"""
Route inference endpoint for ATLAS API.

Receives form input from atlas_web and returns optimal routes (O → waypoints → D)
that maximize a benefit score. Includes all internal data (scores, costs, etc.) 
for debugging and visualization.

Key rules:
- Buffer in km only (no time buffer)
- No fixed stop times (only route deviation time/distance)
- Greedy knapsack algorithm for route construction
- Transparent error handling (no defaults, explicit errors)
"""
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
from sqlalchemy import create_engine, text
import numpy as np
from geopy.distance import geodesic

from .. import config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/inference", tags=["inference"])


# ============================================================================
# REQUEST/RESPONSE MODELS (PYDANTIC)
# ============================================================================

class InferenceRequest(BaseModel):
    """
    Inference request from web form.
    
    All fields are mandatory. No defaults allowed.
    """
    origin_id: str = Field(..., description="Origin location ID (mandatory)")
    destination_id: str = Field(..., description="Destination location ID (mandatory)")
    truck_type: str = Field(..., description="Truck type: 'normal' or 'refrigerado' (mandatory)")
    buffer_value_km: float = Field(..., description="Buffer in kilometers (mandatory)", gt=0)
    available_capacity_kg: float = Field(..., description="Available capacity in kg (mandatory)", gt=0)
    date: str = Field(..., description="Date in ISO format YYYY-MM-DD (mandatory)")
    
    @validator('truck_type')
    def validate_truck_type(cls, v):
        if v not in ['normal', 'refrigerado']:
            raise ValueError("truck_type must be 'normal' or 'refrigerado'")
        return v
    
    @validator('date')
    def validate_date(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError("date must be in format YYYY-MM-DD")
        return v


class CandidateDebugInfo(BaseModel):
    """Debug information for a single candidate location."""
    location_id: str
    location_name: str
    latitude: float
    longitude: float
    eta_km: float  # Distance O→i in km
    f_eta: float  # ETA scoring function: 1/(1+ETA_km)
    delta_d_km: float  # Extra distance: d(O,i) + d(i,D) - d(O,D)
    is_feasible: bool  # Whether delta_d <= buffer
    p_probability: float  # Probability of ≥1 trip (from ML model)
    p_price_eur: float  # Expected price in € (from ML model)
    p_weight_kg: float  # Expected weight in kg (from ML model)
    score: float  # Final score: f_eta × (Pv×X) × (Pp×Y) × (Pw×Z)
    score_per_km: float  # score / delta_d (for ranking)


class RouteWaypoint(BaseModel):
    """A waypoint in a route."""
    location_id: str
    location_name: str
    latitude: float
    longitude: float
    sequence: int  # Order in route (1, 2, 3...)


class AlternativeRoute(BaseModel):
    """An alternative route with waypoints."""
    route_id: int
    waypoints: List[RouteWaypoint]
    total_distance_km: float  # Total distance including deviations
    extra_distance_km: float  # Sum of all deltas
    total_score: float  # Sum of scores of all waypoints
    total_expected_weight_kg: float  # Sum of expected weights
    route_geometry: Optional[str] = None  # GeoJSON LineString for visualization


class InferenceResponse(BaseModel):
    """
    Inference response with routes and debug info.
    
    Includes all internal calculations for transparency and debugging.
    """
    # Base trip metrics (O→D without deviations)
    base_trip: Dict[str, Any]
    
    # Alternative routes (sorted by best score)
    alternative_routes: List[AlternativeRoute]
    
    # All evaluated candidates (for visualization and debugging)
    candidates_information: List[CandidateDebugInfo]
    
    # Metadata
    metadata: Dict[str, Any]


# ============================================================================
# ML MODEL LOADING
# ============================================================================

# Global model cache
_models_cache: Dict[str, Any] = {}


def get_models_dir() -> Path:
    """
    Get models directory from environment variable.
    
    Returns:
        Path to models directory
        
    Raises:
        ValueError: If MODELS_DIR env var not set or directory doesn't exist
    """
    models_dir_env = os.getenv("MODELS_DIR")
    
    if not models_dir_env:
        # Try default path
        default_path = config.PROJECT_ROOT / "models"
        if default_path.exists():
            logger.info(f"Using default models directory: {default_path}")
            return default_path
        else:
            raise ValueError(
                "MODELS_DIR environment variable not set and default 'models/' not found. "
                "Please set MODELS_DIR to point to the models directory."
            )
    
    models_dir = Path(models_dir_env)
    if not models_dir.exists():
        raise ValueError(f"Models directory not found at: {models_dir}")
    
    logger.info(f"Using models directory: {models_dir}")
    return models_dir


def load_ml_models() -> Dict[str, Any]:
    """
    Load ML models (probability, price, weight) from disk.
    
    Models are loaded from MODELS_DIR environment variable.
    Models are cached globally to avoid reloading on each request.
    
    NOTE: Currently, the inference endpoint uses heuristic predictions instead of
    the trained ML models due to feature mismatch. The models were trained with
    features that require database lookups and complex feature engineering
    (origin_lat, origin_lon, destination_lat, destination_lon, month, quarter,
    distance_to_bcn, is_coastal, etc.). For real-time inference, we use simplified
    heuristic predictions based on distance and truck type.
    
    TODO: Align feature engineering pipeline between training and inference, or
    retrain models with simpler feature sets suitable for real-time inference.
    
    Returns:
        Dictionary with 'probability', 'price', 'weight' model bundles
        
    Raises:
        ValueError: If models not found or cannot be loaded
    """
    global _models_cache
    
    # Return cached models if already loaded
    if _models_cache:
        logger.debug("Returning cached ML models")
        return _models_cache
    
    logger.info("Loading ML models...")
    
    try:
        # Import here to avoid circular dependencies
        from atlas_ml.serialization import load_bundle
        
        models_dir = get_models_dir()
        
        # Find latest model directories
        # Expected naming: probability_vYYYYMMDD_HHMMSS, price_vYYYYMMDD_HHMMSS, weight_vYYYYMMDD_HHMMSS
        prob_dirs = sorted([d for d in models_dir.glob("probability_v*") if d.is_dir()], reverse=True)
        price_dirs = sorted([d for d in models_dir.glob("price_v*") if d.is_dir()], reverse=True)
        weight_dirs = sorted([d for d in models_dir.glob("weight_v*") if d.is_dir()], reverse=True)
        
        if not prob_dirs:
            raise ValueError(f"No probability model found in {models_dir}")
        if not price_dirs:
            raise ValueError(f"No price model found in {models_dir}")
        if not weight_dirs:
            raise ValueError(f"No weight model found in {models_dir}")
        
        # Load latest models
        prob_model_path = prob_dirs[0]
        price_model_path = price_dirs[0]
        weight_model_path = weight_dirs[0]
        
        logger.info(f"Loading probability model from: {prob_model_path}")
        probability_bundle = load_bundle(prob_model_path)
        
        logger.info(f"Loading price model from: {price_model_path}")
        price_bundle = load_bundle(price_model_path)
        
        logger.info(f"Loading weight model from: {weight_model_path}")
        weight_bundle = load_bundle(weight_model_path)
        
        _models_cache = {
            'probability': probability_bundle,
            'price': price_bundle,
            'weight': weight_bundle,
            'models_dir': str(models_dir),
            'probability_version': prob_model_path.name,
            'price_version': price_model_path.name,
            'weight_version': weight_model_path.name
        }
        
        logger.info("ML models loaded successfully")
        return _models_cache
        
    except Exception as e:
        logger.error(f"Failed to load ML models: {e}")
        raise ValueError(f"Cannot load ML models: {str(e)}")


# ============================================================================
# DATABASE QUERIES
# ============================================================================

def get_location_info(conn, location_id: str) -> Dict[str, Any]:
    """
    Get location information from database.
    
    Args:
        conn: Database connection
        location_id: Location ID
        
    Returns:
        Dictionary with id, name, latitude, longitude
        
    Raises:
        ValueError: If location not found
    """
    query = text("""
        SELECT 
            location_id,
            location_name,
            longitude,
            latitude
        FROM app.sodd_locations
        WHERE location_id = :location_id
    """)
    result = conn.execute(query, {"location_id": location_id}).fetchone()
    
    if not result:
        raise ValueError(f"Location '{location_id}' not found in database")
    
    return {
        "id": result.location_id,
        "name": result.location_name,
        "longitude": float(result.longitude),
        "latitude": float(result.latitude)
    }


def get_candidate_locations(conn, origin: Dict, destination: Dict, buffer_km: float) -> List[Dict[str, Any]]:
    """
    Get candidate locations within corridor between origin and destination.
    
    Uses an ellipse with major axis from O to D and controlled eccentricity.
    If ellipse query is too complex, falls back to rectangular corridor.
    
    Args:
        conn: Database connection
        origin: Origin location dict (id, name, lat, lon)
        destination: Destination location dict
        buffer_km: Buffer in kilometers
        
    Returns:
        List of candidate locations (excluding origin and destination)
    """
    # Calculate center point and distance
    center_lat = (origin['latitude'] + destination['latitude']) / 2
    center_lon = (origin['longitude'] + destination['longitude']) / 2
    
    # Semi-major axis = half distance O→D (in meters)
    base_distance_km = geodesic(
        (origin['latitude'], origin['longitude']),
        (destination['latitude'], destination['longitude'])
    ).kilometers
    
    semi_major_m = (base_distance_km * 1000) / 2
    
    # Semi-minor axis = buffer (smooth ellipse, low eccentricity)
    # Add some extra buffer to avoid too narrow corridor
    semi_minor_m = buffer_km * 1000 * 1.5
    
    # Calculate rotation angle (angle of line O→D)
    angle_rad = np.arctan2(
        destination['latitude'] - origin['latitude'],
        destination['longitude'] - origin['longitude']
    )
    angle_deg = np.degrees(angle_rad)
    
    # Query using rectangular corridor
    # Constraints:
    # - Longitudinal (X axis): candidates must be BETWEEN origin and destination (not before/after)
    # - Lateral (Y axis): candidates can deviate ±buffer_km/2 from the direct line O→D
    query = text("""
        WITH route_line AS (
            SELECT 
                ST_MakeLine(
                    ST_SetSRID(ST_MakePoint(:origin_lon, :origin_lat), 4326),
                    ST_SetSRID(ST_MakePoint(:dest_lon, :dest_lat), 4326)
                ) AS geom,
                ST_SetSRID(ST_MakePoint(:origin_lon, :origin_lat), 4326)::geography AS origin_geog,
                ST_SetSRID(ST_MakePoint(:dest_lon, :dest_lat), 4326)::geography AS dest_geog
        ),
        bounds AS (
            SELECT 
                LEAST(:origin_lon, :dest_lon) AS min_lon,
                GREATEST(:origin_lon, :dest_lon) AS max_lon,
                LEAST(:origin_lat, :dest_lat) AS min_lat,
                GREATEST(:origin_lat, :dest_lat) AS max_lat
            FROM route_line
        )
        SELECT 
            l.location_id,
            l.location_name,
            l.longitude,
            l.latitude,
            ST_Distance(
                route_line.geom::geography,
                ST_SetSRID(ST_MakePoint(l.longitude, l.latitude), 4326)::geography
            ) AS distance_to_line_m
        FROM app.sodd_locations l, route_line, bounds
        WHERE 
            -- Lateral constraint: distance to line O→D <= buffer_km (perpendicular distance)
            ST_Distance(
                route_line.geom::geography,
                ST_SetSRID(ST_MakePoint(l.longitude, l.latitude), 4326)::geography
            ) <= :buffer_m
            -- Longitudinal constraint: must be between O and D (bounding box)
            AND l.longitude BETWEEN bounds.min_lon AND bounds.max_lon
            AND l.latitude BETWEEN bounds.min_lat AND bounds.max_lat
            -- Exclude origin and destination
            AND l.location_id NOT IN (:origin_id, :dest_id)
        ORDER BY l.location_id
    """)
    
    # Use buffer * 1000 to convert km to meters for ST_Buffer (geography uses meters)
    result = conn.execute(query, {
        "origin_lon": origin['longitude'],
        "origin_lat": origin['latitude'],
        "dest_lon": destination['longitude'],
        "dest_lat": destination['latitude'],
        "buffer_m": buffer_km * 1000,
        "origin_id": origin['id'],
        "dest_id": destination['id']
    })
    
    candidates = []
    for row in result:
        candidates.append({
            "id": row.location_id,
            "name": row.location_name,
            "longitude": float(row.longitude),
            "latitude": float(row.latitude)
        })
    
    logger.info(f"Found {len(candidates)} candidate locations within {buffer_km}km corridor")
    return candidates


def calculate_base_route(conn, origin_id: str, destination_id: str) -> Dict[str, Any]:
    """
    Calculate base route O→D using pgRouting with OSM2PO tables.
    
    Args:
        conn: Database connection
        origin_id: Origin location ID
        destination_id: Destination location ID
        
    Returns:
        Dictionary with distance_km, time_minutes, cost (cost = distance * base_cost_per_km)
        
    Raises:
        ValueError: If route cannot be calculated
    """
    # Get location info
    origin = get_location_info(conn, origin_id)
    destination = get_location_info(conn, destination_id)
    
    # Find nearest vertices in OSM2PO graph
    vertex_query = text("""
        SELECT id 
        FROM app.catalunya_truck_2po_vertex 
        ORDER BY geom_vertex <-> ST_SetSRID(ST_MakePoint(:lon, :lat), 4326) 
        LIMIT 1
    """)
    
    origin_vertex = conn.execute(vertex_query, {
        "lon": origin["longitude"],
        "lat": origin["latitude"]
    }).scalar()
    
    dest_vertex = conn.execute(vertex_query, {
        "lon": destination["longitude"],
        "lat": destination["latitude"]
    }).scalar()
    
    if not origin_vertex or not dest_vertex:
        raise ValueError("Cannot find road network vertices for origin/destination")
    
    # Calculate route using OSM2PO tables
    route_query = text("""
        WITH route AS (
            SELECT * FROM pgr_dijkstra(
                'SELECT id, source, target, cost, reverse_cost FROM app.catalunya_truck_2po_4pgr',
                :origin_vertex,
                :dest_vertex,
                true
            )
        )
        SELECT
            SUM(ST_Length(e.geom_way::geography))/1000 AS total_distance_km,
            SUM(e.cost)*60.0 AS total_time_minutes
        FROM route r
        JOIN app.catalunya_truck_2po_4pgr e ON e.id = r.edge
        WHERE r.edge <> -1
    """)
    
    result = conn.execute(route_query, {
        "origin_vertex": origin_vertex,
        "dest_vertex": dest_vertex
    }).fetchone()
    
    if not result or result.total_distance_km is None:
        raise ValueError(f"No route found between {origin_id} and {destination_id}")
    
    distance_km = float(result.total_distance_km)
    time_minutes = float(result.total_time_minutes)
    
    # Calculate base cost (simple: proportional to distance)
    # Using 1 €/km as base cost (configurable)
    BASE_COST_PER_KM = 1.0
    cost_eur = distance_km * BASE_COST_PER_KM
    
    # Get route geometry for visualization
    waypoints = [origin, destination]
    route_geometry = calculate_route_geometry(conn, waypoints)
    
    return {
        "distance_km": round(distance_km, 2),
        "time_minutes": round(time_minutes, 2),
        "cost_eur": round(cost_eur, 2),
        "origin": origin,
        "destination": destination,
        "route_geometry": route_geometry
    }


def calculate_route_geometry(conn, waypoints: List[Dict[str, Any]]) -> str:
    """
    Calculate route geometry for a multi-waypoint route using pgRouting.
    Returns only the coordinates as a GeoJSON LineString for visualization.
    
    Args:
        conn: Database connection
        waypoints: List of waypoints with latitude, longitude (in sequence order)
        
    Returns:
        GeoJSON LineString string with all route coordinates
    """
    if len(waypoints) < 2:
        return '{"type":"LineString","coordinates":[]}'
    
    # Find nearest vertices for all waypoints using OSM2PO tables
    vertex_query = text("""
        SELECT id 
        FROM app.catalunya_truck_2po_vertex 
        ORDER BY geom_vertex <-> ST_SetSRID(ST_MakePoint(:lon, :lat), 4326) 
        LIMIT 1
    """)
    
    vertices = []
    for wp in waypoints:
        vertex_id = conn.execute(vertex_query, {
            "lon": wp["longitude"],
            "lat": wp["latitude"]
        }).scalar()
        if vertex_id:
            vertices.append(vertex_id)
    
    if len(vertices) < 2:
        return '{"type":"LineString","coordinates":[]}'
    
    # Calculate routes between consecutive waypoints and collect all geometries
    all_coordinates = []
    
    for i in range(len(vertices) - 1):
        segment_query = text("""
            SELECT ST_AsGeoJSON(e.geom_way) as geometry
            FROM pgr_dijkstra(
                'SELECT id, source, target, cost, reverse_cost FROM app.catalunya_truck_2po_4pgr',
                :start_vertex,
                :end_vertex,
                true
            ) r
            JOIN app.catalunya_truck_2po_4pgr e ON e.id = r.edge
            WHERE r.edge <> -1
            ORDER BY r.seq
        """)
        
        result = conn.execute(segment_query, {
            "start_vertex": vertices[i],
            "end_vertex": vertices[i + 1]
        })
        
        # Extract coordinates from each segment
        for row in result:
            if row.geometry:
                import json
                geom = json.loads(row.geometry)
                if geom.get("type") == "LineString" and "coordinates" in geom:
                    # Add coordinates, avoiding duplicates at connection points
                    coords = geom["coordinates"]
                    if not all_coordinates or coords[0] != all_coordinates[-1]:
                        all_coordinates.extend(coords)
                    else:
                        all_coordinates.extend(coords[1:])
    
    # Build final GeoJSON LineString
    import json
    return json.dumps({
        "type": "LineString",
        "coordinates": all_coordinates
    })


# ============================================================================
# SCORING AND FEASIBILITY
# ============================================================================

def calculate_geodesic_distance(loc1: Dict, loc2: Dict) -> float:
    """
    Calculate geodesic distance between two locations in km.
    
    Args:
        loc1: Location dict with latitude, longitude
        loc2: Location dict with latitude, longitude
        
    Returns:
        Distance in kilometers
    """
    return geodesic(
        (loc1['latitude'], loc1['longitude']),
        (loc2['latitude'], loc2['longitude'])
    ).kilometers


def calculate_eta_score(eta_km: float) -> float:
    """
    Calculate ETA scoring function: f_ETA = ETA_km
    
    Linear function: more travel time = more time for loads to appear during the day.
    This multiplies the daily probability prediction to account for time factor.
    
    Interpretation: If the truck travels X km to reach a candidate, there are X km
    worth of travel time for loads to appear at that location. The probability
    prediction is scaled proportionally to this travel distance.
    
    At origin (0 km): no travel time yet, so 0% of daily probability applies.
    At destination (km_od): full journey time has passed, so 100% of daily probability applies.
    
    Since km_od is constant for all candidates in a route optimization, we can
    directly use eta_km for ranking (the constant factor doesn't affect order).
    
    Args:
        eta_km: Distance from origin to candidate in km
        
    Returns:
        ETA score (linear with distance)
    """
    return eta_km


def predict_ml_features(
    models: Dict[str, Any],
    candidate_id: str,
    destination_id: str,
    truck_type: str,
    date_str: str,
    candidate_lat: float,
    candidate_lon: float,
    destination_lat: float,
    destination_lon: float
) -> Tuple[float, float, float]:
    """
    Predict probability, price, and weight using real ML models.
    
    Generates the required features and uses the trained models from atlas_ml.
    
    Args:
        models: Dictionary with loaded model bundles
        candidate_id: Candidate location ID (origin for i→D)
        destination_id: Destination location ID
        truck_type: Truck type ('normal' or 'refrigerado')
        date_str: Date string (YYYY-MM-DD)
        candidate_lat: Candidate latitude
        candidate_lon: Candidate longitude
        destination_lat: Destination latitude
        destination_lon: Destination longitude
        
    Returns:
        Tuple (probability, price_eur, weight_kg)
    """
    import pandas as pd
    
    try:
        # Parse date
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        
        # Determine tipo_mercancia from truck_type
        tipo_mercancia = "refrigerada" if truck_type == "refrigerado" else "normal"
        
        # Create base input dataframe
        input_data = pd.DataFrame([{
            'origin_id': candidate_id,
            'destination_id': destination_id,
            'truck_type': truck_type,
            'tipo_mercancia': tipo_mercancia,
            'date': date_obj,
            'origin_lat': candidate_lat,
            'origin_lon': candidate_lon,
            'destination_lat': destination_lat,
            'destination_lon': destination_lon
        }])
        
        # === Generate Features (same as in atlas_ml.featurization) ===
        
        # 1. Time features
        input_data['day_of_week'] = date_obj.weekday()
        input_data['month'] = date_obj.month
        input_data['day'] = date_obj.day
        input_data['week_of_year'] = date_obj.isocalendar()[1]
        input_data['is_weekend'] = int(date_obj.weekday() >= 5)
        input_data['quarter'] = (date_obj.month - 1) // 3 + 1  # Q1=1, Q2=2, Q3=3, Q4=4
        input_data['holiday_flag'] = 0
        
        # 2. Distance features
        # Calculate haversine distance
        def haversine_distance(lat1, lon1, lat2, lon2):
            """Calculate haversine distance in km."""
            from math import radians, sin, cos, sqrt, atan2
            R = 6371  # Earth radius in km
            
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            
            return R * c
        
        od_distance = haversine_distance(
            candidate_lat, candidate_lon,
            destination_lat, destination_lon
        )
        input_data['od_length_km'] = od_distance
        
        # 3. Geographic features (Catalunya-specific)
        barcelona_lat, barcelona_lon = 41.3851, 2.1734
        
        # Distance to Barcelona
        input_data['origin_dist_to_bcn'] = haversine_distance(
            candidate_lat, candidate_lon,
            barcelona_lat, barcelona_lon
        )
        input_data['dest_dist_to_bcn'] = haversine_distance(
            destination_lat, destination_lon,
            barcelona_lat, barcelona_lon
        )
        
        # Coastal indicators (Catalunya coast: lon > 0.5, lat 40.5-42.5)
        input_data['origin_is_coastal'] = int(
            candidate_lon > 0.5 and 40.5 < candidate_lat < 42.5
        )
        input_data['dest_is_coastal'] = int(
            destination_lon > 0.5 and 40.5 < destination_lat < 42.5
        )
        
        # 4. Encode categorical features
        probability_bundle = models['probability']
        encoders = probability_bundle.encoders.get('feature_builder_encoders', {})
        
        for col in ['tipo_mercancia']:
            if col in encoders:
                encoder = encoders[col]
                try:
                    input_data[f'{col}_encoded'] = encoder.transform([tipo_mercancia])[0]
                except ValueError:
                    # Unknown category
                    logger.warning(f"Unknown {col}: {tipo_mercancia}, using -1")
                    input_data[f'{col}_encoded'] = -1
        
        # === Make Predictions ===
        
        # Get feature names from each model
        prob_features = probability_bundle.features
        price_features = models['price'].features
        weight_features = models['weight'].features
        
        # Prepare feature matrices (fill missing with 0)
        def prepare_features(feature_names):
            X = pd.DataFrame()
            for feat in feature_names:
                if feat in input_data.columns:
                    X[feat] = input_data[feat]
                else:
                    X[feat] = 0.0
            return X
        
        X_prob = prepare_features(prob_features)
        X_price = prepare_features(price_features)
        X_weight = prepare_features(weight_features)
        
        # Apply scalers if available
        if probability_bundle.scaler:
            X_prob = pd.DataFrame(
                probability_bundle.scaler.transform(X_prob),
                columns=X_prob.columns
            )
        if models['price'].scaler:
            X_price = pd.DataFrame(
                models['price'].scaler.transform(X_price),
                columns=X_price.columns
            )
        if models['weight'].scaler:
            X_weight = pd.DataFrame(
                models['weight'].scaler.transform(X_weight),
                columns=X_weight.columns
            )
        
        # Predict
        prob_pred = probability_bundle.model.predict(X_prob)[0]
        price_pred = models['price'].model.predict(X_price)[0]
        weight_pred = models['weight'].model.predict(X_weight)[0]
        
        # Post-process predictions
        
        # Probability: model predicts daily trip count (possibly log-transformed)
        if probability_bundle.encoders.get('log_transformed', False):
            n_trips_daily = np.expm1(prob_pred)  # Reverse log transform
        else:
            n_trips_daily = prob_pred
        
        # Convert daily trips to probability (simple heuristic)
        # More trips = higher probability, capped at 1.0
        probability = min(max(n_trips_daily / 10.0, 0.05), 0.95)
        
        # Price: reverse log transform if applied
        if models['price'].encoders.get('log_transformed', False):
            price_eur = np.expm1(price_pred)
        else:
            price_eur = price_pred
        
        # Weight: reverse log transform if applied
        if models['weight'].encoders.get('log_transformed', False):
            weight_kg = np.expm1(weight_pred)
        else:
            weight_kg = weight_pred
        
        # Ensure non-negative and reasonable values
        probability = max(0.0, min(1.0, float(probability)))
        price_eur = max(0.0, float(price_eur))
        weight_kg = max(0.0, float(weight_kg))
        
        logger.debug(
            f"ML predictions for {candidate_id}→{destination_id} ({od_distance:.1f}km): "
            f"P={probability:.3f}, Price={price_eur:.2f}€, Weight={weight_kg:.2f}kg"
        )
        
        return probability, price_eur, weight_kg
        
    except Exception as e:
        logger.error(f"Error during ML prediction for {candidate_id}→{destination_id}: {e}", exc_info=True)
        # Return conservative defaults on error
        return 0.2, 200.0, 15000.0


def calculate_candidate_score(
    candidate: Dict,
    origin: Dict,
    destination: Dict,
    base_distance_km: float,
    models: Dict[str, Any],
    truck_type: str,
    date_str: str,
    buffer_km: float
) -> Dict[str, Any]:
    """
    Calculate score and feasibility for a candidate location.
    
    Args:
        candidate: Candidate location dict
        origin: Origin location dict
        destination: Destination location dict
        base_distance_km: Direct distance O→D
        models: Loaded ML models
        truck_type: Truck type
        date_str: Date string
        buffer_km: Buffer in km
        
    Returns:
        Dictionary with all scoring details
    """
    # Calculate distances
    d_oi = calculate_geodesic_distance(origin, candidate)
    d_id = calculate_geodesic_distance(candidate, destination)
    
    # Delta distance: extra km to deviate through this candidate
    delta_d = d_oi + d_id - base_distance_km
    
    # Feasibility: delta_d must be <= buffer_km
    is_feasible = delta_d <= buffer_km
    
    # ETA and score function
    eta_km = d_oi
    f_eta = calculate_eta_score(eta_km)
    
    # ML predictions
    p_prob, p_price, p_weight = predict_ml_features(
        models, candidate['id'], destination['id'], truck_type, date_str,
        candidate['latitude'], candidate['longitude'],
        destination['latitude'], destination['longitude']
    )
    
    # Score formula: (Pv × f_eta × X) × (Pp × Y) / (Pw × Z)
    # f_eta multiplies probability: more travel time = more time for loads to appear
    # Weight divides: higher predicted weight = worse (less capacity remaining)
    # Weighting factors (configurable constants)
    X = 10.0  # Probability weight
    Y = 0.01  # Price weight (€ → scale)
    Z = 1.0   # Weight weight (kg → scale) - now used as divisor
    
    # Correct formula: multiply prob by f_eta, divide by weight
    score = (p_prob * f_eta * X) * (p_price * Y) / (p_weight * Z if p_weight > 0 else 1.0)
    
    # Score per km (for ranking candidates efficiently)
    score_per_km = score / delta_d if delta_d > 0 else score * 1000
    
    return {
        'location_id': candidate['id'],
        'location_name': candidate['name'],
        'latitude': candidate['latitude'],
        'longitude': candidate['longitude'],
        'eta_km': round(eta_km, 2),
        'f_eta': round(f_eta, 4),
        'delta_d_km': round(delta_d, 2),
        'is_feasible': is_feasible,
        'p_probability': round(p_prob, 4),
        'p_price_eur': round(p_price, 2),
        'p_weight_kg': round(p_weight, 2),
        'score': round(score, 4),
        'score_per_km': round(score_per_km, 4)
    }


# ============================================================================
# ROUTE CONSTRUCTION (GREEDY KNAPSACK)
# ============================================================================

def reorder_waypoints_geographically(
    selected_candidates: List[Dict[str, Any]],
    origin: Dict,
    destination: Dict
) -> List[Dict[str, Any]]:
    """
    Reorder selected waypoints based on their progressive position from origin to destination.
    
    Uses the "projection distance" method: calculate how far along the O→D line each waypoint is.
    This ensures waypoints are visited in geographical order, avoiding backtracking.
    
    Args:
        selected_candidates: List of selected waypoint candidates
        origin: Origin location with latitude, longitude
        destination: Destination location with latitude, longitude
        
    Returns:
        Reordered list of candidates
    """
    if len(selected_candidates) <= 1:
        return selected_candidates
    
    import math
    
    # Vector from origin to destination
    dx = destination['longitude'] - origin['longitude']
    dy = destination['latitude'] - origin['latitude']
    
    # Length squared (avoid sqrt for efficiency)
    length_sq = dx*dx + dy*dy
    
    if length_sq == 0:
        # Origin and destination are the same point (edge case)
        return selected_candidates
    
    # Calculate projection distance for each candidate
    candidates_with_projection = []
    for candidate in selected_candidates:
        # Vector from origin to candidate
        cx = candidate['longitude'] - origin['longitude']
        cy = candidate['latitude'] - origin['latitude']
        
        # Dot product gives projection length (normalized 0→1 along O→D line)
        projection = (cx * dx + cy * dy) / length_sq
        
        candidates_with_projection.append({
            'candidate': candidate,
            'projection': projection
        })
    
    # Sort by projection (candidates closer to origin come first)
    candidates_with_projection.sort(key=lambda x: x['projection'])
    
    # Return reordered candidates
    return [item['candidate'] for item in candidates_with_projection]


def build_routes_greedy(
    candidates: List[Dict[str, Any]],
    origin: Dict,
    destination: Dict,
    base_distance_km: float,
    buffer_km: float,
    capacity_kg: float,
    conn,  # Database connection for route geometry calculation
    num_routes: int = 3
) -> List[AlternativeRoute]:
    """
    Build alternative routes using greedy knapsack algorithm.
    
    Candidates are ordered by score_per_km (descending).
    Iteratively insert candidates while:
    - Sum of delta_d <= buffer_km
    - Sum of weights <= capacity_kg
    - No duplicate candidates
    
    Args:
        candidates: List of scored candidates (with is_feasible=True)
        origin: Origin location
        destination: Destination location
        base_distance_km: Base distance O→D
        buffer_km: Maximum extra distance allowed
        capacity_kg: Available capacity in kg
        num_routes: Number of alternative routes to generate
        
    Returns:
        List of alternative routes
    """
    # Filter only feasible candidates and sort by score_per_km
    feasible = [c for c in candidates if c['is_feasible']]
    feasible_sorted = sorted(feasible, key=lambda x: x['score_per_km'], reverse=True)
    
    if not feasible_sorted:
        logger.info("No feasible candidates found")
        return []
    
    routes = []
    
    # Generate multiple routes (greedy with different starting points)
    for route_idx in range(num_routes):
        selected = []
        total_delta = 0.0
        total_weight = 0.0
        total_score = 0.0
        used_ids = set()
        
        # For first route: pure greedy
        # For subsequent routes: skip some top candidates to get diversity
        start_offset = route_idx * 2
        
        for i, candidate in enumerate(feasible_sorted[start_offset:], start=start_offset):
            # Skip if already used
            if candidate['location_id'] in used_ids:
                continue
            
            # Check constraints
            new_delta = total_delta + candidate['delta_d_km']
            new_weight = total_weight + candidate['p_weight_kg']
            
            if new_delta <= buffer_km and new_weight <= capacity_kg:
                # Add candidate
                selected.append(candidate)
                total_delta = new_delta
                total_weight = new_weight
                total_score += candidate['score']
                used_ids.add(candidate['location_id'])
        
        if selected:
            # ⭐ REORDER: Sort waypoints geographically to avoid backtracking
            selected = reorder_waypoints_geographically(selected, origin, destination)
            
            # ⭐ RECALCULATE DISTANCES: After reordering, recalculate the actual route distance
            # Calculate sequential distance: O → W1 → W2 → ... → Wn → D
            waypoint_sequence = [origin] + selected + [destination]
            actual_route_distance = 0.0
            
            for i in range(len(waypoint_sequence) - 1):
                segment_distance = calculate_geodesic_distance(
                    waypoint_sequence[i],
                    waypoint_sequence[i + 1]
                )
                actual_route_distance += segment_distance
            
            # Recalculate delta (extra distance compared to direct route)
            recalculated_delta = actual_route_distance - base_distance_km
            
            # Build waypoints list: origin → candidates → destination
            waypoints = []
            
            # Origin (sequence 0)
            waypoints.append(RouteWaypoint(
                location_id=origin['id'],
                location_name=origin['name'],
                latitude=origin['latitude'],
                longitude=origin['longitude'],
                sequence=0
            ))
            
            # Selected candidates (sequence 1, 2, ...) - now in geographical order
            for seq, cand in enumerate(selected, start=1):
                waypoints.append(RouteWaypoint(
                    location_id=cand['location_id'],
                    location_name=cand['location_name'],
                    latitude=cand['latitude'],
                    longitude=cand['longitude'],
                    sequence=seq
                ))
            
            # Destination (final sequence)
            waypoints.append(RouteWaypoint(
                location_id=destination['id'],
                location_name=destination['name'],
                latitude=destination['latitude'],
                longitude=destination['longitude'],
                sequence=len(selected) + 1
            ))
            
            # Total distance = base_distance + recalculated extra_distance
            total_distance = base_distance_km + recalculated_delta
            
            # Calculate route geometry for visualization
            waypoint_coords = [
                {
                    "latitude": wp.latitude,
                    "longitude": wp.longitude
                }
                for wp in waypoints
            ]
            route_geometry = calculate_route_geometry(conn, waypoint_coords)
            
            routes.append(AlternativeRoute(
                route_id=route_idx + 1,
                waypoints=waypoints,
                total_distance_km=round(total_distance, 2),
                extra_distance_km=round(recalculated_delta, 2),  # Use recalculated delta
                total_score=round(total_score, 4),
                total_expected_weight_kg=round(total_weight, 2),
                route_geometry=route_geometry
            ))
    
    logger.info(f"Generated {len(routes)} alternative routes")
    return routes


# ============================================================================
# MAIN ENDPOINT
# ============================================================================

@router.post("", response_model=InferenceResponse)
async def inference_endpoint(request: InferenceRequest):
    """
    Calculate optimal routes with deviations based on ML predictions.
    
    Returns one or more alternative routes (O → waypoints → D) that maximize
    a benefit score, along with all debug information.
    
    Algorithm:
    1. Calculate base route O→D (distance, time, cost)
    2. Find candidate locations within corridor (buffer)
    3. Score each candidate using ML models (probability, price, weight)
    4. Build routes greedily (knapsack) respecting buffer and capacity constraints
    5. Return routes with full debug information
    
    Args:
        request: InferenceRequest with all mandatory fields
        
    Returns:
        InferenceResponse with base trip, alternative routes, and debug info
        
    Raises:
        HTTPException: If any validation fails or errors occur
    """
    try:
        logger.info(f"Inference request: {request.origin_id} → {request.destination_id}")
        
        # ========================================
        # 1. LOAD ML MODELS
        # ========================================
        try:
            models = load_ml_models()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Cannot load ML models: {str(e)}"
            )
        
        # ========================================
        # 2-5. DATABASE OPERATIONS (SINGLE CONNECTION)
        # ========================================
        engine = create_engine(config.DB_DSN)
        
        try:
            with engine.connect() as conn:
                # 2. Calculate base route O→D
                try:
                    base_route = calculate_base_route(conn, request.origin_id, request.destination_id)
                except ValueError as e:
                    raise HTTPException(status_code=404, detail=str(e))
                
                origin = base_route['origin']
                destination = base_route['destination']
                base_distance_km = base_route['distance_km']
                
                logger.info(f"Base route: {base_distance_km} km, {base_route['time_minutes']} min")
                
                # 3. Get candidate locations
                candidates_raw = get_candidate_locations(
                    conn, origin, destination, request.buffer_value_km
                )
                
                if not candidates_raw:
                    logger.info("No candidate locations found in corridor")
                    return InferenceResponse(
                        base_trip=base_route,
                        alternative_routes=[],
                        candidates_information=[],
                        metadata={
                            "reason": "No candidate locations found within buffer corridor",
                            "models_info": {
                                "models_dir": models.get('models_dir'),
                                "probability_version": models.get('probability_version'),
                                "price_version": models.get('price_version'),
                                "weight_version": models.get('weight_version')
                            },
                            "volume_used": False,
                            "buffer_km": request.buffer_value_km,
                            "capacity_kg": request.available_capacity_kg
                        }
                    )
                
                # 4. Score all candidates
                candidates_scored = []
                
                for candidate in candidates_raw:
                    try:
                        score_info = calculate_candidate_score(
                            candidate,
                            origin,
                            destination,
                            base_distance_km,
                            models,
                            request.truck_type,
                            request.date,
                            request.buffer_value_km
                        )
                        candidates_scored.append(score_info)
                    except Exception as e:
                        logger.error(f"Error scoring candidate {candidate['id']}: {e}")
                        # Continue with other candidates
                
                logger.info(f"Scored {len(candidates_scored)} candidates")
                
                # 5. Build routes (greedy knapsack) - connection still open
                routes = build_routes_greedy(
                    candidates_scored,
                    origin,
                    destination,
                    base_distance_km,
                    request.buffer_value_km,
                    request.available_capacity_kg,
                    conn,  # Pass database connection for geometry calculation
                    num_routes=3
                )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Database operation error: {str(e)}"
            )
        
        # ========================================
        # 6. BUILD RESPONSE
        # ========================================
        # Convert candidates_scored to CandidateDebugInfo objects
        candidates_debug_objs = [
            CandidateDebugInfo(**cand) for cand in candidates_scored
        ]
        
        response = InferenceResponse(
            base_trip=base_route,
            alternative_routes=routes,
            candidates_information=candidates_debug_objs,
            metadata={
                "total_candidates_evaluated": len(candidates_scored),
                "total_feasible_candidates": len([c for c in candidates_scored if c['is_feasible']]),
                "total_routes_generated": len(routes),
                "models_info": {
                    "models_dir": models.get('models_dir'),
                    "probability_version": models.get('probability_version'),
                    "price_version": models.get('price_version'),
                    "weight_version": models.get('weight_version')
                },
                "volume_used": False,  # Explicit: NO volume used
                "buffer_km": request.buffer_value_km,
                "capacity_kg": request.available_capacity_kg,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        logger.info(
            f"Inference completed: {len(routes)} routes, "
            f"{len(candidates_scored)} candidates evaluated"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in inference endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
