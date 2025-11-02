"""
Route calculation endpoint for ATLAS API.

Calculates optimal route between origin and destination using OSM2PO graph with pgRouting.
Uses new tables: app.catalunya_truck_2po_4pgr (edges) and app.catalunya_truck_2po_vertex (vertices)
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any
import logging
import json
from sqlalchemy import create_engine, text, exc

from .. import config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/route", tags=["route"])

@router.get("", response_model=Dict[str, Any])
async def calculate_route(
    origin_id: str = Query(..., description="Origin location ID"),
    destination_id: str = Query(..., description="Destination location ID")
):
    """
    Calculate optimal route between origin and destination.
    
    Uses OSM2PO graph with pgRouting (Dijkstra algorithm).
    
    Args:
        origin_id: Location ID for origin (e.g., "08019")
        destination_id: Location ID for destination (e.g., "25120")
        
    Returns:
        JSON with route information
    """
    logger.info(f"Route request: {origin_id} -> {destination_id}")
    
    try:
        engine = create_engine(config.DB_DSN)
        
        with engine.connect() as conn:
            # 1. Get origin coordinates from app.sodd_locations
            origin_query = text("""
                SELECT 
                    location_id,
                    location_name,
                    longitude,
                    latitude
                FROM app.sodd_locations
                WHERE location_id = :origin_id
            """)
            origin_result = conn.execute(origin_query, {"origin_id": origin_id}).fetchone()
            
            if not origin_result:
                raise HTTPException(
                    status_code=404,
                    detail=f"Origin location '{origin_id}' not found"
                )
            
            origin_info = {
                "id": origin_result.location_id,
                "name": origin_result.location_name,
                "longitude": float(origin_result.longitude),
                "latitude": float(origin_result.latitude)
            }
            
            # 2. Get destination coordinates from app.sodd_locations
            dest_query = text("""
                SELECT 
                    location_id,
                    location_name,
                    longitude,
                    latitude
                FROM app.sodd_locations
                WHERE location_id = :dest_id
            """)
            dest_result = conn.execute(dest_query, {"dest_id": destination_id}).fetchone()
            
            if not dest_result:
                raise HTTPException(
                    status_code=404,
                    detail=f"Destination location '{destination_id}' not found"
                )
            
            dest_info = {
                "id": dest_result.location_id,
                "name": dest_result.location_name,
                "longitude": float(dest_result.longitude),
                "latitude": float(dest_result.latitude)
            }
            
            # 3. Find nearest vertex to origin using OSM2PO tables
            origin_vertex_query = text("""
                SELECT id 
                FROM app.catalunya_truck_2po_vertex 
                ORDER BY geom_vertex <-> ST_SetSRID(
                    ST_MakePoint(:lon, :lat), 4326
                ) 
                LIMIT 1
            """)
            origin_vertex = conn.execute(
                origin_vertex_query, 
                {"lon": origin_info["longitude"], "lat": origin_info["latitude"]}
            ).scalar()
            
            if not origin_vertex:
                raise HTTPException(
                    status_code=500,
                    detail="Could not find nearest road network vertex to origin"
                )
            
            # 4. Find nearest vertex to destination using OSM2PO tables
            dest_vertex_query = text("""
                SELECT id 
                FROM app.catalunya_truck_2po_vertex 
                ORDER BY geom_vertex <-> ST_SetSRID(
                    ST_MakePoint(:lon, :lat), 4326
                ) 
                LIMIT 1
            """)
            dest_vertex = conn.execute(
                dest_vertex_query,
                {"lon": dest_info["longitude"], "lat": dest_info["latitude"]}
            ).scalar()
            
            if not dest_vertex:
                raise HTTPException(
                    status_code=500,
                    detail="Could not find nearest road network vertex to destination"
                )
            
            # 5. Calculate route using pgr_dijkstra on OSM2PO tables - get segments
            # Note: cost units depend on OSM2PO configuration (will be divided by 60 to get minutes)
            route_query = text("""
                WITH route AS (
                    SELECT * FROM pgr_dijkstra(
                        'SELECT id, source, target, cost, reverse_cost FROM app.catalunya_truck_2po_4pgr',
                        :origin_vertex, :dest_vertex, true
                    )
                )
                SELECT
                    e.id,
                    e.cost AS segment_cost,
                    ST_Length(e.geom_way::geography)/1000 AS segment_km,
                    ST_AsGeoJSON(e.geom_way) AS geometry
                FROM route r
                JOIN app.catalunya_truck_2po_4pgr e ON e.id = r.edge
                WHERE r.edge <> -1
                ORDER BY r.seq
            """)
            
            segments_result = conn.execute(
                route_query,
                {"origin_vertex": origin_vertex, "dest_vertex": dest_vertex}
            ).fetchall()
            
            if not segments_result:
                raise HTTPException(
                    status_code=404,
                    detail=f"No route found between {origin_id} and {destination_id}"
                )
            
            # Build segments array with geometries
            segments = []
            total_km = 0
            total_cost = 0  # cost units from database (will be divided by 60 to get minutes)
            
            for seg in segments_result:
                segment_km = float(seg.segment_km) if seg.segment_km else 0
                segment_cost = float(seg.segment_cost) if seg.segment_cost else 0
                total_km += segment_km
                total_cost += segment_cost
                
                segments.append({
                    "id": seg.id,
                    "distance_km": round(segment_km, 3),
                    "time_minutes": round(segment_cost * 60.0, 2),
                    "geometry": seg.geometry
                })
            
            # 6. Build response
            # cost is in HOURS, so multiply by 60 to get minutes
            total_minutes = total_cost * 60.0
            total_hours = total_cost
            
            response = {
                "origin": origin_info,
                "destination": dest_info,
                "summary": {
                    "total_distance_km": round(total_km, 2),
                    "total_time_minutes": round(total_minutes, 2),
                    "total_time_hours": round(total_hours, 2),
                    "total_segments": len(segments)
                },
                "segments": segments
            }
            
            logger.info(
                f"Route calculated: {origin_id} -> {destination_id}, "
                f"{response['summary']['total_distance_km']} km, "
                f"{response['summary']['total_time_minutes']} min"
            )
            
            return response
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating route: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating route: {str(e)}"
        )


@router.get("/health", response_model=Dict[str, Any])
async def route_health():
    """Health check for routing service."""
    try:
        engine = create_engine(config.DB_DSN)
        
        with engine.connect() as conn:
            health_query = text("""
                SELECT
                    (SELECT COUNT(*) FROM app.catalunya_truck_2po_4pgr) as edge_count,
                    (SELECT COUNT(*) FROM app.catalunya_truck_2po_vertex) as vertex_count
            """)
            health_result = conn.execute(health_query).fetchone()
            
            if health_result:
                return {
                    "status": "healthy",
                    "routing_service": "operational",
                    "graph": {
                        "edge_count": int(health_result[0]) if health_result[0] else 0,
                        "vertex_count": int(health_result[1]) if health_result[1] else 0
                    }
                }
            else:
                return {"status": "degraded", "routing_service": "unable_to_check"}
            
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "message": str(e),
            "routing_service": "unavailable"
        }

