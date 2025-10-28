"""
Route calculation endpoint for ATLAS API.

Calculates optimal route between origin and destination using pgRouting.
"""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
import logging
from sqlalchemy import create_engine, text

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
    
    Uses pgRouting (Dijkstra algorithm) to find shortest path on road network.
    
    Args:
        origin_id: Location ID for origin
        destination_id: Location ID for destination
        
    Returns:
        JSON with route information:
        - origin: Origin location details
        - destination: Destination location details
        - route_segments: List of route segments with geometry
        - summary: Total distance, time, number of segments
        
    Example:
        GET /route?origin_id=08019&destination_id=25120
    """
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
            
            # 3. Find nearest road network node to origin
            origin_node_query = text("""
                SELECT id 
                FROM app.ways_vertices_pgr 
                ORDER BY the_geom <-> ST_SetSRID(
                    ST_MakePoint(:lon, :lat), 4326
                ) 
                LIMIT 1
            """)
            origin_node = conn.execute(
                origin_node_query, 
                {"lon": origin_info["longitude"], "lat": origin_info["latitude"]}
            ).scalar()
            
            if not origin_node:
                raise HTTPException(
                    status_code=500,
                    detail="Could not find nearest road network node to origin"
                )
            
            logger.info(f"Origin node: {origin_node}")
            
            # 4. Find nearest road network node to destination
            dest_node_query = text("""
                SELECT id 
                FROM app.ways_vertices_pgr 
                ORDER BY the_geom <-> ST_SetSRID(
                    ST_MakePoint(:lon, :lat), 4326
                ) 
                LIMIT 1
            """)
            dest_node = conn.execute(
                dest_node_query,
                {"lon": dest_info["longitude"], "lat": dest_info["latitude"]}
            ).scalar()
            
            if not dest_node:
                raise HTTPException(
                    status_code=500,
                    detail="Could not find nearest road network node to destination"
                )
            
            logger.info(f"Destination node: {dest_node}")
            
            # 5. Calculate route using pgr_dijkstra
            route_query = text("""
                SELECT 
                    r.seq,
                    r.node,
                    r.edge,
                    r.cost,
                    r.agg_cost,
                    w.name,
                    w.length_m,
                    w.length_m / 1000 AS length_km,
                    w.cost_s,
                    w.cost_s / 60 AS cost_minutes,
                    w.maxspeed_forward,
                    w.maxspeed_backward,
                    ST_AsGeoJSON(w.the_geom) as geometry
                FROM pgr_dijkstra(
                    'SELECT gid as id, source, target, cost, reverse_cost FROM app.ways',
                    :origin_node,
                    :dest_node,
                    directed := true
                ) r
                LEFT JOIN app.ways w ON r.edge = w.gid
                ORDER BY r.seq
            """)
            
            route_result = conn.execute(
                route_query,
                {"origin_node": origin_node, "dest_node": dest_node}
            )
            
            # 6. Process route segments
            segments = []
            total_distance_km = 0
            total_time_minutes = 0
            
            for row in route_result:
                # Skip the last row (destination node with no edge)
                if row.edge is None:
                    continue
                
                segment = {
                    "seq": row.seq,
                    "node": row.node,
                    "edge": row.edge,
                    "cost": float(row.cost) if row.cost else None,
                    "agg_cost": float(row.agg_cost) if row.agg_cost else None,
                    "name": row.name,
                    "length_m": float(row.length_m) if row.length_m else None,
                    "length_km": float(row.length_km) if row.length_km else None,
                    "cost_s": float(row.cost_s) if row.cost_s else None,
                    "cost_minutes": float(row.cost_minutes) if row.cost_minutes else None,
                    "maxspeed_forward": row.maxspeed_forward,
                    "maxspeed_backward": row.maxspeed_backward,
                    "geometry": row.geometry  # GeoJSON string
                }
                
                segments.append(segment)
                
                if row.length_km:
                    total_distance_km += float(row.length_km)
                if row.cost_minutes:
                    total_time_minutes += float(row.cost_minutes)
            
            if not segments:
                raise HTTPException(
                    status_code=404,
                    detail=f"No route found between {origin_id} and {destination_id}"
                )
            
            # 7. Build response
            response = {
                "origin": origin_info,
                "destination": dest_info,
                "route_nodes": {
                    "origin_node": origin_node,
                    "destination_node": dest_node
                },
                "segments": segments,
                "summary": {
                    "total_segments": len(segments),
                    "total_distance_km": round(total_distance_km, 2),
                    "total_time_minutes": round(total_time_minutes, 2),
                    "total_time_hours": round(total_time_minutes / 60, 2)
                }
            }
            
            logger.info(
                f"Route calculated: {origin_id} -> {destination_id}, "
                f"{len(segments)} segments, {total_distance_km:.2f} km, "
                f"{total_time_minutes:.2f} min"
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
