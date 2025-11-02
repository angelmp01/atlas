"""
OSM2PO-based Route Optimizer

Calculates optimal routes using pgRouting with the OSM2PO graph.
Uses parametrized SQL queries for safe database access.

New graph schema (imported from OSM2PO):
    - Table: app.catalunya_truck_2po_4pgr
      Columns: id, source, target, cost, reverse_cost, geom_way
    - Table: app.catalunya_truck_2po_vertex
      Columns: id, geom_vertex

Output format:
    {
        "distance_km": float,
        "duration_min": float,
        "geojson": GeoJSON LineString or FeatureCollection
    }
"""

import logging
import json
from typing import Dict, Tuple, Optional, Any
from sqlalchemy import create_engine, text, exc

logger = logging.getLogger(__name__)


class OSM2PORouter:
    """
    Route calculation using OSM2PO imported graph with pgRouting.
    
    This class handles:
    - Finding nearest graph vertices from coordinates
    - Calculating shortest paths using pgr_dijkstra
    - Building GeoJSON responses
    """
    
    def __init__(self, db_dsn: str):
        """
        Initialize the router with database connection.
        
        Args:
            db_dsn: PostgreSQL DSN (connection string)
        """
        self.db_dsn = db_dsn
        self.engine = create_engine(db_dsn)
        self.edge_table = "app.catalunya_truck_2po_4pgr"
        self.vertex_table = "app.catalunya_truck_2po_vertex"
        
        logger.info(
            f"OSM2PORouter initialized - "
            f"Edge table: {self.edge_table}, "
            f"Vertex table: {self.vertex_table}"
        )
    
    def calculate_route(
        self,
        lon1: float,
        lat1: float,
        lon2: float,
        lat2: float
    ) -> Dict[str, Any]:
        """
        Calculate optimal route between two coordinates.
        
        This function:
        1. Finds the nearest graph vertices to origin and destination
        2. Uses pgr_dijkstra to find the shortest path
        3. Returns distance, duration, and GeoJSON geometry
        
        Args:
            lon1: Origin longitude
            lat1: Origin latitude
            lon2: Destination longitude
            lat2: Destination latitude
            
        Returns:
            Dictionary with:
            - distance_km: Total distance in kilometers
            - duration_min: Total duration in minutes
            - geojson: GeoJSON LineString of the route geometry
            
        Raises:
            ValueError: If coordinates are invalid or route not found
            Exception: If database connection fails
        """
        logger.info(
            f"Starting route calculation: "
            f"({lon1:.4f}, {lat1:.4f}) -> ({lon2:.4f}, {lat2:.4f})"
        )
        
        try:
            # 1. Validate coordinates
            if not self._validate_coordinates(lon1, lat1, lon2, lat2):
                raise ValueError("Invalid coordinates provided")
            
            # 2. Find nearest vertices
            origin_vertex = self._find_nearest_vertex(lon1, lat1)
            dest_vertex = self._find_nearest_vertex(lon2, lat2)
            
            if not origin_vertex or not dest_vertex:
                raise ValueError(
                    f"Could not find vertices for coordinates: "
                    f"origin={origin_vertex}, dest={dest_vertex}"
                )
            
            logger.info(
                f"Found vertices - origin: {origin_vertex}, destination: {dest_vertex}"
            )
            
            # 3. Calculate route using pgr_dijkstra
            route_result = self._dijkstra_route(origin_vertex, dest_vertex)
            
            if not route_result:
                raise ValueError(
                    f"No route found between vertices {origin_vertex} and {dest_vertex}"
                )
            
            # 4. Build response
            response = {
                "distance_km": round(route_result["distance_km"], 2),
                "duration_min": round(route_result["duration_min"], 2),
                "geojson": route_result["geojson"]
            }
            
            logger.info(
                f"Route calculated successfully: "
                f"{response['distance_km']} km, "
                f"{response['duration_min']} min"
            )
            
            return response
            
        except ValueError as e:
            logger.error(f"Validation error in route calculation: {e}")
            raise
        except exc.SQLAlchemyError as e:
            logger.error(f"Database error in route calculation: {e}")
            raise Exception(f"Database error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in route calculation: {e}")
            raise
    
    def _validate_coordinates(
        self,
        lon1: float,
        lat1: float,
        lon2: float,
        lat2: float
    ) -> bool:
        """
        Validate coordinate ranges (WGS84 / EPSG:4326).
        
        Args:
            lon1, lat1, lon2, lat2: Coordinates to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Longitude: -180 to 180
            # Latitude: -90 to 90
            assert -180 <= lon1 <= 180, f"Invalid lon1: {lon1}"
            assert -90 <= lat1 <= 90, f"Invalid lat1: {lat1}"
            assert -180 <= lon2 <= 180, f"Invalid lon2: {lon2}"
            assert -90 <= lat2 <= 90, f"Invalid lat2: {lat2}"
            
            # Make sure origin and destination are different
            if (lon1, lat1) == (lon2, lat2):
                raise ValueError("Origin and destination cannot be the same")
            
            return True
        except (AssertionError, ValueError) as e:
            logger.warning(f"Coordinate validation failed: {e}")
            return False
    
    def _find_nearest_vertex(self, lon: float, lat: float) -> Optional[int]:
        """
        Find the nearest graph vertex to given coordinates.
        
        Uses PostgreSQL distance operator (<->) to find the closest vertex.
        
        Args:
            lon: Longitude (EPSG:4326)
            lat: Latitude (EPSG:4326)
            
        Returns:
            Vertex ID or None if not found
        """
        try:
            with self.engine.connect() as conn:
                query = text(f"""
                    SELECT id
                    FROM {self.vertex_table}
                    ORDER BY geom_vertex <-> ST_SetSRID(
                        ST_MakePoint(:lon, :lat), 4326
                    )
                    LIMIT 1
                """)
                
                result = conn.execute(
                    query,
                    {"lon": lon, "lat": lat}
                ).scalar()
                
                if result is None:
                    logger.warning(
                        f"No vertex found near ({lon:.4f}, {lat:.4f})"
                    )
                
                return result
                
        except exc.SQLAlchemyError as e:
            logger.error(f"Database error finding nearest vertex: {e}")
            return None
    
    def _dijkstra_route(self, source: int, target: int) -> Optional[Dict[str, Any]]:
        """
        Calculate shortest path using pgr_dijkstra.
        
        Args:
            source: Source vertex ID
            target: Target vertex ID
            
        Returns:
            Dictionary with distance_km, duration_min, and geojson
            or None if route not found
        """
        try:
            with self.engine.connect() as conn:
                # Query using pgr_dijkstra
                # directed=true means we respect the cost direction (one-way streets)
                query = text(f"""
                    WITH route AS (
                        SELECT * FROM pgr_dijkstra(
                            'SELECT id, source, target, cost, reverse_cost 
                             FROM {self.edge_table}',
                            :source, :target, true
                        )
                    )
                    SELECT
                        SUM(ST_Length(e.geom_way::geography))/1000 AS total_km,
                        SUM(e.cost)*60.0                           AS total_min,
                        ST_LineMerge(ST_Union(e.geom_way))         AS geom_route
                    FROM route r
                    JOIN {self.edge_table} e ON e.id = r.edge
                    WHERE r.edge <> -1
                """)
                
                result = conn.execute(
                    query,
                    {"source": source, "target": target}
                ).fetchone()
                
                if not result or result.geom_route is None:
                    logger.warning(
                        f"Dijkstra returned no route between {source} -> {target}"
                    )
                    return None
                
                # Convert PostGIS geometry to GeoJSON
                geom_geojson = self._geometry_to_geojson(result.geom_route)
                
                return {
                    "distance_km": float(result.total_km) if result.total_km else 0,
                    "duration_min": float(result.total_min) if result.total_min else 0,
                    "geojson": geom_geojson
                }
                
        except exc.SQLAlchemyError as e:
            logger.error(f"Database error in dijkstra calculation: {e}")
            return None
    
    def _geometry_to_geojson(self, geometry: Any) -> Dict[str, Any]:
        """
        Convert PostGIS geometry to GeoJSON.
        
        Args:
            geometry: PostGIS geometry object
            
        Returns:
            GeoJSON dictionary (LineString)
        """
        try:
            # Get the raw GeoJSON string from the geometry
            # PostGIS ST_AsGeoJSON would give us the string directly
            # But we can also work with the geometry object if it has a GeoJSON method
            
            if hasattr(geometry, '__geo_interface__'):
                # If the geometry object supports __geo_interface__
                geo_dict = geometry.__geo_interface__
            else:
                # Otherwise try to convert to string and parse
                geo_str = str(geometry)
                geo_dict = json.loads(geo_str)
            
            return geo_dict
            
        except Exception as e:
            logger.warning(f"Error converting geometry to GeoJSON: {e}")
            # Return a minimal valid GeoJSON on error
            return {
                "type": "LineString",
                "coordinates": []
            }
    
    def calculate_route_with_sql_function(
        self,
        lon1: float,
        lat1: float,
        lon2: float,
        lat2: float
    ) -> Dict[str, Any]:
        """
        Alternative method: Use SQL function app.route_by_coords().
        
        This is more efficient if the function is pre-created in the database.
        
        Args:
            lon1, lat1, lon2, lat2: Coordinates
            
        Returns:
            Dictionary with distance_km, duration_min, and geojson
        """
        logger.info(
            f"Using SQL function method for route: "
            f"({lon1:.4f}, {lat1:.4f}) -> ({lon2:.4f}, {lat2:.4f})"
        )
        
        try:
            with self.engine.connect() as conn:
                query = text("""
                    SELECT distance_km, duration_min, geojson
                    FROM app.route_by_coords(:lon1, :lat1, :lon2, :lat2)
                """)
                
                result = conn.execute(
                    query,
                    {
                        "lon1": lon1,
                        "lat1": lat1,
                        "lon2": lon2,
                        "lat2": lat2
                    }
                ).fetchone()
                
                if not result:
                    raise ValueError("SQL function returned no result")
                
                return {
                    "distance_km": round(float(result.distance_km), 2),
                    "duration_min": round(float(result.duration_min), 2),
                    "geojson": json.loads(result.geojson) if isinstance(result.geojson, str) else result.geojson
                }
                
        except exc.SQLAlchemyError as e:
            logger.error(f"Database error using SQL function: {e}")
            # Fallback to direct method
            logger.info("Falling back to direct dijkstra method")
            return self.calculate_route(lon1, lat1, lon2, lat2)
