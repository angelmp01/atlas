"""
Mock and real data providers for ATLAS API.

Provides locations and goods types, either from database or mock data.
"""
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def get_locations_from_db(db_dsn: str) -> List[Dict[str, Any]]:
    """
    Get locations from database.
    
    Returns list of locations with id, name, latitude, longitude.
    Raises exception if database fails (no silent fallback).
    """
    from sqlalchemy import create_engine, text
    
    engine = create_engine(db_dsn)
    
    # Query locations from app.sodd_locations table
    # Columnas reales: location_id, location_name, longitude, latitude, geom
    query = text("""
        SELECT DISTINCT
            location_id as id,
            location_name as name,
            latitude,
            longitude
        FROM app.sodd_locations
        WHERE location_name IS NOT NULL
        ORDER BY location_name
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query)
        locations = []
        
        for row in result:
            locations.append({
                "id": row.id,
                "name": row.name,
                "latitude": float(row.latitude) if row.latitude else None,
                "longitude": float(row.longitude) if row.longitude else None,
            })
        
        logger.info(f"Loaded {len(locations)} locations from database")
        return locations


def get_mock_locations() -> List[Dict[str, Any]]:
    """
    Return mock locations for testing (Catalan capitals).
    """
    return [
        {"id": "08019", "name": "Barcelona", "latitude": 41.3851, "longitude": 2.1734},
        {"id": "25120", "name": "Lleida", "latitude": 41.6176, "longitude": 0.6200},
        {"id": "43148", "name": "Tarragona", "latitude": 41.1189, "longitude": 1.2445},
        {"id": "17079", "name": "Girona", "latitude": 41.9794, "longitude": 2.8214},
    ]


def get_goods_types() -> List[Dict[str, Any]]:
    """
    Return available goods types.
    """
    return [
        {"id": "normal", "name": "Normal", "description": "Mercancía estándar"},
        {"id": "refrigerada", "name": "Refrigerada", "description": "Mercancía refrigerada"},
    ]


def get_truck_types() -> List[Dict[str, Any]]:
    """
    Return available truck types.
    """
    return [
        {"id": "normal", "name": "Normal", "description": "Camión estándar"},
        {"id": "refrigerado", "name": "Refrigerado", "description": "Camión frigorífico"},
    ]
