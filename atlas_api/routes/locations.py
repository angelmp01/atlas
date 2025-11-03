"""
Locations endpoint for ATLAS API.

Returns list of available locations for origin and destination selection.
"""
from fastapi import APIRouter, HTTPException, Response
from typing import List, Dict, Any
import logging

from ..data.mock_data import get_locations_from_db
from .. import config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/locations", tags=["locations"])


@router.get("", response_model=List[Dict[str, Any]])
async def get_locations(response: Response):
    """
    Get list of available locations.
    
    Returns:
        List of locations with id, name, latitude, longitude
        
    Example:
        GET /locations
        [
            {"id": "08019", "name": "Barcelona", "latitude": 41.3851, "longitude": 2.1734},
            {"id": "25120", "name": "Lleida", "latitude": 41.6176, "longitude": 0.6200}
        ]
    """
    try:
        # Set cache headers - cache for 24 hours (locations don't change)
        response.headers["Cache-Control"] = "public, max-age=86400"
        response.headers["ETag"] = "locations-v1"
        
        locations = get_locations_from_db(config.DB_DSN)
        logger.info(f"Returning {len(locations)} locations from database")
        return locations
    except Exception as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al cargar localizaciones de la base de datos: {str(e)}"
        )
