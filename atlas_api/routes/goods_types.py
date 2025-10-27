"""
Goods types endpoint for ATLAS API.

Returns list of available goods types for merchandise selection.
"""
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import logging

from ..data.mock_data import get_goods_types, get_truck_types

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/goods-types", tags=["goods"])


@router.get("", response_model=List[Dict[str, Any]])
async def get_goods():
    """
    Get list of available goods types.
    
    Returns:
        List of goods types with id, name, description
        
    Example:
        GET /goods-types
        [
            {"id": "normal", "name": "Normal", "description": "Mercancía estándar"},
            {"id": "refrigerada", "name": "Refrigerada", "description": "Mercancía refrigerada"}
        ]
    """
    try:
        goods = get_goods_types()
        logger.info(f"Returning {len(goods)} goods types")
        return goods
    except Exception as e:
        logger.error(f"Error getting goods types: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting goods types: {str(e)}")


@router.get("/truck-types", response_model=List[Dict[str, Any]])
async def get_trucks():
    """
    Get list of available truck types.
    
    Returns:
        List of truck types with id, name, description
        
    Example:
        GET /goods-types/truck-types
        [
            {"id": "normal", "name": "Normal", "description": "Camión estándar"},
            {"id": "refrigerado", "name": "Refrigerado", "description": "Camión frigorífico"}
        ]
    """
    try:
        trucks = get_truck_types()
        logger.info(f"Returning {len(trucks)} truck types")
        return trucks
    except Exception as e:
        logger.error(f"Error getting truck types: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting truck types: {str(e)}")
