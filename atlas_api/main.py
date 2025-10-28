"""
ATLAS API - Main application entry point.

FastAPI application providing REST endpoints for ATLAS web interface.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import sys

from . import config
from .routes import locations, goods_types, route

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="ATLAS API",
    description="API for ATLAS logistics optimization platform",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(locations.router)
app.include_router(goods_types.router)
app.include_router(route.router)


@app.get("/")
async def root():
    """
    Root endpoint - API health check.
    """
    return {
        "status": "ok",
        "message": "ATLAS API is running",
        "version": "0.1.0",
        "endpoints": [
            "/locations",
            "/goods-types",
            "/goods-types/truck-types",
        ]
    }


@app.get("/health")
async def health():
    """
    Health check endpoint.
    """
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting ATLAS API on {config.API_HOST}:{config.API_PORT}")
    logger.info(f"CORS enabled for: {config.CORS_ORIGINS}")
    
    uvicorn.run(
        "atlas_api.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.API_RELOAD
    )
