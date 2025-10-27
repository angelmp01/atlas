"""
ATLAS Web - Main application entry point.

FastAPI application serving web interface for ATLAS platform.
"""
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import logging
import sys
from pathlib import Path

from . import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="ATLAS Web",
    description="Web interface for ATLAS logistics platform",
    version="0.1.0",
)

# Setup paths
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Setup templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Render main page with form and map.
    """
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "api_base_url": config.API_BASE_URL
        }
    )


@app.get("/health")
async def health():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "api_url": config.API_BASE_URL
    }


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting ATLAS Web on {config.WEB_HOST}:{config.WEB_PORT}")
    logger.info(f"API Base URL: {config.API_BASE_URL}")
    
    uvicorn.run(
        "atlas_web.main:app",
        host=config.WEB_HOST,
        port=config.WEB_PORT,
        reload=config.WEB_RELOAD
    )
