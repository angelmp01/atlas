"""
Configuration for ATLAS API.

Simple configuration loading from environment variables or defaults.
"""
import os
from pathlib import Path

# API Configuration
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_RELOAD = os.getenv("API_RELOAD", "true").lower() == "true"

# Database Configuration (reuse existing config)
DB_DSN = os.getenv(
    "PG_DSN", 
    "postgresql://appuser:pgai1234@192.168.50.10:25432/gisdb"
)

# CORS Configuration (allow web frontend)
CORS_ORIGINS = [
    "http://localhost:8001",
    "http://127.0.0.1:8001",
    "http://localhost:3000",  # Development
]

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
