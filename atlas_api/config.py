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

# Database Configuration
# Priority: 1) Full DSN, 2) Individual components, 3) Default
if os.getenv("PG_DSN"):
    DB_DSN = os.getenv("PG_DSN")
elif all([os.getenv("DB_HOST"), os.getenv("DB_USER"), os.getenv("DB_PASSWORD"), os.getenv("DB_NAME")]):
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_DSN = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
else:
    # Default for local development
    DB_DSN = "postgresql://appuser:pgai1234@atlasproject.duckdns.org:25432/gisdb"

# CORS Configuration (allow web frontend)
# Can be overridden with CORS_ORIGINS environment variable (comma-separated)
cors_origins_env = os.getenv("CORS_ORIGINS")
if cors_origins_env:
    CORS_ORIGINS = [origin.strip() for origin in cors_origins_env.split(",")]
else:
    CORS_ORIGINS = [
        "http://localhost:8001",
        "http://127.0.0.1:8001",
        "http://localhost:3000",  # Development
        "https://atlasproject.duckdns.org",  # Production web
        "https://api.atlasproject.duckdns.org",  # Production API (for docs)
    ]

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent

# ML Models directory (configurable via environment variable)
MODELS_DIR_ENV = os.getenv("MODELS_DIR")
if MODELS_DIR_ENV:
    MODELS_DIR = Path(MODELS_DIR_ENV)
else:
    MODELS_DIR = PROJECT_ROOT / "models"
