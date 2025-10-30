"""
Configuration for ATLAS Web.

Simple configuration for web server and API connection.
"""
import os

# Web Server Configuration
WEB_HOST = os.getenv("WEB_HOST", "127.0.0.1")
WEB_PORT = int(os.getenv("WEB_PORT", "8001"))
WEB_RELOAD = os.getenv("WEB_RELOAD", "true").lower() == "true"

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

# Map Configuration (Catalunya center)
MAP_CENTER_LAT = float(os.getenv("MAP_CENTER_LAT", "41.5"))
MAP_CENTER_LON = float(os.getenv("MAP_CENTER_LON", "1.5"))
MAP_ZOOM = int(os.getenv("MAP_ZOOM", "8"))


class Config:
    """Configuration class for easier access."""
    def __init__(self):
        self.HOST = WEB_HOST
        self.PORT = WEB_PORT
        self.DEBUG = WEB_RELOAD
        self.API_BASE_URL = API_BASE_URL
        self.MAP_CENTER_LAT = MAP_CENTER_LAT
        self.MAP_CENTER_LON = MAP_CENTER_LON
        self.MAP_ZOOM = MAP_ZOOM
