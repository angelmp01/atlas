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
