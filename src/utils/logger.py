"""
Logging utilities for the Atlas project.

This module provides centralized logging configuration for all Atlas components.
"""

import logging
import os
from pathlib import Path
from datetime import datetime


def setup_logger(name: str, log_file: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name (str): Logger name
        log_file (str, optional): Log file name. If None, uses console only.
        level (int): Logging level
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(log_dir / log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_project_logger(module_name: str) -> logging.Logger:
    """
    Get a standardized logger for Atlas project modules.
    
    Args:
        module_name (str): Name of the module requesting the logger
        
    Returns:
        logging.Logger: Configured logger
    """
    log_file = f"atlas_{module_name}_{datetime.now().strftime('%Y%m%d')}.log"
    return setup_logger(f"atlas.{module_name}", log_file)
