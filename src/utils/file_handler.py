"""
File handling utilities for the Atlas project.

This module provides utilities for file operations, data validation, and format detection.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd


class FileHandler:
    """Utility class for handling various file operations."""
    
    def __init__(self):
        self.supported_formats = ['.csv', '.json', '.xlsx', '.txt']
    
    def is_supported_format(self, file_path: Path) -> bool:
        """
        Check if the file format is supported.
        
        Args:
            file_path (Path): Path to the file
            
        Returns:
            bool: True if format is supported
        """
        return file_path.suffix.lower() in self.supported_formats
    
    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """
        Get information about a file.
        
        Args:
            file_path (Path): Path to the file
            
        Returns:
            Dict[str, Any]: File information
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        stat = file_path.stat()
        
        return {
            'name': file_path.name,
            'size_bytes': stat.st_size,
            'size_mb': round(stat.st_size / (1024 * 1024), 2),
            'extension': file_path.suffix.lower(),
            'modified_time': stat.st_mtime,
            'is_supported': self.is_supported_format(file_path)
        }
    
    def validate_directory(self, dir_path: Path, create_if_missing: bool = True) -> bool:
        """
        Validate that a directory exists and is accessible.
        
        Args:
            dir_path (Path): Path to the directory
            create_if_missing (bool): Whether to create directory if it doesn't exist
            
        Returns:
            bool: True if directory is valid and accessible
        """
        if not dir_path.exists():
            if create_if_missing:
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    return True
                except Exception:
                    return False
            else:
                return False
        
        return dir_path.is_dir()
    
    def scan_directory(self, dir_path: Path, extensions: Optional[List[str]] = None) -> List[Path]:
        """
        Scan directory for files with specified extensions.
        
        Args:
            dir_path (Path): Directory to scan
            extensions (List[str], optional): List of extensions to filter by
            
        Returns:
            List[Path]: List of matching files
        """
        if not self.validate_directory(dir_path, create_if_missing=False):
            raise ValueError(f"Invalid directory: {dir_path}")
        
        if extensions is None:
            extensions = self.supported_formats
        
        files = []
        for ext in extensions:
            pattern = f"*{ext}"
            files.extend(dir_path.glob(pattern))
        
        return sorted(files)
    
    def read_config_file(self, config_path: Path) -> Dict[str, Any]:
        """
        Read configuration from JSON file.
        
        Args:
            config_path (Path): Path to configuration file
            
        Returns:
            Dict[str, Any]: Configuration data
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to read config file {config_path}: {e}")
    
    def write_config_file(self, config_path: Path, config_data: Dict[str, Any]) -> None:
        """
        Write configuration to JSON file.
        
        Args:
            config_path (Path): Path to configuration file
            config_data (Dict[str, Any]): Configuration data to write
        """
        try:
            # Ensure parent directory exists
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise ValueError(f"Failed to write config file {config_path}: {e}")
    
    def create_sample_data_file(self, output_path: Path, num_rows: int = 100) -> None:
        """
        Create a sample data file for testing purposes.
        
        Args:
            output_path (Path): Path where to save the sample file
            num_rows (int): Number of sample rows to generate
        """
        import random
        from datetime import datetime, timedelta
        
        # Generate sample delivery data
        sample_data = []
        
        for i in range(num_rows):
            # Random coordinates (roughly in Spain)
            pickup_lat = round(random.uniform(36.0, 43.5), 6)
            pickup_lon = round(random.uniform(-9.0, 3.0), 6)
            delivery_lat = round(random.uniform(36.0, 43.5), 6)
            delivery_lon = round(random.uniform(-9.0, 3.0), 6)
            
            # Random time windows
            base_time = datetime.now() + timedelta(days=random.randint(1, 30))
            pickup_start = base_time.strftime('%Y-%m-%d %H:%M:%S')
            pickup_end = (base_time + timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S')
            delivery_start = (base_time + timedelta(hours=3)).strftime('%Y-%m-%d %H:%M:%S')
            delivery_end = (base_time + timedelta(hours=6)).strftime('%Y-%m-%d %H:%M:%S')
            
            sample_data.append({
                'delivery_id': f'DEL_{i+1:04d}',
                'pickup_latitude': pickup_lat,
                'pickup_longitude': pickup_lon,
                'delivery_latitude': delivery_lat,
                'delivery_longitude': delivery_lon,
                'pickup_address': f'Pickup Address {i+1}',
                'delivery_address': f'Delivery Address {i+1}',
                'package_weight': round(random.uniform(0.5, 50.0), 2),
                'package_volume': round(random.uniform(0.01, 2.0), 3),
                'delivery_deadline': (base_time + timedelta(days=1)).strftime('%Y-%m-%d'),
                'pickup_time_window_start': pickup_start,
                'pickup_time_window_end': pickup_end,
                'delivery_time_window_start': delivery_start,
                'delivery_time_window_end': delivery_end,
                'priority': random.choice(['low', 'normal', 'high', 'urgent']),
                'package_type': random.choice(['documents', 'electronics', 'clothing', 'food', 'general'])
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(sample_data)
        
        if output_path.suffix.lower() == '.csv':
            df.to_csv(output_path, index=False)
        elif output_path.suffix.lower() == '.json':
            df.to_json(output_path, orient='records', indent=2)
        elif output_path.suffix.lower() == '.xlsx':
            df.to_excel(output_path, index=False)
        else:
            raise ValueError(f"Unsupported output format: {output_path.suffix}")
