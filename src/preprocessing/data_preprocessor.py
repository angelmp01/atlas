"""
Atlas Data Preprocessing Module

This module handles the preprocessing of raw transportation and delivery data
for the Atlas analytics system. It reads various raw data formats
and converts them into a standardized CSV format for further processing.

Author: Atlas Team
Date: July 2025
"""

import os
import sys
import pandas as pd
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import argparse

from utils.logger import setup_logger
from utils.file_handler import FileHandler


class DataPreprocessor:
    """
    Main class for preprocessing raw transportation and delivery data.
    
    This class handles multiple data sources and formats, converting them
    into a standardized format suitable for analytics and processing.
    """
    
    def __init__(self, input_dir: str, output_dir: str, config_path: Optional[str] = None):
        """
        Initialize the DataPreprocessor.
        
        Args:
            input_dir (str): Path to directory containing raw data files
            output_dir (str): Path to directory where processed CSV will be saved
            config_path (str, optional): Path to configuration file
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.config_path = config_path
        
        # Ensure directories exist
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logger('data_preprocessor', 'preprocessing.log')
        
        # Initialize file handler
        self.file_handler = FileHandler()
        
        # Load configuration if provided
        self.config = self._load_config() if config_path else self._get_default_config()
        
        self.logger.info(f"DataPreprocessor initialized with input_dir: {self.input_dir}, output_dir: {self.output_dir}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        # Start with default config
        config = self._get_default_config()
        
        if self.config_path is None:
            self.logger.warning("No config path provided. Using default config.")
            return config
            
        try:
            with open(self.config_path, 'r') as f:
                loaded_config = json.load(f)
            
            # Merge loaded config with defaults
            if 'preprocessing' in loaded_config:
                config.update(loaded_config['preprocessing'])
            
            self.logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            self.logger.warning(f"Failed to load config from {self.config_path}: {e}. Using default config.")
            return config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for preprocessing."""
        return {
            "supported_formats": [".csv", ".json", ".xlsx", ".txt"],
            "output_filename": "processed_delivery_data.csv",
            "required_columns": [
                "delivery_id",
                "pickup_latitude",
                "pickup_longitude", 
                "delivery_latitude",
                "delivery_longitude",
                "pickup_address",
                "delivery_address",
                "package_weight",
                "package_volume",
                "delivery_deadline",
                "pickup_time_window_start",
                "pickup_time_window_end",
                "delivery_time_window_start", 
                "delivery_time_window_end",
                "priority",
                "package_type"
            ],
            "coordinate_precision": 6,
            "weight_unit": "kg",
            "volume_unit": "m3"
        }
    
    def discover_data_files(self) -> List[Path]:
        """
        Discover all supported data files in the input directory.
        
        Returns:
            List[Path]: List of paths to discovered data files
        """
        supported_extensions = self.config["supported_formats"]
        data_files = []
        
        for ext in supported_extensions:
            # Use ** to search recursively in subdirectories
            pattern = f"**/*{ext}"
            files = list(self.input_dir.glob(pattern))
            data_files.extend(files)
        
        self.logger.info(f"Discovered {len(data_files)} data files: {[f.name for f in data_files]}")
        return data_files
    
    def read_data_file(self, file_path: Path) -> pd.DataFrame:
        """
        Read a data file based on its extension.
        
        Args:
            file_path (Path): Path to the data file
            
        Returns:
            pd.DataFrame: DataFrame containing the file data
        """
        try:
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix.lower() == '.json':
                df = self._read_json_file(file_path)
            elif file_path.suffix.lower() == '.xlsx':
                df = pd.read_excel(file_path)
            elif file_path.suffix.lower() == '.txt':
                # Assume tab-separated or detect delimiter
                df = pd.read_csv(file_path, sep='\t')
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            self.logger.info(f"Successfully read {file_path.name}: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to read file {file_path}: {e}")
            raise
    
    def _read_json_file(self, file_path: Path) -> pd.DataFrame:
        """
        Read JSON file with intelligent structure detection.
        
        This method handles different JSON structures:
        - Simple JSON arrays
        - Flat JSON objects
        - GeoJSON-like structures with features array
        - ESRI JSON format with features
        
        Args:
            file_path (Path): Path to the JSON file
            
        Returns:
            pd.DataFrame: DataFrame containing the JSON data
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Case 1: Data is already a list (simple JSON array)
            if isinstance(data, list):
                return pd.DataFrame(data)
            
            # Case 2: Data is a dictionary
            if isinstance(data, dict):
                # Check for GeoJSON-like structure with 'features' key
                if 'features' in data:
                    features = data['features']
                    
                    # Extract attributes from each feature
                    records = []
                    for feature in features:
                        if isinstance(feature, dict):
                            # Try to get attributes (common in ESRI JSON)
                            if 'attributes' in feature:
                                record = feature['attributes'].copy()
                            else:
                                record = feature.copy()
                            
                            # Add geometry information if available
                            if 'geometry' in feature and feature['geometry']:
                                geom = feature['geometry']
                                if isinstance(geom, dict):
                                    # Add basic geometry info
                                    record['geometry_type'] = geom.get('type', 'unknown')
                                    
                                    # For coordinate extraction (simplified)
                                    if 'coordinates' in geom:
                                        coords = geom['coordinates']
                                        if coords and isinstance(coords, list):
                                            if len(coords) > 0 and isinstance(coords[0], list):
                                                # Multi-dimensional coordinates (polyline, polygon)
                                                record['has_geometry'] = True
                                                record['coordinate_count'] = len(coords)
                                            else:
                                                # Simple point coordinates
                                                if len(coords) >= 2:
                                                    record['longitude'] = coords[0]
                                                    record['latitude'] = coords[1]
                            
                            records.append(record)
                    
                    if records:
                        df = pd.DataFrame(records)
                        self.logger.info(f"Extracted {len(records)} features from GeoJSON-like structure")
                        return df
                
                # Case 3: Try to use the dictionary directly as a single record
                try:
                    return pd.DataFrame([data])
                except:
                    # Case 4: Try to normalize nested JSON
                    return pd.json_normalize(data)
            
            # Fallback: try pandas read_json
            return pd.read_json(file_path)
            
        except Exception as e:
            self.logger.warning(f"Custom JSON parsing failed for {file_path}: {e}")
            # Fallback to pandas read_json
            try:
                return pd.read_json(file_path)
            except Exception as e2:
                self.logger.error(f"All JSON parsing methods failed for {file_path}: {e2}")
                raise Exception(f"Could not parse JSON file {file_path}: {e2}")
    
    def standardize_columns(self, df: pd.DataFrame, source_file: str) -> pd.DataFrame:
        """
        Standardize column names and data types.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            source_file (str): Name of source file for logging
            
        Returns:
            pd.DataFrame: DataFrame with standardized columns
        """
        # Create a copy to avoid modifying original
        standardized_df = df.copy()
        
        # Add source file column
        standardized_df['source_file'] = source_file
        standardized_df['processed_timestamp'] = datetime.now().isoformat()
        
        # Column mapping for common variations
        column_mapping = {
            # Delivery ID variations
            'id': 'delivery_id',
            'delivery_id': 'delivery_id',
            'order_id': 'delivery_id',
            'shipment_id': 'delivery_id',
            
            # Pickup location variations
            'pickup_lat': 'pickup_latitude',
            'pickup_lng': 'pickup_longitude',
            'pickup_lon': 'pickup_longitude',
            'origin_lat': 'pickup_latitude',
            'origin_lng': 'pickup_longitude',
            'origin_lon': 'pickup_longitude',
            
            # Delivery location variations
            'delivery_lat': 'delivery_latitude',
            'delivery_lng': 'delivery_longitude',
            'delivery_lon': 'delivery_longitude',
            'destination_lat': 'delivery_latitude',
            'destination_lng': 'delivery_longitude',
            'destination_lon': 'delivery_longitude',
            
            # Address variations
            'pickup_addr': 'pickup_address',
            'delivery_addr': 'delivery_address',
            'origin_address': 'pickup_address',
            'destination_address': 'delivery_address',
            
            # Package attributes
            'weight': 'package_weight',
            'volume': 'package_volume',
            'size': 'package_volume',
            'deadline': 'delivery_deadline',
            'due_date': 'delivery_deadline',
            
            # Time windows
            'pickup_start': 'pickup_time_window_start',
            'pickup_end': 'pickup_time_window_end',
            'delivery_start': 'delivery_time_window_start',
            'delivery_end': 'delivery_time_window_end',
        }
        
        # Apply column mapping
        standardized_df.columns = [column_mapping.get(col.lower(), col) for col in standardized_df.columns]
        
        self.logger.info(f"Standardized columns for {source_file}")
        return standardized_df
    
    def validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean the data.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        initial_rows = len(df)
        
        # Remove rows with missing critical coordinates
        critical_coords = ['pickup_latitude', 'pickup_longitude', 'delivery_latitude', 'delivery_longitude']
        df = df.dropna(subset=[col for col in critical_coords if col in df.columns])
        
        # Validate coordinate ranges
        coord_columns = [col for col in ['pickup_latitude', 'delivery_latitude'] if col in df.columns]
        for col in coord_columns:
            df = df[(df[col] >= -90) & (df[col] <= 90)]
        
        coord_columns = [col for col in ['pickup_longitude', 'delivery_longitude'] if col in df.columns]
        for col in coord_columns:
            df = df[(df[col] >= -180) & (df[col] <= 180)]
        
        # Round coordinates to specified precision
        precision = self.config["coordinate_precision"]
        for col in ['pickup_latitude', 'pickup_longitude', 'delivery_latitude', 'delivery_longitude']:
            if col in df.columns:
                df[col] = df[col].round(precision)
        
        # Fill missing values with defaults
        if 'package_weight' in df.columns:
            df['package_weight'] = df['package_weight'].fillna(1.0)  # Default 1kg
        
        if 'package_volume' in df.columns:
            df['package_volume'] = df['package_volume'].fillna(0.01)  # Default 0.01 m3
        
        if 'priority' in df.columns:
            df['priority'] = df['priority'].fillna('normal')
        
        if 'package_type' in df.columns:
            df['package_type'] = df['package_type'].fillna('general')
        
        final_rows = len(df)
        removed_rows = initial_rows - final_rows
        
        self.logger.info(f"Data cleaning: removed {removed_rows} invalid rows, {final_rows} rows remaining")
        
        return df
    
    def process_all_files(self) -> pd.DataFrame:
        """
        Process all discovered data files and combine them into a single DataFrame.
        
        Returns:
            pd.DataFrame: Combined and processed DataFrame
        """
        data_files = self.discover_data_files()
        
        if not data_files:
            raise ValueError(f"No supported data files found in {self.input_dir}")
        
        combined_data = []
        
        for file_path in data_files:
            try:
                # Read the file
                df = self.read_data_file(file_path)
                
                # Standardize columns
                df = self.standardize_columns(df, file_path.name)
                
                # Add to combined data
                combined_data.append(df)
                
            except Exception as e:
                self.logger.error(f"Failed to process file {file_path}: {e}")
                continue
        
        if not combined_data:
            raise ValueError("No files were successfully processed")
        
        # Combine all DataFrames
        combined_df = pd.concat(combined_data, ignore_index=True, sort=False)
        
        # Clean and validate the combined data
        combined_df = self.validate_and_clean_data(combined_df)
        
        self.logger.info(f"Successfully combined {len(data_files)} files into DataFrame with {len(combined_df)} rows")
        
        return combined_df
    
    def save_processed_data(self, df: pd.DataFrame) -> str:
        """
        Save the processed DataFrame to CSV.
        
        Args:
            df (pd.DataFrame): Processed DataFrame to save
            
        Returns:
            str: Path to the saved file
        """
        output_filename = self.config["output_filename"]
        output_path = self.output_dir / output_filename
        
        try:
            df.to_csv(output_path, index=False)
            self.logger.info(f"Processed data saved to {output_path}")
            
            # Log summary statistics
            self.logger.info(f"Output file summary: {len(df)} rows, {len(df.columns)} columns")
            self.logger.info(f"Columns: {list(df.columns)}")
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save processed data: {e}")
            raise
    
    def run(self) -> str:
        """
        Execute the complete preprocessing pipeline.
        
        Returns:
            str: Path to the output CSV file
        """
        self.logger.info("Starting data preprocessing pipeline")
        
        try:
            # Process all files
            processed_df = self.process_all_files()
            
            # Save the processed data
            output_path = self.save_processed_data(processed_df)
            
            self.logger.info("Data preprocessing pipeline completed successfully")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Data preprocessing pipeline failed: {e}")
            raise


def main():
    """Main function to run the data preprocessor from command line."""
    parser = argparse.ArgumentParser(description='Atlas Data Preprocessor')
    parser.add_argument('--input-dir', required=True, help='Directory containing raw data files')
    parser.add_argument('--output-dir', required=True, help='Directory to save processed CSV file')
    parser.add_argument('--config', help='Path to configuration file (optional)')
    
    args = parser.parse_args()
    
    try:
        preprocessor = DataPreprocessor(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            config_path=args.config
        )
        
        output_file = preprocessor.run()
        print(f"Preprocessing completed successfully. Output saved to: {output_file}")
        
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
