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
from datetime import datetime, date
import argparse
import calendar
import numpy as np

from utils.logger import setup_logger
from utils.file_handler import FileHandler


class DataPreprocessor:
    """
    Main class for preprocessing raw transportation and delivery data.
    
    This class handles multiple data sources and formats, converting them
    into a standardized format suitable for analytics and processing.
    """
    
    def __init__(self, input_dir: str, output_dir: str, config_path: Optional[str] = None, 
                 target_objectid: Optional[int] = None, distribution_mode: str = "exact", 
                 random_seed: Optional[int] = None):
        """
        Initialize the DataPreprocessor.
        
        Args:
            input_dir (str): Path to directory containing raw data files
            output_dir (str): Path to directory where processed CSV will be saved
            config_path (str, optional): Path to configuration file
            target_objectid (int, optional): Process only this specific OBJECTID for validation
            distribution_mode (str): Distribution mode for trip generation:
                - "exact": Generate exactly the average number of trips per day
                - "poisson": Use Poisson distribution around the average (realistic variability)
                - "normal": Use truncated normal distribution around the average
            random_seed (int, optional): Seed for random number generation for reproducibility
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.config_path = config_path
        self.target_objectid = target_objectid
        self.distribution_mode = distribution_mode.lower()
        
        # Validate distribution mode
        valid_modes = ["exact", "poisson", "normal"]
        if self.distribution_mode not in valid_modes:
            raise ValueError(f"distribution_mode must be one of {valid_modes}, got '{distribution_mode}'")
        
        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
            self.logger_info = f"Random seed set to {random_seed}"
        else:
            self.logger_info = "No random seed set"
        
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
        self.logger.info(f"Distribution mode: {self.distribution_mode}")
        if self.target_objectid:
            self.logger.info(f"Processing only OBJECTID: {self.target_objectid} (validation mode)")
    
    def _generate_trips_for_day(self, average_trips: float) -> int:
        """
        Generate number of trips for a day based on distribution mode.
        
        Args:
            average_trips (float): Average number of trips for this day type
            
        Returns:
            int: Actual number of trips to generate
        """
        if average_trips <= 0:
            return 0
            
        if self.distribution_mode == "exact":
            return int(average_trips)
            
        elif self.distribution_mode == "poisson":
            # Poisson distribution is ideal for counting events (trips)
            return np.random.poisson(average_trips)
            
        elif self.distribution_mode == "normal":
            # Normal distribution with standard deviation = mean/3 (99.7% within reasonable range)
            std_dev = max(0.5, average_trips / 3)  # Minimum std of 0.5 to avoid too narrow distribution
            trips = np.random.normal(average_trips, std_dev)
            return max(0, int(round(trips)))  # Ensure non-negative integer
            
        else:
            # Fallback to exact
            return int(average_trips)
    
    def _generate_cargo_type(self) -> str:
        """
        Generate cargo type based on configuration percentages.
        
        Returns:
            str: Either 'normal' or 'refrigerada'
        """
        normal_percentage = self.config.get('cargo_types', {}).get('normal_percentage', 70)
        
        # Generate random number between 0-100
        random_value = np.random.uniform(0, 100)
        
        if random_value <= normal_percentage:
            return 'normal'
        else:
            return 'refrigerada'
    
    def _generate_volume(self) -> int:
        """
        Generate random volume in palets based on truck capacity.
        
        Returns:
            int: Number of palets (between min and max capacity)
        """
        min_palets = self.config.get('truck_capacity', {}).get('min_palets', 1)
        max_palets = self.config.get('truck_capacity', {}).get('max_palets', 10)
        
        return np.random.randint(min_palets, max_palets + 1)
    
    def _generate_weight(self, volumen_palets: int) -> int:
        """
        Generate weight based on volume with some variation.
        
        Args:
            volumen_palets (int): Number of palets
            
        Returns:
            int: Weight in kilograms (integer)
        """
        min_kg_per_palet = self.config.get('weight_per_palet', {}).get('min_kg', 50)
        max_kg_per_palet = self.config.get('weight_per_palet', {}).get('max_kg', 800)
        
        # Generate weight per palet with variation
        weight_per_palet = np.random.uniform(min_kg_per_palet, max_kg_per_palet)
        total_weight = weight_per_palet * volumen_palets
        
        # Return as integer
        return int(round(total_weight))
    
    def _generate_price(self, shape_length: float) -> int:
        """
        Generate price based on route distance with variation.
        
        Args:
            shape_length (float): Route distance in meters
            
        Returns:
            int: Price in euros (integer)
        """
        if shape_length is None or shape_length <= 0:
            shape_length = 50000  # Default 50km if no distance available
        
        # Convert meters to kilometers
        distance_km = shape_length / 1000
        
        # Get pricing configuration
        base_price_per_km = self.config.get('pricing', {}).get('base_price_per_km', 1.2)
        variation_percent = self.config.get('pricing', {}).get('price_variation_percent', 20)
        
        # Calculate base price
        base_price = distance_km * base_price_per_km
        
        # Add variation (±20% by default)
        variation_factor = np.random.uniform(
            1 - (variation_percent / 100), 
            1 + (variation_percent / 100)
        )
        
        final_price = base_price * variation_factor
        
        # Return as integer
        return int(round(final_price))
    
    def _generate_2024_dates(self) -> List[Dict[str, Any]]:
        """
        Generate all dates for 2024 with their corresponding day of week.
        
        Returns:
            List[Dict]: List of dictionaries with date and day_name
        """
        dates_2024 = []
        start_date = date(2024, 1, 1)
        end_date = date(2024, 12, 31)
        
        current_date = start_date
        while current_date <= end_date:
            day_name = calendar.day_name[current_date.weekday()]
            # Map to Spanish day names as used in the JSON
            day_mapping = {
                'Monday': 'Lunes',
                'Tuesday': 'Martes', 
                'Wednesday': 'Miercoles',
                'Thursday': 'Jueves',
                'Friday': 'Viernes',
                'Saturday': 'Sabado',
                'Sunday': 'Domingo'
            }
            
            dates_2024.append({
                'date': current_date,
                'day_name_spanish': day_mapping[day_name],
                'day_name_english': day_name
            })
            
            # Move to next day
            if current_date.month == 12 and current_date.day == 31:
                break
            elif current_date.day == calendar.monthrange(current_date.year, current_date.month)[1]:
                current_date = current_date.replace(month=current_date.month + 1, day=1)
            else:
                current_date = current_date.replace(day=current_date.day + 1)
        
        return dates_2024
    
    def _expand_feature_to_trips(self, feature: Dict[str, Any], dates_2024: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Expand a single feature into individual trip records for 2024.
        
        Args:
            feature (Dict): Feature from JSON with attributes and geometry
            dates_2024 (List): List of all 2024 dates with day names
            
        Returns:
            List[Dict]: List of individual trip records
        """
        attributes = feature.get('attributes', {})
        geometry = feature.get('geometry', {})
        
        # Extract origin and destination coordinates if available
        origin_coords = None
        destination_coords = None
        
        if geometry and 'paths' in geometry and geometry['paths']:
            paths = geometry['paths']
            if len(paths) > 0 and len(paths[0]) > 0:
                # First coordinate as origin
                if len(paths[0][0]) >= 2:
                    origin_coords = {
                        'longitude': paths[0][0][0],
                        'latitude': paths[0][0][1]
                    }
                
                # Last coordinate as destination
                if len(paths[0][-1]) >= 2:
                    destination_coords = {
                        'longitude': paths[0][-1][0],
                        'latitude': paths[0][-1][1]
                    }
        
        # Day mapping for attribute names
        day_trips = {
            'Lunes': attributes.get('viajes_OD_Lunes', 0),
            'Martes': attributes.get('viajes_OD_Martes', 0),
            'Miercoles': attributes.get('viajes_OD_Miercoles', 0),
            'Jueves': attributes.get('viajes_OD_Jueves', 0),
            'Viernes': attributes.get('viajes_OD_Viernes', 0),
            'Sabado': attributes.get('viajes_OD_Sabado', 0),
            'Domingo': attributes.get('viajes_OD_Domingo', 0)
        }
        
        trips = []
        
        # For each date in 2024
        for date_info in dates_2024:
            day_spanish = date_info['day_name_spanish']
            average_trips = day_trips.get(day_spanish, 0)
            
            # Generate actual number of trips for this day based on distribution mode
            actual_trips = self._generate_trips_for_day(average_trips)
            
            # Create actual_trips individual records for this date
            for trip_num in range(actual_trips):
                # Generate cargo characteristics
                cargo_type = self._generate_cargo_type()
                volumen_palets = self._generate_volume()
                peso_kg = self._generate_weight(volumen_palets)
                precio_euros = self._generate_price(attributes.get('SHAPE_Length', 0))
                
                trip_record = {
                    # Original feature info
                    'objectid': attributes.get('OBJECTID'),
                    'fecha': date_info['date'].strftime('%Y-%m-%d'),
                    'dia_semana': day_spanish,
                    
                    # Origin information
                    'nombre_zona_origen': attributes.get('nombre_zona_origen'),
                    'cod_zona_origen': attributes.get('cod_zona_origen'),
                    'origen_longitude': origin_coords['longitude'] if origin_coords else None,
                    'origen_latitude': origin_coords['latitude'] if origin_coords else None,
                    
                    # Destination information
                    'nombre_zona_destino': attributes.get('nombre_zona_destino'),
                    'cod_zona_destino': attributes.get('cod_zona_destino'),
                    'destino_longitude': destination_coords['longitude'] if destination_coords else None,
                    'destino_latitude': destination_coords['latitude'] if destination_coords else None,
                    
                    # Cargo information (new columns)
                    'tipo_mercancia': cargo_type,
                    'volumen': volumen_palets,
                    'peso': peso_kg,
                    'precio': precio_euros
                }
                
                trips.append(trip_record)
        
        return trips
    
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
        Read JSON file with intelligent structure detection and expand to individual trips.
        
        This method handles different JSON structures and expands features into individual
        trip records for the year 2024 based on daily averages.
        
        Args:
            file_path (Path): Path to the JSON file
            
        Returns:
            pd.DataFrame: DataFrame containing expanded trip records
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Generate all dates for 2024
            dates_2024 = self._generate_2024_dates()
            self.logger.info(f"Generated {len(dates_2024)} dates for 2024")
            
            # Case 1: Data is already a list (simple JSON array)
            if isinstance(data, list):
                self.logger.warning("Simple JSON array detected. Hermes expansion not applicable.")
                return pd.DataFrame(data)
            
            # Case 2: Data is a dictionary - Check for GeoJSON-like structure with 'features'
            if isinstance(data, dict) and 'features' in data:
                features = data['features']
                
                # For memory efficiency, write progressively to CSV
                return self._process_features_progressively(features, dates_2024, file_path)
            
            # Fallback cases
            if isinstance(data, dict):
                try:
                    return pd.DataFrame([data])
                except:
                    return pd.json_normalize(data)
            
            # Final fallback
            return pd.read_json(file_path)
            
        except Exception as e:
            self.logger.error(f"All JSON parsing methods failed for {file_path}: {e}")
            raise Exception(f"Could not parse JSON file {file_path}: {e}")
    
    def _process_features_progressively(self, features: List[Dict], dates_2024: List[Dict], file_path: Path) -> pd.DataFrame:
        """
        Process features progressively, writing to CSV in chunks to avoid memory issues.
        
        Args:
            features (List[Dict]): List of features from JSON
            dates_2024 (List[Dict]): List of all 2024 dates
            file_path (Path): Path to original JSON file
            
        Returns:
            pd.DataFrame: Summary DataFrame with basic statistics
        """
        # Determine output file path
        output_filename = self.config["output_filename"]
        name_parts = output_filename.rsplit('.', 1)
        if len(name_parts) == 2:
            base_name, extension = name_parts
            base_name = f"{base_name}_{self.distribution_mode}"
            
            if self.target_objectid is not None:
                base_name = f"{base_name}_objectid_{self.target_objectid}"
            else:
                base_name = f"{base_name}_full_dataset"
            
            output_filename = f"{base_name}.{extension}"
        else:
            output_filename = f"{output_filename}_{self.distribution_mode}"
            if self.target_objectid is not None:
                output_filename = f"{output_filename}_objectid_{self.target_objectid}"
            else:
                output_filename = f"{output_filename}_full_dataset"
        
        output_path = self.output_dir / output_filename
        
        # Progressive processing variables
        processed_features = 0
        total_trips_written = 0
        header_written = False
        batch_size = 50  # Process features in batches
        
        # Calculate total number of batches for progress tracking
        total_batches = (len(features) + batch_size - 1) // batch_size  # Ceiling division
        
        self.logger.info(f"Starting progressive processing of {len(features)} features in {total_batches} batches")
        
        for i in range(0, len(features), batch_size):
            batch_number = i // batch_size + 1
            batch_features = features[i:i + batch_size]
            batch_trips = []
            
            for feature in batch_features:
                if isinstance(feature, dict) and 'attributes' in feature:
                    attributes = feature['attributes']
                    objectid = attributes.get('OBJECTID')
                    
                    # If target_objectid is specified, only process that feature
                    if self.target_objectid is not None:
                        if objectid != self.target_objectid:
                            continue
                    
                    # Expand this feature to individual trips
                    feature_trips = self._expand_feature_to_trips(feature, dates_2024)
                    batch_trips.extend(feature_trips)
                    processed_features += 1
                    
                    # If we're in validation mode and found our target, break
                    if self.target_objectid is not None and objectid == self.target_objectid:
                        break
            
            # Write this batch to CSV
            if batch_trips:
                batch_df = pd.DataFrame(batch_trips)
                
                # Write to CSV (append mode after first batch)
                write_mode = 'w' if not header_written else 'a'
                write_header = not header_written
                
                batch_df.to_csv(output_path, mode=write_mode, header=write_header, index=False)
                
                total_trips_written += len(batch_trips)
                header_written = True
                
                self.logger.info(f"Batch {batch_number}/{total_batches}: Processed {len(batch_features)} features, "
                               f"wrote {len(batch_trips)} trips. Total: {total_trips_written} trips")
                
                # Clear batch from memory
                del batch_trips, batch_df
            
            # If we're in validation mode and found our target, break
            if self.target_objectid is not None and processed_features > 0:
                break
        
        self.logger.info(f"Progressive processing complete: {processed_features} features -> {total_trips_written} trip records")
        
        if self.target_objectid is not None:
            self.logger.info(f"Validation mode: processed only OBJECTID {self.target_objectid}")
        
        # Return a summary DataFrame for compatibility
        summary_data = {
            'processed_features': [processed_features],
            'total_trips': [total_trips_written],
            'output_file': [str(output_path)],
            'processing_mode': [f"{self.distribution_mode}_{'single_route' if self.target_objectid else 'full_dataset'}"]
        }
        
        return pd.DataFrame(summary_data)
    
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
            pd.DataFrame: Combined and processed DataFrame or summary info
        """
        data_files = self.discover_data_files()
        
        if not data_files:
            raise ValueError(f"No supported data files found in {self.input_dir}")
        
        combined_data = []
        progressive_processing_used = False
        
        for file_path in data_files:
            try:
                # Read the file
                df = self.read_data_file(file_path)
                
                # For Hermes JSON files, check if progressive processing was used
                if file_path.suffix.lower() == '.json' and not df.empty:
                    if 'processed_features' in df.columns:
                        # This is a summary from progressive processing
                        self.logger.info(f"Hermes JSON processed progressively: {df['processed_features'].iloc[0]} features -> {df['total_trips'].iloc[0]} trips")
                        progressive_processing_used = True
                        combined_data.append(df)
                    elif 'objectid' in df.columns:
                        # Standard processing - no need to add source file info
                        self.logger.info(f"Hermes JSON processed: {len(df)} trip records from {file_path.name}")
                        combined_data.append(df)
                    else:
                        # Other JSON format
                        df = self.standardize_columns(df, file_path.name)
                        combined_data.append(df)
                else:
                    # Standardize columns for other file types
                    df = self.standardize_columns(df, file_path.name)
                    combined_data.append(df)
                
            except Exception as e:
                self.logger.error(f"Failed to process file {file_path}: {e}")
                continue
        
        if not combined_data:
            raise ValueError("No files were successfully processed")
        
        # If progressive processing was used, return summary info
        if progressive_processing_used:
            combined_df = pd.concat(combined_data, ignore_index=True, sort=False)
            self.logger.info(f"Progressive processing completed for {len(data_files)} files")
            return combined_df
        
        # Standard processing - combine all DataFrames
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
        # Check if this is a summary from progressive processing
        if 'output_file' in df.columns and len(df) == 1:
            output_path = df['output_file'].iloc[0]
            
            # Log summary statistics for progressive processing
            processed_features = df['processed_features'].iloc[0]
            total_trips = df['total_trips'].iloc[0]
            processing_mode = df['processing_mode'].iloc[0]
            
            self.logger.info(f"Progressive processing completed: {output_path}")
            self.logger.info(f"Output file summary: {total_trips} rows (from {processed_features} features)")
            
            # Additional statistics for trip data
            self.logger.info(f"Processing mode: {processing_mode}")
            
            return str(output_path)
        
        # Standard processing - save DataFrame to CSV
        output_filename = self.config["output_filename"]
        
        # Split filename to insert distribution mode and scope info
        name_parts = output_filename.rsplit('.', 1)
        if len(name_parts) == 2:
            base_name, extension = name_parts
            # Add distribution mode to filename
            base_name = f"{base_name}_{self.distribution_mode}"
            
            # Add scope info based on whether we're processing a specific OBJECTID or full dataset
            if self.target_objectid is not None:
                base_name = f"{base_name}_objectid_{self.target_objectid}"
            else:
                base_name = f"{base_name}_full_dataset"
            
            output_filename = f"{base_name}.{extension}"
        else:
            # Add distribution mode to filename
            output_filename = f"{output_filename}_{self.distribution_mode}"
            
            # Add scope info based on whether we're processing a specific OBJECTID or full dataset
            if self.target_objectid is not None:
                output_filename = f"{output_filename}_objectid_{self.target_objectid}"
            else:
                output_filename = f"{output_filename}_full_dataset"
        
        output_path = self.output_dir / output_filename
        
        try:
            df.to_csv(output_path, index=False)
            self.logger.info(f"Processed data saved to {output_path}")
            
            # Log summary statistics
            self.logger.info(f"Output file summary: {len(df)} rows, {len(df.columns)} columns")
            self.logger.info(f"Columns: {list(df.columns)}")
            
            # Additional statistics for trip data
            if 'objectid' in df.columns:
                unique_features = df['objectid'].nunique()
                date_range = f"{df['fecha'].min()} to {df['fecha'].max()}" if 'fecha' in df.columns else 'N/A'
                self.logger.info(f"Trip data summary: {unique_features} unique routes, date range: {date_range}")
            
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
