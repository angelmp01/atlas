"""
Atlas SpanishODData Preprocessing Module

This module handles the preprocessing of SpanishODData CSV files by adding
synthetic cargo fields (tipo_mercancia, volumen, peso, precio) to existing
trip records without expanding rows. It processes all CSV files in a given
directory and generates corresponding processed files.

Author: Atlas Team
Date: October 2025
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
import numpy as np

from utils.logger import setup_logger
from utils.file_handler import FileHandler


class SpanishODDataPreprocessor:
    """
    Preprocessor for SpanishODData CSV files.
    
    This class processes all CSV files in a given directory, adding synthetic 
    cargo fields to existing trip records without expanding the number of rows. 
    Each row gets additional columns: tipo_mercancia, volumen, peso, precio.
    
    The processor maintains the original file names with a 'processed_' prefix
    and timestamp for the output files.
    """
    
    def __init__(self, input_dir: str, output_dir: str, config_path: Optional[str] = None, 
                 random_seed: Optional[int] = None):
        """
        Initialize the SpanishODDataPreprocessor.
        
        Args:
            input_dir (str): Path to directory containing SpanishODData CSV files
            output_dir (str): Path to directory where processed CSV will be saved
            config_path (str, optional): Path to configuration file
            random_seed (int, optional): Seed for random number generation for reproducibility
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.config_path = config_path
        
        # Validate input directory exists
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        if not self.input_dir.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {self.input_dir}")
        
        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
            self.logger_info = f"Random seed set to {random_seed}"
        else:
            self.logger_info = "No random seed set"
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logger('spanishoddata_preprocessor', 'spanishoddata_preprocessing.log')
        
        # Initialize file handler
        self.file_handler = FileHandler()
        
        # Load configuration if provided
        self.config = self._load_config() if config_path else self._get_default_config()
        
        self.logger.info(f"SpanishODDataPreprocessor initialized with input_dir: {self.input_dir}, output_dir: {self.output_dir}")
        self.logger.info(self.logger_info)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        config = self._get_default_config()
        
        if not self.config_path:
            return config
        
        try:
            with open(str(self.config_path), 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
            
            # Update default config with loaded values
            if 'spanishoddata_preprocessing' in loaded_config:
                config.update(loaded_config['spanishoddata_preprocessing'])
            elif 'preprocessing' in loaded_config:
                config.update(loaded_config['preprocessing'])
            
            self.logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            self.logger.warning(f"Failed to load config from {self.config_path}: {e}. Using default config.")
            return config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for SpanishODData preprocessing."""
        return {
            "cargo_types": {
                "normal_percentage": 70  # 70% normal, 30% refrigerada
            },
            "truck_capacity": {
                "min_palets": 1,
                "max_palets": 10
            },
            "weight_per_palet": {
                "min_kg": 50,
                "max_kg": 800
            },
            "pricing": {
                "base_price_per_km": 1.2,  # Base price per kilometer
                "price_variation_percent": 20  # ±20% variation
            },
            "output_filename": "processed_spanishoddata.csv"
        }
    
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
    
    def _generate_price(self, trips_total_length_km: float, n_trips: float) -> int:
        """
        Generate price based on route distance with variation.
        
        Args:
            trips_total_length_km (float): Total trip distance in kilometers
            
        Returns:
            int: Price in euros (integer)
        """
        if trips_total_length_km is None or trips_total_length_km <= 0:
            trips_total_length_km = 50  # Default 50km if no distance available
        
        # Get pricing configuration
        base_price_per_km = self.config.get('pricing', {}).get('base_price_per_km', 1.2)
        variation_percent = self.config.get('pricing', {}).get('price_variation_percent', 20)
        
        # Calculate base price
        base_price = trips_total_length_km / n_trips * base_price_per_km
        
        # Add variation (±20% by default)
        variation_factor = np.random.uniform(
            1 - (variation_percent / 100), 
            1 + (variation_percent / 100)
        )
        
        final_price = base_price * variation_factor
        
        # Return as integer
        return int(round(final_price))
    
    def _add_synthetic_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add synthetic cargo fields to the DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame with SpanishODData
            
        Returns:
            pd.DataFrame: DataFrame with added synthetic fields
        """
        self.logger.info("Adding synthetic cargo fields to dataset")
        
        # Create lists to store the generated values
        tipos_mercancia = []
        volumenes = []
        pesos = []
        precios = []
        
        # Generate synthetic data for each row
        for row_num, (index, row) in enumerate(df.iterrows()):
            # Generate volume first (needed for weight calculation)
            volumen = self._generate_volume()
            
            # Generate other fields
            tipo_mercancia = self._generate_cargo_type()
            peso = self._generate_weight(volumen)
            precio = self._generate_price(row.get('trips_total_length_km', 0), row.get('n_trips', 1))
            
            # Append to lists
            tipos_mercancia.append(tipo_mercancia)
            volumenes.append(volumen)
            pesos.append(peso)
            precios.append(precio)
            
            # Log progress for large datasets
            if (row_num + 1) % 10000 == 0:
                self.logger.info(f"Processed {row_num + 1} rows")
        
        # Add the new columns to the DataFrame
        df['tipo_mercancia'] = tipos_mercancia
        df['volumen'] = volumenes
        df['peso'] = pesos
        df['precio'] = precios
        
        self.logger.info(f"Successfully added synthetic fields to {len(df)} rows")
        return df
    
    def discover_csv_files(self) -> List[Path]:
        """
        Discover all CSV files in the input directory.
        
        Returns:
            List[Path]: List of paths to discovered CSV files
        """
        csv_files = list(self.input_dir.glob("*.csv"))
        
        if not csv_files:
            self.logger.warning(f"No CSV files found in {self.input_dir}")
        else:
            self.logger.info(f"Found {len(csv_files)} CSV files: {[f.name for f in csv_files]}")
        
        return csv_files
    
    def process_single_file(self, input_file: Path) -> pd.DataFrame:
        """
        Process a single SpanishODData CSV file by adding synthetic cargo fields.
        
        Args:
            input_file (Path): Path to the CSV file to process
            
        Returns:
            pd.DataFrame: Processed DataFrame with synthetic fields
        """
        self.logger.info(f"Loading SpanishODData from {input_file}")
        
        try:
            # Load the CSV file
            df = pd.read_csv(input_file)
            
            self.logger.info(f"Loaded {len(df)} rows from {input_file.name}")
            self.logger.info(f"Original columns: {list(df.columns)}")
            
            # Validate expected columns
            expected_columns = [
                'date', 'origin_id', 'origin_name', 'origin_longitude', 'origin_latitude',
                'destination_id', 'destination_name', 'destination_longitude', 
                'destination_latitude', 'n_trips', 'trips_total_length_km'
            ]
            
            missing_columns = [col for col in expected_columns if col not in df.columns]
            if missing_columns:
                self.logger.warning(f"Missing expected columns in {input_file.name}: {missing_columns}")
            
            # Add synthetic fields
            processed_df = self._add_synthetic_fields(df)
            
            self.logger.info(f"Successfully processed {input_file.name} - Final columns: {len(processed_df.columns)}")
            return processed_df
            
        except Exception as e:
            self.logger.error(f"Error processing {input_file.name}: {e}")
            raise
    
    def process_data(self) -> List[tuple[Path, pd.DataFrame]]:
        """
        Process all SpanishODData CSV files in the input directory.
        
        Returns:
            List[tuple[Path, pd.DataFrame]]: List of tuples containing (input_file_path, processed_dataframe)
        """
        csv_files = self.discover_csv_files()
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {self.input_dir}")
        
        processed_files = []
        
        for csv_file in csv_files:
            self.logger.info(f"Processing file {csv_file.name}...")
            processed_df = self.process_single_file(csv_file)
            processed_files.append((csv_file, processed_df))
        
        self.logger.info(f"Successfully processed {len(processed_files)} files")
        return processed_files
    
    def save_processed_data(self, input_file: Path, df: pd.DataFrame) -> str:
        """
        Save the processed DataFrame to CSV with a name based on the input file.
        
        Args:
            input_file (Path): Original input file path (for naming)
            df (pd.DataFrame): Processed DataFrame
            
        Returns:
            str: Path to the saved file
        """
        # Generate output filename based on input filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_name = input_file.stem  # filename without extension
        output_filename = f"processed_{input_name}_{timestamp}.csv"
        output_path = self.output_dir / output_filename
        
        try:
            # Save to CSV
            df.to_csv(output_path, index=False)
            
            self.logger.info(f"Processed data from {input_file.name} saved to {output_path}")
            self.logger.info(f"Output contains {len(df)} rows and {len(df.columns)} columns")
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Error saving processed data from {input_file.name}: {e}")
            raise
    
    def run(self) -> List[str]:
        """
        Execute the complete preprocessing pipeline for all CSV files.
        
        Returns:
            List[str]: List of paths to the output CSV files
        """
        self.logger.info("Starting SpanishODData preprocessing pipeline")
        
        try:
            # Process all CSV files
            processed_files = self.process_data()
            
            # Save all processed files
            output_paths = []
            for input_file, processed_df in processed_files:
                output_path = self.save_processed_data(input_file, processed_df)
                output_paths.append(output_path)
            
            self.logger.info("SpanishODData preprocessing pipeline completed successfully")
            return output_paths
            
        except Exception as e:
            self.logger.error(f"SpanishODData preprocessing pipeline failed: {e}")
            raise


def main():
    """Main function to run the SpanishODData preprocessor from command line."""
    parser = argparse.ArgumentParser(description='Atlas SpanishODData Preprocessor')
    parser.add_argument('--input-dir', required=True, 
                       help='Directory containing SpanishODData CSV files')
    parser.add_argument('--output-dir', required=True, 
                       help='Directory to save processed CSV files')
    parser.add_argument('--config', 
                       help='Path to configuration file (optional)')
    parser.add_argument('--seed', type=int,
                       help='Random seed for reproducible results')
    
    args = parser.parse_args()
    
    try:
        preprocessor = SpanishODDataPreprocessor(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            config_path=args.config,
            random_seed=args.seed
        )
        
        output_files = preprocessor.run()
        print(f"SpanishODData preprocessing completed successfully.")
        print(f"Output files: {output_files}")
        
    except Exception as e:
        print(f"SpanishODData preprocessing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()