"""
I/O module for ATLAS ML package.

Provides database access using SQLAlchemy + GeoAlchemy2, data readers/writers,
and utilities for exporting processed data to Parquet format.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from geoalchemy2 import Geometry
from pyproj import Transformer
from sqlalchemy import (
    Column, Date, Float, Integer, String, Text, create_engine, text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from .config import Config

logger = logging.getLogger(__name__)

Base = declarative_base()


class SoddLocation(Base):
    """ORM model for app.sodd_locations table."""
    
    __tablename__ = 'sodd_locations'
    __table_args__ = {'schema': 'app'}
    
    location_id = Column(Integer, primary_key=True)
    location_name = Column(String)
    geom = Column(Geometry('POINT'))


class SoddLoad(Base):
    """ORM model for app.sodd_loads_filtered table."""
    
    __tablename__ = 'sodd_loads_filtered'
    __table_args__ = {'schema': 'app'}
    
    id = Column(Integer, primary_key=True)
    date = Column(Date)
    origin_id = Column(Integer)
    destination_id = Column(Integer)
    n_trips = Column(Integer)  # Total trip count
    n_trips_logistic = Column(Integer)  # n_trips * 0.04 (freight vehicle estimate)
    trips_total_length_km = Column(Float)  # Sum of all trip distances (must divide by n_trips)
    tipo_mercancia = Column(String)  # 'normal' or 'refrigerada'
    volumen = Column(Float)
    peso = Column(Float)
    precio = Column(Float)
    geom = Column(Geometry('LINESTRING'))


class DatabaseManager:
    """Manages database connections and queries for ATLAS ML."""
    
    def __init__(self, config: Config):
        """
        Initialize database manager.
        
        Args:
            config: Configuration object with database settings
        """
        self.config = config
        self.engine = create_engine(
            config.database.dsn,
            echo=config.database.echo
        )
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Create coordinate transformer for lat/lon conversion
        # Assuming data is in Web Mercator (EPSG:3857) or similar
        self.transformer_to_wgs84 = Transformer.from_crs(
            "EPSG:3857", "EPSG:4326", always_xy=True
        )
        self.transformer_from_wgs84 = Transformer.from_crs(
            "EPSG:4326", "EPSG:3857", always_xy=True
        )
    
    def get_session(self):
        """Get database session."""
        return self.SessionLocal()
    
    def read_locations(self) -> pd.DataFrame:
        """
        Read all locations from the database.
        
        Returns:
            DataFrame with location_id, location_name, lat, lon columns
        """
        query = text("""
            SELECT 
                location_id,
                location_name,
                ST_X(ST_Transform(geom, 4326)) as lon,
                ST_Y(ST_Transform(geom, 4326)) as lat
            FROM app.sodd_locations
            ORDER BY location_id
        """)
        
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn)
        
        logger.info(f"Loaded {len(df)} locations from database")
        return df
    
    def read_loads(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Read load data from the database.
        
        Args:
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with load data including origin/destination coordinates
        """
        # Base query with geometry transformations
        query_parts = ["""
            SELECT 
                l.date,
                l.origin_id,
                l.destination_id,
                l.n_trips,
                l.n_trips_logistic,
                l.trips_total_length_km,
                l.tipo_mercancia,
                l.volumen,
                l.peso,
                l.precio,
                o.location_name as origin_name,
                d.location_name as destination_name,
                ST_X(ST_Transform(o.geom, 4326)) as origin_lon,
                ST_Y(ST_Transform(o.geom, 4326)) as origin_lat,
                ST_X(ST_Transform(d.geom, 4326)) as destination_lon,
                ST_Y(ST_Transform(d.geom, 4326)) as destination_lat
            FROM app.sodd_loads_filtered l
            JOIN app.sodd_locations o ON l.origin_id = o.location_id
            JOIN app.sodd_locations d ON l.destination_id = d.location_id
        """]
        
        # Add date filters
        where_conditions = []
        if start_date:
            where_conditions.append(f"l.date >= '{start_date}'")
        if end_date:
            where_conditions.append(f"l.date <= '{end_date}'")
        
        if where_conditions:
            query_parts.append("WHERE " + " AND ".join(where_conditions))
        
        query_parts.append("ORDER BY l.date, l.origin_id, l.destination_id")
        
        if limit:
            query_parts.append(f"LIMIT {limit}")
        
        query = text(" ".join(query_parts))
        
        logger.info(f"Querying database for loads between {start_date or 'start'} and {end_date or 'end'}...")
        logger.info("This may take 1-2 minutes for large datasets (millions of records)...")
        
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn)
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        
        logger.info(f"✓ Loaded {len(df):,} load records from database")
        return df
    
    def read_loads_with_time_bins(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Read load data with time bin information.
        
        DEPRECATED: Time-binned training is no longer supported.
        This method is kept for backward compatibility but returns empty DataFrame.
        
        Args:
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            
        Returns:
            Empty DataFrame (time-binned training deprecated)
        """
        # Check if time-binned table exists
        check_query = text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'app' 
                AND table_name = 'sodd_loads_time_bins'
            )
        """)
        
        with self.engine.connect() as conn:
            table_exists = conn.execute(check_query).scalar()
        
        if not table_exists:
            logger.warning("Time-binned load data not available (deprecated feature).")
            return pd.DataFrame()
        
        # Query time-binned data
        query_parts = ["""
            SELECT 
                date,
                origin_id,
                destination_id,
                tau_bin,
                n_trips as n_trips_bin
            FROM app.sodd_loads_time_bins
        """]
        
        where_conditions = []
        if start_date:
            where_conditions.append(f"date >= '{start_date}'")
        if end_date:
            where_conditions.append(f"date <= '{end_date}'")
        
        if where_conditions:
            query_parts.append("WHERE " + " AND ".join(where_conditions))
        
        query_parts.append("ORDER BY date, origin_id, destination_id, tau_bin")
        
        query = text(" ".join(query_parts))
        
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn)
        
        df['date'] = pd.to_datetime(df['date'])
        
        logger.info(f"Loaded {len(df)} time-binned load records from database")
        return df
    
    def get_od_pairs_in_radius(
        self,
        origin_id: int,
        destination_lat: float,
        destination_lon: float,
        radius_km: float
    ) -> List[int]:
        """
        Get destination IDs within radius of a reference destination.
        
        Used for density features - counts destinations reachable from origin
        that are within a certain distance of the reference destination.
        
        Args:
            origin_id: Origin location ID
            destination_lat: Reference destination latitude
            destination_lon: Reference destination longitude
            radius_km: Search radius in kilometers
            
        Returns:
            List of destination location IDs within radius
        """
        query = text("""
            SELECT DISTINCT l.destination_id
            FROM app.sodd_loads_filtered l
            JOIN app.sodd_locations d ON l.destination_id = d.location_id
            WHERE l.origin_id = :origin_id
            AND ST_DWithin(
                d.geom,
                ST_Transform(ST_SetSRID(ST_MakePoint(:lon, :lat), 4326), 3857),
                :radius_m
            )
        """)
        
        with self.engine.connect() as conn:
            result = conn.execute(
                query,
                {
                    "origin_id": origin_id,
                    "lat": destination_lat,
                    "lon": destination_lon,
                    "radius_m": radius_km * 1000  # Convert to meters
                }
            )
            destination_ids = [row[0] for row in result]
        
        return destination_ids
    
    def export_to_parquet(
        self,
        df: pd.DataFrame,
        filename: str,
        directory: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Export DataFrame to Parquet format.
        
        Args:
            df: DataFrame to export
            filename: Output filename (without extension)
            directory: Output directory (defaults to config.paths.processed_dir)
            
        Returns:
            Path to exported file
        """
        if directory is None:
            directory = self.config.paths.processed_dir
        
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        output_path = directory / f"{filename}.parquet"
        df.to_parquet(output_path, index=False)
        
        logger.info(f"Exported {len(df)} records to {output_path}")
        return output_path
    
    def load_from_parquet(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Load DataFrame from Parquet file.
        
        Args:
            filepath: Path to Parquet file
            
        Returns:
            Loaded DataFrame
        """
        df = pd.read_parquet(filepath)
        logger.info(f"Loaded {len(df)} records from {filepath}")
        return df


class DatasetBuilder:
    """Builds training datasets from raw load data."""
    
    def __init__(self, db_manager: DatabaseManager, config: Config):
        """
        Initialize dataset builder.
        
        Args:
            db_manager: Database manager instance
            config: Configuration object
        """
        self.db = db_manager
        self.config = config
    
    def build_base_dataset(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        export_parquet: bool = True
    ) -> pd.DataFrame:
        """
        Build base dataset with all loads and location information.
        
        Args:
            start_date: Start date filter
            end_date: End date filter
            export_parquet: Whether to export to Parquet
            
        Returns:
            DataFrame with enhanced load data
        """
        logger.info(f"Building base dataset from {start_date or 'start'} to {end_date or 'end'}...")
        
        # Load raw data
        df = self.db.read_loads(start_date, end_date)
        
        if df.empty:
            logger.warning("No load data found for specified date range")
            return df
        
        logger.info(f"Processing {len(df):,} raw records - adding features...")
        
        # Add calculated distance per trip (trips_total_length_km is cumulative for all trips)
        # We need to divide by n_trips to get average distance per trip
        if 'trips_total_length_km' not in df.columns or df['trips_total_length_km'].isna().any():
            from .utils import haversine_distance
            df['calculated_distance_km'] = df.apply(
                lambda row: haversine_distance(
                    row['origin_lat'], row['origin_lon'],
                    row['destination_lat'], row['destination_lon']
                ),
                axis=1
            )
            # Use calculated distance where original is missing
            df['od_length_km'] = df['trips_total_length_km'].fillna(df['calculated_distance_km'])
        else:
            df['od_length_km'] = df['trips_total_length_km']
        
        # CRITICAL FIX: Divide by n_trips to get distance per trip (not total)
        # trips_total_length_km contains the sum of all trip distances, not individual trip distance
        # Note: n_trips_logistic = n_trips * 0.04 (freight vehicle estimate), so we use n_trips here
        if 'n_trips' in df.columns:
            # Avoid division by zero - if n_trips is 0 or NaN, keep original distance
            df['od_length_km'] = df.apply(
                lambda row: row['od_length_km'] / row['n_trips'] 
                if pd.notna(row['n_trips']) and row['n_trips'] > 0 
                else row['od_length_km'],
                axis=1
            )
            logger.info("✓ Normalized od_length_km by dividing trips_total_length_km by n_trips")
        
        logger.info("Adding time features (day_of_week, week_of_year, etc.)...")
        
        # Add time features
        from .utils import create_time_features
        time_features = df['date'].apply(create_time_features)
        time_df = pd.DataFrame(list(time_features))
        df = pd.concat([df, time_df], axis=1)
        
        # Add truck type mapping (basic heuristic)
        df['truck_type'] = df['tipo_mercancia'].map({
            'refrigerada': 'refrigerado',
            'normal': 'normal'
        }).fillna('normal')
        
        logger.info(f"✓ Base dataset complete: {len(df):,} records with {len(df.columns)} features")
        
        if export_parquet:
            self.db.export_to_parquet(df, "base_loads_dataset")
        
        return df
    
    def build_od_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build origin-destination aggregated features.
        
        Args:
            df: Base dataset
            
        Returns:
            DataFrame with OD-level aggregates
        """
        logger.info(f"Step 1/3: Aggregating {len(df):,} records to daily OD level...")
        logger.info("Grouping by (date, origin, destination) - this takes 2-5 minutes for 20M+ records...")
        
        # Daily aggregates by (origin_id, destination_id)
        # Use 'first' instead of mode() for categorical - much faster and equivalent for most cases
        od_daily = df.groupby(['date', 'origin_id', 'destination_id']).agg({
            'n_trips': 'sum',  # Total trips
            'n_trips_logistic': 'sum',  # Freight vehicle estimate (n_trips * 0.04)
            'precio': ['mean', 'median', 'std'],
            'peso': ['mean', 'median', 'std'],
            'volumen': ['mean', 'median', 'std'],
            'od_length_km': 'mean',
            'truck_type': 'first',  # Take first value (faster than mode)
            'tipo_mercancia': 'first'
        }).reset_index()
        
        logger.info(f"Step 2/3: Flattening {len(od_daily):,} aggregated rows...")
        
        # Flatten column names
        od_daily.columns = ['_'.join(col).strip('_') for col in od_daily.columns]
        
        # Rename columns for clarity
        rename_dict = {
            'date_': 'date',
            'origin_id_': 'origin_id',
            'destination_id_': 'destination_id',
            'n_trips_sum': 'n_trips_total',  # Total trips per day
            'n_trips_logistic_sum': 'n_trips_daily',  # Freight vehicles per day (target variable)
            'precio_mean': 'precio_mean_daily',
            'precio_median': 'precio_median_daily',
            'precio_std': 'precio_std_daily',
            'peso_mean': 'peso_mean_daily',
            'peso_median': 'peso_median_daily',
            'peso_std': 'peso_std_daily',
            'volumen_mean': 'volumen_mean_daily',
            'volumen_median': 'volumen_median_daily',
            'volumen_std': 'volumen_std_daily',
            'od_length_km_mean': 'od_length_km',
            'truck_type_first': 'truck_type',
            'tipo_mercancia_first': 'tipo_mercancia'
        }
        
        od_daily = od_daily.rename(columns=rename_dict)
        
        logger.info(f"Step 3/3: ✓ Aggregation completed - {len(od_daily):,} daily OD pairs created")
        
        return od_daily


def create_database_manager(config: Config) -> DatabaseManager:
    """
    Factory function to create DatabaseManager instance.
    
    Args:
        config: Configuration object
        
    Returns:
        Configured DatabaseManager
    """
    return DatabaseManager(config)


def create_dataset_builder(config: Config) -> DatasetBuilder:
    """
    Factory function to create DatasetBuilder instance.
    
    Args:
        config: Configuration object
        
    Returns:
        Configured DatasetBuilder
    """
    db_manager = create_database_manager(config)
    return DatasetBuilder(db_manager, config)