"""
Feature engineering module for ATLAS ML package.

Provides feature builders for probability, price, and weight models with 
historical aggregates, target encoding patterns, and time-based features.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import Config
from .io import DatabaseManager, DatasetBuilder
from .utils import (
    create_lag_features, create_rolling_features, create_time_features,
    haversine_distance, target_encode, to_tau_bin
)

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """Base class for feature builders."""
    
    def __init__(self, config: Config, db_manager: DatabaseManager):
        """
        Initialize feature builder.
        
        Args:
            config: Configuration object
            db_manager: Database manager instance
        """
        self.config = config
        self.db = db_manager
        self.encoders = {}
        self.scalers = {}
    
    def create_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create base features common to all models.
        
        Args:
            df: Input dataframe with load data
            
        Returns:
            DataFrame with base features added
        """
        # Modify in place to save memory
        
        # Time features
        if 'date' in df.columns:
            time_features = df['date'].apply(create_time_features)
            time_df = pd.DataFrame(list(time_features))
            # Only add columns that don't already exist to avoid duplicates
            new_cols = [col for col in time_df.columns if col not in df.columns]
            if new_cols:
                df = pd.concat([df, time_df[new_cols]], axis=1)
        
        # Distance features
        if all(col in df.columns for col in ['origin_lat', 'origin_lon', 'destination_lat', 'destination_lon']):
            if 'od_length_km' not in df.columns:
                df['od_length_km'] = df.apply(
                    lambda row: haversine_distance(
                        row['origin_lat'], row['origin_lon'],
                        row['destination_lat'], row['destination_lon']
                    ), axis=1
                )
            
            # Add absolute geographic coordinates (Catalunya-specific patterns)
            # These capture spatial effects: coastal vs inland, urban vs rural, port zones, etc.
            # NOTE: Coordinates already exist in dataframe, we just add derived features
            
            # Geographic zone indicators (Catalunya-specific)
            # Barcelona metropolitan area (high traffic zone)
            barcelona_lat, barcelona_lon = 41.3851, 2.1734
            df['origin_dist_to_bcn'] = df.apply(
                lambda row: haversine_distance(row['origin_lat'], row['origin_lon'], barcelona_lat, barcelona_lon),
                axis=1
            )
            df['dest_dist_to_bcn'] = df.apply(
                lambda row: haversine_distance(row['destination_lat'], row['destination_lon'], barcelona_lat, barcelona_lon),
                axis=1
            )
            
            # Coastal vs inland indicator (Mediterranean coast effect)
            # Catalunya coast is roughly at lon > 0.5 and lat between 40.5-42.5
            df['origin_is_coastal'] = ((df['origin_lon'] > 0.5) & 
                                       (df['origin_lat'] > 40.5) & 
                                       (df['origin_lat'] < 42.5)).astype(int)
            df['dest_is_coastal'] = ((df['destination_lon'] > 0.5) & 
                                     (df['destination_lat'] > 40.5) & 
                                     (df['destination_lat'] < 42.5)).astype(int)
        
        # NOTE: We DON'T create log-transformed features anymore
        # Reasons:
        # 1. They cause confusion (log_precio would need precio as input, which we don't have at inference)
        # 2. XGBoost can learn non-linear transformations automatically via tree splits
        # 3. Simplifies feature engineering and reduces feature count
        
        return df
    
    def create_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Create categorical features with encoding.
        
        Args:
            df: Input dataframe
            fit: Whether to fit encoders (True for training, False for inference)
            
        Returns:
            DataFrame with encoded categorical features
        """
        # Avoid expensive copy() for large datasets - modify in place
        # Only encode tipo_mercancia (normal/refrigerada)
        categorical_cols = ['tipo_mercancia']
        
        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    if col not in self.encoders:
                        self.encoders[col] = LabelEncoder()
                        df[f"{col}_encoded"] = self.encoders[col].fit_transform(df[col].astype(str))
                    else:
                        df[f"{col}_encoded"] = self.encoders[col].transform(df[col].astype(str))
                else:
                    if col in self.encoders:
                        # Handle unseen categories
                        df[f"{col}_encoded"] = df[col].astype(str).map(
                            dict(zip(self.encoders[col].classes_, self.encoders[col].transform(self.encoders[col].classes_)))
                        ).fillna(-1)  # Unknown category
                    else:
                        df[f"{col}_encoded"] = 0  # Default value
        
        return df


class ProbabilityFeatureBuilder(FeatureBuilder):
    """Feature builder for probability estimation models."""
    
    def build_features(
        self,
        df: pd.DataFrame,
        include_tau: bool = False,  # Not used anymore, kept for compatibility
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Build features for probability estimation (simplified).
        
        Only creates simple features: temporal, geographic, categorical.
        No historical aggregations, no tau features, no DB queries.
        
        Args:
            df: Input dataframe
            include_tau: DEPRECATED (not used, kept for compatibility)
            fit: Whether to fit encoders
            
        Returns:
            DataFrame with probability model features
        """
        logger.info("Building features for probability estimation (simplified)")
        logger.info(f"Starting feature engineering on {len(df):,} rows...")
        
        # Base features (temporal, geographic)
        logger.info("[1/2] Creating base features (time, distance, etc.)...")
        df = self.create_base_features(df)
        
        # Categorical encoding
        logger.info("[2/2] Encoding categorical features...")
        df = self.create_categorical_features(df, fit=fit)
        
        logger.info("Feature engineering completed!")
        logger.info(f"Final dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
        
        return df


class RegressionFeatureBuilder(FeatureBuilder):
    """Feature builder for price and weight regression models.
    
    Uses the same base features as ProbabilityFeatureBuilder for consistency.
    No distance bins needed - XGBoost learns optimal splits automatically.
    """
    
    def build_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Build features for regression models (price/weight).
        
        Uses same feature engineering as probability model for consistency.
        
        Args:
            df: Input dataframe
            target_col: Target column name ('precio' or 'peso')
            fit: Whether to fit encoders
            
        Returns:
            DataFrame with regression model features
        """
        logger.info(f"Building features for {target_col} regression")
        
        # Use same base features as probability model
        df = self.create_base_features(df)
        
        # Use same categorical encoding as probability model
        df = self.create_categorical_features(df, fit=fit)
        
        return df


def build_probability_dataset(config: Config, quick_test: bool = False) -> pd.DataFrame:
    """
    Build complete dataset for probability training.
    
    Loads daily aggregated trip data and builds simple features.
    Uses uniform distribution approach (daily counts distributed evenly over 24 hours).
    
    Args:
        config: Configuration object
        quick_test: If True, filter to only a few OD pairs for fast testing
        
    Returns:
        Feature dataframe ready for probability model training
    """
    from .io import create_database_manager, create_dataset_builder
    
    logger.info("=" * 70)
    logger.info("BUILDING PROBABILITY TRAINING DATASET")
    if quick_test:
        logger.info("*** QUICK TEST MODE: Using only 5 OD pairs ***")
    logger.info("=" * 70)
    logger.info("This process has 4 main steps:")
    logger.info("  1. Load raw data from database (~1-2 min)")
    logger.info("  2. Add calculated features (~30 sec)")
    logger.info("  3. Aggregate to daily OD pairs (~2-5 min)")
    logger.info("  4. Encode categorical features (~10 sec)")
    logger.info("=" * 70)
    
    # Initialize data components
    db_manager = create_database_manager(config)
    dataset_builder = create_dataset_builder(config)
    feature_builder = ProbabilityFeatureBuilder(config, db_manager)
    
    # Load base dataset (all 2024 data)
    logger.info("\n[STEP 1/4] Loading raw data from database...")
    base_df = dataset_builder.build_base_dataset(
        start_date='2024-01-01',
        end_date='2024-12-31',
        od_pairs_filter=config.quick_test_od_pairs if quick_test else None
    )
    
    if base_df.empty:
        raise ValueError("No training data found for 2024")
    
    # Aggregate to daily level
    logger.info("\n[STEP 2/4] Aggregating to daily OD level...")
    df = dataset_builder.build_od_aggregates(base_df)
    
    # Build features (simplified: no tau, no historical)
    logger.info("\n[STEP 3/4] Building features (encodings, etc.)...")
    df = feature_builder.build_features(df, include_tau=False)
    
    logger.info("\n[STEP 4/4] Validating dataset...")
    
    # Check for duplicate columns (critical for XGBoost)
    if df.columns.duplicated().any():
        dup_cols = df.columns[df.columns.duplicated()].tolist()
        logger.warning(f"Duplicate columns detected: {dup_cols}")
        logger.warning("Removing duplicate columns...")
        df = df.loc[:, ~df.columns.duplicated()]
    
    # Optimize memory usage by downcasting numeric types
    logger.info("  - Optimizing memory usage (downcasting dtypes)...")
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    logger.info("=" * 70)
    logger.info(f"[OK] DATASET READY: {len(df):,} samples x {len(df.columns)} features")
    logger.info("=" * 70)
    return df


def build_price_dataset(config: Config, quick_test: bool = False) -> pd.DataFrame:
    """
    Build complete dataset for price regression training.
    
    Args:
        config: Configuration object
        quick_test: If True, filter to only a few OD pairs for fast testing
        
    Returns:
        Feature dataframe ready for price model training
    """
    from .io import create_database_manager, create_dataset_builder
    
    logger.info("Building price dataset")
    if quick_test:
        logger.info("*** QUICK TEST MODE: Using only 5 OD pairs ***")
    
    # Initialize data components
    db_manager = create_database_manager(config)
    dataset_builder = create_dataset_builder(config)
    feature_builder = RegressionFeatureBuilder(config, db_manager)
    
    # Load base dataset (all 2024 data)
    base_df = dataset_builder.build_base_dataset(
        start_date='2024-01-01',
        end_date='2024-12-31',
        od_pairs_filter=config.quick_test_od_pairs if quick_test else None
    )
    
    if base_df.empty:
        raise ValueError("No training data found for 2024")
    
    # Filter out records with missing prices
    df = base_df[base_df['precio'].notna() & (base_df['precio'] > 0)].copy()
    
    # Build features
    df = feature_builder.build_features(df, target_col='precio')
    
    logger.info(f"Built price dataset with {len(df)} samples and {len(df.columns)} features")
    return df


def build_weight_dataset(config: Config, quick_test: bool = False) -> pd.DataFrame:
    """
    Build complete dataset for weight regression training.
    
    Args:
        config: Configuration object
        quick_test: If True, filter to only a few OD pairs for fast testing
        
    Returns:
        Feature dataframe ready for weight model training
    """
    from .io import create_database_manager, create_dataset_builder
    
    logger.info("Building weight dataset")
    if quick_test:
        logger.info("*** QUICK TEST MODE: Using only 5 OD pairs ***")
    
    # Initialize data components
    db_manager = create_database_manager(config)
    dataset_builder = create_dataset_builder(config)
    feature_builder = RegressionFeatureBuilder(config, db_manager)
    
    # Load base dataset (all 2024 data)
    base_df = dataset_builder.build_base_dataset(
        start_date='2024-01-01',
        end_date='2024-12-31',
        od_pairs_filter=config.quick_test_od_pairs if quick_test else None
    )
    
    if base_df.empty:
        raise ValueError("No training data found for 2024")
    
    # Filter out records with missing weights
    df = base_df[base_df['peso'].notna() & (base_df['peso'] > 0)].copy()
    
    # Build features
    df = feature_builder.build_features(df, target_col='peso')
    
    logger.info(f"Built weight dataset with {len(df)} samples and {len(df.columns)} features")
    return df


def get_feature_names(task_type: str, include_tau: bool = False) -> List[str]:
    """
    Get expected feature names for a given task type.
    
    Args:
        task_type: Type of task ('probability', 'price', 'weight')
        include_tau: Whether to include tau features (for probability models)
        
    Returns:
        List of expected feature names
    """
    base_features = [
        # Temporal features (from date input)
        'day_of_week', 'week_of_year', 'holiday_flag', 'month', 'quarter',
        # Geographic features (from origin/destination coordinates)
        'od_length_km',  # Haversine distance
        'origin_lat', 'origin_lon', 'destination_lat', 'destination_lon',  # Absolute coordinates
        'origin_dist_to_bcn', 'dest_dist_to_bcn',  # Distance to Barcelona
        'origin_is_coastal', 'dest_is_coastal',  # Coastal vs inland indicator
        # Categorical features (from user input)
        'tipo_mercancia_encoded',  # normal/refrigerada (truck_type is redundant, removed)
        # Note: target encoding removed (not computed anymore)
    ]
    
    # Historical features
    rolling_features = []
    for window in [7, 14, 28]:
        rolling_features.extend([
            f'n_trips_rolling_mean_{window}d',
            f'n_trips_rolling_std_{window}d',
            f'precio_rolling_mean_{window}d',
            f'precio_rolling_std_{window}d',
            f'peso_rolling_mean_{window}d',
            f'peso_rolling_std_{window}d'
        ])
    
    lag_features = [
        'n_trips_lag_1d', 'n_trips_lag_7d', 'n_trips_lag_14d'
    ]
    
    features = base_features + rolling_features + lag_features
    
    # Task-specific features
    if task_type == 'probability' and include_tau:
        tau_features = [
            'tau_bin', 'tau_hour', 'tau_normalized', 'tau_dow_interaction'
        ]
        features.extend(tau_features)
    
    # No task-specific features for price/weight - use same base features as probability
    
    return features