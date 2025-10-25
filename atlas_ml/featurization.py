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
        
        # Truck type standardization
        if 'tipo_mercancia' in df.columns and 'truck_type' not in df.columns:
            df['truck_type'] = df['tipo_mercancia'].map({
                'refrigerada': 'refrigerado',
                'normal': 'normal'
            }).fillna('normal')
        
        # Log-transform skewed numerical features
        numerical_features = ['precio', 'peso', 'volumen', 'od_length_km']
        for feature in numerical_features:
            if feature in df.columns:
                log_feature = f"log_{feature}"
                df[log_feature] = np.log1p(df[feature].clip(lower=0))
        
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
        categorical_cols = ['truck_type', 'tipo_mercancia']
        
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
    """Feature builder for price and weight regression models."""
    
    def build_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Build features for regression models (price/weight).
        
        Simplified: only base features + categorical encoding.
        Historical features removed for now.
        
        Args:
            df: Input dataframe
            target_col: Target column name ('precio' or 'peso')
            fit: Whether to fit encoders
            
        Returns:
            DataFrame with regression model features
        """
        logger.info(f"Building features for {target_col} regression")
        
        # Base features
        df = self.create_base_features(df)
        
        # Categorical encoding
        df = self.create_categorical_features(df, fit=fit)
        
        # Distance-based features
        if 'od_length_km' in df.columns:
            # Distance bins
            df['distance_bin'] = pd.cut(
                df['od_length_km'],
                bins=[0, 50, 100, 200, 500, np.inf],
                labels=['very_short', 'short', 'medium', 'long', 'very_long']
            ).astype(str)
            
            # Distance per weight/volume ratios
            if 'peso' in df.columns and target_col != 'peso':
                df['km_per_kg'] = df['od_length_km'] / (df['peso'] + 1e-6)
            
            if 'volumen' in df.columns:
                df['km_per_m3'] = df['od_length_km'] / (df['volumen'] + 1e-6)
        
        return df


def build_probability_dataset(config: Config) -> pd.DataFrame:
    """
    Build complete dataset for probability training.
    
    Loads daily aggregated trip data and builds simple features.
    Uses uniform distribution approach (daily counts distributed evenly over 24 hours).
    
    Args:
        config: Configuration object
        
    Returns:
        Feature dataframe ready for probability model training
    """
    from .io import create_database_manager, create_dataset_builder
    
    logger.info("=" * 70)
    logger.info("BUILDING PROBABILITY TRAINING DATASET")
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
        end_date='2024-12-31'
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
    logger.info(f"[OK] DATASET READY: {len(df):,} samples × {len(df.columns)} features")
    logger.info("=" * 70)
    return df


def build_price_dataset(config: Config) -> pd.DataFrame:
    """
    Build complete dataset for price regression training.
    
    Args:
        config: Configuration object
        
    Returns:
        Feature dataframe ready for price model training
    """
    from .io import create_database_manager, create_dataset_builder
    
    logger.info("Building price dataset")
    
    # Initialize data components
    db_manager = create_database_manager(config)
    dataset_builder = create_dataset_builder(config)
    feature_builder = RegressionFeatureBuilder(config, db_manager)
    
    # Load base dataset (all 2024 data)
    base_df = dataset_builder.build_base_dataset(
        start_date='2024-01-01',
        end_date='2024-12-31'
    )
    
    if base_df.empty:
        raise ValueError("No training data found for 2024")
    
    # Filter out records with missing prices
    df = base_df[base_df['precio'].notna() & (base_df['precio'] > 0)].copy()
    
    # Build features
    df = feature_builder.build_features(df, target_col='precio')
    
    logger.info(f"Built price dataset with {len(df)} samples and {len(df.columns)} features")
    return df


def build_weight_dataset(config: Config) -> pd.DataFrame:
    """
    Build complete dataset for weight regression training.
    
    Args:
        config: Configuration object
        
    Returns:
        Feature dataframe ready for weight model training
    """
    from .io import create_database_manager, create_dataset_builder
    
    logger.info("Building weight dataset")
    
    # Initialize data components
    db_manager = create_database_manager(config)
    dataset_builder = create_dataset_builder(config)
    feature_builder = RegressionFeatureBuilder(config, db_manager)
    
    # Load base dataset (all 2024 data)
    base_df = dataset_builder.build_base_dataset(
        start_date='2024-01-01',
        end_date='2024-12-31'
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
        'day_of_week', 'week_of_year', 'holiday_flag', 'month', 'quarter',
        'od_length_km', 'log_od_length_km',
        'truck_type_encoded', 'tipo_mercancia_encoded',
        'origin_id_target_encoded', 'destination_id_target_encoded'
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
    
    if task_type in ['price', 'weight']:
        regression_features = [
            'distance_bin', 'km_per_kg', 'km_per_m3'
        ]
        features.extend(regression_features)
    
    return features