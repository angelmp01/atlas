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
    
    def create_target_encoded_features(
        self, 
        df: pd.DataFrame, 
        target_col: str,
        categorical_cols: List[str],
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Create target-encoded features for high-cardinality categoricals.
        
        Args:
            df: Input dataframe
            target_col: Target column for encoding
            categorical_cols: List of categorical columns to encode
            fit: Whether to fit encoders
            
        Returns:
            DataFrame with target-encoded features
        """
        # Modify in place to save memory
        
        if target_col not in df.columns:
            logger.warning(f"Target column {target_col} not found for target encoding")
            return df
        
        for col in categorical_cols:
            if col in df.columns:
                encoded_col = f"{col}_target_encoded"
                
                if fit:
                    df[encoded_col] = target_encode(
                        df[col],
                        df[target_col],
                        min_samples_leaf=self.config.features.min_samples_leaf,
                        smoothing=self.config.features.smoothing
                    )
                    
                    # Store encoding mapping for inference
                    encoding_map = df.groupby(col)[encoded_col].first().to_dict()
                    self.encoders[f"{col}_target_map"] = encoding_map
                    
                else:
                    # Apply stored encoding
                    if f"{col}_target_map" in self.encoders:
                        global_mean = df[target_col].mean() if target_col in df.columns else 0
                        df[encoded_col] = df[col].map(self.encoders[f"{col}_target_map"]).fillna(global_mean)
                    else:
                        df[encoded_col] = 0
        
        return df
    
    def create_historical_features(self, df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
        """
        Create historical aggregation features.
        
        Args:
            df: Input dataframe with date column
            group_cols: Columns to group by for historical aggregates
            
        Returns:
            DataFrame with historical features
        """
        if 'date' not in df.columns:
            logger.warning("Date column not found for historical features")
            return df
        
        # Modify in place to save memory
        df['date'] = pd.to_datetime(df['date'])
        
        # Rolling features for trip counts
        if 'n_trips' in df.columns:
            df = create_rolling_features(
                df, 'n_trips', group_cols, 
                windows=self.config.features.rolling_windows
            )
        
        # Rolling features for price
        if 'precio' in df.columns:
            df = create_rolling_features(
                df, 'precio', group_cols,
                windows=self.config.features.rolling_windows
            )
        
        # Rolling features for weight
        if 'peso' in df.columns:
            df = create_rolling_features(
                df, 'peso', group_cols,
                windows=self.config.features.rolling_windows
            )
        
        # Lag features
        if 'n_trips' in df.columns:
            df = create_lag_features(
                df, 'n_trips', group_cols,
                lags=[1, 7, 14]
            )
        
        return df
    
    def create_density_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create destination density features.
        
        These features count the number of destinations reachable from an origin
        that are within certain distance radii of the reference destination.
        
        Args:
            df: Input dataframe with origin_id, destination coordinates
            
        Returns:
            DataFrame with density features added
        """
        if not all(col in df.columns for col in ['origin_id', 'destination_lat', 'destination_lon']):
            logger.warning("Required columns for density features not found")
            return df
        
        # Modify in place to save memory
        
        for radius in self.config.features.density_radii:
            density_col = f"destinations_within_{int(radius)}km"
            df[density_col] = 0
            
            # This is computationally expensive - consider caching or pre-computing
            for idx, row in df.iterrows():
                try:
                    dest_ids = self.db.get_od_pairs_in_radius(
                        row['origin_id'],
                        row['destination_lat'],
                        row['destination_lon'],
                        radius
                    )
                    df.loc[idx, density_col] = len(dest_ids)
                except Exception as e:
                    logger.warning(f"Failed to compute density for row {idx}: {e}")
                    df.loc[idx, density_col] = 0
        
        return df


class ProbabilityFeatureBuilder(FeatureBuilder):
    """Feature builder for probability estimation models."""
    
    def build_features(
        self,
        df: pd.DataFrame,
        include_tau: bool = True,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Build features for probability estimation.
        
        Args:
            df: Input dataframe
            include_tau: Whether to include tau (elapsed time) features
            fit: Whether to fit encoders
            
        Returns:
            DataFrame with probability model features
        """
        logger.info("Building features for probability estimation")
        logger.info(f"Starting feature engineering on {len(df):,} rows...")
        
        # Base features
        logger.info("[1/6] Creating base features (time, distance, etc.)...")
        df = self.create_base_features(df)
        
        # Categorical encoding
        logger.info("[2/6] Encoding categorical features...")
        df = self.create_categorical_features(df, fit=fit)
        
        # Target encoding for origin-destination pairs
        if 'n_trips' in df.columns:
            logger.info("[3/6] Creating target-encoded features for OD pairs...")
            df = self.create_target_encoded_features(
                df, 'n_trips', ['origin_id', 'destination_id'], fit=fit
            )
        
        # Historical features
        logger.info("[4/6] Computing historical rolling features...")
        df = self.create_historical_features(df, ['origin_id', 'destination_id'])
        
        # Tau (elapsed time) features
        if include_tau and 'tau_minutes' in df.columns:
            logger.info("[5/6] Creating elapsed time (tau) features...")
            df['tau_bin'] = df['tau_minutes'].apply(
                lambda x: to_tau_bin(x, self.config.features.tau_bin_minutes)
            )
            
            # Tau-based features
            df['tau_hour'] = df['tau_minutes'] / 60.0
            df['tau_normalized'] = df['tau_minutes'] / (self.config.features.max_tau_hours * 60)
            
            # Interaction with day of week
            if 'day_of_week' in df.columns:
                df['tau_dow_interaction'] = df['tau_bin'] * df['day_of_week']
        
        # Density features (optional, expensive)
        # df = self.create_density_features(df)
        
        logger.info("[6/6] Feature engineering completed!")
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
        
        # Target encoding
        if target_col in df.columns:
            df = self.create_target_encoded_features(
                df, target_col, ['origin_id', 'destination_id'], fit=fit
            )
        
        # Historical features
        df = self.create_historical_features(df, ['origin_id', 'destination_id'])
        
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
    
    This is a convenience function that combines data loading and feature engineering
    for probability models. It handles both Regime A and Regime B training data.
    
    Args:
        config: Configuration object
        
    Returns:
        Feature dataframe ready for probability model training
    """
    from .io import create_database_manager, create_dataset_builder
    
    logger.info("Building probability dataset")
    
    # Initialize data components
    db_manager = create_database_manager(config)
    dataset_builder = create_dataset_builder(config)
    feature_builder = ProbabilityFeatureBuilder(config, db_manager)
    
    # Check training regime
    if config.training.training_regime == "regime_a":
        # Try to load time-binned data
        df = db_manager.read_loads_with_time_bins(
            config.training.start_date,
            config.training.end_date
        )
        
        if df.empty:
            logger.warning("No time-binned data found, falling back to Regime B")
            config.training.training_regime = "regime_b"
    
    if config.training.training_regime == "regime_b" or df.empty:
        # Load daily aggregated data
        base_df = dataset_builder.build_base_dataset(
            config.training.start_date,
            config.training.end_date
        )
        
        if base_df.empty:
            raise ValueError("No training data found for specified date range")
        
        # Aggregate to daily level
        df = dataset_builder.build_od_aggregates(base_df)
    
    # Build features
    df = feature_builder.build_features(df, include_tau=(config.training.training_regime == "regime_a"))
    
    # Check for duplicate columns (critical for XGBoost)
    if df.columns.duplicated().any():
        dup_cols = df.columns[df.columns.duplicated()].tolist()
        logger.warning(f"Duplicate columns detected: {dup_cols}")
        logger.warning("Removing duplicate columns...")
        df = df.loc[:, ~df.columns.duplicated()]
    
    # Optimize memory usage by downcasting numeric types
    logger.info("Optimizing memory usage...")
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    logger.info(f"Built probability dataset with {len(df)} samples and {len(df.columns)} features")
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
    
    # Load base dataset
    base_df = dataset_builder.build_base_dataset(
        config.training.start_date,
        config.training.end_date
    )
    
    if base_df.empty:
        raise ValueError("No training data found for specified date range")
    
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
    
    # Load base dataset
    base_df = dataset_builder.build_base_dataset(
        config.training.start_date,
        config.training.end_date
    )
    
    if base_df.empty:
        raise ValueError("No training data found for specified date range")
    
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