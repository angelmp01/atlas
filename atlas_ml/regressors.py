"""
Regression models for ATLAS ML package.

Implements models for E[price_{i→d}] and E[weight_{i→d}]: expected price and weight
for loads from origin i to destination d.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Suppress XGBoost deprecation warnings (we're using the correct new syntax)
warnings.filterwarnings('ignore', message='.*tree method.*deprecated.*')
warnings.filterwarnings('ignore', message='.*Falling back to prediction using DMatrix.*')

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from .config import Config
from .evaluation import CrossValidator, create_model_card
from .featurization import RegressionFeatureBuilder, build_price_dataset, build_weight_dataset
from .io import create_database_manager
from .probability import CandidateInput
from .serialization import ModelBundle, save_model_bundle
from .utils import set_random_seed

logger = logging.getLogger(__name__)


class RegressionEstimator:
    """
    Base class for price and weight regression models.
    
    Provides shared functionality for feature engineering, training,
    and inference for both price and weight estimation tasks.
    """
    
    def __init__(self, config: Config, target_type: str):
        """
        Initialize regression estimator.
        
        Args:
            config: Configuration object
            target_type: Either 'price' or 'weight'
        """
        self.config = config
        self.target_type = target_type
        self.model = None
        self.feature_builder = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
        # Set target column name
        self.target_column = 'precio' if target_type == 'price' else 'peso'
        
        # Set model type
        if target_type == 'price':
            self.model_type = config.model.price_model_type
        else:
            self.model_type = config.model.weight_model_type
        
        set_random_seed(config.random_state)
    
    def _create_model(self) -> Any:
        """Create model instance based on configuration."""
        if self.model_type == 'xgboost' and HAS_XGBOOST:
            params = self.config.model.xgb_params.copy()
            params['objective'] = 'reg:squarederror'  # Use regression objective
            return xgb.XGBRegressor(**params)
        else:
            return RandomForestRegressor(**self.config.model.rf_params)
    
    def fit(self, df: pd.DataFrame, wandb_logger=None) -> Dict[str, Any]:
        """
        Train the regression model.
        
        Args:
            df: Training dataset with features and target
            wandb_logger: Optional WandbLogger for logging metrics
            
        Returns:
            Training results dictionary
        """
        import time
        start_time = time.time()
        
        logger.info(f"Training {self.target_type} regression model")
        
        # Validate target column
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset")
        
        # Filter valid targets (positive values)
        df_clean = df[df[self.target_column] > 0].copy()
        
        if len(df_clean) == 0:
            raise ValueError(f"No valid {self.target_type} data found")
        
        logger.info(f"Training with {len(df_clean)} samples (filtered from {len(df)})")
        
        # Build features
        self.feature_builder = RegressionFeatureBuilder(
            self.config, 
            create_database_manager(self.config)
        )
        df_features = self.feature_builder.build_features(
            df_clean, 
            target_col=self.target_column, 
            fit=True
        )
        
        # Prepare training data
        # Exclude: target, IDs, raw categoricals, and historical/leakage features
        exclude_cols = [
            'date', self.target_column, 'origin_id', 'destination_id',
            'tipo_mercancia',  # Use encoded version instead
            # Data leakage - features we won't have at inference time
            'n_trips_total', 'n_trips', 'n_trips_logistic',  # Don't use probability predictions
            'peso', 'precio', 'volumen',  # Can't use target variables as features
            'trips_total_length_km',  # Redundant with od_length_km
            # Historical aggregates - same-variable leakage
            'precio_mean_daily', 'precio_median_daily', 'precio_std_daily',
            'peso_mean_daily', 'peso_median_daily', 'peso_std_daily',
            'volumen_mean_daily', 'volumen_median_daily', 'volumen_std_daily',
            # Log-transformed versions
            'log_precio', 'log_peso', 'log_volumen'
        ]
        
        # Exclude all categorical/string columns (keep only their encoded versions)
        for col in df_features.columns:
            if col not in exclude_cols:
                # Exclude if column is object/string type or ends with common categorical suffixes
                if df_features[col].dtype == 'object' or df_features[col].dtype.name == 'category':
                    exclude_cols.append(col)
                    logger.debug(f"Excluding non-numeric column: {col} (dtype: {df_features[col].dtype})")
        
        feature_columns = [col for col in df_features.columns 
                          if col not in exclude_cols]
        
        logger.info(f"Using {len(feature_columns)} features for {self.target_type} model")
        logger.info(f"Features: {sorted(feature_columns)[:10]}... (showing first 10)")
        
        X = df_features[feature_columns].fillna(0)
        y = df_features[self.target_column]
        
        # Reset index to avoid XGBoost QuantileDMatrix errors with non-unique indices
        df_features = df_features.reset_index(drop=True)
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        # Optional: log-transform target for price to handle skewness
        if self.target_type == 'price':
            y = np.log1p(y)  # log(1 + price) to handle zeros
            self._log_transformed = True
        else:
            self._log_transformed = False
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Train model
        self.model = self._create_model()
        
        # Cross-validation
        cv = CrossValidator(self.config, wandb_logger=wandb_logger)
        cv_results = cv.cross_validate_regressor(
            type(self.model),
            self.model.get_params(),
            X_scaled.assign(date=df_features['date']),
            y,
            task_name=self.target_type  # "price" or "weight"
        )
        
        # Final training on all data
        X_train = X_scaled.drop(columns=['date'], errors='ignore')
        
        # Convert to numpy for better GPU transfer efficiency
        X_train_array = X_train.values if hasattr(X_train, 'values') else X_train
        y_array = y.values if hasattr(y, 'values') else y
        self.model.fit(X_train_array, y_array)
        
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # Calculate training time
        training_time_seconds = time.time() - start_time
        
        # Log performance
        mae = cv_results['overall_metrics'].get('mae', 'N/A')
        r2 = cv_results['overall_metrics'].get('r2', 'N/A')
        logger.info(f"{self.target_type.title()} model training completed. MAE: {mae}, R²: {r2}")
        logger.info(f"Training time: {training_time_seconds:.2f} seconds")
        
        return {
            'cv_results': cv_results,
            'feature_names': self.feature_names,
            'training_samples': len(df_clean),
            'log_transformed': self._log_transformed,
            'training_time_seconds': training_time_seconds
        }
    
    def predict(self, candidates: List[CandidateInput]) -> List[float]:
        """
        Predict values for candidate locations.
        
        Args:
            candidates: List of candidate inputs
            
        Returns:
            List of predicted values (price in € or weight in kg)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        if self.feature_builder is None:
            raise ValueError("Feature builder not initialized. Call fit() first.")
        
        # Convert candidates to DataFrame
        candidate_data = []
        for c in candidates:
            candidate_data.append({
                'origin_id': c.i_location_id,
                'destination_id': c.d_location_id,
                'truck_type': c.truck_type,
                'tipo_mercancia': c.tipo_mercancia,
                'day_of_week': c.day_of_week,
                'week_of_year': c.week_of_year,
                'holiday_flag': c.holiday_flag,
                'od_length_km': 100.0,  # Default, should be updated with real data
            })
        
        df = pd.DataFrame(candidate_data)
        
        # Build features
        df_features = self.feature_builder.build_features(
            df, 
            target_col=self.target_column, 
            fit=False
        )
        
        # Prepare prediction data
        X = df_features[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        
        # Reverse log transformation if applied
        if hasattr(self, '_log_transformed') and self._log_transformed:
            predictions = np.expm1(predictions)  # exp(pred) - 1
        
        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)
        
        return predictions.tolist()


class PriceEstimator(RegressionEstimator):
    """Price estimation model for E[price_{i→d}]."""
    
    def __init__(self, config: Config):
        """Initialize price estimator."""
        super().__init__(config, 'price')


class WeightEstimator(RegressionEstimator):
    """Weight estimation model for E[weight_{i→d}]."""
    
    def __init__(self, config: Config):
        """Initialize weight estimator."""
        super().__init__(config, 'weight')


def train_price(config: Config, wandb_logger=None, quick_test: bool = False) -> ModelBundle:
    """
    Train price estimation model.
    
    This function builds the price dataset, trains a regression model,
    and saves the complete model bundle for inference.
    
    Args:
        config: Configuration object
        wandb_logger: Optional WandbLogger for logging metrics
        quick_test: If True, use only a few OD pairs for fast testing
        
    Returns:
        Trained ModelBundle for price estimation
    """
    logger.info("Starting price model training")
    
    # Build dataset
    try:
        df = build_price_dataset(config, quick_test=quick_test)
    except Exception as e:
        logger.error(f"Failed to build price dataset: {e}")
        raise
    
    if df.empty:
        raise ValueError("No price training data available")
    
    # Initialize and train estimator
    estimator = PriceEstimator(config)
    training_results = estimator.fit(df, wandb_logger=wandb_logger)
    
    # Create model card
    model_card = create_model_card(
        model_type=estimator.model_type,
        task_type="price",
        features=estimator.feature_names,
        cv_results=training_results['cv_results'],
        config=config,
        additional_info={
            'training_samples': training_results['training_samples'],
            'log_transformed': training_results['log_transformed'],
            'target_column': estimator.target_column
        }
    )
    
    # Log summary metrics to wandb
    if wandb_logger:
        wandb_logger.log_task_summary(
            task="price",
            cv_metrics=training_results['cv_results']['cv_metrics'],
            overall_metrics=training_results['cv_results']['overall_metrics'],
            n_folds=training_results['cv_results']['n_folds']
        )
        
        # Log hyperparameters
        wandb_logger.log_hyperparameters(
            estimator.model.get_params(),
            prefix="price/model_"
        )
        
        # Log training time
        wandb_logger.log_metrics({
            "price/training_time_seconds": training_results.get('training_time_seconds', 0),
            "price/training_time_minutes": training_results.get('training_time_seconds', 0) / 60
        })
        
        # Log dataset info
        wandb_logger.log_dataset_info({
            "n_samples_price": training_results['training_samples'],
            "n_features_price": len(estimator.feature_names),
            "feature_names_price": estimator.feature_names[:20],
            "log_transformed": training_results['log_transformed']
        })
    
    # Save model bundle
    from .serialization import ModelVersionManager
    version_manager = ModelVersionManager(config.paths.models_dir)
    bundle_path = version_manager.get_bundle_path('price', quick_test=quick_test)
    
    # Prepare artifacts
    if estimator.feature_builder is None:
        raise ValueError("Feature builder not initialized after training")
    
    encoders = {
        'feature_builder_encoders': estimator.feature_builder.encoders,
        'scaler': estimator.scaler,
        'log_transformed': training_results['log_transformed']
    }
    
    bundle = save_model_bundle(
        model=estimator.model,
        model_type=estimator.model_type,
        task_type="price",
        features=estimator.feature_names,
        model_path=bundle_path,
        encoders=encoders,
        metadata=model_card,
        config=config
    )
    
    # Log model bundle as artifact to wandb
    if wandb_logger:
        wandb_logger.log_model_bundle(
            model_path=bundle_path,
            task="price",
            metrics=training_results['cv_results']['overall_metrics']
        )
    
    logger.info(f"Price model training completed. Bundle saved to: {bundle_path}")
    return bundle


def train_weight(config: Config, wandb_logger=None, quick_test: bool = False) -> ModelBundle:
    """
    Train weight estimation model.
    
    This function builds the weight dataset, trains a regression model,
    and saves the complete model bundle for inference.
    
    Args:
        config: Configuration object
        wandb_logger: Optional WandbLogger for logging metrics
        quick_test: If True, use only a few OD pairs for fast testing
        
    Returns:
        Trained ModelBundle for weight estimation
    """
    logger.info("Starting weight model training")
    
    # Build dataset
    try:
        df = build_weight_dataset(config, quick_test=quick_test)
    except Exception as e:
        logger.error(f"Failed to build weight dataset: {e}")
        raise
    
    if df.empty:
        raise ValueError("No weight training data available")
    
    # Initialize and train estimator
    estimator = WeightEstimator(config)
    training_results = estimator.fit(df, wandb_logger=wandb_logger)
    
    # Create model card
    model_card = create_model_card(
        model_type=estimator.model_type,
        task_type="weight",
        features=estimator.feature_names,
        cv_results=training_results['cv_results'],
        config=config,
        additional_info={
            'training_samples': training_results['training_samples'],
            'log_transformed': training_results['log_transformed'],
            'target_column': estimator.target_column
        }
    )
    
    # Log summary metrics to wandb
    if wandb_logger:
        wandb_logger.log_task_summary(
            task="weight",
            cv_metrics=training_results['cv_results']['cv_metrics'],
            overall_metrics=training_results['cv_results']['overall_metrics'],
            n_folds=training_results['cv_results']['n_folds']
        )
        
        # Log hyperparameters
        wandb_logger.log_hyperparameters(
            estimator.model.get_params(),
            prefix="weight/model_"
        )
        
        # Log training time
        wandb_logger.log_metrics({
            "weight/training_time_seconds": training_results.get('training_time_seconds', 0),
            "weight/training_time_minutes": training_results.get('training_time_seconds', 0) / 60
        })
        
        # Log dataset info
        wandb_logger.log_dataset_info({
            "n_samples_weight": training_results['training_samples'],
            "n_features_weight": len(estimator.feature_names),
            "feature_names_weight": estimator.feature_names[:20],
            "log_transformed": training_results['log_transformed']
        })
    
    # Save model bundle
    from .serialization import ModelVersionManager
    version_manager = ModelVersionManager(config.paths.models_dir)
    bundle_path = version_manager.get_bundle_path('weight', quick_test=quick_test)
    
    # Prepare artifacts
    if estimator.feature_builder is None:
        raise ValueError("Feature builder not initialized after training")
    
    encoders = {
        'feature_builder_encoders': estimator.feature_builder.encoders,
        'scaler': estimator.scaler,
        'log_transformed': training_results['log_transformed']
    }
    
    bundle = save_model_bundle(
        model=estimator.model,
        model_type=estimator.model_type,
        task_type="weight",
        features=estimator.feature_names,
        model_path=bundle_path,
        encoders=encoders,
        metadata=model_card,
        config=config
    )
    
    # Log model bundle as artifact to wandb
    if wandb_logger:
        wandb_logger.log_model_bundle(
            model_path=bundle_path,
            task="weight",
            metrics=training_results['cv_results']['overall_metrics']
        )
    
    logger.info(f"Weight model training completed. Bundle saved to: {bundle_path}")
    return bundle


def predict_price(candidates: List[CandidateInput], bundle: ModelBundle) -> List[float]:
    """
    Predict expected prices for candidate locations using trained model bundle.
    
    Args:
        candidates: List of candidate inputs
        bundle: Trained price model bundle
        
    Returns:
        List of predicted prices in euros
    """
    # Reconstruct estimator from bundle
    config = Config()  # Use default config for inference
    estimator = PriceEstimator(config)
    
    # Load model components
    estimator.model = bundle.model
    estimator.feature_names = bundle.features
    
    # Load encoders and preprocessing
    encoders = bundle.encoders
    if 'feature_builder_encoders' in encoders:
        from .io import create_database_manager
        feature_builder = RegressionFeatureBuilder(config, create_database_manager(config))
        feature_builder.encoders = encoders['feature_builder_encoders']
        estimator.feature_builder = feature_builder
    
    if 'scaler' in encoders:
        estimator.scaler = encoders['scaler']
    
    if 'log_transformed' in encoders:
        estimator._log_transformed = encoders['log_transformed']
    
    # Predict
    return estimator.predict(candidates)


def predict_weight(candidates: List[CandidateInput], bundle: ModelBundle) -> List[float]:
    """
    Predict expected weights for candidate locations using trained model bundle.
    
    Args:
        candidates: List of candidate inputs
        bundle: Trained weight model bundle
        
    Returns:
        List of predicted weights in kilograms
    """
    # Reconstruct estimator from bundle
    config = Config()  # Use default config for inference
    estimator = WeightEstimator(config)
    
    # Load model components
    estimator.model = bundle.model
    estimator.feature_names = bundle.features
    
    # Load encoders and preprocessing
    encoders = bundle.encoders
    if 'feature_builder_encoders' in encoders:
        from .io import create_database_manager
        feature_builder = RegressionFeatureBuilder(config, create_database_manager(config))
        feature_builder.encoders = encoders['feature_builder_encoders']
        estimator.feature_builder = feature_builder
    
    if 'scaler' in encoders:
        estimator.scaler = encoders['scaler']
    
    if 'log_transformed' in encoders:
        estimator._log_transformed = encoders['log_transformed']
    
    # Predict
    return estimator.predict(candidates)


def create_synthetic_price_data(n_samples: int = 1000, config: Optional[Config] = None) -> pd.DataFrame:
    """
    Create synthetic price data for testing purposes.
    
    Args:
        n_samples: Number of samples to generate
        config: Configuration object (optional)
        
    Returns:
        DataFrame with synthetic price data
    """
    np.random.seed(42)
    
    # Generate synthetic features
    data = {
        'origin_id': np.random.randint(1, 51, n_samples),
        'destination_id': np.random.randint(1, 51, n_samples),
        'od_length_km': np.random.exponential(100, n_samples),
        'truck_type': np.random.choice(['normal', 'refrigerado'], n_samples),
        'tipo_mercancia': np.random.choice(['normal', 'refrigerada'], n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'week_of_year': np.random.randint(1, 53, n_samples),
        'holiday_flag': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'peso': np.random.gamma(2, 200, n_samples),  # Weight affects price
        'volumen': np.random.gamma(1.5, 5, n_samples),
        'date': pd.date_range('2024-01-01', periods=n_samples, freq='h')
    }
    
    df = pd.DataFrame(data)
    
    # Generate realistic prices based on distance and weight
    base_price = 0.8 * df['od_length_km'] + 0.1 * df['peso']  # €/km + €/kg
    
    # Add noise and variations
    truck_multiplier = np.where(df['truck_type'] == 'refrigerado', 1.3, 1.0)
    holiday_multiplier = np.where(df['holiday_flag'] == 1, 1.2, 1.0)
    noise = np.random.normal(1.0, 0.2, n_samples)
    
    df['precio'] = base_price * truck_multiplier * holiday_multiplier * noise
    df['precio'] = np.maximum(df['precio'], 10)  # Minimum price
    
    return df


def create_synthetic_weight_data(n_samples: int = 1000, config: Optional[Config] = None) -> pd.DataFrame:
    """
    Create synthetic weight data for testing purposes.
    
    Args:
        n_samples: Number of samples to generate
        config: Configuration object (optional)
        
    Returns:
        DataFrame with synthetic weight data
    """
    np.random.seed(42)
    
    # Generate synthetic features
    data = {
        'origin_id': np.random.randint(1, 51, n_samples),
        'destination_id': np.random.randint(1, 51, n_samples),
        'od_length_km': np.random.exponential(100, n_samples),
        'truck_type': np.random.choice(['normal', 'refrigerado'], n_samples),
        'tipo_mercancia': np.random.choice(['normal', 'refrigerada'], n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'week_of_year': np.random.randint(1, 53, n_samples),
        'holiday_flag': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'precio': np.random.gamma(3, 100, n_samples),  # Price correlates with weight
        'volumen': np.random.gamma(1.5, 5, n_samples),
        'date': pd.date_range('2024-01-01', periods=n_samples, freq='h')
    }
    
    df = pd.DataFrame(data)
    
    # Generate realistic weights based on volume and cargo type
    base_weight = 80 * df['volumen']  # kg per m³
    
    # Add variations
    cargo_multiplier = np.where(df['tipo_mercancia'] == 'refrigerada', 0.8, 1.0)  # Refrigerated lighter
    noise = np.random.normal(1.0, 0.3, n_samples)
    
    df['peso'] = base_weight * cargo_multiplier * noise
    df['peso'] = np.maximum(df['peso'], 50)  # Minimum weight
    
    return df