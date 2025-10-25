"""
Trip count prediction module for ATLAS ML package.

Predicts the expected number of freight vehicles (n_trips_logistic) per day
for origin-destination pairs. This provides interpretable estimates for route optimization.

The model predicts: E[n_trips_logistic_{i→d}] = expected daily freight vehicles from i to d
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from .config import Config
from .evaluation import CrossValidator, create_model_card
from .featurization import ProbabilityFeatureBuilder, build_probability_dataset
from .io import create_database_manager
from .serialization import ModelBundle, save_model_bundle
from .utils import set_random_seed, to_tau_bin

logger = logging.getLogger(__name__)


@dataclass
class CandidateInput:
    """Input data for a single candidate location at elapsed time τ."""
    
    i_location_id: int          # Origin zone (candidate)
    d_location_id: int          # Destination zone (final destination)
    truck_type: str             # "normal" | "refrigerado"
    tipo_mercancia: str         # "normal" | "refrigerada"
    day_of_week: int            # 0..6 (Monday=0, Sunday=6)
    week_of_year: int           # 1..53
    holiday_flag: int           # 0/1
    tau_minutes: int            # Elapsed minutes since departure (currently unused)


@dataclass
class CandidateOutput:
    """Output data for a single candidate location."""
    
    i_location_id: int          # Origin zone
    d_location_id: int          # Destination zone
    tau_minutes: int            # Elapsed time (currently unused)
    expected_trips_per_day: float  # Expected freight vehicles per day (n_trips_logistic)
    exp_price: float            # Expected price (€)
    exp_weight: float           # Expected weight (kg)


# NOTE: ShapeFunctionLearner removed - using uniform distribution instead
# No temporal shape function needed without real tau_minutes data


class ProbabilityEstimator:
    """
    Probability estimator using daily trip count prediction with uniform distribution.
    
    Predicts daily trip counts and converts to instantaneous probability using:
    P(≥1 load in 1 min) = 1 - exp(-λ) where λ = n_trips_daily / (24 * 60)
    
    This approach assumes trips are uniformly distributed over 24 hours.
    """
    
    def __init__(self, config: Config):
        """
        Initialize probability estimator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.model = None
        self.feature_builder = None
        self.scaler = StandardScaler()
        
        set_random_seed(config.random_state)
    
    def _create_model(self, model_type: str, is_classifier: bool = True) -> Any:
        """Create model instance based on configuration."""
        if model_type == 'xgboost' and HAS_XGBOOST:
            if is_classifier:
                params = self.config.model.xgb_params.copy()
                params['objective'] = 'binary:logistic'
                return xgb.XGBClassifier(**params)
            else:
                params = self.config.model.xgb_params.copy()
                # Use squared error for log-transformed target
                params['objective'] = 'reg:squarederror'
                return xgb.XGBRegressor(**params)
        else:
            if is_classifier:
                return RandomForestClassifier(**self.config.model.rf_params)
            else:
                return RandomForestRegressor(**self.config.model.rf_params)
    
    def fit(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train probability model using daily trip counts with uniform distribution.
        
        Approach:
        1. Predict daily trip counts (n_trips_logistic from DB) using XGBoost/RandomForest
        2. Convert to instantaneous probability: P = 1 - exp(-λ) where λ = n_trips_daily/(24*60)
        3. Assumes uniform distribution over 24 hours (simple, honest approach)
        
        Target variable source:
        - Primary: n_trips_logistic from app.sodd_loads_filtered (freight vehicle count estimate)
        - This represents the estimated number of freight transport vehicles per OD pair per day
        
        Args:
            df: DataFrame with daily aggregated data (columns: n_trips_daily or n_trips_logistic, ...)
            
        Returns:
            Training results dictionary with CV metrics and feature names
        """
        logger.info("Training probability model (daily counts + uniform distribution)")
        logger.info("Target variable: n_trips_logistic (freight vehicle count from database)")
        
        if 'n_trips_daily' not in df.columns:
            # Primary source: n_trips_logistic from sodd_loads_filtered table
            if 'n_trips_logistic' in df.columns:
                df['n_trips_daily'] = df['n_trips_logistic']
                logger.info(f"Using n_trips_logistic as target (range: {df['n_trips_daily'].min():.0f} - {df['n_trips_daily'].max():.0f})")
            # Fallback to 'n_trips' for backward compatibility
            elif 'n_trips' in df.columns:
                df['n_trips_daily'] = df['n_trips']
                logger.warning("Using 'n_trips' as fallback target (n_trips_logistic not found)")
            else:
                raise ValueError("No valid target found: requires n_trips_daily, n_trips_logistic, or n_trips column")
        
        df = df.copy()
        
        # Build features (without tau)
        self.feature_builder = ProbabilityFeatureBuilder(self.config, create_database_manager(self.config))
        df_features = self.feature_builder.build_features(df, include_tau=False, fit=True)
        
        # Train daily count model
        excluded_cols = ['date', 'n_trips_daily', 'origin_id', 'destination_id', 'truck_type', 'tipo_mercancia']
        feature_columns = [col for col in df_features.columns if col not in excluded_cols]
        
        X = df_features[feature_columns].fillna(0)
        y_raw = df_features['n_trips_daily'].clip(lower=0)  # Ensure non-negative
        
        # Log-transform target to handle heavy-tailed distribution (75% < 1, 0.03% > 500)
        # This allows model to learn both low and high trip counts effectively
        y = np.log1p(y_raw)  # log(1 + n_trips) to handle zeros
        logger.info(f"Target stats - Raw: min={y_raw.min():.2f}, max={y_raw.max():.2f}, mean={y_raw.mean():.2f}")
        logger.info(f"Target stats - Log: min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}")
        
        # Reset index to avoid XGBoost QuantileDMatrix errors with non-unique indices
        df_features = df_features.reset_index(drop=True)
        X = X.reset_index(drop=True)
        # y is now numpy array from log1p, convert to Series for compatibility
        y = pd.Series(y, name='n_trips_log')
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Train count model
        self.model = self._create_model(self.config.model.probability_model_type, is_classifier=False)
        
        # Cross-validation for count model
        cv = CrossValidator(self.config)
        cv_results = cv.cross_validate_regressor(
            type(self.model),
            self.model.get_params(),
            X_scaled.assign(date=df_features['date']),
            y,
            is_poisson=False  # Using log-transformed target with squared error
        )
        
        # Final training on all data
        X_train = X_scaled.drop(columns=['date'], errors='ignore')
        self.model.fit(X_train, y)
        
        # NOTE: No temporal shape function - using uniform distribution
        # (honest approach without real time-of-day data)
        self.shape_learner = None
        
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        logger.info(f"Training completed. MAE: {cv_results['overall_metrics'].get('mae', 'N/A'):.3f}")
        logger.info("Note: Target was log-transformed (log1p) to handle heavy-tailed distribution")
        
        return {
            'cv_results': cv_results,
            'feature_names': self.feature_names,
            'training_samples': len(df),
            'log_transformed': True  # Flag for inference to apply expm1
        }
    
    def predict_probability(self, candidates: List[CandidateInput]) -> List[float]:
        """
        Predict expected daily trip counts for candidate locations.
        
        Returns the predicted number of freight vehicles (n_trips_logistic) per day
        for each origin-destination pair. This is more interpretable than probability
        and matches the training data directly.
        
        Args:
            candidates: List of candidate inputs
            
        Returns:
            List of predicted daily trip counts (n_trips_logistic per day)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        if self.feature_builder is None:
            raise ValueError("Feature builder not initialized. Call fit() first or set feature_builder manually.")
        
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
                'tau_minutes': c.tau_minutes,
                'od_length_km': 100.0,  # Default, will be updated with real data if available
            })
        
        df = pd.DataFrame(candidate_data)
        
        # Build features (no tau, no historical features)
        df_features = self.feature_builder.build_features(df, include_tau=False, fit=False)
        
        # Prepare prediction data
        X = df_features[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Predict daily trip counts (n_trips_logistic from training)
        daily_rates = self.model.predict(X_scaled)
        daily_rates = np.maximum(daily_rates, 0)  # Ensure non-negative
        
        # Return daily trip counts directly (no conversion to per-minute probability)
        # This is more interpretable: "Expected X freight vehicles per day on this route"
        return daily_rates.tolist()


def train_probability(config: Config) -> ModelBundle:
    """
    Train probability estimation model.
    
    This is the main entry point for training probability models. It automatically
    detects the appropriate training regime based on data availability.
    
    Args:
        config: Configuration object
        
    Returns:
        Trained ModelBundle
    """
    logger.info("Starting probability model training")
    
    # Build dataset
    try:
        df = build_probability_dataset(config)
    except Exception as e:
        logger.error(f"Failed to build probability dataset: {e}")
        raise
    
    if df.empty:
        raise ValueError("No training data available")
    
    # Initialize estimator
    estimator = ProbabilityEstimator(config)
    
    # Train (only one training mode supported)
    training_results = estimator.fit(df)
    
    # Create model card
    model_card = create_model_card(
        model_type=config.model.probability_model_type,
        task_type="probability",
        features=estimator.feature_names,
        cv_results=training_results['cv_results'],
        config=config,
        additional_info={
            'training_samples': training_results['training_samples']
        }
    )
    
    # Save model bundle
    from .serialization import ModelVersionManager
    version_manager = ModelVersionManager(config.paths.models_dir)
    bundle_path = version_manager.get_bundle_path('probability')
    
    # Prepare artifacts
    if estimator.feature_builder is None:
        raise ValueError("Feature builder not initialized after training")
    
    encoders = {
        'feature_builder_encoders': estimator.feature_builder.encoders,
        'scaler': estimator.scaler,
        'log_transformed': training_results.get('log_transformed', False)  # Flag for inference
    }
    
    bundle = save_model_bundle(
        model=estimator.model,
        model_type=config.model.probability_model_type,
        task_type="probability",
        features=estimator.feature_names,
        model_path=bundle_path,
        encoders=encoders,
        metadata=model_card,
        config=config
    )
    
    logger.info(f"Probability model training completed. Bundle saved to: {bundle_path}")
    return bundle


def predict_probability(candidates: List[CandidateInput], bundle: ModelBundle) -> List[float]:
    """
    Predict probabilities for candidate locations using trained model bundle.
    
    Args:
        candidates: List of candidate inputs
        bundle: Trained model bundle
        
    Returns:
        List of probabilities
    """
    # Reconstruct estimator from bundle
    config = Config()  # Use default config for inference
    estimator = ProbabilityEstimator(config)
    
    # Load model components
    estimator.model = bundle.model
    estimator.feature_names = bundle.features
    
    # Load encoders
    encoders = bundle.encoders
    if 'feature_builder_encoders' in encoders:
        from .io import create_database_manager
        feature_builder = ProbabilityFeatureBuilder(config, create_database_manager(config))
        feature_builder.encoders = encoders['feature_builder_encoders']
        estimator.feature_builder = feature_builder
    
    if 'scaler' in encoders:
        estimator.scaler = encoders['scaler']
    
    # Predict
    return estimator.predict_probability(candidates)


def predict_all(candidates: List[CandidateInput], bundle: ModelBundle) -> List[CandidateOutput]:
    """
    Predict all outputs (trip counts, price, weight) for candidates.
    
    Note: This function requires price and weight model bundles to be available.
    For now, it only predicts trip counts and sets price/weight to placeholder values.
    
    Args:
        candidates: List of candidate inputs
        bundle: Trip count prediction model bundle
        
    Returns:
        List of complete candidate outputs
    """
    expected_trips = predict_probability(candidates, bundle)  # Returns daily trip counts
    
    outputs = []
    for candidate, trips in zip(candidates, expected_trips):
        output = CandidateOutput(
            i_location_id=candidate.i_location_id,
            d_location_id=candidate.d_location_id,
            tau_minutes=candidate.tau_minutes,
            expected_trips_per_day=trips,  # Daily freight vehicle count
            exp_price=100.0,  # Placeholder - implement price prediction
            exp_weight=500.0   # Placeholder - implement weight prediction
        )
        outputs.append(output)
    
    return outputs