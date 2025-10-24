"""
Probability estimation module for ATLAS ML package.

Implements models for π_{i→d}(τ): probability that at arrival (elapsed time τ since trip start)
there is at least one available load from zone i to destination d.

Supports both Regime A (bin-level training) and Regime B (daily counts + shape function).
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
from .utils import set_random_seed, to_tau_bin, clamp_probability

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
    tau_minutes: int            # Elapsed minutes since departure


@dataclass
class CandidateOutput:
    """Output data for a single candidate location."""
    
    i_location_id: int          # Origin zone
    d_location_id: int          # Destination zone
    tau_minutes: int            # Elapsed time
    pi: float                   # Probability of ≥1 available load π_{i→d}(τ)
    exp_price: float            # Expected price (€)
    exp_weight: float           # Expected weight (kg)


# NOTE: ShapeFunctionLearner removed - using uniform distribution for Regime B
# No temporal shape function needed without real tau_minutes data


class ProbabilityEstimator:
    """
    Probability estimation for Regime B only (daily counts + uniform distribution).
    
    Predicts daily trip counts and converts to instantaneous probability using:
    P(≥1 load) = 1 - exp(-λ) where λ = n_trips_daily / (24 * 60)
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
                params['objective'] = 'count:poisson'
                return xgb.XGBRegressor(**params)
        else:
            if is_classifier:
                return RandomForestClassifier(**self.config.model.rf_params)
            else:
                return RandomForestRegressor(**self.config.model.rf_params)
    
    def fit(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train Regime B: daily counts with uniform distribution.
        
        Args:
            df: DataFrame with daily aggregated data (columns: n_trips_daily, ...)
            
        Returns:
            Training results dictionary
        """
        logger.info("Training probability model (Regime B: daily counts + uniform distribution)")
        
        if 'n_trips_daily' not in df.columns:
            # Try to use 'n_trips' as daily count
            if 'n_trips' in df.columns:
                df['n_trips_daily'] = df['n_trips']
            else:
                raise ValueError("Regime B requires daily trip counts")
        
        df = df.copy()
        
        # Build features (without tau)
        self.feature_builder = ProbabilityFeatureBuilder(self.config, create_database_manager(self.config))
        df_features = self.feature_builder.build_features(df, include_tau=False, fit=True)
        
        # Train daily count model
        excluded_cols = ['date', 'n_trips_daily', 'origin_id', 'destination_id', 'truck_type', 'tipo_mercancia']
        feature_columns = [col for col in df_features.columns if col not in excluded_cols]
        
        X = df_features[feature_columns].fillna(0)
        y = df_features['n_trips_daily'].clip(lower=0)  # Ensure non-negative
        
        # Reset index to avoid XGBoost QuantileDMatrix errors with non-unique indices
        df_features = df_features.reset_index(drop=True)
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
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
            is_poisson=True
        )
        
        # Final training on all data
        X_train = X_scaled.drop(columns=['date'], errors='ignore')
        self.model.fit(X_train, y)
        
        # NOTE: Shape function disabled - no temporal data available
        # Using uniform distribution instead (honest approach without real tau_minutes data)
        self.shape_learner = None
        
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        logger.info(f"Regime B training completed. MAE: {cv_results['overall_metrics'].get('mae', 'N/A'):.3f}")
        
        return {
            'regime': 'regime_b',
            'cv_results': cv_results,
            'feature_names': self.feature_names,
            'training_samples': len(df)
        }
    
    def predict_probability(self, candidates: List[CandidateInput]) -> List[float]:
        """
        Predict probability π_{i→d}(τ) for candidate locations.
        
        Args:
            candidates: List of candidate inputs
            
        Returns:
            List of probabilities [0, 1]
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
        
        # Predict daily rates (Regime B)
        daily_rates = self.model.predict(X_scaled)
        daily_rates = np.maximum(daily_rates, 0)  # Ensure non-negative
        
        # Convert to instantaneous probabilities (uniform distribution)
        # No shape function - using honest uniform distribution
        probabilities = []
        
        for rate in daily_rates:
            # Rate per minute (uniform distribution over 24 hours)
            lambda_per_minute = rate / (24 * 60)
            
            # Instantaneous probability (1-minute window)
            # P(≥1 arrival in 1 minute) = 1 - exp(-λ)
            probability = 1 - np.exp(-lambda_per_minute)
            probabilities.append(probability)
        
        probabilities = np.array(probabilities)
        
        # Clamp probabilities to [0, 1]
        probabilities = [clamp_probability(p) for p in probabilities]
        
        return probabilities


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
        'scaler': estimator.scaler
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
    Predict all outputs (probability, price, weight) for candidates.
    
    Note: This function requires price and weight model bundles to be available.
    For now, it only predicts probabilities and sets price/weight to placeholder values.
    
    Args:
        candidates: List of candidate inputs
        bundle: Probability model bundle
        
    Returns:
        List of complete candidate outputs
    """
    probabilities = predict_probability(candidates, bundle)
    
    outputs = []
    for candidate, prob in zip(candidates, probabilities):
        output = CandidateOutput(
            i_location_id=candidate.i_location_id,
            d_location_id=candidate.d_location_id,
            tau_minutes=candidate.tau_minutes,
            pi=prob,
            exp_price=100.0,  # Placeholder - implement price prediction
            exp_weight=500.0   # Placeholder - implement weight prediction
        )
        outputs.append(output)
    
    return outputs