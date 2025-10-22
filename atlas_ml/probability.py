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


class ShapeFunctionLearner:
    """
    Learns the time distribution shape function for Regime B.
    
    In Regime B, we estimate daily trip counts and then distribute them across
    time bins using a learned shape function f(τ_bin | features).
    """
    
    def __init__(self, config: Config):
        """
        Initialize shape function learner.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.model = None
        self.feature_builder = None
        self.total_bins = (config.features.max_tau_hours * 60) // config.features.tau_bin_minutes
    
    def fit(self, df: pd.DataFrame) -> None:
        """
        Learn the shape function from historical data.
        
        This creates synthetic pseudo-proportions by analyzing historical
        patterns across distance deciles and day-of-week combinations.
        
        Args:
            df: DataFrame with historical load data
        """
        logger.info("Learning time distribution shape function")
        
        # Create distance deciles
        df = df.copy()
        df['distance_decile'] = pd.qcut(
            df['od_length_km'], 
            q=10, 
            labels=False, 
            duplicates='drop'
        ).fillna(5)  # Default to middle decile
        
        # Create synthetic bin data
        shape_data = []
        
        for _, group in df.groupby(['distance_decile', 'day_of_week']):
            if len(group) < 10:  # Skip small groups
                continue
            
            # Create probabilistic distribution based on distance and day patterns
            # This is a simplified heuristic - replace with domain knowledge if available
            tau_bins = np.arange(self.total_bins)
            
            # Distance-based pattern: shorter trips peak earlier
            distance_factor = group['distance_decile'].iloc[0]
            peak_bin = int(self.total_bins * (0.2 + 0.6 * distance_factor / 9))
            
            # Day-of-week pattern: weekends different from weekdays
            dow_factor = 1.0 if group['day_of_week'].iloc[0] < 5 else 0.7
            
            # Create bell-curve like distribution
            shape_weights = np.exp(-0.5 * ((tau_bins - peak_bin) / (self.total_bins / 6)) ** 2)
            shape_weights *= dow_factor
            shape_weights /= shape_weights.sum()  # Normalize to probabilities
            
            # Create training samples
            for bin_idx, weight in enumerate(shape_weights):
                if weight > 0.001:  # Skip very low probability bins
                    shape_data.append({
                        'distance_decile': distance_factor,
                        'day_of_week': group['day_of_week'].iloc[0],
                        'tau_bin': bin_idx,
                        'shape_weight': weight,
                        'holiday_flag': group['holiday_flag'].iloc[0] if 'holiday_flag' in group else 0
                    })
        
        if not shape_data:
            logger.warning("No data available for shape function learning. Using uniform distribution.")
            self._create_uniform_model()
            return
        
        shape_df = pd.DataFrame(shape_data)
        
        # Train multinomial model for shape function
        features = ['distance_decile', 'day_of_week', 'holiday_flag']
        X = shape_df[features]
        
        # Convert to multinomial classification problem
        # Each time bin is a class, weighted by shape_weight
        y_multinomial = []
        sample_weights = []
        
        for _, row in shape_df.iterrows():
            y_multinomial.append(row['tau_bin'])
            sample_weights.append(row['shape_weight'])
        
        y_multinomial = np.array(y_multinomial)
        sample_weights = np.array(sample_weights)
        
        # Train model
        if HAS_XGBOOST and self.config.model.probability_model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                objective='multi:softprob',
                n_estimators=50,  # Simpler model for shape function
                max_depth=4,
                random_state=self.config.random_state
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=50,
                max_depth=8,
                random_state=self.config.random_state
            )
        
        self.model.fit(X, y_multinomial, sample_weight=sample_weights)
        logger.info(f"Trained shape function on {len(shape_df)} samples")
    
    def predict_shape(self, distance_decile: int, day_of_week: int, holiday_flag: int = 0) -> np.ndarray:
        """
        Predict time distribution shape for given conditions.
        
        Args:
            distance_decile: Distance decile (0-9)
            day_of_week: Day of week (0-6)
            holiday_flag: Holiday flag (0/1)
            
        Returns:
            Array of probabilities for each time bin (sums to 1)
        """
        if self.model is None:
            # Fallback: uniform distribution
            return np.ones(self.total_bins) / self.total_bins
        
        X = np.array([[distance_decile, day_of_week, holiday_flag]])
        
        try:
            shape_probs = self.model.predict_proba(X)[0]
            
            # Ensure we have probabilities for all bins
            if len(shape_probs) < self.total_bins:
                # Pad with small values
                padded_probs = np.zeros(self.total_bins)
                padded_probs[:len(shape_probs)] = shape_probs
                padded_probs[len(shape_probs):] = 1e-6
                shape_probs = padded_probs
            
            # Normalize
            shape_probs = shape_probs / shape_probs.sum()
            return shape_probs
            
        except Exception as e:
            logger.warning(f"Shape function prediction failed: {e}. Using uniform distribution.")
            return np.ones(self.total_bins) / self.total_bins
    
    def _create_uniform_model(self) -> None:
        """Create a dummy model that returns uniform distribution."""
        class UniformModel:
            def predict_proba(self, X):
                n_samples = X.shape[0]
                n_bins = (24 * 60) // 10  # Default bin count
                return np.ones((n_samples, n_bins)) / n_bins
        
        self.model = UniformModel()


class ProbabilityEstimator:
    """
    Main class for probability estimation with support for both training regimes.
    
    Regime A: Train directly on time-binned data with binary labels
    Regime B: Train daily count model + shape function, then compute π = 1 - exp(-λW)
    """
    
    def __init__(self, config: Config):
        """
        Initialize probability estimator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.regime = config.training.training_regime
        self.model = None
        self.shape_learner = None
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
    
    def fit_regime_a(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train Regime A: direct time-bin classification.
        
        Args:
            df: DataFrame with time-binned data (columns: tau_bin, n_trips_bin, ...)
            
        Returns:
            Training results dictionary
        """
        logger.info("Training Regime A: time-bin classification")
        
        # Create binary labels (1 if n_trips_bin >= 1, 0 otherwise)
        if 'n_trips_bin' not in df.columns:
            raise ValueError("Regime A requires 'n_trips_bin' column")
        
        df = df.copy()
        df['has_load'] = (df['n_trips_bin'] >= 1).astype(int)
        
        # Build features
        self.feature_builder = ProbabilityFeatureBuilder(self.config, create_database_manager(self.config))
        df_features = self.feature_builder.build_features(df, include_tau=True, fit=True)
        
        # Prepare training data
        feature_columns = [col for col in df_features.columns 
                          if col not in ['date', 'has_load', 'n_trips_bin', 'origin_id', 'destination_id']]
        
        X = df_features[feature_columns].fillna(0)
        y = df_features['has_load']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Train model
        self.model = self._create_model(self.config.model.probability_model_type, is_classifier=True)
        
        # Cross-validation
        cv = CrossValidator(self.config)
        cv_results = cv.cross_validate_classifier(
            type(self.model), 
            self.model.get_params(),
            X_scaled.assign(date=df_features['date']),
            y
        )
        
        # Final training on all data
        X_train = X_scaled.drop(columns=['date'], errors='ignore')
        self.model.fit(X_train, y)
        
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        logger.info(f"Regime A training completed. ROC-AUC: {cv_results['overall_metrics'].get('roc_auc', 'N/A'):.3f}")
        
        return {
            'regime': 'regime_a',
            'cv_results': cv_results,
            'feature_names': self.feature_names,
            'training_samples': len(df)
        }
    
    def fit_regime_b(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train Regime B: daily counts + shape function.
        
        Args:
            df: DataFrame with daily aggregated data (columns: n_trips_daily, ...)
            
        Returns:
            Training results dictionary
        """
        logger.info("Training Regime B: daily counts + shape function")
        
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
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
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
        
        # Train shape function
        self.shape_learner = ShapeFunctionLearner(self.config)
        self.shape_learner.fit(df)
        
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
        
        # Build features
        if self.regime == 'regime_a':
            df_features = self.feature_builder.build_features(df, include_tau=True, fit=False)
        else:
            df_features = self.feature_builder.build_features(df, include_tau=False, fit=False)
        
        # Prepare prediction data
        X = df_features[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        if self.regime == 'regime_a':
            # Direct probability prediction
            probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        else:  # regime_b
            # Predict daily rates
            daily_rates = self.model.predict(X_scaled)
            daily_rates = np.maximum(daily_rates, 0)  # Ensure non-negative
            
            # Convert to time-bin rates using shape function
            probabilities = []
            W = self.config.training.wait_tolerance_minutes  # Additional wait time
            
            for i, (rate, candidate) in enumerate(zip(daily_rates, candidates)):
                # Get distance decile (approximate)
                distance_decile = min(9, int(df_features.loc[i, 'od_length_km'] / 50))  # 50km per decile
                
                # Get shape distribution
                shape_probs = self.shape_learner.predict_shape(
                    distance_decile, candidate.day_of_week, candidate.holiday_flag
                )
                
                # Convert tau to bin
                tau_bin = to_tau_bin(candidate.tau_minutes, self.config.features.tau_bin_minutes)
                tau_bin = min(tau_bin, len(shape_probs) - 1)
                
                # Rate for this specific time bin
                bin_rate = rate * shape_probs[tau_bin]
                
                # Add wait tolerance: rate over (current_bin + wait_time_bins)
                wait_bins = W // self.config.features.tau_bin_minutes
                total_rate = 0
                for j in range(tau_bin, min(tau_bin + wait_bins + 1, len(shape_probs))):
                    total_rate += rate * shape_probs[j]
                
                # Convert to probability: π = 1 - exp(-λ * 1) where λ is rate per time unit
                # Here we assume rate is per day, convert to rate per bin_duration
                bin_duration_days = self.config.features.tau_bin_minutes / (24 * 60)
                effective_rate = total_rate * bin_duration_days
                
                probability = 1 - np.exp(-effective_rate)
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
    
    # Train based on regime
    if config.training.training_regime == 'regime_a':
        training_results = estimator.fit_regime_a(df)
    else:
        training_results = estimator.fit_regime_b(df)
    
    # Create model card
    model_card = create_model_card(
        model_type=config.model.probability_model_type,
        task_type="probability",
        training_regime=config.training.training_regime,
        features=estimator.feature_names,
        cv_results=training_results['cv_results'],
        config=config,
        additional_info={
            'training_samples': training_results['training_samples'],
            'regime_explanation': training_results['regime']
        }
    )
    
    # Save model bundle
    from .serialization import ModelVersionManager
    version_manager = ModelVersionManager(config.paths.models_dir)
    bundle_path = version_manager.get_bundle_path(
        'probability',
        training_regime=config.training.training_regime
    )
    
    # Prepare artifacts
    encoders = {
        'feature_builder_encoders': estimator.feature_builder.encoders,
        'scaler': estimator.scaler
    }
    
    if estimator.shape_learner:
        encoders['shape_learner'] = estimator.shape_learner
    
    bundle = save_model_bundle(
        model=estimator.model,
        model_type=config.model.probability_model_type,
        task_type="probability",
        features=estimator.feature_names,
        model_path=bundle_path,
        encoders=encoders,
        metadata=model_card,
        training_regime=config.training.training_regime,
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
    estimator.regime = bundle.training_regime or 'regime_b'
    
    # Load encoders
    encoders = bundle.encoders
    if 'feature_builder_encoders' in encoders:
        feature_builder = ProbabilityFeatureBuilder(config, None)  # No DB needed for inference
        feature_builder.encoders = encoders['feature_builder_encoders']
        estimator.feature_builder = feature_builder
    
    if 'scaler' in encoders:
        estimator.scaler = encoders['scaler']
    
    if 'shape_learner' in encoders:
        estimator.shape_learner = encoders['shape_learner']
    
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