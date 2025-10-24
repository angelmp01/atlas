"""
Evaluation module for ATLAS ML package.

Provides temporal cross-validation, metrics calculation, and model card generation
for assessing model performance and tracking model quality over time.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, auc, average_precision_score, brier_score_loss,
    mean_absolute_error, mean_squared_error, precision_recall_curve,
    roc_auc_score, roc_curve
)

from .config import Config

logger = logging.getLogger(__name__)


class TemporalSplitter:
    """Handles temporal cross-validation splits for time series data."""
    
    def __init__(
        self,
        train_months: int = 6,
        test_months: int = 1,
        min_train_samples: int = 1000
    ):
        """
        Initialize temporal splitter.
        
        Args:
            train_months: Number of months for training
            test_months: Number of months for testing
            min_train_samples: Minimum samples required for training
        """
        self.train_months = train_months
        self.test_months = test_months
        self.min_train_samples = min_train_samples
    
    def split(self, df: pd.DataFrame, date_col: str = 'date') -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate temporal cross-validation splits.
        
        The splitting strategy ensures that training data always comes before
        test data to prevent data leakage. Each fold uses a sliding window
        approach where the test period follows the training period.
        
        Args:
            df: DataFrame with date column
            date_col: Name of date column
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        if date_col not in df.columns:
            raise ValueError(f"Date column '{date_col}' not found in DataFrame")
        
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        df_copy = df_copy.sort_values(date_col)
        
        # Get date range
        min_date = df_copy[date_col].min()
        max_date = df_copy[date_col].max()
        
        splits = []
        current_date = min_date
        
        while current_date <= max_date:
            # Define training period
            train_start = current_date
            train_end = train_start + timedelta(days=30 * self.train_months)
            
            # Define test period
            test_start = train_end
            test_end = test_start + timedelta(days=30 * self.test_months)
            
            # Get indices
            train_mask = (df_copy[date_col] >= train_start) & (df_copy[date_col] < train_end)
            test_mask = (df_copy[date_col] >= test_start) & (df_copy[date_col] < test_end)
            
            train_indices = df_copy.index[train_mask].values
            test_indices = df_copy.index[test_mask].values
            
            # Check minimum sample requirements
            if len(train_indices) >= self.min_train_samples and len(test_indices) > 0:
                splits.append((train_indices, test_indices))
                logger.debug(
                    f"Created split: train {train_start.date()} to {train_end.date()} "
                    f"({len(train_indices)} samples), "
                    f"test {test_start.date()} to {test_end.date()} "
                    f"({len(test_indices)} samples)"
                )
            
            # Move to next period (sliding window with 1-month step)
            current_date += timedelta(days=30)
            
            # Stop if we can't create a complete test period
            if test_end > max_date:
                break
        
        logger.info(f"Created {len(splits)} temporal CV splits")
        return splits


class ModelEvaluator:
    """Evaluates model performance using various metrics."""
    
    @staticmethod
    def evaluate_binary_classifier(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        y_pred_binary: Optional[np.ndarray] = None,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Evaluate binary classification model.
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            y_pred_binary: Predicted binary labels (optional)
            threshold: Classification threshold
            
        Returns:
            Dictionary of metrics
        """
        if y_pred_binary is None:
            y_pred_binary = (y_pred_proba >= threshold).astype(int)
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred_binary)
        
        # Probabilistic metrics
        try:
            roc_auc = roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            roc_auc = np.nan
        
        try:
            pr_auc = average_precision_score(y_true, y_pred_proba)
        except ValueError:
            pr_auc = np.nan
        
        brier_score = brier_score_loss(y_true, y_pred_proba)
        
        # Precision and recall at threshold
        precision = np.sum((y_pred_binary == 1) & (y_true == 1)) / (np.sum(y_pred_binary == 1) + 1e-10)
        recall = np.sum((y_pred_binary == 1) & (y_true == 1)) / (np.sum(y_true == 1) + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        return {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'brier_score': brier_score,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'threshold': threshold
        }
    
    @staticmethod
    def evaluate_regressor(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate regression model.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # Mean Absolute Percentage Error (safe version)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'r2': r2
        }
    
    @staticmethod
    def evaluate_poisson_regressor(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_rate: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate Poisson regression model.
        
        Args:
            y_true: True count values
            y_pred: Predicted count values
            y_pred_rate: Predicted rate parameters (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = ModelEvaluator.evaluate_regressor(y_true, y_pred)
        
        # Poisson deviance
        if y_pred_rate is not None:
            # Clip to avoid log(0)
            y_pred_rate_clipped = np.clip(y_pred_rate, 1e-10, None)
            
            # Poisson deviance: 2 * sum(y * log(y/mu) - (y - mu))
            # where y is observed and mu is predicted rate
            deviance_terms = []
            for i in range(len(y_true)):
                y_obs = y_true[i]
                mu_pred = y_pred_rate_clipped[i]
                
                if y_obs > 0:
                    deviance = y_obs * np.log(y_obs / mu_pred) - (y_obs - mu_pred)
                else:
                    deviance = -mu_pred
                
                deviance_terms.append(deviance)
            
            mean_deviance = 2 * np.mean(deviance_terms)
            metrics['poisson_deviance'] = mean_deviance
        
        return metrics


class CrossValidator:
    """Handles cross-validation with temporal splits."""
    
    def __init__(self, config: Config):
        """
        Initialize cross-validator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.splitter = TemporalSplitter(
            train_months=config.training.cv_months_train,
            test_months=config.training.cv_months_test
        )
    
    def cross_validate_classifier(
        self,
        model_class: Any,
        model_params: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        date_col: str = 'date'
    ) -> Dict[str, Any]:
        """
        Perform temporal cross-validation for classification.
        
        Args:
            model_class: Model class to instantiate
            model_params: Model hyperparameters
            X: Feature matrix
            y: Target vector
            date_col: Date column name
            
        Returns:
            Cross-validation results
        """
        # Create combined dataframe for splitting
        df_combined = X.copy()
        df_combined['target'] = y
        
        # Generate splits
        splits = self.splitter.split(df_combined, date_col)
        
        if not splits:
            raise ValueError("No valid temporal splits could be created")
        
        fold_results = []
        all_predictions = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            logger.info(f"Processing fold {fold_idx + 1}/{len(splits)}")
            
            # Get train/test data
            X_train = X.loc[train_idx].drop(columns=[date_col], errors='ignore')
            X_test = X.loc[test_idx].drop(columns=[date_col], errors='ignore')
            
            # Ensure y is 1D numpy array, not DataFrame
            y_train = y.loc[train_idx]
            if isinstance(y_train, pd.DataFrame):
                y_train = y_train.iloc[:, 0]  # Take first column as Series
            y_train = y_train.values  # Convert to numpy array
            
            y_test = y.loc[test_idx]
            if isinstance(y_test, pd.DataFrame):
                y_test = y_test.iloc[:, 0]
            y_test = y_test.values  # Convert to numpy array
            
            # Train model
            model = model_class(**model_params)
            model.fit(X_train, y_train)
            
            # Predict
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred_binary = model.predict(X_test)
            
            # Evaluate
            fold_metrics = ModelEvaluator.evaluate_binary_classifier(
                y_test, y_pred_proba, y_pred_binary  # y_test is already numpy array
            )
            fold_metrics['fold'] = fold_idx
            fold_metrics['train_size'] = len(train_idx)
            fold_metrics['test_size'] = len(test_idx)
            
            fold_results.append(fold_metrics)
            
            # Store predictions for later analysis
            pred_df = pd.DataFrame({
                'fold': fold_idx,
                'index': test_idx,
                'y_true': y_test,  # Already numpy array
                'y_pred_proba': y_pred_proba,
                'y_pred_binary': y_pred_binary
            })
            all_predictions.append(pred_df)
        
        # Aggregate results
        fold_df = pd.DataFrame(fold_results)
        predictions_df = pd.concat(all_predictions, ignore_index=True)
        
        # Calculate overall metrics
        overall_metrics = ModelEvaluator.evaluate_binary_classifier(
            predictions_df['y_true'].values,
            predictions_df['y_pred_proba'].values,
            predictions_df['y_pred_binary'].values
        )
        
        # Calculate cross-validation statistics
        cv_metrics = {}
        for metric in ['accuracy', 'roc_auc', 'pr_auc', 'brier_score', 'precision', 'recall', 'f1_score']:
            values = fold_df[metric].dropna()
            if len(values) > 0:
                cv_metrics[f'{metric}_mean'] = values.mean()
                cv_metrics[f'{metric}_std'] = values.std()
                cv_metrics[f'{metric}_min'] = values.min()
                cv_metrics[f'{metric}_max'] = values.max()
        
        return {
            'cv_metrics': cv_metrics,
            'overall_metrics': overall_metrics,
            'fold_results': fold_results,
            'predictions': predictions_df,
            'n_folds': len(splits)
        }
    
    def cross_validate_regressor(
        self,
        model_class: Any,
        model_params: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        date_col: str = 'date',
        is_poisson: bool = False
    ) -> Dict[str, Any]:
        """
        Perform temporal cross-validation for regression.
        
        Args:
            model_class: Model class to instantiate
            model_params: Model hyperparameters
            X: Feature matrix
            y: Target vector
            date_col: Date column name
            is_poisson: Whether to compute Poisson-specific metrics
            
        Returns:
            Cross-validation results
        """
        # Create combined dataframe for splitting
        df_combined = X.copy()
        df_combined['target'] = y
        
        # Generate splits
        splits = self.splitter.split(df_combined, date_col)
        
        if not splits:
            raise ValueError("No valid temporal splits could be created")
        
        fold_results = []
        all_predictions = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            logger.info(f"Processing fold {fold_idx + 1}/{len(splits)}")
            
            # Get train/test data
            X_train = X.loc[train_idx].drop(columns=[date_col], errors='ignore')
            X_test = X.loc[test_idx].drop(columns=[date_col], errors='ignore')
            
            # Ensure y is 1D numpy array, not DataFrame - XGBoost requires this
            y_train = y.loc[train_idx]
            if isinstance(y_train, pd.DataFrame):
                y_train = y_train.iloc[:, 0]  # Take first column as Series
            y_train = y_train.values  # Convert to numpy array
            
            y_test = y.loc[test_idx]
            if isinstance(y_test, pd.DataFrame):
                y_test = y_test.iloc[:, 0]
            y_test = y_test.values  # Convert to numpy array
            
            # Train model
            model = model_class(**model_params)
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Evaluate
            if is_poisson:
                fold_metrics = ModelEvaluator.evaluate_poisson_regressor(
                    y_test, y_pred
                )
            else:
                fold_metrics = ModelEvaluator.evaluate_regressor(
                    y_test, y_pred  # y_test is already numpy array
                )
            
            fold_metrics['fold'] = fold_idx
            fold_metrics['train_size'] = len(train_idx)
            fold_metrics['test_size'] = len(test_idx)
            
            fold_results.append(fold_metrics)
            
            # Store predictions
            pred_df = pd.DataFrame({
                'fold': fold_idx,
                'index': test_idx,
                'y_true': y_test,  # Already numpy array
                'y_pred': y_pred
            })
            all_predictions.append(pred_df)
        
        # Aggregate results
        fold_df = pd.DataFrame(fold_results)
        predictions_df = pd.concat(all_predictions, ignore_index=True)
        
        # Calculate overall metrics
        if is_poisson:
            overall_metrics = ModelEvaluator.evaluate_poisson_regressor(
                predictions_df['y_true'].values,
                predictions_df['y_pred'].values
            )
        else:
            overall_metrics = ModelEvaluator.evaluate_regressor(
                predictions_df['y_true'].values,
                predictions_df['y_pred'].values
            )
        
        # Calculate cross-validation statistics
        cv_metrics = {}
        for metric in ['mae', 'mse', 'rmse', 'mape', 'r2']:
            if metric in fold_df.columns:
                values = fold_df[metric].dropna()
                if len(values) > 0:
                    cv_metrics[f'{metric}_mean'] = values.mean()
                    cv_metrics[f'{metric}_std'] = values.std()
                    cv_metrics[f'{metric}_min'] = values.min()
                    cv_metrics[f'{metric}_max'] = values.max()
        
        return {
            'cv_metrics': cv_metrics,
            'overall_metrics': overall_metrics,
            'fold_results': fold_results,
            'predictions': predictions_df,
            'n_folds': len(splits)
        }


def create_model_card(
    model_type: str,
    task_type: str,
    training_regime: Optional[str],
    features: List[str],
    cv_results: Dict[str, Any],
    config: Config,
    additional_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a comprehensive model card with training information and metrics.
    
    A model card documents the model's intended use, training process, performance,
    and limitations. This is essential for model governance and reproducibility.
    
    Args:
        model_type: Type of model ('xgboost', 'randomforest')
        task_type: Task type ('probability', 'price', 'weight')
        training_regime: Training regime for probability models
        features: List of feature names
        cv_results: Cross-validation results
        config: Configuration used for training
        additional_info: Additional information to include
        
    Returns:
        Model card dictionary
    """
    model_card = {
        # Model identification
        "model_info": {
            "model_type": model_type,
            "task_type": task_type,
            "training_regime": training_regime,
            "version": "1.0.0",
            "created_at": datetime.now().isoformat(),
        },
        
        # Training configuration
        "training_config": {
            "features": {
                "count": len(features),
                "names": features[:20],  # Limit for readability
            },
            "model_params": getattr(config.model, f"{model_type}_params", {}),
            "cv_setup": {
                "train_months": config.training.cv_months_train,
                "test_months": config.training.cv_months_test,
                "n_folds": cv_results.get("n_folds", 0),
            },
            "data_filters": {
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
            }
        },
        
        # Performance metrics
        "performance": {
            "cv_metrics": cv_results.get("cv_metrics", {}),
            "overall_metrics": cv_results.get("overall_metrics", {}),
            "best_fold_score": None,
            "worst_fold_score": None,
        },
        
        # Model interpretation
        "interpretation": {
            "training_regime_explanation": _get_regime_explanation(training_regime),
            "metric_explanations": _get_metric_explanations(task_type),
            "limitations": _get_model_limitations(task_type, training_regime),
        },
        
        # Technical details
        "technical": {
            "random_state": config.random_state,
            "feature_engineering": {
                "target_encoding": True,
                "historical_features": True,
                "density_features": False,  # Currently disabled
            },
        }
    }
    
    # Add fold performance range
    if 'fold_results' in cv_results:
        fold_df = pd.DataFrame(cv_results['fold_results'])
        if task_type == 'probability':
            score_col = 'roc_auc'
        else:
            score_col = 'mae'
        
        if score_col in fold_df.columns:
            scores = fold_df[score_col].dropna()
            if len(scores) > 0:
                model_card["performance"]["best_fold_score"] = scores.max() if task_type == 'probability' else scores.min()
                model_card["performance"]["worst_fold_score"] = scores.min() if task_type == 'probability' else scores.max()
    
    # Add additional information
    if additional_info:
        model_card["additional"] = additional_info
    
    return model_card


def _get_regime_explanation(training_regime: Optional[str]) -> str:
    """Get explanation of training regime."""
    if training_regime == "regime_a":
        return (
            "Regime A: Trained on time-binned data with actual elapsed time labels. "
            "Probability π(τ) is estimated directly from bin-level occurrence patterns."
        )
    elif training_regime == "regime_b":
        return (
            "Regime B: Trained on daily aggregated data. Daily trip counts are "
            "distributed across time bins using a learned shape function. "
            "Probability π(τ) = 1 - exp(-λ(τ) * W) where λ(τ) is the estimated rate."
        )
    else:
        return "No specific training regime (applies to price/weight models)."


def _get_metric_explanations(task_type: str) -> Dict[str, str]:
    """Get explanations of metrics for different task types."""
    if task_type == "probability":
        return {
            "roc_auc": "Area under ROC curve - measures discrimination ability",
            "pr_auc": "Area under Precision-Recall curve - handles class imbalance",
            "brier_score": "Mean squared difference between predicted probabilities and outcomes (lower is better)",
            "accuracy": "Fraction of correct binary predictions at default threshold"
        }
    else:
        return {
            "mae": "Mean Absolute Error - average absolute difference between predictions and actuals",
            "rmse": "Root Mean Squared Error - square root of average squared differences",
            "mape": "Mean Absolute Percentage Error - average percentage difference",
            "r2": "R-squared - proportion of variance explained by the model"
        }


def _get_model_limitations(task_type: str, training_regime: Optional[str]) -> List[str]:
    """Get list of model limitations and caveats."""
    limitations = [
        "Model trained on historical data and may not capture future trends",
        "Performance may degrade if operational patterns change significantly",
        "Feature importance may vary across different time periods"
    ]
    
    if task_type == "probability":
        limitations.extend([
            "Probability estimates are conditioned on historical load patterns",
            "Model assumes independence between different origin-destination pairs"
        ])
        
        if training_regime == "regime_b":
            limitations.extend([
                "Regime B uses simplified time distribution assumptions",
                "Actual time-of-day patterns may differ from learned shape function"
            ])
    
    if task_type in ["price", "weight"]:
        limitations.extend([
            "Estimates are averages and do not capture full distribution",
            "Model may not handle extreme values or outliers well"
        ])
    
    return limitations