"""
Weights & Biases integration for ATLAS ML.

Provides logging for experiments, metrics, hyperparameters, and model artifacts.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import wandb

logger = logging.getLogger(__name__)


class WandbLogger:
    """
    Handles Weights & Biases logging for training runs.
    
    Supports logging:
    - Metrics per fold and overall metrics
    - Hyperparameters and configuration
    - Model artifacts
    - Dataset information
    """
    
    def __init__(
        self,
        project: str = "atlas-ml",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        tags: Optional[list] = None,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True
    ):
        """
        Initialize wandb logger.
        
        Args:
            project: wandb project name
            entity: wandb entity (username or team)
            name: Run name (auto-generated if None)
            tags: List of tags for the run
            config: Configuration dictionary to log
            enabled: Enable/disable wandb logging
        """
        self.enabled = enabled
        self.run = None
        
        if not enabled:
            logger.info("wandb logging is disabled")
            return
        
        try:
            # Initialize wandb run
            self.run = wandb.init(
                project=project,
                entity=entity,
                name=name,
                tags=tags or [],
                config=config or {},
                reinit=True  # Allow multiple runs in same process
            )
            logger.info(f"wandb run initialized: {self.run.name} ({self.run.id})")
            
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}. Continuing without wandb logging.")
            self.enabled = False
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None, prefix: str = ""):
        """
        Log metrics to wandb.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Step number (e.g., fold number, epoch)
            prefix: Prefix for metric names (e.g., "probability/", "price/")
        """
        if not self.enabled or self.run is None:
            return
        
        try:
            # Add prefix to all metric keys
            prefixed_metrics = {
                f"{prefix}{key}" if prefix else key: value
                for key, value in metrics.items()
            }
            
            if step is not None:
                wandb.log(prefixed_metrics, step=step)
            else:
                wandb.log(prefixed_metrics)
                
        except Exception as e:
            logger.warning(f"Failed to log metrics to wandb: {e}")
    
    def log_fold_metrics(
        self,
        fold: int,
        task: str,
        metrics: Dict[str, float]
    ):
        """
        Log metrics for a specific fold and task.
        
        Args:
            fold: Fold number (0-indexed)
            task: Task name ("probability", "price", "weight")
            metrics: Dictionary of metrics (mae, rmse, r2, etc.)
        """
        if not self.enabled or self.run is None:
            return
        
        prefix = f"{task}/"
        
        # Log metrics WITHOUT fold in the name - use step parameter instead
        # This creates a single metric line with 7 data points (one per fold)
        fold_metrics = {
            key: value
            for key, value in metrics.items()
            if key not in ['fold', 'train_size', 'test_size']  # Exclude metadata
        }
        
        # Add fold number with prefix to avoid conflicts between tasks
        fold_metric_name = f"{prefix}fold"
        fold_metrics[fold_metric_name] = fold
        
        # Define X-axis as task-specific "fold" for these metrics
        if fold == 0:  # Only define once, on first fold
            for metric_name in fold_metrics.keys():
                if metric_name != fold_metric_name:
                    wandb.define_metric(f"{prefix}{metric_name}", step_metric=fold_metric_name)
        
        # Log metrics - wandb will use the fold_metric_name as X-axis
        # No need to pass step parameter, wandb will use the fold metric value
        self.log_metrics(fold_metrics, prefix="")
    
    def log_task_summary(
        self,
        task: str,
        cv_metrics: Dict[str, Any],
        overall_metrics: Dict[str, float],
        n_folds: int
    ):
        """
        Log summary metrics for a task (after all folds).
        
        Args:
            task: Task name ("probability", "price", "weight")
            cv_metrics: Cross-validation metrics (mean, std)
            overall_metrics: Overall metrics on full dataset
            n_folds: Number of folds used
        """
        if not self.enabled or self.run is None:
            return
        
        prefix = f"{task}/"
        
        summary_metrics = {
            # CV metrics (mean ± std)
            "cv/mae_mean": cv_metrics.get("mae_mean"),
            "cv/mae_std": cv_metrics.get("mae_std"),
            "cv/rmse_mean": cv_metrics.get("rmse_mean"),
            "cv/r2_mean": cv_metrics.get("r2_mean"),
            # Overall metrics
            "overall/mae": overall_metrics.get("mae"),
            "overall/rmse": overall_metrics.get("rmse"),
            "overall/r2": overall_metrics.get("r2"),
            # Config
            "n_folds": n_folds
        }
        
        # Update run.summary directly (appears in Runs table)
        # Do NOT use wandb.log() here to avoid breaking step-based metrics
        try:
            for key, value in summary_metrics.items():
                if value is not None:
                    self.run.summary[f"{prefix}{key}"] = value
            logger.debug(f"Updated run.summary with {len(summary_metrics)} metrics for task {task}")
        except Exception as e:
            logger.warning(f"Failed to update run summary: {e}")
    
    def log_hyperparameters(self, params: Dict[str, Any], prefix: str = ""):
        """
        Log hyperparameters.
        
        Args:
            params: Dictionary of hyperparameters
            prefix: Prefix for parameter names
        """
        if not self.enabled or self.run is None:
            return
        
        try:
            # Update config with hyperparameters
            prefixed_params = {
                f"{prefix}{key}" if prefix else key: value
                for key, value in params.items()
            }
            wandb.config.update(prefixed_params)
            
        except Exception as e:
            logger.warning(f"Failed to log hyperparameters to wandb: {e}")
    
    def log_dataset_info(self, info: Dict[str, Any]):
        """
        Log dataset information.
        
        Args:
            info: Dictionary with dataset info (n_samples, date_range, features, etc.)
        """
        if not self.enabled or self.run is None:
            return
        
        try:
            wandb.config.update({"dataset": info})
            
        except Exception as e:
            logger.warning(f"Failed to log dataset info to wandb: {e}")
    
    def log_artifact(
        self,
        artifact_path: Union[str, Path],
        artifact_name: str,
        artifact_type: str = "model",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log a model or file artifact to wandb.
        
        Args:
            artifact_path: Path to artifact file or directory
            artifact_name: Name for the artifact
            artifact_type: Type of artifact ("model", "dataset", etc.)
            metadata: Optional metadata dictionary
        """
        if not self.enabled or self.run is None:
            return
        
        try:
            artifact_path = Path(artifact_path)
            
            # Create artifact
            artifact = wandb.Artifact(
                name=artifact_name,
                type=artifact_type,
                metadata=metadata or {}
            )
            
            # Add file or directory
            if artifact_path.is_dir():
                artifact.add_dir(str(artifact_path))
            else:
                artifact.add_file(str(artifact_path))
            
            # Log artifact
            self.run.log_artifact(artifact)
            logger.info(f"Logged artifact: {artifact_name} ({artifact_type})")
            
        except Exception as e:
            logger.warning(f"Failed to log artifact to wandb: {e}")
    
    def log_model_bundle(
        self,
        model_path: Path,
        task: str,
        metrics: Dict[str, Any]
    ):
        """
        Log a complete model bundle (model + encoders + metadata).
        
        Args:
            model_path: Path to model directory
            task: Task name ("probability", "price", "weight")
            metrics: Model metrics to include in metadata
        """
        if not self.enabled or self.run is None:
            return
        
        # Calculate model size in MB
        model_file = model_path / "model.joblib"
        model_size_mb = 0.0
        if model_file.exists():
            model_size_mb = model_file.stat().st_size / (1024 * 1024)  # Convert bytes to MB
        
        # Log model size as a metric
        self.log_metrics({f"{task}/model_size_mb": model_size_mb})
        
        artifact_name = f"{task}_model_{self.run.id}"
        metadata = {
            "task": task,
            "metrics": metrics,
            "model_path": str(model_path),
            "model_size_mb": round(model_size_mb, 2)
        }
        
        self.log_artifact(
            artifact_path=model_path,
            artifact_name=artifact_name,
            artifact_type="model",
            metadata=metadata
        )
    
    def finish(self):
        """Finish the wandb run."""
        if not self.enabled or self.run is None:
            return
        
        try:
            wandb.finish()
            logger.info("wandb run finished")
        except Exception as e:
            logger.warning(f"Error finishing wandb run: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()
