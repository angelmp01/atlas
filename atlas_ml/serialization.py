"""
Serialization module for ATLAS ML package.

Provides safe model and encoder persistence with versioning and metadata tracking.
Implements ModelBundle for packaging trained models with their artifacts.
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import pandas as pd

from .config import Config

logger = logging.getLogger(__name__)


class ModelBundle:
    """
    Container for trained models, encoders, and metadata.
    
    A ModelBundle packages all artifacts needed for inference:
    - Trained model objects
    - Feature encoders and preprocessors  
    - Model metadata and performance metrics
    - Training configuration
    """
    
    def __init__(
        self,
        model_path: Path,
        model_type: str,
        task_type: str,
        features: List[str],
        version: str = "1.0.0"
    ):
        """
        Initialize ModelBundle.
        
        Args:
            model_path: Base path for model artifacts
            model_type: Type of model (e.g., 'xgboost', 'randomforest')
            task_type: Task type ('probability', 'price', 'weight')
            features: List of feature names
            version: Model version string
        """
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.task_type = task_type
        self.features = features
        self.version = version
        
        # Derived paths
        self.model_file = self.model_path / "model.joblib"
        self.encoders_file = self.model_path / "encoders.joblib"
        self.metadata_file = self.model_path / "model_card.json"
        self.scaler_file = self.model_path / "scaler.joblib"
        
        # Loaded objects (lazy loading)
        self._model = None
        self._encoders = None
        self._metadata = None
        self._scaler = None
    
    @property
    def model(self):
        """Lazy load trained model."""
        if self._model is None and self.model_file.exists():
            self._model = joblib.load(self.model_file)
            logger.debug(f"Loaded model from {self.model_file}")
        return self._model
    
    @property
    def encoders(self) -> Dict[str, Any]:
        """Lazy load feature encoders."""
        if self._encoders is None and self.encoders_file.exists():
            self._encoders = joblib.load(self.encoders_file)
            logger.debug(f"Loaded encoders from {self.encoders_file}")
        return self._encoders or {}
    
    @property
    def scaler(self):
        """Lazy load feature scaler."""
        if self._scaler is None and self.scaler_file.exists():
            self._scaler = joblib.load(self.scaler_file)
            logger.debug(f"Loaded scaler from {self.scaler_file}")
        return self._scaler
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Lazy load model metadata."""
        if self._metadata is None and self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self._metadata = json.load(f)
            logger.debug(f"Loaded metadata from {self.metadata_file}")
        return self._metadata or {}
    
    def save_model(self, model: Any) -> None:
        """Save trained model."""
        self.model_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.model_file)
        self._model = model
        logger.info(f"Saved model to {self.model_file}")
    
    def save_encoders(self, encoders: Dict[str, Any]) -> None:
        """Save feature encoders."""
        self.model_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(encoders, self.encoders_file)
        self._encoders = encoders
        logger.info(f"Saved encoders to {self.encoders_file}")
    
    def save_scaler(self, scaler: Any) -> None:
        """Save feature scaler."""
        self.model_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, self.scaler_file)
        self._scaler = scaler
        logger.info(f"Saved scaler to {self.scaler_file}")
    
    def save_metadata(self, metadata: Dict[str, Any]) -> None:
        """Save model metadata."""
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Add standard metadata
        full_metadata = {
            "model_type": self.model_type,
            "task_type": self.task_type,
            "features": self.features,
            "version": self.version,
            "created_at": datetime.now().isoformat(),
            "bundle_hash": self._compute_bundle_hash(),
            **metadata
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(full_metadata, f, indent=2, default=str)
        
        self._metadata = full_metadata
        logger.info(f"Saved metadata to {self.metadata_file}")
    
    def _compute_bundle_hash(self) -> str:
        """Compute hash of bundle for integrity checking."""
        hash_obj = hashlib.sha256()
        
        # Hash model type, task, features
        hash_obj.update(f"{self.model_type}:{self.task_type}".encode())
        hash_obj.update(":".join(sorted(self.features)).encode())
        
        return hash_obj.hexdigest()[:16]
    
    def is_valid(self) -> bool:
        """Check if bundle has all required components."""
        required_files = [self.model_file, self.metadata_file]
        return all(f.exists() for f in required_files)
    
    def get_info(self) -> Dict[str, Any]:
        """Get bundle information summary."""
        return {
            "model_type": self.model_type,
            "task_type": self.task_type,
            "version": self.version,
            "num_features": len(self.features),
            "valid": self.is_valid(),
            "model_path": str(self.model_path),
            "created_at": self.metadata.get("created_at"),
            "cv_score": self.metadata.get("cv_metrics", {}).get("mean_score")
        }


def save_model_bundle(
    model: Any,
    model_type: str,
    task_type: str,
    features: List[str],
    model_path: Union[str, Path],
    encoders: Optional[Dict[str, Any]] = None,
    scaler: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
    config: Optional[Config] = None
) -> ModelBundle:
    """
    Save a complete model bundle with all artifacts.
    
    Args:
        model: Trained model object
        model_type: Type of model
        task_type: Task type ('probability', 'price', 'weight')
        features: List of feature names
        model_path: Path to save bundle
        encoders: Feature encoders dictionary
        scaler: Feature scaler object
        metadata: Additional metadata
        config: Configuration object
        
    Returns:
        ModelBundle instance
    """
    # Create bundle
    bundle = ModelBundle(
        model_path=model_path,
        model_type=model_type,
        task_type=task_type,
        features=features
    )
    
    # Save components
    bundle.save_model(model)
    
    if encoders:
        bundle.save_encoders(encoders)
    
    if scaler:
        bundle.save_scaler(scaler)
    
    # Prepare metadata
    model_metadata = metadata or {}
    if config:
        model_metadata["config"] = {
            "model_params": getattr(config.model, f"{model_type}_params", {}),
            "feature_params": config.features.dict(),
            "training_params": config.training.dict()
        }
    
    bundle.save_metadata(model_metadata)
    
    logger.info(f"Saved complete model bundle to {model_path}")
    return bundle


def load_bundle(model_path: Union[str, Path]) -> ModelBundle:
    """
    Load a model bundle from disk.
    
    Args:
        model_path: Path to model bundle directory
        
    Returns:
        Loaded ModelBundle
        
    Raises:
        FileNotFoundError: If bundle path doesn't exist
        ValueError: If bundle is invalid or corrupted
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model bundle not found at {model_path}")
    
    # Load metadata first to get bundle info
    metadata_file = model_path / "model_card.json"
    if not metadata_file.exists():
        raise ValueError(f"Model metadata not found at {metadata_file}")
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Create bundle from metadata
    bundle = ModelBundle(
        model_path=model_path,
        model_type=metadata.get("model_type", "unknown"),
        task_type=metadata.get("task_type", "unknown"),
        features=metadata.get("features", []),
        version=metadata.get("version", "1.0.0")
    )
    
    # Validate bundle
    if not bundle.is_valid():
        raise ValueError(f"Invalid model bundle at {model_path}")
    
    logger.info(f"Loaded model bundle from {model_path}")
    return bundle


def list_model_bundles(models_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    List all model bundles in a directory.
    
    Args:
        models_dir: Directory containing model bundles
        
    Returns:
        List of bundle information dictionaries
    """
    models_dir = Path(models_dir)
    
    if not models_dir.exists():
        return []
    
    bundles = []
    for bundle_dir in models_dir.iterdir():
        if bundle_dir.is_dir():
            try:
                bundle = load_bundle(bundle_dir)
                bundles.append(bundle.get_info())
            except (FileNotFoundError, ValueError) as e:
                logger.warning(f"Failed to load bundle from {bundle_dir}: {e}")
    
    return bundles


def cleanup_old_bundles(
    models_dir: Union[str, Path],
    keep_latest: int = 5,
    task_type: Optional[str] = None
) -> int:
    """
    Clean up old model bundles, keeping only the latest N versions.
    
    Args:
        models_dir: Directory containing model bundles
        keep_latest: Number of latest bundles to keep per task type
        task_type: Optional task type filter
        
    Returns:
        Number of bundles deleted
    """
    models_dir = Path(models_dir)
    
    if not models_dir.exists():
        return 0
    
    # Get all bundles with creation times
    bundles = []
    for bundle_dir in models_dir.iterdir():
        if bundle_dir.is_dir():
            try:
                bundle = load_bundle(bundle_dir)
                info = bundle.get_info()
                created_at = info.get("created_at")
                if created_at and (task_type is None or info["task_type"] == task_type):
                    bundles.append((bundle_dir, created_at, info["task_type"]))
            except (FileNotFoundError, ValueError):
                continue
    
    if not bundles:
        return 0
    
    # Group by task type and sort by creation time
    from itertools import groupby
    deleted_count = 0
    
    bundles.sort(key=lambda x: x[2])  # Sort by task_type
    
    for task, group in groupby(bundles, key=lambda x: x[2]):
        task_bundles = list(group)
        task_bundles.sort(key=lambda x: x[1], reverse=True)  # Sort by created_at desc
        
        # Keep only the latest N bundles for this task
        to_delete = task_bundles[keep_latest:]
        
        for bundle_dir, _, _ in to_delete:
            try:
                import shutil
                shutil.rmtree(bundle_dir)
                deleted_count += 1
                logger.info(f"Deleted old bundle: {bundle_dir}")
            except Exception as e:
                logger.warning(f"Failed to delete bundle {bundle_dir}: {e}")
    
    return deleted_count


class ModelVersionManager:
    """Manages model versioning and deployment."""
    
    def __init__(self, models_dir: Union[str, Path]):
        """
        Initialize version manager.
        
        Args:
            models_dir: Directory for storing model bundles
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def get_bundle_path(
        self,
        task_type: str,
        version: Optional[str] = None
    ) -> Path:
        """
        Get path for a model bundle.
        
        Args:
            task_type: Task type ('probability', 'price', 'weight')
            version: Model version (if None, use timestamp)
            
        Returns:
            Path for model bundle
        """
        if version is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version = f"v{timestamp}"
        
        bundle_name = f"{task_type}_{version}"
        return self.models_dir / bundle_name
    
    def get_latest_bundle(
        self,
        task_type: str
    ) -> Optional[ModelBundle]:
        """
        Get the latest model bundle for a task type.
        
        Args:
            task_type: Task type to search for
            
        Returns:
            Latest ModelBundle or None if not found
        """
        bundles = list_model_bundles(self.models_dir)
        
        # Filter by task type
        filtered_bundles = [
            b for b in bundles
            if b["task_type"] == task_type and b["valid"]
        ]
        
        if not filtered_bundles:
            return None
        
        # Sort by creation time and get latest
        latest = max(filtered_bundles, key=lambda x: x["created_at"] or "")
        
        return load_bundle(latest["model_path"])