"""
Configuration module for ATLAS ML package.

Provides Pydantic dataclasses for settings, database connections, model paths,
and default parameters used throughout the ML pipeline.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


class DatabaseConfig(BaseModel):
    """Database connection configuration."""
    
    dsn: str = Field(
        default_factory=lambda: os.getenv(
            "PG_DSN", 
            "postgresql://appuser:pgai1234@192.168.50.10:25432/gisdb"
        ),
        description="PostgreSQL connection string"
    )
    echo: bool = Field(default=False, description="Enable SQLAlchemy logging")


class ModelConfig(BaseModel):
    """Configuration for machine learning models."""
    
    # Model types
    probability_model_type: str = Field(
        default="xgboost",
        description="Model type for probability estimation: 'xgboost' or 'randomforest'"
    )
    price_model_type: str = Field(
        default="xgboost", 
        description="Model type for price regression: 'xgboost' or 'randomforest'"
    )
    weight_model_type: str = Field(
        default="xgboost",
        description="Model type for weight regression: 'xgboost' or 'randomforest'"
    )
    
    # Model hyperparameters
    xgb_params: Dict = Field(
        default_factory=lambda: {
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "objective": "binary:logistic",  # Will be overridden for regressors
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "gpu_hist",  # Use "gpu_hist" for GPU acceleration / "hist" for CPU
            "device": "cuda",     # Uncomment to enable GPU (requires CUDA + xgboost[cuda])
        },
        description="XGBoost hyperparameters"
    )
    
    rf_params: Dict = Field(
        default_factory=lambda: {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "n_jobs": -1,
        },
        description="Random Forest hyperparameters"
    )


class FeatureConfig(BaseModel):
    """Configuration for feature engineering."""
    
    # Time binning
    tau_bin_minutes: int = Field(
        default=10,
        description="Time bin size in minutes for elapsed time discretization"
    )
    max_tau_hours: int = Field(
        default=24,
        description="Maximum elapsed time to consider in hours"
    )
    
    # Rolling window sizes for historical aggregates (in days)
    rolling_windows: List[int] = Field(
        default=[7, 14, 28],
        description="Rolling window sizes for historical features"
    )
    
    # Distance thresholds for density features (in km)
    density_radii: List[float] = Field(
        default=[25.0, 50.0],
        description="Distance radii for destination density features"
    )
    
    # Target encoding parameters
    min_samples_leaf: int = Field(
        default=10,
        description="Minimum samples for target encoding categories"
    )
    smoothing: float = Field(
        default=1.0,
        description="Smoothing factor for target encoding"
    )


class TrainingConfig(BaseModel):
    """Configuration for training procedures."""
    
    # Probability training regime
    training_regime: str = Field(
        default="regime_b",
        description="Training regime: 'regime_a' (bin-level) or 'regime_b' (daily counts)"
    )
    
    # Temporal cross-validation
    cv_months_train: int = Field(
        default=6,
        description="Number of months to use for training in temporal CV"
    )
    cv_months_test: int = Field(
        default=1,
        description="Number of months to use for testing in temporal CV"
    )
    
    # Date filtering
    start_date: Optional[str] = Field(
        default=None,
        description="Start date for training data (YYYY-MM-DD)"
    )
    end_date: Optional[str] = Field(
        default=None,
        description="End date for training data (YYYY-MM-DD)"
    )
    
    # Probability specific
    wait_tolerance_minutes: int = Field(
        default=30,
        description="Additional wait time W for probability calculation"
    )


class PathConfig(BaseModel):
    """Configuration for file paths and directories."""
    
    # Base directories
    data_dir: Path = Field(
        default_factory=lambda: Path("data"),
        description="Base data directory"
    )
    models_dir: Path = Field(
        default_factory=lambda: Path("models"),
        description="Directory to save trained models"
    )
    logs_dir: Path = Field(
        default_factory=lambda: Path("logs"),
        description="Directory for log files"
    )
    
    # Processed data paths
    processed_dir: Path = Field(
        default_factory=lambda: Path("data/processed"),
        description="Directory for processed data files"
    )
    
    def ensure_dirs(self) -> None:
        """Create directories if they don't exist."""
        for path in [self.data_dir, self.models_dir, self.logs_dir, self.processed_dir]:
            path.mkdir(parents=True, exist_ok=True)


class Config(BaseModel):
    """Main configuration class for ATLAS ML package."""
    
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    paths: PathConfig = Field(default_factory=PathConfig)
    
    # Global settings
    random_state: int = Field(default=42, description="Global random state")
    verbose: bool = Field(default=True, description="Enable verbose logging")
    
    class Config:
        """Pydantic config."""
        extra = "forbid"
        validate_assignment = True
    
    def model_post_init(self, __context) -> None:
        """Post-initialization setup."""
        self.paths.ensure_dirs()


@dataclass
class HolidayCalendar:
    """Simple holiday calendar for feature engineering."""
    
    holidays: List[str]  # List of dates in "YYYY-MM-DD" format
    
    def is_holiday(self, date_str: str) -> bool:
        """Check if a date is a holiday."""
        return date_str in self.holidays


# Default Spanish holidays for 2024 (extend as needed)
DEFAULT_HOLIDAYS = HolidayCalendar(
    holidays=[
        "2024-01-01",  # New Year
        "2024-01-06",  # Epiphany
        "2024-03-29",  # Good Friday
        "2024-05-01",  # Labor Day
        "2024-08-15",  # Assumption
        "2024-10-12",  # National Day
        "2024-11-01",  # All Saints
        "2024-12-06",  # Constitution Day
        "2024-12-08",  # Immaculate Conception
        "2024-12-25",  # Christmas
    ]
)