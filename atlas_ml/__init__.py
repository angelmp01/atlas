"""
ATLAS ML Package

AI-based Transportation Logistics Assistant System - Machine Learning Module

This package provides machine learning capabilities for estimating:
1. π_{i→d}(τ): probability of available loads from zone i to destination d at elapsed time τ
2. E[price_{i→d}] and E[weight_{i→d}]: expected price and weight for loads

The package supports both training and inference modes for production use.

Key Features:
- Two training regimes for probability estimation (Regime A: bin-level, Regime B: daily counts)
- Temporal cross-validation for robust model evaluation
- Feature engineering with historical aggregates and target encoding
- Model serialization with versioning and metadata tracking
- Clean separation of training and inference interfaces

Example Usage:

Training:
    from atlas_ml import Config, train_probability, train_price, train_weight
    
    config = Config()
    
    # Train models
    prob_bundle = train_probability(config)
    price_bundle = train_price(config)
    weight_bundle = train_weight(config)

Inference:
    from atlas_ml import CandidateInput, predict_all, load_bundle
    
    # Load trained models
    prob_bundle = load_bundle("models/probability_regime_b_v20241022")
    
    # Define candidates
    candidates = [
        CandidateInput(
            i_location_id=1,
            d_location_id=2,
            truck_type='normal',
            tipo_mercancia='normal',
            day_of_week=1,
            week_of_year=10,
            holiday_flag=0,
            tau_minutes=120
        )
    ]
    
    # Get predictions
    results = predict_all(candidates, prob_bundle)
    for result in results:
        print(f"Origin {result.i_location_id} → Destination {result.d_location_id}")
        print(f"  Probability: {result.pi:.3f}")
        print(f"  Expected price: €{result.exp_price:.2f}")
        print(f"  Expected weight: {result.exp_weight:.1f} kg")
"""

from .config import Config, ModelConfig, DatabaseConfig, FeatureConfig, TrainingConfig, PathConfig
from .probability import (
    CandidateInput,
    CandidateOutput,
    predict_probability,
    predict_all,
    train_probability,
    ProbabilityEstimator,
    ShapeFunctionLearner
)
from .regressors import (
    predict_price, 
    predict_weight, 
    train_price, 
    train_weight,
    PriceEstimator,
    WeightEstimator,
    create_synthetic_price_data,
    create_synthetic_weight_data
)
from .serialization import (
    ModelBundle, 
    load_bundle, 
    save_model_bundle,
    ModelVersionManager,
    list_model_bundles,
    cleanup_old_bundles
)
from .featurization import (
    build_probability_dataset,
    build_price_dataset,
    build_weight_dataset,
    get_feature_names,
    ProbabilityFeatureBuilder,
    RegressionFeatureBuilder
)
from .evaluation import (
    TemporalSplitter,
    ModelEvaluator,
    CrossValidator,
    create_model_card
)
from .io import (
    DatabaseManager,
    DatasetBuilder,
    create_database_manager,
    create_dataset_builder
)
from .utils import (
    to_tau_bin,
    set_random_seed,
    create_time_features,
    haversine_distance,
    target_encode,
    setup_logging
)

__version__ = "0.1.0"
__author__ = "ATLAS ML Team"
__description__ = "Machine Learning module for AI-based Transportation Logistics Assistant System"

# Main public API
__all__ = [
    # Configuration
    "Config",
    "ModelConfig", 
    "DatabaseConfig",
    "FeatureConfig",
    "TrainingConfig",
    "PathConfig",
    
    # Core data structures
    "CandidateInput",
    "CandidateOutput",
    "ModelBundle",
    
    # Training functions (CLI-friendly)
    "train_probability",
    "train_price", 
    "train_weight",
    
    # Inference functions
    "predict_probability",
    "predict_price",
    "predict_weight", 
    "predict_all",
    
    # Model management
    "load_bundle",
    "save_model_bundle",
    "ModelVersionManager",
    "list_model_bundles",
    "cleanup_old_bundles",
    
    # Dataset building (for custom training)
    "build_probability_dataset",
    "build_price_dataset", 
    "build_weight_dataset",
    
    # Utilities
    "to_tau_bin",
    "set_random_seed",
    "setup_logging",
    
    # Advanced/Expert API
    "ProbabilityEstimator",
    "PriceEstimator", 
    "WeightEstimator",
    "ShapeFunctionLearner",
    "ProbabilityFeatureBuilder",
    "RegressionFeatureBuilder", 
    "TemporalSplitter",
    "ModelEvaluator",
    "CrossValidator",
    "DatabaseManager",
    "DatasetBuilder",
    
    # Testing utilities
    "create_synthetic_price_data",
    "create_synthetic_weight_data",
]

# Package metadata
__package_info__ = {
    "name": "atlas_ml",
    "version": __version__,
    "description": __description__,
    "author": __author__,
    "python_requires": ">=3.11",
    "dependencies": [
        "pandas>=2.0.0",
        "numpy>=1.24.0", 
        "scikit-learn>=1.3.0",
        "xgboost>=1.7.0",
        "sqlalchemy>=2.0.0",
        "geoalchemy2>=0.13.0",
        "pyproj>=3.5.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "orjson>=3.8.0",
        "joblib>=1.3.0"
    ]
}

def get_package_info():
    """Get package information dictionary."""
    return __package_info__.copy()

def print_package_info():
    """Print package information."""
    info = get_package_info()
    print(f"{info['name']} v{info['version']}")
    print(f"{info['description']}")
    print(f"Author: {info['author']}")
    print(f"Python: {info['python_requires']}")
    print("Dependencies:")
    for dep in info['dependencies']:
        print(f"  - {dep}")

# Version check and compatibility
def check_dependencies():
    """
    Check if required dependencies are available.
    
    Returns:
        Dict with dependency status
    """
    status = {}
    
    try:
        import pandas
        status['pandas'] = pandas.__version__
    except ImportError:
        status['pandas'] = 'NOT INSTALLED'
    
    try:
        import numpy
        status['numpy'] = numpy.__version__
    except ImportError:
        status['numpy'] = 'NOT INSTALLED'
    
    try:
        import sklearn
        status['scikit-learn'] = sklearn.__version__
    except ImportError:
        status['scikit-learn'] = 'NOT INSTALLED'
    
    try:
        import xgboost
        status['xgboost'] = xgboost.__version__
    except ImportError:
        status['xgboost'] = 'NOT INSTALLED (OPTIONAL)'
    
    try:
        import sqlalchemy
        status['sqlalchemy'] = sqlalchemy.__version__
    except ImportError:
        status['sqlalchemy'] = 'NOT INSTALLED'
    
    try:
        import geoalchemy2
        status['geoalchemy2'] = geoalchemy2.__version__
    except ImportError:
        status['geoalchemy2'] = 'NOT INSTALLED'
    
    try:
        import pydantic
        status['pydantic'] = pydantic.__version__
    except ImportError:
        status['pydantic'] = 'NOT INSTALLED'
    
    return status