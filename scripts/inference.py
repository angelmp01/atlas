"""
Interactive inference script for testing trained ATLAS models.

Usage:
    python scripts/inference.py --model probability --origin 08001 --destination 08100
    python scripts/inference.py --model price --origin 08001 --destination 08100 --date 2024-06-15
    python scripts/inference.py --model weight --origin 08001 --destination 08100
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add parent directory to path to import atlas_ml
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from atlas_ml.serialization import load_bundle
from atlas_ml.config import Config

logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG to see feature values
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_latest_model(model_type: str, models_dir: Path = Path("models")) -> Path:
    """
    Find the latest trained model of given type.
    
    Args:
        model_type: Type of model ('probability', 'price', 'weight')
        models_dir: Directory containing model bundles
        
    Returns:
        Path to latest model bundle
    """
    pattern = f"{model_type}_*"
    model_dirs = sorted(models_dir.glob(pattern), reverse=True)
    
    if not model_dirs:
        raise FileNotFoundError(f"No trained models found for type: {model_type}")
    
    return model_dirs[0]


def prepare_input_features(
    origin_id: int,
    destination_id: int,
    date: Optional[str] = None,
    truck_type: str = "normal",
    tipo_mercancia: str = "normal",
    bundle: Any = None,
    use_db: bool = True
) -> pd.DataFrame:
    """
    Prepare input features for model inference.
    
    Args:
        origin_id: Origin location ID
        destination_id: Destination location ID
        date: Date string in YYYY-MM-DD format (defaults to today)
        truck_type: Type of truck ('normal', 'refrigerado')
        tipo_mercancia: Type of merchandise ('normal', 'refrigerada')
        bundle: Loaded model bundle with encoders
        use_db: If True, load historical stats from database
        
    Returns:
        DataFrame with all required features
    """
    # Parse date
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    date_obj = pd.to_datetime(date)
    
    # Create base input
    input_data = pd.DataFrame({
        'origin_id': [origin_id],
        'destination_id': [destination_id],
        'date': [date_obj],
        'truck_type': [truck_type],
        'tipo_mercancia': [tipo_mercancia],
    })
    
    # Add time features
    input_data['day_of_week'] = date_obj.dayofweek
    input_data['month'] = date_obj.month
    input_data['day'] = date_obj.day
    input_data['week_of_year'] = date_obj.isocalendar().week
    input_data['is_weekend'] = int(date_obj.dayofweek >= 5)
    input_data['quarter'] = date_obj.quarter
    
    # Load historical data from database if requested
    if use_db:
        try:
            from atlas_ml.io import create_database_manager
            config = Config()
            db = create_database_manager(config)
            
            # Get historical stats for this OD pair
            from sqlalchemy import text
            query = text("""
                SELECT 
                    AVG(precio) as precio_mean_daily,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY precio) as precio_median_daily,
                    STDDEV(precio) as precio_std_daily,
                    AVG(peso) as peso_mean_daily,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY peso) as peso_median_daily,
                    STDDEV(peso) as peso_std_daily,
                    AVG(volumen) as volumen_mean_daily,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY volumen) as volumen_median_daily,
                    STDDEV(volumen) as volumen_std_daily
                FROM app.sodd_loads
                WHERE origin_id = :origin_id
                  AND destination_id = :destination_id
                  AND date < :ref_date  -- Use only historical data
            """)
            
            with db.engine.connect() as conn:
                result = conn.execute(query, {
                    'origin_id': str(origin_id).zfill(5),  # Convert to string with leading zeros
                    'destination_id': str(destination_id).zfill(5),
                    'ref_date': date
                }).fetchone()
            
            if result and result[0] is not None:
                input_data['precio_mean_daily'] = float(result[0] or 0)
                input_data['precio_median_daily'] = float(result[1] or 0)
                input_data['precio_std_daily'] = float(result[2] or 0)
                input_data['peso_mean_daily'] = float(result[3] or 0)
                input_data['peso_median_daily'] = float(result[4] or 0)
                input_data['peso_std_daily'] = float(result[5] or 0)
                input_data['volumen_mean_daily'] = float(result[6] or 0)
                input_data['volumen_median_daily'] = float(result[7] or 0)
                input_data['volumen_std_daily'] = float(result[8] or 0)
                logger.info(f"Loaded historical stats from database for OD pair {origin_id}→{destination_id}")
            else:
                logger.warning(f"No historical data found for OD pair {origin_id}→{destination_id}, using zeros")
                for col in ['precio_mean_daily', 'precio_median_daily', 'precio_std_daily',
                           'peso_mean_daily', 'peso_median_daily', 'peso_std_daily',
                           'volumen_mean_daily', 'volumen_median_daily', 'volumen_std_daily']:
                    input_data[col] = 0.0
            
        except Exception as e:
            logger.warning(f"Failed to load historical data from DB: {e}, using placeholder zeros")
            for col in ['precio_mean_daily', 'precio_median_daily', 'precio_std_daily',
                       'peso_mean_daily', 'peso_median_daily', 'peso_std_daily',
                       'volumen_mean_daily', 'volumen_median_daily', 'volumen_std_daily']:
                input_data[col] = 0.0
    else:
        # Use placeholder zeros
        for col in ['precio_mean_daily', 'precio_median_daily', 'precio_std_daily',
                   'peso_mean_daily', 'peso_median_daily', 'peso_std_daily',
                   'volumen_mean_daily', 'volumen_median_daily', 'volumen_std_daily']:
            input_data[col] = 0.0
    
    # Load geographic features (coordinates and distance)
    if use_db:
        try:
            from atlas_ml.io import create_database_manager
            config = Config()
            db = create_database_manager(config)
            
            from sqlalchemy import text
            geo_query = text("""
                SELECT 
                    o.location_id as origin_id,
                    d.location_id as destination_id,
                    ST_X(ST_Transform(o.geom, 4326)) as origin_lon,
                    ST_Y(ST_Transform(o.geom, 4326)) as origin_lat,
                    ST_X(ST_Transform(d.geom, 4326)) as destination_lon,
                    ST_Y(ST_Transform(d.geom, 4326)) as destination_lat,
                    ST_Distance(
                        ST_Transform(o.geom, 3857),
                        ST_Transform(d.geom, 3857)
                    ) / 1000.0 as od_length_km
                FROM app.sodd_locations o
                CROSS JOIN app.sodd_locations d
                WHERE o.location_id = :origin_id
                  AND d.location_id = :destination_id
            """)
            
            with db.engine.connect() as conn:
                geo_result = conn.execute(geo_query, {
                    'origin_id': str(origin_id).zfill(5),
                    'destination_id': str(destination_id).zfill(5)
                }).fetchone()
            
            if geo_result:
                input_data['origin_lon'] = float(geo_result[2])
                input_data['origin_lat'] = float(geo_result[3])
                input_data['destination_lon'] = float(geo_result[4])
                input_data['destination_lat'] = float(geo_result[5])
                input_data['od_length_km'] = float(geo_result[6])
                input_data['log_od_length_km'] = np.log1p(input_data['od_length_km'])
                logger.info(f"Loaded geographic data: distance = {input_data['od_length_km'].iloc[0]:.2f} km")
            else:
                logger.warning(f"No geographic data found for OD pair, using zeros")
                for col in ['origin_lon', 'origin_lat', 'destination_lon', 'destination_lat', 'od_length_km', 'log_od_length_km']:
                    input_data[col] = 0.0
        except Exception as e:
            logger.warning(f"Failed to load geographic data: {e}")
            for col in ['origin_lon', 'origin_lat', 'destination_lon', 'destination_lat', 'od_length_km', 'log_od_length_km']:
                input_data[col] = 0.0
    else:
        for col in ['origin_lon', 'origin_lat', 'destination_lon', 'destination_lat', 'od_length_km', 'log_od_length_km']:
            input_data[col] = 0.0
    
    # Holiday flag (simplified - you could load a holiday calendar)
    input_data['holiday_flag'] = 0
    
    # Encode categorical features
    encoders = bundle.encoders.get('feature_builder_encoders', {})
    
    for col in ['truck_type', 'tipo_mercancia']:
        if col in encoders:
            encoder = encoders[col]
            try:
                input_data[f'{col}_encoded'] = encoder.transform(input_data[col])
            except ValueError:
                # Unknown category
                logger.warning(f"Unknown value for {col}: {input_data[col].iloc[0]}, using default")
                input_data[f'{col}_encoded'] = -1
    
    logger.info(f"Prepared input with {len(input_data.columns)} features")
    return input_data


def run_inference(
    model_type: str,
    origin_id: int,
    destination_id: int,
    date: Optional[str] = None,
    truck_type: str = "normal",
    tipo_mercancia: str = "normal",
    model_path: Optional[str] = None,
    use_db: bool = True,
    elapsed_time: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run inference with a trained model.
    
    Args:
        model_type: Type of model to use ('probability', 'price', 'weight')
        origin_id: Origin location ID
        destination_id: Destination location ID
        date: Date for prediction (YYYY-MM-DD)
        truck_type: Type of truck
        tipo_mercancia: Type of merchandise
        model_path: Optional path to specific model bundle
        
    Returns:
        Dictionary with prediction results and metadata
    """
    # Load model bundle
    if model_path:
        bundle_path = Path(model_path)
    else:
        bundle_path = find_latest_model(model_type)
    
    logger.info(f"Loading model from: {bundle_path}")
    bundle = load_bundle(str(bundle_path))
    
    # Display model info
    logger.info(f"Model type: {bundle.metadata.get('model_type')}")
    logger.info(f"Task type: {bundle.metadata.get('task_type')}")
    logger.info(f"Training date: {bundle.metadata.get('training_date')}")
    logger.info(f"Features: {len(bundle.features)}")
    
    # Prepare input features
    input_data = prepare_input_features(
        origin_id=origin_id,
        destination_id=destination_id,
        date=date,
        truck_type=truck_type,
        tipo_mercancia=tipo_mercancia,
        bundle=bundle,
        use_db=use_db
    )
    
    # Get required features from model
    required_features = bundle.features
    
    # Extract metadata values needed for shape function (before filtering/scaling)
    od_length_km = float(input_data['od_length_km'].values[0]) if 'od_length_km' in input_data.columns else 0.0
    day_of_week = int(input_data['day_of_week'].values[0]) if 'day_of_week' in input_data.columns else 0
    holiday_flag = int(input_data['holiday_flag'].values[0]) if 'holiday_flag' in input_data.columns else 0
    
    # Filter to only required features (fill missing with 0)
    X = pd.DataFrame()
    for feat in required_features:
        if feat in input_data.columns:
            X[feat] = input_data[feat]
        else:
            logger.warning(f"Missing feature: {feat}, using 0.0")
            X[feat] = 0.0
    
    # Apply scaling if model has scaler
    if bundle.scaler:
        logger.info("Applying feature scaling")
        X_scaled = bundle.scaler.transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Make prediction
    logger.info("Running prediction...")
    logger.debug(f"Input features shape: {X.shape}")
    logger.debug(f"Feature values (first row): {X.iloc[0].to_dict()}")
    prediction = bundle.model.predict(X)
    
    # Post-process based on model type
    result = {
        'origin_id': origin_id,
        'destination_id': destination_id,
        'date': date or datetime.now().strftime("%Y-%m-%d"),
        'truck_type': truck_type,
        'tipo_mercancia': tipo_mercancia,
        'model_type': model_type,
        'model_path': str(bundle_path),
        'raw_prediction': float(prediction[0])
    }
    
    if model_type == 'probability':
        # Regime B: model predicts daily trip count, convert to probability
        # Using UNIFORM DISTRIBUTION (no shape function - no real temporal data available)
        n_trips_daily = float(prediction[0])
        result['n_trips_daily'] = n_trips_daily
        
        # Simple uniform distribution: divide daily trips by minutes in a day
        # Assumes trips are evenly distributed (honest approach without temporal data)
        lambda_per_minute = n_trips_daily / (24 * 60)  # trips per minute
        
        # Probability at specific moment (1-minute window for practical purposes)
        # This represents: "At time τ, what's the probability of finding a load?"
        time_window_minutes = 1  # Instantaneous probability (1-minute granularity)
        lambda_in_window = lambda_per_minute * time_window_minutes
        
        # Apply Poisson probability formula: P(≥1 arrival) = 1 - exp(-λ)
        probability = 1 - np.exp(-lambda_in_window)
        
        result['probability'] = float(probability)
        result['lambda_per_minute'] = float(lambda_per_minute)
        result['time_window_minutes'] = time_window_minutes
        
        if elapsed_time is not None:
            result['elapsed_time_minutes'] = elapsed_time
            result['interpretation'] = f"{result['probability']:.1%} instantaneous probability of loads at τ={elapsed_time}min on this route (based on {n_trips_daily:.1f} daily trips, uniform distribution)"
        else:
            result['interpretation'] = f"{result['probability']:.1%} instantaneous probability of loads on this route (based on {n_trips_daily:.1f} daily trips, uniform distribution)"
    elif model_type == 'price':
        # If model was log-transformed, reverse it
        if bundle.encoders.get('log_transformed', False):
            actual_price = np.expm1(prediction[0])  # inverse of log1p
            result['price_eur'] = float(actual_price)
        else:
            result['price_eur'] = float(prediction[0])
        result['interpretation'] = f"Estimated price: €{result['price_eur']:.2f}"
    elif model_type == 'weight':
        if bundle.encoders.get('log_transformed', False):
            actual_weight = np.expm1(prediction[0])
            result['weight_kg'] = float(actual_weight)
        else:
            result['weight_kg'] = float(prediction[0])
        result['interpretation'] = f"Estimated weight: {result['weight_kg']:.2f} kg"
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Run inference with trained ATLAS models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict probability for a route
  python scripts/inference.py --model probability --origin 1 --destination 100
  
  # Predict price for specific date
  python scripts/inference.py --model price --origin 1 --destination 100 --date 2024-06-15
  
  # Predict weight with refrigerated truck
  python scripts/inference.py --model weight --origin 1 --destination 100 --truck-type refrigerado
  
  # Use specific model version
  python scripts/inference.py --model price --origin 1 --destination 100 --model-path models/price_xgboost_v20251023_120000/
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['probability', 'price', 'weight'],
        help='Type of model to use for prediction'
    )
    
    parser.add_argument(
        '--origin',
        type=int,
        required=True,
        help='Origin location ID'
    )
    
    parser.add_argument(
        '--destination',
        type=int,
        required=True,
        help='Destination location ID'
    )
    
    parser.add_argument(
        '--date',
        type=str,
        default=None,
        help='Date for prediction (YYYY-MM-DD), defaults to today'
    )
    
    parser.add_argument(
        '--truck-type',
        type=str,
        default='normal',
        choices=['normal', 'refrigerado'],
        help='Type of truck'
    )
    
    parser.add_argument(
        '--tipo-mercancia',
        type=str,
        default='normal',
        choices=['normal', 'refrigerada'],
        help='Type of merchandise'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to specific model bundle (optional, uses latest if not specified)'
    )
    
    parser.add_argument(
        '--no-db',
        action='store_true',
        help='Do not load historical stats from database (use placeholder zeros)'
    )
    
    parser.add_argument(
        '--elapsed-time',
        type=int,
        default=None,
        help='Elapsed time in minutes since trip start (for time-dependent probability, only for Regime B with shape function)'
    )
    
    args = parser.parse_args()
    
    try:
        result = run_inference(
            model_type=args.model,
            origin_id=args.origin,
            destination_id=args.destination,
            date=args.date,
            truck_type=args.truck_type,
            tipo_mercancia=args.tipo_mercancia,
            model_path=args.model_path,
            use_db=not args.no_db,
            elapsed_time=args.elapsed_time
        )
        
        # Display results
        print("\n" + "="*60)
        print("PREDICTION RESULTS")
        print("="*60)
        print(f"Route: {result['origin_id']} → {result['destination_id']}")
        print(f"Date: {result['date']}")
        print(f"Truck type: {result['truck_type']}")
        print(f"Merchandise: {result['tipo_mercancia']}")
        print(f"Model: {result['model_type']}")
        print("-"*60)
        print(f"✓ {result['interpretation']}")
        print(f"  Raw prediction: {result['raw_prediction']:.6f}")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
