"""
Utility functions for ATLAS ML package.

Provides common helpers for time binning, encoding, random seeds, and calendar utilities.
"""

import logging
import random
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    # XGBoost uses numpy random state internally


def to_tau_bin(tau_minutes: int, bin_size: int = 10) -> int:
    """
    Convert elapsed time in minutes to time bin.
    
    This function maps continuous elapsed time since trip departure to discrete
    time bins for feature engineering and modeling.
    
    Args:
        tau_minutes: Elapsed time in minutes since trip start
        bin_size: Size of each time bin in minutes (default: 10)
        
    Returns:
        Time bin index (0-based)
        
    Examples:
        >>> to_tau_bin(0, 10)    # Start of trip
        0
        >>> to_tau_bin(15, 10)   # 15 minutes elapsed
        1
        >>> to_tau_bin(25, 10)   # 25 minutes elapsed
        2
        >>> to_tau_bin(-5, 10)   # Negative time (clamped)
        0
    """
    # Clamp negative values to 0
    tau_minutes = max(0, tau_minutes)
    
    # Convert to bin index
    bin_index = tau_minutes // bin_size
    
    # Optional: cap maximum bin (e.g., 24 hours = 144 bins at 10min resolution)
    max_bins = (24 * 60) // bin_size
    bin_index = min(bin_index, max_bins - 1)
    
    return int(bin_index)


def tau_bin_to_minutes(bin_index: int, bin_size: int = 10) -> Tuple[int, int]:
    """
    Convert time bin back to minute range.
    
    Args:
        bin_index: Time bin index
        bin_size: Size of each time bin in minutes
        
    Returns:
        Tuple of (start_minutes, end_minutes) for the bin
    """
    start_minutes = bin_index * bin_size
    end_minutes = (bin_index + 1) * bin_size
    return start_minutes, end_minutes


def get_day_of_week(date: pd.Timestamp) -> int:
    """
    Get day of week from date.
    
    Args:
        date: Pandas timestamp
        
    Returns:
        Day of week (0=Monday, 6=Sunday)
    """
    return date.dayofweek


def get_week_of_year(date: pd.Timestamp) -> int:
    """
    Get week of year from date.
    
    Args:
        date: Pandas timestamp
        
    Returns:
        Week of year (1-53)
    """
    return date.isocalendar().week


def get_holiday_flag(date: pd.Timestamp, holidays: Optional[List[str]] = None) -> int:
    """
    Check if date is a holiday.
    
    Args:
        date: Pandas timestamp
        holidays: List of holiday dates in "YYYY-MM-DD" format
        
    Returns:
        1 if holiday, 0 otherwise
    """
    if holidays is None:
        # Default Spanish holidays for 2024
        holidays = [
            "2024-01-01", "2024-01-06", "2024-03-29", "2024-05-01",
            "2024-08-15", "2024-10-12", "2024-11-01", "2024-12-06",
            "2024-12-08", "2024-12-25"
        ]
    
    date_str = date.strftime("%Y-%m-%d")
    return 1 if date_str in holidays else 0


def create_time_features(date: pd.Timestamp, holidays: Optional[List[str]] = None) -> dict:
    """
    Create all time-based features from a date.
    
    Args:
        date: Pandas timestamp
        holidays: List of holiday dates
        
    Returns:
        Dictionary with time features
    """
    return {
        "day_of_week": get_day_of_week(date),
        "week_of_year": get_week_of_year(date),
        "holiday_flag": get_holiday_flag(date, holidays),
        "month": date.month,
        "quarter": date.quarter,
    }


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great circle distance between two points on Earth.
    
    Uses the Haversine formula to compute the distance between two points
    given their latitude and longitude coordinates.
    
    Args:
        lat1, lon1: Latitude and longitude of first point in degrees
        lat2, lon2: Latitude and longitude of second point in degrees
        
    Returns:
        Distance in kilometers
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Earth radius in kilometers
    r = 6371
    
    return r * c


def target_encode(
    series: pd.Series,
    target: pd.Series,
    min_samples_leaf: int = 10,
    smoothing: float = 1.0,
    noise_level: float = 0.01
) -> pd.Series:
    """
    Apply target encoding to a categorical series.
    
    Target encoding replaces categorical values with the mean of the target
    variable for that category, with smoothing to prevent overfitting.
    
    Args:
        series: Categorical series to encode
        target: Target variable series
        min_samples_leaf: Minimum samples required for a category
        smoothing: Smoothing factor (higher = more smoothing toward global mean)
        noise_level: Small amount of noise to add for regularization
        
    Returns:
        Target-encoded series
    """
    # Calculate global mean
    global_mean = target.mean()
    
    # Calculate category statistics
    category_stats = pd.DataFrame({
        'sum': target.groupby(series).sum(),
        'count': target.groupby(series).count(),
        'mean': target.groupby(series).mean()
    }).fillna(0)
    
    # Apply smoothing formula
    # smoothed_mean = (count * category_mean + smoothing * global_mean) / (count + smoothing)
    category_stats['smoothed_mean'] = (
        (category_stats['count'] * category_stats['mean'] + smoothing * global_mean) /
        (category_stats['count'] + smoothing)
    )
    
    # For categories with too few samples, use global mean
    category_stats.loc[category_stats['count'] < min_samples_leaf, 'smoothed_mean'] = global_mean
    
    # Map back to original series
    encoded = series.map(category_stats['smoothed_mean']).fillna(global_mean)
    
    # Add small amount of noise for regularization
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, len(encoded))
        encoded = encoded + noise
    
    return encoded


def create_lag_features(
    df: pd.DataFrame,
    value_col: str,
    group_cols: List[str],
    date_col: str = "date",
    lags: List[int] = [1, 7, 14]
) -> pd.DataFrame:
    """
    Create lagged features for time series data.
    
    Args:
        df: Input dataframe
        value_col: Column to create lags for
        group_cols: Columns to group by for lag calculation
        date_col: Date column name
        lags: List of lag periods (in days)
        
    Returns:
        DataFrame with lag features added
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    for lag in lags:
        lag_col = f"{value_col}_lag_{lag}d"
        df[lag_col] = (
            df.sort_values([*group_cols, date_col])
            .groupby(group_cols)[value_col]
            .shift(lag)
        )
    
    return df


def create_rolling_features(
    df: pd.DataFrame,
    value_col: str,
    group_cols: List[str],
    date_col: str = "date",
    windows: List[int] = [7, 14, 28]
) -> pd.DataFrame:
    """
    Create rolling window features for time series data.
    
    Args:
        df: Input dataframe
        value_col: Column to create rolling features for
        group_cols: Columns to group by for rolling calculation
        date_col: Date column name
        windows: List of window sizes (in days)
        
    Returns:
        DataFrame with rolling features added
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    for window in windows:
        mean_col = f"{value_col}_rolling_mean_{window}d"
        std_col = f"{value_col}_rolling_std_{window}d"
        
        # Sort by group and date
        df_sorted = df.sort_values([*group_cols, date_col])
        
        # Calculate rolling statistics
        rolling_stats = (
            df_sorted.groupby(group_cols)[value_col]
            .rolling(window=window, min_periods=1)
            .agg(['mean', 'std'])
            .reset_index()
        )
        
        df[mean_col] = rolling_stats['mean']
        df[std_col] = rolling_stats['std'].fillna(0)
    
    return df


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger("atlas_ml")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def validate_coordinates(lat: float, lon: float) -> bool:
    """
    Validate latitude and longitude coordinates.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        
    Returns:
        True if coordinates are valid
    """
    return -90 <= lat <= 90 and -180 <= lon <= 180


def clamp_probability(prob: float) -> float:
    """
    Clamp probability to valid range [0, 1].
    
    Args:
        prob: Probability value
        
    Returns:
        Clamped probability
    """
    return max(0.0, min(1.0, prob))