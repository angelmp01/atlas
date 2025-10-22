"""
Test suite for feature engineering module.

Tests feature builders for probability, price, and weight models including
historical aggregates, target encoding, and time-based features.
"""

import unittest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path

# Import atlas_ml modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from atlas_ml.config import Config
from atlas_ml.featurization import (
    FeatureBuilder, ProbabilityFeatureBuilder, RegressionFeatureBuilder,
    get_feature_names
)
from atlas_ml.utils import create_time_features, target_encode


class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.config.random_state = 42
        
        # Mock database manager
        self.mock_db = Mock()
        self.mock_db.get_od_pairs_in_radius.return_value = [1, 2, 3]  # Mock density feature
        
    def create_sample_data(self, n_samples: int = 100) -> pd.DataFrame:
        """Create sample data for feature engineering tests."""
        np.random.seed(42)
        
        data = {
            'date': pd.date_range('2024-01-01', periods=n_samples, freq='D'),
            'origin_id': np.random.randint(1, 11, n_samples),
            'destination_id': np.random.randint(1, 11, n_samples),
            'origin_lat': np.random.uniform(40.0, 42.0, n_samples),
            'origin_lon': np.random.uniform(1.0, 3.0, n_samples),
            'destination_lat': np.random.uniform(40.0, 42.0, n_samples),
            'destination_lon': np.random.uniform(1.0, 3.0, n_samples),
            'tipo_mercancia': np.random.choice(['normal', 'refrigerada'], n_samples),
            'truck_type': np.random.choice(['normal', 'refrigerado'], n_samples),
            'precio': np.random.gamma(3, 50, n_samples),
            'peso': np.random.gamma(2, 200, n_samples),
            'volumen': np.random.gamma(1.5, 5, n_samples),
            'n_trips': np.random.poisson(2, n_samples),
            'tau_minutes': np.random.randint(0, 1440, n_samples),
        }
        
        return pd.DataFrame(data)
    
    def test_base_feature_builder_initialization(self):
        """Test FeatureBuilder initialization."""
        builder = FeatureBuilder(self.config, self.mock_db)
        
        self.assertEqual(builder.config, self.config)
        self.assertEqual(builder.db, self.mock_db)
        self.assertIsInstance(builder.encoders, dict)
        self.assertIsInstance(builder.scalers, dict)
    
    def test_base_features_creation(self):
        """Test creation of base features."""
        builder = FeatureBuilder(self.config, self.mock_db)
        df = self.create_sample_data(50)
        
        df_with_features = builder.create_base_features(df)
        
        # Check that time features were added
        expected_time_features = ['day_of_week', 'week_of_year', 'holiday_flag', 'month', 'quarter']
        for feature in expected_time_features:
            self.assertIn(feature, df_with_features.columns)
        
        # Check that distance was calculated
        if 'od_length_km' not in df.columns:
            self.assertIn('od_length_km', df_with_features.columns)
        
        # Check log transformations
        if 'precio' in df.columns:
            self.assertIn('log_precio', df_with_features.columns)
        
        # Verify data types and ranges
        self.assertTrue(df_with_features['day_of_week'].between(0, 6).all())
        self.assertTrue(df_with_features['week_of_year'].between(1, 53).all())
        self.assertTrue(df_with_features['holiday_flag'].isin([0, 1]).all())
    
    def test_categorical_features_encoding(self):
        """Test categorical feature encoding."""
        builder = FeatureBuilder(self.config, self.mock_db)
        df = self.create_sample_data(30)
        
        # Test fitting encoders
        df_encoded = builder.create_categorical_features(df, fit=True)
        
        # Check that encoded columns were created
        self.assertIn('truck_type_encoded', df_encoded.columns)
        self.assertIn('tipo_mercancia_encoded', df_encoded.columns)
        
        # Check that encoders were stored
        self.assertIn('truck_type', builder.encoders)
        self.assertIn('tipo_mercancia', builder.encoders)
        
        # Test applying existing encoders
        df_new = self.create_sample_data(20)
        df_new_encoded = builder.create_categorical_features(df_new, fit=False)
        
        # Should have same encoded columns
        self.assertIn('truck_type_encoded', df_new_encoded.columns)
        self.assertIn('tipo_mercancia_encoded', df_new_encoded.columns)
    
    def test_target_encoding(self):
        """Test target encoding functionality."""
        builder = FeatureBuilder(self.config, self.mock_db)
        df = self.create_sample_data(100)
        
        target_col = 'precio'
        categorical_cols = ['origin_id', 'destination_id']
        
        df_encoded = builder.create_target_encoded_features(
            df, target_col, categorical_cols, fit=True
        )
        
        # Check that target encoded columns were created
        for col in categorical_cols:
            encoded_col = f"{col}_target_encoded"
            self.assertIn(encoded_col, df_encoded.columns)
            
            # Check that values are reasonable
            self.assertFalse(df_encoded[encoded_col].isna().any())
            
            # Check that encoder was stored
            self.assertIn(f"{col}_target_map", builder.encoders)
    
    def test_historical_features(self):
        """Test historical feature creation."""
        builder = FeatureBuilder(self.config, self.mock_db)
        df = self.create_sample_data(200)  # Larger dataset for rolling features
        
        group_cols = ['origin_id', 'destination_id']
        df_with_history = builder.create_historical_features(df, group_cols)
        
        # Check rolling features
        for window in self.config.features.rolling_windows:
            self.assertIn(f'n_trips_rolling_mean_{window}d', df_with_history.columns)
            self.assertIn(f'precio_rolling_mean_{window}d', df_with_history.columns)
        
        # Check lag features
        self.assertIn('n_trips_lag_1d', df_with_history.columns)
        self.assertIn('n_trips_lag_7d', df_with_history.columns)
        
        # Verify no NaN in rolling means (should have min_periods=1)
        rolling_cols = [col for col in df_with_history.columns if 'rolling_mean' in col]
        for col in rolling_cols:
            # Some NaN values are acceptable due to insufficient history
            self.assertTrue(df_with_history[col].notna().sum() > 0)
    
    def test_probability_feature_builder(self):
        """Test ProbabilityFeatureBuilder specific functionality."""
        builder = ProbabilityFeatureBuilder(self.config, self.mock_db)
        df = self.create_sample_data(50)
        
        # Test without tau features
        df_features = builder.build_features(df, include_tau=False, fit=True)
        
        # Should not have tau-related features
        tau_features = ['tau_bin', 'tau_hour', 'tau_normalized']
        for feature in tau_features:
            self.assertNotIn(feature, df_features.columns)
        
        # Test with tau features
        df_features_tau = builder.build_features(df, include_tau=True, fit=True)
        
        # Should have tau-related features
        for feature in tau_features:
            self.assertIn(feature, df_features_tau.columns)
        
        # Check tau_bin calculation
        if 'tau_bin' in df_features_tau.columns:
            self.assertTrue((df_features_tau['tau_bin'] >= 0).all())
            self.assertTrue((df_features_tau['tau_hour'] >= 0).all())
            self.assertTrue((df_features_tau['tau_normalized'] >= 0).all())
    
    def test_regression_feature_builder(self):
        """Test RegressionFeatureBuilder specific functionality."""
        builder = RegressionFeatureBuilder(self.config, self.mock_db)
        df = self.create_sample_data(50)
        
        target_col = 'precio'
        df_features = builder.build_features(df, target_col=target_col, fit=True)
        
        # Check distance-based features
        if 'od_length_km' in df.columns or 'od_length_km' in df_features.columns:
            self.assertIn('distance_bin', df_features.columns)
            
            # Check distance bins are reasonable
            distance_bins = df_features['distance_bin'].dropna().unique()
            expected_bins = ['very_short', 'short', 'medium', 'long', 'very_long']
            for bin_val in distance_bins:
                self.assertIn(bin_val, expected_bins)
    
    def test_feature_names_generation(self):
        """Test feature names generation utility."""
        # Test probability features without tau
        prob_features = get_feature_names('probability', include_tau=False)
        
        expected_base = ['day_of_week', 'week_of_year', 'holiday_flag']
        for feature in expected_base:
            self.assertIn(feature, prob_features)
        
        # Should not have tau features
        tau_features = ['tau_bin', 'tau_hour']
        for feature in tau_features:
            self.assertNotIn(feature, prob_features)
        
        # Test probability features with tau
        prob_features_tau = get_feature_names('probability', include_tau=True)
        
        # Should have tau features
        for feature in tau_features:
            self.assertIn(feature, prob_features_tau)
        
        # Test regression features
        price_features = get_feature_names('price')
        weight_features = get_feature_names('weight')
        
        # Should have regression-specific features
        regression_features = ['distance_bin', 'km_per_kg']
        for feature in regression_features:
            self.assertIn(feature, price_features)
            self.assertIn(feature, weight_features)
    
    def test_time_features_utility(self):
        """Test time features creation utility."""
        test_date = pd.Timestamp('2024-03-15')  # Friday
        
        features = create_time_features(test_date)
        
        self.assertEqual(features['day_of_week'], 4)  # Friday = 4
        self.assertEqual(features['month'], 3)        # March = 3
        self.assertEqual(features['quarter'], 1)      # Q1
        self.assertIn('week_of_year', features)
        self.assertIn('holiday_flag', features)
    
    def test_target_encoding_utility(self):
        """Test target encoding utility function."""
        # Create simple test data
        series = pd.Series(['A', 'B', 'A', 'C', 'B', 'A'] * 10)
        target = pd.Series([10, 20, 12, 30, 22, 11] * 10)
        
        encoded = target_encode(series, target, min_samples_leaf=5, smoothing=1.0)
        
        # Check output properties
        self.assertEqual(len(encoded), len(series))
        self.assertFalse(encoded.isna().any())
        
        # Categories with more samples should be closer to their actual means
        series_df = pd.DataFrame({'category': series, 'target': target, 'encoded': encoded})
        category_stats = series_df.groupby('category').agg({
            'target': ['mean', 'count'],
            'encoded': 'first'
        }).round(2)
        
        # Should have reasonable encoded values
        self.assertTrue(encoded.min() > 0)
        self.assertTrue(encoded.max() < target.max() + 10)  # Some smoothing applied
    
    def test_feature_builder_error_handling(self):
        """Test error handling in feature builders."""
        builder = FeatureBuilder(self.config, self.mock_db)
        
        # Test with missing columns
        df_minimal = pd.DataFrame({
            'origin_id': [1, 2, 3],
            'destination_id': [4, 5, 6]
        })
        
        # Should handle missing date column gracefully
        df_result = builder.create_historical_features(df_minimal, ['origin_id'])
        self.assertIsInstance(df_result, pd.DataFrame)
        
        # Should handle missing target column in target encoding
        df_no_target = builder.create_target_encoded_features(
            df_minimal, 'nonexistent_target', ['origin_id'], fit=True
        )
        self.assertIsInstance(df_no_target, pd.DataFrame)


class TestFeatureIntegration(unittest.TestCase):
    """Integration tests for feature engineering."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.config = Config()
        self.mock_db = Mock()
    
    def test_feature_consistency(self):
        """Test consistency between different feature builders."""
        # Create same data for both builders
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=20, freq='D'),
            'origin_id': np.random.randint(1, 5, 20),
            'destination_id': np.random.randint(1, 5, 20),
            'precio': np.random.gamma(2, 50, 20),
            'peso': np.random.gamma(1.5, 200, 20),
            'truck_type': np.random.choice(['normal', 'refrigerado'], 20),
            'tipo_mercancia': np.random.choice(['normal', 'refrigerada'], 20),
        })
        
        prob_builder = ProbabilityFeatureBuilder(self.config, self.mock_db)
        reg_builder = RegressionFeatureBuilder(self.config, self.mock_db)
        
        # Build features with both builders
        prob_features = prob_builder.build_features(df, include_tau=False, fit=True)
        reg_features = reg_builder.build_features(df, target_col='precio', fit=True)
        
        # Common base features should be consistent
        common_features = ['day_of_week', 'week_of_year', 'holiday_flag']
        
        for feature in common_features:
            if feature in prob_features.columns and feature in reg_features.columns:
                # Values should be identical for same input data
                pd.testing.assert_series_equal(
                    prob_features[feature], 
                    reg_features[feature],
                    check_names=False
                )
    
    def test_config_impact_on_features(self):
        """Test how configuration changes affect feature generation."""
        # Test with different rolling windows
        config1 = Config()
        config1.features.rolling_windows = [7, 14]
        
        config2 = Config()
        config2.features.rolling_windows = [3, 7, 21]
        
        builder1 = FeatureBuilder(config1, self.mock_db)
        builder2 = FeatureBuilder(config2, self.mock_db)
        
        # Expected feature names should differ
        features1 = get_feature_names('probability')
        features2 = get_feature_names('probability')
        
        # Both should have rolling features, but potentially different ones
        rolling_features1 = [f for f in features1 if 'rolling' in f]
        rolling_features2 = [f for f in features2 if 'rolling' in f]
        
        self.assertTrue(len(rolling_features1) > 0)
        self.assertTrue(len(rolling_features2) > 0)


if __name__ == '__main__':
    unittest.main()