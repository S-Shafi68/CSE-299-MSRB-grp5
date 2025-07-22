"""
Comprehensive tests for Week 2 components.
Tests Ridge, Lasso, and Preprocessing integration.
"""

import unittest
import numpy as np
from models.linear_regression import LinearRegression
from models.ridge import Ridge
from models.lasso import Lasso
from preprocessing.scalers import StandardScaler, MinMaxScaler
from utils.metrics import mse, r2_score


class TestWeek2Models(unittest.TestCase):
    """Test Week 2 regression models."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.X = np.random.randn(100, 3)
        self.y = self.X @ [1.5, -2.0, 0.5] + np.random.randn(100) * 0.1
        
        # Split data
        self.X_train, self.X_test = self.X[:80], self.X[80:]
        self.y_train, self.y_test = self.y[:80], self.y[80:]
    
    def test_ridge_regression(self):
        """Test Ridge regression implementation."""
        model = Ridge(alpha=1.0)
        model.fit(self.X_train, self.y_train)
        
        # Check model is fitted
        self.assertIsNotNone(model.coef_)
        self.assertIsNotNone(model.intercept_)
        
        # Make predictions
        y_pred = model.predict(self.X_test)
        
        # Check predictions are reasonable
        self.assertEqual(len(y_pred), len(self.y_test))
        self.assertLess(mse(self.y_test, y_pred), 1.0)  # Should be < 1.0 for this simple case
    
    def test_lasso_regression(self):
        """Test Lasso regression implementation."""
        model = Lasso(alpha=0.1, max_iter=1000)
        model.fit(self.X_train, self.y_train)
        
        # Check model is fitted
        self.assertIsNotNone(model.coef_)
        self.assertIsNotNone(model.intercept_)
        
        # Make predictions
        y_pred = model.predict(self.X_test)
        
        # Check predictions are reasonable
        self.assertEqual(len(y_pred), len(self.y_test))
        self.assertLess(mse(self.y_test, y_pred), 1.0)
    
    def test_model_comparison(self):
        """Test all three models on same data."""
        models = {
            'Linear': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1, max_iter=1000)
        }
        
        results = {}
        for name, model in models.items():
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            results[name] = {
                'mse': mse(self.y_test, y_pred),
                'r2': r2_score(self.y_test, y_pred)
            }
        
        # All models should have reasonable performance
        for name, metrics in results.items():
            self.assertLess(metrics['mse'], 2.0, f"{name} MSE too high")
            self.assertGreater(metrics['r2'], 0.5, f"{name} R² too low")
        
        print("\n=== Model Comparison Results ===")
        for name, metrics in results.items():
            print(f"{name}: MSE={metrics['mse']:.4f}, R²={metrics['r2']:.4f}")


class TestPreprocessingIntegration(unittest.TestCase):
    """Test preprocessing integration with models."""
    
    def setUp(self):
        """Set up test data."""
        # Create data with different scales
        np.random.seed(42)
        self.X = np.column_stack([
            np.random.randn(100) * 100,    # Large scale feature
            np.random.randn(100) * 0.01,   # Small scale feature
            np.random.randn(100)           # Normal scale feature
        ])
        self.y = np.random.randn(100)
        
        self.X_train, self.X_test = self.X[:80], self.X[80:]
        self.y_train, self.y_test = self.y[:80], self.y[80:]
    
    def test_standard_scaler_with_models(self):
        """Test StandardScaler with all models."""
        # Apply scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        # Test with Ridge
        model = Ridge(alpha=1.0)
        model.fit(X_train_scaled, self.y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Should make valid predictions
        self.assertEqual(len(y_pred), len(self.y_test))
        self.assertFalse(np.any(np.isnan(y_pred)), "Predictions contain NaN")
    
    def test_minmax_scaler_with_models(self):
        """Test MinMaxScaler with all models."""
        # Apply scaling
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        # Check scaling worked
        self.assertAlmostEqual(np.min(X_train_scaled), -1.0, places=10)
        self.assertAlmostEqual(np.max(X_train_scaled), 1.0, places=10)
        
        # Test with Lasso
        model = Lasso(alpha=0.1, max_iter=1000)
        model.fit(X_train_scaled, self.y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Should make valid predictions
        self.assertEqual(len(y_pred), len(self.y_test))
        self.assertFalse(np.any(np.isnan(y_pred)), "Predictions contain NaN")


if __name__ == '__main__':
    unittest.main(verbosity=2)
