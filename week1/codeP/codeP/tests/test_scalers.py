"""
Unit tests for preprocessing scalers.
"""

import unittest
import numpy as np
from preprocessing.scalers import StandardScaler, MinMaxScaler


class TestStandardScaler(unittest.TestCase):
    """Test cases for StandardScaler."""
    
    def setUp(self):
        """Set up test data."""
        self.X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        self.scaler = StandardScaler()
    
    def test_fit_transform(self):
        """Test fit_transform method."""
        X_scaled = self.scaler.fit_transform(self.X)
        
        # Check that mean is approximately 0
        np.testing.assert_array_almost_equal(np.mean(X_scaled, axis=0), [0, 0], decimal=10)
        
        # Check that std is approximately 1
        np.testing.assert_array_almost_equal(np.std(X_scaled, axis=0), [1, 1], decimal=10)
    
    def test_inverse_transform(self):
        """Test inverse_transform method."""
        X_scaled = self.scaler.fit_transform(self.X)
        X_restored = self.scaler.inverse_transform(X_scaled)
        
        # Should get back original data
        np.testing.assert_array_almost_equal(X_restored, self.X, decimal=10)
    
    def test_not_fitted_error(self):
        """Test error when not fitted."""
        with self.assertRaises(ValueError):
            self.scaler.transform(self.X)


class TestMinMaxScaler(unittest.TestCase):
    """Test cases for MinMaxScaler."""
    
    def setUp(self):
        """Set up test data."""
        self.X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        self.scaler = MinMaxScaler()
    
    def test_fit_transform_default_range(self):
        """Test fit_transform with default range [0, 1]."""
        X_scaled = self.scaler.fit_transform(self.X)
        
        # Check that min is 0 and max is 1
        np.testing.assert_array_almost_equal(np.min(X_scaled, axis=0), [0, 0])
        np.testing.assert_array_almost_equal(np.max(X_scaled, axis=0), [1, 1])
    
    def test_fit_transform_custom_range(self):
        """Test fit_transform with custom range [-1, 1]."""
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_scaled = scaler.fit_transform(self.X)
        
        # Check that min is -1 and max is 1
        np.testing.assert_array_almost_equal(np.min(X_scaled, axis=0), [-1, -1])
        np.testing.assert_array_almost_equal(np.max(X_scaled, axis=0), [1, 1])
    
    def test_inverse_transform(self):
        """Test inverse_transform method."""
        X_scaled = self.scaler.fit_transform(self.X)
        X_restored = self.scaler.inverse_transform(X_scaled)
        
        # Should get back original data
        np.testing.assert_array_almost_equal(X_restored, self.X, decimal=10)


if __name__ == '__main__':
    unittest.main()
