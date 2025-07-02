"""
Data preprocessing scalers for feature normalization.

This module provides scaling transformations to normalize features
for machine learning algorithms.
"""

import numpy as np
from models.base_model import BaseTransformer
from logger import get_logger


class StandardScaler(BaseTransformer):
    """
    Standardize features by removing the mean and scaling to unit variance.
    
    The standard score of a sample x is calculated as:
        z = (x - u) / s
    where u is the mean of the training samples and s is the standard deviation.
    
    Parameters:
    -----------
    with_mean : bool, default=True
        If True, center the data before scaling.
    with_std : bool, default=True
        If True, scale the data to unit variance.
    """
    
    def __init__(self, with_mean=True, with_std=True):
        super().__init__()
        self.with_mean = with_mean
        self.with_std = with_std
        
        # Statistics computed during fitting
        self.mean_ = None
        self.std_ = None
        self.n_features_in_ = None
        
        # Setup logging
        self.logger = get_logger(f"{self.__class__.__name__}")
    
    def fit(self, X, y=None):
        """
        Compute the mean and std to be used for later scaling.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y : array-like, optional
            Ignored. This parameter exists only for compatibility.
            
        Returns:
        --------
        self : StandardScaler
            Returns self for method chaining.
        """
        # Validate input
        X = self._validate_input(X, y=None)
        self.n_features_in_ = X.shape[1]
        
        self.logger.info(f"Fitting StandardScaler on data with shape {X.shape}")
        
        # Compute statistics
        if self.with_mean:
            self.mean_ = np.mean(X, axis=0)
        else:
            self.mean_ = np.zeros(X.shape[1])
            
        if self.with_std:
            self.std_ = np.std(X, axis=0, ddof=0)
            # Avoid division by zero
            self.std_[self.std_ == 0] = 1.0
        else:
            self.std_ = np.ones(X.shape[1])
        
        self.logger.debug(f"Computed mean: {self.mean_}")
        self.logger.debug(f"Computed std: {self.std_}")
        
        return self
    
    def transform(self, X):
        """
        Perform standardization by centering and scaling.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The data used to scale along the features axis.
            
        Returns:
        --------
        X_scaled : array-like, shape (n_samples, n_features)
            Transformed array.
        """
        # Check if fitted
        if self.mean_ is None or self.std_ is None:
            raise ValueError("StandardScaler must be fitted before transform")
        
        # Validate input
        X = self._validate_input(X, y=None)
        
        # Check feature consistency
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but StandardScaler was fitted with {self.n_features_in_} features")
        
        # Apply standardization
        X_scaled = X.copy()
        if self.with_mean:
            X_scaled = X_scaled - self.mean_
        if self.with_std:
            X_scaled = X_scaled / self.std_
        
        return X_scaled
    
    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input samples.
        y : array-like, optional
            Target values (ignored).
            
        Returns:
        --------
        X_scaled : array-like, shape (n_samples, n_features)
            Transformed array.
        """
        return self.fit(X, y).transform(X)
    
    def inverse_transform(self, X):
        """
        Scale back the data to the original representation.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The data used to scale back.
            
        Returns:
        --------
        X_original : array-like, shape (n_samples, n_features)
            Transformed array.
        """
        # Check if fitted
        if self.mean_ is None or self.std_ is None:
            raise ValueError("StandardScaler must be fitted before inverse_transform")
        
        # Validate input
        X = self._validate_input(X, y=None)
        
        # Apply inverse transformation
        X_original = X.copy()
        if self.with_std:
            X_original = X_original * self.std_
        if self.with_mean:
            X_original = X_original + self.mean_
        
        return X_original
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'with_mean': self.with_mean,
            'with_std': self.with_std
        }
    
    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key}")
        return self
    
    def __str__(self):
        """String representation of the scaler."""
        if self.mean_ is not None:
            return f"StandardScaler(fitted=True, n_features={len(self.mean_)})"
        else:
            return f"StandardScaler(fitted=False)"
    
    def __repr__(self):
        """Detailed string representation of the scaler."""
        return f"StandardScaler(with_mean={self.with_mean}, with_std={self.with_std})"


class MinMaxScaler(BaseTransformer):
    """
    Transform features by scaling each feature to a given range.
    
    This estimator scales and translates each feature individually such
    that it is in the given range on the training set, e.g. between
    zero and one.
    
    The transformation is given by:
        X_scaled = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_scaled * (max - min) + min
    
    Parameters:
    -----------
    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.
    """
    
    def __init__(self, feature_range=(0, 1)):
        super().__init__()
        self.feature_range = feature_range
        
        # Statistics computed during fitting
        self.min_ = None
        self.max_ = None
        self.scale_ = None
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None
        self.n_features_in_ = None
        
        # Setup logging
        self.logger = get_logger(f"{self.__class__.__name__}")
    
    def fit(self, X, y=None):
        """
        Compute the minimum and maximum to be used for later scaling.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The data used to compute the per-feature minimum and maximum
            used for later scaling along the features axis.
        y : array-like, optional
            Ignored. This parameter exists only for compatibility.
            
        Returns:
        --------
        self : MinMaxScaler
            Returns self for method chaining.
        """
        # Validate input
        X = self._validate_input(X, y=None)
        self.n_features_in_ = X.shape[1]
        
        self.logger.info(f"Fitting MinMaxScaler on data with shape {X.shape}")
        
        # Compute statistics
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        
        # Avoid division by zero
        self.data_range_[self.data_range_ == 0] = 1.0
        
        # Compute scaling parameters
        feature_min, feature_max = self.feature_range
        self.scale_ = (feature_max - feature_min) / self.data_range_
        self.min_ = feature_min - self.data_min_ * self.scale_
        
        self.logger.debug(f"Data min: {self.data_min_}")
        self.logger.debug(f"Data max: {self.data_max_}")
        self.logger.debug(f"Scale: {self.scale_}")
        
        return self
    
    def transform(self, X):
        """
        Scale features according to feature_range.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data that will be transformed.
            
        Returns:
        --------
        X_scaled : array-like, shape (n_samples, n_features)
            Transformed data.
        """
        # Check if fitted
        if self.scale_ is None:
            raise ValueError("MinMaxScaler must be fitted before transform")
        
        # Validate input
        X = self._validate_input(X, y=None)
        
        # Check feature consistency
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but MinMaxScaler was fitted with {self.n_features_in_} features")
        
        # Apply scaling
        X_scaled = X * self.scale_ + self.min_
        
        return X_scaled
    
    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input samples.
        y : array-like, optional
            Target values (ignored).
            
        Returns:
        --------
        X_scaled : array-like, shape (n_samples, n_features)
            Transformed array.
        """
        return self.fit(X, y).transform(X)
    
    def inverse_transform(self, X):
        """
        Undo the scaling of X according to feature_range.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data that will be transformed.
            
        Returns:
        --------
        X_original : array-like, shape (n_samples, n_features)
            Transformed data.
        """
        # Check if fitted
        if self.scale_ is None:
            raise ValueError("MinMaxScaler must be fitted before inverse_transform")
        
        # Validate input
        X = self._validate_input(X, y=None)
        
        # Apply inverse transformation
        X_original = (X - self.min_) / self.scale_
        
        return X_original
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'feature_range': self.feature_range
        }
    
    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key}")
        return self
    
    def __str__(self):
        """String representation of the scaler."""
        if self.scale_ is not None:
            return f"MinMaxScaler(fitted=True, n_features={len(self.scale_)})"
        else:
            return f"MinMaxScaler(fitted=False)"
    
    def __repr__(self):
        """Detailed string representation of the scaler."""
        return f"MinMaxScaler(feature_range={self.feature_range})"
