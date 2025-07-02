"""
Base model interface/abstract classes for the ML library recreation project.
Defines the common interface that all models should implement.
"""

from abc import ABC, abstractmethod
import numpy as np
import logging


class BaseModel(ABC):
    """Abstract base class for all machine learning models."""
    
    def __init__(self):
        """Initialize the base model."""
        self.is_fitted = False
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def fit(self, X, y):
        """
        Fit the model to training data.
        
        Args:
            X (array-like): Training features of shape (n_samples, n_features)
            y (array-like): Training targets of shape (n_samples,)
            
        Returns:
            self: Returns self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X (array-like): Features of shape (n_samples, n_features)
            
        Returns:
            array: Predictions of shape (n_samples,)
        """
        pass
    
    def _validate_input(self, X, y=None):
        """
        Validate input data.
        
        Args:
            X (array-like): Input features
            y (array-like, optional): Target values
            
        Returns:
            tuple: (X, y) as numpy arrays
        """
        # FIRST: Handle tuple inputs before any operations
        if isinstance(X, tuple):
            X = X[0] if len(X) == 1 else np.array(X)
        if isinstance(y, tuple) and y is not None:
            y = y[0] if len(y) == 1 else np.array(y)
        
        # Convert to numpy arrays
        X = np.asarray(X)
        
        # Validate X
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got {X.ndim}D array")
        
        if X.shape[0] == 0:
            raise ValueError("X cannot be empty")
        
        if y is not None:
            y = np.asarray(y)
            
            # Validate y
            if y.ndim != 1:
                raise ValueError(f"y must be 1D, got {y.ndim}D array")
            
            if X.shape[0] != y.shape[0]:
                raise ValueError(f"X and y must have same number of samples. "
                               f"Got X: {X.shape[0]}, y: {y.shape[0]}")
            
            return X, y
        return X
    
    def _check_fitted(self):
        """Check if the model has been fitted."""
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
    
    def get_params(self):
        """
        Get parameters of the model.
        
        Returns:
            dict: Dictionary of parameter names and values
        """
        # Get all attributes that don't start with underscore
        params = {}
        for attr_name in dir(self):
            if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                attr_value = getattr(self, attr_name)
                if not isinstance(attr_value, (logging.Logger, type(None))):
                    params[attr_name] = attr_value
        return params
    
    def set_params(self, **params):
        """
        Set parameters of the model.
        
        Args:
            **params: Dictionary of parameter names and values
            
        Returns:
            self: Returns self for method chaining
        """
        for param_name, param_value in params.items():
            if hasattr(self, param_name):
                setattr(self, param_name, param_value)
            else:
                raise ValueError(f"Invalid parameter: {param_name}")
        return self


class BaseRegressor(BaseModel):
    """Base class for regression models."""
    
    def __init__(self):
        """Initialize the base regressor."""
        super().__init__()
        self.model_type = "regressor"
    
    def score(self, X, y):
        """
        Calculate R² score.
        
        Args:
            X (array-like): Features
            y (array-like): True targets
            
        Returns:
            float: R² score
        """
        self._check_fitted()
        X, y = self._validate_input(X, y)
        
        y_pred = self.predict(X)
        
        # Calculate R² score
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        
        return 1 - (ss_res / ss_tot)


class BaseClassifier(BaseModel):
    """Base class for classification models."""
    
    def __init__(self):
        """Initialize the base classifier."""
        super().__init__()
        self.model_type = "classifier"
        self.classes_ = None
    
    def score(self, X, y):
        """
        Calculate accuracy score.
        
        Args:
            X (array-like): Features
            y (array-like): True labels
            
        Returns:
            float: Accuracy score
        """
        self._check_fitted()
        X, y = self._validate_input(X, y)
        
        y_pred = self.predict(X)
        return np.mean(y == y_pred)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X (array-like): Features
            
        Returns:
            array: Class probabilities of shape (n_samples, n_classes)
        """
        # Default implementation - subclasses should override if they support probabilities
        raise NotImplementedError("This classifier does not support probability predictions")
    
    def _encode_labels(self, y):
        """
        Encode string labels to integers.
        
        Args:
            y (array-like): Labels
            
        Returns:
            array: Encoded labels
        """
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        
        # Create mapping from class to index
        class_to_idx = {cls: idx for idx, cls in enumerate(self.classes_)}
        
        # Encode labels
        y_encoded = np.array([class_to_idx[label] for label in y])
        
        return y_encoded
    
    def _decode_labels(self, y_encoded):
        """
        Decode integer labels back to original classes.
        
        Args:
            y_encoded (array-like): Encoded labels
            
        Returns:
            array: Original labels
        """
        if self.classes_ is None:
            return y_encoded
        
        return self.classes_[y_encoded]


class BaseClusterer(BaseModel):
    """Base class for clustering models."""
    
    def __init__(self):
        """Initialize the base clusterer."""
        super().__init__()
        self.model_type = "clusterer"
        self.labels_ = None
        self.cluster_centers_ = None
    
    def fit(self, X, y=None):
        """
        Fit clustering model. Note: y is ignored for clustering.
        
        Args:
            X (array-like): Features
            y: Ignored, present for API consistency
            
        Returns:
            self: Returns self for method chaining
        """
        # Base implementation - subclasses should override
        pass
    
    def predict(self, X):
        """
        Predict cluster labels for new data.
        
        Args:
            X (array-like): Features
            
        Returns:
            array: Cluster labels
        """
        # Default implementation - subclasses should override
        self._check_fitted()
        X, _ = self._validate_input(X)
        
        # For most clustering algorithms, this would assign to nearest cluster center
        raise NotImplementedError("Subclass must implement predict method")
    
    def fit_predict(self, X, y=None):
        """
        Fit model and predict cluster labels.
        
        Args:
            X (array-like): Features
            y: Ignored, present for API consistency
            
        Returns:
            array: Cluster labels
        """
        self.fit(X, y)
        return self.labels_


class BaseTransformer(BaseModel):
    """Base class for data transformers (PCA, scalers, etc.)."""
    
    def __init__(self):
        """Initialize the base transformer."""
        super().__init__()
        self.model_type = "transformer"
    
    def predict(self, X):
        """
        Transform is the main method for transformers, not predict.
        This method redirects to transform for consistency.
        
        Args:
            X (array-like): Features to transform
            
        Returns:
            array: Transformed features
        """
        return self.transform(X)
    
    @abstractmethod
    def transform(self, X):
        """
        Transform the input features.
        
        Args:
            X (array-like): Features to transform
            
        Returns:
            array: Transformed features
        """
        pass
    
    def fit_transform(self, X, y=None):
        """
        Fit transformer and transform the data.
        
        Args:
            X (array-like): Features
            y: Target values (ignored by most transformers)
            
        Returns:
            array: Transformed features
        """
        self.fit(X, y)
        return self.transform(X)
    
    def inverse_transform(self, X):
        """
        Inverse transform the features.
        
        Args:
            X (array-like): Transformed features
            
        Returns:
            array: Original features
        """
        # Default implementation - not all transformers support inverse transform
        raise NotImplementedError("This transformer does not support inverse transform")


# Utility functions for model validation
def check_array(array, accept_sparse=False, dtype=None, ensure_2d=True):
    """
    Validate and convert input to numpy array.
    
    Args:
        array: Input array
        accept_sparse (bool): Whether to accept sparse matrices
        dtype: Desired data type
        ensure_2d (bool): Whether to ensure 2D array
        
    Returns:
        array: Validated numpy array
    """
    array = np.asarray(array, dtype=dtype)
    
    if ensure_2d and array.ndim == 1:
        array = array.reshape(-1, 1)
    
    if array.ndim < 1:
        raise ValueError("Input must be at least 1D")
    
    return array


def check_X_y(X, y, accept_sparse=False, dtype=None):
    """
    Validate X and y arrays.
    
    Args:
        X: Input features
        y: Target values
        accept_sparse (bool): Whether to accept sparse matrices
        dtype: Desired data type
        
    Returns:
        tuple: (X, y) as validated numpy arrays
    """
    X = check_array(X, accept_sparse=accept_sparse, dtype=dtype, ensure_2d=True)
    y = check_array(y, accept_sparse=False, dtype=dtype, ensure_2d=False)
    
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y must have the same number of samples. "
                        f"Got X: {X.shape[0]}, y: {y.shape[0]}")
    
    return X, y