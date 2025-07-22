"""
Categorical Data Encoders Implementation
"""

import numpy as np
from models.base import BaseTransformer
from logger import get_logger

logger = get_logger("encoders")

class LabelEncoder(BaseTransformer):
    """
    Encode categorical labels as integers.
    
    This encoder converts categorical labels (strings or other objects) 
    into integer labels in the range [0, n_classes-1].
    
    Example:
    --------
    >>> encoder = LabelEncoder()
    >>> labels = ['cat', 'dog', 'cat', 'bird', 'dog']
    >>> encoded = encoder.fit_transform(labels)
    >>> print(encoded)  # [0, 1, 0, 2, 1]
    >>> print(encoder.classes_)  # ['bird', 'cat', 'dog']
    """
    
    def __init__(self):
        super().__init__()
        self.classes_ = None
        self.class_to_index = None
        self.index_to_class = None
        
        # Setup logging
        self.logger = get_logger(f"{self.__class__.__name__}")
    
    def fit(self, y):
        """
        Fit label encoder to target values.
        
        Parameters:
        -----------
        y : array-like, shape (n_samples,)
            Target values to encode
            
        Returns:
        --------
        self : LabelEncoder
            Returns self for method chaining
        """
        y = np.array(y)
        
        # Get unique classes in sorted order for consistency
        self.classes_ = np.unique(y)
        
        # Create mapping dictionaries
        self.class_to_index = {cls: idx for idx, cls in enumerate(self.classes_)}
        self.index_to_class = {idx: cls for idx, cls in enumerate(self.classes_)}
        
        self.is_fitted = True
        
        self.logger.info(f"LabelEncoder fitted with {len(self.classes_)} classes")
        self.logger.debug(f"Classes: {self.classes_}")
        
        return self
    
    def transform(self, y):
        """
        Transform labels to encoded integers.
        
        Parameters:
        -----------
        y : array-like, shape (n_samples,)
            Target values to encode
            
        Returns:
        --------
        y_encoded : array, shape (n_samples,)
            Encoded labels as integers
        """
        if not self.is_fitted:
            raise ValueError("LabelEncoder must be fitted before transform")
        
        y = np.array(y)
        
        # Check for unknown labels
        unknown_labels = set(y) - set(self.classes_)
        if unknown_labels:
            raise ValueError(f"Unknown labels found: {unknown_labels}")
        
        # Encode labels
        encoded = np.array([self.class_to_index[label] for label in y])
        
        self.logger.debug(f"Transformed {len(y)} labels")
        return encoded
    
    def fit_transform(self, y):
        """
        Fit label encoder and transform labels.
        
        Parameters:
        -----------
        y : array-like, shape (n_samples,)
            Target values to encode
            
        Returns:
        --------
        y_encoded : array, shape (n_samples,)
            Encoded labels as integers
        """
        return self.fit(y).transform(y)
    
    def inverse_transform(self, y_encoded):
        """
        Transform encoded labels back to original labels.
        
        Parameters:
        -----------
        y_encoded : array-like, shape (n_samples,)
            Encoded labels as integers
            
        Returns:
        --------
        y_original : array, shape (n_samples,)
            Original labels
        """
        if not self.is_fitted:
            raise ValueError("LabelEncoder must be fitted before inverse_transform")
        
        y_encoded = np.array(y_encoded)
        
        # Check for valid indices
        max_index = len(self.classes_) - 1
        if np.any(y_encoded < 0) or np.any(y_encoded > max_index):
            raise ValueError(f"Encoded labels must be in range [0, {max_index}]")
        
        # Decode labels
        original = np.array([self.index_to_class[idx] for idx in y_encoded])
        
        self.logger.debug(f"Inverse transformed {len(y_encoded)} labels")
        return original
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {}
    
    def set_params(self, **params):
        """Set parameters for this estimator."""
        # LabelEncoder has no parameters to set
        return self
    
    def __str__(self):
        """String representation of the encoder."""
        if self.classes_ is not None:
            return f"LabelEncoder(fitted=True, n_classes={len(self.classes_)})"
        else:
            return f"LabelEncoder(fitted=False)"
    
    def __repr__(self):
        """Detailed string representation of the encoder."""
        return "LabelEncoder()"


class OneHotEncoder(BaseTransformer):
    """
    Encode categorical features as one-hot vectors.
    
    This encoder converts categorical features into a binary matrix where
    each category becomes a separate binary feature.
    
    Parameters:
    -----------
    sparse : bool, default=False
        Whether to return sparse matrix (not implemented in this version)
    drop : str or None, default=None
        Strategy for dropping one category per feature to avoid collinearity
    handle_unknown : str, default='error'
        How to handle unknown categories ('error' or 'ignore')
    
    Example:
    --------
    >>> encoder = OneHotEncoder()
    >>> features = [['cat'], ['dog'], ['cat'], ['bird']]
    >>> encoded = encoder.fit_transform(features)
    >>> print(encoded.shape)  # (4, 3)
    """
    
    def __init__(self, sparse=False, drop=None, handle_unknown='error'):
        super().__init__()
        self.sparse = sparse
        self.drop = drop
        self.handle_unknown = handle_unknown
        
        # Fitted attributes
        self.categories_ = None
        self.feature_names_in_ = None
        self.feature_names_out_ = None
        self.n_features_in_ = None
        
        # Setup logging
        self.logger = get_logger(f"{self.__class__.__name__}")
        
        # Validate parameters
        if handle_unknown not in ['error', 'ignore']:
            raise ValueError("handle_unknown must be 'error' or 'ignore'")
    
    def fit(self, X, y=None):
        """
        Fit OneHotEncoder to feature data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature data containing categorical values
        y : array-like, optional
            Ignored. This parameter exists only for compatibility.
            
        Returns:
        --------
        self : OneHotEncoder
            Returns self for method chaining
        """
        X = self._validate_input(X, y=None)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.n_features_in_ = X.shape[1]
        self.categories_ = []
        self.feature_names_out_ = []
        
        # Find unique categories for each feature
        for col_idx in range(X.shape[1]):
            unique_values = np.unique(X[:, col_idx])
            self.categories_.append(unique_values)
            
            # Generate feature names for output
            for value in unique_values:
                feature_name = f"x{col_idx}_{value}"
                self.feature_names_out_.append(feature_name)
        
        self.is_fitted = True
        
        total_features = sum(len(cats) for cats in self.categories_)
        self.logger.info(f"OneHotEncoder fitted with {total_features} output features from {self.n_features_in_} input features")
        
        return self
    
    def transform(self, X):
        """
        Transform features to one-hot encoding.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature data to encode
            
        Returns:
        --------
        X_encoded : array, shape (n_samples, n_encoded_features)
            One-hot encoded features
        """
        if not self.is_fitted:
            raise ValueError("OneHotEncoder must be fitted before transform")
        
        X = self._validate_input(X, y=None)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but OneHotEncoder was fitted with {self.n_features_in_} features")
        
        encoded_features = []
        
        for col_idx in range(X.shape[1]):
            categories = self.categories_[col_idx]
            col_data = X[:, col_idx]
            
            # Create one-hot encoding for this column
            col_encoded = np.zeros((len(col_data), len(categories)))
            
            for i, value in enumerate(col_data):
                if value in categories:
                    cat_idx = np.where(categories == value)[0][0]
                    col_encoded[i, cat_idx] = 1
                else:
                    if self.handle_unknown == 'error':
                        raise ValueError(f"Unknown category '{value}' in feature {col_idx}")
                    # If handle_unknown == 'ignore', leave as all zeros
            
            encoded_features.append(col_encoded)
        
        # Concatenate all encoded features
        X_encoded = np.concatenate(encoded_features, axis=1)
        
        self.logger.debug(f"Transformed {X.shape[0]} samples to {X_encoded.shape[1]} features")
        return X_encoded
    
    def fit_transform(self, X, y=None):
        """
        Fit OneHotEncoder and transform features.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature data to encode
        y : array-like, optional
            Ignored. This parameter exists only for compatibility.
            
        Returns:
        --------
        X_encoded : array, shape (n_samples, n_encoded_features)
            One-hot encoded features
        """
        return self.fit(X, y).transform(X)
    
    def inverse_transform(self, X_encoded):
        """
        Transform one-hot encoded data back to original representation.
        
        Parameters:
        -----------
        X_encoded : array-like, shape (n_samples, n_encoded_features)
            One-hot encoded data
            
        Returns:
        --------
        X_original : array, shape (n_samples, n_features)
            Original categorical data
        """
        if not self.is_fitted:
            raise ValueError("OneHotEncoder must be fitted before inverse_transform")
        
        X_encoded = np.array(X_encoded)
        
        expected_features = sum(len(cats) for cats in self.categories_)
        if X_encoded.shape[1] != expected_features:
            raise ValueError(f"X_encoded has {X_encoded.shape[1]} features, expected {expected_features}")
        
        X_original = []
        feature_start = 0
        
        for col_idx in range(self.n_features_in_):
            categories = self.categories_[col_idx]
            n_categories = len(categories)
            
            # Extract one-hot columns for this feature
            col_encoded = X_encoded[:, feature_start:feature_start + n_categories]
            
            # Find the category with maximum value (should be 1 for valid one-hot)
            category_indices = np.argmax(col_encoded, axis=1)
            col_original = categories[category_indices]
            
            X_original.append(col_original)
            feature_start += n_categories
        
        X_original = np.column_stack(X_original)
        
        self.logger.debug(f"Inverse transformed {X_encoded.shape[0]} samples")
        return X_original
    
    def get_feature_names_out(self, input_features=None):
        """
        Get feature names for the encoded output.
        
        Parameters:
        -----------
        input_features : array-like, optional
            Input feature names (ignored in this implementation)
            
        Returns:
        --------
        feature_names : array, shape (n_encoded_features,)
            Feature names for encoded output
        """
        if not self.is_fitted:
            raise ValueError("OneHotEncoder must be fitted before getting feature names")
        
        return np.array(self.feature_names_out_)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'sparse': self.sparse,
            'drop': self.drop,
            'handle_unknown': self.handle_unknown
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
        """String representation of the encoder."""
        if self.categories_ is not None:
            total_features = sum(len(cats) for cats in self.categories_)
            return f"OneHotEncoder(fitted=True, n_features_out={total_features})"
        else:
            return f"OneHotEncoder(fitted=False)"
    
    def __repr__(self):
        """Detailed string representation of the encoder."""
        return f"OneHotEncoder(sparse={self.sparse}, drop={self.drop}, handle_unknown='{self.handle_unknown}')"
