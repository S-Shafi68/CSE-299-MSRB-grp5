"""
Principal Component Analysis (PCA) Implementation
"""

import numpy as np
from .base import BaseTransformer
from logger import get_logger

logger = get_logger("pca")

class PCA(BaseTransformer):
    """
    Principal Component Analysis for dimensionality reduction.
    
    Parameters:
    -----------
    n_components : int, float, or None, default=None
        Number of components to keep
        - If int: exact number of components
        - If float: proportion of variance to retain (0 < n_components < 1)
        - If None: keep all components
    whiten : bool, default=False
        Whether to whiten the components (scale to unit variance)
    random_state : int, default=None
        Random seed for reproducibility
    """
    
    def __init__(self, n_components=None, whiten=False, random_state=None):
        super().__init__()
        self.n_components = n_components
        self.whiten = whiten
        self.random_state = random_state
        
        # Fitted attributes
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.n_samples_ = None
        
        logger.debug(f"Initialized PCA with n_components={n_components}, whiten={whiten}")
    
    def _center_data(self, X):
        """Center the data by subtracting the mean."""
        self.mean_ = np.mean(X, axis=0)
        return X - self.mean_
    
    def _determine_n_components(self, n_features, explained_variance_ratio):
        """Determine the number of components to keep."""
        if self.n_components is None:
            return n_features
        elif isinstance(self.n_components, int):
            return min(self.n_components, n_features)
        elif isinstance(self.n_components, float):
            # Keep components that explain the desired variance
            cumsum_var = np.cumsum(explained_variance_ratio)
            n_comp = np.argmax(cumsum_var >= self.n_components) + 1
            return min(n_comp, n_features)
        else:
            raise ValueError("n_components must be int, float, or None")
    
    def fit(self, X, y=None):
        """
        Fit PCA on training data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, optional
            Ignored. Present for API consistency.
            
        Returns:
        --------
        self : PCA
            Returns self for method chaining
        """
        X = self._validate_input(X)
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        self.n_samples_, self.n_features_in_ = X.shape
        
        # Center the data
        X_centered = self._center_data(X)
        
        # Compute covariance matrix
        cov_matrix = np.cov(X_centered.T)
        
        # Perform eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store explained variance
        self.explained_variance_ = eigenvalues
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = eigenvalues / total_variance
        
        # Determine number of components to keep
        n_components = self._determine_n_components(
            self.n_features_in_, self.explained_variance_ratio_
        )
        
        # Keep only the selected components
        self.components_ = eigenvectors[:, :n_components].T
        self.explained_variance_ = self.explained_variance_[:n_components]
        self.explained_variance_ratio_ = self.explained_variance_ratio_[:n_components]
        
        # Calculate singular values (for sklearn compatibility)
        self.singular_values_ = np.sqrt(self.explained_variance_ * (self.n_samples_ - 1))
        
        # Update output dimensions
        self.n_features_out_ = n_components
        
        self.is_fitted = True
        logger.info(f"PCA fitted with {n_components} components explaining "
                   f"{np.sum(self.explained_variance_ratio_):.4f} of total variance")
        
        return self
    
    def transform(self, X):
        """
        Transform data to lower dimensional space.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to transform
            
        Returns:
        --------
        X_transformed : array, shape (n_samples, n_components)
            Transformed data
        """
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before transform")
        
        X = self._validate_input(X)
        
        # Check feature consistency
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but PCA was fitted with {self.n_features_in_} features")
        
        # Center the data using training mean
        X_centered = X - self.mean_
        
        # Project onto principal components
        X_transformed = X_centered @ self.components_.T
        
        # Apply whitening if requested
        if self.whiten:
            X_transformed /= np.sqrt(self.explained_variance_)
        
        logger.debug(f"Transformed {X.shape[0]} samples from {X.shape[1]} to {X_transformed.shape[1]} dimensions")
        return X_transformed
    
    def inverse_transform(self, X_transformed):
        """
        Transform data back to original space.
        
        Parameters:
        -----------
        X_transformed : array-like, shape (n_samples, n_components)
            Transformed data
            
        Returns:
        --------
        X_original : array, shape (n_samples, n_features)
            Data in original space
        """
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before inverse_transform")
        
        X_transformed = np.array(X_transformed)
        
        # Undo whitening if it was applied
        if self.whiten:
            X_transformed = X_transformed * np.sqrt(self.explained_variance_)
        
        # Project back to original space
        X_reconstructed = X_transformed @ self.components_
        
        # Add back the mean
        X_original = X_reconstructed + self.mean_
        
        return X_original
    
    def fit_transform(self, X, y=None):
        """Fit PCA and transform the data."""
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before getting feature names")
        
        n_components = self.components_.shape[0]
        return [f"PC{i+1}" for i in range(n_components)]
    
    def score(self, X, y=None):
        """
        Return the average log-likelihood of the data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test data
        y : array-like, optional
            Ignored
            
        Returns:
        --------
        score : float
            Average log-likelihood
        """
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before scoring")
        
        X = self._validate_input(X)
        X_transformed = self.transform(X)
        X_reconstructed = self.inverse_transform(X_transformed)
        
        # Calculate reconstruction error
        mse = np.mean((X - X_reconstructed) ** 2)
        return -mse  # Negative MSE as score (higher is better)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'n_components': self.n_components,
            'whiten': self.whiten,
            'random_state': self.random_state
        }
    
    def __str__(self):
        if self.is_fitted:
            return f"PCA(n_components={self.n_features_out_}, explained_variance={np.sum(self.explained_variance_ratio_):.4f})"
        else:
            return f"PCA(n_components={self.n_components}, fitted=False)"
    
    def __repr__(self):
        return f"PCA(n_components={self.n_components}, whiten={self.whiten})"
