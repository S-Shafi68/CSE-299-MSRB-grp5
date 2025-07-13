"""
Ridge Regression Implementation

Ridge regression with L2 regularization for handling multicollinearity
and preventing overfitting.
"""

import numpy as np
from .base import BaseRegressor
from logger import get_logger


class Ridge(BaseRegressor):
    """
    Ridge Regression with L2 regularization.
    
    Ridge regression adds a penalty term (L2 regularization) to the standard
    linear regression loss function to prevent overfitting and handle multicollinearity.
    
    The objective function becomes:
    minimize: ||y - Xβ||² + α||β||²
    
    Parameters:
    -----------
    alpha : float, default=1.0
        Regularization strength. Higher values specify stronger regularization.
    fit_intercept : bool, default=True
        Whether to fit an intercept term.
    max_iter : int, default=1000
        Maximum number of iterations (for compatibility, not used in direct solution).
    tol : float, default=1e-6
        Tolerance for optimization (for compatibility, not used in direct solution).
    """
    
    def __init__(self, alpha=1.0, fit_intercept=True, max_iter=1000, tol=1e-6):
        super().__init__()
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        
        # Initialize parameters
        self.coef_ = None
        self.intercept_ = None
        self.n_features_in_ = None
        
        # Setup logging
        self.logger = get_logger(f"{self.__class__.__name__}")
        
    def fit(self, X, y):
        """
        Fit Ridge regression model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : Ridge
            Returns self for method chaining
        """
        # Validate input
        X, y = self._validate_input(X, y)
        self.n_features_in_ = X.shape[1]
        
        self.logger.info(f"Fitting Ridge regression with alpha={self.alpha}")
        self.logger.debug(f"Training data shape: {X.shape}")
        
        # Center the data if fitting intercept
        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            X_centered = X - X_mean
            y_centered = y - y_mean
        else:
            X_mean = np.zeros(X.shape[1])
            y_mean = 0.0
            X_centered = X.copy()
            y_centered = y.copy()
        
        # Ridge regression solution: β = (X^T X + αI)^(-1) X^T y
        try:
            # Create regularization matrix (don't regularize intercept)
            I = np.eye(X_centered.shape[1])
            
            # Compute X^T X + αI
            XtX_regularized = X_centered.T @ X_centered + self.alpha * I
            
            # Compute X^T y
            Xty = X_centered.T @ y_centered
            
            # Solve the regularized normal equation
            self.coef_ = np.linalg.solve(XtX_regularized, Xty)
            
            # Calculate intercept if needed
            if self.fit_intercept:
                self.intercept_ = y_mean - X_mean @ self.coef_
            else:
                self.intercept_ = 0.0
                
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse if singular matrix
            self.logger.warning("Singular matrix encountered, using pseudoinverse")
            if self.fit_intercept:
                X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
                # Add regularization to all coefficients except intercept
                reg_matrix = np.eye(X_with_intercept.shape[1])
                reg_matrix[0, 0] = 0  # Don't regularize intercept
                reg_matrix *= self.alpha
                
                XtX_reg = X_with_intercept.T @ X_with_intercept + reg_matrix
                Xty = X_with_intercept.T @ y
                
                coefficients = np.linalg.pinv(XtX_reg) @ Xty
                self.intercept_ = coefficients[0]
                self.coef_ = coefficients[1:]
            else:
                I = np.eye(X.shape[1])
                XtX_regularized = X.T @ X + self.alpha * I
                Xty = X.T @ y
                self.coef_ = np.linalg.pinv(XtX_regularized) @ Xty
                self.intercept_ = 0.0
        
        self.logger.info(f"Ridge regression fitted successfully")
        self.logger.debug(f"Coefficients: {self.coef_}")
        self.logger.debug(f"Intercept: {self.intercept_}")
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the Ridge model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Predicted values
        """
        # Check if model is fitted
        if self.coef_ is None:
            raise ValueError("Model must be fitted before making predictions")
        # Handle tuple input - IMPORTANT FIX
        if isinstance(X, tuple):
         X = X[0]  # Extract array from tuple
        
        # Validate input
        X = self._validate_input(X, y=None)
        
        # Check feature consistency
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but Ridge was fitted with {self.n_features_in_} features")
        
        # Make predictions: y = Xβ + intercept
        predictions = X @ self.coef_ + self.intercept_
        
        # Ensure we return a proper numpy array, not tuple
        return np.asarray(predictions).flatten()
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'alpha': self.alpha,
            'fit_intercept': self.fit_intercept,
            'max_iter': self.max_iter,
            'tol': self.tol
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
        """String representation of the model."""
        if self.coef_ is not None:
            return f"Ridge(alpha={self.alpha}, fitted=True, n_features={len(self.coef_)})"
        else:
            return f"Ridge(alpha={self.alpha}, fitted=False)"
    
    def __repr__(self):
        """Detailed string representation of the model."""
        return (f"Ridge(alpha={self.alpha}, fit_intercept={self.fit_intercept}, "
                f"max_iter={self.max_iter}, tol={self.tol})")
