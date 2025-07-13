"""
Linear Regression implementation for the ML library recreation project.
Implements ordinary least squares linear regression from scratch.
"""

import numpy as np
from .base import BaseRegressor  # Changed from models.base



class LinearRegression(BaseRegressor):
    """
    Linear Regression using Ordinary Least Squares.
    
    Fits a linear model with coefficients w = (w1, ..., wp) to minimize
    the residual sum of squares between the observed targets in the dataset,
    and the targets predicted by the linear approximation.
    
    Parameters:
    -----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    """
    
    def __init__(self, fit_intercept=True):
        """
        Initialize Linear Regression model.
        
        Args:
            fit_intercept (bool): Whether to fit intercept term
        """
        super().__init__()
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self.n_features_in_ = None
        
    def fit(self, X, y):
        """
        Fit linear model using ordinary least squares.
        
        Args:
            X (array-like): Training features of shape (n_samples, n_features)
            y (array-like): Training targets of shape (n_samples,)
            
        Returns:
            self: Returns self for method chaining
        """
        # Validate input
        X, y = self._validate_input(X, y)
        
        # Store number of features
        self.n_features_in_ = X.shape[1]
        
        self.logger.info(f"Fitting Linear Regression on {X.shape[0]} samples "
                        f"with {X.shape[1]} features")
        
        if self.fit_intercept:
            # Add bias column (column of ones) for intercept
            X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
            
            # Solve normal equation: (X^T X)^-1 X^T y
            try:
                # Use pseudoinverse for numerical stability
                coefficients = np.linalg.pinv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
                
                # Extract intercept and coefficients
                self.intercept_ = coefficients[0]
                self.coef_ = coefficients[1:]
                
            except np.linalg.LinAlgError:
                # Fallback to direct pseudoinverse
                self.logger.warning("Using pseudoinverse fallback due to singular matrix")
                coefficients = np.linalg.pinv(X_with_bias) @ y
                self.intercept_ = coefficients[0]
                self.coef_ = coefficients[1:]
        else:
            # No intercept case
            self.intercept_ = 0.0
            
            try:
                # Solve: (X^T X)^-1 X^T y
                self.coef_ = np.linalg.pinv(X.T @ X) @ X.T @ y
            except np.linalg.LinAlgError:
                # Fallback to direct pseudoinverse
                self.logger.warning("Using pseudoinverse fallback due to singular matrix")
                self.coef_ = np.linalg.pinv(X) @ y
        
        # Mark as fitted
        self.is_fitted = True
        
        self.logger.info(f"Model fitted successfully. Intercept: {self.intercept_:.4f}")
        self.logger.debug(f"Coefficients: {self.coef_}")
        
        return self
    
    def predict(self, X):
        """
        Predict using the linear model.
        
        Args:
            X (array-like): Samples of shape (n_samples, n_features)
            
        Returns:
            array: Predicted values of shape (n_samples,)
        """
        # Check if model is fitted
        self._check_fitted()
        
        # Validate input
        X, _ = self._validate_input(X)
        
        # Check feature consistency
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but model was trained "
                           f"with {self.n_features_in_} features")
        
        # Make predictions: y = X * coef + intercept
        predictions = X @ self.coef_
        
        if self.fit_intercept:
            predictions += self.intercept_
            
        return predictions
    
    def get_coefficients(self):
        """
        Get the model coefficients.
        
        Returns:
            dict: Dictionary with 'coef' and 'intercept' keys
        """
        self._check_fitted()
        return {
            'coef': self.coef_.copy() if self.coef_ is not None else None,
            'intercept': self.intercept_
        }
    
    def score(self, X, y):
        """
        Return the coefficient of determination R² of the prediction.
        
        Args:
            X (array-like): Test samples
            y (array-like): True values for X
            
        Returns:
            float: R² score
        """
        # Use parent class implementation
        return super().score(X, y)
    
    def _residuals(self, X, y):
        """
        Calculate residuals (y_true - y_pred).
        
        Args:
            X (array-like): Features
            y (array-like): True targets
            
        Returns:
            array: Residuals
        """
        self._check_fitted()
        X, y = self._validate_input(X, y)
        
        y_pred = self.predict(X)
        return y - y_pred
    
    def mean_squared_error(self, X, y):
        """
        Calculate mean squared error.
        
        Args:
            X (array-like): Features
            y (array-like): True targets
            
        Returns:
            float: Mean squared error
        """
        residuals = self._residuals(X, y)
        return np.mean(residuals ** 2)
    
    def mean_absolute_error(self, X, y):
        """
        Calculate mean absolute error.
        
        Args:
            X (array-like): Features
            y (array-like): True targets
            
        Returns:
            float: Mean absolute error
        """
        residuals = self._residuals(X, y)
        return np.mean(np.abs(residuals))
    
    def __str__(self):
        """String representation of the model."""
        if not self.is_fitted:
            return "LinearRegression(not fitted)"
        
        coef_str = ", ".join([f"{c:.4f}" for c in self.coef_])
        return f"LinearRegression(intercept={self.intercept_:.4f}, coef=[{coef_str}])"
    
    def __repr__(self):
        """Detailed representation of the model."""
        return f"LinearRegression(fit_intercept={self.fit_intercept})"