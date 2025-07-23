"""
Logistic Regression Implementation
"""

import numpy as np
from .base import BaseClassifier  # Changed from models.base to .base
from logger import get_logger

logger = get_logger("logistic_regression")

class LogisticRegression(BaseClassifier):
    """
    Logistic Regression classifier implementation.
    
    Parameters:
    -----------
    learning_rate : float, default=0.01
        Learning rate for gradient descent
    max_iter : int, default=1000
        Maximum number of iterations
    tolerance : float, default=1e-6
        Tolerance for stopping criterion
    fit_intercept : bool, default=True
        Whether to fit intercept term
    """
    
    def __init__(self, learning_rate=0.01, max_iter=1000, tolerance=1e-6, fit_intercept=True):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.fit_intercept = fit_intercept
        
        # Initialize model parameters
        self.coef_ = None
        self.classes_ = None
        self.multiclass = False
        self.is_fitted = False
        
        logger.debug(f"Initialized LogisticRegression with lr={learning_rate}, max_iter={max_iter}")
    
    def _sigmoid(self, z):
        """Sigmoid activation function with numerical stability."""
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _compute_cost(self, X, y, coef):
        """Compute logistic regression cost function."""
        m = X.shape[0]
        z = X @ coef
        predictions = self._sigmoid(z)
        
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        
        cost = -(1/m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        return cost
    
    def _compute_gradient(self, X, y, coef):
        """Compute gradient for logistic regression."""
        m = X.shape[0]
        z = X @ coef
        predictions = self._sigmoid(z)
        gradient = (1/m) * X.T @ (predictions - y)
        return gradient
    
    def _fit_binary(self, X, y):
        """Fit binary logistic regression using gradient descent."""
        n_features = X.shape[1]
        coef = np.zeros(n_features)
        
        for iteration in range(self.max_iter):
            # Compute cost and gradient
            cost = self._compute_cost(X, y, coef)
            gradient = self._compute_gradient(X, y, coef)
            
            # Update coefficients
            new_coef = coef - self.learning_rate * gradient
            
            # Check for convergence
            if np.linalg.norm(new_coef - coef) < self.tolerance:
                logger.debug(f"Converged after {iteration + 1} iterations")
                break
            
            coef = new_coef
        
        return coef
    
    def fit(self, X, y):
        """
        Fit the logistic regression model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        """
        X, y = self._validate_input(X, y)
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Add intercept term if needed
        if self.fit_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])
        
        if n_classes == 2:
            # Binary classification
            self.multiclass = False
            y_binary = np.where(y == self.classes_[0], 0, 1)
            self.coef_ = self._fit_binary(X, y_binary)
        else:
            # Multiclass classification (one-vs-rest)
            self.multiclass = True
            self.coef_ = []
            for class_label in self.classes_:
                y_binary = np.where(y == class_label, 1, 0)
                coef = self._fit_binary(X, y_binary)
                self.coef_.append(coef)
            self.coef_ = np.array(self.coef_)
        
        self.is_fitted = True
        logger.info(f"Logistic regression fitted for {n_classes} classes")
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test data
            
        Returns:
        --------
        probabilities : array, shape (n_samples, n_classes)
            Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = self._validate_input(X)
        
        # Add intercept term if fit_intercept is True
        if self.fit_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])
        
        if self.multiclass:
            # One-vs-rest multiclass prediction
            probabilities = []
            for i, coef in enumerate(self.coef_):
                # Ensure coefficient dimensions match input
                if coef.shape[0] != X.shape[1]:
                    raise ValueError(f"Feature dimension mismatch: expected {coef.shape[0]}, got {X.shape[1]}")
                
                logits = X @ coef
                probs = self._sigmoid(logits)
                probabilities.append(probs)
            
            probabilities = np.column_stack(probabilities)
            # Normalize probabilities to sum to 1
            probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
            
        else:
            # Binary classification
            if self.coef_.shape[0] != X.shape[1]:
                raise ValueError(f"Feature dimension mismatch: expected {self.coef_.shape[0]}, got {X.shape[1]}")
            
            logits = X @ self.coef_
            prob_positive = self._sigmoid(logits)
            prob_negative = 1 - prob_positive
            probabilities = np.column_stack([prob_negative, prob_positive])
        
        return probabilities
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test data
            
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Predicted class labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        probabilities = self.predict_proba(X)
        
        if self.multiclass:
            predictions = np.argmax(probabilities, axis=1)
            return np.array([self.classes_[idx] for idx in predictions])
        else:
            predictions = np.argmax(probabilities, axis=1)
            return np.array([self.classes_[idx] for idx in predictions])
