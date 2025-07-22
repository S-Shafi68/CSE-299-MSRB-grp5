"""
Support Vector Machine Classifier Implementation with Fixed Broadcasting
"""

import numpy as np
from .base import BaseClassifier
from logger import get_logger

logger = get_logger("svm")

class SupportVectorMachine(BaseClassifier):
    """
    Support Vector Machine classifier with multiclass support and fixed broadcasting.
    """
    
    def __init__(self, C=1.0, kernel='linear', gamma='scale', degree=3, max_iter=1000, tolerance=1e-3):
        super().__init__()
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.max_iter = max_iter
        self.tolerance = tolerance
        
        # Model parameters
        self.classes_ = None
        self.n_features_in_ = None
        self.multiclass = False
        self.binary_classifiers = []
        
        # Binary classification parameters
        self.alpha = None
        self.b = None
        self.X_support = None
        self.y_support = None
        self.support_indices = None
        
        # Validate parameters
        if C <= 0:
            raise ValueError("C must be positive")
        if kernel not in ['linear', 'rbf', 'poly']:
            raise ValueError("kernel must be 'linear', 'rbf', or 'poly'")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        
        logger.debug(f"Initialized SVM with C={C}, kernel={kernel}, gamma={gamma}")
    
    def _kernel_function(self, x1, x2):
        """Compute kernel function between two vectors."""
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'rbf':
            if self.gamma == 'scale':
                gamma = 1.0 / (x1.shape[0] * np.var(x1) + 1e-8)
            else:
                gamma = float(self.gamma)
            return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)
        elif self.kernel == 'poly':
            if self.gamma == 'scale':
                gamma = 1.0 / (x1.shape[0] * np.var(x1) + 1e-8)
            else:
                gamma = float(self.gamma)
            return (gamma * np.dot(x1, x2) + 1) ** self.degree
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def _compute_kernel_matrix(self, X1, X2=None):
        """Compute kernel matrix between two sets of vectors."""
        if X2 is None:
            X2 = X1
        
        n1, n2 = X1.shape[0], X2.shape[0]
        K = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self._kernel_function(X1[i], X2[j])
        
        return K
    
    def _simplified_smo(self, X, y):
        """Simplified Sequential Minimal Optimization algorithm."""
        n_samples = X.shape[0]
        alpha = np.zeros(n_samples)
        b = 0.0
        K = self._compute_kernel_matrix(X)
        
        for iteration in range(self.max_iter):
            alpha_prev = alpha.copy()
            
            for i in range(n_samples):
                # Fixed: Proper matrix multiplication for prediction
                prediction = np.dot(alpha * y, K[i]) + b
                error_i = prediction - y[i]
                
                if (y[i] * error_i < -self.tolerance and alpha[i] < self.C) or \
                   (y[i] * error_i > self.tolerance and alpha[i] > 0):
                    
                    j = np.random.choice([idx for idx in range(n_samples) if idx != i])
                    prediction_j = np.dot(alpha * y, K[j]) + b
                    error_j = prediction_j - y[j]
                    
                    alpha_i_old, alpha_j_old = alpha[i], alpha[j]
                    
                    if y[i] != y[j]:
                        L = max(0, alpha[j] - alpha[i])
                        H = min(self.C, self.C + alpha[j] - alpha[i])
                    else:
                        L = max(0, alpha[i] + alpha[j] - self.C)
                        H = min(self.C, alpha[i] + alpha[j])
                    
                    if L == H:
                        continue
                    
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue
                    
                    alpha[j] = alpha[j] - (y[j] * (error_i - error_j)) / eta
                    alpha[j] = max(L, min(H, alpha[j]))
                    
                    if abs(alpha[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    alpha[i] = alpha[i] + y[i] * y[j] * (alpha_j_old - alpha[j])
                    
                    b1 = b - error_i - y[i] * (alpha[i] - alpha_i_old) * K[i, i] - \
                         y[j] * (alpha[j] - alpha_j_old) * K[i, j]
                    b2 = b - error_j - y[i] * (alpha[i] - alpha_i_old) * K[i, j] - \
                         y[j] * (alpha[j] - alpha_j_old) * K[j, j]
                    
                    if 0 < alpha[i] < self.C:
                        b = b1
                    elif 0 < alpha[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2
            
            if np.allclose(alpha, alpha_prev, atol=self.tolerance):
                logger.debug(f"SVM converged after {iteration + 1} iterations")
                break
        
        return alpha, b
    
    def fit(self, X, y):
        """Fit the SVM classifier with multiclass support."""
        X, y = self._validate_input(X, y)
        
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        
        if len(self.classes_) == 2:
            # Binary classification
            self.multiclass = False
            y_binary = np.where(y == self.classes_[0], -1, 1)
            self.alpha, self.b = self._simplified_smo(X, y_binary)
            
            support_mask = self.alpha > 1e-5
            self.support_indices = np.where(support_mask)[0]
            self.X_support = X[support_mask]
            self.y_support = y_binary[support_mask]
            self.alpha = self.alpha[support_mask]
            
            logger.info(f"Binary SVM fitted with {len(self.support_indices)} support vectors")
            
        else:
            # Multiclass classification (One-vs-Rest)
            self.multiclass = True
            self.binary_classifiers = []
            
            for class_label in self.classes_:
                logger.debug(f"Training binary classifier for class {class_label}")
                
                y_binary = np.where(y == class_label, 1, -1)
                alpha, b = self._simplified_smo(X, y_binary)
                
                support_mask = alpha > 1e-5
                binary_classifier = {
                    'alpha': alpha[support_mask],
                    'b': b,
                    'X_support': X[support_mask],
                    'y_support': y_binary[support_mask],
                    'support_indices': np.where(support_mask)[0],
                    'class_label': class_label
                }
                self.binary_classifiers.append(binary_classifier)
            
            total_sv = sum(len(clf['support_indices']) for clf in self.binary_classifiers)
            logger.info(f"Multiclass SVM fitted with {total_sv} total support vectors for {len(self.classes_)} classes")
        
        self.is_fitted = True
        return self
    
    def _decision_function_binary(self, X, classifier=None):
        """Compute decision function for binary classification with fixed broadcasting."""
        if classifier is None:
            # Use main binary classifier
            K = self._compute_kernel_matrix(X, self.X_support)
            # Fixed: Proper matrix multiplication to avoid broadcasting error
            decision = np.array([np.dot(self.alpha * self.y_support, K[i]) + self.b for i in range(K.shape[0])])
        else:
            # Use specific binary classifier
            K = self._compute_kernel_matrix(X, classifier['X_support'])
            # Fixed: Proper matrix multiplication to avoid broadcasting error
            decision = np.array([np.dot(classifier['alpha'] * classifier['y_support'], K[i]) + classifier['b'] 
                               for i in range(K.shape[0])])
        
        return decision
    
    def predict(self, X):
        """Make predictions on new data."""
        if not self.is_fitted:
            raise ValueError("SVM must be fitted before making predictions")
        
        X = self._validate_input(X)
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but SVM was fitted with {self.n_features_in_} features")
        
        if not self.multiclass:
            # Binary classification
            decision = self._decision_function_binary(X)
            predictions = np.where(decision >= 0, self.classes_[1], self.classes_[0])
        else:
            # Multiclass classification (One-vs-Rest)
            decision_scores = []
            
            for classifier in self.binary_classifiers:
                decision = self._decision_function_binary(X, classifier)
                decision_scores.append(decision)
            
            decision_scores = np.array(decision_scores).T
            predicted_indices = np.argmax(decision_scores, axis=1)
            predictions = np.array([self.classes_[idx] for idx in predicted_indices])
        
        logger.debug(f"Made predictions for {len(X)} samples")
        return predictions
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("SVM must be fitted before making predictions")
        
        X = self._validate_input(X)
        
        if not self.multiclass:
            # Binary classification
            decision = self._decision_function_binary(X)
            # Apply sigmoid to convert decision values to probabilities
            prob_positive = 1 / (1 + np.exp(-decision))
            prob_negative = 1 - prob_positive
            return np.column_stack([prob_negative, prob_positive])
        else:
            # Multiclass classification
            decision_scores = []
            
            for classifier in self.binary_classifiers:
                decision = self._decision_function_binary(X, classifier)
                # Apply sigmoid to convert decision values to probabilities
                prob = 1 / (1 + np.exp(-decision))
                decision_scores.append(prob)
            
            probabilities = np.array(decision_scores).T
            # Normalize probabilities to sum to 1
            row_sums = probabilities.sum(axis=1, keepdims=True)
            probabilities = probabilities / (row_sums + 1e-8)  # Add small epsilon to avoid division by zero
            
            return probabilities
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'C': self.C,
            'kernel': self.kernel,
            'gamma': self.gamma,
            'degree': self.degree,
            'max_iter': self.max_iter,
            'tolerance': self.tolerance
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
        """String representation of the classifier."""
        if self.is_fitted:
            if self.multiclass:
                return f"SupportVectorMachine(C={self.C}, kernel='{self.kernel}', fitted=True, multiclass=True)"
            else:
                return f"SupportVectorMachine(C={self.C}, kernel='{self.kernel}', fitted=True, n_support={len(self.support_indices)})"
        else:
            return f"SupportVectorMachine(C={self.C}, kernel='{self.kernel}', fitted=False)"
    
    def __repr__(self):
        """Detailed string representation of the classifier."""
        return f"SupportVectorMachine(C={self.C}, kernel='{self.kernel}', gamma={self.gamma}, degree={self.degree})"
