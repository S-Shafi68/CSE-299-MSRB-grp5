"""
Random Forest Implementation for Classification
"""

import numpy as np
from .base import BaseClassifier
from .decision_tree import DecisionTreeClassifier
from logger import get_logger

logger = get_logger("random_forest")

class RandomForestClassifier(BaseClassifier):
    """
    Random Forest Classifier implementation.
    
    Parameters:
    -----------
    n_estimators : int, default=100
        Number of trees in the forest
    max_depth : int, default=None
        Maximum depth of trees
    min_samples_split : int, default=2
        Minimum samples required to split a node
    min_samples_leaf : int, default=1
        Minimum samples required in a leaf node
    max_features : str or int, default='sqrt'
        Number of features to consider at each split
    bootstrap : bool, default=True
        Whether to use bootstrap sampling
    random_state : int, default=None
        Random seed for reproducibility
    """
    
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features='sqrt', bootstrap=True,
                 random_state=None):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        
        # Model attributes
        self.estimators_ = []
        self.feature_importances_ = None
        self.n_features_in_ = None
        self.oob_score_ = None
        
        # Validate parameters
        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1")
        if max_features not in ['sqrt', 'log2', None] and not isinstance(max_features, int):
            raise ValueError("max_features must be 'sqrt', 'log2', None, or an integer")
        
        logger.debug(f"Initialized RandomForestClassifier with n_estimators={n_estimators}")
    
    def _get_max_features(self, n_features):
        """Get the number of features to consider at each split."""
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features))
        elif self.max_features is None:
            return n_features
        else:
            return min(self.max_features, n_features)
    
    def _bootstrap_sample(self, X, y):
        """Create a bootstrap sample of the data."""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices], indices
    
    def _get_feature_subset(self, n_features):
        """Get a random subset of features."""
        max_features = self._get_max_features(n_features)
        return np.random.choice(n_features, size=max_features, replace=False)
    
    def fit(self, X, y):
        """
        Fit the random forest classifier.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        """
        X, y = self._validate_input(X, y)
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.estimators_ = []
        
        # Initialize feature importances
        self.feature_importances_ = np.zeros(self.n_features_in_)
        
        # For OOB score calculation
        oob_predictions = np.zeros((X.shape[0], len(self.classes_)))
        oob_counts = np.zeros(X.shape[0])
        
        # Build each tree
        for i in range(self.n_estimators):
            # Create bootstrap sample
            if self.bootstrap:
                X_bootstrap, y_bootstrap, bootstrap_indices = self._bootstrap_sample(X, y)
                # Track out-of-bag samples
                oob_mask = np.ones(X.shape[0], dtype=bool)
                oob_mask[bootstrap_indices] = False
                oob_indices = np.where(oob_mask)[0]
            else:
                X_bootstrap, y_bootstrap = X, y
                oob_indices = []
            
            # Create and fit tree
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                criterion='gini',
                random_state=self.random_state + i if self.random_state else None
            )
            
            tree.fit(X_bootstrap, y_bootstrap)
            self.estimators_.append(tree)
            
            # Accumulate feature importances
            self.feature_importances_ += tree.feature_importances_
            
            # Calculate OOB predictions
            if self.bootstrap and len(oob_indices) > 0:
                oob_pred = tree.predict(X[oob_indices])
                for j, idx in enumerate(oob_indices):
                    class_idx = np.where(self.classes_ == oob_pred[j])[0][0]
                    oob_predictions[idx, class_idx] += 1
                    oob_counts[idx] += 1
        
        # Normalize feature importances
        self.feature_importances_ /= self.n_estimators
        
        # Calculate OOB score
        if self.bootstrap:
            oob_mask = oob_counts > 0
            if np.any(oob_mask):
                oob_predictions_norm = oob_predictions[oob_mask] / oob_counts[oob_mask, np.newaxis]
                oob_pred_classes = self.classes_[np.argmax(oob_predictions_norm, axis=1)]
                self.oob_score_ = np.mean(oob_pred_classes == y[oob_mask])
        
        self.is_fitted = True
        logger.info(f"Random forest fitted with {self.n_estimators} trees")
        
        return self
    
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
            raise ValueError("Random forest must be fitted before making predictions")
        
        X = self._validate_input(X)
        
        # Get predictions from all trees
        predictions = np.zeros((X.shape[0], len(self.classes_)))
        
        for tree in self.estimators_:
            tree_pred = tree.predict(X)
            for i, pred in enumerate(tree_pred):
                class_idx = np.where(self.classes_ == pred)[0][0]
                predictions[i, class_idx] += 1
        
        # Return majority vote
        final_predictions = self.classes_[np.argmax(predictions, axis=1)]
        
        logger.debug(f"Made predictions for {len(X)} samples using {len(self.estimators_)} trees")
        return final_predictions
    
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
            raise ValueError("Random forest must be fitted before making predictions")
        
        X = self._validate_input(X)
        
        # Get predictions from all trees
        predictions = np.zeros((X.shape[0], len(self.classes_)))
        
        for tree in self.estimators_:
            tree_pred = tree.predict(X)
            for i, pred in enumerate(tree_pred):
                class_idx = np.where(self.classes_ == pred)[0][0]
                predictions[i, class_idx] += 1
        
        # Normalize to get probabilities
        probabilities = predictions / self.n_estimators
        
        return probabilities
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'random_state': self.random_state
        }
    
    def __str__(self):
        if self.is_fitted:
            return f"RandomForestClassifier(n_estimators={self.n_estimators}, fitted=True, oob_score={self.oob_score_:.4f if self.oob_score_ else None})"
        else:
            return f"RandomForestClassifier(n_estimators={self.n_estimators}, fitted=False)"
    
    def __repr__(self):
        return f"RandomForestClassifier(n_estimators={self.n_estimators}, max_depth={self.max_depth})"
