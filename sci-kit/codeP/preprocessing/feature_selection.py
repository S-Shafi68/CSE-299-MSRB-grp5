"""
Feature Selection Utilities
"""

import numpy as np
from models.base import BaseTransformer
from logger import get_logger

logger = get_logger("feature_selection")

class SelectKBest(BaseTransformer):
    """
    Select k best features based on univariate statistical tests.
    
    Parameters:
    -----------
    score_func : callable, default=f_classif
        Function to compute univariate scores
    k : int, default=10
        Number of top features to select
    """
    
    def __init__(self, score_func=None, k=10):
        super().__init__()
        self.score_func = score_func or self._f_classif
        self.k = k
        
        # Fitted attributes
        self.scores_ = None
        self.pvalues_ = None
        self.selected_features_ = None
        
        logger.debug(f"Initialized SelectKBest with k={k}")
    
    def _f_classif(self, X, y):
        """F-score for classification (ANOVA F-test)."""
        classes = np.unique(y)
        n_classes = len(classes)
        n_samples, n_features = X.shape
        
        scores = np.zeros(n_features)
        
        for i in range(n_features):
            feature = X[:, i]
            
            # Calculate between-class and within-class variance
            overall_mean = np.mean(feature)
            
            # Between-class sum of squares
            ss_between = 0
            for class_label in classes:
                class_mask = y == class_label
                class_mean = np.mean(feature[class_mask])
                class_size = np.sum(class_mask)
                ss_between += class_size * (class_mean - overall_mean) ** 2
            
            # Within-class sum of squares
            ss_within = 0
            for class_label in classes:
                class_mask = y == class_label
                class_feature = feature[class_mask]
                class_mean = np.mean(class_feature)
                ss_within += np.sum((class_feature - class_mean) ** 2)
            
            # F-statistic
            ms_between = ss_between / (n_classes - 1)
            ms_within = ss_within / (n_samples - n_classes)
            
            if ms_within > 0:
                scores[i] = ms_between / ms_within
            else:
                scores[i] = 0
        
        return scores
    
    def fit(self, X, y):
        """
        Fit the feature selector.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : SelectKBest
            Returns self for method chaining
        """
        X, y = self._validate_input(X, y)
        
        # Calculate feature scores
        self.scores_ = self.score_func(X, y)
        
        # Select k best features
        k_best = min(self.k, X.shape[1])
        self.selected_features_ = np.argsort(self.scores_)[-k_best:]
        
        self.n_features_out_ = k_best
        self.is_fitted = True
        
        logger.info(f"Selected {k_best} best features out of {X.shape[1]}")
        
        return self
    
    def transform(self, X):
        """
        Transform data by selecting k best features.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to transform
            
        Returns:
        --------
        X_selected : array, shape (n_samples, k)
            Transformed data with selected features
        """
        if not self.is_fitted:
            raise ValueError("SelectKBest must be fitted before transform")
        
        X = self._validate_input(X)
        return X[:, self.selected_features_]
    
    def get_support(self, indices=False):
        """
        Get a mask or integer index of selected features.
        
        Parameters:
        -----------
        indices : bool, default=False
            If True, return feature indices. If False, return boolean mask.
            
        Returns:
        --------
        support : array
            Boolean mask or integer indices of selected features
        """
        if not self.is_fitted:
            raise ValueError("SelectKBest must be fitted before getting support")
        
        if indices:
            return self.selected_features_
        else:
            mask = np.zeros(self.n_features_in_, dtype=bool)
            mask[self.selected_features_] = True
            return mask

class RFE(BaseTransformer):
    """
    Recursive Feature Elimination.
    
    Parameters:
    -----------
    estimator : object
        The base estimator from which to perform feature elimination
    n_features_to_select : int, default=None
        Number of features to select
    step : int, default=1
        Number of features to remove at each iteration
    """
    
    def __init__(self, estimator, n_features_to_select=None, step=1):
        super().__init__()
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        
        # Fitted attributes
        self.support_ = None
        self.ranking_ = None
        
        logger.debug(f"Initialized RFE with n_features_to_select={n_features_to_select}")
    
    def fit(self, X, y):
        """
        Fit the RFE feature selector.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : RFE
            Returns self for method chaining
        """
        X, y = self._validate_input(X, y)
        
        n_features = X.shape[1]
        if self.n_features_to_select is None:
            n_features_to_select = n_features // 2
        else:
            n_features_to_select = self.n_features_to_select
        
        # Initialize support and ranking
        support = np.ones(n_features, dtype=bool)
        ranking = np.ones(n_features, dtype=int)
        
        # Recursive elimination
        n_remaining = n_features
        current_ranking = 1
        
        while n_remaining > n_features_to_select:
            # Train estimator on current features
            estimator_clone = type(self.estimator)(**self.estimator.get_params())
            estimator_clone.fit(X[:, support], y)
            
            # Get feature importance
            if hasattr(estimator_clone, 'feature_importances_'):
                importances = estimator_clone.feature_importances_
            elif hasattr(estimator_clone, 'coef_'):
                importances = np.abs(estimator_clone.coef_)
                if importances.ndim > 1:
                    importances = np.mean(importances, axis=0)
            else:
                raise ValueError("Estimator must have feature_importances_ or coef_ attribute")
            
            # Remove least important features
            n_to_remove = min(self.step, n_remaining - n_features_to_select)
            worst_features = np.argsort(importances)[:n_to_remove]
            
            # Update support and ranking
            current_features = np.where(support)[0]
            for idx in worst_features:
                feature_idx = current_features[idx]
                support[feature_idx] = False
                ranking[feature_idx] = current_ranking
            
            n_remaining -= n_to_remove
            current_ranking += 1
        
        self.support_ = support
        self.ranking_ = ranking
        self.n_features_out_ = n_features_to_select
        self.is_fitted = True
        
        logger.info(f"RFE selected {n_features_to_select} features out of {n_features}")
        
        return self
    
    def transform(self, X):
        """
        Transform data by selecting features.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to transform
            
        Returns:
        --------
        X_selected : array, shape (n_samples, n_features_to_select)
            Transformed data with selected features
        """
        if not self.is_fitted:
            raise ValueError("RFE must be fitted before transform")
        
        X = self._validate_input(X)
        return X[:, self.support_]
    
    def get_support(self, indices=False):
        """Get a mask or integer index of selected features."""
        if not self.is_fitted:
            raise ValueError("RFE must be fitted before getting support")
        
        if indices:
            return np.where(self.support_)[0]
        else:
            return self.support_
