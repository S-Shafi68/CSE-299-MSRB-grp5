"""
Decision Tree Implementation for Classification and Regression
"""

import numpy as np
from .base import BaseClassifier, BaseRegressor
from logger import get_logger

logger = get_logger("decision_tree")

class DecisionTreeNode:
    """Node class for decision tree."""
    
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, 
                 value=None, samples=None, impurity=None):
        self.feature_idx = feature_idx  # Index of feature to split on
        self.threshold = threshold      # Threshold value for split
        self.left = left               # Left child node
        self.right = right             # Right child node
        self.value = value             # Value for leaf nodes
        self.samples = samples         # Number of samples in node
        self.impurity = impurity       # Impurity measure of node

class DecisionTreeClassifier(BaseClassifier):
    """
    Decision Tree Classifier implementation.
    
    Parameters:
    -----------
    max_depth : int, default=None
        Maximum depth of the tree
    min_samples_split : int, default=2
        Minimum samples required to split a node
    min_samples_leaf : int, default=1
        Minimum samples required in a leaf node
    criterion : str, default='gini'
        Splitting criterion ('gini' or 'entropy')
    random_state : int, default=None
        Random seed for reproducibility
    """
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 criterion='gini', random_state=None):
        super().__init__()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.random_state = random_state
        
        # Model attributes
        self.tree_ = None
        self.feature_importances_ = None
        self.n_features_in_ = None
        
        # Validate parameters
        if criterion not in ['gini', 'entropy']:
            raise ValueError("criterion must be 'gini' or 'entropy'")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1")
        
        logger.debug(f"Initialized DecisionTreeClassifier with criterion={criterion}")
    
    def _calculate_gini(self, y):
        """Calculate Gini impurity."""
        if len(y) == 0:
            return 0
        
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)
    
    def _calculate_entropy(self, y):
        """Calculate entropy."""
        if len(y) == 0:
            return 0
        
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    def _calculate_impurity(self, y):
        """Calculate impurity based on criterion."""
        if self.criterion == 'gini':
            return self._calculate_gini(y)
        else:  # entropy
            return self._calculate_entropy(y)
    
    def _find_best_split(self, X, y):
        """Find the best split for the current node."""
        best_gain = 0
        best_feature_idx = None
        best_threshold = None
        
        n_samples, n_features = X.shape
        current_impurity = self._calculate_impurity(y)
        
        for feature_idx in range(n_features):
            # Get unique values for this feature
            feature_values = np.unique(X[:, feature_idx])
            
            # Try each unique value as threshold
            for threshold in feature_values:
                # Split data
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                # Calculate weighted impurity after split
                left_impurity = self._calculate_impurity(y[left_mask])
                right_impurity = self._calculate_impurity(y[right_mask])
                
                left_weight = np.sum(left_mask) / n_samples
                right_weight = np.sum(right_mask) / n_samples
                
                weighted_impurity = (left_weight * left_impurity + 
                                   right_weight * right_impurity)
                
                # Calculate information gain
                gain = current_impurity - weighted_impurity
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold
        
        return best_feature_idx, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree."""
        n_samples, n_features = X.shape
        
        # Check stopping conditions
        if (self.max_depth is not None and depth >= self.max_depth or
            n_samples < self.min_samples_split or
            len(np.unique(y)) == 1):
            
            # Create leaf node
            leaf_value = np.bincount(y).argmax()  # Most common class
            return DecisionTreeNode(value=leaf_value, samples=n_samples,
                                  impurity=self._calculate_impurity(y))
        
        # Find best split
        best_feature_idx, best_threshold, best_gain = self._find_best_split(X, y)
        
        if best_feature_idx is None or best_gain == 0:
            # No good split found, create leaf
            leaf_value = np.bincount(y).argmax()
            return DecisionTreeNode(value=leaf_value, samples=n_samples,
                                  impurity=self._calculate_impurity(y))
        
        # Split data
        left_mask = X[:, best_feature_idx] <= best_threshold
        right_mask = ~left_mask
        
        # Check minimum samples per leaf
        if (np.sum(left_mask) < self.min_samples_leaf or 
            np.sum(right_mask) < self.min_samples_leaf):
            leaf_value = np.bincount(y).argmax()
            return DecisionTreeNode(value=leaf_value, samples=n_samples,
                                  impurity=self._calculate_impurity(y))
        
        # Recursively build left and right subtrees
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return DecisionTreeNode(feature_idx=best_feature_idx, threshold=best_threshold,
                              left=left_child, right=right_child, samples=n_samples,
                              impurity=self._calculate_impurity(y))
    
    def _calculate_feature_importances(self, node, total_samples):
        """Calculate feature importances recursively."""
        if node.feature_idx is None:  # Leaf node
            return
        
        # Calculate importance for current node
        left_samples = node.left.samples if node.left else 0
        right_samples = node.right.samples if node.right else 0
        
        importance = (node.samples / total_samples) * node.impurity
        if node.left:
            importance -= (left_samples / total_samples) * node.left.impurity
        if node.right:
            importance -= (right_samples / total_samples) * node.right.impurity
        
        self.feature_importances_[node.feature_idx] += importance
        
        # Recursively calculate for children
        if node.left:
            self._calculate_feature_importances(node.left, total_samples)
        if node.right:
            self._calculate_feature_importances(node.right, total_samples)
    
    def fit(self, X, y):
        """
        Fit the decision tree classifier.
        
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
        
        # Encode labels to integers
        self.classes_ = np.unique(y)
        y_encoded = np.searchsorted(self.classes_, y)
        
        self.n_features_in_ = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features_in_)
        
        # Build the tree
        self.tree_ = self._build_tree(X, y_encoded)
        
        # Calculate feature importances
        self._calculate_feature_importances(self.tree_, X.shape[0])
        
        # Normalize feature importances
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ /= np.sum(self.feature_importances_)
        
        self.is_fitted = True
        logger.info(f"Decision tree fitted with {len(self.classes_)} classes")
        
        return self
    
    def _predict_sample(self, x, node):
        """Predict a single sample using the tree."""
        if node.value is not None:  # Leaf node
            return node.value
        
        if x[node.feature_idx] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
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
            raise ValueError("Decision tree must be fitted before making predictions")
        
        X = self._validate_input(X)
        
        predictions = []
        for sample in X:
            pred_idx = self._predict_sample(sample, self.tree_)
            predictions.append(self.classes_[pred_idx])
        
        return np.array(predictions)
    
    def get_depth(self):
        """Get the depth of the tree."""
        def _get_depth(node):
            if node.value is not None:  # Leaf node
                return 1
            left_depth = _get_depth(node.left) if node.left else 0
            right_depth = _get_depth(node.right) if node.right else 0
            return 1 + max(left_depth, right_depth)
        
        return _get_depth(self.tree_) if self.tree_ else 0
    
    def get_n_leaves(self):
        """Get the number of leaves in the tree."""
        def _count_leaves(node):
            if node.value is not None:  # Leaf node
                return 1
            left_leaves = _count_leaves(node.left) if node.left else 0
            right_leaves = _count_leaves(node.right) if node.right else 0
            return left_leaves + right_leaves
        
        return _count_leaves(self.tree_) if self.tree_ else 0
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'criterion': self.criterion,
            'random_state': self.random_state
        }
    
    def __str__(self):
        if self.is_fitted:
            return f"DecisionTreeClassifier(max_depth={self.max_depth}, fitted=True, depth={self.get_depth()})"
        else:
            return f"DecisionTreeClassifier(max_depth={self.max_depth}, fitted=False)"
    
    def __repr__(self):
        return f"DecisionTreeClassifier(max_depth={self.max_depth}, criterion='{self.criterion}')"
