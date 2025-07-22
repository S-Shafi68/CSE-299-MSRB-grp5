"""
k-Nearest Neighbors Classifier Implementation
"""

import numpy as np
from .base import BaseClassifier
from logger import get_logger

logger = get_logger("knn")

class KNearestNeighbors(BaseClassifier):
    """
    k-Nearest Neighbors classifier implementation.
    
    This classifier implements the k-nearest neighbors algorithm for classification.
    It stores all training data and makes predictions based on the majority vote
    of the k nearest neighbors in the feature space.
    
    Parameters:
    -----------
    k : int, default=5
        Number of neighbors to consider for classification
    distance_metric : str, default='euclidean'
        Distance metric to use ('euclidean', 'manhattan', 'cosine')
    weights : str, default='uniform'
        Weight function used in prediction ('uniform', 'distance')
    """
    
    def __init__(self, k=5, distance_metric='euclidean', weights='uniform'):
        super().__init__()
        self.k = k
        self.distance_metric = distance_metric
        self.weights = weights
        
        # Training data storage (lazy learning)
        self.X_train = None
        self.y_train = None
        self.classes_ = None
        self.n_features_in_ = None
        
        # Validate parameters
        if k <= 0:
            raise ValueError("k must be a positive integer")
        if distance_metric not in ['euclidean', 'manhattan', 'cosine']:
            raise ValueError("distance_metric must be 'euclidean', 'manhattan', or 'cosine'")
        if weights not in ['uniform', 'distance']:
            raise ValueError("weights must be 'uniform' or 'distance'")
        
        logger.debug(f"Initialized KNN with k={k}, distance_metric={distance_metric}, weights={weights}")
    
    def _calculate_distance(self, x1, x2):
        """
        Calculate distance between two points based on the specified metric.
        
        Parameters:
        -----------
        x1, x2 : array-like
            Two data points to calculate distance between
            
        Returns:
        --------
        distance : float
            Distance between the two points
        """
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.distance_metric == 'cosine':
            # Cosine distance = 1 - cosine similarity
            dot_product = np.dot(x1, x2)
            norm_x1 = np.linalg.norm(x1)
            norm_x2 = np.linalg.norm(x2)
            if norm_x1 == 0 or norm_x2 == 0:
                return 1.0  # Maximum distance for zero vectors
            cosine_sim = dot_product / (norm_x1 * norm_x2)
            return 1 - cosine_sim
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def _get_neighbors(self, x):
        """
        Get k nearest neighbors for a single point.
        
        Parameters:
        -----------
        x : array-like
            Single data point to find neighbors for
            
        Returns:
        --------
        neighbors : list of tuples
            List of (distance, label, index) for k nearest neighbors
        """
        distances = []
        
        # Calculate distances to all training points
        for i, x_train in enumerate(self.X_train):
            dist = self._calculate_distance(x, x_train)
            distances.append((dist, self.y_train[i], i))
        
        # Sort by distance and get k nearest
        distances.sort(key=lambda x: x[0])
        return distances[:self.k]
    
    def _predict_single(self, x):
        """
        Predict class for a single sample.
        
        Parameters:
        -----------
        x : array-like
            Single data point to predict
            
        Returns:
        --------
        prediction : class label
            Predicted class for the input point
        """
        neighbors = self._get_neighbors(x)
        
        if self.weights == 'uniform':
            # Simple majority vote
            votes = {}
            for _, label, _ in neighbors:
                votes[label] = votes.get(label, 0) + 1
            return max(votes, key=votes.get)
        
        elif self.weights == 'distance':
            # Distance-weighted voting
            votes = {}
            for distance, label, _ in neighbors:
                # Avoid division by zero for identical points
                weight = 1 / (distance + 1e-8)
                votes[label] = votes.get(label, 0) + weight
            return max(votes, key=votes.get)
    
    def fit(self, X, y):
        """
        Fit the KNN classifier.
        
        In KNN, fitting simply stores the training data since it's a lazy learning algorithm.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : KNearestNeighbors
            Returns self for method chaining
        """
        X, y = self._validate_input(X, y)
        
        # Store training data (lazy learning approach)
        self.X_train = X.copy()
        self.y_train = y.copy()
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        
        # Validate k against number of samples
        if self.k > len(X):
            logger.warning(f"k={self.k} is larger than number of samples ({len(X)}). Using k={len(X)}")
            self.k = len(X)
        
        self.is_fitted = True
        logger.info(f"KNN fitted with {len(X)} training samples, {len(self.classes_)} classes")
        logger.debug(f"Classes: {self.classes_}")
        
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
            raise ValueError("KNN must be fitted before making predictions")
        
        X = self._validate_input(X)
        
        # Check feature consistency
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but KNN was fitted with {self.n_features_in_} features")
        
        predictions = []
        for x in X:
            pred = self._predict_single(x)
            predictions.append(pred)
        
        logger.debug(f"Made predictions for {len(X)} samples")
        return np.array(predictions)
    
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
            Class probabilities for each sample
        """
        if not self.is_fitted:
            raise ValueError("KNN must be fitted before making predictions")
        
        X = self._validate_input(X)
        
        # Check feature consistency
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but KNN was fitted with {self.n_features_in_} features")
        
        probabilities = []
        
        for x in X:
            neighbors = self._get_neighbors(x)
            
            # Initialize probability dictionary for all classes
            class_probs = {class_label: 0.0 for class_label in self.classes_}
            
            if self.weights == 'uniform':
                # Equal weight for all neighbors
                for _, label, _ in neighbors:
                    class_probs[label] += 1.0 / self.k
            
            elif self.weights == 'distance':
                # Distance-weighted probabilities
                total_weight = 0.0
                weighted_votes = {class_label: 0.0 for class_label in self.classes_}
                
                for distance, label, _ in neighbors:
                    weight = 1 / (distance + 1e-8)  # Avoid division by zero
                    weighted_votes[label] += weight
                    total_weight += weight
                
                # Normalize by total weight
                if total_weight > 0:
                    for class_label in self.classes_:
                        class_probs[class_label] = weighted_votes[class_label] / total_weight
            
            # Convert to array in consistent order
            prob_array = [class_probs[class_label] for class_label in self.classes_]
            probabilities.append(prob_array)
        
        return np.array(probabilities)
    
    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        """
        Find the k-neighbors of a point.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Query points
        n_neighbors : int, optional
            Number of neighbors to get (default is the value passed to the constructor)
        return_distance : bool, default=True
            Whether to return distances
            
        Returns:
        --------
        distances : array, shape (n_samples, n_neighbors)
            Distances to the neighbors (only if return_distance=True)
        indices : array, shape (n_samples, n_neighbors)
            Indices of the neighbors in the training data
        """
        if not self.is_fitted:
            raise ValueError("KNN must be fitted before calling kneighbors")
        
        if n_neighbors is None:
            n_neighbors = self.k
        
        X = self._validate_input(X)
        
        distances_list = []
        indices_list = []
        
        for x in X:
            # Get all distances
            all_distances = []
            for i, x_train in enumerate(self.X_train):
                dist = self._calculate_distance(x, x_train)
                all_distances.append((dist, i))
            
            # Sort and get top n_neighbors
            all_distances.sort(key=lambda x: x[0])
            top_neighbors = all_distances[:n_neighbors]
            
            distances_list.append([dist for dist, _ in top_neighbors])
            indices_list.append([idx for _, idx in top_neighbors])
        
        distances_array = np.array(distances_list)
        indices_array = np.array(indices_list)
        
        if return_distance:
            return distances_array, indices_array
        else:
            return indices_array
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'k': self.k,
            'distance_metric': self.distance_metric,
            'weights': self.weights
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
            return f"KNearestNeighbors(k={self.k}, fitted=True, n_samples={len(self.X_train)})"
        else:
            return f"KNearestNeighbors(k={self.k}, fitted=False)"
    
    def __repr__(self):
        """Detailed string representation of the classifier."""
        return f"KNearestNeighbors(k={self.k}, distance_metric='{self.distance_metric}', weights='{self.weights}')"
