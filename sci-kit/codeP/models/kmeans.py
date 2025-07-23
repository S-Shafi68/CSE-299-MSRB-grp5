"""
K-Means Clustering Implementation
"""

import numpy as np
from .base import BaseClusterer
from logger import get_logger

logger = get_logger("kmeans")

class KMeans(BaseClusterer):
    """
    K-Means clustering algorithm implementation.
    
    Parameters:
    -----------
    n_clusters : int, default=8
        Number of clusters to form
    init : str, default='k-means++'
        Initialization method ('k-means++' or 'random')
    max_iter : int, default=300
        Maximum number of iterations
    tolerance : float, default=1e-4
        Tolerance for convergence
    random_state : int, default=None
        Random seed for reproducibility
    """
    
    def __init__(self, n_clusters=8, init='k-means++', max_iter=300, 
                 tolerance=1e-4, random_state=None):
        super().__init__()
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.random_state = random_state
        
        # Results
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None
        
        # Validate parameters
        if n_clusters <= 0:
            raise ValueError("n_clusters must be positive")
        if init not in ['k-means++', 'random']:
            raise ValueError("init must be 'k-means++' or 'random'")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        
        logger.debug(f"Initialized KMeans with n_clusters={n_clusters}, init={init}")
    
    def _init_centroids(self, X):
        """Initialize cluster centroids."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        centroids = np.zeros((self.n_clusters, n_features))
        
        if self.init == 'random':
            # Random initialization
            for i in range(self.n_clusters):
                centroids[i] = X[np.random.choice(n_samples)]
        
        elif self.init == 'k-means++':
            # K-means++ initialization
            # Choose first centroid randomly
            centroids[0] = X[np.random.choice(n_samples)]
            
            # Choose remaining centroids
            for i in range(1, self.n_clusters):
                # Calculate distances to nearest centroid
                distances = np.array([min([np.linalg.norm(x - c)**2 for c in centroids[:i]]) 
                                    for x in X])
                
                # Choose next centroid with probability proportional to distance
                probabilities = distances / distances.sum()
                cumulative_prob = probabilities.cumsum()
                r = np.random.random()
                
                for j, p in enumerate(cumulative_prob):
                    if r < p:
                        centroids[i] = X[j]
                        break
        
        return centroids
    
    def _assign_clusters(self, X, centroids):
        """Assign each point to nearest centroid."""
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
    
    def _update_centroids(self, X, labels):
        """Update centroids based on current assignments."""
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        
        for i in range(self.n_clusters):
            if np.sum(labels == i) > 0:
                centroids[i] = X[labels == i].mean(axis=0)
            else:
                # If no points assigned to centroid, reinitialize randomly
                centroids[i] = X[np.random.choice(X.shape[0])]
        
        return centroids
    
    def _calculate_inertia(self, X, labels, centroids):
        """Calculate within-cluster sum of squares."""
        inertia = 0
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - centroids[i])**2)
        return inertia
    
    def fit(self, X, y=None):
        """
        Fit K-Means clustering to data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, optional
            Ignored. Present for API consistency.
        """
        X = self._validate_input(X)
        
        if X.shape[0] < self.n_clusters:
            raise ValueError(f"Number of samples ({X.shape[0]}) must be >= n_clusters ({self.n_clusters})")
        
        # Initialize centroids
        centroids = self._init_centroids(X)
        
        # Main K-means loop
        for iteration in range(self.max_iter):
            # Assign points to clusters
            labels = self._assign_clusters(X, centroids)
            
            # Update centroids
            new_centroids = self._update_centroids(X, labels)
            
            # Check for convergence
            if np.all(np.abs(centroids - new_centroids) < self.tolerance):
                logger.debug(f"K-Means converged after {iteration + 1} iterations")
                break
            
            centroids = new_centroids
        
        self.n_iter_ = iteration + 1
        self.cluster_centers_ = centroids
        self.labels_ = labels
        self.inertia_ = self._calculate_inertia(X, labels, centroids)
        self.n_clusters_ = self.n_clusters
        
        self.is_fitted = True
        logger.info(f"K-Means fitted with {self.n_clusters} clusters, inertia: {self.inertia_:.4f}")
        
        return self
    
    def predict(self, X):
        """
        Predict cluster labels for new data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to predict
            
        Returns:
        --------
        labels : array, shape (n_samples,)
            Cluster labels
        """
        if not self.is_fitted:
            raise ValueError("KMeans must be fitted before making predictions")
        
        X = self._validate_input(X)
        return self._assign_clusters(X, self.cluster_centers_)
    
    def fit_predict(self, X, y=None):
        """Fit the model and predict cluster labels."""
        self.fit(X, y)
        return self.labels_
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'n_clusters': self.n_clusters,
            'init': self.init,
            'max_iter': self.max_iter,
            'tolerance': self.tolerance,
            'random_state': self.random_state
        }
    
    def __str__(self):
        if self.is_fitted:
            return f"KMeans(n_clusters={self.n_clusters}, fitted=True, inertia={self.inertia_:.4f})"
        else:
            return f"KMeans(n_clusters={self.n_clusters}, fitted=False)"
    
    def __repr__(self):
        return f"KMeans(n_clusters={self.n_clusters}, init='{self.init}', max_iter={self.max_iter})"
