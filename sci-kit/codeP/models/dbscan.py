"""
DBSCAN Clustering Implementation
"""

import numpy as np
from .base import BaseClusterer
from logger import get_logger

logger = get_logger("dbscan")

class DBSCAN(BaseClusterer):
    """
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise).
    
    Parameters:
    -----------
    eps : float, default=0.5
        Maximum distance between two samples for them to be considered neighbors
    min_samples : int, default=5
        Minimum number of samples in a neighborhood for a point to be core
    metric : str, default='euclidean'
        Distance metric to use
    """
    
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean'):
        super().__init__()
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        
        # Results
        self.labels_ = None
        self.core_sample_indices_ = None
        self.n_clusters_ = None
        
        # Validate parameters
        if eps <= 0:
            raise ValueError("eps must be positive")
        if min_samples < 1:
            raise ValueError("min_samples must be at least 1")
        if metric not in ['euclidean', 'manhattan']:
            raise ValueError("metric must be 'euclidean' or 'manhattan'")
        
        logger.debug(f"Initialized DBSCAN with eps={eps}, min_samples={min_samples}")
    
    def _calculate_distance(self, x1, x2):
        """Calculate distance between two points."""
        if self.metric == 'euclidean':
            return np.linalg.norm(x1 - x2)
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
    
    def _get_neighbors(self, X, point_idx):
        """Get neighbors within eps distance of a point."""
        neighbors = []
        for i, point in enumerate(X):
            if self._calculate_distance(X[point_idx], point) <= self.eps:
                neighbors.append(i)
        return neighbors
    
    def _expand_cluster(self, X, labels, point_idx, neighbors, cluster_id):
        """Expand cluster from a core point."""
        labels[point_idx] = cluster_id
        i = 0
        
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            
            # If neighbor is noise, add to current cluster
            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id
            
            # If neighbor is unvisited
            elif labels[neighbor_idx] == 0:
                labels[neighbor_idx] = cluster_id
                
                # Get neighbors of neighbor
                neighbor_neighbors = self._get_neighbors(X, neighbor_idx)
                
                # If neighbor is also a core point, add its neighbors
                if len(neighbor_neighbors) >= self.min_samples:
                    for nn in neighbor_neighbors:
                        if nn not in neighbors:
                            neighbors.append(nn)
            
            i += 1
    
    def fit(self, X, y=None):
        """
        Fit DBSCAN clustering to data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, optional
            Ignored. Present for API consistency.
        """
        X = self._validate_input(X)
        n_samples = X.shape[0]
        
        # Initialize labels: 0 = unvisited, -1 = noise, >0 = cluster
        labels = np.zeros(n_samples, dtype=int)
        cluster_id = 0
        core_samples = []
        
        # Process each point
        for point_idx in range(n_samples):
            # Skip if already processed
            if labels[point_idx] != 0:
                continue
            
            # Get neighbors
            neighbors = self._get_neighbors(X, point_idx)
            
            # If not enough neighbors, mark as noise
            if len(neighbors) < self.min_samples:
                labels[point_idx] = -1
            else:
                # Core point - start new cluster
                cluster_id += 1
                core_samples.append(point_idx)
                self._expand_cluster(X, labels, point_idx, neighbors, cluster_id)
        
        self.labels_ = labels
        self.core_sample_indices_ = np.array(core_samples)
        self.n_clusters_ = cluster_id
        
        self.is_fitted = True
        logger.info(f"DBSCAN fitted with {self.n_clusters_} clusters and {np.sum(labels == -1)} noise points")
        
        return self
    
    def fit_predict(self, X, y=None):
        """Fit the model and predict cluster labels."""
        self.fit(X, y)
        return self.labels_
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'eps': self.eps,
            'min_samples': self.min_samples,
            'metric': self.metric
        }
    
    def __str__(self):
        if self.is_fitted:
            return f"DBSCAN(eps={self.eps}, min_samples={self.min_samples}, fitted=True, n_clusters={self.n_clusters_})"
        else:
            return f"DBSCAN(eps={self.eps}, min_samples={self.min_samples}, fitted=False)"
    
    def __repr__(self):
        return f"DBSCAN(eps={self.eps}, min_samples={self.min_samples}, metric='{self.metric}')"
