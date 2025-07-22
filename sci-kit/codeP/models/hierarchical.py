"""
Hierarchical Clustering Implementation
"""

import numpy as np
from .base import BaseClusterer
from logger import get_logger

logger = get_logger("hierarchical")

class AgglomerativeClustering(BaseClusterer):
    """
    Agglomerative (bottom-up) hierarchical clustering.
    
    Parameters:
    -----------
    n_clusters : int, default=2
        Number of clusters to find
    linkage : str, default='ward'
        Linkage criterion ('ward', 'complete', 'average', 'single')
    distance_threshold : float, default=None
        Distance threshold for clustering
    """
    
    def __init__(self, n_clusters=2, linkage='ward', distance_threshold=None):
        super().__init__()
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.distance_threshold = distance_threshold
        
        # Results
        self.labels_ = None
        self.n_clusters_ = None
        self.children_ = None
        self.distances_ = None
        
        # Validate parameters
        if linkage not in ['ward', 'complete', 'average', 'single']:
            raise ValueError("linkage must be 'ward', 'complete', 'average', or 'single'")
        
        logger.debug(f"Initialized AgglomerativeClustering with n_clusters={n_clusters}, linkage={linkage}")
    
    def _calculate_distance_matrix(self, X):
        """Calculate pairwise distance matrix."""
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = np.linalg.norm(X[i] - X[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
    
    def _cluster_distance(self, cluster1, cluster2, X, distance_matrix):
        """Calculate distance between two clusters."""
        if self.linkage == 'single':
            # Minimum distance between any two points in clusters
            min_dist = float('inf')
            for i in cluster1:
                for j in cluster2:
                    min_dist = min(min_dist, distance_matrix[i, j])
            return min_dist
        
        elif self.linkage == 'complete':
            # Maximum distance between any two points in clusters
            max_dist = 0
            for i in cluster1:
                for j in cluster2:
                    max_dist = max(max_dist, distance_matrix[i, j])
            return max_dist
        
        elif self.linkage == 'average':
            # Average distance between all pairs of points
            total_dist = 0
            count = 0
            for i in cluster1:
                for j in cluster2:
                    total_dist += distance_matrix[i, j]
                    count += 1
            return total_dist / count if count > 0 else 0
        
        elif self.linkage == 'ward':
            # Ward's method - minimize within-cluster variance
            cluster1_points = X[list(cluster1)]
            cluster2_points = X[list(cluster2)]
            
            # Calculate centroids
            centroid1 = np.mean(cluster1_points, axis=0)
            centroid2 = np.mean(cluster2_points, axis=0)
            
            # Calculate merged centroid
            n1, n2 = len(cluster1), len(cluster2)
            merged_centroid = (n1 * centroid1 + n2 * centroid2) / (n1 + n2)
            
            # Calculate increase in sum of squared errors
            sse_increase = (n1 * n2 / (n1 + n2)) * np.sum((centroid1 - centroid2)**2)
            return sse_increase
    
    def fit(self, X, y=None):
        """
        Fit hierarchical clustering to data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, optional
            Ignored. Present for API consistency.
        """
        X = self._validate_input(X)
        n_samples = X.shape[0]
        
        # Initialize each point as its own cluster
        clusters = [{i} for i in range(n_samples)]
        
        # Calculate initial distance matrix
        distance_matrix = self._calculate_distance_matrix(X)
        
        # Store merge history
        merge_history = []
        merge_distances = []
        
        # Main agglomerative loop
        while len(clusters) > 1:
            # Find closest pair of clusters
            min_distance = float('inf')
            merge_indices = (0, 1)
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist = self._cluster_distance(clusters[i], clusters[j], X, distance_matrix)
                    if dist < min_distance:
                        min_distance = dist
                        merge_indices = (i, j)
            
            # Check distance threshold
            if self.distance_threshold is not None and min_distance > self.distance_threshold:
                break
            
            # Merge clusters
            i, j = merge_indices
            merged_cluster = clusters[i] | clusters[j]
            
            # Store merge information
            merge_history.append((list(clusters[i]), list(clusters[j])))
            merge_distances.append(min_distance)
            
            # Update clusters list
            new_clusters = []
            for k, cluster in enumerate(clusters):
                if k != i and k != j:
                    new_clusters.append(cluster)
            new_clusters.append(merged_cluster)
            clusters = new_clusters
            
            # Stop if we've reached desired number of clusters
            if len(clusters) == self.n_clusters:
                break
        
        # Assign final labels
        labels = np.zeros(n_samples, dtype=int)
        for cluster_id, cluster in enumerate(clusters):
            for point_id in cluster:
                labels[point_id] = cluster_id
        
        self.labels_ = labels
        self.n_clusters_ = len(clusters)
        self.children_ = merge_history
        self.distances_ = merge_distances
        
        self.is_fitted = True
        logger.info(f"Hierarchical clustering fitted with {self.n_clusters_} clusters")
        
        return self
    
    def fit_predict(self, X, y=None):
        """Fit the model and predict cluster labels."""
        self.fit(X, y)
        return self.labels_
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'n_clusters': self.n_clusters,
            'linkage': self.linkage,
            'distance_threshold': self.distance_threshold
        }
    
    def __str__(self):
        if self.is_fitted:
            return f"AgglomerativeClustering(n_clusters={self.n_clusters_}, linkage='{self.linkage}', fitted=True)"
        else:
            return f"AgglomerativeClustering(n_clusters={self.n_clusters}, linkage='{self.linkage}', fitted=False)"
    
    def __repr__(self):
        return f"AgglomerativeClustering(n_clusters={self.n_clusters}, linkage='{self.linkage}')"
