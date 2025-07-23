"""
Clustering Evaluation Metrics
"""

import numpy as np
from logger import get_logger

logger = get_logger("clustering_metrics")

def silhouette_score(X, labels):
    """
    Calculate silhouette score for clustering.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    labels : array-like, shape (n_samples,)
        Cluster labels
        
    Returns:
    --------
    score : float
        Mean silhouette score
    """
    if len(np.unique(labels)) < 2:
        return 0.0
    
    n_samples = X.shape[0]
    silhouette_values = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Calculate a(i) - mean distance to points in same cluster
        same_cluster_mask = labels == labels[i]
        same_cluster_points = X[same_cluster_mask]
        
        if len(same_cluster_points) > 1:
            a_i = np.mean([np.linalg.norm(X[i] - point) for point in same_cluster_points if not np.array_equal(X[i], point)])
        else:
            a_i = 0
        
        # Calculate b(i) - mean distance to points in nearest cluster
        b_i = float('inf')
        for cluster_id in np.unique(labels):
            if cluster_id != labels[i]:
                other_cluster_points = X[labels == cluster_id]
                if len(other_cluster_points) > 0:
                    mean_dist = np.mean([np.linalg.norm(X[i] - point) for point in other_cluster_points])
                    b_i = min(b_i, mean_dist)
        
        # Calculate silhouette value
        if max(a_i, b_i) > 0:
            silhouette_values[i] = (b_i - a_i) / max(a_i, b_i)
        else:
            silhouette_values[i] = 0
    
    return np.mean(silhouette_values)

def within_cluster_sum_of_squares(X, labels, centers):
    """
    Calculate within-cluster sum of squares (inertia).
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    labels : array-like, shape (n_samples,)
        Cluster labels
    centers : array-like, shape (n_clusters, n_features)
        Cluster centers
        
    Returns:
    --------
    wcss : float
        Within-cluster sum of squares
    """
    wcss = 0
    for i, center in enumerate(centers):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            wcss += np.sum((cluster_points - center)**2)
    
    return wcss

def calinski_harabasz_score(X, labels):
    """
    Calculate Calinski-Harabasz score (variance ratio criterion).
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    labels : array-like, shape (n_samples,)
        Cluster labels
        
    Returns:
    --------
    score : float
        Calinski-Harabasz score
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0
    
    n_samples, n_features = X.shape
    n_clusters = len(unique_labels)
    
    # Calculate overall centroid
    overall_centroid = np.mean(X, axis=0)
    
    # Calculate between-cluster dispersion
    between_cluster_dispersion = 0
    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) > 0:
            cluster_centroid = np.mean(cluster_points, axis=0)
            between_cluster_dispersion += len(cluster_points) * np.sum((cluster_centroid - overall_centroid)**2)
    
    # Calculate within-cluster dispersion
    within_cluster_dispersion = 0
    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) > 0:
            cluster_centroid = np.mean(cluster_points, axis=0)
            within_cluster_dispersion += np.sum((cluster_points - cluster_centroid)**2)
    
    # Calculate score
    if within_cluster_dispersion == 0:
        return 0.0
    
    score = (between_cluster_dispersion / (n_clusters - 1)) / (within_cluster_dispersion / (n_samples - n_clusters))
    return score

def davies_bouldin_score(X, labels):
    """
    Calculate Davies-Bouldin score.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    labels : array-like, shape (n_samples,)
        Cluster labels
        
    Returns:
    --------
    score : float
        Davies-Bouldin score (lower is better)
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0
    
    n_clusters = len(unique_labels)
    
    # Calculate cluster centroids and within-cluster distances
    centroids = []
    within_cluster_distances = []
    
    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) > 0:
            centroid = np.mean(cluster_points, axis=0)
            centroids.append(centroid)
            
            # Average distance from centroid to points in cluster
            avg_distance = np.mean([np.linalg.norm(point - centroid) for point in cluster_points])
            within_cluster_distances.append(avg_distance)
        else:
            centroids.append(np.zeros(X.shape[1]))
            within_cluster_distances.append(0)
    
    centroids = np.array(centroids)
    within_cluster_distances = np.array(within_cluster_distances)
    
    # Calculate Davies-Bouldin score
    db_score = 0
    for i in range(n_clusters):
        max_ratio = 0
        for j in range(n_clusters):
            if i != j:
                centroid_distance = np.linalg.norm(centroids[i] - centroids[j])
                if centroid_distance > 0:
                    ratio = (within_cluster_distances[i] + within_cluster_distances[j]) / centroid_distance
                    max_ratio = max(max_ratio, ratio)
        db_score += max_ratio
    
    return db_score / n_clusters

def adjusted_rand_score(labels_true, labels_pred):
    """
    Calculate Adjusted Rand Index between two clusterings.
    
    Parameters:
    -----------
    labels_true : array-like, shape (n_samples,)
        True cluster labels
    labels_pred : array-like, shape (n_samples,)
        Predicted cluster labels
        
    Returns:
    --------
    ari : float
        Adjusted Rand Index
    """
    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)
    
    # Build contingency table
    n_samples = len(labels_true)
    true_labels = np.unique(labels_true)
    pred_labels = np.unique(labels_pred)
    
    contingency = np.zeros((len(true_labels), len(pred_labels)))
    for i, true_label in enumerate(true_labels):
        for j, pred_label in enumerate(pred_labels):
            contingency[i, j] = np.sum((labels_true == true_label) & (labels_pred == pred_label))
    
    # Calculate ARI
    sum_comb_c = np.sum([n * (n - 1) / 2 for n in np.sum(contingency, axis=1)])
    sum_comb_k = np.sum([n * (n - 1) / 2 for n in np.sum(contingency, axis=0)])
    sum_comb = np.sum([n * (n - 1) / 2 for n in contingency.flatten()])
    
    expected_index = sum_comb_c * sum_comb_k / (n_samples * (n_samples - 1) / 2)
    max_index = (sum_comb_c + sum_comb_k) / 2
    
    if max_index == expected_index:
        return 1.0
    
    return (sum_comb - expected_index) / (max_index - expected_index)
