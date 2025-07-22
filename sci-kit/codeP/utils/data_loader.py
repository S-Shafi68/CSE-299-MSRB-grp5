"""
Data loading utilities for the ML library recreation project.
Handles loading, preprocessing, and splitting of datasets.
"""

import numpy as np
import os
from pathlib import Path
import logging


class DataLoader:
    """Handles loading and preprocessing of datasets."""
    
    def __init__(self, data_dir="DATA"):
        """
        Initialize DataLoader.
        
        Args:
            data_dir (str): Directory containing datasets
        """
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(__name__)
        
    def load_dataset(self, dataset_name, test_size=0.2, random_state=42, normalize=False):
        """
        Load a dataset and split into train/test sets.
        
        Args:
            dataset_name (str): Name of the dataset
            test_size (float): Proportion of test set
            random_state (int): Random state for reproducibility
            normalize (bool): Whether to normalize features
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        # Load the dataset
        X, y = self._load_raw_dataset(dataset_name)
        
        # Split into train/test
        X_train, X_test, y_train, y_test = self.train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Normalize if requested
        if normalize:
            X_train, X_test = self._normalize_features(X_train, X_test)
            
        return X_train, X_test, y_train, y_test
    
    def _load_raw_dataset(self, dataset_name):
        """
        Load raw dataset from file or generate synthetic data.
        
        Args:
            dataset_name (str): Name of the dataset
            
        Returns:
            tuple: (X, y) features and targets
        """
        if dataset_name == "iris":
            return self._load_iris()
        elif dataset_name == "boston":
            return self._load_boston()
        elif dataset_name == "wine":
            return self._load_wine()
        elif dataset_name == "digits":
            return self._load_digits()
        else:
            # Try to load from file
            return self._load_from_file(dataset_name)
    
    def _load_iris(self):
        """Load or generate Iris dataset."""
        # For now, generate synthetic iris-like data
        # In a real implementation, you'd load from sklearn.datasets or a file
        np.random.seed(42)
        
        # Generate 150 samples with 4 features (sepal/petal length/width)
        n_samples = 150
        n_features = 4
        
        # Create three clusters (species)
        X = np.zeros((n_samples, n_features))
        y = np.zeros(n_samples)
        
        for i in range(3):
            start_idx = i * 50
            end_idx = (i + 1) * 50
            
            # Different means for each species
            means = [5.0 + i, 3.0 + i*0.5, 3.5 + i, 1.0 + i*0.5]
            
            for j in range(n_features):
                X[start_idx:end_idx, j] = np.random.normal(means[j], 0.5, 50)
            
            y[start_idx:end_idx] = i
            
        self.logger.info(f"Generated synthetic Iris dataset: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def _load_boston(self):
        """Load or generate Boston housing dataset."""
        # Generate synthetic regression data similar to Boston housing
        np.random.seed(42)
        
        n_samples = 506
        n_features = 13
        
        # Generate random features
        X = np.random.randn(n_samples, n_features)
        
        # Create a linear relationship with some noise
        true_coef = np.random.randn(n_features)
        y = X @ true_coef + np.random.normal(0, 1, n_samples)
        
        # Scale to reasonable housing price range
        y = (y - y.min()) / (y.max() - y.min()) * 40 + 10  # 10-50k range
        
        self.logger.info(f"Generated synthetic Boston dataset: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def _load_wine(self):
        """Load or generate Wine dataset."""
        # Generate synthetic wine classification data
        np.random.seed(42)
        
        n_samples = 178
        n_features = 13
        n_classes = 3
        
        X = np.zeros((n_samples, n_features))
        y = np.zeros(n_samples)
        
        samples_per_class = n_samples // n_classes
        
        for i in range(n_classes):
            start_idx = i * samples_per_class
            end_idx = (i + 1) * samples_per_class if i < n_classes - 1 else n_samples
            
            # Different feature distributions for each wine class
            class_mean = np.random.randn(n_features) * 2 + i * 3
            class_cov = np.eye(n_features) * (0.5 + i * 0.3)
            
            X[start_idx:end_idx] = np.random.multivariate_normal(
                class_mean, class_cov, end_idx - start_idx
            )
            y[start_idx:end_idx] = i
            
        self.logger.info(f"Generated synthetic Wine dataset: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def _load_digits(self):
        """Load or generate Digits dataset."""
        # Generate synthetic digit classification data
        np.random.seed(42)
        
        n_samples = 1797
        n_features = 64  # 8x8 pixel images
        n_classes = 10
        
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)
        
        self.logger.info(f"Generated synthetic Digits dataset: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def _load_from_file(self, dataset_name):
        """
        Load dataset from CSV file.
        
        Args:
            dataset_name (str): Name of the dataset file (without extension)
            
        Returns:
            tuple: (X, y) features and targets
        """
        csv_path = self.data_dir / f"{dataset_name}.csv"
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")
        
        # Simple CSV loading (assumes last column is target)
        data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
        X = data[:, :-1]
        y = data[:, -1]
        
        self.logger.info(f"Loaded dataset from file: {csv_path}")
        return X, y
    
    def train_test_split(self, X, y, test_size=0.2, random_state=42):
        """
        Split dataset into training and testing sets.
        
        Args:
            X (array): Features
            y (array): Targets
            test_size (float): Proportion of test set
            random_state (int): Random state for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        np.random.seed(random_state)
        
        n_samples = X.shape[0]
        n_test = int(n_samples * test_size)
        
        # Random indices for test set
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]
        
        return X_train, X_test, y_train, y_test
    
    def _normalize_features(self, X_train, X_test):
        """
        Normalize features using training set statistics.
        
        Args:
            X_train (array): Training features
            X_test (array): Test features
            
        Returns:
            tuple: (X_train_norm, X_test_norm)
        """
        # Calculate mean and std from training data
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        
        # Avoid division by zero
        std = np.where(std == 0, 1, std)
        
        # Normalize both sets using training statistics
        X_train_norm = (X_train - mean) / std
        X_test_norm = (X_test - mean) / std
        
        self.logger.info("Features normalized using training set statistics")
        return X_train_norm, X_test_norm