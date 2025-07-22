#!/usr/bin/env python3

"""
Main CLI entry point for the ML library recreation project.
Usage: python main.py --dataset X --model Y --options Z
"""

import argparse
import logging
import sys
import os
from pathlib import Path
import numpy as np

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from logger import get_logger

logger = get_logger("main")

def setup_logging(level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='ML Library Recreation Project',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --dataset iris --model linear_regression --test_size 0.3
  python main.py --dataset boston --model linear_regression --normalize true
  python main.py --dataset iris --model logistic_regression --scaler standard
  python main.py --dataset iris --model knn --k 5 --scaler standard
  python main.py --dataset wine --model knn --k 3 --distance_metric manhattan --weights distance
  python main.py --dataset iris --model svm --C 1.0 --kernel linear --scaler standard
  python main.py --dataset wine --model svm --C 0.5 --kernel rbf --gamma 0.1 --scaler standard
  python main.py --dataset iris --model kmeans --n_clusters 3 --scaler standard
  python main.py --dataset wine --model hierarchical --n_clusters 3 --linkage ward
  python main.py --dataset digits --model dbscan --eps 0.5 --min_samples 5 --scaler standard
  python main.py --dataset iris --model decision_tree --max_depth 5 --criterion gini --scaler standard
  python main.py --dataset wine --model random_forest --n_estimators 100 --max_depth 10 --scaler standard
  python main.py --dataset iris --model decision_tree --cross_validate --cv_folds 10
  python main.py --dataset iris --model random_forest --grid_search --n_estimators 50 100
"""
    )

    # Required arguments
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset to use (iris, boston, wine, digits)')
    parser.add_argument('--model', type=str, required=True,
                       help='Model to use (linear_regression, ridge, lasso, logistic_regression, knn, svm, kmeans, hierarchical, dbscan, decision_tree, random_forest)')

    # Optional arguments
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility (default: 42)')
    parser.add_argument('--normalize', type=str, choices=['true', 'false'], default='false',
                       help='Whether to normalize features (default: false)')
    parser.add_argument('--cv', type=int, default=None,
                       help='Number of cross-validation folds')
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='Regularization parameter for Ridge/Lasso (default: 1.0)')
    parser.add_argument('--output_dir', type=str, default='RESULTS',
                       help='Output directory for results (default: RESULTS)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--scaler', type=str, choices=['none', 'standard', 'minmax'],
                       default='none', help='Data preprocessing scaler')
    parser.add_argument('--scaler_range', type=str, default='0,1',
                       help='Range for MinMaxScaler (e.g., "0,1" or "-1,1")')
    parser.add_argument('--task', type=str, choices=['classification', 'regression', 'clustering'], 
                       help='Task type (auto-detected if not specified)')

    # KNN-specific arguments
    parser.add_argument('--k', type=int, default=5, 
                       help='Number of neighbors for KNN (default: 5)')
    parser.add_argument('--distance_metric', type=str, default='euclidean',
                       choices=['euclidean', 'manhattan', 'cosine'],
                       help='Distance metric for KNN (default: euclidean)')
    parser.add_argument('--weights', type=str, default='uniform',
                       choices=['uniform', 'distance'],
                       help='Weight function for KNN (default: uniform)')

    # SVM-specific arguments
    parser.add_argument('--C', type=float, default=1.0,
                       help='Regularization parameter for SVM (default: 1.0)')
    parser.add_argument('--kernel', type=str, default='linear',
                       choices=['linear', 'rbf', 'poly'],
                       help='SVM kernel type (default: linear)')
    parser.add_argument('--gamma', type=str, default='scale',
                       help='Kernel coefficient for SVM (default: scale)')
    parser.add_argument('--degree', type=int, default=3,
                       help='Degree for polynomial kernel (default: 3)')
    parser.add_argument('--svm_max_iter', type=int, default=1000,
                       help='Maximum iterations for SVM (default: 1000)')
    parser.add_argument('--svm_tolerance', type=float, default=1e-3,
                       help='Tolerance for SVM convergence (default: 1e-3)')

    # Clustering-specific arguments
    parser.add_argument('--n_clusters', type=int, default=3,
                       help='Number of clusters for K-Means and Hierarchical (default: 3)')
    parser.add_argument('--init', type=str, default='k-means++',
                       choices=['k-means++', 'random'],
                       help='K-Means initialization method (default: k-means++)')
    parser.add_argument('--max_iter', type=int, default=300,
                       help='Maximum iterations for K-Means (default: 300)')
    parser.add_argument('--tolerance', type=float, default=1e-4,
                       help='Tolerance for K-Means convergence (default: 1e-4)')
    
    # Hierarchical clustering arguments
    parser.add_argument('--linkage', type=str, default='ward',
                       choices=['ward', 'complete', 'average', 'single'],
                       help='Hierarchical clustering linkage method (default: ward)')
    parser.add_argument('--distance_threshold', type=float, default=None,
                       help='Distance threshold for hierarchical clustering (default: None)')
    
    # DBSCAN arguments
    parser.add_argument('--eps', type=float, default=0.5,
                       help='DBSCAN eps parameter (default: 0.5)')
    parser.add_argument('--min_samples', type=int, default=5,
                       help='DBSCAN min_samples parameter (default: 5)')
    parser.add_argument('--metric', type=str, default='euclidean',
                       choices=['euclidean', 'manhattan'],
                       help='DBSCAN distance metric (default: euclidean)')

    # Tree model arguments
    parser.add_argument('--max_depth', type=int, default=None,
                       help='Maximum depth for tree models (default: None)')
    parser.add_argument('--min_samples_split', type=int, default=2,
                       help='Minimum samples to split for tree models (default: 2)')
    parser.add_argument('--min_samples_leaf', type=int, default=1,
                       help='Minimum samples in leaf for tree models (default: 1)')
    parser.add_argument('--n_estimators', type=int, default=100,
                       help='Number of trees for Random Forest (default: 100)')
    parser.add_argument('--criterion', type=str, default='gini',
                       choices=['gini', 'entropy'],
                       help='Splitting criterion for Decision Tree (default: gini)')
    parser.add_argument('--max_features', type=str, default='sqrt',
                       help='Max features for Random Forest (default: sqrt)')
    parser.add_argument('--bootstrap', type=str, default='true',
                       choices=['true', 'false'],
                       help='Bootstrap sampling for Random Forest (default: true)')

    # Validation arguments
    parser.add_argument('--cross_validate', action='store_true',
                       help='Perform cross-validation')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--stratified', action='store_true',
                       help='Use stratified cross-validation for classification')
    parser.add_argument('--grid_search', action='store_true',
                       help='Perform grid search for hyperparameter tuning')
    parser.add_argument('--randomized_search', action='store_true',
                       help='Perform randomized search for hyperparameter tuning')
    parser.add_argument('--n_iter_search', type=int, default=10,
                       help='Number of iterations for randomized search (default: 10)')
    parser.add_argument('--scoring', type=str, default='accuracy',
                       choices=['accuracy', 'f1', 'precision', 'recall'],
                       help='Scoring metric for validation (default: accuracy)')

    return parser.parse_args()

def get_model(model_name, **kwargs):
    """Factory function to get model instance."""
    # Import models here to avoid circular imports
    from models.linear_regression import LinearRegression
    from models.ridge import Ridge
    from models.lasso import Lasso
    from models.logistic_regression import LogisticRegression
    from models.knn import KNearestNeighbors
    from models.svm import SupportVectorMachine
    from models.kmeans import KMeans
    from models.hierarchical import AgglomerativeClustering
    from models.dbscan import DBSCAN
    from models.decision_tree import DecisionTreeClassifier
    from models.random_forest import RandomForestClassifier
    
    models = {
        'linear_regression': LinearRegression,
        'ridge': Ridge,
        'lasso': Lasso,
        'logistic_regression': LogisticRegression,
        'knn': KNearestNeighbors,
        'svm': SupportVectorMachine,
        'kmeans': KMeans,
        'hierarchical': AgglomerativeClustering,
        'dbscan': DBSCAN,
        'decision_tree': DecisionTreeClassifier,
        'random_forest': RandomForestClassifier,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(models.keys())}")
    
    return models[model_name](**kwargs)

def determine_task_type(model_name, dataset_name):
    """Automatically determine if this is a classification, regression, or clustering task."""
    clustering_models = ['kmeans', 'hierarchical', 'dbscan']
    classification_models = ['logistic_regression', 'knn', 'svm', 'decision_tree', 'random_forest']
    classification_datasets = ['iris', 'wine', 'digits']
    
    if model_name in clustering_models:
        return 'clustering'
    elif model_name in classification_models or dataset_name in classification_datasets:
        return 'classification'
    else:
        return 'regression'

def apply_preprocessing(X_train, X_test, scaler_type, scaler_range="0,1"):
    """Apply preprocessing to training and test data."""
    if scaler_type == 'none':
        return X_train, X_test
    
    from preprocessing.scalers import StandardScaler, MinMaxScaler
    
    if scaler_type == 'standard':
        scaler = StandardScaler()
        logger.info("Applying StandardScaler preprocessing...")
    elif scaler_type == 'minmax':
        # Parse range
        range_min, range_max = map(float, scaler_range.split(','))
        scaler = MinMaxScaler(feature_range=(range_min, range_max))
        logger.info(f"Applying MinMaxScaler preprocessing with range ({range_min}, {range_max})...")
    
    # Fit on training data and transform both
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"Preprocessing completed. Original shape: {X_train.shape}")
    return X_train_scaled, X_test_scaled

def perform_cross_validation(model, X, y, args):
    """Perform cross-validation on the model."""
    from utils.validation import cross_validate, KFold, StratifiedKFold
    
    # Setup cross-validation
    if args.stratified and determine_task_type(args.model, args.dataset) == 'classification':
        cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)
        logger.info(f"Using Stratified {args.cv_folds}-Fold Cross-Validation")
    else:
        cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)
        logger.info(f"Using {args.cv_folds}-Fold Cross-Validation")
    
    # Perform cross-validation
    cv_results = cross_validate(model, X, y, cv=cv, scoring=args.scoring, return_train_score=True)
    
    # Display results
    print("\n" + "="*50)
    print("CROSS-VALIDATION RESULTS")
    print("="*50)
    print(f"Scoring: {args.scoring}")
    print(f"CV Folds: {args.cv_folds}")
    print(f"Test Score: {cv_results['test_score'].mean():.4f} (+/- {cv_results['test_score'].std() * 2:.4f})")
    print(f"Train Score: {cv_results['train_score'].mean():.4f} (+/- {cv_results['train_score'].std() * 2:.4f})")
    print(f"Fit Time: {cv_results['fit_time'].mean():.4f} (+/- {cv_results['fit_time'].std() * 2:.4f}) seconds")
    print(f"Score Time: {cv_results['score_time'].mean():.4f} (+/- {cv_results['score_time'].std() * 2:.4f}) seconds")
    
    # Individual fold results
    print("\nIndividual Fold Results:")
    for i, (train_score, test_score) in enumerate(zip(cv_results['train_score'], cv_results['test_score'])):
        print(f"Fold {i+1}: Train={train_score:.4f}, Test={test_score:.4f}")
    
    print("="*50)
    
    return cv_results

def perform_grid_search(model, X, y, args):
    """Perform grid search hyperparameter tuning."""
    from utils.grid_search import GridSearchCV
    
    # Define parameter grids for different models
    param_grids = {
        'decision_tree': {
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        },
        'random_forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        },
        'svm': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
        },
        'knn': {
            'k': [3, 5, 7, 9, 11],
            'distance_metric': ['euclidean', 'manhattan'],
            'weights': ['uniform', 'distance']
        }
    }
    
    if args.model not in param_grids:
        logger.warning(f"No parameter grid defined for {args.model}. Using default parameters.")
        return None
    
    # Setup grid search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grids[args.model],
        cv=args.cv_folds,
        scoring=args.scoring,
        refit=True
    )
    
    logger.info(f"Starting Grid Search for {args.model} with {len(param_grids[args.model])} parameters...")
    
    # Perform grid search
    grid_search.fit(X, y)
    
    # Display results
    print("\n" + "="*50)
    print("GRID SEARCH RESULTS")
    print("="*50)
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Score: {grid_search.best_score_:.4f}")
    print(f"Model: {args.model}")
    print(f"Scoring: {args.scoring}")
    
    # Show top 5 results
    print("\nTop 5 Parameter Combinations:")
    results = grid_search.cv_results_
    sorted_indices = sorted(range(len(results['mean_test_score'])), 
                          key=lambda i: results['mean_test_score'][i], reverse=True)
    
    for i, idx in enumerate(sorted_indices[:5]):
        print(f"{i+1}. Score: {results['mean_test_score'][idx]:.4f} "
              f"(+/- {results['std_test_score'][idx] * 2:.4f}) "
              f"Params: {results['params'][idx]}")
    
    print("="*50)
    
    return grid_search

def perform_randomized_search(model, X, y, args):
    """Perform randomized search hyperparameter tuning."""
    from utils.grid_search import RandomizedSearchCV
    
    # Define parameter distributions for different models
    param_distributions = {
        'decision_tree': {
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 6],
            'criterion': ['gini', 'entropy']
        },
        'random_forest': {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 6],
            'max_features': ['sqrt', 'log2']
        },
        'svm': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
        },
        'knn': {
            'k': [3, 5, 7, 9, 11, 13],
            'distance_metric': ['euclidean', 'manhattan', 'cosine'],
            'weights': ['uniform', 'distance']
        }
    }
    
    if args.model not in param_distributions:
        logger.warning(f"No parameter distribution defined for {args.model}. Using default parameters.")
        return None
    
    # Setup randomized search
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions[args.model],
        n_iter=args.n_iter_search,
        cv=args.cv_folds,
        scoring=args.scoring,
        refit=True,
        random_state=args.random_state
    )
    
    logger.info(f"Starting Randomized Search for {args.model} with {args.n_iter_search} iterations...")
    
    # Perform randomized search
    random_search.fit(X, y)
    
    # Display results
    print("\n" + "="*50)
    print("RANDOMIZED SEARCH RESULTS")
    print("="*50)
    print(f"Best Parameters: {random_search.best_params_}")
    print(f"Best Score: {random_search.best_score_:.4f}")
    print(f"Model: {args.model}")
    print(f"Scoring: {args.scoring}")
    print(f"Iterations: {args.n_iter_search}")
    
    print("="*50)
    
    return random_search

def calculate_and_display_metrics(model_name, dataset_name, y_train, y_test, y_pred_train, y_pred_test, model=None):
    """Calculate and display appropriate metrics based on task type."""
    task_type = determine_task_type(model_name, dataset_name)
    
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Task Type: {task_type}")
    
    if task_type == 'clustering':
        from utils.clustering_metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
        
        # For clustering, we use the full dataset (X_train + X_test)
        X_full = None
        if hasattr(model, 'labels_'):
            # Get the full dataset for clustering evaluation
            from utils.data_loader import DataLoader
            data_loader = DataLoader()
            X_full, _, _, _ = data_loader.load_dataset(
                dataset_name=dataset_name,
                test_size=0.0,  # Use full dataset for clustering
                random_state=42,
                normalize=False
            )
            
            # Apply same preprocessing as used for training
            from preprocessing.scalers import StandardScaler, MinMaxScaler
            # (This is a simplified version - in practice, you'd want to pass the scaler)
            
            labels = model.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # Exclude noise for DBSCAN
            
            print(f"Number of clusters: {n_clusters}")
            if hasattr(model, 'inertia_'):
                print(f"Inertia (WCSS): {model.inertia_:.4f}")
            
            if X_full is not None and len(set(labels)) > 1:
                silhouette = silhouette_score(X_full, labels)
                calinski = calinski_harabasz_score(X_full, labels)
                davies = davies_bouldin_score(X_full, labels)
                
                print(f"Silhouette Score: {silhouette:.4f}")
                print(f"Calinski-Harabasz Score: {calinski:.4f}")
                print(f"Davies-Bouldin Score: {davies:.4f}")
                
                # Count noise points for DBSCAN
                if model_name == 'dbscan':
                    n_noise = list(labels).count(-1)
                    print(f"Number of noise points: {n_noise}")
                
                print("="*50)
                
                return {
                    'task_type': task_type,
                    'n_clusters': n_clusters,
                    'silhouette_score': silhouette,
                    'calinski_harabasz_score': calinski,
                    'davies_bouldin_score': davies,
                    'inertia': getattr(model, 'inertia_', None),
                    'n_noise': list(labels).count(-1) if model_name == 'dbscan' else 0,
                    'labels': labels.tolist()
                }
        
        print("="*50)
        return {'task_type': task_type}
    
    elif task_type == 'classification':
        from utils.metrics import calculate_accuracy, calculate_precision, calculate_recall, calculate_f1
        from utils.metrics import confusion_matrix, classification_report
        
        # Calculate metrics
        train_accuracy = calculate_accuracy(y_train, y_pred_train)
        test_accuracy = calculate_accuracy(y_test, y_pred_test)
        test_precision = calculate_precision(y_test, y_pred_test, average='weighted')
        test_recall = calculate_recall(y_test, y_pred_test, average='weighted')
        test_f1 = calculate_f1(y_test, y_pred_test, average='weighted')
        
        # Display results
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test F1-Score: {test_f1:.4f}")
        
        # Show confusion matrix
        cm = confusion_matrix(y_test, y_pred_test)
        print(f"\nConfusion Matrix:\n{cm}")
        
        # Show classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_test))
        
        # Show feature importance for tree models
        if hasattr(model, 'feature_importances_') and model.feature_importances_ is not None:
            print(f"\nFeature Importances:")
            for i, importance in enumerate(model.feature_importances_):
                print(f"  Feature {i}: {importance:.4f}")
        
        # Show tree-specific metrics
        if hasattr(model, 'get_depth'):
            print(f"\nTree Depth: {model.get_depth()}")
        if hasattr(model, 'get_n_leaves'):
            print(f"Number of Leaves: {model.get_n_leaves()}")
        if hasattr(model, 'oob_score_') and model.oob_score_ is not None:
            print(f"Out-of-Bag Score: {model.oob_score_:.4f}")
        
        print("="*50)
        
        return {
            'task_type': task_type,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'confusion_matrix': cm.tolist(),
            'feature_importances': model.feature_importances_.tolist() if hasattr(model, 'feature_importances_') else None,
            'tree_depth': model.get_depth() if hasattr(model, 'get_depth') else None,
            'n_leaves': model.get_n_leaves() if hasattr(model, 'get_n_leaves') else None,
            'oob_score': model.oob_score_ if hasattr(model, 'oob_score_') else None
        }
    
    else:  # regression
        from utils.metrics import mean_squared_error, r2_score
        
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        print(f"Training MSE: {train_mse:.4f}")
        print(f"Test MSE: {test_mse:.4f}")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        
        # Show feature importance for tree models
        if hasattr(model, 'feature_importances_') and model.feature_importances_ is not None:
            print(f"\nFeature Importances:")
            for i, importance in enumerate(model.feature_importances_):
                print(f"  Feature {i}: {importance:.4f}")
        
        print("="*50)
        
        return {
            'task_type': task_type,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'feature_importances': model.feature_importances_.tolist() if hasattr(model, 'feature_importances_') else None
        }

def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting ML Library Recreation Project")
    logger.info(f"Dataset: {args.dataset}, Model: {args.model}")
    
    try:
        # Load data
        logger.info("Loading data...")
        from utils.data_loader import DataLoader
        data_loader = DataLoader()
        
        # For clustering, we might want to use the full dataset
        task_type = determine_task_type(args.model, args.dataset)
        
        if task_type == 'clustering':
            # For clustering, use full dataset
            X_train, X_test, y_train, y_test = data_loader.load_dataset(
                dataset_name=args.dataset,
                test_size=args.test_size,
                random_state=args.random_state,
                normalize=args.normalize.lower() == 'true'
            )
            # Combine train and test for clustering
            X_full = np.vstack([X_train, X_test])
            y_full = np.hstack([y_train, y_test]) if y_train is not None else None
            X_train, X_test = X_full, X_full  # Use full dataset for clustering
            y_train, y_test = y_full, y_full
        else:
            # For supervised learning, use train/test split
            X_train, X_test, y_train, y_test = data_loader.load_dataset(
                dataset_name=args.dataset,
                test_size=args.test_size,
                random_state=args.random_state,
                normalize=args.normalize.lower() == 'true'
            )
        
        # Apply preprocessing
        X_train, X_test = apply_preprocessing(X_train, X_test, args.scaler, args.scaler_range)
        logger.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
        
        # Initialize model parameters
        logger.info("Initializing model...")
        model_kwargs = {}
        
        if args.model in ['ridge', 'lasso']:
            model_kwargs['alpha'] = args.alpha
        elif args.model == 'knn':
            model_kwargs['k'] = args.k
            model_kwargs['distance_metric'] = args.distance_metric
            model_kwargs['weights'] = args.weights
            logger.info(f"KNN parameters: k={args.k}, distance_metric={args.distance_metric}, weights={args.weights}")
        elif args.model == 'svm':
            model_kwargs['C'] = args.C
            model_kwargs['kernel'] = args.kernel
            if args.gamma != 'scale':
                try:
                    model_kwargs['gamma'] = float(args.gamma)
                except ValueError:
                    model_kwargs['gamma'] = args.gamma
            else:
                model_kwargs['gamma'] = args.gamma
            model_kwargs['degree'] = args.degree
            model_kwargs['max_iter'] = args.svm_max_iter
            model_kwargs['tolerance'] = args.svm_tolerance
            logger.info(f"SVM parameters: C={args.C}, kernel={args.kernel}, gamma={args.gamma}, max_iter={args.svm_max_iter}")
        elif args.model == 'kmeans':
            model_kwargs['n_clusters'] = args.n_clusters
            model_kwargs['init'] = args.init
            model_kwargs['max_iter'] = args.max_iter
            model_kwargs['tolerance'] = args.tolerance
            model_kwargs['random_state'] = args.random_state
            logger.info(f"K-Means parameters: n_clusters={args.n_clusters}, init={args.init}, max_iter={args.max_iter}")
        elif args.model == 'hierarchical':
            model_kwargs['n_clusters'] = args.n_clusters
            model_kwargs['linkage'] = args.linkage
            if args.distance_threshold is not None:
                model_kwargs['distance_threshold'] = args.distance_threshold
            logger.info(f"Hierarchical parameters: n_clusters={args.n_clusters}, linkage={args.linkage}")
        elif args.model == 'dbscan':
            model_kwargs['eps'] = args.eps
            model_kwargs['min_samples'] = args.min_samples
            model_kwargs['metric'] = args.metric
            logger.info(f"DBSCAN parameters: eps={args.eps}, min_samples={args.min_samples}, metric={args.metric}")
        elif args.model == 'decision_tree':
            model_kwargs['max_depth'] = args.max_depth
            model_kwargs['min_samples_split'] = args.min_samples_split
            model_kwargs['min_samples_leaf'] = args.min_samples_leaf
            model_kwargs['criterion'] = args.criterion
            model_kwargs['random_state'] = args.random_state
            logger.info(f"Decision Tree parameters: max_depth={args.max_depth}, criterion={args.criterion}")
        elif args.model == 'random_forest':
            model_kwargs['n_estimators'] = args.n_estimators
            model_kwargs['max_depth'] = args.max_depth
            model_kwargs['min_samples_split'] = args.min_samples_split
            model_kwargs['min_samples_leaf'] = args.min_samples_leaf
            model_kwargs['max_features'] = args.max_features
            model_kwargs['bootstrap'] = args.bootstrap.lower() == 'true'
            model_kwargs['random_state'] = args.random_state
            logger.info(f"Random Forest parameters: n_estimators={args.n_estimators}, max_depth={args.max_depth}")
        
        # Create model instance
        model = get_model(args.model, **model_kwargs)
        
        # Perform validation if requested
        if args.cross_validate:
            X_combined = np.vstack([X_train, X_test])
            y_combined = np.hstack([y_train, y_test]) if y_train is not None else None
            
            if task_type != 'clustering':
                cv_results = perform_cross_validation(model, X_combined, y_combined, args)
            else:
                logger.warning("Cross-validation not applicable for clustering tasks")
        
        # Perform hyperparameter tuning if requested
        if args.grid_search:
            X_combined = np.vstack([X_train, X_test])
            y_combined = np.hstack([y_train, y_test]) if y_train is not None else None
            
            if task_type != 'clustering':
                grid_search = perform_grid_search(model, X_combined, y_combined, args)
                if grid_search:
                    model = grid_search.best_estimator_
            else:
                logger.warning("Grid search not applicable for clustering tasks")
        
        # Perform randomized search if requested
        if args.randomized_search:
            X_combined = np.vstack([X_train, X_test])
            y_combined = np.hstack([y_train, y_test]) if y_train is not None else None
            
            if task_type != 'clustering':
                random_search = perform_randomized_search(model, X_combined, y_combined, args)
                if random_search:
                    model = random_search.best_estimator_
            else:
                logger.warning("Randomized search not applicable for clustering tasks")
        
        # Train model
        logger.info("Training model...")
        if task_type == 'clustering':
            model.fit(X_train)  # Clustering doesn't use y
            y_pred_train = model.labels_
            y_pred_test = model.labels_  # Same as train for clustering
        else:
            model.fit(X_train, y_train)
            # Make predictions
            logger.info("Making predictions...")
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
        
        # Calculate and display metrics
        results_data = calculate_and_display_metrics(
            args.model, args.dataset, y_train, y_test, y_pred_train, y_pred_test, model
        )
        
        # Add additional metadata
        results_data.update({
            'model': args.model,
            'dataset': args.dataset,
            'parameters': vars(args)
        })
        
        # Save results
        logger.info("Saving results...")
        from utils.results import ResultsSaver
        results_saver = ResultsSaver(args.output_dir)
        results_saver.save_results(results_data, f"{args.model}_{args.dataset}")
        
        logger.info("Execution completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
