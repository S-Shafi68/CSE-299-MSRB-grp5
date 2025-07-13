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
"""
    )

    # Required arguments
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset to use (iris, boston, wine, digits)')
    parser.add_argument('--model', type=str, required=True,
                       help='Model to use (linear_regression, ridge, lasso, logistic_regression, knn, svm)')

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
    parser.add_argument('--task', type=str, choices=['classification', 'regression'], 
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
    
    models = {
        'linear_regression': LinearRegression,
        'ridge': Ridge,
        'lasso': Lasso,
        'logistic_regression': LogisticRegression,
        'knn': KNearestNeighbors,
        'svm': SupportVectorMachine,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(models.keys())}")
    
    return models[model_name](**kwargs)

def determine_task_type(model_name, dataset_name):
    """Automatically determine if this is a classification or regression task."""
    classification_models = ['logistic_regression', 'knn', 'svm']
    classification_datasets = ['iris', 'wine', 'digits']
    
    if model_name in classification_models or dataset_name in classification_datasets:
        return 'classification'
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

def calculate_and_display_metrics(model_name, dataset_name, y_train, y_test, y_pred_train, y_pred_test):
    """Calculate and display appropriate metrics based on task type."""
    task_type = determine_task_type(model_name, dataset_name)
    
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Task Type: {task_type}")
    
    if task_type == 'classification':
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
        
        print("="*50)
        
        return {
            'task_type': task_type,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'confusion_matrix': cm.tolist()
        }
    else:
        from utils.metrics import mean_squared_error, r2_score
        
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        print(f"Training MSE: {train_mse:.4f}")
        print(f"Test MSE: {test_mse:.4f}")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print("="*50)
        
        return {
            'task_type': task_type,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2
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
        X_train, X_test, y_train, y_test = data_loader.load_dataset(
            dataset_name=args.dataset,
            test_size=args.test_size,
            random_state=args.random_state,
            normalize=args.normalize.lower() == 'true'
        )
        
        # Apply preprocessing
        X_train, X_test = apply_preprocessing(X_train, X_test, args.scaler, args.scaler_range)
        logger.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
        
        # Initialize model
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
        
        model = get_model(args.model, **model_kwargs)
        
        # Train model
        logger.info("Training model...")
        model.fit(X_train, y_train)
        
        # Make predictions
        logger.info("Making predictions...")
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate and display metrics
        results_data = calculate_and_display_metrics(
            args.model, args.dataset, y_train, y_test, y_pred_train, y_pred_test
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
