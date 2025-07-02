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
from models.ridge import Ridge
from models.lasso import Lasso 
from logger import get_logger

logger = get_logger("main")


# Add the current directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

from utils.data_loader import DataLoader
from utils.results import ResultsSaver
from models.linear_regression import LinearRegression


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
        """
    )
    
    # Required arguments
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset to use (iris, boston, wine, digits)')
    parser.add_argument('--model', type=str, required=True,
                       help='Model to use (linear_regression, ridge, lasso, etc.)')
    
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
    # Add this with your other arguments
    parser.add_argument('--scaler', type=str, choices=['none', 'standard', 'minmax'], 
                   default='none', help='Data preprocessing scaler')
    parser.add_argument('--scaler_range', type=str, default='0,1', 
                   help='Range for MinMaxScaler (e.g., "0,1" or "-1,1")')

    
    return parser.parse_args()


def get_model(model_name, **kwargs):
    """Factory function to get model instance."""
    models = {
        'linear_regression': LinearRegression,
         'ridge': Ridge,
         'lasso': Lasso,
        # Future models will be added here
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(models.keys())}")
    
    return models[model_name](**kwargs)


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
        data_loader = DataLoader()
        X_train, X_test, y_train, y_test = data_loader.load_dataset(
            dataset_name=args.dataset,
            test_size=args.test_size,
            random_state=args.random_state,
            normalize=args.normalize.lower() == 'true'
            
        )
        # Apply preprocessing - ADD THIS LINE
        X_train, X_test = apply_preprocessing(X_train, X_test, args.scaler, args.scaler_range)
        
        logger.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
        
        # Initialize model
        logger.info("Initializing model...")
        model_kwargs = {}
        if args.model in ['ridge', 'lasso']:
            model_kwargs['alpha'] = args.alpha
            
        model = get_model(args.model, **model_kwargs)
        
        # Train model
        logger.info("Training model...")
        model.fit(X_train, y_train)
        
        # Make predictions
        logger.info("Making predictions...")
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics (basic for now)
        from utils.metrics import mean_squared_error, r2_score
        
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score  (y_test, y_pred_test)
        
        # Display results
        print("\n" + "="*50)
        print("RESULTS")
        print("="*50)
        print(f"Model: {args.model}")
        print(f"Dataset: {args.dataset}")
        print(f"Training MSE: {train_mse:.4f}")
        print(f"Test MSE: {test_mse:.4f}")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print("="*50)
        
        # Save results
        logger.info("Saving results...")
        results_saver = ResultsSaver(args.output_dir)
        results_data = {
            'model': args.model,
            'dataset': args.dataset,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'parameters': vars(args)
        }
        results_saver.save_results(results_data, f"{args.model}_{args.dataset}")
        
        logger.info("Execution completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()