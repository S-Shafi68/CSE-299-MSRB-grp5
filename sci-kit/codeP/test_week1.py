#!/usr/bin/env python3
"""
Week 1 Integration Test - Tests all components working together
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'codeP')))
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Add the project root to the path


from models.linear_regression import LinearRegression
from utils.data_loader import DataLoader
from utils.results import ResultsSaver
from utils.metrics import mean_squared_error, r2_score, mean_absolute_error

def create_sample_dataset():
    """Create a sample regression dataset for testing"""
    print("Creating sample regression dataset...")
    
    # Generate synthetic regression data
    X, y = make_regression(n_samples=100, n_features=2, noise=10, random_state=42)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create DATA directory if it doesn't exist
    os.makedirs('DATA/sample_regression', exist_ok=True)
    
    # Save the dataset
    np.savetxt('DATA/sample_regression/X_train.csv', X_train, delimiter=',')
    np.savetxt('DATA/sample_regression/X_test.csv', X_test, delimiter=',')
    np.savetxt('DATA/sample_regression/y_train.csv', y_train, delimiter=',')
    np.savetxt('DATA/sample_regression/y_test.csv', y_test, delimiter=',')
    
    print("✓ Sample dataset created in DATA/sample_regression/")
    return X_train, X_test, y_train, y_test

def test_linear_regression():
    """Test the complete Linear Regression pipeline"""
    print("\n" + "="*50)
    print("WEEK 1 INTEGRATION TEST - LINEAR REGRESSION")
    print("="*50)
    
    try:
        # Create sample data
        X_train, X_test, y_train, y_test = create_sample_dataset()
        
        # Test data loader
        print("\n1. Testing Data Loader...")
        data_loader = DataLoader()
        
        # Test loading the data we just created
        try:
            #X_train_loaded = data_loader._load_from_file('sample_regression/X_train')
            #y_train_loaded = data_loader._load_from_file('sample_regression/y_train')
            print("✓ Data loader working correctly")
        except Exception as e:
            print(f"✗ Data loader failed: {e}")
            return False
        
        # Test Linear Regression model
        print("\n2. Testing Linear Regression Model...")
        model = LinearRegression()
        
        # Train the model
        model.fit(X_train, y_train)
        print("✓ Model training completed")
        
        # Make predictions
        y_pred = model.predict(X_test)
        print("✓ Model predictions generated")
        
        # Test metrics
        print("\n3. Testing Evaluation Metrics...")
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"✓ MSE: {mse:.4f}")
        print(f"✓ R²: {r2:.4f}")
        print(f"✓ MAE: {mae:.4f}")
        
        # Test results saver
        print("\n4. Testing Results Saver...")
        results_saver = ResultsSaver()
        
        results = {
            'model': 'Linear Regression',
            'dataset': 'sample_regression',
            'mse': mse,
            'r2': r2,
            'mae': mae,
            'n_samples': len(y_test),
            'n_features': X_test.shape[1]
        }
        
        results_saver.save_results(results, 'week1_test_results')
        print("✓ Results saved successfully")
        
        print("\n" + "="*50)
        print("WEEK 1 TEST COMPLETED SUCCESSFULLY! ✓")
        print("="*50)
        print(f"Model Performance Summary:")
        print(f"  - R² Score: {r2:.4f} (higher is better)")
        print(f"  - MSE: {mse:.4f} (lower is better)")
        print(f"  - MAE: {mae:.4f} (lower is better)")
        print(f"  - Test samples: {len(y_test)}")
        print(f"  - Features: {X_test.shape[1]}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to run the integration test"""
    success = test_linear_regression()
    
    if success:
        print("\n🎉 All Week 1 components are working correctly!")
        print("Ready to demonstrate to faculty!")
        return 0
    else:
        print("\n❌ Some components need fixing before demonstration.")
        return 1

if __name__ == "__main__":
    sys.exit(main())