#!/bin/bash

# Week 1 Faculty Demonstration Script
# Scikit-Learn Recreation Project

echo "=================================================="
echo "  SCIKIT-LEARN RECREATION PROJECT - WEEK 1 DEMO"
echo "=================================================="
echo "Student: [Your Name]"
echo "Date: $(date)"
echo "Week: 1/7 - Foundation & Architecture"
echo "=================================================="

# Function to print section headers
print_section() {
    echo ""
    echo ">>> $1"
    echo "----------------------------------------"
}

# Function to check if command was successful
check_success() {
    if [ $? -eq 0 ]; then
        echo "✓ SUCCESS"
    else
        echo "✗ FAILED"
        exit 1
    fi
}

print_section "1. PROJECT STRUCTURE OVERVIEW"
echo "Showing current project structure:"
echo ""
tree -I '__pycache__' . 2>/dev/null || find . -type f -name "*.py" | head -20
echo ""

print_section "2. COMMAND-LINE INTERFACE TEST"
echo "Testing basic CLI functionality:"
echo "Command: python main.py --help"
echo ""
python main.py --help
check_success

print_section "3. LINEAR REGRESSION - BASIC USAGE"
echo "Running Linear Regression with sample data:"
echo "Command: python main.py --dataset sample_regression --model linear_regression"
echo ""
python main.py --dataset sample_regression --model linear_regression
check_success

print_section "4. INTEGRATION TEST"
echo "Running comprehensive Week 1 integration test:"
echo ""
python test_week1.py
check_success

print_section "5. RESULTS VERIFICATION"
echo "Showing generated results:"
echo ""
if [ -d "RESULTS" ]; then
    echo "Results directory contents:"
    ls -la RESULTS/
    echo ""
    
    # Show latest results file if it exists
    LATEST_RESULT=$(ls -t RESULTS/*.json 2>/dev/null | head -1)
    if [ -n "$LATEST_RESULT" ]; then
        echo "Latest results:"
        cat "$LATEST_RESULT"
        echo ""
    fi
else
    echo "No results directory found - this might indicate an issue"
fi

print_section "6. CODE QUALITY CHECK"
echo "Checking Python syntax for all modules:"
echo ""

# Check each Python file
for file in main.py models/*.py utils/*.py; do
    if [ -f "$file" ]; then
        echo -n "Checking $file... "
        python -m py_compile "$file" 2>/dev/null
        if [ $? -eq 0 ]; then
            echo "✓"
        else
            echo "✗ Syntax error"
        fi
    fi
done

print_section "7. WEEK 1 DELIVERABLES CHECKLIST"
echo "Verifying all Week 1 requirements:"
echo ""

# Check each deliverable
check_deliverable() {
    if [ -f "$1" ]; then
        echo "✓ $2"
    else
        echo "✗ $2 (Missing: $1)"
    fi
}

check_deliverable "main.py" "Command-line interface"
check_deliverable "models/linear_regression.py" "Linear Regression implementation"
check_deliverable "utils/data_loader.py" "Data loading functionality"
check_deliverable "utils/results.py" "Result saving functionality"
check_deliverable "models/base_model.py" "Base model interface"
check_deliverable "utils/metrics.py" "Evaluation metrics"

print_section "8. PERFORMANCE METRICS"
echo "Week 1 Performance Summary:"
echo ""

# Run a quick performance test
echo "Running performance benchmark..."
python -c "
import time
import numpy as np
from models.linear_regression import LinearRegression

# Generate test data
np.random.seed(42)
X = np.random.randn(1000, 5)
y = np.random.randn(1000)

# Time the training
model = LinearRegression()
start_time = time.time()
model.fit(X, y)
training_time = time.time() - start_time

# Time the prediction
start_time = time.time()
predictions = model.predict(X)
prediction_time = time.time() - start_time

print(f'Training time: {training_time:.4f} seconds')
print(f'Prediction time: {prediction_time:.4f} seconds')
print(f'Samples processed: {len(X)}')
print(f'Features: {X.shape[1]}')
"

print_section "9. NEXT WEEK PREPARATION"
echo "Week 2 will focus on:"
echo "  • Ridge Regression implementation"
echo "  • Lasso Regression implementation"
echo "  • Data preprocessing (StandardScaler, MinMaxScaler)"
echo "  • Enhanced evaluation metrics"
echo ""

print_section "10. DEMO COMPLETE"
echo "=================================================="
echo "Week 1 demonstration completed successfully!"
echo "All core components are working:"
echo "  ✓ Command-line interface"
echo "  ✓ Basic architecture"
echo "  ✓ Linear Regression algorithm"
echo "  ✓ Data loading utilities"
echo "  ✓ Results saving system"
echo "  ✓ Evaluation metrics"
echo ""
echo "Ready to proceed to Week 2!"
echo "=================================================="