#!/bin/bash

# Week 1 Environment Setup Script
# Sets up the project structure and dependencies

echo "=================================================="
echo "  SCIKIT-LEARN RECREATION PROJECT - SETUP"
echo "=================================================="

# Function to create directory if it doesn't exist
create_dir() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
        echo "✓ Created directory: $1"
    else
        echo "✓ Directory exists: $1"
    fi
}

# Function to create empty __init__.py files
create_init() {
    if [ ! -f "$1/__init__.py" ]; then
        touch "$1/__init__.py"
        echo "✓ Created __init__.py in $1"
    fi
}

echo "Setting up project structure..."
echo ""

# Create main directories
create_dir "models"
create_dir "utils" 
create_dir "preprocessing"
create_dir "tests"
create_dir "DATA"
create_dir "RESULTS"
create_dir "RESULTS/model_outputs"
create_dir "RESULTS/visualizations"
create_dir "RESULTS/benchmarks"
create_dir "codeB"

# Create __init__.py files for Python packages
create_init "models"
create_init "utils"
create_init "preprocessing"
create_init "tests"

echo ""
echo "Checking Python dependencies..."

# Check if required packages are available
check_package() {
    python -c "import $1" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "✓ $1 is available"
    else
        echo "✗ $1 is missing"
        echo "  Install with: pip install $1"
    fi
}

check_package "numpy"
check_package "argparse"
check_package "json"
check_package "logging"
check_package "sklearn"

echo ""
echo "Setting up sample data..."

# Create a simple sample dataset if it doesn't exist
if [ ! -f "DATA/sample_data.csv" ]; then
    python -c "
import numpy as np
import os

# Create sample regression data
np.random.seed(42)
X = np.random.randn(100, 3)
y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + np.random.randn(100) * 0.1

# Save as CSV
data = np.column_stack([X, y])
header = 'feature1,feature2,feature3,target'
np.savetxt('DATA/sample_data.csv', data, delimiter=',', header=header, comments='')
print('✓ Created sample dataset: DATA/sample_data.csv')
"
else
    echo "✓ Sample dataset already exists"
fi

echo ""
echo "Making scripts executable..."
chmod +x demo_week1.sh
chmod +x setup_week1.sh
chmod +x test_week1.py
echo "✓ Scripts are now executable"

echo ""
echo "=================================================="
echo "Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Run integration test: python test_week1.py"
echo "2. Test CLI interface: python main.py --help"
echo "3. Run faculty demo: ./demo_week1.sh"
echo "=================================================="