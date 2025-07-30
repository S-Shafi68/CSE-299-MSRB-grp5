#!/bin/bash

# This script runs the Python experiments and unit tests

echo "Running Ridge model..."
python main.py --dataset boston --model ridge --alpha 1.0 --test_size 0.2 --verbose

echo "Running Lasso model..."
python main.py --dataset boston --model lasso --alpha 1.0 --test_size 0.2 --verbose

echo "Running unit tests..."
python -m unittest tests.test_scalers -v

echo "All tasks complete."
