#!/bin/bash

# This script runs experiments with tree-based models and hyperparameter tuning

echo "--- Running Decision Tree ---"
python main.py --dataset iris --model decision_tree --max_depth 5 --criterion gini --scaler standard

echo "--- Running Random Forest ---"
python main.py --dataset wine --model random_forest --n_estimators 100 --max_depth 10 --scaler standard

echo "--- Running Cross-Validation ---"
python main.py --dataset iris --model decision_tree --cross_validate --cv_folds 10

echo "--- Running Grid Search ---"
python main.py --dataset iris --model decision_tree --grid_search --max_depth 5

echo "--- Running Decision Tree with Cross-Validation ---"
python main.py --dataset iris --model decision_tree --max_depth 5 --cross_validate --cv_folds 10 --stratified

echo "--- Running Random Forest with Grid Search ---"
python main.py --dataset wine --model random_forest --grid_search --scoring f1

echo "--- Running Decision Tree with Randomized Search ---"
python main.py --dataset digits --model decision_tree --randomized_search --n_iter_search 20

echo "All tasks complete."
