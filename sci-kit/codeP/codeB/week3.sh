#!/bin/bash

# This script runs a series of classification model experiments

echo "Running Logistic Regression on Iris..."
python main.py --dataset iris --model logistic_regression --scaler standard --verbose

echo "Running KNN (k=5) on Iris..."
python main.py --dataset iris --model knn --k 5 --scaler standard

echo "Running KNN (k=3, Manhattan) on Wine..."
python main.py --dataset wine --model knn --k 3 --distance_metric manhattan --scaler minmax

echo "Running KNN (k=7, Distance Weighted) on Iris..."
python main.py --dataset iris --model knn --k 7 --weights distance --scaler standard --verbose

echo "Running KNN (k=5, Cosine) on Digits..."
python main.py --dataset digits --model knn --k 5 --distance_metric cosine --scaler standard

echo "Running SVM (RBF kernel) on Iris..."
python main.py --dataset iris --model svm --C 1.0 --kernel rbf --scaler standard

echo "Running SVM (Linear kernel) on Wine..."
python main.py --dataset wine --model svm --C 0.5 --kernel linear --scaler minmax

echo "All experiments complete."
