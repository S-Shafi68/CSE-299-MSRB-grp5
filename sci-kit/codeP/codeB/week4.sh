#!/bin/bash

# This script runs a series of clustering model experiments

echo "Running K-Means on Iris..."
python main.py --dataset iris --model kmeans --n_clusters 3 --scaler standard

echo "Running Hierarchical Clustering on Wine..."
python main.py --dataset wine --model hierarchical --n_clusters 3 --linkage ward

echo "Running DBSCAN on Digits..."
python main.py --dataset digits --model dbscan --eps 0.5 --min_samples 5

echo "All clustering tasks complete."
