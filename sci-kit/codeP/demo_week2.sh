#!/bin/bash
echo "=== Week 2 Demo: Ridge, Lasso & Preprocessing ==="
echo

echo "1. Testing Ridge Regression:"
python main.py --dataset boston --model ridge --alpha 1.0 --test_size 0.2 --verbose

echo -e "\n2. Testing Lasso Regression:"
python main.py --dataset boston --model lasso --alpha 0.1 --test_size 0.2 --verbose

echo -e "\n3. Testing with StandardScaler:"
python main.py --dataset boston --model ridge --alpha 1.0 --scaler standard --test_size 0.2 --verbose

echo -e "\n4. Testing with MinMaxScaler:"
python main.py --dataset boston --model lasso --alpha 0.1 --scaler minmax --scaler_range "-1,1" --test_size 0.2 --verbose

echo -e "\n5. Model Comparison on Iris Dataset:"
echo "Linear Regression:"
python main.py --dataset iris --model linear_regression --test_size 0.3

echo -e "\nRidge Regression:"
python main.py --dataset iris --model ridge --alpha 0.5 --test_size 0.3

echo -e "\nLasso Regression:"
python main.py --dataset iris --model lasso --alpha 0.1 --test_size 0.3

echo -e "\n=== Week 2 Demo Complete! ==="
