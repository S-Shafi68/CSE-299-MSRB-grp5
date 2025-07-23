#!/bin/bash

# =============================================================================
# Regression Workflow Automation Script
# ML Library Recreation Project
# =============================================================================

echo "=== Regression Workflow Automation ==="

# Setup
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="regression_logs_${TIMESTAMP}"
mkdir -p "$LOG_DIR"
RESULTS_DIR="RESULTS/regression_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}===================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

run_experiment() {
    local model_name=$1
    local dataset=$2
    shift 2
    local args=("$@")
    
    print_header "Running $model_name on $dataset dataset"
    
    local log_file="$LOG_DIR/${model_name}_${dataset}.log"
    local start_time=$(date +%s)
    
    echo "Command: python main.py --dataset $dataset --model $model_name ${args[*]}"
    echo "Log file: $log_file"
    echo ""
    
    if python main.py --dataset "$dataset" --model "$model_name" "${args[@]}" \
       --output_dir "$RESULTS_DIR" --verbose > "$log_file" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        print_success "$model_name completed in ${duration}s"
        
        # Extract R² score from log
        if grep -q "Test R²" "$log_file"; then
            local r2_score=$(grep "Test R²" "$log_file" | tail -1 | awk '{print $NF}')
            echo "  → Test R²: $r2_score"
        fi
    else
        echo -e "${RED}✗ $model_name failed - check $log_file${NC}"
        return 1
    fi
    echo ""
}

main() {
    print_header "Regression Models Comparison"
    echo "Starting at: $(date)"
    echo ""
    
    # =============================================================================
    # LINEAR REGRESSION MODELS
    # =============================================================================
    
    print_header "LINEAR REGRESSION MODELS"
    
    # Linear Regression
    run_experiment "linear_regression" "boston" \
        --scaler standard --test_size 0.2 --random_state 42
    
    # Ridge Regression
    run_experiment "ridge" "boston" \
        --alpha 1.0 --scaler standard --test_size 0.2 --random_state 42
    
    # Lasso Regression
    run_experiment "lasso" "boston" \
        --alpha 1.0 --scaler standard --test_size 0.2 --random_state 42
    
    # =============================================================================
    # REGULARIZATION COMPARISON
    # =============================================================================
    
    print_header "REGULARIZATION COMPARISON"
    
    # Different alpha values for Ridge
    local alphas=(0.1 1.0 10.0 100.0)
    
    for alpha in "${alphas[@]}"; do
        run_experiment "ridge" "boston" \
            --alpha "$alpha" --scaler standard --test_size 0.2 --random_state 42
    done
    
    # =============================================================================
    # PREPROCESSING COMPARISON
    # =============================================================================
    
    print_header "PREPROCESSING COMPARISON"
    
    # Compare different scalers
    local scalers=("none" "standard" "minmax")
    
    for scaler in "${scalers[@]}"; do
        run_experiment "linear_regression" "boston" \
            --scaler "$scaler" --test_size 0.2 --random_state 42
    done
    
    print_success "Regression workflow completed!"
}

# Run main function
main "$@"
