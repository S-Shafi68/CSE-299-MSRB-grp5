#!/bin/bash

# =============================================================================
# Clustering Workflow Automation Script
# ML Library Recreation Project
# =============================================================================

echo "=== Clustering Workflow Automation ==="

# Setup
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="clustering_logs_${TIMESTAMP}"
mkdir -p "$LOG_DIR"
RESULTS_DIR="RESULTS/clustering_${TIMESTAMP}"
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
        
        # Extract silhouette score from log
        if grep -q "Silhouette Score" "$log_file"; then
            local silhouette=$(grep "Silhouette Score" "$log_file" | tail -1 | awk '{print $NF}')
            echo "  → Silhouette Score: $silhouette"
        fi
    else
        echo -e "${RED}✗ $model_name failed - check $log_file${NC}"
        return 1
    fi
    echo ""
}

main() {
    print_header "Clustering Models Comparison"
    echo "Starting at: $(date)"
    echo ""
    
    # =============================================================================
    # K-MEANS CLUSTERING
    # =============================================================================
    
    print_header "K-MEANS CLUSTERING"
    
    # Different number of clusters
    local clusters=(2 3 4 5)
    
    for k in "${clusters[@]}"; do
        run_experiment "kmeans" "iris" \
            --n_clusters "$k" --scaler standard --random_state 42
    done
    
    # =============================================================================
    # HIERARCHICAL CLUSTERING
    # =============================================================================
    
    print_header "HIERARCHICAL CLUSTERING"
    
    # Different linkage methods
    local linkages=("ward" "complete" "average" "single")
    
    for linkage in "${linkages[@]}"; do
        run_experiment "hierarchical" "wine" \
            --n_clusters 3 --linkage "$linkage" --scaler standard
    done
    
    # =============================================================================
    # DBSCAN CLUSTERING
    # =============================================================================
    
    print_header "DBSCAN CLUSTERING"
    
    # Different eps values
    local eps_values=(0.3 0.5 0.7 1.0)
    
    for eps in "${eps_values[@]}"; do
        run_experiment "dbscan" "digits" \
            --eps "$eps" --min_samples 5 --scaler standard
    done
    
    print_success "Clustering workflow completed!"
}

# Run main function
main "$@"
