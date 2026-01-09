#!/bin/bash
# Process all datasets in Phase 2 using the Docker container

set -e

# Configuration
IMAGE_NAME="airrml2025-winning-solution:latest"
TRAIN_DIR="${1:-/path/to/train_datasets}"
TEST_DIR="${2:-/path/to/test_datasets}"
OUTPUT_DIR="${3:-/path/to/output}"
N_JOBS="${4:-4}"
DEVICE="${5:-cpu}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}AIRR-ML 2025 - Batch Processing${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Train directory: $TRAIN_DIR"
echo "Test directory: $TEST_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "N jobs: $N_JOBS"
echo "Device: $DEVICE"
echo ""

# Validate directories
if [ ! -d "$TRAIN_DIR" ]; then
    echo -e "${RED}Error: Training directory does not exist: $TRAIN_DIR${NC}"
    echo "Usage: $0 <train_dir> <test_dir> <output_dir> [n_jobs] [device]"
    exit 1
fi

if [ ! -d "$TEST_DIR" ]; then
    echo -e "${RED}Error: Test directory does not exist: $TEST_DIR${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Track success/failure
declare -a SUCCESS_DATASETS
declare -a FAILED_DATASETS

# Process each dataset
for dataset_num in {1..8}; do
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Processing Dataset $dataset_num${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    # Find training directory
    train_dataset_dir=""
    for pattern in "train_dataset_${dataset_num}" "dataset_${dataset_num}" "train_ds${dataset_num}" "ds${dataset_num}"; do
        if [ -d "$TRAIN_DIR/$pattern" ]; then
            train_dataset_dir="$TRAIN_DIR/$pattern"
            break
        fi
    done
    
    if [ -z "$train_dataset_dir" ]; then
        echo -e "${YELLOW}Warning: Could not find training directory for dataset $dataset_num${NC}"
        FAILED_DATASETS+=($dataset_num)
        continue
    fi
    
    echo -e "${GREEN}Found training directory: $train_dataset_dir${NC}"
    
    # Find all test directories for this dataset
    test_dataset_dirs=""
    for test_idx in {1..10}; do
        for pattern in "test_dataset_${dataset_num}_${test_idx}" "test_ds${dataset_num}_${test_idx}"; do
            if [ -d "$TEST_DIR/$pattern" ]; then
                if [ -z "$test_dataset_dirs" ]; then
                    test_dataset_dirs="/data/test/$pattern"
                else
                    test_dataset_dirs="$test_dataset_dirs /data/test/$pattern"
                fi
            fi
        done
    done
    
    # Try without index if nothing found
    if [ -z "$test_dataset_dirs" ]; then
        for pattern in "test_dataset_${dataset_num}" "test_ds${dataset_num}"; do
            if [ -d "$TEST_DIR/$pattern" ]; then
                test_dataset_dirs="/data/test/$pattern"
                break
            fi
        done
    fi
    
    if [ -z "$test_dataset_dirs" ]; then
        echo -e "${YELLOW}Warning: Could not find test directories for dataset $dataset_num${NC}"
        FAILED_DATASETS+=($dataset_num)
        continue
    fi
    
    echo -e "${GREEN}Found test directories: $test_dataset_dirs${NC}"
    echo ""
    
    # Run Docker container
    echo -e "${GREEN}Running Docker container...${NC}"
    
    if docker run --rm \
        -v "$TRAIN_DIR:/data/train" \
        -v "$TEST_DIR:/data/test" \
        -v "$OUTPUT_DIR:/output" \
        "$IMAGE_NAME" \
        --train_dir "/data/train/$(basename $train_dataset_dir)" \
        --test_dir $test_dataset_dirs \
        --out_dir /output \
        --n_jobs "$N_JOBS" \
        --device "$DEVICE"; then
        
        echo -e "${GREEN}✓ Dataset $dataset_num completed successfully${NC}"
        SUCCESS_DATASETS+=($dataset_num)
    else
        echo -e "${RED}✗ Dataset $dataset_num failed${NC}"
        FAILED_DATASETS+=($dataset_num)
    fi
    
    echo ""
done

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}PROCESSING COMPLETE${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}Successful datasets: ${SUCCESS_DATASETS[@]:-none}${NC}"
if [ ${#FAILED_DATASETS[@]} -gt 0 ]; then
    echo -e "${RED}Failed datasets: ${FAILED_DATASETS[@]}${NC}"
fi
echo ""

# Generate final submission file if all succeeded
if [ ${#FAILED_DATASETS[@]} -eq 0 ]; then
    echo -e "${GREEN}All datasets processed successfully!${NC}"
    echo -e "${GREEN}Generating final submission file...${NC}"
    
    # Use Python to concatenate outputs
    python3 << EOF
import sys
sys.path.insert(0, '/app')
from submission.utils import concatenate_output_files

try:
    concatenate_output_files('$OUTPUT_DIR', 'submission.csv')
    print("✓ Final submission file created: $OUTPUT_DIR/submission.csv")
except Exception as e:
    print(f"Error creating submission file: {e}")
    sys.exit(1)
EOF
    
else
    echo -e "${YELLOW}Some datasets failed. Please review and rerun failed datasets.${NC}"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
