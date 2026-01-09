#!/bin/bash
# Test script for the Docker container

set -e

IMAGE_NAME="airrml2025-winning-solution:latest"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}AIRR-ML 2025 - Docker Test Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Test 1: Check if image exists
echo -e "${BLUE}[Test 1] Checking if Docker image exists...${NC}"
if docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Docker image found${NC}"
else
    echo -e "${RED}✗ Docker image not found: $IMAGE_NAME${NC}"
    echo "Please run ./build_docker.sh first"
    exit 1
fi
echo ""

# Test 2: Check image size
echo -e "${BLUE}[Test 2] Checking image size...${NC}"
IMAGE_SIZE=$(docker image inspect "$IMAGE_NAME" --format='{{.Size}}' | awk '{print $1/1024/1024/1024}')
echo "Image size: ${IMAGE_SIZE} GB"
if (( $(echo "$IMAGE_SIZE > 10" | bc -l) )); then
    echo -e "${RED}⚠ Warning: Image is quite large (>10GB)${NC}"
else
    echo -e "${GREEN}✓ Image size is reasonable${NC}"
fi
echo ""

# Test 3: Test help command
echo -e "${BLUE}[Test 3] Testing help command...${NC}"
if docker run --rm "$IMAGE_NAME" --help > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Help command works${NC}"
else
    echo -e "${RED}✗ Help command failed${NC}"
    exit 1
fi
echo ""

# Test 4: Check Python packages
echo -e "${BLUE}[Test 4] Checking Python packages...${NC}"
REQUIRED_PACKAGES=("numpy" "pandas" "scikit-learn" "xgboost" "torch" "torch_geometric")
ALL_PACKAGES_OK=true

for package in "${REQUIRED_PACKAGES[@]}"; do
    if docker run --rm --entrypoint python3 "$IMAGE_NAME" -c "import $package" 2>/dev/null; then
        echo -e "${GREEN}✓ $package${NC}"
    else
        echo -e "${RED}✗ $package not found${NC}"
        ALL_PACKAGES_OK=false
    fi
done

if [ "$ALL_PACKAGES_OK" = true ]; then
    echo -e "${GREEN}✓ All required packages present${NC}"
else
    echo -e "${RED}✗ Some packages missing${NC}"
    exit 1
fi
echo ""

# Test 5: Check if winning approach files exist
echo -e "${BLUE}[Test 5] Checking winning approach files...${NC}"
ALL_DATASETS_OK=true

for dataset_num in {1..8}; do
    if docker run --rm --entrypoint sh "$IMAGE_NAME" -c "test -f /app/winningApproach/ds${dataset_num}/code/predict.py" 2>/dev/null; then
        echo -e "${GREEN}✓ DS${dataset_num} predict.py${NC}"
    else
        echo -e "${RED}✗ DS${dataset_num} predict.py not found${NC}"
        ALL_DATASETS_OK=false
    fi
done

if [ "$ALL_DATASETS_OK" = true ]; then
    echo -e "${GREEN}✓ All dataset files present${NC}"
else
    echo -e "${RED}✗ Some dataset files missing${NC}"
    exit 1
fi
echo ""

# Test 6: Check submission module
echo -e "${BLUE}[Test 6] Checking submission module...${NC}"
if docker run --rm --entrypoint python3 "$IMAGE_NAME" -c "from submission.predictor import ImmuneStatePredictor" 2>/dev/null; then
    echo -e "${GREEN}✓ Submission module imports correctly${NC}"
else
    echo -e "${RED}✗ Submission module import failed${NC}"
    exit 1
fi
echo ""

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✓ ALL TESTS PASSED${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "The Docker container is ready for use!"
echo ""
echo "Next steps:"
echo "1. Test on actual data:"
echo "   docker run --rm -v /path/to/data:/data -v /path/to/output:/output \\"
echo "     $IMAGE_NAME \\"
echo "     --train_dir /data/train_dataset_1 \\"
echo "     --test_dir /data/test_dataset_1_1 \\"
echo "     --out_dir /output --n_jobs 4 --device cpu"
echo ""
echo "2. Save image for submission:"
echo "   docker save $IMAGE_NAME | gzip > airrml2025_submission.tar.gz"
echo ""
