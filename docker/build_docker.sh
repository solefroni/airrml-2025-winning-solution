#!/bin/bash
# Build Docker image for AIRR-ML 2025 submission

set -e

# Configuration
IMAGE_NAME="airrml2025-winning-solution"
IMAGE_TAG="v1.0"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}AIRR-ML 2025 - Docker Build Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Get the directory containing this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if winningApproach directory exists
if [ ! -d "../winningApproach" ]; then
    echo -e "${RED}Error: winningApproach directory not found!${NC}"
    echo "Please ensure ../winningApproach exists relative to the dockers directory"
    exit 1
fi

# Copy winningApproach to build context
echo -e "${GREEN}[1/4] Copying winningApproach to build context...${NC}"
rm -rf winningApproach
cp -r ../winningApproach ./winningApproach
echo "✓ Copied winningApproach"

# Build Docker image
echo ""
echo -e "${GREEN}[2/4] Building Docker image...${NC}"
docker build -t "$FULL_IMAGE_NAME" .

# Also tag as latest
echo ""
echo -e "${GREEN}[3/4] Tagging as latest...${NC}"
docker tag "$FULL_IMAGE_NAME" "${IMAGE_NAME}:latest"

# Clean up
echo ""
echo -e "${GREEN}[4/4] Cleaning up...${NC}"
rm -rf winningApproach
echo "✓ Removed temporary build files"

# Success message
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✓ Docker image built successfully!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Image name: $FULL_IMAGE_NAME"
echo "Also tagged as: ${IMAGE_NAME}:latest"
echo ""
echo "Test the image with:"
echo "  docker run --rm $FULL_IMAGE_NAME --help"
echo ""
echo "Run on a dataset:"
echo "  docker run --rm -v /path/to/data:/data -v /path/to/output:/output \\"
echo "    $FULL_IMAGE_NAME \\"
echo "    --train_dir /data/train_dataset_1 \\"
echo "    --test_dir /data/test_dataset_1_1 \\"
echo "    --out_dir /output \\"
echo "    --n_jobs 4 \\"
echo "    --device cpu"
echo ""
echo "Save the image to a tar file:"
echo "  docker save $FULL_IMAGE_NAME | gzip > ${IMAGE_NAME}_${IMAGE_TAG}.tar.gz"
echo ""
