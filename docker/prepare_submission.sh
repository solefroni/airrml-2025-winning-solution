#!/bin/bash
# Prepare complete submission package for Phase 2

set -e

# Configuration
IMAGE_NAME="airrml2025-winning-solution"
IMAGE_TAG="v1.0"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"
SUBMISSION_DIR="submission_package"
KAGGLE_SUBMISSION_FILE="../submission/BEST_SUBMISSION_0.84003.csv"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}AIRR-ML 2025 - Submission Preparation${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if Docker image exists
if ! docker image inspect "$FULL_IMAGE_NAME" > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker image not found: $FULL_IMAGE_NAME${NC}"
    echo "Please run ./build_docker.sh first"
    exit 1
fi

# Create submission directory
echo -e "${GREEN}[1/6] Creating submission directory...${NC}"
rm -rf "$SUBMISSION_DIR"
mkdir -p "$SUBMISSION_DIR"
echo "✓ Created $SUBMISSION_DIR"
echo ""

# Save Docker image
echo -e "${GREEN}[2/6] Saving Docker image...${NC}"
echo "This may take several minutes..."
docker save "$FULL_IMAGE_NAME" | gzip > "$SUBMISSION_DIR/${IMAGE_NAME}_${IMAGE_TAG}.tar.gz"
IMAGE_SIZE=$(du -h "$SUBMISSION_DIR/${IMAGE_NAME}_${IMAGE_TAG}.tar.gz" | cut -f1)
echo "✓ Saved Docker image: ${IMAGE_SIZE}"
echo ""

# Copy Kaggle submission file
echo -e "${GREEN}[3/6] Copying Kaggle submission file...${NC}"
if [ -f "$KAGGLE_SUBMISSION_FILE" ]; then
    cp "$KAGGLE_SUBMISSION_FILE" "$SUBMISSION_DIR/kaggle_submission.csv"
    echo "✓ Copied Kaggle submission file"
else
    echo -e "${YELLOW}⚠ Warning: Kaggle submission file not found at: $KAGGLE_SUBMISSION_FILE${NC}"
    echo "Please manually add the submission.csv file from Kaggle"
fi
echo ""

# Copy code repository link info
echo -e "${GREEN}[4/6] Creating repository information...${NC}"
cat > "$SUBMISSION_DIR/REPOSITORY_INFO.txt" << EOF
AIRR-ML 2025 Challenge - Phase 2 Submission
============================================

Team: solefroni
Private Leaderboard Score: 0.69353
Rank: Top 10

Code Repository
---------------
GitHub URL: [TO BE ADDED]

Repository Structure:
- winningApproach/    : Complete source code for all 8 datasets
  - ds1/ to ds8/      : Dataset-specific implementations
  - README.md         : Usage instructions
  - METHODOLOGY.md    : Detailed methodology
  - requirements.txt  : Python dependencies
  - assemble.py       : Combine predictions

Docker Container
----------------
Image file: ${IMAGE_NAME}_${IMAGE_TAG}.tar.gz
Image size: ${IMAGE_SIZE}

To load the Docker image:
  docker load < ${IMAGE_NAME}_${IMAGE_TAG}.tar.gz

To test the Docker image:
  docker run --rm ${IMAGE_NAME}:${IMAGE_TAG} --help

To run on a dataset:
  docker run --rm \\
    -v /path/to/train:/data/train \\
    -v /path/to/test:/data/test \\
    -v /path/to/output:/output \\
    ${IMAGE_NAME}:${IMAGE_TAG} \\
    --train_dir /data/train/train_dataset_1 \\
    --test_dir /data/test/test_dataset_1_1 \\
    --out_dir /output \\
    --n_jobs 4 \\
    --device cpu

Kaggle Submission
-----------------
File: kaggle_submission.csv
Private Score: 0.69353
Public Score: 0.84003

Contact
-------
[TO BE ADDED]

Date: $(date +%Y-%m-%d)
EOF
echo "✓ Created repository information"
echo ""

# Create usage instructions
echo -e "${GREEN}[5/6] Creating usage instructions...${NC}"
cat > "$SUBMISSION_DIR/USAGE_INSTRUCTIONS.md" << 'EOF'
# AIRR-ML 2025 Phase 2 - Usage Instructions

## Quick Start

### 1. Load Docker Image

```bash
docker load < airrml2025-winning-solution_v1.0.tar.gz
```

### 2. Verify Image

```bash
docker images | grep airrml2025
docker run --rm airrml2025-winning-solution:v1.0 --help
```

### 3. Run on a Single Dataset

```bash
docker run --rm \
  -v /path/to/train_data:/data/train \
  -v /path/to/test_data:/data/test \
  -v /path/to/output:/output \
  airrml2025-winning-solution:v1.0 \
  --train_dir /data/train/train_dataset_1 \
  --test_dir /data/test/test_dataset_1_1 \
  --out_dir /output \
  --n_jobs 4 \
  --device cpu
```

### 4. Process All Datasets

```bash
# Create a batch script
for i in {1..8}; do
  docker run --rm \
    -v /path/to/train:/data/train \
    -v /path/to/test:/data/test \
    -v /path/to/output:/output \
    airrml2025-winning-solution:v1.0 \
    --train_dir /data/train/train_dataset_${i} \
    --test_dir /data/test/test_dataset_${i}_* \
    --out_dir /output \
    --n_jobs 4 \
    --device cpu
done
```

## Command Line Options

```
--train_dir PATH         Training data directory (required)
--test_dir PATH [...]    Test data directory/directories (required)
--out_dir PATH          Output directory (required)
--n_jobs INT            Number of parallel jobs (default: 4)
--device {cpu,cuda}     Compute device (default: cpu)
--skip_training         Use pre-trained models
```

## Output Format

The container generates predictions in the competition format:

```csv
ID,dataset,label_positive_probability,junction_aa,v_call,j_call
```

## System Requirements

- **CPU**: 4+ cores recommended
- **RAM**: 16GB minimum, 32GB+ for DS8
- **Disk**: ~20GB for container + data
- **OS**: Linux (tested on Ubuntu 20.04+)

## Dataset Naming Convention

Directories should be named:
- Training: `train_dataset_X` (where X = 1-8)
- Testing: `test_dataset_X_Y` (where Y = test set number)

Alternative patterns supported:
- `dataset_X`, `datasetX`
- `train_dsX`, `test_dsX_Y`
- `dsX`

## Troubleshooting

### Out of Memory (DS8)

```bash
docker run --rm --memory="32g" ...
```

### Permission Issues

```bash
docker run --rm --user $(id -u):$(id -g) ...
```

### GPU Support

For CUDA support, install NVIDIA Docker runtime and use:

```bash
docker run --rm --gpus all ... --device cuda
```

## Performance Notes

- DS1-DS7: ~10-60 minutes per dataset
- DS8: ~2-4 hours (requires downsampling, ensemble model)
- Total estimated time for 100 datasets: ~100-200 hours on 4-core CPU

## Contact

For issues or questions, please contact the team through the competition organizers.
EOF
echo "✓ Created usage instructions"
echo ""

# Create checksums
echo -e "${GREEN}[6/6] Creating checksums...${NC}"
cd "$SUBMISSION_DIR"
sha256sum * > CHECKSUMS.txt 2>/dev/null || true
cd ..
echo "✓ Created checksums"
echo ""

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✓ SUBMISSION PACKAGE READY${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Package contents:"
ls -lh "$SUBMISSION_DIR/"
echo ""
echo "Package size:"
du -sh "$SUBMISSION_DIR/"
echo ""
echo -e "${YELLOW}IMPORTANT: Before submitting${NC}"
echo "1. Update REPOSITORY_INFO.txt with your GitHub URL and contact info"
echo "2. Verify kaggle_submission.csv is present and correct"
echo "3. Test the Docker image on sample data"
echo "4. Compress the entire package:"
echo "   tar -czf submission_package.tar.gz $SUBMISSION_DIR/"
echo ""
echo -e "${GREEN}Ready to submit to AIRR-ML 2025 Phase 2 organizers!${NC}"
echo ""
