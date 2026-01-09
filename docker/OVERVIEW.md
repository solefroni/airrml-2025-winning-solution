# AIRR-ML 2025 Docker Submission - Complete Overview

## What This Package Does

This Docker submission wraps your winning AIRR-ML solution into a standardized container that the competition organizers can run on ~100 Phase 2 datasets. Your solution achieved:
- **Private Score**: 0.69353 (Top 10)
- **Public Score**: 0.84003

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Container                         │
│  ┌────────────────────────────────────────────────────────┐ │
│  │         submission/ (Uniform Interface)                │ │
│  │  ┌──────────────────────────────────────────────────┐  │ │
│  │  │  main.py: Entry point                            │  │ │
│  │  │    - Parses command line args                    │  │ │
│  │  │    - Calls ImmuneStatePredictor                  │  │ │
│  │  └──────────────────────────────────────────────────┘  │ │
│  │  ┌──────────────────────────────────────────────────┐  │ │
│  │  │  predictor.py: ImmuneStatePredictor             │  │ │
│  │  │    - Detects dataset number (1-8)               │  │ │
│  │  │    - Routes to appropriate dsX/code/            │  │ │
│  │  │    - Handles training and prediction            │  │ │
│  │  └──────────────────────────────────────────────────┘  │ │
│  │  ┌──────────────────────────────────────────────────┐  │ │
│  │  │  utils.py: Helper functions                     │  │ │
│  │  │    - Dataset pairing                            │  │ │
│  │  │    - Output concatenation                       │  │ │
│  │  └──────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │         winningApproach/ (Your Solution)              │ │
│  │                                                        │ │
│  │  ds1/  ds2/  ds3/  ds4/  ds5/  ds6/  ds7/  ds8/      │ │
│  │   ├── code/                                          │ │
│  │   │   ├── train.py    (Dataset-specific training)   │ │
│  │   │   └── predict.py  (Dataset-specific prediction) │ │
│  │   ├── model/                                         │ │
│  │   │   └── *.pkl       (Pre-trained models)          │ │
│  │   └── output/         (Generated at runtime)        │ │
│  │                                                        │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## How It Works

### 1. Input
Organizers run:
```bash
docker run --rm \
  -v /phase2/train:/data/train \
  -v /phase2/test:/data/test \
  -v /phase2/output:/output \
  airrml2025-winning-solution:v1.0 \
  --train_dir /data/train/train_dataset_3 \
  --test_dir /data/test/test_dataset_3_1 \
  --out_dir /output \
  --n_jobs 4 --device cpu
```

### 2. Processing Flow

1. **main.py** receives command line arguments
2. **ImmuneStatePredictor** is initialized
3. **Dataset detection**: Analyzes directory name to determine dataset number (1-8)
4. **Training** (if needed):
   - Locates `dsX/code/train.py`
   - Sets environment variables
   - Runs training script
   - Saves model to output directory
5. **Prediction**:
   - Locates `dsX/code/predict.py`
   - For each test directory:
     - Sets environment variables
     - Runs prediction script
   - Collects all predictions
6. **Output formatting**:
   - Ensures correct column format
   - Saves as CSV in competition format

### 3. Output
```
/output/
├── ds3_predictions.csv     # Combined test + ranked sequences
└── [model files if trained]
```

## Dataset-Specific Implementations

| Dataset | Approach | Special Notes |
|---------|----------|---------------|
| **DS1** | VDJdb + Log-transformed features | Uses external reference DB |
| **DS2** | XGBoost + Gapped k-mers | Chi-squared feature selection |
| **DS3** | Logistic Regression + SARS-CoV-2 | Parse Bioscience reference |
| **DS4** | XGBoost + Pattern discovery | Fisher's exact test |
| **DS5** | Logistic Regression + k-mers | Aggregated pattern stats |
| **DS6** | Logistic Regression + HER2/neu | Parse Bioscience reference |
| **DS7** | XGBoost + Differential sequences | 600 key sequences |
| **DS8** | GCN + XGBoost Ensemble | **Requires downsampling** |

### DS8 Special Handling
DS8 is the most complex:
- Input repertoires have millions of sequences
- Must downsample to 10,000 templates per repertoire
- Uses graph neural network + traditional ML ensemble
- Requires 16-32GB RAM
- Takes 2-4 hours per dataset

The container automatically:
1. Runs `downsample_samples.py` for training data
2. Runs `downsample_test.py` for each test set
3. Caches downsampled data for reuse
4. Trains GCN and XGBoost models
5. Combines predictions via meta-learner

## Files in This Directory

### Core Files
- **Dockerfile**: Defines Docker image (Python 3.9 + dependencies)
- **requirements.txt**: Python packages needed
- **submission/__init__.py**: Module initialization
- **submission/main.py**: Entry point (CLI interface)
- **submission/predictor.py**: Main predictor class
- **submission/utils.py**: Helper functions

### Scripts
- **build_docker.sh**: Builds Docker image
- **test_docker.sh**: Runs validation tests
- **prepare_submission.sh**: Creates submission package
- **run_all_datasets.sh**: Batch processes all datasets

### Documentation
- **README.md**: Complete documentation
- **QUICKSTART.md**: Quick reference guide
- **OVERVIEW.md**: This file
- **SUBMISSION_CHECKLIST.md**: Submission task list

### Generated (not in git)
- **winningApproach/**: Copied during build from `../winningApproach/`
- **submission_package/**: Created by `prepare_submission.sh`

## Build Process

### What Happens During Build

```bash
./build_docker.sh
```

1. **Copy source code**: `../winningApproach/` → `./winningApproach/`
2. **Build Docker image**:
   - Start from `python:3.9-slim`
   - Install system dependencies
   - Install Python packages
   - Copy winningApproach/
   - Copy submission/
   - Set up entry point
3. **Tag image**: `airrml2025-winning-solution:v1.0` and `:latest`
4. **Cleanup**: Remove temporary `./winningApproach/` copy

Result: ~3-5GB Docker image with everything needed to run your solution

## Testing Process

### What Happens During Test

```bash
./test_docker.sh
```

1. **Image exists**: Verifies Docker image built successfully
2. **Image size**: Checks reasonable size (<10GB)
3. **Help command**: Tests entry point works
4. **Python packages**: Verifies all dependencies installed
5. **Dataset files**: Checks all ds1-ds8 code present
6. **Module import**: Tests submission module loads

All tests must pass before submitting!

## Submission Package

### What's Included

```bash
./prepare_submission.sh
```

Creates `submission_package/` with:

1. **Docker image**: `airrml2025-winning-solution_v1.0.tar.gz`
   - Compressed Docker image (~2-4GB)
   - Load with: `docker load < file.tar.gz`

2. **Kaggle submission**: `kaggle_submission.csv`
   - Your final Kaggle submission (Private: 0.69353)
   - For organizers' analysis

3. **Repository info**: `REPOSITORY_INFO.txt`
   - GitHub URL (you must add this)
   - Contact information
   - Run instructions

4. **Usage guide**: `USAGE_INSTRUCTIONS.md`
   - How to load and run container
   - Command line options
   - Troubleshooting

5. **Checksums**: `CHECKSUMS.txt`
   - SHA256 hashes for verification

### Delivery Methods

**Small package (<5GB)**:
- Email attachment or
- Upload to competition portal

**Large package (>5GB)**:
- Upload to Google Drive, Dropbox, or similar
- Share link in submission email
- Or use WeTransfer, Filemail, etc.

## System Requirements

### Docker Image
- Base: Python 3.9 on Debian (slim)
- Size: ~3-5GB
- Required packages: numpy, pandas, scikit-learn, xgboost, PyTorch, PyTorch Geometric

### Runtime Requirements
- **CPU**: 4+ cores (8+ for DS8)
- **RAM**: 16GB minimum, 32GB for DS8
- **Disk**: 20GB free (for container + data + outputs)
- **OS**: Linux (tested on Ubuntu 20.04+)
- **Docker**: Version 20.10+

### GPU Support
- Optional but recommended for DS8
- Requires NVIDIA Docker runtime
- Use `--device cuda` flag

## Phase 2 Expectations

### Dataset Scale
- ~100 datasets (vs. 8 in Phase 1)
- Each dataset may have multiple test sets
- Total: ~500-1000 train+test combinations

### Processing Time (Estimated)
- **DS1-DS7**: ~10-60 minutes per dataset
- **DS8**: ~2-4 hours per dataset
- **Total Phase 2**: 100-200 CPU-hours
  - On 8-core machine: ~2-3 weeks of wall time
  - On HPC cluster: few days with parallelization

### Computational Resources (Organizers')
Likely using:
- HPC cluster or cloud instances
- Multiple parallel jobs
- Sufficient RAM for DS8 (32GB+ per job)

## What Organizers Will Do

### Option B (Your Choice)
1. Receive your Docker image
2. Load it on their infrastructure
3. For each of ~100 datasets:
   ```bash
   docker run --rm \
     -v /their/data:/data \
     -v /their/output:/output \
     airrml2025-winning-solution:v1.0 \
     --train_dir /data/train_dataset_X \
     --test_dir /data/test_dataset_X_1 ... \
     --out_dir /output \
     --n_jobs 4 --device cpu
   ```
4. Collect all predictions
5. Analyze performance across dataset types
6. Generate results for manuscript

### Timeline (Option B)
- **Jan 15**: You submit Docker + materials
- **Late Jan**: Organizers test your container
- **Feb-Mar**: Organizers run full analysis
- **Mar 30**: Results due from them
- **2025-2026**: Manuscript preparation

## Advantages of This Approach

### For You
- ✅ No need to process 100 datasets yourself
- ✅ No need for large computational resources
- ✅ Organizers handle any environment issues
- ✅ Your code runs in controlled environment

### For Organizers
- ✅ Standardized interface across all teams
- ✅ Reproducible results
- ✅ Can run on their optimized infrastructure
- ✅ Easy to parallelize across datasets

### For Science
- ✅ Fully reproducible methodology
- ✅ Open-source for community
- ✅ Standardized comparison framework
- ✅ Long-term reusability

## Troubleshooting Common Issues

### Build Fails

**Problem**: "winningApproach directory not found"
```bash
# Solution:
cd /home/dsi/solefroni/AIRRML/dockers
ls -la ../winningApproach  # Verify it exists
./build_docker.sh
```

**Problem**: "No space left on device"
```bash
# Solution: Clean up Docker
docker system prune -a
df -h  # Check disk space
```

### Tests Fail

**Problem**: Package import errors
```bash
# Solution: Check requirements.txt has correct versions
# Rebuild image:
docker rmi airrml2025-winning-solution:latest
./build_docker.sh
```

### Runtime Errors

**Problem**: Out of memory (DS8)
```bash
# Solution: Increase Docker memory
docker run --rm --memory="32g" ...
```

**Problem**: Dataset not detected
```bash
# Solution: Ensure directory name contains "dataset_X" or "dsX"
# Example: train_dataset_1, test_dataset_1_1
```

## Next Steps

1. **Build**: `./build_docker.sh`
2. **Test**: `./test_docker.sh`
3. **Create GitHub repo**: See SUBMISSION_CHECKLIST.md
4. **Prepare package**: `./prepare_submission.sh`
5. **Update info**: Edit `submission_package/REPOSITORY_INFO.txt`
6. **Submit**: Email to organizers by Jan 15, 2026

## Support

Need help?
- Check **README.md** for detailed docs
- See **QUICKSTART.md** for quick reference
- Review **SUBMISSION_CHECKLIST.md** for tasks
- Test locally before submitting

## Summary

This Docker submission provides:
- ✅ Uniform interface (compliant with organizers' requirements)
- ✅ All 8 dataset implementations included
- ✅ Pre-trained models bundled
- ✅ Automated preprocessing (DS8 downsampling)
- ✅ Comprehensive documentation
- ✅ Testing and validation tools
- ✅ Ready for Phase 2 evaluation

**You're ready to compete in Phase 2!** 🏆
