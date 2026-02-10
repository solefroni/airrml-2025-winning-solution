# AIRR-ML 2025 - Docker Submission Package

This directory contains the Docker container for the winning solution (Private Leaderboard Score: **0.69353**) of the AIRR-ML 2025 Challenge.

## Contents

```
dockers/
├── Dockerfile                 # Docker image definition
├── build_docker.sh           # Build script
├── requirements.txt          # Python dependencies
├── submission/               # Uniform interface implementation
│   ├── __init__.py
│   ├── main.py              # Main entry point
│   ├── predictor.py         # ImmuneStatePredictor class
│   └── utils.py             # Utility functions
└── README.md                # This file
```

## Quick Start

### 1. Build the Docker Image

```bash
cd /dsi/efroni-lab/sol/AIRRML_from_home/dockers
chmod +x build_docker.sh
./build_docker.sh
```

This will:
- Copy the `winningApproach/` directory into the build context
- Build the Docker image tagged as `airrml2025-winning-solution:v1.0`
- Tag it as `airrml2025-winning-solution:latest`
- Clean up temporary files

### 2. Test the Docker Image

```bash
# Show help message
docker run --rm airrml2025-winning-solution:latest --help
```

### 3. Run on a Dataset

```bash
docker run --rm \
  -v /path/to/train_data:/data/train \
  -v /path/to/test_data:/data/test \
  -v /path/to/output:/output \
  airrml2025-winning-solution:latest \
  --train_dir /data/train/train_dataset_1 \
  --test_dir /data/test/test_dataset_1_1 /data/test/test_dataset_1_2 \
  --out_dir /output \
  --n_jobs 4 \
  --device cpu
```

### 4. Save Docker Image to File

For submission to organizers:

```bash
# Save as compressed tar
docker save airrml2025-winning-solution:v1.0 | gzip > airrml2025-winning-solution_v1.0.tar.gz

# Check size
ls -lh airrml2025-winning-solution_v1.0.tar.gz
```

To load the image on another machine:

```bash
docker load < airrml2025-winning-solution_v1.0.tar.gz
```

## Command Line Interface

The Docker container provides a uniform interface as required by the organizers:

```
python3 -m submission.main [OPTIONS]

Required Arguments:
  --train_dir PATH         Directory containing training data
  --test_dir PATH [PATH]   One or more test data directories
  --out_dir PATH          Output directory for predictions

Optional Arguments:
  --n_jobs INT            Number of parallel jobs (default: 4)
  --device {cpu,cuda}     Compute device (default: cpu)
  --skip_training         Use pre-trained models (skip training)
```

## Dataset Naming Convention

The predictor automatically detects which dataset (1-8) based on directory names. Ensure your directories contain one of these patterns:

- `train_dataset_X` or `test_dataset_X_Y` (where X=1-8, Y=test set number)
- `dataset_X` or `datasetX`
- `train_dsX` or `test_dsX_Y`
- `dsX`

## Output Format

The container generates predictions in the competition format:

```csv
ID,dataset,label_positive_probability,junction_aa,v_call,j_call
sample_123,1,0.75,CASSLAPGATNEKLFF,TRBV19,TRBJ1-1
...
```

## Processing Multiple Datasets

To process all 8 datasets in Phase 2:

```bash
#!/bin/bash
# Example batch processing script

TRAIN_DIR="/path/to/train_datasets"
TEST_DIR="/path/to/test_datasets"
OUTPUT_DIR="/path/to/output"

for dataset_num in {1..8}; do
    echo "Processing dataset $dataset_num..."
    
    # Find all test sets for this dataset
    test_dirs=$(find "$TEST_DIR" -type d -name "*dataset_${dataset_num}_*" | tr '\n' ' ')
    
    docker run --rm \
      -v "$TRAIN_DIR:/data/train" \
      -v "$TEST_DIR:/data/test" \
      -v "$OUTPUT_DIR:/output" \
      airrml2025-winning-solution:latest \
      --train_dir "/data/train/train_dataset_${dataset_num}" \
      --test_dir $test_dirs \
      --out_dir /output \
      --n_jobs 4 \
      --device cpu
done

# Combine all predictions into final submission
python3 -c "
from submission.utils import concatenate_output_files
concatenate_output_files('$OUTPUT_DIR', 'final_submission.csv')
"
```

## System Requirements

### Computational Resources

- **CPU**: Minimum 4 cores recommended
- **RAM**: Minimum 16GB, 32GB+ recommended for DS8
- **Disk**: ~20GB for Docker image + data

### Software Requirements

- Docker >= 20.10
- For GPU support: NVIDIA Docker runtime

## Dataset-Specific Notes

### DS1: T1D-based (Synthetic)
- Uses VDJdb reference sequences
- Pre-trained model included

### DS2: Synthetic
- Gapped k-mer features
- XGBoost classifier

### DS3: SARS-CoV-2 (Synthetic)
- Uses Parse Bioscience reference
- Pre-trained model included

### DS4-DS6: Synthetic
- Statistical pattern discovery
- Pre-trained models included

### DS7: HSV (Real-World)
- Differential sequence features
- Pre-trained model included

### DS8: Type 1 Diabetes (Real-World)
- **Important**: Requires downsampling preprocessing
- GCN + XGBoost ensemble
- Processing time: ~2-4 hours per repertoire
- Memory intensive: 16GB+ RAM recommended

## Troubleshooting

### Out of Memory Errors (DS8)

If you encounter OOM errors on DS8:

```bash
# Increase Docker memory limit
docker run --rm --memory="32g" ...
```

### Permission Issues

If you get permission errors accessing output:

```bash
# Run with user ID mapping
docker run --rm --user $(id -u):$(id -g) ...
```

### Dataset Detection Fails

If the predictor cannot detect the dataset number:

```bash
# Ensure directory name contains "dataset_X" or "dsX" where X is 1-8
# Example: train_dataset_1, test_dataset_1_1, etc.
```

## Validation

To validate your submission file:

```python
from submission.utils import validate_submission_file

stats = validate_submission_file("submission.csv")
print(stats)
```

## Contact & Support

For issues or questions about this Docker container:
- Check the `winningApproach/README.md` for methodology details
- Review `winningApproach/METHODOLOGY.md` for algorithmic details

## License

This code is provided for the AIRR-ML 2025 Challenge Phase 2 evaluation.

## Acknowledgments

This solution represents the winning approach (Private Score: 0.69353) from the AIRR-ML 2025 Kaggle competition.
