# AIRR-ML Competition 2025 - Winning Solution

This repository contains the code for the winning solution of the AIRR-ML Competition 2025 (Private Score: 0.69353).

## Solution Overview

| Dataset | Type | Model Architecture | Key Features |
|---------|------|--------------------|--------------|
| **DS1** | Synthetic (T1D-based) | Logistic Regression (Calibrated) | VDJdb antigen-specific TCR counts (Proinsulin, GAD65) |
| **DS2** | Synthetic | XGBoost | Gapped k-mer patterns (e.g., A.K.L) selected via Chi-squared |
| **DS3** | Synthetic (SARS-CoV-2 based) | Logistic Regression | SARS-CoV-2 specific CDR3 counts from Parse Bioscience |
| **DS4** | Synthetic | XGBoost | Statistical pattern discovery (Fisher's exact test) |
| **DS5** | Synthetic | Logistic Regression | Aggregated sequence pattern statistics |
| **DS6** | Synthetic (HER2-based) | Logistic Regression | HER2/neu specific CDR3 counts from Parse Bioscience |
| **DS7** | HSV (Real-World) | XGBoost | Differential sequences + diversity metrics |
| **DS8** | Type 1 Diabetes (Real-World) | GCN + XGBoost Ensemble | Graph-based repertoire embedding + feature-based classification |

## Repository Structure

```
winningApproach/
├── README.md                # This file
├── requirements.txt         # Python dependencies
├── assemble.py              # Combines all predictions into final submission
├── ds1/ ... ds8/            # Per-dataset folders
│   ├── code/
│   │   ├── train.py         # Model training script
│   │   └── predict.py       # Prediction generation script
│   ├── model/               # Pre-trained model artifacts
│   ├── input/               # (Place input data here)
│   └── output/              # Generated predictions (created by predict.py)
└── output/                  # Final combined submission file
```

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Data Paths

Each `dsX/code/train.py` and `dsX/code/predict.py` script contains path variables at the top that must be set to point to your data locations:

- `TRAIN_DATASET_DIR`: Path to training data for that dataset
- `TEST_DATASET_DIR`: Path to test data for that dataset

Update these paths before running.

### 3. Train Models (Optional)

Pre-trained models are provided in `dsX/model/`. To retrain:

```bash
cd ds1/code && python train.py
cd ds2/code && python train.py
# ... repeat for ds3-ds8
```

### 4. Generate Predictions

For each dataset, run the prediction script:

```bash
cd ds1/code && python predict.py
cd ds2/code && python predict.py
cd ds3/code && python predict.py
cd ds4/code && python predict.py
cd ds5/code && python predict.py
cd ds6/code && python predict.py
cd ds7/code && python predict.py
cd ds8/code && python predict.py
```

Each script generates:
- `dsX/output/dsX_test_predictions.csv` — Predictions for test samples
- `dsX/output/dsX_ranked_sequences.csv` — Top 50,000 ranked training sequences

### 5. Assemble Final Submission

Combine all predictions into a single submission file:

```bash
python assemble.py
```

This creates `output/final_submission.csv` containing all test predictions and ranked sequences for all 8 datasets.

## External Data Requirements

Some models require external biological databases:

1. **VDJdb**: DS1 uses TCR sequences reactive to T1D-associated antigens (Proinsulin, GAD65)
2. **Parse Bioscience**: DS3 and DS6 use antigen-reactive TCR sequences (SARS-CoV-2, HER2/neu)

## Per-Dataset Details

### DS1: Synthetic (T1D-based)
- Counts TCRs matching VDJdb Proinsulin and GAD65 sequences
- Log-transformed diversity features with polynomial expansion
- Isotonic calibration

### DS2: Synthetic
- Gapped k-mer tokenization (k=4,5,6 with gaps)
- Chi-squared feature selection (5,000 from 2.2M features)
- XGBoost classifier
- `train.py` included for reproducibility (dynamic gapped tokenizer → Chi-squared → XGBoost pipeline)

### DS3: Synthetic (SARS-CoV-2 based)
- Counts TCRs matching Parse Bioscience SARS-CoV-2 sequences
- Logistic Regression classifier

### DS4: Synthetic
- Statistical pattern discovery using Fisher's exact test
- Strict significance thresholds (p < 0.0001, fold_change > 4.0)
- Top 50 most significant patterns as features
- XGBoost classifier

### DS5: Synthetic
- Aggregated k-mer pattern statistics
- Logistic Regression classifier

### DS6: Synthetic (HER2-based)
- Counts TCRs matching Parse Bioscience HER2/neu sequences
- Logistic Regression classifier

### DS7: HSV (Real-World)
- 600 differential TCR sequences (binary features)
- Repertoire diversity metrics (Shannon, Simpson, clonality)
- V/J gene diversity features
- XGBoost classifier

### DS8: Type 1 Diabetes (Real-World)
- Graph Convolutional Network (GCN) on TCR repertoire graphs
- XGBoost on 200 repertoire-level features
- Meta-learner ensemble combining both models
- **Requires preprocessing**: Run `downsample_samples.py` first (see below)

## Preprocessing: Downsampling (DS8 Only)

DS8 repertoires are very large (millions of sequences). Before training or prediction, you must downsample them to a manageable size.

### Step 1: Downsample Training Data

```bash
cd ds8/code

# Set environment variable to your training data location
export DS8_TRAIN_DATA=/path/to/train_dataset_8

# Run downsampling
python downsample_samples.py
```

### Step 2: Downsample Test Data

```bash
# Set environment variable to your test data location
export DS8_TEST_DATA=/path/to/test_dataset_8_1
export DS8_TEST_CACHE=../cache/test1
python downsample_test.py

# Repeat for each test set
export DS8_TEST_DATA=/path/to/test_dataset_8_2
export DS8_TEST_CACHE=../cache/test2
python downsample_test.py
```

### Parameters
- **Target templates**: 10,000 per repertoire
- **Random seed**: 42 (deterministic)
- **Method**: Immunarch-style multinomial sampling (preserves abundance distribution)

The downsampled files are cached as `.pkl` files for reuse.
