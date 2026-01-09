#!/usr/bin/env python3
"""
Assemble Final Submission

This script combines predictions from all 8 datasets (DS1-DS8) into a single
submission file in the required competition format.

Usage:
    python assemble.py

Prerequisites:
    Run predict.py for each dataset first to generate predictions in dsX/output/
"""

import pandas as pd
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FINAL_SUBMISSION_FILE = OUTPUT_DIR / "final_submission.csv"

# Expected columns in submission format
COLUMNS = ['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']

def load_dataset_predictions(ds_num):
    """Load predictions for a single dataset from dsX/output/."""
    ds_dir = BASE_DIR / f"ds{ds_num}" / "output"
    
    all_data = []
    
    # Load test predictions
    test_file = ds_dir / f"ds{ds_num}_test_predictions.csv"
    if test_file.exists():
        df = pd.read_csv(test_file)
        all_data.append(df)
        print(f"  DS{ds_num}: Loaded {len(df)} test predictions")
    else:
        print(f"  DS{ds_num}: WARNING - No test predictions found at {test_file}")
    
    # Load ranked sequences
    ranked_file = ds_dir / f"ds{ds_num}_ranked_sequences.csv"
    if ranked_file.exists():
        df = pd.read_csv(ranked_file)
        all_data.append(df)
        print(f"  DS{ds_num}: Loaded {len(df)} ranked sequences")
    else:
        # Try alternative name
        ranked_file = ds_dir / "ranked_sequences.csv"
        if ranked_file.exists():
            df = pd.read_csv(ranked_file)
            all_data.append(df)
            print(f"  DS{ds_num}: Loaded {len(df)} ranked sequences")
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        # Ensure correct columns
        for col in COLUMNS:
            if col not in combined.columns:
                combined[col] = -999.0
        return combined[COLUMNS]
    
    return None

def main():
    print("=" * 70)
    print("ASSEMBLING FINAL SUBMISSION")
    print("=" * 70)
    print()
    print("Loading predictions from each dataset...")
    print()
    
    all_predictions = []
    
    for ds_num in range(1, 9):
        df = load_dataset_predictions(ds_num)
        if df is not None:
            all_predictions.append(df)
    
    if not all_predictions:
        print("\nERROR: No predictions found. Run predict.py for each dataset first.")
        return
    
    # Combine all
    final_df = pd.concat(all_predictions, ignore_index=True)
    
    # Save
    final_df.to_csv(FINAL_SUBMISSION_FILE, index=False)
    
    print()
    print("=" * 70)
    print(f"SUBMISSION FILE SAVED: {FINAL_SUBMISSION_FILE}")
    print(f"Total rows: {len(final_df)}")
    print("=" * 70)

if __name__ == "__main__":
    main()

