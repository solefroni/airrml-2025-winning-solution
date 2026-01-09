#!/usr/bin/env python3
"""
Generate Ranked Training Sequences ONLY
========================================

Fast optimized version that only generates the ranked training sequences.
Test predictions are already saved from previous run.

Optimizations:
- Pre-compute V/J calls using groupby (1000x faster)
- Vectorized calculations
- Skip test prediction generation
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import json
import pickle
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - Update these paths for your environment
# =============================================================================
import os
CODE_DIR = Path(__file__).parent.resolve()
TRAIN_DOWNSAMPLE_CACHE = Path(os.environ.get('DS8_TRAIN_CACHE', '../cache/train_downsample'))
TRAIN_METADATA = Path(os.environ.get('DS8_TRAIN_DATA', '../input')) / 'metadata.csv'
OUTPUT_DIR = CODE_DIR.parent / 'output'
NUM_TRAIN_SEQUENCES = 50000

print("="*70)
print("FAST RANKING: Training Sequences Only")
print("="*70)
print(f"Output: {OUTPUT_DIR}")
print()

# Load existing test predictions
print("Loading existing test predictions...")
test_files = [
    'test_dataset_8_1_predictions.csv',
    'test_dataset_8_2_predictions.csv',
    'test_dataset_8_3_predictions.csv'
]

all_test_predictions = []
for test_file in test_files:
    test_path = OUTPUT_DIR / test_file
    if test_path.exists():
        df = pd.read_csv(test_path)
        all_test_predictions.extend(df.to_dict('records'))
        print(f"  ✓ Loaded {len(df)} predictions from {test_file}")
    else:
        print(f"  ✗ WARNING: {test_file} not found!")

print(f"\nTotal test predictions: {len(all_test_predictions)}")

print()
print("="*70)
print("RANKING TRAINING SEQUENCES")
print("="*70)

# Load training metadata
print("\nLoading training metadata...")
train_metadata = pd.read_csv(TRAIN_METADATA)
print(f"  Training samples: {len(train_metadata)}")

# Collect all sequences
print("\nCollecting training sequences...")
all_sequences = []

for idx, row in tqdm(train_metadata.iterrows(), total=len(train_metadata), desc="Loading repertoires"):
    rep_id = row['repertoire_id']
    cache_file = TRAIN_DOWNSAMPLE_CACHE / f"{rep_id}.pkl"
    
    if not cache_file.exists():
        continue
    
    try:
        with open(cache_file, 'rb') as f:
            df = pickle.load(f)
        
        # Add each sequence with its frequency
        for _, seq_row in df.iterrows():
            all_sequences.append({
                'junction_aa': seq_row['junction_aa'],
                'v_call': seq_row.get('v_call', 'UNKNOWN'),
                'j_call': seq_row.get('j_call', 'UNKNOWN'),
                'templates': seq_row['templates'],
                'repertoire_id': rep_id,
                'label': row['label_positive']
            })
    except Exception as e:
        continue

print(f"  Collected: {len(all_sequences):,} total sequences")

# Convert to DataFrame
print("\nRanking by disease association...")
seq_df = pd.DataFrame(all_sequences)

# Calculate importance score: frequency in positive samples
positive_seqs = seq_df[seq_df['label'] == 1]
negative_seqs = seq_df[seq_df['label'] == 0]

# Count frequency in each group (VECTORIZED - FAST!)
pos_freq = positive_seqs.groupby('junction_aa')['templates'].sum()
neg_freq = negative_seqs.groupby('junction_aa')['templates'].sum()

# FULLY VECTORIZED scoring (NO PYTHON LOOPS!)
print("  Calculating importance scores (fully vectorized)...")

# Create DataFrame with pos/neg frequencies for all unique sequences
all_seqs = pd.DataFrame({'junction_aa': seq_df['junction_aa'].unique()})
all_seqs['pos_freq'] = all_seqs['junction_aa'].map(pos_freq).fillna(0)
all_seqs['neg_freq'] = all_seqs['junction_aa'].map(neg_freq).fillna(0)

# Vectorized calculations (MUCH FASTER!)
pos_count = all_seqs['pos_freq'] + 1  # Pseudocount
neg_count = all_seqs['neg_freq'] + 1
log_fc = np.log2(pos_count / neg_count)
total_freq = pos_count + neg_count
all_seqs['importance_score'] = log_fc * np.log(total_freq)

print(f"  Unique sequences: {len(all_seqs):,}")

# OPTIMIZED: Pre-compute most common V/J calls for all sequences using groupby
print("  Pre-computing V/J gene calls (optimized)...")
vj_calls = seq_df.groupby('junction_aa').agg({
    'v_call': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'UNKNOWN',
    'j_call': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'UNKNOWN'
}).to_dict('index')

importance_df = all_seqs
importance_df = importance_df.sort_values('importance_score', ascending=False)

print(f"  Top sequence score: {importance_df.iloc[0]['importance_score']:.4f}")

# Select top N sequences
top_sequences = importance_df.head(NUM_TRAIN_SEQUENCES)

print(f"\nSelecting top {NUM_TRAIN_SEQUENCES:,} sequences...")

# OPTIMIZED: Build ranked list using pre-computed V/J calls (FAST!)
ranked_sequences = []
for idx, (_, row) in enumerate(tqdm(top_sequences.iterrows(), total=len(top_sequences), desc="Building ranked list")):
    seq = row['junction_aa']
    
    # Lookup pre-computed V/J calls (O(1) operation!)
    v_call = vj_calls.get(seq, {}).get('v_call', 'UNKNOWN')
    j_call = vj_calls.get(seq, {}).get('j_call', 'UNKNOWN')
    
    ranked_sequences.append({
        'ID': f'train_dataset_8_seq_top_{idx + 1}',
        'dataset': 'train_dataset_8',
        'label_positive_probability': -999.0,
        'junction_aa': seq,
        'v_call': v_call,
        'j_call': j_call
    })

print(f"  ✓ Ranked {len(ranked_sequences):,} sequences")

# Save ranked sequences
ranked_df = pd.DataFrame(ranked_sequences)
ranked_file = OUTPUT_DIR / 'ranked_sequences.csv'
ranked_df.to_csv(ranked_file, index=False)
print(f"\n✓ Saved ranked sequences to: {ranked_file}")

# Combine all predictions into final submission file
print("\n"+"="*70)
print("CREATING FINAL SUBMISSION FILE")
print("="*70)

final_predictions = all_test_predictions + ranked_sequences
final_df = pd.DataFrame(final_predictions)

# Ensure correct column order
final_df = final_df[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]

submission_file = OUTPUT_DIR / 'ds8_ensemble_submission.csv'
final_df.to_csv(submission_file, index=False)

print(f"\n✓ Final submission file created!")
print(f"  Location: {submission_file}")
print(f"  Total entries: {len(final_df):,}")
print(f"    - Test predictions: {len(all_test_predictions):,}")
print(f"    - Ranked sequences: {len(ranked_sequences):,}")

# Save metadata
metadata = {
    'created': datetime.now().isoformat(),
    'test_predictions': len(all_test_predictions),
    'ranked_sequences': len(ranked_sequences),
    'total_entries': len(final_df),
    'test_datasets': ['test_dataset_8_1', 'test_dataset_8_2', 'test_dataset_8_3'],
    'num_train_sequences': NUM_TRAIN_SEQUENCES,
    'models': ['GCN', 'XGBoost', 'Ensemble']
}

metadata_file = OUTPUT_DIR / 'submission_metadata.json'
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n✓ Metadata saved to: {metadata_file}")

print("\n"+"="*70)
print("✓ SUCCESS - All files generated!")
print("="*70)
print(f"\nSubmission ready at: {submission_file}")

