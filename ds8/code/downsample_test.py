#!/usr/bin/env python3
"""
Downsample Test Dataset Samples
===============================

This script downsamples test repertoire samples using the EXACT same method
as training data to ensure consistency.

Parameters (MUST match training):
- Target templates: 10,000
- Random state: 42
- Method: Immunarch-style multinomial sampling

Usage:
    python downsample_test.py
    
Environment variables:
    DS8_TEST_DATA: Path to test dataset directory (default: ../input/test)
    DS8_TEST_CACHE: Path to cache directory for downsampled test files (default: ../cache/test)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle
import json
from datetime import datetime

# =============================================================================
# CONFIGURATION - Update these paths for your environment
# =============================================================================
CODE_DIR = Path(__file__).parent.resolve()
TEST_DATA_DIR = Path(os.environ.get('DS8_TEST_DATA', '../input/test'))
CACHE_DIR = Path(os.environ.get('DS8_TEST_CACHE', '../cache/test'))

# Create directory
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# MUST match training parameters exactly
TARGET_TEMPLATES = 10000
RANDOM_STATE = 42


def downsample_repertoire(df, target_templates, random_state=None):
    """
    Downsample a repertoire using immunarch-style multinomial sampling.
    EXACT copy of training method for consistency.
    """
    total_templates = df['templates'].sum()
    
    if total_templates <= target_templates:
        return df.copy()
    
    if random_state is not None:
        np.random.seed(random_state)
    
    # Clean data
    df_clean = df.copy()
    df_clean['templates'] = pd.to_numeric(df_clean['templates'], errors='coerce')
    
    valid_mask = (
        df_clean['templates'].notna() & 
        np.isfinite(df_clean['templates']) & 
        (df_clean['templates'] > 0)
    )
    
    df_clean = df_clean[valid_mask].copy()
    
    if len(df_clean) == 0:
        raise ValueError("No valid template values found in repertoire")
    
    total_templates_clean = df_clean['templates'].sum()
    
    if total_templates_clean <= target_templates:
        return df_clean.reset_index(drop=True)
    
    # Multinomial sampling
    templates_array = df_clean['templates'].values.astype(float)
    epsilon = 1e-10
    templates_array = templates_array + epsilon
    probabilities = templates_array / templates_array.sum()
    probabilities = np.clip(probabilities, 0.0, 1.0)
    probabilities = probabilities / probabilities.sum()
    
    try:
        sampled_counts = np.random.multinomial(target_templates, probabilities)
    except ValueError:
        indices = np.random.choice(
            len(probabilities), 
            size=target_templates, 
            replace=True, 
            p=probabilities
        )
        sampled_counts = np.bincount(indices, minlength=len(probabilities))
    
    df_downsampled = df_clean.copy()
    df_downsampled['templates'] = sampled_counts
    df_downsampled = df_downsampled[df_downsampled['templates'] > 0].copy()
    
    return df_downsampled.reset_index(drop=True)


def main():
    print("="*60)
    print("Downsampling Test Repertoire Samples")
    print("="*60)
    print(f"Test data directory: {TEST_DATA_DIR}")
    print(f"Cache directory: {CACHE_DIR}")
    print(f"Target templates per sample: {TARGET_TEMPLATES:,}")
    print(f"Random state: {RANDOM_STATE}")
    print()
    
    # Get all .tsv files (test sets don't have metadata.csv)
    tsv_files = sorted(list(TEST_DATA_DIR.glob("*.tsv")))
    
    if not tsv_files:
        print(f"ERROR: No .tsv files found in {TEST_DATA_DIR}")
        print("Please set DS8_TEST_DATA environment variable to the correct path.")
        return
    
    print(f"Found {len(tsv_files)} repertoires")
    
    stats = {
        'total_samples': len(tsv_files),
        'target_templates': TARGET_TEMPLATES,
        'random_state': RANDOM_STATE,
        'downsampled_samples': 0,
        'skipped_samples': 0,
        'errors': 0,
        'total_original_templates': 0,
        'total_downsampled_templates': 0
    }
    
    cache_info = {}
    
    print("\nDownsampling samples...")
    for tsv_file in tqdm(tsv_files):
        rep_id = tsv_file.stem
        
        try:
            df = pd.read_csv(tsv_file, sep='\t')
            
            if 'junction_aa' not in df.columns or 'templates' not in df.columns:
                stats['errors'] += 1
                continue
            
            original_templates = df['templates'].sum()
            stats['total_original_templates'] += original_templates
            
            if original_templates > TARGET_TEMPLATES:
                downsampled_df = downsample_repertoire(df, TARGET_TEMPLATES, random_state=RANDOM_STATE)
                stats['downsampled_samples'] += 1
                stats['total_downsampled_templates'] += TARGET_TEMPLATES
            else:
                downsampled_df = df.copy()
                stats['skipped_samples'] += 1
                stats['total_downsampled_templates'] += original_templates
            
            # Save to cache
            cache_file = CACHE_DIR / f"{rep_id}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(downsampled_df, f)
            
            cache_info[rep_id] = {
                'filename': tsv_file.name,
                'original_templates': int(original_templates),
                'downsampled_templates': int(downsampled_df['templates'].sum()),
                'original_unique': int(df['junction_aa'].nunique()),
                'downsampled_unique': int(downsampled_df['junction_aa'].nunique()),
                'cache_file': str(cache_file)
            }
            
        except Exception as e:
            print(f"Error processing {rep_id}: {e}")
            stats['errors'] += 1
    
    # Save cache index
    cache_index_file = CACHE_DIR / "cache_index.json"
    
    with open(cache_index_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'target_templates': TARGET_TEMPLATES,
            'random_state': RANDOM_STATE,
            'stats': stats,
            'samples': cache_info
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Downsampling Complete")
    print(f"{'='*60}")
    print(f"Total samples processed: {len(cache_info)}")
    print(f"Samples downsampled: {stats['downsampled_samples']}")
    print(f"Samples already small enough: {stats['skipped_samples']}")
    print(f"Errors: {stats['errors']}")
    print(f"Total original templates: {stats['total_original_templates']:,}")
    print(f"Total downsampled templates: {stats['total_downsampled_templates']:,}")
    if stats['total_original_templates'] > 0:
        print(f"Reduction: {(1 - stats['total_downsampled_templates']/stats['total_original_templates'])*100:.2f}%")
    print(f"\nCache saved to: {CACHE_DIR}")


if __name__ == "__main__":
    main()







