#!/usr/bin/env python3
"""
Downsample Repertoire Samples
=============================

This script downsamples each repertoire sample to a fixed number of templates
by randomly sampling CDR3 sequences weighted by their template counts.

The downsampled data is cached for reuse by the training and prediction scripts.

This is a REQUIRED preprocessing step for DS8 (and similar large repertoire datasets)
before running train.py or predict.py.

Usage:
    python downsample_samples.py
    
Environment variables:
    DS8_TRAIN_DATA: Path to training dataset directory (default: ../input)
    DS8_CACHE_DIR: Path to cache directory for downsampled files (default: ../cache)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import pickle
import json
from datetime import datetime

# =============================================================================
# CONFIGURATION - Update these paths for your environment
# =============================================================================
CODE_DIR = Path(__file__).parent.resolve()
DATA_DIR = Path(os.environ.get('DS8_TRAIN_DATA', '../input'))
CACHE_DIR = Path(os.environ.get('DS8_CACHE_DIR', '../cache/downsample'))

# Create directories
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Downsampling parameters
TARGET_TEMPLATES = 10000  # Number of templates per repertoire after downsampling
RANDOM_STATE = 42


def downsample_repertoire(df, target_templates, random_state=None):
    """
    Downsample a repertoire to target_templates using immunarch-style method.
    
    Uses multinomial sampling to select individual templates (not sequences)
    to match target count. This preserves the stochastic nature of sampling.
    
    Args:
        df: DataFrame with 'junction_aa' and 'templates' columns
        target_templates: Target number of templates
        random_state: Random seed for reproducibility
    
    Returns:
        DataFrame with downsampled sequences
    """
    total_templates = df['templates'].sum()
    
    if total_templates <= target_templates:
        # Already smaller than target, return as is
        return df.copy()
    
    # Set random seed
    if random_state is not None:
        np.random.seed(random_state)
    
    # Clean data: filter out invalid template values
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
    
    # Calculate probabilities for multinomial sampling
    templates_array = df_clean['templates'].values.astype(float)
    epsilon = 1e-10
    templates_array = templates_array + epsilon
    probabilities = templates_array / templates_array.sum()
    probabilities = np.clip(probabilities, 0.0, 1.0)
    probabilities = probabilities / probabilities.sum()
    
    # Sample using multinomial distribution
    try:
        sampled_counts = np.random.multinomial(target_templates, probabilities)
    except ValueError:
        # Fallback to weighted random sampling
        indices = np.random.choice(
            len(probabilities), 
            size=target_templates, 
            replace=True, 
            p=probabilities
        )
        sampled_counts = np.bincount(indices, minlength=len(probabilities))
    
    # Create downsampled DataFrame
    df_downsampled = df_clean.copy()
    df_downsampled['templates'] = sampled_counts
    df_downsampled = df_downsampled[df_downsampled['templates'] > 0].copy()
    
    return df_downsampled.reset_index(drop=True)


def main():
    print("="*60)
    print("Downsampling Repertoire Samples")
    print("="*60)
    print(f"Data directory: {DATA_DIR}")
    print(f"Cache directory: {CACHE_DIR}")
    print(f"Target templates per sample: {TARGET_TEMPLATES:,}")
    print(f"Random state: {RANDOM_STATE}")
    print()
    
    # Load metadata
    print("Loading metadata...")
    metadata_path = DATA_DIR / "metadata.csv"
    if not metadata_path.exists():
        print(f"ERROR: Metadata file not found at {metadata_path}")
        print("Please set DS8_TRAIN_DATA environment variable to the correct path.")
        return
    
    metadata_df = pd.read_csv(metadata_path)
    print(f"Total samples: {len(metadata_df)}")
    
    # Statistics
    stats = {
        'total_samples': len(metadata_df),
        'target_templates': TARGET_TEMPLATES,
        'random_state': RANDOM_STATE,
        'downsampled_samples': 0,
        'skipped_samples': 0,
        'total_original_templates': 0,
        'total_downsampled_templates': 0,
    }
    
    # Process each sample
    print("\nDownsampling samples...")
    cache_info = {}
    
    for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
        rep_id = row['repertoire_id']
        filename = row['filename']
        filepath = DATA_DIR / filename
        
        if not filepath.exists():
            continue
        
        try:
            df = pd.read_csv(filepath, sep='\t')
            if 'junction_aa' not in df.columns or 'templates' not in df.columns:
                continue
            
            original_templates = df['templates'].sum()
            stats['total_original_templates'] += original_templates
            
            # Downsample
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
                'filename': filename,
                'original_templates': int(original_templates),
                'downsampled_templates': int(downsampled_df['templates'].sum()),
                'original_unique': int(df['junction_aa'].nunique()),
                'downsampled_unique': int(downsampled_df['junction_aa'].nunique()),
                'cache_file': str(cache_file),
            }
            
        except Exception as e:
            print(f"Error processing {rep_id}: {e}")
            continue
    
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
    print(f"Total original templates: {stats['total_original_templates']:,}")
    print(f"Total downsampled templates: {stats['total_downsampled_templates']:,}")
    if stats['total_original_templates'] > 0:
        print(f"Reduction: {(1 - stats['total_downsampled_templates']/stats['total_original_templates'])*100:.2f}%")
    print(f"\nCache index saved to: {cache_index_file}")
    print(f"Cache directory: {CACHE_DIR}")


if __name__ == "__main__":
    main()







