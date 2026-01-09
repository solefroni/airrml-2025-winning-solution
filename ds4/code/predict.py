#!/usr/bin/env python3
"""
Generate predictions for DS4 using the Hybrid Gapped XGBoost model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import pickle
import json
import sys
import xgboost as xgb
import os

# Configuration - Relative paths
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
MODEL_DIR = BASE_DIR / 'model'
OUTPUT_DIR = BASE_DIR / 'output'

# =============================================================================
# CONFIGURATION - Update these paths for your environment
# =============================================================================
CODE_DIR = Path(__file__).parent.resolve()
TRAIN_DATASET_DIR = Path(os.environ.get('DS4_TRAIN_DATA', '../input/train'))
TEST_DATASET_DIR = Path(os.environ.get('DS4_TEST_DATA', '../input/test'))

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_artifacts():
    print("Loading model and artifacts...")
    model = pickle.load(open(MODEL_DIR / 'xgboost_model.pkl', 'rb'))
    scaler = pickle.load(open(MODEL_DIR / 'scaler.pkl', 'rb'))
    disease_patterns = pickle.load(open(MODEL_DIR / 'disease_patterns.pkl', 'rb'))
    
    with open(MODEL_DIR / 'results.json', 'r') as f:
        results = json.load(f)
        
    return model, scaler, disease_patterns, results

def extract_kmers(sequences, k_sizes=[3, 4, 5, 6]):
    """Extract k-mers (standard + gapped) from a list of sequences."""
    counters = {k: Counter() for k in k_sizes}
    
    for seq in sequences:
        seq_len = len(seq)
        for k in k_sizes:
            if seq_len >= k:
                # Standard
                for i in range(seq_len - k + 1):
                    counters[k][seq[i:i+k]] += 1
                
                # Gapped (1 gap) for k>=4
                if k >= 4 and seq_len >= k+1:
                    for i in range(seq_len - (k+1) + 1):
                        sub = seq[i:i+k+1]
                        mid = k // 2
                        gapped = sub[:mid] + '.' + sub[mid+1:]
                        counters[k][gapped] += 1
                        
    return counters

def extract_features_single(filepath, disease_patterns_by_k):
    """Extract features for a single repertoire file."""
    try:
        df = pd.read_csv(filepath, sep='\t', usecols=['junction_aa', 'v_call', 'j_call'], low_memory=False)
        df = df.dropna(subset=['junction_aa'])
        sequences = df['junction_aa'].tolist()
        v_genes = df['v_call'].dropna().tolist()
        j_genes = df['j_call'].dropna().tolist()
    except:
        return None
        
    # Extract k-mers
    kmers_by_k = extract_kmers(sequences, k_sizes=[3, 4, 5, 6])
    
    features = {}
    
    # 1. Pattern Features
    for k, patterns in disease_patterns_by_k.items():
        if not patterns:
            continue
            
        counter = kmers_by_k[k]
        present = [p for p in patterns if p in counter]
        
        count = len(present)
        abundance = sum(counter[p] for p in present)
        
        if abundance > 0:
            probs = [counter[p]/abundance for p in present]
            diversity = -sum(p * np.log(p) for p in probs)
        else:
            diversity = 0
            
        features[f'k{k}_count'] = count
        features[f'k{k}_abundance'] = abundance
        features[f'k{k}_diversity'] = diversity
        
    # 2. V/J Features
    features['v_unique_count'] = len(set(v_genes))
    features['j_unique_count'] = len(set(j_genes))
    features['v_diversity'] = len(set(v_genes))/len(v_genes) if v_genes else 0
    features['j_diversity'] = len(set(j_genes))/len(j_genes) if j_genes else 0
    
    return features

def generate_predictions():
    model, scaler, disease_patterns, results = load_artifacts()
    feature_names = results['feature_names']
    
    # Process Test Files
    print(f"Processing Test Files from {TEST_DATASET_DIR}...")
    test_predictions = []
    
    test_files = sorted(list(TEST_DATASET_DIR.glob('*.tsv')))
    if not test_files:
        print(f"Warning: No test files found in {TEST_DATASET_DIR}")
        return

    for i, filepath in enumerate(test_files):
        if i % 50 == 0: print(f"  {i}/{len(test_files)}")
        
        features = extract_features_single(filepath, disease_patterns)
        if features is None:
            # Fallback
            prob = 0.5
        else:
            # Create feature vector in correct order
            vec = [features.get(col, 0) for col in feature_names]
            vec_scaled = scaler.transform([vec])
            prob = model.predict_proba(vec_scaled)[0, 1]
            
        test_predictions.append({
            'ID': filepath.stem,
            'dataset': 'test_dataset_4',
            'label_positive_probability': prob,
            'junction_aa': -999.0,
            'v_call': -999.0,
            'j_call': -999.0
        })
        
    test_df = pd.DataFrame(test_predictions)
    output_path = OUTPUT_DIR / 'ds4_test_predictions.csv'
    test_df.to_csv(output_path, index=False)
    print(f"Saved {len(test_df)} test predictions to {output_path}")
    
    return test_df

if __name__ == "__main__":
    generate_predictions()
