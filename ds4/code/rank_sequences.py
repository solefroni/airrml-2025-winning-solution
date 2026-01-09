#!/usr/bin/env python3
"""
Generate 50,000 ranked sequences for DS4.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import pickle
import sys
import multiprocessing as mp
import time
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
TRAIN_DATASET_DIR = Path(os.environ.get('DS4_TRAIN_DATA', '../input'))
METADATA_FILE = TRAIN_DATASET_DIR / 'metadata.csv'

# Output file
RANKED_FILE = OUTPUT_DIR / 'ds4_ranked_sequences.csv'

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_patterns():
    with open(MODEL_DIR / 'disease_patterns.pkl', 'rb') as f:
        disease_patterns_by_k = pickle.load(f)
    
    # Flatten to single set for fast lookup, but keep track of score (length)
    patterns_set = set()
    pattern_scores = {}
    
    for k, patterns in disease_patterns_by_k.items():
        for p in patterns:
            patterns_set.add(p)
            pattern_scores[p] = k
            
    return patterns_set, pattern_scores

def process_file(args):
    filepath, patterns_set, pattern_scores = args
    candidates = []
    
    try:
        df = pd.read_csv(filepath, sep='\t', usecols=['junction_aa', 'v_call', 'j_call'])
        df = df.dropna(subset=['junction_aa'])
        
        for idx, row in df.iterrows():
            seq = row['junction_aa']
            v = row['v_call']
            j = row['j_call']
            
            best_score = 0
            found = False
            
            # Check for patterns
            seq_len = len(seq)
            # Check k=6, 5, 4, 3 (prioritize longer)
            for k in [6, 5, 4, 3]:
                if seq_len >= k:
                    # Check standard
                    for i in range(seq_len - k + 1):
                        sub = seq[i:i+k]
                        if sub in patterns_set:
                            best_score = max(best_score, pattern_scores[sub])
                            found = True
                    
                    # Check gapped (1 gap)
                    if k >= 4 and seq_len >= k+1:
                        for i in range(seq_len - (k+1) + 1):
                            sub = seq[i:i+k+1]
                            mid = k // 2
                            gapped = sub[:mid] + '.' + sub[mid+1:]
                            if gapped in patterns_set:
                                best_score = max(best_score, pattern_scores[gapped])
                                found = True
                
                if found and best_score >= 6: # Optimization: Found best possible score
                    break
            
            if found:
                candidates.append({
                    'sequence': seq,
                    'v_call': v,
                    'j_call': j,
                    'score': best_score
                })
                
    except Exception:
        return []
        
    return candidates

def main():
    print("Generating DS4 ranked sequences...")
    t0 = time.time()
    
    # 1. Load Patterns
    patterns_set, pattern_scores = load_patterns()
    print(f"Loaded {len(patterns_set)} unique patterns.")
    
    # 2. Get Positive Samples First (Highest density of signals)
    metadata = pd.read_csv(METADATA_FILE)
    pos_files = metadata[metadata['label_positive'] == 1]['filename'].tolist()
    neg_files = metadata[metadata['label_positive'] == 0]['filename'].tolist()
    
    # Prioritize positive files
    all_files = [TRAIN_DATASET_DIR / f for f in pos_files] + [TRAIN_DATASET_DIR / f for f in neg_files]
    all_files = [f for f in all_files if f.exists()]
    
    print(f"Scanning {len(all_files)} files (prioritizing {len(pos_files)} positives)...")
    
    # 3. Parallel Scanning
    chunk_size = 40
    all_candidates = {} # seq -> (score, v, j)
    
    pool = mp.Pool(processes=min(48, mp.cpu_count()))
    
    for i in range(0, len(all_files), chunk_size):
        chunk = all_files[i:i+chunk_size]
        print(f"  Processing chunk {i//chunk_size + 1} ({len(chunk)} files)...")
        
        args_list = [(f, patterns_set, pattern_scores) for f in chunk]
        results = pool.map(process_file, args_list)
        
        for file_candidates in results:
            for c in file_candidates:
                seq = c['sequence']
                score = c['score']
                if seq not in all_candidates or score > all_candidates[seq][0]:
                    all_candidates[seq] = (score, c['v_call'], c['j_call'])
        
        print(f"    Found {len(all_candidates)} unique candidates so far.")
        
        # Stop early if we have enough
        if len(all_candidates) > 70000: # Buffer for filtering
            print("  Found enough candidates. Stopping scan.")
            break
            
    pool.close()
    pool.join()
    
    # 4. Rank and Select
    print("Ranking sequences...")
    ranked_list = []
    for seq, (score, v, j) in all_candidates.items():
        ranked_list.append({
            'junction_aa': seq,
            'v_call': v,
            'j_call': j,
            'score': score
        })
        
    ranked_list.sort(key=lambda x: x['score'], reverse=True)
    final_list = ranked_list[:50000]
    
    # 5. Format for Submission
    output_rows = []
    for i, item in enumerate(final_list):
        output_rows.append({
            'ID': f'train_dataset_4_seq_top_{i+1}',
            'dataset': 'train_dataset_4',
            'label_positive_probability': -999.0,
            'junction_aa': item['junction_aa'],
            'v_call': item['v_call'],
            'j_call': item['j_call']
        })
        
    df = pd.DataFrame(output_rows)
    df.to_csv(RANKED_FILE, index=False)
    
    print(f"Saved {len(df)} ranked sequences to {RANKED_FILE}")
    print(f"Total time: {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
