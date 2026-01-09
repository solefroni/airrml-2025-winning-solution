#!/usr/bin/env python3
"""
Generate Ensemble Predictions for DS8 (Type 1 Diabetes)

Creates competition-formatted submission file with:
1. Test predictions from GCN + XGBoost ensemble
2. Top 50,000 training sequences ranked by ensemble importance
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch_geometric.data import DataLoader
import json
import pickle
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add current directory for local imports
CODE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(CODE_DIR))

from ensemble_predictor import EnsemblePredictor
from cvc_embedder import CVCEmbedder
from graph_builder import GraphBuilder
from cache_utils import set_cache_dir
import graph_classification as gc_module
from extract_repertoire_features import extract_all_features

# =============================================================================
# CONFIGURATION - Update these paths for your environment
# =============================================================================
DATASET_NUM = 8

# Data directories (update these to your data locations)
TRAIN_DATA_DIR = Path(os.environ.get('DS8_TRAIN_DATA', '../input/train'))
TRAIN_DOWNSAMPLE_CACHE = Path(os.environ.get('DS8_TRAIN_CACHE', '../cache/train_downsample'))

# Test datasets (list of tuples: name, cache_path)
# Update these paths for your test data
TEST_DATASETS = [
    ("test_dataset_8_1", Path(os.environ.get('DS8_TEST1_CACHE', '../cache/test1'))),
    ("test_dataset_8_2", Path(os.environ.get('DS8_TEST2_CACHE', '../cache/test2'))),
    ("test_dataset_8_3", Path(os.environ.get('DS8_TEST3_CACHE', '../cache/test3')))
]

GRAPH_CACHE_DIR = Path(os.environ.get('DS8_GRAPH_CACHE', '../cache/graphs'))

# Output directory
OUTPUT_DIR = CODE_DIR.parent / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
K_VALUE = 30
EMBEDDER_TYPE = 'cvc'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_TRAIN_SEQUENCES = 50000

print("="*70)
print("ENSEMBLE PREDICTION GENERATION FOR DS8")
print("="*70)
print(f"Device: {DEVICE}")
print(f"Output: {OUTPUT_DIR}")
print()

# Initialize ensemble
print("Loading ensemble model...")

# Model paths (relative to ds8 folder)
MODEL_DIR = CODE_DIR.parent / 'model'
GCN_MODEL_PATH = MODEL_DIR / 'gcn_model.pth'
XGB_MODEL_PATH = MODEL_DIR / 'xgboost_model.pkl'
META_MODEL_PATH = MODEL_DIR / 'meta_learner.pkl'
ENSEMBLE_CONFIG_PATH = MODEL_DIR / 'ensemble_config.json'

# Load ensemble configuration
with open(ENSEMBLE_CONFIG_PATH, 'r') as f:
    ensemble_config = json.load(f)

print(f"  GCN validation AUC: {ensemble_config.get('gcn_val_auc', 'N/A'):.4f}")
print(f"  XGBoost validation AUC: {ensemble_config.get('xgb_val_auc', 'N/A'):.4f}")
print(f"  Ensemble validation AUC: {ensemble_config.get('ensemble_val_auc', 'N/A'):.4f}")

# Initialize ensemble with all model paths
ensemble = EnsemblePredictor(
    gcn_model_path=GCN_MODEL_PATH,
    xgb_model_path=XGB_MODEL_PATH,
    meta_model_path=META_MODEL_PATH,
    gcn_config=ensemble_config.get('gcn_config', {}),
    device=str(DEVICE)
)

print("  All models loaded successfully")
print()

# Set up graph building
set_cache_dir(GRAPH_CACHE_DIR)
gc_module.EMBEDDER_TYPE = EMBEDDER_TYPE
gc_module.GRAPH_CACHE_DIR = GRAPH_CACHE_DIR

embedder = CVCEmbedder(device=DEVICE, verbose=False)
graph_builder = GraphBuilder(embedder=embedder, k_neighbors=K_VALUE, device=str(DEVICE),
                             verbose=False, use_hnswlib=True)

# ============================================================================
# PART 1: Generate Test Predictions
# ============================================================================
print("="*70)
print("PART 1: GENERATING TEST PREDICTIONS")
print("="*70)
print()

all_test_predictions = []

for test_dataset_name, test_cache_dir in TEST_DATASETS:
    print(f"\nProcessing {test_dataset_name}...")
    print("-"*70)
    
    # Get all sample IDs from cache
    sample_files = sorted(list(test_cache_dir.glob("*.pkl")))
    sample_ids = [f.stem for f in sample_files if f.stem != 'cache_index']
    
    print(f"  Found: {len(sample_ids)} samples")
    
    if len(sample_ids) == 0:
        print(f"  WARNING: No samples found in {test_cache_dir}")
        continue
    
    # Create fake metadata (no labels for test)
    fake_metadata = pd.DataFrame({
        'repertoire_id': sample_ids,
        'label_positive': [0] * len(sample_ids)  # Dummy labels
    })
    
    # Build graphs
    print(f"  Building graphs...")
    test_graphs, _ = gc_module.build_graph_dataset(
        sample_ids, fake_metadata, test_cache_dir, graph_builder, K_VALUE
    )
    
    print(f"  Built: {len(test_graphs)} graphs")
    
    # Create graph loader
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
    
    # Extract features for XGBoost
    print(f"  Extracting features for XGBoost...")
    all_features = []
    for sample_id in tqdm(sample_ids, desc="  Feature extraction"):
        cache_file = test_cache_dir / f"{sample_id}.pkl"
        if not cache_file.exists():
            continue
        
        try:
            with open(cache_file, 'rb') as f:
                df = pickle.load(f)
            
            features = extract_all_features(df)
            features['repertoire_id'] = sample_id
            all_features.append(features)
        except Exception as e:
            print(f"    WARNING: Failed to extract features for {sample_id}: {e}")
            continue
    
    # Create features DataFrame
    features_df = pd.DataFrame(all_features)
    features_df.set_index('repertoire_id', inplace=True)
    features_df = features_df.fillna(0)
    
    print(f"  Extracted: {len(features_df)} feature sets")
    
    # Get ensemble predictions
    print(f"  Getting ensemble predictions...")
    predictions = ensemble.predict_ensemble(
        graph_loader=test_loader,
        features_df=features_df
    )
    
    # Store predictions
    for i, sample_id in enumerate(sample_ids):
        if i < len(predictions):
            all_test_predictions.append({
                'ID': sample_id,
                'dataset': test_dataset_name,
                'label_positive_probability': predictions[i],
                'junction_aa': -999.0,
                'v_call': -999.0,
                'j_call': -999.0
            })
    
    print(f"  ✓ Completed {test_dataset_name}: {len(predictions)} predictions")

print(f"\n{'='*70}")
print(f"TOTAL TEST PREDICTIONS: {len(all_test_predictions)}")
print(f"{'='*70}")
print()

# Save individual test set predictions
for test_dataset_name, _ in TEST_DATASETS:
    test_preds = [p for p in all_test_predictions if p['dataset'] == test_dataset_name]
    if test_preds:
        df = pd.DataFrame(test_preds)
        output_file = OUTPUT_DIR / f"{test_dataset_name}_predictions.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved {len(test_preds)} predictions to: {output_file.name}")

# ============================================================================
# PART 2: Rank Training Sequences
# ============================================================================
print()
print("="*70)
print("PART 2: RANKING TRAINING SEQUENCES")
print("="*70)
print()

print("Loading training metadata...")
train_metadata = pd.read_csv(TRAIN_DATA_DIR / "metadata.csv")
print(f"  Training samples: {len(train_metadata)}")

# Rank sequences by ensemble importance
print("\nRanking sequences by ensemble importance...")
print("  Strategy: Combine XGBoost feature importance + GCN node centrality")

# Get XGBoost feature importance
xgb_model = ensemble.xgb_model
if hasattr(xgb_model, 'feature_importances_'):
    feature_importance = xgb_model.feature_importances_
    print(f"  XGBoost features: {len(feature_importance)}")
else:
    print("  WARNING: No feature importance available from XGBoost")
    feature_importance = None

# Collect all sequences from training data
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

# Rank by frequency in positive samples (simple but effective)
print("\nRanking by disease association...")
seq_df = pd.DataFrame(all_sequences)

# Calculate importance score: frequency in positive samples
positive_seqs = seq_df[seq_df['label'] == 1]
negative_seqs = seq_df[seq_df['label'] == 0]

# Count frequency in each group
pos_freq = positive_seqs.groupby('junction_aa')['templates'].sum()
neg_freq = negative_seqs.groupby('junction_aa')['templates'].sum()

# Calculate log fold change (disease association score)
importance_scores = []
for seq in seq_df['junction_aa'].unique():
    pos_count = pos_freq.get(seq, 0) + 1  # Pseudocount
    neg_count = neg_freq.get(seq, 0) + 1
    
    # Log fold change
    log_fc = np.log2(pos_count / neg_count)
    
    # Also consider total frequency
    total_freq = pos_count + neg_count
    
    # Combined score
    score = log_fc * np.log(total_freq)
    
    importance_scores.append({
        'junction_aa': seq,
        'importance_score': score,
        'pos_freq': pos_count - 1,
        'neg_freq': neg_count - 1
    })

importance_df = pd.DataFrame(importance_scores)
importance_df = importance_df.sort_values('importance_score', ascending=False)

print(f"  Unique sequences: {len(importance_df):,}")
print(f"  Top sequence score: {importance_df.iloc[0]['importance_score']:.4f}")

# Select top N sequences
top_sequences = importance_df.head(NUM_TRAIN_SEQUENCES)

print(f"\nSelecting top {NUM_TRAIN_SEQUENCES:,} sequences...")

# Get V/J calls for top sequences (use most common for each sequence)
ranked_sequences = []
for idx, row in tqdm(top_sequences.iterrows(), total=len(top_sequences), desc="Building ranked list"):
    seq = row['junction_aa']
    
    # Find most common V/J for this sequence
    seq_entries = seq_df[seq_df['junction_aa'] == seq]
    if len(seq_entries) > 0:
        v_call = seq_entries['v_call'].mode()[0] if len(seq_entries['v_call'].mode()) > 0 else 'UNKNOWN'
        j_call = seq_entries['j_call'].mode()[0] if len(seq_entries['j_call'].mode()) > 0 else 'UNKNOWN'
    else:
        v_call = 'UNKNOWN'
        j_call = 'UNKNOWN'
    
    ranked_sequences.append({
        'ID': f'train_dataset_8_seq_top_{len(ranked_sequences) + 1}',
        'dataset': 'train_dataset_8',
        'label_positive_probability': -999.0,
        'junction_aa': seq,
        'v_call': v_call,
        'j_call': j_call
    })

print(f"  ✓ Ranked {len(ranked_sequences):,} sequences")

# Save ranked sequences
ranked_df = pd.DataFrame(ranked_sequences)
ranked_file = OUTPUT_DIR / "ranked_sequences.csv"
ranked_df.to_csv(ranked_file, index=False)
print(f"\nSaved ranked sequences to: {ranked_file.name}")

# ============================================================================
# PART 3: Create Final Submission File
# ============================================================================
print()
print("="*70)
print("PART 3: CREATING FINAL SUBMISSION FILE")
print("="*70)
print()

# Combine test predictions and ranked sequences
test_df = pd.DataFrame(all_test_predictions)
submission_df = pd.concat([test_df, ranked_df], ignore_index=True)

print(f"Submission file contents:")
print(f"  Test predictions: {len(test_df):,}")
print(f"  Training sequences: {len(ranked_df):,}")
print(f"  Total rows: {len(submission_df):,}")

# Save final submission
submission_file = OUTPUT_DIR / "ds8_ensemble_submission.csv"
submission_df.to_csv(submission_file, index=False)

print(f"\n✓ Saved final submission to: {submission_file}")
print(f"  Format: {list(submission_df.columns)}")
print(f"  Size: {submission_file.stat().st_size / 1024 / 1024:.2f} MB")

# Save metadata
metadata = {
    'model_type': 'ensemble',
    'components': ['GCN', 'XGBoost', 'LogisticRegression_MetaLearner'],
    'gcn_config': {k: str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v 
                   for k, v in ensemble.gcn_config.items()},
    'k_value': K_VALUE,
    'embedder_type': EMBEDDER_TYPE,
    'test_predictions': {
        ds_name: len([p for p in all_test_predictions if p['dataset'] == ds_name])
        for ds_name, _ in TEST_DATASETS
    },
    'total_test_predictions': len(test_df),
    'training_sequences': len(ranked_df),
    'total_rows': len(submission_df),
    'timestamp': datetime.now().isoformat(),
    'device': str(DEVICE)
}

metadata_file = OUTPUT_DIR / "submission_metadata.json"
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Saved metadata to: {metadata_file.name}")

# Display summary
print()
print("="*70)
print("SUBMISSION COMPLETE")
print("="*70)
print()
print("Files created:")
print(f"  1. {submission_file.name} - Main submission file ({len(submission_df):,} rows)")
for ds_name, _ in TEST_DATASETS:
    count = len([p for p in all_test_predictions if p['dataset'] == ds_name])
    print(f"  2. {ds_name}_predictions.csv - {count} predictions")
print(f"  3. ranked_sequences.csv - Top {NUM_TRAIN_SEQUENCES:,} sequences")
print(f"  4. submission_metadata.json - Metadata")
print()
print("Test prediction breakdown:")
for ds_name, _ in TEST_DATASETS:
    count = len([p for p in all_test_predictions if p['dataset'] == ds_name])
    print(f"  {ds_name}: {count} samples")
print()
print("Ready for competition submission!")
print("="*70)


