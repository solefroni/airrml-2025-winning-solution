#!/usr/bin/env python3
"""
DS2 Training Script: Dynamic Exhaustive Gapped Tokenizer → Chi-Squared Selection → XGBoost

This script reproduces the DS2 model training pipeline. It was intentionally added
for Phase 2 reproducibility (organizers may retrain on new datasets).

Pipeline:
- Dynamic exhaustive gapped tokenizer (lengths 4, 5, 6 with max 3 gaps)
- CountVectorizer (binary, min_df=2)
- SelectKBest (chi2, k=5000)
- XGBoost (n_estimators=300, max_depth=5, learning_rate=0.05)

Usage:
  Set DS2_TRAIN_DATA to the path containing metadata.csv and repertoire .tsv files.
  Example: export DS2_TRAIN_DATA=/path/to/train_dataset_2
  Then: python train.py

Output:
  Saves to ds2/model/:
  - hybrid_gapped_xgboost_pipeline.pkl
  - classification_results.json
  - top_features.csv, top_50_features.csv
"""

import os
import sys
import pandas as pd
import numpy as np
import itertools
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - Use environment variable or default
# =============================================================================
CODE_DIR = Path(__file__).parent.resolve()
TRAIN_DATA_DIR = Path(os.environ.get('DS2_TRAIN_DATA', str(CODE_DIR.parent / 'input' / 'train')))
METADATA_FILE = TRAIN_DATA_DIR / 'metadata.csv'
MODEL_DIR = CODE_DIR.parent / 'model'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Pipeline hyperparameters (match predict.py expectations)
CHI2_K = 5000
XGB_N_ESTIMATORS = 300
XGB_MAX_DEPTH = 5
XGB_LEARNING_RATE = 0.05
RANDOM_STATE = 42

# ============================================================================
# DYNAMIC GAPPED TOKENIZER (must match predict.py for compatibility)
# ============================================================================
def generate_masks(length, max_gaps):
    """Generate all binary masks for a given length with up to max_gaps. Start/end anchored."""
    inner_length = length - 2
    masks = []
    for inner in itertools.product([0, 1], repeat=inner_length):
        if inner.count(0) <= max_gaps:
            mask = (1,) + inner + (1,)
            masks.append(mask)
    return masks

MASKS = {}
for L in [4, 5, 6]:
    MASKS[L] = generate_masks(L, max_gaps=3)

def dynamic_gapped_tokenizer(sequence):
    """Tokenize sequence using all pre-computed masks for lengths 4, 5, 6."""
    tokens = []
    seq_len = len(sequence)
    for i in range(seq_len):
        for L in [4, 5, 6]:
            if i + L <= seq_len:
                sub = sequence[i : i + L]
                for mask in MASKS[L]:
                    token_parts = []
                    gap_count = 0
                    for j, bit in enumerate(mask):
                        if bit == 1:
                            if gap_count > 0:
                                token_parts.append('.' * gap_count)
                                gap_count = 0
                            token_parts.append(sub[j])
                        else:
                            gap_count += 1
                    token = "".join(token_parts)
                    tokens.append(token)
    return tokens

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("DS2: Dynamic Exhaustive Gapped Tokenizer → Chi-Squared → XGBoost")
    print("=" * 70)
    print(f"Train data directory: {TRAIN_DATA_DIR}")
    print(f"Metadata file: {METADATA_FILE}")
    print(f"Model output directory: {MODEL_DIR}")
    print()

    if not METADATA_FILE.exists():
        print(f"ERROR: Metadata file not found: {METADATA_FILE}")
        print("Set DS2_TRAIN_DATA to the directory containing metadata.csv and repertoire .tsv files.")
        sys.exit(1)

    # Load metadata
    print("Step 1: Loading metadata...")
    metadata = pd.read_csv(METADATA_FILE)
    print(f"  Total samples: {len(metadata)}")
    print(f"  Positive: {metadata['label_positive'].sum()}, Negative: {(~metadata['label_positive']).sum()}")
    print()

    # Train/validation split (80/20)
    train_idx, val_idx = train_test_split(
        metadata.index,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=metadata['label_positive']
    )
    train_meta = metadata.loc[train_idx].reset_index(drop=True)
    val_meta = metadata.loc[val_idx].reset_index(drop=True)
    y_train = train_meta['label_positive'].values.astype(int)
    y_val = val_meta['label_positive'].values.astype(int)
    print(f"  Training: {len(train_meta)}, Validation: {len(val_meta)}")
    print()

    # Load sequences as space-separated text per sample
    print("Step 2: Loading repertoire sequences...")
    def load_sequences_text(meta_df):
        texts = []
        for _, row in meta_df.iterrows():
            fpath = TRAIN_DATA_DIR / row['filename']
            if not fpath.exists():
                texts.append("")
                continue
            try:
                df = pd.read_csv(fpath, sep='\t', usecols=['junction_aa'], low_memory=False)
                df = df.dropna(subset=['junction_aa'])
                seqs = [str(s).strip() for s in df['junction_aa'].tolist() if len(str(s).strip()) >= 3]
                texts.append(" ".join(seqs))
            except Exception:
                texts.append("")
        return texts

    train_texts = load_sequences_text(train_meta)
    val_texts = load_sequences_text(val_meta)
    print(f"  Loaded {len(train_texts)} training samples")
    print()

    # Build pipeline
    print("Step 3: Building pipeline and training...")
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(
            tokenizer=dynamic_gapped_tokenizer,
            binary=True,
            lowercase=False,
            min_df=2,
            max_features=None,
            token_pattern=None
        )),
        ('selection', SelectKBest(score_func=chi2, k=CHI2_K)),
        ('classifier', xgb.XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS,
            max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE,
            n_jobs=-1,
            eval_metric='logloss',
            random_state=RANDOM_STATE,
            use_label_encoder=False
        ))
    ])

    pipeline.fit(train_texts, y_train)
    vectorizer = pipeline.named_steps['vectorizer']
    selector = pipeline.named_steps['selection']
    classifier = pipeline.named_steps['classifier']

    n_total = len(vectorizer.get_feature_names_out())
    n_selected = selector.get_support().sum()
    print(f"  Total features: {n_total:,}, Selected (chi2): {n_selected:,}")
    print()

    # Evaluate
    print("Step 4: Evaluating...")
    cv_scores = cross_val_score(pipeline, train_texts, y_train, cv=5, scoring='roc_auc')
    y_val_proba = pipeline.predict_proba(val_texts)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_proba)
    print(f"  CV AUC (5-fold): {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    print(f"  Validation AUC: {val_auc:.4f}")
    print()

    # Feature names and importance
    all_names = np.array(vectorizer.get_feature_names_out())
    sel_mask = selector.get_support()
    selected_names = all_names[sel_mask]
    importances = classifier.feature_importances_
    order = np.argsort(importances)[::-1]

    # Save pipeline
    pipeline_path = MODEL_DIR / 'hybrid_gapped_xgboost_pipeline.pkl'
    with open(pipeline_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"  Saved pipeline: {pipeline_path}")

    # Save top features
    top_df = pd.DataFrame({
        'feature': selected_names[order],
        'importance': importances[order]
    })
    top_df.to_csv(MODEL_DIR / 'top_features.csv', index=False)
    top_df.head(50).to_csv(MODEL_DIR / 'top_50_features.csv', index=False)
    print(f"  Saved top_features.csv, top_50_features.csv")

    # Save results JSON
    results = {
        'cv_auc_mean': float(cv_scores.mean()),
        'cv_auc_std': float(cv_scores.std()),
        'test_auc': float(val_auc),
        'n_features_total': int(n_total),
        'n_features_selected': int(n_selected),
    }
    with open(MODEL_DIR / 'classification_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved classification_results.json")
    print()
    print("=" * 70)
    print("DS2 training complete.")
    print("=" * 70)

if __name__ == '__main__':
    main()
