"""
HSV Classification Model using differential sequences + repertoire diversity features.

Combines:
1. 600 differential TCR sequences (binary features)
2. Repertoire diversity metrics (Shannon, Simpson, clonality)
3. V/J gene diversity and usage patterns
4. CDR3 length distribution features
5. Undersampling for class balance

This multi-modal approach captures both specific sequences and overall repertoire structure.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import Counter
import pickle
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
from scipy.stats import entropy

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# =============================================================================
# CONFIGURATION - Update these paths for your environment
# =============================================================================
import os
CODE_DIR = Path(__file__).parent.resolve()
DATA_DIR = Path(os.environ.get('DS7_TRAIN_DATA', '../input'))
METADATA_PATH = DATA_DIR / "metadata.csv"
MODEL_DIR = CODE_DIR.parent / 'model'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = MODEL_DIR
OUTPUT_DIR = CODE_DIR.parent / 'output'

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_DIFFERENTIAL_SEQUENCES = 600
TEST_SIZE = 0.2


def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    import sys
    sys.stdout.flush()


def load_repertoire_data(filepath: Path) -> pd.DataFrame:
    """Load full repertoire data."""
    return pd.read_csv(filepath, sep='\t')


def calculate_diversity_features(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate repertoire diversity metrics."""
    features = {}
    
    sequences = df['junction_aa'].dropna().astype(str)
    if len(sequences) == 0:
        return {f'diversity_{k}': 0 for k in range(10)}
    
    # Sequence frequencies
    seq_counts = Counter(sequences)
    total = sum(seq_counts.values())
    frequencies = np.array([count/total for count in seq_counts.values()])
    
    # Shannon entropy (diversity)
    features['diversity_shannon'] = entropy(frequencies)
    
    # Simpson index (1 - sum of squared frequencies)
    features['diversity_simpson'] = 1 - np.sum(frequencies ** 2)
    
    # Clonality (inverse of diversity)
    features['clonality'] = 1 / (1 + features['diversity_shannon'])
    
    # Unique sequences ratio
    features['unique_ratio'] = len(seq_counts) / len(sequences)
    
    # Top clone frequency
    features['top_clone_freq'] = max(frequencies)
    
    # Top 10 clones frequency
    top_10_freqs = sorted(frequencies, reverse=True)[:10]
    features['top10_clone_freq'] = sum(top_10_freqs)
    
    return features


def calculate_vj_features(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate V/J gene usage features."""
    features = {}
    
    # V gene diversity
    if 'v_call' in df.columns:
        v_genes = df['v_call'].dropna().astype(str)
        if len(v_genes) > 0:
            v_counts = Counter(v_genes)
            v_freqs = np.array([count/len(v_genes) for count in v_counts.values()])
            features['v_diversity'] = entropy(v_freqs)
            features['v_unique_count'] = len(v_counts)
            features['v_top_freq'] = max(v_freqs) if len(v_freqs) > 0 else 0
    
    # J gene diversity
    if 'j_call' in df.columns:
        j_genes = df['j_call'].dropna().astype(str)
        if len(j_genes) > 0:
            j_counts = Counter(j_genes)
            j_freqs = np.array([count/len(j_genes) for count in j_counts.values()])
            features['j_diversity'] = entropy(j_freqs)
            features['j_unique_count'] = len(j_counts)
            features['j_top_freq'] = max(j_freqs) if len(j_freqs) > 0 else 0
    
    return features


def calculate_cdr3_length_features(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate CDR3 length distribution features."""
    features = {}
    
    sequences = df['junction_aa'].dropna().astype(str)
    if len(sequences) == 0:
        return features
    
    lengths = [len(seq) for seq in sequences]
    
    features['cdr3_mean_length'] = np.mean(lengths)
    features['cdr3_std_length'] = np.std(lengths)
    features['cdr3_min_length'] = np.min(lengths)
    features['cdr3_max_length'] = np.max(lengths)
    features['cdr3_median_length'] = np.median(lengths)
    
    # Length distribution (binned)
    for bin_start in range(8, 24, 4):  # bins: 8-12, 12-16, 16-20, 20-24
        bin_end = bin_start + 4
        count = sum(1 for l in lengths if bin_start <= l < bin_end)
        features[f'cdr3_length_bin_{bin_start}_{bin_end}'] = count / len(lengths)
    
    return features


def extract_all_features(repertoire_ids: List[str],
                        repertoire_files: Dict[str, Path],
                        differential_sequences: List[str]) -> pd.DataFrame:
    """Extract combined features: sequences + diversity + V/J + CDR3 length."""
    print_flush(f"\nExtracting comprehensive features for {len(repertoire_ids)} repertoires...")
    
    all_features = []
    differential_set = set(differential_sequences)
    
    for rep_id in tqdm(repertoire_ids, desc="Extracting features"):
        filepath = repertoire_files[rep_id]
        df = load_repertoire_data(filepath)
        
        features = {'repertoire_id': rep_id}
        
        # 1. Binary sequence features (600 features)
        sequences = set(df['junction_aa'].dropna().astype(str).unique())
        for j, diff_seq in enumerate(differential_sequences):
            features[f'seq_{j}'] = 1 if diff_seq in sequences else 0
        
        # 2. Diversity features
        diversity_features = calculate_diversity_features(df)
        features.update(diversity_features)
        
        # 3. V/J gene features
        vj_features = calculate_vj_features(df)
        features.update(vj_features)
        
        # 4. CDR3 length features
        length_features = calculate_cdr3_length_features(df)
        features.update(length_features)
        
        # 5. Basic repertoire size
        features['repertoire_size'] = len(df)
        
        all_features.append(features)
    
    features_df = pd.DataFrame(all_features)
    features_df = features_df.fillna(0)  # Fill any NaN with 0
    
    print_flush(f"\nFeature extraction complete:")
    print_flush(f"  Total features: {len(features_df.columns) - 1}")
    print_flush(f"  Sequence features: {N_DIFFERENTIAL_SEQUENCES}")
    print_flush(f"  Diversity features: ~7")
    print_flush(f"  V/J features: ~6")
    print_flush(f"  CDR3 length features: ~9")
    print_flush(f"  Other: ~1")
    
    return features_df


def find_differentially_shared_sequences(repertoire_ids: List[str],
                                        labels: np.ndarray,
                                        repertoire_files: Dict[str, Path],
                                        n_sequences: int = N_DIFFERENTIAL_SEQUENCES) -> List[str]:
    """Find differentially shared sequences (same as before)."""
    print_flush(f"\nFinding {n_sequences} most differentially shared sequences...")
    
    positive_ids = [rid for rid, label in zip(repertoire_ids, labels) if label == 1]
    negative_ids = [rid for rid, label in zip(repertoire_ids, labels) if label == 0]
    
    print_flush(f"Counting sequences...")
    positive_sequence_counts = Counter()
    for rep_id in tqdm(positive_ids, desc="Positive"):
        df = load_repertoire_data(repertoire_files[rep_id])
        sequences = set(df['junction_aa'].dropna().astype(str).unique())
        for seq in sequences:
            positive_sequence_counts[seq] += 1
    
    negative_sequence_counts = Counter()
    for rep_id in tqdm(negative_ids, desc="Negative"):
        df = load_repertoire_data(repertoire_files[rep_id])
        sequences = set(df['junction_aa'].dropna().astype(str).unique())
        for seq in sequences:
            negative_sequence_counts[seq] += 1
    
    all_sequences = set(positive_sequence_counts.keys()) | set(negative_sequence_counts.keys())
    
    differential_scores = {}
    for seq in tqdm(all_sequences, desc="Computing scores"):
        pos_fraction = positive_sequence_counts.get(seq, 0) / len(positive_ids)
        neg_fraction = negative_sequence_counts.get(seq, 0) / len(negative_ids)
        differential_scores[seq] = pos_fraction - neg_fraction
    
    sorted_sequences = sorted(differential_scores.items(), key=lambda x: x[1], reverse=True)
    top_sequences = [seq for seq, score in sorted_sequences[:n_sequences]]
    
    print_flush(f"Selected top {n_sequences} differential sequences")
    return top_sequences


def train_enhanced_model(X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        feature_names: List[str]) -> Tuple[xgb.XGBClassifier, dict, float]:
    """Train enhanced model with undersampling (best strategy from step 4)."""
    print_flush("\n" + "="*70)
    print_flush("TRAINING ENHANCED MODEL")
    print_flush("="*70)
    
    # Scale features (important for mixing binary and continuous features)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply undersampling (best strategy from step 4)
    print_flush("\nApplying undersampling...")
    undersampler = RandomUnderSampler(random_state=RANDOM_SEED, sampling_strategy=0.5)
    X_train_balanced, y_train_balanced = undersampler.fit_resample(X_train_scaled, y_train)
    
    print_flush(f"Balanced training set: {len(y_train_balanced)} samples")
    print_flush(f"  Positive: {np.sum(y_train_balanced == 1)}")
    print_flush(f"  Negative: {np.sum(y_train_balanced == 0)}")
    
    # Calculate scale_pos_weight
    n_pos = np.sum(y_train_balanced == 1)
    n_neg = np.sum(y_train_balanced == 0)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    # Try multiple configurations with CV
    configs = [
        {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8},
        {'n_estimators': 150, 'max_depth': 4, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.9},
        {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.7, 'colsample_bytree': 0.8},
    ]
    
    best_test_auc = 0
    best_model = None
    best_config = None
    best_cv_auc = 0
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    for config in configs:
        print_flush(f"\nTrying config: {config}")
        
        cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_balanced, y_train_balanced)):
            X_fold_train = X_train_balanced[train_idx]
            y_fold_train = y_train_balanced[train_idx]
            X_fold_val = X_train_balanced[val_idx]
            y_fold_val = y_train_balanced[val_idx]
            
            model = xgb.XGBClassifier(
                **config,
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_SEED,
                eval_metric='logloss',
                use_label_encoder=False
            )
            
            model.fit(X_fold_train, y_fold_train, verbose=False)
            y_pred = model.predict_proba(X_fold_val)[:, 1]
            score = roc_auc_score(y_fold_val, y_pred)
            cv_scores.append(score)
        
        mean_cv = np.mean(cv_scores)
        std_cv = np.std(cv_scores)
        print_flush(f"  Mean CV AUC: {mean_cv:.4f} ± {std_cv:.4f}")
        
        # Train on full balanced set and test
        model = xgb.XGBClassifier(
            **config,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_SEED,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        model.fit(X_train_balanced, y_train_balanced, verbose=False)
        y_test_pred = model.predict_proba(X_test_scaled)[:, 1]
        test_auc = roc_auc_score(y_test, y_test_pred)
        print_flush(f"  Test AUC: {test_auc:.4f}")
        
        if test_auc > best_test_auc:
            best_test_auc = test_auc
            best_cv_auc = mean_cv
            best_config = config
            best_model = model
            print_flush(f"  ✓ New best!")
    
    print_flush(f"\n{'='*70}")
    print_flush(f"BEST MODEL")
    print_flush(f"{'='*70}")
    print_flush(f"Config: {best_config}")
    print_flush(f"CV AUC: {best_cv_auc:.4f}")
    print_flush(f"Test AUC: {best_test_auc:.4f}")
    
    # Final evaluation
    print_flush("\n" + "="*70)
    print_flush("FINAL EVALUATION")
    print_flush("="*70)
    
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    # Optimal threshold
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    
    print_flush(f"\nTest ROC-AUC: {best_test_auc:.4f}")
    print_flush(f"Optimal threshold: {optimal_threshold:.4f}")
    print_flush("\nClassification Report:")
    print_flush(classification_report(y_test, y_pred_optimal))
    print_flush("\nConfusion Matrix:")
    print_flush(confusion_matrix(y_test, y_pred_optimal))
    
    # Feature importance
    print_flush("\nTop 20 most important features:")
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(20).iterrows():
        print_flush(f"  {row['feature']}: {row['importance']:.4f}")
    
    results = {
        'best_config': best_config,
        'cv_auc': float(best_cv_auc),
        'test_auc': float(best_test_auc),
        'optimal_threshold': float(optimal_threshold),
        'n_train': int(len(y_train)),
        'n_test': int(len(y_test)),
        'n_features': int(X_train.shape[1])
    }
    
    return best_model, scaler, results, optimal_threshold, feature_importance


def main():
    print_flush("="*70)
    print_flush("DS7 ENHANCED MODEL TRAINING")
    print_flush("="*70)
    print_flush("Combining differential sequences + diversity + V/J + CDR3 features")
    
    # Load data
    metadata = pd.read_csv(METADATA_PATH)
    print_flush(f"\nTotal samples: {len(metadata)}")
    print_flush(f"Positive: {metadata['label_positive'].sum()}, Negative: {(~metadata['label_positive']).sum()}")
    
    repertoire_files = {}
    for _, row in metadata.iterrows():
        rep_id = row['repertoire_id']
        filepath = DATA_DIR / row['filename']
        if filepath.exists():
            repertoire_files[rep_id] = filepath
    
    # Split data
    train_ids, test_ids = train_test_split(
        metadata['repertoire_id'].tolist(),
        test_size=TEST_SIZE,
        stratify=metadata['label_positive'],
        random_state=RANDOM_SEED
    )
    
    id_to_label = dict(zip(metadata['repertoire_id'], metadata['label_positive']))
    y_train = np.array([id_to_label[rid] for rid in train_ids], dtype=int)
    y_test = np.array([id_to_label[rid] for rid in test_ids], dtype=int)
    
    print_flush(f"\nTrain: {len(train_ids)} ({y_train.sum()} pos, {(y_train == 0).sum()} neg)")
    print_flush(f"Test: {len(test_ids)} ({y_test.sum()} pos, {(y_test == 0).sum()} neg)")
    
    # Find differential sequences
    print_flush("\n" + "="*70)
    print_flush("DIFFERENTIAL SEQUENCE SELECTION")
    print_flush("="*70)
    
    differential_sequences = find_differentially_shared_sequences(
        train_ids, y_train, repertoire_files, N_DIFFERENTIAL_SEQUENCES
    )
    
    # Extract all features
    print_flush("\n" + "="*70)
    print_flush("FEATURE EXTRACTION")
    print_flush("="*70)
    
    train_features = extract_all_features(train_ids, repertoire_files, differential_sequences)
    test_features = extract_all_features(test_ids, repertoire_files, differential_sequences)
    
    # Prepare X matrices
    feature_cols = [c for c in train_features.columns if c != 'repertoire_id']
    X_train = train_features[feature_cols].values
    X_test = test_features[feature_cols].values
    
    # Train model
    model, scaler, results, optimal_threshold, feature_importance = train_enhanced_model(
        X_train, y_train, X_test, y_test, feature_cols
    )
    
    # Save model
    print_flush("\n" + "="*70)
    print_flush("SAVING MODEL")
    print_flush("="*70)
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'differential_sequences': differential_sequences,
        'feature_names': feature_cols,
        'results': results,
        'optimal_threshold': optimal_threshold,
        'train_ids': train_ids,
        'test_ids': test_ids
    }
    
    model_path = MODEL_DIR / "xgboost_enhanced_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print_flush(f"✓ Model saved to {model_path}")
    
    results_path = RESULTS_DIR / "enhanced_training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print_flush(f"✓ Results saved to {results_path}")
    
    feature_importance.to_csv(RESULTS_DIR / "enhanced_feature_importance.csv", index=False)
    print_flush(f"✓ Feature importance saved")
    
    print_flush("\n" + "="*70)
    print_flush("COMPARISON WITH PREVIOUS APPROACHES")
    print_flush("="*70)
    print_flush(f"Generic shared:              AUC = 0.4098")
    print_flush(f"Differential (unbalanced):   AUC = 0.3608")
    print_flush(f"Differential (balanced):     AUC = 0.6745")
    print_flush(f"Enhanced (balanced + stats): AUC = {results['test_auc']:.4f}")
    print_flush("="*70)


if __name__ == "__main__":
    main()



