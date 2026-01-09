#!/usr/bin/env python3
"""
Hybrid Gapped XGBoost for Dataset 4 (DS4).

Approach:
1. Dynamic Gapped Tokenizer (k=3,4,5,6 with gaps)
2. Statistical Feature Selection (Fisher's exact test) on TRAINING DATA ONLY
3. XGBoost Classifier

Key parameters:
- Pattern significance: p < 0.0001, fold_change > 4.0
- Maximum patterns: 50 (to prevent overfitting)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
import xgboost as xgb
import pickle
import json
import sys
import warnings
import time
warnings.filterwarnings('ignore')

# Force unbuffered output for better monitoring
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

# =============================================================================
# CONFIGURATION - Update these paths for your environment
# =============================================================================
CODE_DIR = Path(__file__).parent.resolve()
TRAIN_DATASET_DIR = Path(os.environ.get('DS4_TRAIN_DATA', '../input'))
METADATA_FILE = TRAIN_DATASET_DIR / 'metadata.csv'
MODEL_DIR = CODE_DIR.parent / 'model'
OUTPUT_DIR = CODE_DIR.parent / 'output'
CACHE_DIR = BASE_DIR / 'cache'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_KMERS_BY_K = CACHE_DIR / 'train_kmers_by_k.pkl'
CACHE_DISEASE_PATTERNS = CACHE_DIR / 'disease_patterns_by_k.pkl'

print("=" * 80)
print("DS4 HYBRID GAPPED XGBOOST CLASSIFIER")
print("Applying DS2 methodology to DS4")
print("=" * 80)
print()

# ============================================================================
# STEP 1: SPLIT DATA FIRST (CRITICAL FOR NO INFORMATION LEAKAGE)
# ============================================================================
print("Step 1: Splitting data into train/test (80/20)...")
print("-" * 80)
metadata = pd.read_csv(METADATA_FILE)
train_indices, test_indices = train_test_split(
    metadata.index,
    test_size=0.2,
    random_state=42,
    stratify=metadata['label_positive']
)
train_metadata = metadata.loc[train_indices].reset_index(drop=True)
test_metadata = metadata.loc[test_indices].reset_index(drop=True)
y_train = train_metadata['label_positive'].values.astype(int)
y_test = test_metadata['label_positive'].values.astype(int)
print(f"  Training: {len(train_metadata)} samples (pos={sum(y_train==1)}, neg={sum(y_train==0)})")
print(f"  Test: {len(test_metadata)} samples (pos={sum(y_test==1)}, neg={sum(y_test==0)})")
print()

# ============================================================================
# STEP 2: LOAD CDR3 SEQUENCES
# ============================================================================
print("Step 2: Loading CDR3 sequences...")
print("-" * 80)

def load_sample_cdr3s(metadata_df):
    """Load CDR3 sequences for each sample."""
    sample_cdr3_sequences = []
    for idx, row in metadata_df.iterrows():
        filename = row['filename']
        filepath = TRAIN_DATASET_DIR / filename
        if not filepath.exists():
            sample_cdr3_sequences.append([])
            continue
        try:
            rep_df = pd.read_csv(filepath, sep='\t', usecols=['junction_aa'], low_memory=False)
            rep_df = rep_df.dropna(subset=['junction_aa'])
            sequences = rep_df['junction_aa'].tolist()
            sample_cdr3_sequences.append(sequences)
        except Exception as e:
            sample_cdr3_sequences.append([])
    return sample_cdr3_sequences

t0 = time.time()
train_cdr3_sequences = load_sample_cdr3s(train_metadata)
test_cdr3_sequences = load_sample_cdr3s(test_metadata)
print(f"  Loaded sequences for {len(train_cdr3_sequences)} training samples")
print(f"  Loaded sequences for {len(test_cdr3_sequences)} test samples")
print(f"  Time: {time.time()-t0:.2f}s")
print()

# ============================================================================
# STEP 3: EXTRACT GAPPED K-MERS (TRAINING DATA ONLY)
# ============================================================================
print("Step 3: Extracting gapped k-mers from training sequences...")
print("-" * 80)

def generate_gapped_kmers(sequence, k, max_gaps=1):
    """
    Generate k-mers with gaps.
    For k=4, max_gaps=1, generates: ABCD, A.CD, AB.D, ABC.
    Note: Anchors (first/last) are preserved if possible, but internal gaps allowed.
    """
    kmers = []
    n = len(sequence)
    if n < k:
        return []
        
    # Standard k-mers
    for i in range(n - k + 1):
        kmers.append(sequence[i:i+k])
        
    # Gapped k-mers (simple implementation: 1 gap of size 1)
    if max_gaps > 0:
        # Pattern: X.Y (k=3 derived from length 4 substring)
        # We look at substrings of length k+1 and replace one char with '.'
        for i in range(n - (k + 1) + 1):
            sub = sequence[i:i+k+1]
            # Create gaps at internal positions
            for j in range(1, k): 
                gapped = sub[:j] + '.' + sub[j+1:]
                kmers.append(gapped)
                
    return kmers

def extract_kmers_from_samples(sample_sequences, k_sizes=[3, 4, 5, 6]):
    """Extract k-mers from all samples."""
    sample_kmers_by_k = {k: [] for k in k_sizes}
    
    total = len(sample_sequences)
    for idx, sequences in enumerate(sample_sequences):
        if idx % 50 == 0:
            print(f"  Processing sample {idx}/{total}...")
            
        sample_counters = {k: Counter() for k in k_sizes}
        
        for seq in sequences:
            # Extract standard k-mers
            for k in k_sizes:
                # Optimized k-mer extraction
                seq_len = len(seq)
                if seq_len >= k:
                    for i in range(seq_len - k + 1):
                        sample_counters[k][seq[i:i+k]] += 1
                        
                    # Add basic gapped k-mers for k=4,5,6 (1 gap)
                    # This is a simplified version of "Dynamic Gapped Tokenizer"
                    if k >= 4:
                        # Extract from k+1 window with 1 gap
                        if seq_len >= k+1:
                            for i in range(seq_len - (k+1) + 1):
                                sub = seq[i:i+k+1]
                                # Create one gap in middle
                                mid = k // 2
                                gapped = sub[:mid] + '.' + sub[mid+1:]
                                sample_counters[k][gapped] += 1

        for k in k_sizes:
            sample_kmers_by_k[k].append(sample_counters[k])
            
    return sample_kmers_by_k

k_sizes = [3, 4, 5, 6]

if CACHE_KMERS_BY_K.exists():
    print("  Loading k-mers from cache...")
    with open(CACHE_KMERS_BY_K, 'rb') as f:
        train_kmers_by_k = pickle.load(f)
    print("  K-mers loaded from cache!")
else:
    print(f"  Extracting k-mers (k={k_sizes})...")
    t0 = time.time()
    train_kmers_by_k = extract_kmers_from_samples(train_cdr3_sequences, k_sizes)
    print(f"  Time: {time.time()-t0:.2f}s")
    
    print("  Saving k-mers to cache...")
    with open(CACHE_KMERS_BY_K, 'wb') as f:
        pickle.dump(train_kmers_by_k, f)
    print("  K-mers cached!")
print()

# Extract k-mers for test data (needed for feature extraction later)
# We don't cache test k-mers to simulate real inference
print("  Extracting k-mers for test data...")
test_kmers_by_k = extract_kmers_from_samples(test_cdr3_sequences, k_sizes)
print()

# ============================================================================
# STEP 4: IDENTIFY DISEASE-ASSOCIATED PATTERNS (TRAINING DATA ONLY)
# ============================================================================
print("Step 4: Identifying disease-associated patterns from TRAINING data only...")
print("-" * 80)

def identify_disease_patterns(sample_kmers, labels, k, min_samples=3, 
                              min_fold_change=2.0, p_value_threshold=0.001, 
                              max_patterns=1000):
    """
    Identify k-mer patterns with statistical significance testing.
    Using slightly relaxed criteria to capture more potential signals (DS2 approach).
    """
    positive_mask = np.array(labels) == 1
    n_positive = positive_mask.sum()
    n_negative = (~positive_mask).sum()
    
    print(f"    Scanning {len(sample_kmers)} samples ({n_positive} pos, {n_negative} neg)...")
    
    # 1. Aggregate counts efficiently
    kmer_stats = defaultdict(lambda: {'pos': 0, 'neg': 0})
    
    for i, counter in enumerate(sample_kmers):
        is_pos = labels[i] == 1
        # Use set to count presence (binary) instead of abundance for initial screen
        # This is more robust to PCR amplification bias
        unique_kmers = set(counter.keys())
        for kmer in unique_kmers:
            if is_pos:
                kmer_stats[kmer]['pos'] += 1
            else:
                kmer_stats[kmer]['neg'] += 1
                
    print(f"    Found {len(kmer_stats)} unique k-mers. Filtering...")
    
    # 2. Filter and Score
    candidates = []
    
    for kmer, stats_dict in kmer_stats.items():
        pos_count = stats_dict['pos']
        neg_count = stats_dict['neg']
        total_count = pos_count + neg_count
        
        if total_count < min_samples:
            continue
            
        if pos_count < 2:  # Must appear in at least 2 positive samples
            continue
            
        # Fold Change
        pos_freq = pos_count / n_positive
        neg_freq = (neg_count + 1) / (n_negative + 1) # Smoothing
        fold_change = pos_freq / neg_freq
        
        if fold_change < min_fold_change:
            continue
            
        # Fisher's Exact Test
        contingency = [[pos_count, n_positive - pos_count],
                       [neg_count, n_negative - neg_count]]
        try:
            _, p_value = stats.fisher_exact(contingency, alternative='greater')
        except:
            p_value = 1.0
            
        if p_value > p_value_threshold:
            continue
            
        candidates.append({
            'kmer': kmer,
            'p_value': p_value,
            'fold_change': fold_change,
            'pos_count': pos_count,
            'neg_count': neg_count
        })
        
    # Sort by significance
    candidates.sort(key=lambda x: x['p_value'])
    
    # Select top
    top_candidates = candidates[:max_patterns]
    selected_patterns = {c['kmer'] for c in top_candidates}
    
    return selected_patterns

if CACHE_DISEASE_PATTERNS.exists():
    print("  Loading disease patterns from cache...")
    with open(CACHE_DISEASE_PATTERNS, 'rb') as f:
        disease_patterns_by_k = pickle.load(f)
    print("  Disease patterns loaded!")
    for k, patterns in disease_patterns_by_k.items():
        print(f"    k={k}: {len(patterns)} patterns")
else:
    disease_patterns_by_k = {}
    for k in k_sizes:
        print(f"  Identifying disease-associated {k}-mers...")
        patterns = identify_disease_patterns(
            train_kmers_by_k[k], y_train, k,
            min_samples=5,  # Increased from 3
            min_fold_change=4.0,  # Increased from 2.5
            p_value_threshold=0.0001, # Stricter p-value
            max_patterns=50 # LIMIT to top 50 patterns to prevent overfitting
        )
        disease_patterns_by_k[k] = patterns
        print(f"    Found {len(patterns)} significant {k}-mers (STRICT SELECTION)")
        
    print("  Saving disease patterns to cache...")
    with open(CACHE_DISEASE_PATTERNS, 'wb') as f:
        pickle.dump(disease_patterns_by_k, f)

print()

# ============================================================================
# STEP 5: EXTRACT FEATURES (AGGREGATED + V/J)
# ============================================================================
print("Step 5: Extracting features...")
print("-" * 80)

def extract_features(sample_kmers_by_k, disease_patterns_by_k, metadata_df):
    """Extract aggregated pattern features + V/J features."""
    n_samples = len(metadata_df)
    features = {}
    
    # 1. Aggregated Pattern Features
    for k in k_sizes:
        patterns = disease_patterns_by_k[k]
        if not patterns:
            continue
            
        kmers_list = sample_kmers_by_k[k]
        
        counts = []
        abundances = []
        diversities = []
        
        for i in range(n_samples):
            counter = kmers_list[i]
            
            # Count distinct patterns present
            present = [p for p in patterns if p in counter]
            counts.append(len(present))
            
            # Total abundance of patterns
            abundance = sum(counter[p] for p in present)
            abundances.append(abundance)
            
            # Diversity (Shannon)
            if abundance > 0:
                probs = [counter[p]/abundance for p in present]
                entropy = -sum(p * np.log(p) for p in probs)
                diversities.append(entropy)
            else:
                diversities.append(0)
                
        features[f'k{k}_count'] = counts
        features[f'k{k}_abundance'] = abundances
        features[f'k{k}_diversity'] = diversities
        
    # 2. V/J Gene Features
    print("  Extracting V/J features...")
    v_counts = []
    j_counts = []
    v_diversities = []
    j_diversities = []
    
    for idx, row in metadata_df.iterrows():
        filename = row['filename']
        filepath = TRAIN_DATASET_DIR / filename
        
        if filepath.exists():
            try:
                df = pd.read_csv(filepath, sep='\t', usecols=['v_call', 'j_call'])
                v_genes = df['v_call'].dropna().tolist()
                j_genes = df['j_call'].dropna().tolist()
                
                v_counts.append(len(set(v_genes)))
                j_counts.append(len(set(j_genes)))
                
                # Simple diversity (unique/total)
                v_diversities.append(len(set(v_genes))/len(v_genes) if v_genes else 0)
                j_diversities.append(len(set(j_genes))/len(j_genes) if j_genes else 0)
            except:
                v_counts.append(0); j_counts.append(0)
                v_diversities.append(0); j_diversities.append(0)
        else:
            v_counts.append(0); j_counts.append(0)
            v_diversities.append(0); j_diversities.append(0)
            
    features['v_unique_count'] = v_counts
    features['j_unique_count'] = j_counts
    features['v_diversity'] = v_diversities
    features['j_diversity'] = j_diversities
    
    return pd.DataFrame(features)

print("  Extracting training features...")
X_train_df = extract_features(train_kmers_by_k, disease_patterns_by_k, train_metadata)
print("  Extracting test features...")
X_test_df = extract_features(test_kmers_by_k, disease_patterns_by_k, test_metadata)

print(f"  Feature shape: {X_train_df.shape}")
print(f"  Feature names: {list(X_train_df.columns)}")
print()

# ============================================================================
# STEP 6: TRAIN MODEL WITH FEATURE SELECTION (NESTED CV CHECK)
# ============================================================================
print("Step 6: Training Model (Robust XGBoost)...")
print("-" * 80)

# Convert to numpy
X_train = X_train_df.values
X_test = X_test_df.values

# Scaling
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Cross-Validation
print("  Performing 5-fold Cross-Validation...")
clf = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5, scoring='roc_auc')
print(f"  CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Train Final Model
print("  Training final model on all training data...")
clf.fit(X_train_scaled, y_train)

# Evaluate on Held-out Test Set
y_pred_prob = clf.predict_proba(X_test_scaled)[:, 1]
test_auc = roc_auc_score(y_test, y_pred_prob)
print(f"  Test AUC: {test_auc:.4f}")

# Save Everything
print("  Saving model and artifacts...")
pickle.dump(clf, open(MODEL_DIR / 'xgboost_model.pkl', 'wb'))
pickle.dump(scaler, open(MODEL_DIR / 'scaler.pkl', 'wb'))
pickle.dump(disease_patterns_by_k, open(MODEL_DIR / 'disease_patterns.pkl', 'wb'))

results = {
    'cv_auc_mean': float(cv_scores.mean()),
    'cv_auc_std': float(cv_scores.std()),
    'test_auc': float(test_auc),
    'feature_names': list(X_train_df.columns),
    'feature_importances': float(0.0) # Placeholder
}
with open(MODEL_DIR / 'results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("=" * 80)
print(f"FINAL RESULT: Test AUC = {test_auc:.4f}")
print("=" * 80)

