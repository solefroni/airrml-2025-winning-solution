#!/usr/bin/env python3
"""
Classify AIRRML Dataset 3 positive and negative samples using SARSCoV2 TCR sequence abundance.
Uses SARSCoV2 CDR3 sequences as features to predict AIRRML labels.
Similar to the Proinsulin classifier for Dataset 1.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# =============================================================================
# CONFIGURATION - Update these paths for your environment
# =============================================================================
CODE_DIR = Path(__file__).parent.resolve()
# External reference data: Parse Bioscience SARS-CoV-2 TCR sequences
PARSE_FILE = Path(os.environ.get('PARSE_SARSCOV2_FILE', '../reference/tcr_with_antigens_tagged.tsv'))
TRAIN_DATASET_DIR = Path(os.environ.get('DS3_TRAIN_DATA', '../input'))
METADATA_FILE = TRAIN_DATASET_DIR / 'metadata.csv'
MODEL_DIR = CODE_DIR.parent / 'model'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = MODEL_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("AIRRML DATASET 3 CLASSIFICATION USING SARSCoV2 TCR SEQUENCE ABUNDANCE")
print("=" * 80)
print()

# Step 1: Extract SARSCoV2 CDR3 sequences
print("Step 1: Extracting SARSCoV2 CDR3 sequences from Parse dataset...")
sarscov2_cdr3s = set()
sarscov2_counts = Counter()

chunk_size = 100000
for chunk in pd.read_csv(PARSE_FILE, sep='\t', chunksize=chunk_size,
                         usecols=['cdr3_aa', 'sample', 'is_positive'],
                         low_memory=False):
    sarscov2_chunk = chunk[(chunk['sample'] == 'SARSCoV2') & 
                            (chunk['is_positive'] == True) & 
                            (chunk['cdr3_aa'].notna())]
    if len(sarscov2_chunk) > 0:
        sarscov2_cdr3s.update(sarscov2_chunk['cdr3_aa'].unique())
        sarscov2_counts.update(sarscov2_chunk['cdr3_aa'])

sarscov2_cdr3s = sorted(list(sarscov2_cdr3s))
print(f"  Found {len(sarscov2_cdr3s):,} unique SARSCoV2 CDR3 sequences")
print(f"  Total SARSCoV2 sequence occurrences: {sum(sarscov2_counts.values()):,}")
print()

# Step 2: Load AIRRML metadata
print("Step 2: Loading AIRRML Dataset 3 metadata...")
metadata = pd.read_csv(METADATA_FILE)
print(f"  Total repertoires: {len(metadata)}")
print(f"  Positive repertoires: {sum(metadata['label_positive'] == True)}")
print(f"  Negative repertoires: {sum(metadata['label_positive'] == False)}")
print()

# Step 3: Build feature matrix
print("Step 3: Building feature matrix (counting SARSCoV2 CDR3s in each repertoire)...")
print("  This may take several minutes...")

X = []  # Feature matrix
y = []  # Labels
repertoire_ids = []

for idx, row in metadata.iterrows():
    repertoire_id = row['repertoire_id']
    filename = row['filename']
    is_positive = row['label_positive']
    
    filepath = TRAIN_DATASET_DIR / filename
    if not filepath.exists():
        continue
    
    try:
        rep_df = pd.read_csv(filepath, sep='\t', usecols=['junction_aa'], low_memory=False)
        rep_df = rep_df.dropna(subset=['junction_aa'])
        
        # Count occurrences of each SARSCoV2 CDR3 in this repertoire
        rep_cdr3_counts = Counter(rep_df['junction_aa'])
        feature_vector = [rep_cdr3_counts.get(cdr3, 0) for cdr3 in sarscov2_cdr3s]
        
        X.append(feature_vector)
        y.append(1 if is_positive else 0)
        repertoire_ids.append(repertoire_id)
        
        if (idx + 1) % 50 == 0:
            print(f"    Processed {idx + 1}/{len(metadata)} repertoires...")
    except Exception as e:
        print(f"    Error processing {filename}: {e}")
        continue

X = np.array(X)
y = np.array(y)

print(f"\n  Feature matrix shape: {X.shape}")
print(f"  Positive samples: {sum(y == 1):,}")
print(f"  Negative samples: {sum(y == 0):,}")
print(f"  Total SARSCoV2 CDR3 occurrences across all repertoires: {X.sum():,}")
print(f"  Repertoires with at least one SARSCoV2 CDR3: {np.sum(X.sum(axis=1) > 0):,}")
print()

# Step 4: Split data
print("Step 4: Splitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Training set: {len(X_train):,} samples")
print(f"  Test set: {len(X_test):,} samples")
print()

# Step 5: Train classifiers
print("=" * 80)
print("TRAINING CLASSIFIERS")
print("=" * 80)
print()

results = {}

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Logistic Regression
print("1. Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr.fit(X_train_scaled, y_train)

# Cross-validation
cv_scores = cross_val_score(lr, X_train_scaled, y_train, cv=5, scoring='roc_auc')
y_pred_lr = lr.predict(X_test_scaled)
y_pred_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]
auc_lr = roc_auc_score(y_test, y_pred_proba_lr)

print(f"  CV AUC (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
print(f"  Test AUC: {auc_lr:.4f}")
print(f"  Test Accuracy: {(y_pred_lr == y_test).mean():.4f}")
print()

results['logistic_regression'] = {
    'cv_auc_mean': float(cv_scores.mean()),
    'cv_auc_std': float(cv_scores.std()),
    'test_auc': float(auc_lr),
    'test_accuracy': float((y_pred_lr == y_test).mean()),
    'classification_report': classification_report(y_test, y_pred_lr, output_dict=True)
}

# 2. Random Forest
print("2. Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
rf.fit(X_train, y_train)

cv_scores_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring='roc_auc')
y_pred_rf = rf.predict(X_test)
y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]
auc_rf = roc_auc_score(y_test, y_pred_proba_rf)

print(f"  CV AUC (5-fold): {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std() * 2:.4f})")
print(f"  Test AUC: {auc_rf:.4f}")
print(f"  Test Accuracy: {(y_pred_rf == y_test).mean():.4f}")
print()

results['random_forest'] = {
    'cv_auc_mean': float(cv_scores_rf.mean()),
    'cv_auc_std': float(cv_scores_rf.std()),
    'test_auc': float(auc_rf),
    'test_accuracy': float((y_pred_rf == y_test).mean()),
    'classification_report': classification_report(y_test, y_pred_rf, output_dict=True)
}

# 3. XGBoost
print("3. XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    scale_pos_weight=sum(y_train == 0) / sum(y_train == 1),
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)

cv_scores_xgb = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='roc_auc')
y_pred_xgb = xgb_model.predict(X_test)
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)

print(f"  CV AUC (5-fold): {cv_scores_xgb.mean():.4f} (+/- {cv_scores_xgb.std() * 2:.4f})")
print(f"  Test AUC: {auc_xgb:.4f}")
print(f"  Test Accuracy: {(y_pred_xgb == y_test).mean():.4f}")
print()

results['xgboost'] = {
    'cv_auc_mean': float(cv_scores_xgb.mean()),
    'cv_auc_std': float(cv_scores_xgb.std()),
    'test_auc': float(auc_xgb),
    'test_accuracy': float((y_pred_xgb == y_test).mean()),
    'classification_report': classification_report(y_test, y_pred_xgb, output_dict=True)
}

# Step 6: Feature importance analysis
print("=" * 80)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)
print()

# Get top important features from Random Forest
feature_importance = rf.feature_importances_
top_indices = np.argsort(feature_importance)[-20:][::-1]

print("Top 20 Most Important SARSCoV2 CDR3 Sequences (Random Forest):")
print("-" * 80)
print(f"{'Rank':<6} {'CDR3 Sequence':<40} {'Importance':>15} {'Occurrences':>15}")
print("-" * 80)
for i, idx in enumerate(top_indices, 1):
    cdr3 = sarscov2_cdr3s[idx]
    importance = feature_importance[idx]
    occurrences = X[:, idx].sum()
    print(f"{i:<6} {cdr3:<40} {importance:>15.6f} {occurrences:>15,}")

# Save top features
top_features_df = pd.DataFrame({
    'cdr3': [sarscov2_cdr3s[idx] for idx in top_indices],
    'importance': [feature_importance[idx] for idx in top_indices],
    'total_occurrences': [X[:, idx].sum() for idx in top_indices]
})
top_features_df.to_csv(OUTPUT_DIR / 'top_sarscov2_features.csv', index=False)

print()

# Step 7: Save results
print("=" * 80)
print("SAVING RESULTS")
print("=" * 80)
print()

# Add metadata
results['metadata'] = {
    'num_sarscov2_cdr3s': len(sarscov2_cdr3s),
    'num_repertoires': len(y),
    'num_positive': int(sum(y == 1)),
    'num_negative': int(sum(y == 0)),
    'train_size': len(X_train),
    'test_size': len(X_test),
    'total_sarscov2_occurrences': int(X.sum())
}

# Save results
with open(OUTPUT_DIR / 'classification_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save trained models and scaler
print("Saving trained models...")
with open(OUTPUT_DIR / 'logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(lr, f)
with open(OUTPUT_DIR / 'random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf, f)
with open(OUTPUT_DIR / 'xgboost_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
with open(OUTPUT_DIR / 'scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open(OUTPUT_DIR / 'sarscov2_cdr3s.pkl', 'wb') as f:
    pickle.dump(sarscov2_cdr3s, f)
print("  Models saved successfully")
print()

# Save confusion matrices
print("Confusion Matrices:")
print("-" * 80)
print("\nLogistic Regression:")
print(confusion_matrix(y_test, y_pred_lr))
print("\nRandom Forest:")
print(confusion_matrix(y_test, y_pred_rf))
print("\nXGBoost:")
print(confusion_matrix(y_test, y_pred_xgb))
print()

# Summary table
print("=" * 80)
print("CLASSIFICATION SUMMARY")
print("=" * 80)
print(f"{'Classifier':<20} {'CV AUC':>12} {'Test AUC':>12} {'Test Accuracy':>15}")
print("-" * 80)
print(f"{'Logistic Regression':<20} {cv_scores.mean():>12.4f} {auc_lr:>12.4f} {(y_pred_lr == y_test).mean():>15.4f}")
print(f"{'Random Forest':<20} {cv_scores_rf.mean():>12.4f} {auc_rf:>12.4f} {(y_pred_rf == y_test).mean():>15.4f}")
print(f"{'XGBoost':<20} {cv_scores_xgb.mean():>12.4f} {auc_xgb:>12.4f} {(y_pred_xgb == y_test).mean():>15.4f}")
print("-" * 80)
print()

print("Results saved to:", OUTPUT_DIR)
print("  - classification_results.json")
print("  - top_sarscov2_features.csv")
print("  - logistic_regression_model.pkl")
print("  - random_forest_model.pkl")
print("  - xgboost_model.pkl")
print("  - scaler.pkl")
print("  - sarscov2_cdr3s.pkl (feature list)")
print()
print("=" * 80)
print("Analysis Complete!")
print("=" * 80)

