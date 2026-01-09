"""
Generate competition submission file for DS7 (HSV classification).

This script:
1. Loads the best trained model (Enhanced - 0.7373 AUC)
2. Generates predictions for test_dataset_7_1 and test_dataset_7_2
3. Ranks training sequences by importance/witness enrichment
4. Creates submission CSV with format:
   - Test predictions: ID, dataset, probability, -999.0, -999.0, -999.0
   - Ranked sequences: ID, dataset, -999.0, junction_aa, v_call, j_call

Target: 50,000 total entries (176 test + ~49,824 ranked sequences)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import Counter
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy

# =============================================================================
# CONFIGURATION - Update these paths for your environment
# =============================================================================
CODE_DIR = Path(__file__).parent.resolve()
TRAIN_DATA_DIR = Path(os.environ.get('DS7_TRAIN_DATA', '../input/train'))
TRAIN_METADATA_PATH = TRAIN_DATA_DIR / "metadata.csv"
TEST_DATA_DIRS = [
    Path(os.environ.get('DS7_TEST1_DATA', '../input/test1')),
    Path(os.environ.get('DS7_TEST2_DATA', '../input/test2'))
]
MODEL_DIR = CODE_DIR.parent / 'model'
MODEL_PATH = MODEL_DIR / "xgboost_enhanced_model.pkl"
OUTPUT_DIR = CODE_DIR.parent / 'output'

# Target entries
N_RANKED_SEQUENCES = 50000


def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    import sys
    sys.stdout.flush()


def load_repertoire_data(filepath: Path) -> pd.DataFrame:
    """Load full repertoire data."""
    return pd.read_csv(filepath, sep='\t')


def calculate_diversity_features(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate diversity features (same as training)."""
    features = {}
    sequences = df['junction_aa'].dropna().astype(str)
    
    if len(sequences) == 0:
        return {f'diversity_{k}': 0 for k in range(10)}
    
    seq_counts = Counter(sequences)
    total = sum(seq_counts.values())
    frequencies = np.array([count/total for count in seq_counts.values()])
    
    features['diversity_shannon'] = entropy(frequencies)
    features['diversity_simpson'] = 1 - np.sum(frequencies ** 2)
    features['clonality'] = 1 / (1 + features['diversity_shannon'])
    features['unique_ratio'] = len(seq_counts) / len(sequences)
    features['top_clone_freq'] = max(frequencies)
    top_10_freqs = sorted(frequencies, reverse=True)[:10]
    features['top10_clone_freq'] = sum(top_10_freqs)
    
    return features


def calculate_vj_features(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate V/J gene features (same as training)."""
    features = {}
    
    if 'v_call' in df.columns:
        v_genes = df['v_call'].dropna().astype(str)
        if len(v_genes) > 0:
            v_counts = Counter(v_genes)
            v_freqs = np.array([count/len(v_genes) for count in v_counts.values()])
            features['v_diversity'] = entropy(v_freqs)
            features['v_unique_count'] = len(v_counts)
            features['v_top_freq'] = max(v_freqs) if len(v_freqs) > 0 else 0
    
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
    """Calculate CDR3 length features (same as training)."""
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
    
    for bin_start in range(8, 24, 4):
        bin_end = bin_start + 4
        count = sum(1 for l in lengths if bin_start <= l < bin_end)
        features[f'cdr3_length_bin_{bin_start}_{bin_end}'] = count / len(lengths)
    
    return features


def extract_features_for_repertoire(rep_id: str, filepath: Path, 
                                    differential_sequences: List[str]) -> Dict[str, float]:
    """Extract all 622 features for a single repertoire."""
    df = load_repertoire_data(filepath)
    features = {'repertoire_id': rep_id}
    
    # 1. Binary sequence features
    sequences = set(df['junction_aa'].dropna().astype(str).unique())
    for j, diff_seq in enumerate(differential_sequences):
        features[f'seq_{j}'] = 1 if diff_seq in sequences else 0
    
    # 2. Diversity features
    features.update(calculate_diversity_features(df))
    
    # 3. V/J features
    features.update(calculate_vj_features(df))
    
    # 4. CDR3 length features
    features.update(calculate_cdr3_length_features(df))
    
    # 5. Repertoire size
    features['repertoire_size'] = len(df)
    
    return features


def rank_training_sequences_by_witness(train_metadata: pd.DataFrame,
                                       train_data_dir: Path,
                                       model,
                                       differential_sequences: List[str],
                                       n_sequences: int = N_RANKED_SEQUENCES) -> pd.DataFrame:
    """
    Rank training sequences using witness enrichment scoring.
    
    Witness score: log2((count_in_positive + 1) / (count_in_negative + 1))
    Higher score = more enriched in positive samples
    """
    print_flush(f"\nRanking training sequences by witness enrichment...")
    
    # Separate positive and negative samples
    positive_reps = train_metadata[train_metadata['label_positive'] == True]
    negative_reps = train_metadata[train_metadata['label_positive'] == False]
    
    print_flush(f"  Positive repertoires: {len(positive_reps)}")
    print_flush(f"  Negative repertoires: {len(negative_reps)}")
    
    # Count sequences in positive and negative samples
    positive_sequence_counts = Counter()
    negative_sequence_counts = Counter()
    sequence_info = {}  # Store sequence details (v_call, j_call, rep_id)
    
    print_flush("\nCounting sequences in positive samples...")
    for _, row in tqdm(positive_reps.iterrows(), total=len(positive_reps)):
        filepath = train_data_dir / row['filename']
        if not filepath.exists():
            continue
        
        df = load_repertoire_data(filepath)
        for _, seq_row in df.iterrows():
            seq = str(seq_row['junction_aa'])
            if pd.notna(seq) and len(seq) > 0:
                positive_sequence_counts[seq] += 1
                if seq not in sequence_info:
                    sequence_info[seq] = {
                        'v_call': str(seq_row.get('v_call', '')),
                        'j_call': str(seq_row.get('j_call', '')),
                        'first_seen_in': row['repertoire_id']
                    }
    
    print_flush("\nCounting sequences in negative samples...")
    for _, row in tqdm(negative_reps.iterrows(), total=len(negative_reps)):
        filepath = train_data_dir / row['filename']
        if not filepath.exists():
            continue
        
        df = load_repertoire_data(filepath)
        for _, seq_row in df.iterrows():
            seq = str(seq_row['junction_aa'])
            if pd.notna(seq) and len(seq) > 0:
                negative_sequence_counts[seq] += 1
                if seq not in sequence_info:
                    sequence_info[seq] = {
                        'v_call': str(seq_row.get('v_call', '')),
                        'j_call': str(seq_row.get('j_call', '')),
                        'first_seen_in': row['repertoire_id']
                    }
    
    # Calculate witness scores
    print_flush("\nCalculating witness enrichment scores...")
    sequence_scores = []
    
    for seq in tqdm(sequence_info.keys(), desc="Scoring sequences"):
        pos_count = positive_sequence_counts.get(seq, 0)
        neg_count = negative_sequence_counts.get(seq, 0)
        
        # Witness score: log2 fold-change with pseudocounts
        witness_score = np.log2((pos_count + 1) / (neg_count + 1))
        
        # Bonus for appearing in differential sequences (model features)
        if seq in differential_sequences:
            witness_score += 2.0  # Boost sequences the model uses
        
        sequence_scores.append({
            'junction_aa': seq,
            'v_call': sequence_info[seq]['v_call'],
            'j_call': sequence_info[seq]['j_call'],
            'witness_score': witness_score,
            'pos_count': pos_count,
            'neg_count': neg_count,
            'repertoire_id': sequence_info[seq]['first_seen_in']
        })
    
    # Convert to DataFrame and sort
    ranked_df = pd.DataFrame(sequence_scores)
    ranked_df = ranked_df.sort_values('witness_score', ascending=False)
    
    print_flush(f"\nRanking statistics:")
    print_flush(f"  Total unique sequences: {len(ranked_df):,}")
    print_flush(f"  Top sequence witness score: {ranked_df.iloc[0]['witness_score']:.4f}")
    print_flush(f"  Top sequence: {ranked_df.iloc[0]['junction_aa']}")
    print_flush(f"  Returning top {n_sequences:,} sequences")
    
    return ranked_df.head(n_sequences)


def main():
    print_flush("="*70)
    print_flush("DS7 COMPETITION SUBMISSION GENERATION")
    print_flush("="*70)
    
    # Load model
    print_flush("\nLoading Enhanced model...")
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    scaler = model_data['scaler']
    differential_sequences = model_data['differential_sequences']
    feature_names = model_data['feature_names']
    optimal_threshold = model_data['optimal_threshold']
    
    print_flush(f"✓ Model loaded")
    print_flush(f"  Test AUC: {model_data['results']['test_auc']:.4f}")
    print_flush(f"  Features: {len(feature_names)}")
    print_flush(f"  Optimal threshold: {optimal_threshold:.4f}")
    
    # Generate test predictions
    print_flush("\n" + "="*70)
    print_flush("GENERATING TEST PREDICTIONS")
    print_flush("="*70)
    
    all_test_predictions = []
    
    for test_dir in TEST_DATA_DIRS:
        print_flush(f"\nProcessing {test_dir.name}...")
        
        test_files = sorted(test_dir.glob("*.tsv"))
        print_flush(f"  Found {len(test_files)} test repertoires")
        
        for test_file in tqdm(test_files, desc=f"  Predicting {test_dir.name}"):
            rep_id = test_file.stem
            
            # Extract features
            features = extract_features_for_repertoire(rep_id, test_file, differential_sequences)
            
            # Convert to array matching training feature order
            feature_values = []
            for fname in feature_names:
                feature_values.append(features.get(fname, 0))
            
            X = np.array([feature_values])
            
            # Scale and predict
            X_scaled = scaler.transform(X)
            prob = model.predict_proba(X_scaled)[0, 1]
            
            all_test_predictions.append({
                'ID': rep_id,
                'dataset': 'train_dataset_7',  # Competition format uses train_dataset_X
                'label_positive_probability': prob,
                'junction_aa': '-999.0',
                'v_call': '-999.0',
                'j_call': '-999.0'
            })
    
    print_flush(f"\n✓ Generated {len(all_test_predictions)} test predictions")
    
    # Rank training sequences
    print_flush("\n" + "="*70)
    print_flush("RANKING TRAINING SEQUENCES")
    print_flush("="*70)
    
    train_metadata = pd.read_csv(TRAIN_METADATA_PATH)
    
    ranked_sequences = rank_training_sequences_by_witness(
        train_metadata,
        TRAIN_DATA_DIR,
        model,
        differential_sequences,
        n_sequences=N_RANKED_SEQUENCES
    )
    
    # Create ranked sequence entries
    ranked_entries = []
    for idx, row in ranked_sequences.iterrows():
        ranked_entries.append({
            'ID': f"train_dataset_7_seq_top_{idx+1}",
            'dataset': 'train_dataset_7',
            'label_positive_probability': '-999.0',
            'junction_aa': row['junction_aa'],
            'v_call': row['v_call'],
            'j_call': row['j_call']
        })
    
    print_flush(f"\n✓ Created {len(ranked_entries)} ranked sequence entries")
    
    # Combine test predictions and ranked sequences
    print_flush("\n" + "="*70)
    print_flush("CREATING SUBMISSION FILE")
    print_flush("="*70)
    
    submission_data = all_test_predictions + ranked_entries
    submission_df = pd.DataFrame(submission_data)
    
    # Save submission file
    submission_path = OUTPUT_DIR / "ds7_submission.csv"
    submission_df.to_csv(submission_path, index=False)
    
    print_flush(f"\n✓ Submission file saved to {submission_path}")
    print_flush(f"  Total entries: {len(submission_df):,}")
    print_flush(f"  Test predictions: {len(all_test_predictions)}")
    print_flush(f"  Ranked sequences: {len(ranked_entries)}")
    
    # Save test predictions separately
    test_pred_df = pd.DataFrame(all_test_predictions)
    test_pred_path = OUTPUT_DIR / "ds7_test_predictions.csv"
    test_pred_df.to_csv(test_pred_path, index=False)
    print_flush(f"  Test predictions also saved to {test_pred_path}")
    
    # Show sample entries
    print_flush("\n" + "="*70)
    print_flush("SAMPLE ENTRIES")
    print_flush("="*70)
    
    print_flush("\nFirst 5 test predictions:")
    print_flush(submission_df.head(5).to_string(index=False))
    
    print_flush("\nFirst 5 ranked sequences:")
    print_flush(submission_df.iloc[len(all_test_predictions):len(all_test_predictions)+5].to_string(index=False))
    
    print_flush("\n" + "="*70)
    print_flush("SUBMISSION GENERATION COMPLETE")
    print_flush("="*70)
    print_flush(f"\nFile ready for competition: {submission_path}")
    print_flush(f"Model used: Enhanced (Test AUC: 0.7373)")
    print_flush("="*70)


if __name__ == "__main__":
    main()

