"""
Generate predictions for test repertoires and ranked sequence list for DS2.

This script:
1. Loads the trained DS2 hybrid gapped XGBoost model (Pipeline: Dynamic Gapped Tokenizer → Chi-Squared → XGBoost)
2. Extracts features from TEST repertoires and generates probabilities
3. Extracts sequences from TRAIN repertoires and ranks them using witness-based ranking

Model:
- Approach: Exhaustive Gapped Tokenizer → Chi-Squared Selection → XGBoost
- Tokenizer: Exhaustive gapped (lengths 4-5, captures internal residues like A.K.L, P.I.S)
- Features: 2,219,785 total → 5,000 selected via Chi-squared
- XGBoost: n_estimators=300, max_depth=5, learning_rate=0.05

Ranking Method:
- Uses witness enrichment scoring (log2 fold-change) to identify sequences
  that appear more frequently in positive samples
- Combines witness score with model feature importance
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import Counter, defaultdict
import pickle
import json
from multiprocessing import Pool, cpu_count
from functools import partial
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable, *args, **kwargs):
        return iterable
import warnings
import itertools
warnings.filterwarnings('ignore')

# ============================================================================
# DYNAMIC GAPPED TOKENIZER (required for loading the pickle model)
# ============================================================================
def generate_masks(length, max_gaps):
    """
    Generates all binary masks (tuples) for a given length with up to max_gaps.
    Ensures start/end are always 1 (anchors).
    """
    inner_length = length - 2
    masks = []
    
    for inner in itertools.product([0, 1], repeat=inner_length):
        if inner.count(0) <= max_gaps:
            mask = (1,) + inner + (1,)
            masks.append(mask)
    
    return masks

# Pre-compute masks for efficiency (Lengths 4, 5, 6)
MASKS = {}
for L in [4, 5, 6]:
    MASKS[L] = generate_masks(L, max_gaps=3)

def dynamic_gapped_tokenizer(sequence):
    """
    Dynamic tokenizer that applies all pre-computed masks for lengths 4, 5, and 6.
    This ensures we capture motifs like C.A.S.S.L.G (6-mer with gaps).
    """
    tokens = []
    seq_len = len(sequence)
    
    for i in range(seq_len):
        for L in [4, 5, 6]:
            if i + L <= seq_len:
                sub = sequence[i : i+L]
                
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

# Alias for pickle compatibility (the model was saved with this name)
exhaustive_gapped_tokenizer = dynamic_gapped_tokenizer

# =============================================================================
# CONFIGURATION - Update these paths for your environment
# =============================================================================
CODE_DIR = Path(__file__).parent.resolve()
TRAIN_DATA_DIR = Path(os.environ.get('DS2_TRAIN_DATA', '../input/train'))
TRAIN_METADATA_PATH = TRAIN_DATA_DIR / "metadata.csv"
TEST_DATA_DIR = Path(os.environ.get('DS2_TEST_DATA', '../input/test'))
MODEL_DIR = CODE_DIR.parent / 'model'
OUTPUT_DIR = CODE_DIR.parent / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Import witness ranking module (local)
sys.path.insert(0, str(CODE_DIR))
from witness_ranking import (
    calculate_witness_score,
    combine_witness_and_model_scores,
    normalize_scores,
    calculate_enrichment_metrics
)

# Helper function for flushing output
def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


# ============================================================================
# OPTIMIZED FEATURE MATCHING
# ============================================================================

def get_sequence_kmers(sequence: str) -> Set[str]:
    """
    Generate all k-mers (contiguous and gapped) for a sequence.
    Uses the same tokenizer as the model to ensure consistency.
    Returns a set for O(1) lookup.
    """
    return set(dynamic_gapped_tokenizer(sequence))


def process_sequence_batch(batch_data: List[Tuple], feature_importance_dict: Dict[str, float], 
                           top_feature_set: Set[str]) -> List[Tuple]:
    """
    Process a batch of sequences to calculate model importance scores.
    
    Args:
        batch_data: List of (junction_aa, data_dict) tuples
        feature_importance_dict: Dict mapping feature names to importance values
        top_feature_set: Set of top feature names for quick lookup
    
    Returns:
        List of (junction_aa, model_score, feature_count) tuples
    """
    results = []
    
    for junction_aa, _ in batch_data:
        # Generate all k-mers for this sequence
        seq_kmers = get_sequence_kmers(junction_aa)
        
        # Find intersection with top features
        matching_features = seq_kmers & top_feature_set
        
        if matching_features:
            # Sum importance of matching features
            total_importance = sum(feature_importance_dict[f] for f in matching_features)
            # Average importance
            model_score = total_importance / len(matching_features)
            feature_count = len(matching_features)
        else:
            model_score = 0.0
            feature_count = 0
        
        results.append((junction_aa, model_score, feature_count))
    
    return results


class DS2PredictorV2:
    """Predictor for DS2 using the hybrid gapped XGBoost model (v2)."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        
        # Will be loaded from model
        self.pipeline = None
        self.vectorizer = None
        self.selector = None
        self.classifier = None
        self.feature_names = None
        self.model_metadata = None
        
    def load_model(self):
        """Load the trained pipeline model."""
        if self.verbose:
            print_flush(f"Loading DS2 v2 model from {MODEL_DIR}...")
        
        # Load pipeline model
        model_path = MODEL_DIR / 'hybrid_gapped_xgboost_pipeline.pkl'
        with open(model_path, 'rb') as f:
            self.pipeline = pickle.load(f)
        
        # Extract components
        self.vectorizer = self.pipeline.named_steps['vectorizer']
        self.selector = self.pipeline.named_steps['selection']
        self.classifier = self.pipeline.named_steps['classifier']
        
        # Get selected feature names
        all_features = np.array(self.vectorizer.get_feature_names_out())
        selected_mask = self.selector.get_support()
        self.feature_names = all_features[selected_mask]
        
        # Load model metadata
        metadata_path = MODEL_DIR / 'classification_results.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.model_metadata = json.load(f)
        
        if self.verbose:
            print_flush(f"Model loaded successfully")
            print_flush(f"  Total features generated: {len(all_features):,}")
            print_flush(f"  Features selected: {len(self.feature_names):,}")
            if self.model_metadata:
                print_flush(f"  Test AUC: {self.model_metadata.get('test_auc', 'N/A'):.4f}")
                print_flush(f"  CV AUC: {self.model_metadata.get('cv_auc_mean', 'N/A'):.4f} (±{self.model_metadata.get('cv_auc_std', 0):.4f})")
    
    def load_sequences_as_text(self, metadata_df: pd.DataFrame, data_dir: Path) -> List[str]:
        """Load all sequences for each sample and join as space-separated text."""
        sequences_text = []
        
        for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), disable=not self.verbose):
            filename = row['filename']
            rep_file = data_dir / filename
            
            if not rep_file.exists():
                if self.verbose:
                    print_flush(f"  Warning: File not found: {rep_file}")
                sequences_text.append("")
                continue
            
            try:
                rep_df = pd.read_csv(rep_file, sep='\t', usecols=['junction_aa'], low_memory=False)
                rep_df = rep_df.dropna(subset=['junction_aa'])
                sequences = [str(s).strip() for s in rep_df['junction_aa'].tolist() if len(str(s).strip()) >= 3]
                # Join sequences with spaces for CountVectorizer
                sequences_text.append(" ".join(sequences))
            except Exception as e:
                if self.verbose:
                    print_flush(f"  Warning: Error loading {rep_file}: {e}")
                sequences_text.append("")
        
        return sequences_text
    
    def predict_test_repertoires(self):
        """Load test repertoires, extract features, and generate predictions."""
        if self.verbose:
            print_flush("\n" + "="*70)
            print_flush("PREDICTING TEST REPERTOIRES (DS2 v2)")
            print_flush("="*70)
        
        # Load test files
        test_files = sorted(list(TEST_DATA_DIR.glob("*.tsv")))
        if self.verbose:
            print_flush(f"Found {len(test_files)} test repertoire files")
        
        # Create metadata-like structure for test files
        test_metadata = pd.DataFrame({
            'repertoire_id': [f.stem for f in test_files],
            'filename': [f.name for f in test_files]
        })
        
        # Load sequences as text
        if self.verbose:
            print_flush("Loading test sequences...")
        test_sequences_text = self.load_sequences_as_text(test_metadata, TEST_DATA_DIR)
        
        # Generate predictions using pipeline
        if self.verbose:
            print_flush("Generating predictions...")
        probabilities = self.pipeline.predict_proba(test_sequences_text)[:, 1]
        
        # Create output DataFrame
        results_df = pd.DataFrame({
            'repertoire_id': test_metadata['repertoire_id'].values,
            'probability': probabilities
        })
        
        output_path = OUTPUT_DIR / "ds2_test_predictions_v2.csv"
        results_df.to_csv(output_path, index=False)
        
        if self.verbose:
            print_flush(f"\nPredictions saved to: {output_path}")
            print_flush(f"Predicted {len(results_df)} repertoires")
            print_flush(f"Probability range: [{probabilities.min():.4f}, {probabilities.max():.4f}]")
            print_flush(f"Mean probability: {probabilities.mean():.4f}")
        
        return results_df
    
    def rank_training_sequences(self, top_n: int = 50000):
        """Extract sequences from training data and rank them using witness-based ranking."""
        if self.verbose:
            print_flush("\n" + "="*70)
            print_flush("RANKING TRAINING SEQUENCES (DS2 v2 - OPTIMIZED)")
            print_flush("="*70)
        
        # Load training data
        metadata = pd.read_csv(TRAIN_METADATA_PATH)
        if metadata['label_positive'].dtype == 'object':
            metadata['label_positive'] = metadata['label_positive'].map({'True': True, 'False': False})
        
        if self.verbose:
            print_flush(f"Loading {len(metadata)} training repertoires...")
        
        repertoires = {}
        labels = {}
        for _, row in tqdm(metadata.iterrows(), total=len(metadata), disable=not self.verbose):
            sample_id = row['repertoire_id']
            filename = row['filename']
            label = bool(row['label_positive'])
            
            tsv_path = TRAIN_DATA_DIR / filename
            try:
                tcr_data = pd.read_csv(tsv_path, sep='\t')
                repertoires[sample_id] = tcr_data
                labels[sample_id] = label
            except Exception as e:
                if self.verbose:
                    print_flush(f"Warning: Failed to load {filename}: {e}")
        
        # Extract all unique sequences with their V/J calls and counts
        sequence_data = defaultdict(lambda: {'v_calls': set(), 'j_calls': set(), 
                                             'pos_count': 0, 'neg_count': 0})
        
        if self.verbose:
            print_flush("\nExtracting sequences from training data...")
        
        for sample_id, tcr_df in tqdm(repertoires.items(), disable=not self.verbose):
            label = labels[sample_id]
            
            for _, row in tcr_df.iterrows():
                junction_aa = str(row.get('junction_aa', ''))
                v_call = str(row.get('v_call', ''))
                j_call = str(row.get('j_call', ''))
                
                if pd.isna(junction_aa) or junction_aa == 'nan' or len(junction_aa) < 3:
                    continue
                
                if label:
                    sequence_data[junction_aa]['pos_count'] += 1
                else:
                    sequence_data[junction_aa]['neg_count'] += 1
                
                if not pd.isna(v_call) and v_call != 'nan':
                    sequence_data[junction_aa]['v_calls'].add(v_call)
                if not pd.isna(j_call) and j_call != 'nan':
                    sequence_data[junction_aa]['j_calls'].add(j_call)
        
        if self.verbose:
            print_flush(f"Found {len(sequence_data):,} unique sequences")
        
        # ====================================================================
        # OPTIMIZED: Pre-compute feature importance lookup structures
        # ====================================================================
        if self.verbose:
            print_flush("\nPreparing optimized feature matching...")
        
        # Get model feature importances from XGBoost
        feature_importances = self.classifier.feature_importances_
        
        # Create importance dictionary for all selected features
        feature_importance_dict = {name: imp for name, imp in zip(self.feature_names, feature_importances)}
        
        # Get top 1000 features by importance
        top_features = sorted(
            [(name, feature_importance_dict[name]) for name in self.feature_names],
            key=lambda x: x[1],
            reverse=True
        )[:1000]
        
        # Create set for O(1) lookup
        top_feature_set = set(name for name, _ in top_features)
        top_feature_importance = {name: imp for name, imp in top_features}
        
        if self.verbose:
            print_flush(f"  Using top {len(top_feature_set):,} features for matching")
            print_flush(f"  Top 5 features: {[f[0] for f in top_features[:5]]}")
        
        # ====================================================================
        # OPTIMIZED: Calculate scores in batches with progress reporting
        # ====================================================================
        if self.verbose:
            print_flush("\nCalculating witness-based importance scores (OPTIMIZED)...")
            print_flush("Using vectorized k-mer matching + log2 fold-change enrichment")
        
        # Convert to list for batch processing
        sequence_list = list(sequence_data.items())
        total_sequences = len(sequence_list)
        
        # Process in batches with progress reporting
        BATCH_SIZE = 10000
        num_batches = (total_sequences + BATCH_SIZE - 1) // BATCH_SIZE
        
        scored_sequences = []
        
        for batch_idx in tqdm(range(num_batches), desc="Processing batches", disable=not self.verbose):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, total_sequences)
            batch = sequence_list[start_idx:end_idx]
            
            # Process each sequence in batch
            for junction_aa, data in batch:
                # Calculate witness enrichment score (log2 fold-change)
                witness_score = calculate_witness_score(
                    data['pos_count'], 
                    data['neg_count'], 
                    method='log2_fold_change'
                )
                
                # Calculate model importance using optimized k-mer matching
                seq_kmers = get_sequence_kmers(junction_aa)
                matching_features = seq_kmers & top_feature_set
                
                if matching_features:
                    total_importance = sum(top_feature_importance[f] for f in matching_features)
                    model_score = total_importance / len(matching_features)
                else:
                    model_score = 0.0
                
                # Use most common V/J call (or first if multiple)
                v_call = sorted(data['v_calls'])[0] if data['v_calls'] else ''
                j_call = sorted(data['j_calls'])[0] if data['j_calls'] else ''
                
                scored_sequences.append({
                    'junction_aa': junction_aa,
                    'v_call': v_call,
                    'j_call': j_call,
                    'witness_score': witness_score,
                    'model_score': model_score,
                    'pos_count': data['pos_count'],
                    'neg_count': data['neg_count'],
                })
        
        if self.verbose:
            print_flush(f"\nScored {len(scored_sequences):,} sequences")
        
        # Extract scores for normalization
        witness_scores = np.array([s['witness_score'] for s in scored_sequences])
        model_scores = np.array([s['model_score'] for s in scored_sequences])
        
        # Normalize and combine scores (60% witness, 40% model)
        combined_scores = combine_witness_and_model_scores(
            witness_scores, model_scores, witness_weight=0.6, normalize=True
        )
        
        # Update importance scores
        for i, score in enumerate(combined_scores):
            scored_sequences[i]['importance_score'] = score
        
        # Sort by combined importance score (descending)
        scored_sequences.sort(key=lambda x: x['importance_score'], reverse=True)
        
        # Calculate and report enrichment metrics
        if self.verbose:
            metrics = calculate_enrichment_metrics(scored_sequences, top_k=1000)
            print_flush(f"\nEnrichment metrics (top 1000 sequences):")
            print_flush(f"  Positive ratio: {metrics['pos_ratio']:.4f}")
            print_flush(f"  Average witness score: {metrics['avg_witness_score']:.4f}")
            print_flush(f"  Sequences only in positive samples: {metrics['pos_only_count']} ({metrics['pos_only_ratio']:.2%})")
        
        # Take top N unique sequences
        seen = set()
        top_sequences = []
        for seq_data in scored_sequences:
            key = (seq_data['junction_aa'], seq_data['v_call'], seq_data['j_call'])
            if key not in seen:
                seen.add(key)
                top_sequences.append(seq_data)
                if len(top_sequences) >= top_n:
                    break
        
        # Create output DataFrame
        output_df = pd.DataFrame(top_sequences)
        output_df = output_df[['junction_aa', 'v_call', 'j_call', 'importance_score']]
        
        output_path = OUTPUT_DIR / "ds2_ranked_sequences_v2.csv"
        output_df.to_csv(output_path, index=False)
        
        if self.verbose:
            print_flush(f"\nRanked sequences saved to: {output_path}")
            print_flush(f"Total unique sequences ranked: {len(top_sequences)}")
            print_flush(f"Importance score range: [{output_df['importance_score'].min():.6f}, {output_df['importance_score'].max():.6f}]")
            print_flush(f"Top 5 sequences:")
            for i, row in output_df.head(5).iterrows():
                print_flush(f"  {i+1}. {row['junction_aa']} | {row['v_call']} | {row['j_call']} | score={row['importance_score']:.6f}")
        
        return output_df
    
    def create_submission_file(self, test_predictions: pd.DataFrame, ranked_sequences: pd.DataFrame):
        """Create combined submission file in the competition format."""
        if self.verbose:
            print_flush("\n" + "="*70)
            print_flush("CREATING SUBMISSION FILE (DS2 v2)")
            print_flush("="*70)
        
        # Create test dataset entries
        test_entries = []
        for _, row in test_predictions.iterrows():
            test_entries.append({
                'ID': row['repertoire_id'],
                'dataset': 'test_dataset_2',
                'label_positive_probability': row['probability'],
                'junction_aa': -999.0,
                'v_call': -999.0,
                'j_call': -999.0
            })
        
        # Create training dataset entries
        train_entries = []
        for idx, (_, row) in enumerate(ranked_sequences.iterrows(), 1):
            train_entries.append({
                'ID': f'train_dataset_2_seq_top_{idx}',
                'dataset': 'train_dataset_2',
                'label_positive_probability': -999.0,
                'junction_aa': row['junction_aa'],
                'v_call': row['v_call'],
                'j_call': row['j_call']
            })
        
        # Combine and create DataFrame
        submission_df = pd.DataFrame(test_entries + train_entries)
        
        # Save submission file
        output_path = OUTPUT_DIR / "ds2_submission_v2.csv"
        submission_df.to_csv(output_path, index=False)
        
        if self.verbose:
            print_flush(f"\nSubmission file saved to: {output_path}")
            print_flush(f"Total entries: {len(submission_df)}")
            print_flush(f"  - Test entries: {len(test_entries)}")
            print_flush(f"  - Training sequence entries: {len(train_entries)}")
            print_flush(f"\nFirst few test entries:")
            print_flush(submission_df[submission_df['dataset'] == 'test_dataset_2'].head(3).to_string())
            print_flush(f"\nFirst few training sequence entries:")
            print_flush(submission_df[submission_df['dataset'] == 'train_dataset_2'].head(3).to_string())
        
        return submission_df


def main():
    """Main function."""
    import argparse
    parser = argparse.ArgumentParser(description="DS2 Predictions and Sequence Ranking")
    parser.add_argument('--skip-test-predictions', action='store_true',
                        help='Skip test predictions if output file exists')
    parser.add_argument('--ranking-only', action='store_true',
                        help='Only run ranking (skip test predictions)')
    args = parser.parse_args()
    
    print_flush("="*70)
    print_flush("DS2 PREDICTIONS AND SEQUENCE RANKING (VERSION 2.1 - OPTIMIZED)")
    print_flush("Model: Hybrid Gapped XGBoost (Test AUC: 0.8944)")
    print_flush("="*70)
    
    # Initialize predictor
    predictor = DS2PredictorV2(verbose=True)
    
    # Load model
    predictor.load_model()
    
    # Check if we should skip test predictions
    test_pred_path = OUTPUT_DIR / "ds2_test_predictions_v2.csv"
    skip_test = args.ranking_only or args.skip_test_predictions
    
    if skip_test and test_pred_path.exists():
        print_flush(f"\n[SKIP] Loading existing test predictions from {test_pred_path}")
        test_predictions = pd.read_csv(test_pred_path)
        print_flush(f"Loaded {len(test_predictions)} predictions")
    else:
        # Generate predictions for test repertoires
        test_predictions = predictor.predict_test_repertoires()
    
    # Rank training sequences
    ranked_sequences = predictor.rank_training_sequences(top_n=50000)
    
    # Create combined submission file
    submission_df = predictor.create_submission_file(test_predictions, ranked_sequences)
    
    print_flush("\n" + "="*70)
    print_flush("COMPLETE (DS2 v2)")
    print_flush("="*70)
    print_flush(f"Test predictions: {OUTPUT_DIR / 'ds2_test_predictions_v2.csv'}")
    print_flush(f"Ranked sequences: {OUTPUT_DIR / 'ds2_ranked_sequences_v2.csv'}")
    print_flush(f"Submission file: {OUTPUT_DIR / 'ds2_submission_v2.csv'}")


if __name__ == "__main__":
    main()
