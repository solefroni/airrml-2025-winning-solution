"""
Generate predictions for test repertoires and ranked sequence list for DS6.

This script:
1. Loads the trained DS6 Logistic Regression model (HER2/neu CDR3-based)
2. Extracts features from TEST repertoires (counts of HER2/neu CDR3 sequences)
3. Extracts sequences from TRAIN repertoires and ranks them by importance
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
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - Update these paths for your environment
# =============================================================================
CODE_DIR = Path(__file__).parent.resolve()
TRAIN_DATA_DIR = Path(os.environ.get('DS6_TRAIN_DATA', '../input/train'))
TRAIN_METADATA_PATH = TRAIN_DATA_DIR / "metadata.csv"
TEST_DATA_DIR = Path(os.environ.get('DS6_TEST_DATA', '../input/test'))
MODEL_DIR = CODE_DIR.parent / 'model'
OUTPUT_DIR = CODE_DIR.parent / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Helper function for flushing output
def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


class DS6Predictor:
    """Predictor for DS6 using HER2/neu CDR3 sequence counts."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        
        # Will be loaded from model
        self.model = None
        self.scaler = None
        self.her2neu_cdr3s = None
        self.results_metadata = None
        
    def load_model(self):
        """Load the trained Logistic Regression model, scaler, and HER2/neu CDR3 list."""
        if self.verbose:
            print_flush(f"Loading model from {MODEL_DIR}...")
        
        # Load Logistic Regression model (best performer with 1.0 AUC)
        with open(MODEL_DIR / 'logistic_regression_model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        
        # Load scaler
        with open(MODEL_DIR / 'scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load HER2/neu CDR3 sequences (feature vocabulary)
        with open(MODEL_DIR / 'her2neu_cdr3s.pkl', 'rb') as f:
            self.her2neu_cdr3s = pickle.load(f)
        
        # Load results metadata
        with open(MODEL_DIR / 'classification_results.json', 'r') as f:
            self.results_metadata = json.load(f)
        
        if self.verbose:
            print_flush(f"Model loaded. HER2/neu CDR3s: {len(self.her2neu_cdr3s)}")
            print_flush(f"Test AUC: {self.results_metadata['logistic_regression']['test_auc']:.4f}")
    
    def extract_features(self, repertoires: Dict[str, pd.DataFrame], 
                        sample_ids: List[str]) -> np.ndarray:
        """Extract features by counting HER2/neu CDR3 sequences in each repertoire."""
        if self.verbose:
            print_flush("\nExtracting features (counting HER2/neu CDR3s)...")
        
        X = []
        
        for sample_id in tqdm(sample_ids, disable=not self.verbose):
            tcr_df = repertoires[sample_id]
            
            # Count occurrences of each HER2/neu CDR3 in this repertoire
            rep_cdr3_counts = Counter(tcr_df['junction_aa'].dropna())
            feature_vector = [rep_cdr3_counts.get(cdr3, 0) for cdr3 in self.her2neu_cdr3s]
            
            X.append(feature_vector)
        
        return np.array(X)
    
    def predict_test_repertoires(self):
        """Load test repertoires, extract features, and generate predictions."""
        if self.verbose:
            print_flush("\n" + "="*70)
            print_flush("PREDICTING TEST REPERTOIRES")
            print_flush("="*70)
        
        # Load test files
        test_files = sorted(list(TEST_DATA_DIR.glob("*.tsv")))
        if self.verbose:
            print_flush(f"Found {len(test_files)} test repertoire files")
        
        repertoires = {}
        for tsv_path in tqdm(test_files, disable=not self.verbose):
            sample_id = tsv_path.stem
            try:
                tcr_data = pd.read_csv(tsv_path, sep='\t')
                repertoires[sample_id] = tcr_data
            except Exception as e:
                if self.verbose:
                    print_flush(f"Warning: Failed to load {tsv_path.name}: {e}")
        
        sample_ids = sorted(list(repertoires.keys()))
        
        # Extract features
        X_test = self.extract_features(repertoires, sample_ids)
        
        # Scale features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Generate predictions
        probabilities = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Create output DataFrame
        results_df = pd.DataFrame({
            'repertoire_id': sample_ids,
            'probability': probabilities
        })
        
        output_path = OUTPUT_DIR / "ds6_test_predictions.csv"
        results_df.to_csv(output_path, index=False)
        
        if self.verbose:
            print_flush(f"\nPredictions saved to: {output_path}")
            print_flush(f"Predicted {len(sample_ids)} repertoires")
            print_flush(f"Probability range: [{probabilities.min():.4f}, {probabilities.max():.4f}]")
            print_flush(f"Mean probability: {probabilities.mean():.4f}")
        
        return results_df
    
    def rank_training_sequences(self, top_n: int = 50000):
        """Extract sequences from training data and rank them by importance."""
        if self.verbose:
            print_flush("\n" + "="*70)
            print_flush("RANKING TRAINING SEQUENCES")
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
        
        # Get feature importance from model (coefficients for Logistic Regression)
        # For Logistic Regression, we use the absolute value of coefficients as importance
        feature_importance = np.abs(self.model.coef_[0])
        
        # Create mapping from HER2/neu CDR3 to importance
        cdr3_to_importance = {cdr3: imp for cdr3, imp in zip(self.her2neu_cdr3s, feature_importance)}
        
        # Extract all unique sequences with their V/J calls
        sequence_data = defaultdict(lambda: {'v_calls': set(), 'j_calls': set(), 
                                             'pos_count': 0, 'neg_count': 0,
                                             'her2neu_count': 0})
        
        if self.verbose:
            print_flush("\nExtracting sequences from training data...")
        
        # Create set of HER2/neu CDR3s for fast lookup
        her2neu_set = set(self.her2neu_cdr3s)
        
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
                
                # Check if this sequence is a HER2/neu CDR3
                if junction_aa in her2neu_set:
                    sequence_data[junction_aa]['her2neu_count'] += 1
                
                if not pd.isna(v_call) and v_call != 'nan':
                    sequence_data[junction_aa]['v_calls'].add(v_call)
                if not pd.isna(j_call) and j_call != 'nan':
                    sequence_data[junction_aa]['j_calls'].add(j_call)
        
        if self.verbose:
            print_flush(f"Found {len(sequence_data)} unique sequences")
        
        # Calculate importance scores for each sequence
        if self.verbose:
            print_flush("\nCalculating importance scores...")
        
        scored_sequences = []
        
        for junction_aa, data in tqdm(sequence_data.items(), disable=not self.verbose):
            # HER2/neu CDR3 importance score
            if junction_aa in cdr3_to_importance:
                her2neu_score = cdr3_to_importance[junction_aa]
            else:
                her2neu_score = 0.0
            
            # Positive/negative ratio
            total_count = data['pos_count'] + data['neg_count']
            pos_ratio = data['pos_count'] / max(total_count, 1)
            
            # HER2/neu association (whether it's a HER2/neu CDR3)
            is_her2neu = 1.0 if junction_aa in her2neu_set else 0.0
            
            # Combined importance score
            # Weight: HER2/neu CDR3 importance (most important), positive ratio, and HER2/neu association
            importance_score = (
                0.6 * her2neu_score +  # HER2/neu CDR3 importance (most important)
                0.3 * pos_ratio +  # Positive sample association
                0.1 * is_her2neu  # HER2/neu association bonus
            )
            
            # Use most common V/J call (or first if multiple)
            v_call = sorted(data['v_calls'])[0] if data['v_calls'] else ''
            j_call = sorted(data['j_calls'])[0] if data['j_calls'] else ''
            
            scored_sequences.append({
                'junction_aa': junction_aa,
                'v_call': v_call,
                'j_call': j_call,
                'importance_score': importance_score,
                'her2neu_score': her2neu_score,
                'pos_ratio': pos_ratio,
                'is_her2neu': is_her2neu,
                'pos_count': data['pos_count'],
                'neg_count': data['neg_count']
            })
        
        # Sort by importance score (descending)
        scored_sequences.sort(key=lambda x: x['importance_score'], reverse=True)
        
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
        
        output_path = OUTPUT_DIR / "ds6_ranked_sequences.csv"
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
            print_flush("CREATING SUBMISSION FILE")
            print_flush("="*70)
        
        # Create test dataset entries
        test_entries = []
        for _, row in test_predictions.iterrows():
            test_entries.append({
                'ID': row['repertoire_id'],
                'dataset': 'test_dataset_6',
                'label_positive_probability': row['probability'],
                'junction_aa': -999.0,
                'v_call': -999.0,
                'j_call': -999.0
            })
        
        # Create training dataset entries
        train_entries = []
        for idx, (_, row) in enumerate(ranked_sequences.iterrows(), 1):
            train_entries.append({
                'ID': f'train_dataset_6_seq_top_{idx}',
                'dataset': 'train_dataset_6',
                'label_positive_probability': -999.0,
                'junction_aa': row['junction_aa'],
                'v_call': row['v_call'],
                'j_call': row['j_call']
            })
        
        # Combine and create DataFrame
        submission_df = pd.DataFrame(test_entries + train_entries)
        
        # Save submission file
        output_path = OUTPUT_DIR / "ds6_submission.csv"
        submission_df.to_csv(output_path, index=False)
        
        if self.verbose:
            print_flush(f"\nSubmission file saved to: {output_path}")
            print_flush(f"Total entries: {len(submission_df)}")
            print_flush(f"  - Test entries: {len(test_entries)}")
            print_flush(f"  - Training sequence entries: {len(train_entries)}")
            print_flush(f"\nFirst few test entries:")
            print_flush(submission_df[submission_df['dataset'] == 'test_dataset_6'].head(3).to_string())
            print_flush(f"\nFirst few training sequence entries:")
            print_flush(submission_df[submission_df['dataset'] == 'train_dataset_6'].head(3).to_string())
        
        return submission_df


def main():
    """Main function."""
    print_flush("="*70)
    print_flush("DS6 PREDICTIONS AND SEQUENCE RANKING")
    print_flush("="*70)
    
    # Initialize predictor
    predictor = DS6Predictor(verbose=True)
    
    # Load model
    predictor.load_model()
    
    # Generate predictions for test repertoires
    test_predictions = predictor.predict_test_repertoires()
    
    # Rank training sequences
    ranked_sequences = predictor.rank_training_sequences(top_n=50000)
    
    # Create combined submission file
    submission_df = predictor.create_submission_file(test_predictions, ranked_sequences)
    
    print_flush("\n" + "="*70)
    print_flush("COMPLETE")
    print_flush("="*70)
    print_flush(f"Test predictions: {OUTPUT_DIR / 'ds6_test_predictions.csv'}")
    print_flush(f"Ranked sequences: {OUTPUT_DIR / 'ds6_ranked_sequences.csv'}")
    print_flush(f"Submission file: {OUTPUT_DIR / 'ds6_submission.csv'}")


if __name__ == "__main__":
    main()







