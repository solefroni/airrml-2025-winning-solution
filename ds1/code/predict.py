"""
Generate predictions for test repertoires and ranked sequence list for DS1.

This script:
1. Loads the trained DS1 model (Log-transformed Proinsulin_Diversity + GAD65_Diversity)
2. Extracts features from TEST repertoires and generates probabilities
3. Extracts sequences from TRAIN repertoires and ranks them using witness-based ranking

Model:
- Approach: Log-transformed diversity features + Polynomial expansion + Isotonic calibration
- Features: Log-transformed Proinsulin_Diversity and GAD65_Diversity (2 base features)
- Polynomial degree: 2 (quadratic + interactions)
- Calibration: Isotonic

Ranking Method:
- Uses witness enrichment scoring (log2 fold-change) to identify sequences
  that appear more frequently in positive samples
- Combines witness score with model importance (Proinsulin and GAD65 TCRs)
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
TRAIN_DATA_DIR = Path(os.environ.get('DS1_TRAIN_DATA', '../input/train'))
TRAIN_METADATA_PATH = TRAIN_DATA_DIR / "metadata.csv"
TEST_DATA_DIR = Path(os.environ.get('DS1_TEST_DATA', '../input/test'))
MODEL_DIR = CODE_DIR.parent / 'model'
OUTPUT_DIR = CODE_DIR.parent / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Reference sequence files (included in model folder)
PROINSULIN_CDR3S_FILE = MODEL_DIR / "proinsulin_cdr3s.pkl"
GAD65_CDR3S_FILE = MODEL_DIR / "gad65_cdr3s.pkl"

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


class DS1PredictorV3:
    """Predictor for DS1 using the final optimization model (v3)."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        
        # Will be loaded from model
        self.model = None
        self.scaler = None
        self.poly = None
        self.proinsulin_cdr3s_set = None
        self.gad65_cdr3s_set = None
        self.model_metadata = None
        
    def load_model(self):
        """Load the trained model, scaler, polynomial features, and reference sequences."""
        if self.verbose:
            print_flush(f"Loading DS1 v3 model from {MODEL_DIR}...")
        
        # Load Proinsulin CDR3 sequences
        with open(PROINSULIN_CDR3S_FILE, 'rb') as f:
            proinsulin_cdr3s = pickle.load(f)
        self.proinsulin_cdr3s_set = set(proinsulin_cdr3s)
        
        # Load GAD65 CDR3 sequences
        with open(GAD65_CDR3S_FILE, 'rb') as f:
            gad65_cdr3s = pickle.load(f)
        self.gad65_cdr3s_set = set(gad65_cdr3s)
        
        # Load model
        with open(MODEL_DIR / 'model_best.pkl', 'rb') as f:
            self.model = pickle.load(f)
        
        # Load scaler
        with open(MODEL_DIR / 'scaler_best.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load polynomial features
        with open(MODEL_DIR / 'polynomial_features_best.pkl', 'rb') as f:
            self.poly = pickle.load(f)
        
        # Load model metadata
        metadata_path = MODEL_DIR / 'optimization_results.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.model_metadata = json.load(f)
        
        if self.verbose:
            print_flush(f"Model loaded successfully")
            print_flush(f"  Proinsulin CDR3 sequences: {len(self.proinsulin_cdr3s_set):,}")
            print_flush(f"  GAD65 CDR3 sequences: {len(self.gad65_cdr3s_set):,}")
            if self.model_metadata:
                best_config = self.model_metadata.get('best_configuration', {})
                print_flush(f"  Model: {best_config.get('name', 'N/A')}")
                print_flush(f"  Test AUC: {best_config.get('test_auc', 'N/A'):.4f}")
                print_flush(f"  Base features: {self.model_metadata.get('metadata', {}).get('base_features', [])}")
                print_flush(f"  Polynomial degree: {self.model_metadata.get('metadata', {}).get('polynomial_degree', 2)}")
    
    def extract_features(self, repertoire_path: Path) -> np.ndarray:
        """Extract Proinsulin_Diversity and GAD65_Diversity features from a repertoire."""
        try:
            rep_df = pd.read_csv(repertoire_path, sep='\t', usecols=['junction_aa'], low_memory=False)
            rep_df = rep_df.dropna(subset=['junction_aa'])
            
            rep_sequences = [str(s).strip() for s in rep_df['junction_aa'].tolist()]
            rep_unique_sequences = set(rep_sequences)
            
            # Calculate diversity features (unique matches)
            proinsulin_diversity = len(rep_unique_sequences & self.proinsulin_cdr3s_set)
            gad65_diversity = len(rep_unique_sequences & self.gad65_cdr3s_set)
            
            return np.array([proinsulin_diversity, gad65_diversity])
        except Exception as e:
            if self.verbose:
                print_flush(f"Warning: Error extracting features from {repertoire_path}: {e}")
            return np.array([0, 0])
    
    def predict_test_repertoires(self):
        """Load test repertoires, extract features, and generate predictions."""
        if self.verbose:
            print_flush("\n" + "="*70)
            print_flush("PREDICTING TEST REPERTOIRES (DS1 v3)")
            print_flush("="*70)
        
        # Load test files
        test_files = sorted(list(TEST_DATA_DIR.glob("*.tsv")))
        if self.verbose:
            print_flush(f"Found {len(test_files)} test repertoire files")
        
        # Extract features for each test repertoire
        features = []
        sample_ids = []
        
        if self.verbose:
            print_flush("Extracting features (Proinsulin_Diversity, GAD65_Diversity)...")
        
        for tsv_path in tqdm(test_files, disable=not self.verbose):
            sample_id = tsv_path.stem
            feature_vector = self.extract_features(tsv_path)
            features.append(feature_vector)
            sample_ids.append(sample_id)
        
        # Convert to numpy array
        X_test = np.array(features)
        
        if self.verbose:
            print_flush(f"Feature matrix shape: {X_test.shape}")
            print_flush(f"Proinsulin_Diversity range: [{X_test[:, 0].min()}, {X_test[:, 0].max()}]")
            print_flush(f"GAD65_Diversity range: [{X_test[:, 1].min()}, {X_test[:, 1].max()}]")
        
        # Apply log transform (log1p = log(1 + x))
        X_test_log = np.log1p(X_test)
        
        # Apply polynomial features
        X_test_poly = self.poly.transform(X_test_log)
        
        # Scale features
        X_test_scaled = self.scaler.transform(X_test_poly)
        
        # Generate predictions
        if self.verbose:
            print_flush("Generating predictions...")
        probabilities = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Create output DataFrame
        results_df = pd.DataFrame({
            'repertoire_id': sample_ids,
            'probability': probabilities,
            'proinsulin_diversity': X_test[:, 0],
            'gad65_diversity': X_test[:, 1]
        })
        
        output_path = OUTPUT_DIR / "ds1_test_predictions_v3.csv"
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
            print_flush("RANKING TRAINING SEQUENCES (DS1 v3)")
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
                                             'pos_count': 0, 'neg_count': 0,
                                             'is_proinsulin': False, 'is_gad65': False})
        
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
                
                # Check if this is a Proinsulin or GAD65 TCR
                is_proinsulin = junction_aa in self.proinsulin_cdr3s_set
                is_gad65 = junction_aa in self.gad65_cdr3s_set
                sequence_data[junction_aa]['is_proinsulin'] = is_proinsulin
                sequence_data[junction_aa]['is_gad65'] = is_gad65
                
                if label:
                    sequence_data[junction_aa]['pos_count'] += 1
                else:
                    sequence_data[junction_aa]['neg_count'] += 1
                
                if not pd.isna(v_call) and v_call != 'nan':
                    sequence_data[junction_aa]['v_calls'].add(v_call)
                if not pd.isna(j_call) and j_call != 'nan':
                    sequence_data[junction_aa]['j_calls'].add(j_call)
        
        if self.verbose:
            proinsulin_count = sum(1 for data in sequence_data.values() if data['is_proinsulin'])
            gad65_count = sum(1 for data in sequence_data.values() if data['is_gad65'])
            both_count = sum(1 for data in sequence_data.values() if data['is_proinsulin'] and data['is_gad65'])
            print_flush(f"Found {len(sequence_data)} unique sequences")
            print_flush(f"  - Proinsulin TCRs: {proinsulin_count}")
            print_flush(f"  - GAD65 TCRs: {gad65_count}")
            print_flush(f"  - Both: {both_count}")
            print_flush(f"  - Other sequences: {len(sequence_data) - proinsulin_count - gad65_count + both_count}")
        
        # Calculate importance scores using WITNESS-BASED ranking
        if self.verbose:
            print_flush("\nCalculating witness-based importance scores...")
            print_flush("Using log2 fold-change enrichment + model importance (Proinsulin + GAD65 TCRs)")
        
        # Model importance: Proinsulin and GAD65 TCRs get high weight (these are the model features)
        proinsulin_weight = 1.0  # High weight for proinsulin TCRs
        gad65_weight = 0.8  # Slightly lower weight for GAD65 TCRs
        other_weight = 0.1  # Lower weight for other sequences
        
        # Calculate witness scores and combine with model scores
        scored_sequences = []
        witness_scores_list = []
        model_scores_list = []
        
        for junction_aa, data in tqdm(sequence_data.items(), disable=not self.verbose):
            # Calculate witness enrichment score (log2 fold-change)
            witness_score = calculate_witness_score(
                data['pos_count'], 
                data['neg_count'], 
                method='log2_fold_change'
            )
            
            # Model importance score: Proinsulin and GAD65 TCRs get high weight
            if data['is_proinsulin'] and data['is_gad65']:
                model_score = max(proinsulin_weight, gad65_weight)  # Both features
            elif data['is_proinsulin']:
                model_score = proinsulin_weight
            elif data['is_gad65']:
                model_score = gad65_weight
            else:
                model_score = other_weight
            
            witness_scores_list.append(witness_score)
            model_scores_list.append(model_score)
            
            # Use most common V/J call (or first if multiple)
            v_call = sorted(data['v_calls'])[0] if data['v_calls'] else ''
            j_call = sorted(data['j_calls'])[0] if data['j_calls'] else ''
            
            scored_sequences.append({
                'junction_aa': junction_aa,
                'v_call': v_call,
                'j_call': j_call,
                'witness_score': witness_score,
                'model_score': model_score,
                'is_proinsulin': data['is_proinsulin'],
                'is_gad65': data['is_gad65'],
                'pos_count': data['pos_count'],
                'neg_count': data['neg_count'],
                'importance_score': None  # Will be set after normalization
            })
        
        # Normalize and combine scores (60% witness, 40% model)
        witness_scores = np.array(witness_scores_list)
        model_scores = np.array(model_scores_list)
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
            
            # Count antigen-specific TCRs in top sequences
            top_1000_proinsulin = sum(1 for seq in scored_sequences[:1000] if seq['is_proinsulin'])
            top_1000_gad65 = sum(1 for seq in scored_sequences[:1000] if seq['is_gad65'])
            print_flush(f"  Proinsulin TCRs in top 1000: {top_1000_proinsulin} ({top_1000_proinsulin/1000:.2%})")
            print_flush(f"  GAD65 TCRs in top 1000: {top_1000_gad65} ({top_1000_gad65/1000:.2%})")
        
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
        
        output_path = OUTPUT_DIR / "ds1_ranked_sequences_v3.csv"
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
            print_flush("CREATING SUBMISSION FILE (DS1 v3)")
            print_flush("="*70)
        
        # Create test dataset entries
        test_entries = []
        for _, row in test_predictions.iterrows():
            test_entries.append({
                'ID': row['repertoire_id'],
                'dataset': 'test_dataset_1',
                'label_positive_probability': row['probability'],
                'junction_aa': -999.0,
                'v_call': -999.0,
                'j_call': -999.0
            })
        
        # Create training dataset entries
        train_entries = []
        for idx, (_, row) in enumerate(ranked_sequences.iterrows(), 1):
            train_entries.append({
                'ID': f'train_dataset_1_seq_top_{idx}',
                'dataset': 'train_dataset_1',
                'label_positive_probability': -999.0,
                'junction_aa': row['junction_aa'],
                'v_call': row['v_call'],
                'j_call': row['j_call']
            })
        
        # Combine and create DataFrame
        submission_df = pd.DataFrame(test_entries + train_entries)
        
        # Save submission file
        output_path = OUTPUT_DIR / "ds1_submission_v3.csv"
        submission_df.to_csv(output_path, index=False)
        
        if self.verbose:
            print_flush(f"\nSubmission file saved to: {output_path}")
            print_flush(f"Total entries: {len(submission_df)}")
            print_flush(f"  - Test entries: {len(test_entries)}")
            print_flush(f"  - Training sequence entries: {len(train_entries)}")
            print_flush(f"\nFirst few test entries:")
            print_flush(submission_df[submission_df['dataset'] == 'test_dataset_1'].head(3).to_string())
            print_flush(f"\nFirst few training sequence entries:")
            print_flush(submission_df[submission_df['dataset'] == 'train_dataset_1'].head(3).to_string())
        
        return submission_df


def main():
    """Main function."""
    print_flush("="*70)
    print_flush("DS1 PREDICTIONS AND SEQUENCE RANKING (VERSION 3)")
    print_flush("Model: Final Optimization (Log-transform + Polynomial + Isotonic)")
    print_flush("Test AUC: 0.9581")
    print_flush("="*70)
    
    # Initialize predictor
    predictor = DS1PredictorV3(verbose=True)
    
    # Load model
    predictor.load_model()
    
    # Generate predictions for test repertoires
    test_predictions = predictor.predict_test_repertoires()
    
    # Rank training sequences
    ranked_sequences = predictor.rank_training_sequences(top_n=50000)
    
    # Create combined submission file
    submission_df = predictor.create_submission_file(test_predictions, ranked_sequences)
    
    print_flush("\n" + "="*70)
    print_flush("COMPLETE (DS1 v3)")
    print_flush("="*70)
    print_flush(f"Test predictions: {OUTPUT_DIR / 'ds1_test_predictions_v3.csv'}")
    print_flush(f"Ranked sequences: {OUTPUT_DIR / 'ds1_ranked_sequences_v3.csv'}")
    print_flush(f"Submission file: {OUTPUT_DIR / 'ds1_submission_v3.csv'}")


if __name__ == "__main__":
    main()





