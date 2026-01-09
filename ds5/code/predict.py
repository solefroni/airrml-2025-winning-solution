"""
Generate predictions for test repertoires and ranked sequence list for DS5.

This script:
1. Loads the trained DS5 Logistic Regression model (aggregated pattern features, k=3,4,5)
2. Extracts features from TEST repertoires (aggregated pattern features)
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
TRAIN_DATA_DIR = Path(os.environ.get('DS5_TRAIN_DATA', '../input/train'))
TRAIN_METADATA_PATH = TRAIN_DATA_DIR / "metadata.csv"
TEST_DATA_DIR = Path(os.environ.get('DS5_TEST_DATA', '../input/test'))
MODEL_DIR = CODE_DIR.parent / 'model'
OUTPUT_DIR = CODE_DIR.parent / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Helper function for flushing output
def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


class DS5Predictor:
    """Predictor for DS5 using aggregated pattern features (k=3,4,5)."""
    
    def __init__(self, k_sizes: List[int] = [3, 4, 5], verbose: bool = True):
        self.k_sizes = k_sizes
        self.verbose = verbose
        
        # Will be loaded from model
        self.model = None
        self.scaler = None
        self.disease_patterns_by_k = None
        self.feature_names = None
        self.results_metadata = None
        
    def load_model(self):
        """Load the trained Logistic Regression model, scaler, and disease patterns."""
        if self.verbose:
            print_flush(f"Loading model from {MODEL_DIR}...")
        
        # Load Logistic Regression model (best performer)
        with open(MODEL_DIR / 'best_model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        
        # Load scaler
        with open(MODEL_DIR / 'scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load disease patterns
        with open(MODEL_DIR / 'disease_patterns.pkl', 'rb') as f:
            self.disease_patterns_by_k = pickle.load(f)
        
        # Load results metadata
        with open(MODEL_DIR / 'results.json', 'r') as f:
            self.results_metadata = json.load(f)
        
        self.feature_names = self.results_metadata['feature_names']
        
        if self.verbose:
            print_flush(f"Model loaded. Features: {len(self.feature_names)}")
            print_flush(f"Pattern counts: {self.results_metadata['n_patterns_by_k']}")
            print_flush(f"Test AUC: {self.results_metadata['test_auc']:.4f}")
    
    def generate_kmers(self, sequence: str, k: int) -> List[str]:
        """Generate k-mers from sequence."""
        if len(sequence) < k:
            return []
        return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    
    def extract_kmers_from_samples(self, sample_sequences: List[List[str]], k: int) -> List[Counter]:
        """Extract k-mers from all samples."""
        sample_kmers = []
        for sequences in sample_sequences:
            kmers = []
            for seq in sequences:
                kmers.extend(self.generate_kmers(seq, k))
            sample_kmers.append(Counter(kmers))
        return sample_kmers
    
    def extract_aggregated_pattern_features(self, sample_kmers_by_k: Dict[int, List[Counter]]) -> pd.DataFrame:
        """Extract aggregated pattern features based on disease-associated patterns."""
        n_samples = len(sample_kmers_by_k[list(self.k_sizes)[0]])
        features = {}
        
        # Basic pattern features for each k
        for k in self.k_sizes:
            disease_patterns = self.disease_patterns_by_k[k]
            sample_kmers = sample_kmers_by_k[k]
            
            pattern_counts = []
            pattern_abundances = []
            pattern_diversities = []
            pattern_max_abundances = []
            pattern_mean_abundances = []
            
            for i in range(n_samples):
                kmer_counter = sample_kmers[i]
                
                # Count disease-associated patterns
                pattern_count = sum(1 for pattern in disease_patterns if pattern in kmer_counter)
                pattern_counts.append(pattern_count)
                
                # Total abundance
                pattern_abundance = sum(kmer_counter.get(pattern, 0) for pattern in disease_patterns)
                pattern_abundances.append(pattern_abundance)
                
                # Pattern diversity (Shannon index)
                pattern_abundances_list = [kmer_counter.get(pattern, 0) for pattern in disease_patterns]
                pattern_abundances_list = [x for x in pattern_abundances_list if x > 0]
                if len(pattern_abundances_list) > 0:
                    total = sum(pattern_abundances_list)
                    probs = [x / total for x in pattern_abundances_list]
                    diversity = -sum(p * np.log(p + 1e-10) for p in probs)
                    pattern_diversities.append(diversity)
                else:
                    pattern_diversities.append(0.0)
                
                # Max and mean abundances
                if pattern_abundances_list:
                    pattern_max_abundances.append(max(pattern_abundances_list))
                    pattern_mean_abundances.append(np.mean(pattern_abundances_list))
                else:
                    pattern_max_abundances.append(0.0)
                    pattern_mean_abundances.append(0.0)
            
            features[f'pattern{k}_count'] = pattern_counts
            features[f'pattern{k}_abundance'] = pattern_abundances
            features[f'pattern{k}_diversity'] = pattern_diversities
            features[f'pattern{k}_max_abundance'] = pattern_max_abundances
            features[f'pattern{k}_mean_abundance'] = pattern_mean_abundances
            
            # Ratio features
            total_kmers = [sum(kmers.values()) for kmers in sample_kmers]
            features[f'pattern{k}_ratio'] = [
                pattern_counts[i] / (total_kmers[i] + 1e-10) for i in range(n_samples)
            ]
            features[f'pattern{k}_abundance_ratio'] = [
                pattern_abundances[i] / (total_kmers[i] + 1e-10) for i in range(n_samples)
            ]
        
        # Combined features across k-mer sizes
        features['pattern_all_count'] = [
            sum(features[f'pattern{k}_count'][i] for k in self.k_sizes) for i in range(n_samples)
        ]
        features['pattern_all_abundance'] = [
            sum(features[f'pattern{k}_abundance'][i] for k in self.k_sizes) for i in range(n_samples)
        ]
        features['pattern_all_diversity'] = [
            sum(features[f'pattern{k}_diversity'][i] for k in self.k_sizes) for i in range(n_samples)
        ]
        
        # Interaction features (ratios between different k-mer sizes)
        if len(self.k_sizes) >= 2:
            for i, k1 in enumerate(self.k_sizes[:-1]):
                for k2 in self.k_sizes[i+1:]:
                    features[f'pattern{k1}_to_{k2}_ratio'] = [
                        (features[f'pattern{k1}_count'][j] + 1e-10) / 
                        (features[f'pattern{k2}_count'][j] + 1e-10) 
                        for j in range(n_samples)
                    ]
        
        # Normalized features by repertoire size
        total_kmers_per_sample = [sum(kmers.values()) for kmers in sample_kmers_by_k[self.k_sizes[0]]]
        for k in self.k_sizes:
            features[f'pattern{k}_count_per_1k'] = [
                (features[f'pattern{k}_count'][i] * 1000) / (total_kmers_per_sample[i] + 1) 
                for i in range(n_samples)
            ]
        
        return pd.DataFrame(features)
    
    def extract_all_features(self, repertoires: Dict[str, pd.DataFrame],
                            sample_ids: List[str]) -> np.ndarray:
        """Extract all features (aggregated patterns)."""
        if self.verbose:
            print_flush("\nExtracting features...")
        
        # Extract sequences and k-mers
        sample_sequences = []
        for sample_id in sample_ids:
            tcr_df = repertoires[sample_id]
            sequences = tcr_df['junction_aa'].dropna().tolist()
            sample_sequences.append(sequences)
        
        # Extract k-mers for each k
        sample_kmers_by_k = {}
        for k in self.k_sizes:
            sample_kmers_by_k[k] = self.extract_kmers_from_samples(sample_sequences, k)
        
        # Extract aggregated pattern features
        pattern_features = self.extract_aggregated_pattern_features(sample_kmers_by_k)
        
        # Reorder columns to match feature_names from training (ensures exact order)
        pattern_features = pattern_features[self.feature_names]
        
        X = pattern_features.values
        
        return X
    
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
        X_test = self.extract_all_features(repertoires, sample_ids)
        
        # Scale features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Generate predictions
        probabilities = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Create output DataFrame
        results_df = pd.DataFrame({
            'repertoire_id': sample_ids,
            'probability': probabilities
        })
        
        output_path = OUTPUT_DIR / "ds5_test_predictions.csv"
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
        
        # Get feature importance from Logistic Regression model (use absolute coefficients)
        feature_importance = np.abs(self.model.coef_[0])
        feature_to_importance = {name: imp for name, imp in zip(self.feature_names, feature_importance)}
        
        # Get pattern feature importances (for k=3,4,5)
        pattern_weights = {}
        for k in self.k_sizes:
            pattern_features = [f for f in self.feature_names if f.startswith(f'pattern{k}_')]
            pattern_weights[k] = sum(feature_to_importance.get(f, 0.0) for f in pattern_features)
        
        # Extract all unique sequences with their V/J calls
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
            print_flush(f"Found {len(sequence_data)} unique sequences")
        
        # Calculate importance scores for each sequence
        if self.verbose:
            print_flush("\nCalculating importance scores...")
        
        scored_sequences = []
        
        for junction_aa, data in tqdm(sequence_data.items(), disable=not self.verbose):
            # Calculate pattern score based on k-mers in sequence
            pattern_score = 0.0
            for k in self.k_sizes:
                kmers = self.generate_kmers(junction_aa, k)
                # Check if any k-mer matches disease patterns
                disease_patterns = self.disease_patterns_by_k[k]
                matching_patterns = sum(1 for kmer in kmers if kmer in disease_patterns)
                if matching_patterns > 0:
                    # Weight by pattern feature importance for this k
                    pattern_score += pattern_weights[k] * (matching_patterns / max(len(kmers), 1))
            
            # Normalize pattern score
            total_kmers = sum(len(self.generate_kmers(junction_aa, k)) for k in self.k_sizes)
            if total_kmers > 0:
                pattern_score = pattern_score / total_kmers
            
            # Calculate positive/negative ratio
            total_count = data['pos_count'] + data['neg_count']
            pos_ratio = data['pos_count'] / max(total_count, 1)
            
            # Combined importance score
            importance_score = (
                0.7 * pattern_score +  # Pattern contribution (most important)
                0.3 * pos_ratio  # Positive sample association
            )
            
            # Use most common V/J call (or first if multiple)
            v_call = sorted(data['v_calls'])[0] if data['v_calls'] else ''
            j_call = sorted(data['j_calls'])[0] if data['j_calls'] else ''
            
            scored_sequences.append({
                'junction_aa': junction_aa,
                'v_call': v_call,
                'j_call': j_call,
                'importance_score': importance_score,
                'pattern_score': pattern_score,
                'pos_ratio': pos_ratio,
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
        
        output_path = OUTPUT_DIR / "ds5_ranked_sequences.csv"
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
                'dataset': 'test_dataset_5',
                'label_positive_probability': row['probability'],
                'junction_aa': -999.0,
                'v_call': -999.0,
                'j_call': -999.0
            })
        
        # Create training dataset entries
        train_entries = []
        for idx, (_, row) in enumerate(ranked_sequences.iterrows(), 1):
            train_entries.append({
                'ID': f'train_dataset_5_seq_top_{idx}',
                'dataset': 'train_dataset_5',
                'label_positive_probability': -999.0,
                'junction_aa': row['junction_aa'],
                'v_call': row['v_call'],
                'j_call': row['j_call']
            })
        
        # Combine and create DataFrame
        submission_df = pd.DataFrame(test_entries + train_entries)
        
        # Save submission file
        output_path = OUTPUT_DIR / "ds5_submission.csv"
        submission_df.to_csv(output_path, index=False)
        
        if self.verbose:
            print_flush(f"\nSubmission file saved to: {output_path}")
            print_flush(f"Total entries: {len(submission_df)}")
            print_flush(f"  - Test entries: {len(test_entries)}")
            print_flush(f"  - Training sequence entries: {len(train_entries)}")
            print_flush(f"\nFirst few test entries:")
            print_flush(submission_df[submission_df['dataset'] == 'test_dataset_5'].head(3).to_string())
            print_flush(f"\nFirst few training sequence entries:")
            print_flush(submission_df[submission_df['dataset'] == 'train_dataset_5'].head(3).to_string())
        
        return submission_df


def main():
    """Main function."""
    print_flush("="*70)
    print_flush("DS5 PREDICTIONS AND SEQUENCE RANKING")
    print_flush("="*70)
    
    # Initialize predictor
    predictor = DS5Predictor(k_sizes=[3, 4, 5], verbose=True)
    
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
    print_flush(f"Test predictions: {OUTPUT_DIR / 'ds5_test_predictions.csv'}")
    print_flush(f"Ranked sequences: {OUTPUT_DIR / 'ds5_ranked_sequences.csv'}")
    print_flush(f"Submission file: {OUTPUT_DIR / 'ds5_submission.csv'}")


if __name__ == "__main__":
    main()







