"""
Forward Feature Selection with XGBoost
=======================================
Iteratively adds the most important feature using XGBoost feature importance.
Starts with 0 features and adds 1 feature at a time until reaching 200 features.
Records performance at each step and creates visualizations.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import Counter, defaultdict
from itertools import product
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score, recall_score,
    balanced_accuracy_score
)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import pickle
import json
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Helper function for flushing output
def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

# =============================================================================
# CONFIGURATION - Update these paths for your environment
# =============================================================================
CODE_DIR = Path(__file__).parent.resolve()
TRAIN_DATA_DIR = Path(os.environ.get('DS1_TRAIN_DATA', '../input'))
METADATA_PATH = TRAIN_DATA_DIR / "metadata.csv"
MODEL_DIR = CODE_DIR.parent / 'model'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = MODEL_DIR
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class ForwardFeatureSelection:
    """Forward feature selection using XGBoost."""
    
    def __init__(self, k_mer_size: int = 3, 
                 target_features: int = 200,
                 verbose: bool = True):
        """
        Initialize forward selection.
        
        Args:
            k_mer_size: Size of k-mers
            target_features: Target number of features to reach
            verbose: Whether to print progress
        """
        self.k_mer_size = k_mer_size
        self.target_features = target_features
        self.verbose = verbose
        
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        self.all_kmers = None
        self.v_genes = None
        self.j_genes = None
        self.scaler = RobustScaler()
        
        # Track selection progress
        self.selection_history = []
        
    def load_data(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, bool]]:
        """Load all TCR repertoire data and labels."""
        if self.verbose:
            print_flush("Loading metadata...")
        
        metadata = pd.read_csv(METADATA_PATH)
        
        if metadata['label_positive'].dtype == 'object':
            metadata['label_positive'] = metadata['label_positive'].map({'True': True, 'False': False})
        
        if self.verbose:
            print_flush(f"Total samples: {len(metadata)}")
            print_flush(f"Positive: {metadata['label_positive'].sum()}, Negative: {(~metadata['label_positive']).sum()}")
            print_flush("\nLoading repertoire data...")
        
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
        
        if self.verbose:
            print_flush(f"\nLoaded {len(repertoires)} repertoires")
        
        return repertoires, labels
    
    def generate_kmers(self, sequence: str, k: int) -> List[str]:
        """Generate k-mers from sequence."""
        if len(sequence) < k:
            return []
        return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    
    def extract_kmer_features(self, repertoires: Dict[str, pd.DataFrame], 
                             sample_ids: List[str]) -> np.ndarray:
        """Extract k-mer frequency features."""
        if self.verbose:
            print_flush(f"\nExtracting {self.k_mer_size}-mer features...")
        
        if self.all_kmers is None:
            self.all_kmers = [''.join(p) for p in product(self.amino_acids, repeat=self.k_mer_size)]
            if self.verbose:
                print_flush(f"Total {self.k_mer_size}-mers: {len(self.all_kmers)}")
        
        kmer_to_idx = {kmer: idx for idx, kmer in enumerate(self.all_kmers)}
        X = np.zeros((len(sample_ids), len(self.all_kmers)))
        
        for i, sample_id in tqdm(enumerate(sample_ids), total=len(sample_ids), disable=not self.verbose):
            tcr_df = repertoires[sample_id]
            kmer_counts = Counter()
            
            for seq in tcr_df['junction_aa'].dropna():
                kmers = self.generate_kmers(str(seq), self.k_mer_size)
                kmer_counts.update(kmers)
            
            total_kmers = sum(kmer_counts.values())
            if total_kmers > 0:
                for kmer, count in kmer_counts.items():
                    if kmer in kmer_to_idx:
                        X[i, kmer_to_idx[kmer]] = count / total_kmers
        
        return X
    
    def extract_gene_usage_features(self, repertoires: Dict[str, pd.DataFrame],
                                   sample_ids: List[str]) -> np.ndarray:
        """Extract V/J gene usage features."""
        if self.verbose:
            print_flush("\nExtracting gene usage features...")
        
        if self.v_genes is None or self.j_genes is None:
            all_v_genes = set()
            all_j_genes = set()
            
            for sample_id in sample_ids:
                tcr_df = repertoires[sample_id]
                all_v_genes.update(tcr_df['v_call'].dropna().unique())
                all_j_genes.update(tcr_df['j_call'].dropna().unique())
            
            self.v_genes = sorted(list(all_v_genes))
            self.j_genes = sorted(list(all_j_genes))
        
        v_gene_to_idx = {gene: idx for idx, gene in enumerate(self.v_genes)}
        j_gene_to_idx = {gene: idx for idx, gene in enumerate(self.j_genes)}
        
        X = np.zeros((len(sample_ids), len(self.v_genes) + len(self.j_genes)))
        
        for i, sample_id in tqdm(enumerate(sample_ids), total=len(sample_ids), disable=not self.verbose):
            tcr_df = repertoires[sample_id]
            
            v_counts = Counter(tcr_df['v_call'].dropna())
            total_v = sum(v_counts.values())
            if total_v > 0:
                for gene, count in v_counts.items():
                    if gene in v_gene_to_idx:
                        X[i, v_gene_to_idx[gene]] = count / total_v
            
            j_counts = Counter(tcr_df['j_call'].dropna())
            total_j = sum(j_counts.values())
            if total_j > 0:
                for gene, count in j_counts.items():
                    if gene in j_gene_to_idx:
                        X[i, len(self.v_genes) + j_gene_to_idx[gene]] = count / total_j
        
        return X
    
    def extract_length_features(self, repertoires: Dict[str, pd.DataFrame],
                               sample_ids: List[str]) -> np.ndarray:
        """Extract CDR3 length features."""
        if self.verbose:
            print_flush("\nExtracting length features...")
        
        length_bins = list(range(5, 26))
        X = np.zeros((len(sample_ids), 5 + len(length_bins)))
        
        for i, sample_id in tqdm(enumerate(sample_ids), total=len(sample_ids), disable=not self.verbose):
            tcr_df = repertoires[sample_id]
            lengths = tcr_df['junction_aa'].dropna().str.len().values
            
            if len(lengths) > 0:
                X[i, 0] = np.mean(lengths)
                X[i, 1] = np.std(lengths)
                X[i, 2] = np.min(lengths)
                X[i, 3] = np.max(lengths)
                X[i, 4] = np.median(lengths)
                
                hist, _ = np.histogram(lengths, bins=length_bins + [100])
                X[i, 5:] = hist / len(lengths)
        
        return X
    
    def extract_diversity_features(self, repertoires: Dict[str, pd.DataFrame],
                                  sample_ids: List[str]) -> np.ndarray:
        """Extract diversity features."""
        if self.verbose:
            print_flush("\nExtracting diversity features...")
        
        X = np.zeros((len(sample_ids), 6))
        
        for i, sample_id in tqdm(enumerate(sample_ids), total=len(sample_ids), disable=not self.verbose):
            tcr_df = repertoires[sample_id]
            sequences = tcr_df['junction_aa'].dropna()
            total_seqs = len(sequences)
            
            if total_seqs > 0:
                unique_seqs = sequences.nunique()
                X[i, 0] = unique_seqs
                X[i, 1] = unique_seqs / total_seqs
                
                seq_counts = sequences.value_counts()
                proportions = seq_counts / total_seqs
                
                simpson = 1 - np.sum(proportions ** 2)
                X[i, 2] = simpson
                
                shannon = -np.sum(proportions * np.log(proportions + 1e-10))
                X[i, 3] = shannon
                
                sorted_props = np.sort(proportions.values)
                n = len(sorted_props)
                index = np.arange(1, n + 1)
                gini = (2 * np.sum(index * sorted_props)) / (n * np.sum(sorted_props)) - (n + 1) / n
                X[i, 4] = gini
                
                X[i, 5] = proportions.iloc[0]
        
        return X
    
    def extract_all_features(self, repertoires: Dict[str, pd.DataFrame],
                            sample_ids: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Extract all features and return feature names."""
        if self.verbose:
            print_flush("\n" + "="*70)
            print_flush("FEATURE EXTRACTION")
            print_flush("="*70)
        
        X_kmer = self.extract_kmer_features(repertoires, sample_ids)
        X_gene = self.extract_gene_usage_features(repertoires, sample_ids)
        X_length = self.extract_length_features(repertoires, sample_ids)
        X_diversity = self.extract_diversity_features(repertoires, sample_ids)
        
        X = np.hstack([X_kmer, X_gene, X_length, X_diversity])
        
        # Create feature names
        feature_names = []
        feature_names.extend([f"kmer_{kmer}" for kmer in self.all_kmers])
        feature_names.extend([f"v_gene_{gene}" for gene in self.v_genes])
        feature_names.extend([f"j_gene_{gene}" for gene in self.j_genes])
        feature_names.extend([
            "length_mean", "length_std", "length_min", "length_max", "length_median"
        ])
        feature_names.extend([f"length_bin_{i}" for i in range(5, 26)])
        feature_names.extend([
            "n_unique", "diversity_ratio", "simpson", "shannon", "gini", "top_clone"
        ])
        
        if self.verbose:
            print_flush(f"\nCombined features: {X.shape}")
            print_flush(f"  K-mers: {X_kmer.shape[1]}")
            print_flush(f"  Genes: {X_gene.shape[1]}")
            print_flush(f"  Length: {X_length.shape[1]}")
            print_flush(f"  Diversity: {X_diversity.shape[1]}")
            print_flush(f"  Total: {len(feature_names)} features")
        
        return X, feature_names
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray,
                     feature_names: List[str]) -> Tuple[xgb.XGBClassifier, Dict, Dict]:
        """Train XGBoost and return model with metrics and feature importance."""
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
            importance_type='gain'
        )
        
        model.fit(X_train, y_train)
        
        # Get predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_prob_train = model.predict_proba(X_train)[:, 1]
        y_prob_test = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'train_roc_auc': roc_auc_score(y_train, y_prob_train),
            'train_f1': f1_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'test_roc_auc': roc_auc_score(y_test, y_prob_test),
            'test_f1': f1_score(y_test, y_pred_test),
            'test_balanced_accuracy': balanced_accuracy_score(y_test, y_pred_test)
        }
        
        # Get feature importance
        feature_importance = model.feature_importances_
        importance_dict = {name: imp for name, imp in zip(feature_names, feature_importance)}
        
        return model, metrics, importance_dict
    
    def forward_selection(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray,
                         feature_names: List[str]) -> Dict:
        """Perform forward feature selection."""
        if self.verbose:
            print_flush("\n" + "="*70)
            print_flush("FORWARD FEATURE SELECTION")
            print_flush("="*70)
            print_flush(f"Starting with 0 features")
            print_flush(f"Target: {self.target_features} features")
            print_flush(f"Adding 1 feature at a time")
            print_flush("="*70)
        
        # Scale all features first
        if self.verbose:
            print_flush("\nScaling all features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        if self.verbose:
            print_flush("Scaling complete!")
        
        # Track selected features
        selected_features = []
        selected_indices = []
        available_indices = np.arange(len(feature_names))
        
        iteration = 0
        
        while len(selected_features) < self.target_features:
            iteration += 1
            n_features = len(selected_features)
            
            if self.verbose:
                print_flush(f"\nIteration {iteration}: {n_features} features selected")
            
            best_score = -np.inf
            best_feature_idx = None
            best_feature_name = None
            best_model = None
            best_metrics = None
            
            # Try each available feature
            if self.verbose:
                print_flush(f"  Evaluating {len(available_indices)} candidate features...")
            
            for candidate_idx in tqdm(available_indices, disable=not self.verbose, desc="  Testing features"):
                # Create feature set with this candidate
                test_indices = selected_indices + [candidate_idx]
                test_feature_names = [feature_names[i] for i in test_indices]
                
                X_train_current = X_train_scaled[:, test_indices]
                X_test_current = X_test_scaled[:, test_indices]
                
                # Train XGBoost
                model, metrics, _ = self.train_xgboost(
                    X_train_current, y_train, X_test_current, y_test, test_feature_names
                )
                
                # Use test ROC-AUC as selection criterion
                score = metrics['test_roc_auc']
                
                if score > best_score:
                    best_score = score
                    best_feature_idx = candidate_idx
                    best_feature_name = feature_names[candidate_idx]
                    best_model = model
                    best_metrics = metrics
            
            # Add best feature
            selected_indices.append(best_feature_idx)
            selected_features.append(best_feature_name)
            available_indices = available_indices[available_indices != best_feature_idx]
            
            # Record progress
            self.selection_history.append({
                'iteration': iteration,
                'n_features': len(selected_features),
                'selected_feature': best_feature_name,
                'metrics': best_metrics
            })
            
            if self.verbose:
                print_flush(f"  Selected: {best_feature_name}")
                print_flush(f"  Test ROC-AUC: {best_metrics['test_roc_auc']:.4f}")
                print_flush(f"  Test Accuracy: {best_metrics['test_accuracy']:.4f}")
                print_flush(f"  Test F1: {best_metrics['test_f1']:.4f}")
        
        # Final model
        if self.verbose:
            print_flush(f"\nFinal: {len(selected_features)} features selected")
        
        X_train_final = X_train_scaled[:, selected_indices]
        X_test_final = X_test_scaled[:, selected_indices]
        
        final_model, final_metrics, _ = self.train_xgboost(
            X_train_final, y_train, X_test_final, y_test, selected_features
        )
        
        return {
            'final_model': final_model,
            'final_features': selected_features,
            'final_feature_indices': selected_indices,
            'history': self.selection_history
        }
    
    def visualize_results(self, output_path: Path):
        """Create visualizations of the selection process."""
        if self.verbose:
            print_flush("\n" + "="*70)
            print_flush("CREATING VISUALIZATIONS")
            print_flush("="*70)
        
        # Extract data for plotting
        iterations = [h['iteration'] for h in self.selection_history]
        n_features = [h['n_features'] for h in self.selection_history]
        test_roc_auc = [h['metrics']['test_roc_auc'] for h in self.selection_history]
        test_accuracy = [h['metrics']['test_accuracy'] for h in self.selection_history]
        test_f1 = [h['metrics']['test_f1'] for h in self.selection_history]
        train_roc_auc = [h['metrics']['train_roc_auc'] for h in self.selection_history]
        selected_features = [h['selected_feature'] for h in self.selection_history]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Forward Feature Selection Progress', fontsize=16, fontweight='bold')
        
        # Plot 1: Number of features vs Test ROC-AUC
        ax1 = axes[0, 0]
        ax1.plot(n_features, test_roc_auc, 'o-', linewidth=2, markersize=4, label='Test ROC-AUC')
        ax1.plot(n_features, train_roc_auc, 's--', linewidth=2, markersize=4, alpha=0.7, label='Train ROC-AUC')
        ax1.set_xlabel('Number of Features', fontsize=12)
        ax1.set_ylabel('ROC-AUC Score', fontsize=12)
        ax1.set_title('ROC-AUC vs Number of Features', fontsize=13)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Number of features vs Test Accuracy
        ax2 = axes[0, 1]
        ax2.plot(n_features, test_accuracy, 'o-', linewidth=2, markersize=4, color='green', label='Test Accuracy')
        ax2.set_xlabel('Number of Features', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Test Accuracy vs Number of Features', fontsize=13)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Number of features vs Test F1
        ax3 = axes[1, 0]
        ax3.plot(n_features, test_f1, 'o-', linewidth=2, markersize=4, color='orange', label='Test F1')
        ax3.set_xlabel('Number of Features', fontsize=12)
        ax3.set_ylabel('F1 Score', fontsize=12)
        ax3.set_title('Test F1 Score vs Number of Features', fontsize=13)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Feature selection order (top 50)
        ax4 = axes[1, 1]
        top_n = min(50, len(selected_features))
        feature_order = selected_features[:top_n]
        y_pos = np.arange(top_n)
        ax4.barh(y_pos, range(1, top_n + 1), color='steelblue', alpha=0.7)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels([f[:30] + '...' if len(f) > 30 else f for f in feature_order], fontsize=8)
        ax4.set_xlabel('Selection Order', fontsize=12)
        ax4.set_title(f'First {top_n} Selected Features (in order)', fontsize=13)
        ax4.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        if self.verbose:
            print_flush(f"Visualization saved to: {output_path}")
        
        plt.close()
    
    def run_selection(self, test_size: float = 0.2, random_state: int = 42):
        """Run complete forward selection pipeline."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.verbose:
            print_flush("="*70)
            print_flush("FORWARD FEATURE SELECTION WITH XGBOOST")
            print_flush("="*70)
            print_flush(f"K-mer size: {self.k_mer_size}")
            print_flush(f"Target features: {self.target_features}")
            print_flush("="*70)
        
        # Load data
        repertoires, labels = self.load_data()
        
        # Split train/test
        sample_ids = list(repertoires.keys())
        y = np.array([labels[sid] for sid in sample_ids])
        
        train_ids, test_ids = train_test_split(
            sample_ids, test_size=test_size, random_state=random_state, stratify=y
        )
        
        train_repertoires = {sid: repertoires[sid] for sid in train_ids}
        test_repertoires = {sid: repertoires[sid] for sid in test_ids}
        train_labels = {sid: labels[sid] for sid in train_ids}
        test_labels = {sid: labels[sid] for sid in test_ids}
        
        if self.verbose:
            print_flush(f"\nTrain/Test Split:")
            print_flush(f"  Train: {len(train_ids)} samples")
            print_flush(f"  Test: {len(test_ids)} samples")
        
        # Extract features
        if self.verbose:
            print_flush("\nExtracting features for training set...")
        X_train, feature_names = self.extract_all_features(train_repertoires, train_ids)
        if self.verbose:
            print_flush("\nExtracting features for test set...")
        X_test, _ = self.extract_all_features(test_repertoires, test_ids)
        
        y_train = np.array([train_labels[sid] for sid in train_ids])
        y_test = np.array([test_labels[sid] for sid in test_ids])
        
        # Perform forward selection
        if self.verbose:
            print_flush("\nStarting forward selection process...")
        results = self.forward_selection(X_train, y_train, X_test, y_test, feature_names)
        
        # Create visualizations
        viz_path = RESULTS_DIR / f"forward_selection_plot_{timestamp}.png"
        self.visualize_results(viz_path)
        
        # Save results
        output_data = {
            'timestamp': timestamp,
            'parameters': {
                'k_mer_size': self.k_mer_size,
                'target_features': self.target_features,
                'test_size': test_size,
                'random_state': random_state
            },
            'data_info': {
                'n_train': len(train_ids),
                'n_test': len(test_ids),
                'n_features_available': len(feature_names),
                'n_features_selected': len(results['final_features'])
            },
            'selected_features': results['final_features'],
            'selection_history': [
                {
                    'iteration': h['iteration'],
                    'n_features': h['n_features'],
                    'selected_feature': h['selected_feature'],
                    'metrics': h['metrics']
                }
                for h in self.selection_history
            ],
            'final_metrics': self.selection_history[-1]['metrics'] if self.selection_history else {}
        }
        
        results_path = RESULTS_DIR / f"forward_selection_results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        model_path = RESULTS_DIR / f"forward_selection_model_{timestamp}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': results['final_model'],
                'scaler': self.scaler,
                'final_features': results['final_features'],
                'final_feature_indices': results['final_feature_indices']
            }, f)
        
        if self.verbose:
            print_flush("\n" + "="*70)
            print_flush("SUMMARY")
            print_flush("="*70)
            print_flush(f"Initial features: {len(feature_names)}")
            print_flush(f"Selected features: {len(results['final_features'])}")
            if self.selection_history:
                final_metrics = self.selection_history[-1]['metrics']
                print_flush(f"Final Test ROC-AUC: {final_metrics['test_roc_auc']:.4f}")
                print_flush(f"Final Test Accuracy: {final_metrics['test_accuracy']:.4f}")
                print_flush(f"Final Test F1: {final_metrics['test_f1']:.4f}")
            print_flush(f"\nResults: {results_path}")
            print_flush(f"Model: {model_path}")
            print_flush(f"Visualization: {viz_path}")
        
        return results


def main():
    """Main function."""
    print_flush("Starting Forward Feature Selection...")
    
    selector = ForwardFeatureSelection(
        k_mer_size=3,
        target_features=200,
        verbose=True
    )
    
    results = selector.run_selection(test_size=0.2, random_state=42)
    
    print_flush("\n" + "="*70)
    print_flush("SELECTION COMPLETE")
    print_flush("="*70)


if __name__ == "__main__":
    main()

