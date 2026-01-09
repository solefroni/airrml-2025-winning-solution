#!/usr/bin/env python3
"""
Extract Repertoire-Level Features
==================================

For each repertoire (50k TCRs), compute comprehensive features:
- V/J/D usage and frequencies
- Entropy of V and J usage
- CDR3 length and composition
- Histogram/moments of CDR3 length
- Amino acid composition
- 2-mer and 3-mer frequencies
- Templates/clonality metrics
- Distribution of templates (mean, variance, Gini, top clone fraction)
- Number of unique CDR3s, clonality indices
- Physicochemical summaries (hydrophobicity, charge, etc.)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from scipy import stats
from scipy.stats import entropy
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
import os
CODE_DIR = Path(__file__).parent.resolve()
CACHE_DIR = Path(os.environ.get('DS8_CACHE_DIR', '../cache'))
CACHE_INDEX = CACHE_DIR / "cache_index.json"
METADATA_PATH = Path(os.environ.get('DS8_TRAIN_DATA', '../input')) / 'metadata.csv'
OUTPUT_DIR = CODE_DIR.parent / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Amino acid properties
AA_HYDROPHOBICITY = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
}

AA_CHARGE = {
    'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0,
    'G': 0, 'H': 1, 'I': 0, 'K': 1, 'L': 0,
    'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1,
    'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0
}

AA_POLARITY = {
    'A': 0, 'C': 0, 'D': 1, 'E': 1, 'F': 0,
    'G': 0, 'H': 1, 'I': 0, 'K': 1, 'L': 0,
    'M': 0, 'N': 1, 'P': 0, 'Q': 1, 'R': 1,
    'S': 1, 'T': 1, 'V': 0, 'W': 0, 'Y': 1
}

def calculate_gini(x):
    """Calculate Gini coefficient."""
    if len(x) == 0:
        return 0.0
    sorted_x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_x)) / (n * np.sum(sorted_x)) - (n + 1) / n

def calculate_shannon_entropy(counts):
    """Calculate Shannon entropy from counts."""
    if len(counts) == 0 or np.sum(counts) == 0:
        return 0.0
    probs = counts / np.sum(counts)
    probs = probs[probs > 0]  # Remove zeros
    return -np.sum(probs * np.log2(probs))

def extract_vjd_usage(df):
    """Extract V, J, D gene usage features."""
    features = {}
    
    # V gene usage
    if 'v_call' in df.columns:
        v_counts = df['v_call'].value_counts()
        v_freqs = v_counts / len(df)
        
        # Individual V gene frequencies (top 50)
        for i, (v_gene, freq) in enumerate(v_counts.head(50).items()):
            features[f'v_freq_{v_gene}'] = freq / len(df)
        
        # V gene entropy
        features['v_entropy'] = calculate_shannon_entropy(v_counts.values)
        features['v_unique_count'] = len(v_counts)
        features['v_diversity'] = len(v_counts) / len(df) if len(df) > 0 else 0
        
        # Top V gene fraction
        if len(v_counts) > 0:
            features['v_top1_freq'] = v_freqs.iloc[0] if len(v_freqs) > 0 else 0
            features['v_top5_freq'] = v_freqs.head(5).sum()
            features['v_top10_freq'] = v_freqs.head(10).sum()
    
    # J gene usage
    if 'j_call' in df.columns:
        j_counts = df['j_call'].value_counts()
        j_freqs = j_counts / len(df)
        
        # Individual J gene frequencies (top 30)
        for i, (j_gene, freq) in enumerate(j_counts.head(30).items()):
            features[f'j_freq_{j_gene}'] = freq / len(df)
        
        # J gene entropy
        features['j_entropy'] = calculate_shannon_entropy(j_counts.values)
        features['j_unique_count'] = len(j_counts)
        features['j_diversity'] = len(j_counts) / len(df) if len(df) > 0 else 0
        
        # Top J gene fraction
        if len(j_counts) > 0:
            features['j_top1_freq'] = j_freqs.iloc[0] if len(j_freqs) > 0 else 0
            features['j_top5_freq'] = j_freqs.head(5).sum()
            features['j_top10_freq'] = j_freqs.head(10).sum()
    
    # D gene usage
    if 'd_call' in df.columns:
        d_counts = df['d_call'].value_counts()
        d_freqs = d_counts / len(df)
        
        # Individual D gene frequencies (top 20)
        for i, (d_gene, freq) in enumerate(d_counts.head(20).items()):
            features[f'd_freq_{d_gene}'] = freq / len(df)
        
        # D gene entropy
        features['d_entropy'] = calculate_shannon_entropy(d_counts.values)
        features['d_unique_count'] = len(d_counts)
        features['d_diversity'] = len(d_counts) / len(df) if len(df) > 0 else 0
        
        # Top D gene fraction
        if len(d_counts) > 0:
            features['d_top1_freq'] = d_freqs.iloc[0] if len(d_freqs) > 0 else 0
            features['d_top5_freq'] = d_freqs.head(5).sum()
    
    return features

def extract_cdr3_features(df):
    """Extract CDR3 length and composition features."""
    features = {}
    
    if 'junction_aa' not in df.columns:
        return features
    
    # CDR3 lengths
    lengths = df['junction_aa'].str.len()
    features['cdr3_length_mean'] = lengths.mean()
    features['cdr3_length_median'] = lengths.median()
    features['cdr3_length_std'] = lengths.std()
    features['cdr3_length_min'] = lengths.min()
    features['cdr3_length_max'] = lengths.max()
    features['cdr3_length_q25'] = lengths.quantile(0.25)
    features['cdr3_length_q75'] = lengths.quantile(0.75)
    features['cdr3_length_skew'] = lengths.skew()
    features['cdr3_length_kurtosis'] = lengths.kurtosis()
    
    # Length histogram (bins: ≤20, 21-25, 26-30, 31+)
    length_bins = pd.cut(lengths, bins=[0, 20, 25, 30, 100], labels=['short', 'medium', 'long', 'very_long'])
    bin_counts = length_bins.value_counts()
    total = len(lengths)
    if total > 0:
        for label in ['short', 'medium', 'long', 'very_long']:
            count = bin_counts.get(label, 0)
            features[f'cdr3_length_{label}_pct'] = count / total if isinstance(count, (int, float)) else 0
    
    # Unique CDR3s
    unique_cdr3s = df['junction_aa'].nunique()
    features['unique_cdr3_count'] = unique_cdr3s
    features['unique_cdr3_fraction'] = unique_cdr3s / len(df) if len(df) > 0 else 0
    
    # Amino acid composition
    all_sequences = df['junction_aa'].astype(str)
    aa_counter = Counter()
    for seq in all_sequences:
        aa_counter.update(seq)
    
    total_aa = sum(aa_counter.values())
    if total_aa > 0:
        for aa in 'ACDEFGHIKLMNPQRSTVWY':
            features[f'aa_freq_{aa}'] = aa_counter.get(aa, 0) / total_aa
    
    return features

def extract_kmer_features(df, k=2, prefix=''):
    """Extract k-mer frequency features."""
    features = {}
    
    if 'junction_aa' not in df.columns:
        return features
    
    # Global k-mer frequencies (across all sequences)
    all_sequences = df['junction_aa'].astype(str)
    kmer_counter = Counter()
    
    for seq in all_sequences:
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            kmer_counter[kmer] += 1
    
    total_kmers = sum(kmer_counter.values())
    if total_kmers > 0:
        # Top 100 most frequent k-mers
        top_kmers = kmer_counter.most_common(100)
        for kmer, count in top_kmers:
            features[f'{prefix}kmer_{kmer}_freq'] = count / total_kmers
    
    return features

def extract_clonality_features(df):
    """Extract clonality and template distribution features."""
    features = {}
    
    if 'templates' not in df.columns:
        return features
    
    templates = df['templates'].values
    total_templates = templates.sum()
    
    if total_templates == 0:
        return features
    
    # Basic statistics
    features['total_templates'] = total_templates
    features['mean_templates'] = templates.mean()
    features['median_templates'] = np.median(templates)
    features['std_templates'] = templates.std()
    features['min_templates'] = templates.min()
    features['max_templates'] = templates.max()
    features['var_templates'] = templates.var()
    
    # Gini coefficient
    features['gini_coefficient'] = calculate_gini(templates)
    
    # Top clone fractions
    sorted_templates = np.sort(templates)[::-1]
    features['top1_clone_fraction'] = sorted_templates[0] / total_templates if len(sorted_templates) > 0 else 0
    features['top5_clone_fraction'] = sorted_templates[:5].sum() / total_templates if len(sorted_templates) >= 5 else sorted_templates.sum() / total_templates
    features['top10_clone_fraction'] = sorted_templates[:10].sum() / total_templates if len(sorted_templates) >= 10 else sorted_templates.sum() / total_templates
    features['top100_clone_fraction'] = sorted_templates[:100].sum() / total_templates if len(sorted_templates) >= 100 else sorted_templates.sum() / total_templates
    
    # Clonality indices
    # Simpson's diversity index (1 - D)
    freqs = templates / total_templates
    simpson_d = np.sum(freqs ** 2)
    features['simpson_diversity'] = 1 - simpson_d
    
    # Shannon entropy of templates
    features['template_entropy'] = calculate_shannon_entropy(templates)
    
    # Effective number of clones
    features['effective_clones'] = 1 / simpson_d if simpson_d > 0 else 0
    
    return features

def extract_physicochemical_features(df):
    """Extract physicochemical properties of CDR3 sequences."""
    features = {}
    
    if 'junction_aa' not in df.columns:
        return features
    
    all_sequences = df['junction_aa'].astype(str)
    
    # Per-sequence properties, then aggregate
    hydrophobicity_values = []
    charge_values = []
    polarity_values = []
    
    for seq in all_sequences:
        if len(seq) == 0:
            continue
        
        # Calculate per-sequence averages
        h_vals = [AA_HYDROPHOBICITY.get(aa, 0) for aa in seq]
        c_vals = [AA_CHARGE.get(aa, 0) for aa in seq]
        p_vals = [AA_POLARITY.get(aa, 0) for aa in seq]
        
        hydrophobicity_values.append(np.mean(h_vals))
        charge_values.append(np.mean(c_vals))
        polarity_values.append(np.mean(p_vals))
    
    if len(hydrophobicity_values) > 0:
        # Hydrophobicity
        features['hydrophobicity_mean'] = np.mean(hydrophobicity_values)
        features['hydrophobicity_std'] = np.std(hydrophobicity_values)
        features['hydrophobicity_median'] = np.median(hydrophobicity_values)
        
        # Charge
        features['charge_mean'] = np.mean(charge_values)
        features['charge_std'] = np.std(charge_values)
        features['charge_median'] = np.median(charge_values)
        
        # Polarity
        features['polarity_mean'] = np.mean(polarity_values)
        features['polarity_std'] = np.std(polarity_values)
        features['polarity_median'] = np.median(polarity_values)
    
    return features

def extract_all_features(df):
    """Extract all repertoire-level features."""
    features = {}
    
    # V/J/D usage
    features.update(extract_vjd_usage(df))
    
    # CDR3 features
    features.update(extract_cdr3_features(df))
    
    # K-mer features (2-mer and 3-mer)
    features.update(extract_kmer_features(df, k=2, prefix='2mer_'))
    features.update(extract_kmer_features(df, k=3, prefix='3mer_'))
    
    # Clonality features
    features.update(extract_clonality_features(df))
    
    # Physicochemical features
    features.update(extract_physicochemical_features(df))
    
    return features

def main():
    print("="*60)
    print("Extracting Repertoire-Level Features")
    print("="*60)
    
    # Load cache index
    print("Loading cache index...")
    with open(CACHE_INDEX, 'r') as f:
        cache_info = json.load(f)
    
    # Load metadata
    print("Loading metadata...")
    metadata_df = pd.read_csv(METADATA_PATH)
    print(f"Loaded {len(metadata_df)} samples from metadata")
    
    # Extract features for each repertoire
    print("\nExtracting features...")
    all_features = []
    sample_ids = []
    labels = []
    
    samples_info = cache_info['samples']
    
    for rep_id, info in tqdm(samples_info.items(), total=len(samples_info)):
        cache_file = Path(info['cache_file'])
        
        if not cache_file.exists():
            continue
        
        # Load downsampled repertoire
        try:
            with open(cache_file, 'rb') as f:
                df = pickle.load(f)
            
            # Extract features
            features = extract_all_features(df)
            features['repertoire_id'] = rep_id
            
            all_features.append(features)
            sample_ids.append(rep_id)
            
            # Get label from metadata
            meta_row = metadata_df[metadata_df['repertoire_id'] == rep_id]
            if len(meta_row) > 0:
                label = meta_row['label_positive'].iloc[0]
                labels.append(1 if label else 0)
            else:
                labels.append(0)
                
        except Exception as e:
            print(f"Error processing {rep_id}: {e}")
            continue
    
    # Create feature matrix
    print("\nCreating feature matrix...")
    features_df = pd.DataFrame(all_features)
    
    # Set repertoire_id as index
    features_df.set_index('repertoire_id', inplace=True)
    
    # Fill NaN values with 0
    features_df = features_df.fillna(0)
    
    # Add labels
    labels_df = pd.DataFrame({
        'repertoire_id': sample_ids,
        'label': labels
    })
    
    print(f"\nExtracted {len(features_df)} samples with {len(features_df.columns)} features")
    print(f"Feature matrix shape: {features_df.shape}")
    print(f"Positive samples: {sum(labels)}, Negative: {len(labels) - sum(labels)}")
    
    # Save feature matrix and labels
    output_file = OUTPUT_DIR / "repertoire_features.csv"
    labels_file = OUTPUT_DIR / "labels.csv"
    
    features_df.to_csv(output_file)
    labels_df.to_csv(labels_file, index=False)
    
    print(f"\nSaved feature matrix to: {output_file}")
    print(f"Saved labels to: {labels_file}")
    
    # Print feature summary
    print(f"\nFeature categories:")
    print(f"  V/J/D usage: {len([c for c in features_df.columns if c.startswith(('v_', 'j_', 'd_'))])} features")
    print(f"  CDR3 features: {len([c for c in features_df.columns if c.startswith('cdr3_')])} features")
    print(f"  2-mer features: {len([c for c in features_df.columns if '2mer_' in c])} features")
    print(f"  3-mer features: {len([c for c in features_df.columns if '3mer_' in c])} features")
    print(f"  Clonality features: {len([c for c in features_df.columns if any(x in c for x in ['template', 'clone', 'gini', 'simpson', 'diversity'])])} features")
    print(f"  Physicochemical features: {len([c for c in features_df.columns if any(x in c for x in ['hydrophobicity', 'charge', 'polarity'])])} features")
    print(f"  AA composition: {len([c for c in features_df.columns if c.startswith('aa_freq_')])} features")
    
    return features_df, labels_df

if __name__ == "__main__":
    main()

