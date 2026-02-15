#!/usr/bin/env python3
"""
Caching Utilities for Graph Classification
==========================================

Provides caching functions for embeddings and graphs.
"""

import pickle
import hashlib
from pathlib import Path
import numpy as np
from typing import List, Optional

# Cache directory - will be set by graph_classification.py or via environment variable
import os
GRAPH_CACHE_DIR = Path(os.environ.get('DS8_GRAPH_CACHE', '../cache/graphs'))
OLD_CACHE_DIR = None
NEW_CACHE_DIR = None

def set_cache_dir(cache_dir, old_cache_dir=None, new_cache_dir=None):
    """Set the cache directory (called from graph_classification.py).
    
    Args:
        cache_dir: Primary cache directory (for saving)
        old_cache_dir: Optional old cache directory (for loading)
        new_cache_dir: Optional new cache directory (for loading)
    """
    global GRAPH_CACHE_DIR, OLD_CACHE_DIR, NEW_CACHE_DIR
    GRAPH_CACHE_DIR = Path(cache_dir)
    GRAPH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (GRAPH_CACHE_DIR / "embeddings").mkdir(parents=True, exist_ok=True)
    (GRAPH_CACHE_DIR / "graphs").mkdir(parents=True, exist_ok=True)
    
    # Set both cache directories for checking
    if old_cache_dir is not None:
        OLD_CACHE_DIR = Path(old_cache_dir)
    else:
        OLD_CACHE_DIR = None
    if new_cache_dir is not None:
        NEW_CACHE_DIR = Path(new_cache_dir)
    else:
        NEW_CACHE_DIR = None


def get_sequence_hash(sequences_or_key, embedder_params=None) -> str:
    """Get hash of sequence list or cache key for caching.
    
    Args:
        sequences_or_key: Either a list of sequences, or a tuple of (sequences, v_genes, j_genes)
        embedder_params: Optional dict of embedder-specific parameters (e.g., {'n_references': 1000, 'organism': 'human', 'chain': 'beta'})
    """
    if isinstance(sequences_or_key, tuple):
        # Handle tuple cache key (sequences, v_genes, j_genes)
        sequences, v_genes, j_genes = sequences_or_key
        seq_str = "|".join(sorted(sequences))
        v_str = "|".join(str(v) for v in v_genes)
        j_str = "|".join(str(j) for j in j_genes)
        combined = f"{seq_str}|||{v_str}|||{j_str}"
    else:
        # Handle list of sequences
        seq_str = "|".join(sorted(sequences_or_key))
        combined = seq_str
    
    # Add embedder-specific parameters to hash if provided
    if embedder_params:
        param_str = "|".join(f"{k}:{v}" for k, v in sorted(embedder_params.items()))
        combined = f"{combined}|||{param_str}"
    
    return hashlib.md5(combined.encode()).hexdigest()


def get_graph_cache_key(rep_id: str, k_value: int, sequences: List[str], embedder_type: str = "cvc") -> str:
    """Get cache key for a graph.
    
    Args:
        rep_id: Repertoire ID
        k_value: K value for KNN
        sequences: List of sequences
        embedder_type: Type of embedder ('cvc' only)
    """
    seq_hash = get_sequence_hash(sequences)
    return f"{rep_id}_k{k_value}_{embedder_type}_{seq_hash}"


def load_cached_embeddings(sequences_or_key, embedder_params=None) -> Optional[np.ndarray]:
    """Load cached embeddings if available. Checks both old and new cache directories.
    
    Args:
        sequences_or_key: Either a list of sequences, or a tuple of (sequences, v_genes, j_genes)
        embedder_params: Optional dict of embedder-specific parameters
    """
    seq_hash = get_sequence_hash(sequences_or_key, embedder_params)
    
    # Extract sequences for comparison
    if isinstance(sequences_or_key, tuple):
        sequences = sequences_or_key[0]
    else:
        sequences = sequences_or_key
    
    # List of cache directories to check (in order: old, new, primary)
    cache_dirs_to_check = []
    if OLD_CACHE_DIR is not None and OLD_CACHE_DIR.exists():
        cache_dirs_to_check.append(OLD_CACHE_DIR)
    if NEW_CACHE_DIR is not None and NEW_CACHE_DIR.exists() and NEW_CACHE_DIR != GRAPH_CACHE_DIR:
        cache_dirs_to_check.append(NEW_CACHE_DIR)
    # Always check primary cache directory last (for saving location)
    if GRAPH_CACHE_DIR not in cache_dirs_to_check:
        cache_dirs_to_check.append(GRAPH_CACHE_DIR)
    
    # Try each cache directory
    for cache_dir in cache_dirs_to_check:
        cache_file = cache_dir / "embeddings" / f"{seq_hash}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    # Verify sequences match (compare as sets to handle different orders)
                    # The hash already ensures same sequences, but we need to verify
                    # and ensure embeddings are in the same order as input sequences
                    cached_seqs = cached_data['sequences']
                    if set(cached_seqs) == set(sequences) and len(cached_seqs) == len(sequences):
                        # Reorder embeddings to match input sequence order
                        seq_to_idx = {seq: idx for idx, seq in enumerate(cached_seqs)}
                        reordered_indices = [seq_to_idx[seq] for seq in sequences]
                        return cached_data['embeddings'][reordered_indices]
            except Exception as e:
                print(f"Warning: Error loading cached embeddings from {cache_dir}: {e}")
                continue
    
    return None


def save_cached_embeddings(sequences_or_key, embeddings: np.ndarray, embedder_params=None):
    """Save embeddings to cache.
    
    Args:
        sequences_or_key: Either a list of sequences, or a tuple of (sequences, v_genes, j_genes)
        embeddings: Embeddings to cache
        embedder_params: Optional dict of embedder-specific parameters
    """
    seq_hash = get_sequence_hash(sequences_or_key, embedder_params)
    cache_file = GRAPH_CACHE_DIR / "embeddings" / f"{seq_hash}.pkl"
    
    # Extract sequences for storage
    if isinstance(sequences_or_key, tuple):
        sequences = sequences_or_key[0]
    else:
        sequences = sequences_or_key
    
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'sequences': sequences,
                'embeddings': embeddings
            }, f)
    except Exception as e:
        print(f"Warning: Error saving cached embeddings: {e}")


def load_cached_graph(rep_id: str, k_value: int, sequences: List[str], embedder_type: str = "cvc"):
    """Load cached graph if available. Checks both old and new cache directories.
    
    Args:
        rep_id: Repertoire ID
        k_value: K value for KNN
        sequences: List of sequences
        embedder_type: Type of embedder ('cvc' only)
    """
    import torch
    from torch_geometric.data import Data
    
    cache_key = get_graph_cache_key(rep_id, k_value, sequences, embedder_type)
    
    # List of cache directories to check (in order: old, new, primary)
    cache_dirs_to_check = []
    if OLD_CACHE_DIR is not None and OLD_CACHE_DIR.exists():
        cache_dirs_to_check.append(OLD_CACHE_DIR)
    if NEW_CACHE_DIR is not None and NEW_CACHE_DIR.exists() and NEW_CACHE_DIR != GRAPH_CACHE_DIR:
        cache_dirs_to_check.append(NEW_CACHE_DIR)
    # Always check primary cache directory last (for saving location)
    if GRAPH_CACHE_DIR not in cache_dirs_to_check:
        cache_dirs_to_check.append(GRAPH_CACHE_DIR)
    
    # Try each cache directory
    for cache_dir in cache_dirs_to_check:
        cache_file = cache_dir / "graphs" / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    # Verify sequences match and embedder type matches
                    if (cached_data['sequences'] == sequences and 
                        cached_data['k'] == k_value and 
                        cached_data.get('embedder_type', 'cvc') == embedder_type):
                        graph = cached_data['graph']
                        # Ensure all tensors are on CPU (fix for device mismatch issues)
                        if isinstance(graph, Data):
                            # Use PyG's built-in method to move all tensors to CPU
                            graph = graph.to('cpu')
                        return graph
            except Exception as e:
                print(f"Warning: Error loading cached graph from {cache_dir}: {e}")
                continue
    
    return None


def save_cached_graph(rep_id: str, k_value: int, sequences: List[str], graph, embedder_type: str = "cvc"):
    """Save graph to cache.
    
    Args:
        rep_id: Repertoire ID
        k_value: K value for KNN
        sequences: List of sequences
        graph: Graph to cache
        embedder_type: Type of embedder ('cvc' only)
    """
    import torch
    from torch_geometric.data import Data
    
    cache_key = get_graph_cache_key(rep_id, k_value, sequences, embedder_type)
    cache_file = GRAPH_CACHE_DIR / "graphs" / f"{cache_key}.pkl"
    
    try:
        # Ensure graph is on CPU before saving (for DataLoader compatibility)
        if isinstance(graph, Data):
            graph = graph.to('cpu')
        
        # Ensure directory exists
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'rep_id': rep_id,
                'k': k_value,
                'sequences': sequences,
                'graph': graph,
                'embedder_type': embedder_type
            }, f)
    except Exception as e:
        print(f"Warning: Error saving cached graph: {e}")
        import traceback
        traceback.print_exc()

