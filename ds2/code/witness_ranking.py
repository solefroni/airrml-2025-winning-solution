"""
Witness-based ranking module for identifying immune-state-associated sequences.

Based on the competition paper concept: "witness sequences" are sequences that
appear in positive-labeled repertoires and are associated with the immune state.

This module provides functions to calculate witness enrichment scores and
combine them with model feature importance.
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


def calculate_witness_score(pos_count: int, neg_count: int, method: str = 'log2_fold_change') -> float:
    """
    Calculate witness enrichment score for a sequence.
    
    Args:
        pos_count: Number of times sequence appears in positive samples
        neg_count: Number of times sequence appears in negative samples
        method: Scoring method ('log2_fold_change', 'fold_change', 'ratio')
    
    Returns:
        Witness enrichment score
    """
    if method == 'log2_fold_change':
        # Log2 fold-change: log2((pos_count + 1) / (neg_count + 1))
        # +1 for pseudocount to avoid division by zero
        return np.log2((pos_count + 1) / (neg_count + 1))
    
    elif method == 'fold_change':
        # Simple fold-change: (pos_count + 1) / (neg_count + 1)
        return (pos_count + 1) / (neg_count + 1)
    
    elif method == 'ratio':
        # Positive ratio: pos_count / (pos_count + neg_count)
        total = pos_count + neg_count
        return pos_count / max(total, 1)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalize scores to [0, 1] range."""
    if len(scores) == 0:
        return scores
    min_score = scores.min()
    max_score = scores.max()
    if max_score == min_score:
        return np.ones_like(scores)
    return (scores - min_score) / (max_score - min_score)


def combine_witness_and_model_scores(
    witness_scores: np.ndarray,
    model_scores: np.ndarray,
    witness_weight: float = 0.6,
    normalize: bool = True
) -> np.ndarray:
    """
    Combine witness enrichment scores with model feature importance scores.
    
    Args:
        witness_scores: Array of witness enrichment scores
        model_scores: Array of model feature importance scores
        witness_weight: Weight for witness score (model_weight = 1 - witness_weight)
        normalize: Whether to normalize scores before combining
    
    Returns:
        Combined importance scores
    """
    if normalize:
        witness_scores = normalize_scores(witness_scores)
        model_scores = normalize_scores(model_scores)
    
    combined = witness_weight * witness_scores + (1 - witness_weight) * model_scores
    return combined


def rank_sequences_by_witness(
    sequence_data: Dict[str, Dict],
    model_importance_scores: Dict[str, float] = None,
    witness_weight: float = 0.6,
    witness_method: str = 'log2_fold_change',
    top_n: int = 50000
) -> List[Dict]:
    """
    Rank sequences by witness enrichment and model importance.
    
    Args:
        sequence_data: Dict mapping sequence to {'pos_count': int, 'neg_count': int, ...}
        model_importance_scores: Dict mapping sequence to model importance score
        witness_weight: Weight for witness score (0.6 = 60% witness, 40% model)
        witness_method: Method for calculating witness score
        top_n: Number of top sequences to return
    
    Returns:
        List of ranked sequence dictionaries with scores
    """
    scored_sequences = []
    
    for junction_aa, data in sequence_data.items():
        # Calculate witness score
        pos_count = data.get('pos_count', 0)
        neg_count = data.get('neg_count', 0)
        witness_score = calculate_witness_score(pos_count, neg_count, method=witness_method)
        
        # Get model importance score (if available)
        model_score = model_importance_scores.get(junction_aa, 0.0) if model_importance_scores else 0.0
        
        # Combine scores
        if model_importance_scores:
            # Normalize scores
            # We'll do this after collecting all scores
            combined_score = None  # Will be calculated after normalization
        else:
            # Only witness score available
            combined_score = witness_score
        
        # Store sequence data
        seq_data = {
            'junction_aa': junction_aa,
            'v_call': data.get('v_call', ''),
            'j_call': data.get('j_call', ''),
            'witness_score': witness_score,
            'model_score': model_score,
            'pos_count': pos_count,
            'neg_count': neg_count,
            'combined_score': combined_score
        }
        scored_sequences.append(seq_data)
    
    # Normalize and combine scores if model scores are available
    if model_importance_scores:
        witness_scores = np.array([s['witness_score'] for s in scored_sequences])
        model_scores = np.array([s['model_score'] for s in scored_sequences])
        combined_scores = combine_witness_and_model_scores(
            witness_scores, model_scores, witness_weight=witness_weight
        )
        
        # Update combined scores
        for i, score in enumerate(combined_scores):
            scored_sequences[i]['combined_score'] = score
    else:
        # Only witness scores, normalize them
        witness_scores = np.array([s['witness_score'] for s in scored_sequences])
        normalized = normalize_scores(witness_scores)
        for i, score in enumerate(normalized):
            scored_sequences[i]['combined_score'] = score
    
    # Sort by combined score (descending)
    scored_sequences.sort(key=lambda x: x['combined_score'], reverse=True)
    
    # Return top N
    return scored_sequences[:top_n]


def calculate_enrichment_metrics(ranked_sequences: List[Dict], top_k: int = 1000) -> Dict:
    """
    Calculate enrichment metrics for top-ranked sequences.
    
    Args:
        ranked_sequences: List of ranked sequence dictionaries
        top_k: Number of top sequences to analyze
    
    Returns:
        Dictionary with enrichment metrics
    """
    top_k_seqs = ranked_sequences[:top_k]
    
    total_pos = sum(s['pos_count'] for s in top_k_seqs)
    total_neg = sum(s['neg_count'] for s in top_k_seqs)
    
    # Calculate metrics
    pos_ratio = total_pos / max(total_pos + total_neg, 1)
    avg_witness_score = np.mean([s['witness_score'] for s in top_k_seqs])
    
    # Sequences that appear only in positive samples
    pos_only = sum(1 for s in top_k_seqs if s['pos_count'] > 0 and s['neg_count'] == 0)
    pos_only_ratio = pos_only / len(top_k_seqs)
    
    return {
        'top_k': top_k,
        'total_sequences': len(top_k_seqs),
        'total_pos_count': total_pos,
        'total_neg_count': total_neg,
        'pos_ratio': pos_ratio,
        'avg_witness_score': avg_witness_score,
        'pos_only_count': pos_only,
        'pos_only_ratio': pos_only_ratio
    }

