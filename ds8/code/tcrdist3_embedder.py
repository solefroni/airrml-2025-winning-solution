#!/usr/bin/env python3
"""
TCRdist3 Embedder for TCR Sequences
===================================

Uses TCRdist3 to generate embeddings by computing distances to reference/prototype TCRs.
TCRdist3 computes CDR-based distance metrics between TCR sequences.
Based on: https://github.com/kmayerb/tcrdist3
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Union, Optional
from tqdm import tqdm
import warnings
import sys
warnings.filterwarnings('ignore')

# Use system/pip-installed tcrdist3 (pip install tcrdist3)
try:
    from tcrdist.repertoire import TCRrep
    import pwseqdist
    TCRDIST3_AVAILABLE = True
except ImportError as e:
    TCRDIST3_AVAILABLE = False
    IMPORT_ERROR = str(e)
    print(f"Warning: TCRdist3 not available: {e}")


class TCRdist3Embedder:
    """
    Embedder using TCRdist3 (distance-based TCR embedding).
    
    TCRdist3 computes distances from input TCR sequences to a set of reference/prototype sequences.
    The embedding is a vector of distances to reference TCRs.
    """
    
    def __init__(
        self,
        device=None,
        batch_size=1024,
        use_fp16=False,
        verbose=True,
        n_references=1000,
        organism='human',
        chain='beta',
        cpus=1,
        reference_tcrs=None
    ):
        """
        Initialize TCRdist3 embedder.
        
        Args:
            device: Not used (TCRdist3 is CPU-based)
            batch_size: Batch size for processing (not used directly)
            use_fp16: Not used
            verbose: Print progress messages
            n_references: Number of reference TCRs to use (default 1000)
            organism: Organism ('human' or 'mouse')
            chain: Chain type ('alpha', 'beta', or ['alpha', 'beta'])
            cpus: Number of CPUs to use for distance computation
            reference_tcrs: Optional DataFrame with reference TCRs. If None, will sample from input.
        """
        if not TCRDIST3_AVAILABLE:
            raise ImportError(
                f"TCRdist3 is not available. Error: {IMPORT_ERROR if 'IMPORT_ERROR' in globals() else 'Unknown'}\n"
                "Please install with: pip install tcrdist3"
            )
        
        self.verbose = verbose
        self.n_references = n_references
        self.organism = organism
        self.chain = chain if isinstance(chain, list) else [chain]
        self.cpus = cpus
        self.reference_tcrs = reference_tcrs
        self.embedding_dim = None  # Will be determined after first embedding
        
        # Initialize a minimal TCRrep to get the distance metrics
        # We'll use these metrics with pwseqdist directly
        dummy_df = pd.DataFrame({
            'cdr3_b_aa': ['CASSPGQGGYEQYF'] if 'beta' in self.chain else ['CASSPGQGGYEQYF'],
            'v_b_gene': ['TRBV19*01'] if 'beta' in self.chain else None,
            'j_b_gene': ['TRBJ2-1*01'] if 'beta' in self.chain else None,
            'cdr3_a_aa': ['CASSPGQGGYEQYF'] if 'alpha' in self.chain else None,
            'v_a_gene': ['TRAV12-1*01'] if 'alpha' in self.chain else None,
            'j_a_gene': ['TRAJ42*01'] if 'alpha' in self.chain else None,
            'count': [1]
        })
        dummy_tr = TCRrep(cell_df=dummy_df, organism=organism, chains=self.chain, compute_distances=False)
        
        # Get the CDR3 distance metric (primary metric for embeddings)
        if 'beta' in self.chain:
            self.metric = dummy_tr.metrics_b.get('cdr3_b_aa')
        elif 'alpha' in self.chain:
            self.metric = dummy_tr.metrics_a.get('cdr3_a_aa')
        else:
            raise ValueError(f"Chain {self.chain} not supported")
        
        if self.metric is None:
            raise ValueError(f"Could not get CDR3 distance metric for chain {self.chain}")
        
        if self.verbose:
            print(f"TCRdist3 Embedder initialized (OPTIMIZED - direct pwseqdist):")
            print(f"  - Reference TCRs: {n_references}")
            print(f"  - Organism: {organism}")
            print(f"  - Chain: {chain}")
            print(f"  - CPUs: {cpus}")
            print(f"  - Numba: Enabled")
            print(f"  - Computing only input→reference distances (not full pairwise)")
    
    def _prepare_dataframe(
        self,
        sequences: List[str],
        v_genes: Optional[List[str]] = None,
        j_genes: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Prepare input dataframe for TCRdist3.
        
        Args:
            sequences: List of CDR3 amino acid sequences
            v_genes: Optional list of V gene names
            j_genes: Optional list of J gene names
            
        Returns:
            DataFrame with columns required by TCRdist3
        """
        data = {}
        
        if 'beta' in self.chain:
            data['cdr3_b_aa'] = sequences
            if v_genes is not None:
                # TCRdist3 expects allele format (e.g., TRBV19*01)
                data['v_b_gene'] = [v if '*' in str(v) else f"{v}*01" if v is not None else None for v in v_genes]
            else:
                data['v_b_gene'] = None
            if j_genes is not None:
                data['j_b_gene'] = [j if '*' in str(j) else f"{j}*01" if j is not None else None for j in j_genes]
            else:
                data['j_b_gene'] = None
        elif 'alpha' in self.chain:
            data['cdr3_a_aa'] = sequences
            if v_genes is not None:
                data['v_a_gene'] = [v if '*' in str(v) else f"{v}*01" if v is not None else None for v in v_genes]
            else:
                data['v_a_gene'] = None
            if j_genes is not None:
                data['j_a_gene'] = [j if '*' in str(j) else f"{j}*01" if j is not None else None for j in j_genes]
            else:
                data['j_a_gene'] = None
        else:
            raise ValueError(f"Chain {self.chain} not supported. Use 'alpha' or 'beta'")
        
        df = pd.DataFrame(data)
        
        # Filter out rows with missing CDR3 (TCRdist3 requires CDR3)
        cdr3_col = 'cdr3_b_aa' if 'beta' in self.chain else 'cdr3_a_aa'
        
        # Convert to string and filter out empty/None values
        df[cdr3_col] = df[cdr3_col].astype(str)
        df = df[
            df[cdr3_col].notna() & 
            (df[cdr3_col] != '') & 
            (df[cdr3_col] != 'nan') &
            (df[cdr3_col].str.len() > 0)
        ].copy()
        
        if len(df) == 0:
            raise ValueError(f"No valid sequences after filtering (CDR3 column: {cdr3_col})")
        
        # Add count column (required by TCRdist3)
        df['count'] = 1
        
        return df
    
    def _get_or_create_references(
        self,
        input_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Get or create reference TCRs for distance computation.
        
        If reference_tcrs is provided, use it. Otherwise, sample from input.
        """
        if self.reference_tcrs is not None:
            return self.reference_tcrs.copy()
        
        # Sample reference TCRs from input
        n_sample = min(self.n_references, len(input_df))
        references = input_df.sample(n=n_sample, random_state=42).copy()
        
        if self.verbose:
            print(f"  Using {len(references)} reference TCRs (sampled from input)")
        
        return references
    
    def embed(
        self,
        sequences: Union[List[str], np.ndarray],
        v_genes: Optional[Union[List[str], np.ndarray]] = None,
        j_genes: Optional[Union[List[str], np.ndarray]] = None,
        batch_size: int = None
    ) -> np.ndarray:
        """
        Generate embeddings for TCR sequences using TCRdist3.
        
        Args:
            sequences: List of CDR3 amino acid sequences
            v_genes: Optional list of V gene names
            j_genes: Optional list of J gene names
            batch_size: Batch size (kept for API compatibility)
            
        Returns:
            numpy array of embeddings (n_sequences, embedding_dim)
        """
        if isinstance(sequences, np.ndarray):
            sequences = sequences.tolist()
        if v_genes is not None and isinstance(v_genes, np.ndarray):
            v_genes = v_genes.tolist()
        if j_genes is not None and isinstance(j_genes, np.ndarray):
            j_genes = j_genes.tolist()
        
        if len(sequences) == 0:
            return np.array([])
        
        # Try to load from cache first
        try:
            from cache_utils import load_cached_embeddings
            
            # Create embedder-specific cache parameters
            embedder_params = {
                'embedder': 'tcrdist3',
                'n_references': self.n_references,
                'organism': self.organism,
                'chain': '_'.join(self.chain) if isinstance(self.chain, list) else self.chain
            }
            
            # Create cache key
            if v_genes is not None and j_genes is not None:
                cache_key = (sequences, tuple(v_genes), tuple(j_genes))
            else:
                cache_key = sequences
            
            cached_embeddings = load_cached_embeddings(cache_key, embedder_params)
            if cached_embeddings is not None:
                if self.verbose:
                    print(f"✓ Loaded TCRdist3 embeddings from cache ({len(sequences):,} sequences)")
                return cached_embeddings
        except Exception as e:
            if self.verbose:
                print(f"  Note: Could not check cache ({e}), will compute embeddings")
        
        if self.verbose:
            print(f"Generating TCRdist3 embeddings for {len(sequences):,} sequences...")
        
        # Prepare input dataframe
        input_df = self._prepare_dataframe(sequences, v_genes, j_genes)
        
        # Get or create reference TCRs
        if self.verbose:
            print(f"  Preparing reference TCRs...")
        reference_df = self._get_or_create_references(input_df)
        
        # Ensure both dataframes have valid sequences
        cdr3_col = 'cdr3_b_aa' if 'beta' in self.chain else 'cdr3_a_aa'
        
        # Filter out any empty or invalid sequences
        reference_df = reference_df[
            reference_df[cdr3_col].notna() & 
            (reference_df[cdr3_col].astype(str).str.len() > 0)
        ].copy()
        input_df = input_df[
            input_df[cdr3_col].notna() & 
            (input_df[cdr3_col].astype(str).str.len() > 0)
        ].copy()
        
        if len(reference_df) == 0:
            raise ValueError(f"No valid reference TCRs after filtering (CDR3 column: {cdr3_col})")
        if len(input_df) == 0:
            raise ValueError(f"No valid input sequences after filtering (CDR3 column: {cdr3_col})")
        
        if self.verbose:
            print(f"  Valid sequences: {len(input_df):,} input, {len(reference_df):,} references")
        
        # OPTIMIZED: Use pwseqdist directly to compute ONLY input→reference distances
        # Process in batches to avoid OOM errors with large datasets
        cdr3_col = 'cdr3_b_aa' if 'beta' in self.chain else 'cdr3_a_aa'
        
        input_seqs = input_df[cdr3_col].tolist()
        reference_seqs = reference_df[cdr3_col].tolist()
        
        if self.verbose:
            print(f"  Computing distances using pwseqdist (Numba-optimized, batched)...")
            print(f"    Input sequences: {len(input_seqs):,}")
            print(f"    Reference sequences: {len(reference_seqs):,}")
            print(f"    Total distances to compute: {len(input_seqs):,} × {len(reference_seqs):,} = {len(input_seqs) * len(reference_seqs):,}")
        
        # Process in batches to avoid OOM
        # Batch size: process 200K sequences at a time (further reduced to avoid OOM)
        # Each batch computes 200K × 1K = 200M distances, ~0.75GB memory
        # pwseqdist uses additional memory for intermediate calculations, so we need headroom
        batch_size = 200_000
        n_batches = (len(input_seqs) + batch_size - 1) // batch_size
        
        if self.verbose:
            print(f"    Processing in {n_batches} batches of ~{batch_size:,} sequences each")
        
        all_embeddings_list = []
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(input_seqs))
            batch_seqs = input_seqs[start_idx:end_idx]
            
            if self.verbose and (i == 0 or (i + 1) % max(1, n_batches // 10) == 0 or i == n_batches - 1):
                print(f"    Batch {i+1}/{n_batches}: processing sequences {start_idx:,} to {end_idx:,}...")
            
            # Compute distances for this batch
            batch_dist = pwseqdist.apply_pairwise_rect(
                metric=self.metric,
                seqs1=batch_seqs,      # Batch of input sequences (rows)
                seqs2=reference_seqs,  # Reference sequences (columns)
                ncpus=self.cpus,
                use_numba=True,        # Enable Numba optimization
                uniqify=True
            )
            
            all_embeddings_list.append(batch_dist.astype(np.float32))
        
        # Concatenate all batches
        all_embeddings = np.vstack(all_embeddings_list)
        
        # Set embedding dimension
        if self.embedding_dim is None:
            self.embedding_dim = all_embeddings.shape[1]
            if self.verbose:
                print(f"  Embedding dimension: {self.embedding_dim}")
        
        if self.verbose:
            print(f"  Generated embeddings: {all_embeddings.shape}")
        
        # Save to cache for future use
        try:
            from cache_utils import save_cached_embeddings
            
            # Create embedder-specific cache parameters
            embedder_params = {
                'embedder': 'tcrdist3',
                'n_references': self.n_references,
                'organism': self.organism,
                'chain': '_'.join(self.chain) if isinstance(self.chain, list) else self.chain
            }
            
            # Create cache key
            if v_genes is not None and j_genes is not None:
                cache_key = (sequences, tuple(v_genes), tuple(j_genes))
            else:
                cache_key = sequences
            
            save_cached_embeddings(cache_key, all_embeddings, embedder_params)
            if self.verbose:
                print(f"  ✓ Saved embeddings to cache for future use")
        except Exception as e:
            if self.verbose:
                print(f"  Warning: Could not save to cache ({e})")
        
        return all_embeddings
    
    def __call__(self, sequences, v_genes=None, j_genes=None):
        """Make embedder callable."""
        return self.embed(sequences, v_genes, j_genes)

