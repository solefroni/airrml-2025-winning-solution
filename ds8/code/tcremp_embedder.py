#!/usr/bin/env python3
"""
TCRemP Embedder for TCR Sequences
==================================

Uses TCRemP (T-cell receptor sequence embedding via prototypes) to generate embeddings.
TCRemP computes distances to prototype TCR sequences to create embeddings.
Based on: https://github.com/antigenomics/tcremp
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Union, Optional, Dict
from tqdm import tqdm
import warnings
import tempfile
import shutil
warnings.filterwarnings('ignore')

# Global flag for TCRemP availability
TCREMP_AVAILABLE = False
IMPORT_ERROR = None

# Add TCRemP to path (try multiple locations for submission vs. development)
for _base in [Path(__file__).parent.parent.parent, Path(__file__).parent.parent]:
    _tcremp = _base / 'tcremp'
    if _tcremp.exists():
        sys.path.insert(0, str(_tcremp))
        break
if os.environ.get('TCREMP_PATH'):
    sys.path.insert(0, os.environ.get('TCREMP_PATH'))

try:
    from tcremp.tcremp_pipeline import TcrempPipeline
    TCREMP_AVAILABLE = True
except ImportError as e:
    TCREMP_AVAILABLE = False
    IMPORT_ERROR = str(e)
    print(f"Warning: TCRemP not available: {e}")
    print("Note: TCRemP requires Python 3.11+ and the 'mirpy' package.")
    print("Please install in a Python 3.11 environment or use a different embedder.")


class TCRemPEmbedder:
    """
    Embedder using TCRemP (prototype-based TCR embedding).
    
    TCRemP computes distances from input TCR sequences to a set of prototype sequences.
    The embedding is a vector of distances to prototypes for V, J, and CDR3 regions.
    """
    
    def __init__(
        self,
        device=None,
        batch_size=1024,
        use_fp16=False,
        verbose=True,
        n_prototypes=3000,
        species='HomoSapiens',
        chain='TRB',
        nproc=1,
        temp_dir=None
    ):
        """
        Initialize TCRemP embedder.
        
        Args:
            device: Not used (TCRemP doesn't use GPU directly)
            batch_size: Not used (TCRemP processes all at once)
            use_fp16: Not used
            verbose: Print progress messages
            n_prototypes: Number of prototypes to use (default 3000)
            species: Species for prototypes ('HomoSapiens', 'MusMusculus', 'MacacaMulatta')
            chain: Chain type ('TRA', 'TRB', or 'TRA_TRB')
            nproc: Number of processes for distance computation
            temp_dir: Temporary directory for TCRemP output (auto-created if None)
        """
        if not TCREMP_AVAILABLE:
            error_msg = (
                "TCRemP is not available. "
                "TCRemP requires Python 3.11+ and the 'mirpy' package.\n"
                f"Current Python version: {sys.version_info.major}.{sys.version_info.minor}\n"
                f"Import error: {IMPORT_ERROR if 'IMPORT_ERROR' in globals() else 'Unknown'}\n"
                "Please either:\n"
                "  1. Use a Python 3.11+ environment with TCRemP installed\n"
                "  2. Use a different embedder (CVC or TCRformer)"
            )
            raise ImportError(error_msg)
        
        self.verbose = verbose
        self.n_prototypes = n_prototypes
        self.species = species
        self.chain = chain
        self.nproc = nproc
        self.temp_dir = temp_dir
        self.embedding_dim = None  # Will be determined after first embedding
        
        if self.verbose:
            print(f"TCRemP Embedder initialized:")
            print(f"  - Prototypes: {n_prototypes}")
            print(f"  - Species: {species}")
            print(f"  - Chain: {chain}")
            print(f"  - Processes: {nproc}")
    
    def _prepare_input_dataframe(
        self,
        sequences: List[str],
        v_genes: Optional[List[str]] = None,
        j_genes: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Prepare input dataframe for TCRemP.
        
        Args:
            sequences: List of CDR3 amino acid sequences
            v_genes: Optional list of V gene names
            j_genes: Optional list of J gene names
            
        Returns:
            DataFrame with columns required by TCRemP
        """
        data = {
            'junction_aa': sequences,
        }
        
        if self.chain == 'TRB':
            data['b_cdr3aa'] = sequences
            if v_genes is not None:
                data['b_v'] = v_genes
            else:
                data['b_v'] = None
            if j_genes is not None:
                data['b_j'] = j_genes
            else:
                data['b_j'] = None
        elif self.chain == 'TRA':
            data['a_cdr3aa'] = sequences
            if v_genes is not None:
                data['a_v'] = v_genes
            else:
                data['a_v'] = None
            if j_genes is not None:
                data['a_j'] = j_genes
            else:
                data['a_j'] = None
        else:
            raise ValueError(f"Chain {self.chain} not yet supported. Use 'TRA' or 'TRB'")
        
        df = pd.DataFrame(data)
        
        # Normalize V/J gene names (remove allele info if present)
        if self.chain == 'TRB':
            if 'b_v' in df.columns:
                df['b_v'] = df['b_v'].astype(str).str.split('*').str[0].str.replace('/', '')
            if 'b_j' in df.columns:
                df['b_j'] = df['b_j'].astype(str).str.split('*').str[0].str.replace('/', '')
        elif self.chain == 'TRA':
            if 'a_v' in df.columns:
                df['a_v'] = df['a_v'].astype(str).str.split('*').str[0].str.replace('/', '')
            if 'a_j' in df.columns:
                df['a_j'] = df['a_j'].astype(str).str.split('*').str[0].str.replace('/', '')
        
        return df
    
    def embed(
        self,
        sequences: Union[List[str], np.ndarray],
        v_genes: Optional[Union[List[str], np.ndarray]] = None,
        j_genes: Optional[Union[List[str], np.ndarray]] = None,
        batch_size: int = None
    ) -> np.ndarray:
        """
        Generate embeddings for TCR sequences using TCRemP.
        
        Args:
            sequences: List of CDR3 amino acid sequences
            v_genes: Optional list of V gene names
            j_genes: Optional list of J gene names
            batch_size: Not used (kept for API compatibility)
            
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
        
        if self.verbose:
            print(f"Generating TCRemP embeddings for {len(sequences):,} sequences...")
        
        # Prepare input dataframe
        input_df = self._prepare_input_dataframe(sequences, v_genes, j_genes)
        
        # Create temporary directory for TCRemP output
        if self.temp_dir is None:
            temp_dir = tempfile.mkdtemp(prefix='tcremp_embed_')
            cleanup_temp = True
        else:
            temp_dir = self.temp_dir
            cleanup_temp = False
        
        try:
            # Get the combined prototype file path
            from tcremp import get_resource_path
            combined_prototype_path = get_resource_path('tcremp_prototypes_olga.tsv')
            
            # Initialize TCRemP pipeline with the combined prototype file
            # TCRemP will split it into TRA and TRB automatically
            pipeline = TcrempPipeline(
                run_name=temp_dir,
                input_data=input_df,
                clonotype_index=None,
                prototypes_path=combined_prototype_path,  # Use combined file
                n=self.n_prototypes,
                species=self.species,
                prototypes_chain=self.chain,
                random_seed=42
            )
            
            # Extract clonotypes
            if self.verbose:
                print("  Extracting clonotypes...")
            pipeline.tcremp_clonotypes(chain=self.chain, unique_clonotypes=False)
            
            # Compute distances
            if self.verbose:
                print(f"  Computing distances to {self.n_prototypes} prototypes...")
            pipeline.tcremp_dists_count(chain=self.chain, nproc=self.nproc, chunk_sz=100)
            
            # Process distances
            if self.verbose:
                print("  Processing distance results...")
            pipeline.tcremp_dists(chain=self.chain)
            
            # Extract embeddings from distance dataframe
            # TCRemP stores distances in columns like: b_0_v, b_0_j, b_0_cdr3, b_1_v, ...
            dist_df = pipeline.dists[self.chain]
            
            # Get distance columns (all columns except metadata)
            metadata_cols = ['cloneId', 'cdr3aa', 'v', 'j', 'chain']
            if self.chain == 'TRB':
                dist_cols = [col for col in dist_df.columns if col.startswith('b_') and col not in metadata_cols]
            elif self.chain == 'TRA':
                dist_cols = [col for col in dist_df.columns if col.startswith('a_') and col not in metadata_cols]
            else:
                dist_cols = [col for col in dist_df.columns if col not in metadata_cols]
            
            # Sort columns to ensure consistent ordering
            dist_cols = sorted(dist_cols)
            
            # Extract embeddings
            embeddings = dist_df[dist_cols].values.astype(np.float32)
            
            # Set embedding dimension
            if self.embedding_dim is None:
                self.embedding_dim = embeddings.shape[1]
                if self.verbose:
                    print(f"  Embedding dimension: {self.embedding_dim}")
            
            # Map back to original sequences (TCRemP may deduplicate)
            # We need to map from clonotypes back to original sequences
            if len(embeddings) != len(sequences):
                if self.verbose:
                    print(f"  Warning: TCRemP deduplicated sequences ({len(sequences)} -> {len(embeddings)})")
                    print(f"  Mapping embeddings back to original sequences...")
                
                # Create mapping from sequences to embeddings
                # TCRemP uses cloneId to group identical clonotypes
                # We need to map each input sequence to its cloneId
                annot_df = pipeline.annot_input[self.chain]
                
                # Create sequence-to-cloneId mapping
                if self.chain == 'TRB':
                    seq_col = 'b_cdr3aa'
                    v_col = 'b_v'
                    j_col = 'b_j'
                else:
                    seq_col = 'a_cdr3aa'
                    v_col = 'a_v'
                    j_col = 'a_j'
                
                # Map input sequences to cloneIds
                input_to_clone = {}
                for idx, row in annot_df.iterrows():
                    seq = row[seq_col]
                    clone_id = row.get('cloneId', idx)
                    if seq not in input_to_clone:
                        input_to_clone[seq] = clone_id
                
                # Map cloneIds to embeddings
                clone_to_embedding = {}
                for idx, row in dist_df.iterrows():
                    clone_id = row['cloneId']
                    clone_to_embedding[clone_id] = embeddings[idx]
                
                # Map original sequences to embeddings
                result_embeddings = []
                for seq in sequences:
                    clone_id = input_to_clone.get(seq, None)
                    if clone_id is not None and clone_id in clone_to_embedding:
                        result_embeddings.append(clone_to_embedding[clone_id])
                    else:
                        # Fallback: use mean embedding or zero embedding
                        if len(result_embeddings) > 0:
                            result_embeddings.append(np.zeros_like(result_embeddings[0]))
                        else:
                            result_embeddings.append(np.zeros(embeddings.shape[1], dtype=np.float32))
                
                embeddings = np.array(result_embeddings)
            
            if self.verbose:
                print(f"  Generated embeddings: {embeddings.shape}")
            
            return embeddings
            
        finally:
            # Cleanup temporary directory
            if cleanup_temp and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def __call__(self, sequences, v_genes=None, j_genes=None):
        """Make embedder callable."""
        return self.embed(sequences, v_genes, j_genes)

