#!/usr/bin/env python3
"""
CVC Embedder for TCR Sequences
==============================

Loads the CVC model and generates embeddings for CDR3 sequences.
Uses the CVC model from https://github.com/RomiGoldner/CVC
Based on the implementation in graph_sample_classification/utils/cvc_embedder.py
"""

import torch
import numpy as np
from pathlib import Path
import sys
from typing import List, Union
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class CVCEmbedder:
    """Wrapper for CVC model to generate embeddings for CDR3 sequences."""
    
    def __init__(self, model_path=None, device=None, batch_size=1024, max_len=64, verbose=True, use_fp16=False):
        """
        Initialize CVC embedder.
        
        Args:
            model_path: Path to CVC model directory. If None, tries to find it.
            device: torch device ('cuda' or 'cpu'). If None, auto-detects.
            batch_size: Batch size for embedding generation
            max_len: Maximum CDR3 sequence length
            verbose: Print status messages.
        """
        self.verbose = verbose
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.max_len = max_len
        self.use_fp16 = use_fp16 and (self.device == 'cuda')
        
        if self.verbose:
            print(f"CVC Embedder: Using device {self.device}")
        
        # Try to find CVC model
        if model_path is None:
            # Check environment variable first, then common locations
            env_cvc_path = os.environ.get('CVC_MODEL_PATH')
            possible_paths = []
            if env_cvc_path:
                possible_paths.append(Path(env_cvc_path))
            possible_paths.extend([
                Path.home() / "CVC",
                Path("./CVC"),
                Path("../CVC"),
            ])
            for path in possible_paths:
                if path.exists():
                    # Check if it's a valid model directory (has config.json or pytorch_model.bin)
                    if (path / "config.json").exists() or (path / "pytorch_model.bin").exists():
                        model_path = path
                        break
        
        if model_path is None:
            raise FileNotFoundError(
                f"CVC model not found. Please download the model first using: "
                f"python -m scripts.download_cvc --model_type CVC"
            )
        
        self.model_path = Path(model_path)
        self._load_model()
    
    def _load_model(self):
        """Load CVC model using transformers library."""
        if self.verbose:
            print(f"Loading CVC model from {self.model_path}")
        
        try:
            # Try to use safetensors first (bypasses PyTorch 2.6+ requirement)
            # If not available, fallback to regular format
            # The error message states: "This version restriction does not apply when loading files with safetensors"
            try:
                self.model = BertModel.from_pretrained(
                    str(self.model_path),
                    add_pooling_layer=False,
                    output_hidden_states=True,
                    use_safetensors=True  # Try safetensors first
                )
            except (OSError, ValueError) as e:
                # If safetensors not available, try regular format
                # This may fail with PyTorch < 2.6, but we'll handle that error
                if "safetensors" in str(e).lower():
                    if self.verbose:
                        print("  Safetensors not found, trying regular format...")
                    self.model = BertModel.from_pretrained(
                        str(self.model_path),
                        add_pooling_layer=False,
                        output_hidden_states=True,
                        use_safetensors=False  # Fallback to regular format
                    )
                else:
                    raise
            self.model.to(self.device)
            
            # Use mixed precision (FP16) for faster inference on GPU
            if self.use_fp16:
                self.model = self.model.half()  # Convert to FP16
                if self.verbose:
                    print("✓ Using FP16 (half precision) for faster inference")
            
            self.model.eval()
            
            # Compile model for faster inference (PyTorch 2.0+)
            # This can provide 1.5-2x speedup on GPU
            # Must be done after model is in eval mode and on device
            try:
                if hasattr(torch, 'compile') and self.device == 'cuda':
                    self.model = torch.compile(self.model, mode='reduce-overhead')
                    if self.verbose:
                        print("✓ Model compiled with torch.compile() for faster inference")
            except Exception as e:
                if self.verbose:
                    print(f"  Note: torch.compile() not available or failed: {e}")
            
            # Load tokenizer
            self.tokenizer = BertTokenizer.from_pretrained(str(self.model_path))
            
            if self.verbose:
                print("✓ CVC model loaded successfully")
        
        except Exception as e:
            raise RuntimeError(f"Failed to load CVC model: {e}")
    
    @staticmethod
    def insert_whitespace(seq: str) -> str:
        """Insert whitespace between amino acids for tokenization."""
        return " ".join(list(seq))
    
    def embed(self, sequences: Union[List[str], np.ndarray], batch_size: int = None) -> np.ndarray:
        """
        Generate embeddings for CDR3 sequences.
        
        Args:
            sequences: List or array of CDR3 sequences (strings)
            batch_size: Batch size for processing (uses self.batch_size if None)
        
        Returns:
            numpy array of shape (n_sequences, embedding_dim)
        """
        if isinstance(sequences, np.ndarray):
            sequences = sequences.tolist()
        
        if len(sequences) == 0:
            return np.array([])
        
        if batch_size is None:
            batch_size = self.batch_size
        
        # Clean and prepare sequences with whitespace
        cleaned_sequences = [seq.strip().upper() for seq in sequences]
        spaced_sequences = [self.insert_whitespace(seq) for seq in cleaned_sequences]
        
        # Accumulate embeddings on GPU first, then transfer to CPU once
        gpu_embeddings = []
        
        # Use inference_mode() instead of no_grad() for slightly better performance
        with torch.inference_mode():
            for i in range(0, len(spaced_sequences), batch_size):
                batch = spaced_sequences[i:i+batch_size]
                
                # Tokenize sequences
                encoded = self.tokenizer(
                    batch,
                    padding="max_length",
                    max_length=self.max_len,
                    return_tensors="pt",
                    truncation=True
                )
                
                # Move to device
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                
                # Get model outputs with autocast if using FP16
                if self.use_fp16:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**encoded, output_hidden_states=True)
                else:
                    outputs = self.model(**encoded, output_hidden_states=True)
                
                # Extract embeddings from last layer (mean pooling over sequence tokens)
                # Optimized batch processing for efficiency
                hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_dim]
                
                # Get attention mask to find actual sequence lengths (exclude CLS token)
                attention_mask = encoded['attention_mask']  # [batch_size, seq_len]
                # Create mask excluding CLS token (position 0) and padding
                seq_mask = attention_mask.clone()
                seq_mask[:, 0] = 0  # Exclude CLS token
                seq_mask = seq_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
                
                # Batch mean pooling: sum over sequence tokens, divide by actual length
                # Skip CLS token by masking it out
                masked_hidden = hidden_states * seq_mask  # [batch_size, seq_len, hidden_dim]
                seq_lengths = seq_mask.squeeze(-1).sum(dim=1, keepdim=True)  # [batch_size, 1]
                # Avoid division by zero
                seq_lengths = torch.clamp(seq_lengths, min=1.0)
                batch_embeddings = masked_hidden.sum(dim=1) / seq_lengths  # [batch_size, hidden_dim]
                
                # Keep on GPU for now (accumulate)
                gpu_embeddings.append(batch_embeddings)
        
        # Concatenate all batches on GPU, then transfer to CPU once
        if gpu_embeddings:
            all_embeddings = torch.cat(gpu_embeddings, dim=0)  # Concatenate on GPU
            # Single CPU transfer at the end
            return all_embeddings.cpu().numpy().astype(np.float32)
        else:
            return np.array([])

