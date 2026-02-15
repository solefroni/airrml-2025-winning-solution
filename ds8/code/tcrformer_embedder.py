#!/usr/bin/env python3
"""
TCRformer Embedder for TCR Sequences
====================================

Loads the TCRformer model (fine-tuned ProtBERT) and generates embeddings for CDR3 sequences.
Uses the TCRformer model from https://github.com/InduKhatri/tcrformer
Based on the embedding configuration (ep=2) which includes V and J gene embeddings.
"""

import os
import torch
import numpy as np
from pathlib import Path
import sys
from typing import List, Union, Optional, Dict
from transformers import BertTokenizer, BertConfig
from tqdm import tqdm
import warnings
import re
import pandas as pd
from sklearn import preprocessing
warnings.filterwarnings('ignore')

# Import TCRformer model classes (optional; fallback classes defined below if import fails)
_tcrformer_bert = Path(__file__).parent / 'tcrformer' / 'models' / 'BERT'
if _tcrformer_bert.exists():
    sys.path.insert(0, str(_tcrformer_bert))
else:
    _tcrformer_bert = Path(__file__).parent.parent.parent / 'tcrformer' / 'models' / 'BERT'
    if _tcrformer_bert.exists():
        sys.path.insert(0, str(_tcrformer_bert))

try:
    from berttcr_ep2 import BertTCRModel, BertTCREmbeddings
except ImportError:
    # If direct import fails, we'll define the classes here
    from transformers import BertPreTrainedModel
    from transformers.models.bert.modeling_bert import BertEncoder, BertPooler
    from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
    from packaging import version
    
    class BertTCREmbeddings(torch.nn.Module):
        """TCRformer embeddings with V and J gene information."""
        def __init__(self, config):
            super().__init__()
            self.word_embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
            self.position_embeddings = torch.nn.Embedding(config.max_position_embeddings, config.hidden_size)
            self.token_type_embeddings = torch.nn.Embedding(config.type_vocab_size, config.hidden_size)
            self.v_gene_embeddings = torch.nn.Embedding(65, config.hidden_size, padding_idx=64)
            self.j_gene_embeddings = torch.nn.Embedding(15, config.hidden_size, padding_idx=14)
            self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
            self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
            if version.parse(torch.__version__) > version.parse("1.6.0"):
                self.register_buffer("token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False)

        def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, 
                   v_gene_ids=None, j_gene_ids=None, past_key_values_length=0):
            if input_ids is not None:
                input_shape = input_ids.size()
            else:
                input_shape = inputs_embeds.size()[:-1]
            seq_length = input_shape[1]
            if position_ids is None:
                position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
            if token_type_ids is None:
                if hasattr(self, "token_type_ids"):
                    buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                    buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                    token_type_ids = buffered_token_type_ids_expanded
                else:
                    token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            v_gene_embeddings = self.v_gene_embeddings(v_gene_ids)
            j_gene_embeddings = self.j_gene_embeddings(j_gene_ids)
            embeddings = inputs_embeds + v_gene_embeddings + j_gene_embeddings + token_type_embeddings
            if self.position_embedding_type == "absolute":
                position_embeddings = self.position_embeddings(position_ids)
                embeddings += position_embeddings
            embeddings = self.LayerNorm(embeddings)
            embeddings = self.dropout(embeddings)
            return embeddings
    
    class BertTCRModel(BertPreTrainedModel):
        """TCRformer model with V and J gene embeddings."""
        def __init__(self, config, add_pooling_layer=True):
            super().__init__(config)
            self.config = config
            self.embeddings = BertTCREmbeddings(config)
            self.encoder = BertEncoder(config)
            self.pooler = BertPooler(config) if add_pooling_layer else None
            self.post_init()
        
        def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                   head_mask=None, vgenes=None, jgenes=None, inputs_embeds=None, encoder_hidden_states=None,
                   encoder_attention_mask=None, past_key_values=None, use_cache=None, output_attentions=None,
                   output_hidden_states=None, return_dict=None):
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            use_cache = False
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
            elif input_ids is not None:
                input_shape = input_ids.size()
            elif inputs_embeds is not None:
                input_shape = inputs_embeds.size()[:-1]
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds")
            batch_size, seq_length = input_shape
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            if attention_mask is None:
                attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
            if token_type_ids is None:
                if hasattr(self.embeddings, "token_type_ids"):
                    buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                    buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                    token_type_ids = buffered_token_type_ids_expanded
                else:
                    token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
            extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)
            encoder_extended_attention_mask = None
            head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
            embedding_output = self.embeddings(
                input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                v_gene_ids=vgenes, j_gene_ids=jgenes, inputs_embeds=inputs_embeds,
                past_key_values_length=past_key_values_length,
            )
            encoder_outputs = self.encoder(
                embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_extended_attention_mask,
                past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions,
                output_hidden_states=output_hidden_states, return_dict=return_dict,
            )
            sequence_output = encoder_outputs[0]
            pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
            if not return_dict:
                return (sequence_output, pooled_output) + encoder_outputs[1:]
            return BaseModelOutputWithPoolingAndCrossAttentions(
                last_hidden_state=sequence_output, pooler_output=pooled_output,
                past_key_values=encoder_outputs.past_key_values, hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions, cross_attentions=encoder_outputs.cross_attentions,
            )


class TCRformerEmbedder:
    """Wrapper for TCRformer model to generate embeddings for CDR3 sequences."""
    
    # V and J gene categories (from TCRformer)
    V_GENE_CATEGORIES = ['NA','TRBV1', 'TRBV10-1', 'TRBV10-2', 'TRBV10-3', 'TRBV11-1', 'TRBV11-2', 'TRBV11-3', 
                        'TRBV12-1', 'TRBV12-2', 'TRBV12-3', 'TRBV12-4', 'TRBV12-5', 'TRBV13', 'TRBV13-1', 
                        'TRBV13-2', 'TRBV13-3', 'TRBV14', 'TRBV15', 'TRBV16', 'TRBV17', 'TRBV18', 'TRBV19', 
                        'TRBV2', 'TRBV20', 'TRBV20-1', 'TRBV23', 'TRBV24', 'TRBV24-1', 'TRBV25-1', 'TRBV26', 
                        'TRBV27', 'TRBV28', 'TRBV29', 'TRBV29-1', 'TRBV3', 'TRBV3-1', 'TRBV30', 'TRBV31', 
                        'TRBV4', 'TRBV4-1', 'TRBV4-2', 'TRBV4-3', 'TRBV5', 'TRBV5-1', 'TRBV5-4', 'TRBV5-5', 
                        'TRBV5-6', 'TRBV5-8', 'TRBV6-1', 'TRBV6-2', 'TRBV6-3', 'TRBV6-4', 'TRBV6-5', 'TRBV6-6', 
                        'TRBV6-9', 'TRBV7-2', 'TRBV7-3', 'TRBV7-4', 'TRBV7-6', 'TRBV7-7', 'TRBV7-8', 'TRBV7-9', 'TRBV9']
    
    J_GENE_CATEGORIES = ['NA','TRBJ1-1', 'TRBJ1-2', 'TRBJ1-3', 'TRBJ1-4', 'TRBJ1-5', 'TRBJ1-6', 'TRBJ2-1', 
                        'TRBJ2-2', 'TRBJ2-3', 'TRBJ2-4', 'TRBJ2-5', 'TRBJ2-6', 'TRBJ2-7']
    
    def __init__(self, model_path=None, base_model='Rostlab/prot_bert', device=None, 
                 batch_size=1024, max_len=40, verbose=True, use_fp16=False):
        """
        Initialize TCRformer embedder.
        
        Args:
            model_path: Path to fine-tuned TCRformer model directory. If None, uses base ProtBERT.
            base_model: Base model name (default: 'Rostlab/prot_bert')
            device: torch device ('cuda' or 'cpu'). If None, auto-detects.
            batch_size: Batch size for embedding generation
            max_len: Maximum CDR3 sequence length
            verbose: Print status messages.
            use_fp16: Use FP16 for faster inference (GPU only)
        """
        self.verbose = verbose
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.max_len = max_len
        self.use_fp16 = use_fp16 and (self.device == 'cuda')
        self.base_model = base_model
        
        if self.verbose:
            print(f"TCRformer Embedder: Using device {self.device}")
        
        # Initialize V/J gene encoders
        self._init_gene_encoders()
        
        # Try to find TCRformer model
        if model_path is None:
            possible_paths = [
                Path(os.environ.get('TCRFORMER_MODEL_PATH', '')),
                Path(__file__).parent / "tcrformer_model",
                Path(__file__).parent.parent.parent / "tcrformer" / "models" / "BERT" / "md2" / "model",
                Path.home() / "tcrformer" / "models" / "BERT" / "md2" / "model",
            ]
            possible_paths = [p for p in possible_paths if p and str(p)]
            for path in possible_paths:
                if path.exists() and ((path / "config.json").exists() or (path / "pytorch_model.bin").exists()):
                    model_path = path
                    break
        
        self.model_path = Path(model_path) if model_path else None
        self._load_model()
    
    def _init_gene_encoders(self):
        """Initialize V and J gene encoders."""
        cate = [self.V_GENE_CATEGORIES, self.J_GENE_CATEGORIES]
        # Use handle_unknown='use_encoded_value' and unknown_value=-1 to handle unknown categories
        # Then we'll map -1 to the padding index (64 for V, 14 for J)
        self.gene_encoder = preprocessing.OrdinalEncoder(
            categories=cate,
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        # Fit with dummy data to initialize
        dummy_df = pd.DataFrame({'V': ['NA'], 'J': ['NA']})
        self.gene_encoder.fit(dummy_df[["V", "J"]])
    
    def _normalize_v_gene(self, v_gene: str) -> str:
        """Normalize V gene name to match TCRformer format."""
        if pd.isna(v_gene) or v_gene == '' or v_gene == 'None':
            return 'NA'
        
        v_gene = str(v_gene).strip().upper()
        
        # Handle TCRBV format: TCRBV09-01 -> find matching TRBV category
        if v_gene.startswith('TCRBV'):
            # Extract base number: TCRBV09-01 -> 9
            match = re.search(r'TCRBV(\d+)', v_gene)
            if match:
                num = int(match.group(1))
                # Try to find exact match first (e.g., TRBV9)
                for cat in self.V_GENE_CATEGORIES:
                    if cat.startswith(f'TRBV{num}-') or cat == f'TRBV{num}':
                        return cat
                # If no exact match, return first variant (e.g., TRBV9 -> TRBV9-1 if exists)
                for cat in self.V_GENE_CATEGORIES:
                    if cat.startswith(f'TRBV{num}-'):
                        return cat
                # If still no match, try simplified name
                simplified = f'TRBV{num}'
                if simplified in self.V_GENE_CATEGORIES:
                    return simplified
        
        # Handle TRBV format
        elif v_gene.startswith('TRBV'):
            # Check if it's already in categories
            if v_gene in self.V_GENE_CATEGORIES:
                return v_gene
            # Try to find variant (e.g., TRBV6 -> TRBV6-1)
            match = re.search(r'TRBV(\d+)', v_gene)
            if match:
                num = int(match.group(1))
                # Find first variant
                for cat in self.V_GENE_CATEGORIES:
                    if cat.startswith(f'TRBV{num}-'):
                        return cat
                # Check if base name exists
                if f'TRBV{num}' in self.V_GENE_CATEGORIES:
                    return f'TRBV{num}'
        
        # If not recognized, return 'NA' (will use padding index)
        return 'NA'
    
    def _normalize_j_gene(self, j_gene: str) -> str:
        """Normalize J gene name to match TCRformer format."""
        if pd.isna(j_gene) or j_gene == '' or j_gene == 'None':
            return 'NA'
        j_gene = str(j_gene).strip().upper()
        if j_gene.startswith('TCRBJ'):
            # Extract: TCRBJ02-02 -> TRBJ2-2
            match = re.search(r'TCRBJ(\d+)-(\d+)', j_gene)
            if match:
                num1, num2 = match.groups()
                return f'TRBJ{num1}-{num2}'
        elif j_gene.startswith('TRBJ'):
            return j_gene
        for cat in self.J_GENE_CATEGORIES:
            if j_gene in cat or cat in j_gene:
                return cat
        return 'NA'
    
    def _load_model(self):
        """Load TCRformer model."""
        if self.verbose:
            if self.model_path:
                print(f"Loading TCRformer model from {self.model_path}")
            else:
                print(f"Loading base ProtBERT model: {self.base_model}")
        
        try:
            if self.model_path and self.model_path.exists():
                # Load fine-tuned model
                config = BertConfig.from_pretrained(str(self.model_path))
                self.model = BertTCRModel.from_pretrained(str(self.model_path), config=config)
            else:
                # Load base ProtBERT and convert to TCRformer architecture
                config = BertConfig.from_pretrained(self.base_model)
                self.model = BertTCRModel(config)
                # Load ProtBERT weights (except V/J gene embeddings which will be random)
                from transformers import BertModel
                base_bert = BertModel.from_pretrained(self.base_model)
                # Copy weights (this is a simplified approach)
                self.model.encoder.load_state_dict(base_bert.encoder.state_dict())
                self.model.embeddings.word_embeddings.load_state_dict(base_bert.embeddings.word_embeddings.state_dict())
                self.model.embeddings.position_embeddings.load_state_dict(base_bert.embeddings.position_embeddings.state_dict())
                self.model.embeddings.token_type_embeddings.load_state_dict(base_bert.embeddings.token_type_embeddings.state_dict())
                self.model.embeddings.LayerNorm.load_state_dict(base_bert.embeddings.LayerNorm.state_dict())
                if self.verbose:
                    print("  Using base ProtBERT (V/J gene embeddings will be random)")
            
            self.model.to(self.device)
            
            if self.use_fp16:
                self.model = self.model.half()
                if self.verbose:
                    print("✓ Using FP16 (half precision) for faster inference")
            
            self.model.eval()
            
            # Load tokenizer
            if self.model_path and (self.model_path / "vocab.txt").exists():
                self.tokenizer = BertTokenizer.from_pretrained(str(self.model_path), do_lower_case=False)
            else:
                self.tokenizer = BertTokenizer.from_pretrained(self.base_model, do_lower_case=False)
            
            if self.verbose:
                print("✓ TCRformer model loaded successfully")
        
        except Exception as e:
            raise RuntimeError(f"Failed to load TCRformer model: {e}")
    
    @staticmethod
    def insert_whitespace(seq: str) -> str:
        """Insert whitespace between amino acids for tokenization."""
        return " ".join(list(seq))
    
    def embed(self, sequences: Union[List[str], np.ndarray], 
              v_genes: Optional[Union[List[str], np.ndarray]] = None,
              j_genes: Optional[Union[List[str], np.ndarray]] = None,
              batch_size: int = None) -> np.ndarray:
        """
        Generate embeddings for CDR3 sequences.
        
        Args:
            sequences: List or array of CDR3 sequences (strings)
            v_genes: Optional list of V gene names (e.g., 'TCRBV09-01' or 'TRBV9')
            j_genes: Optional list of J gene names (e.g., 'TCRBJ02-02' or 'TRBJ2-2')
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
        
        # Normalize and encode V/J genes
        use_genes = (v_genes is not None) and (j_genes is not None)
        if use_genes:
            if isinstance(v_genes, np.ndarray):
                v_genes = v_genes.tolist()
            if isinstance(j_genes, np.ndarray):
                j_genes = j_genes.tolist()
            # Normalize gene names
            normalized_v = [self._normalize_v_gene(v) for v in v_genes]
            normalized_j = [self._normalize_j_gene(j) for j in j_genes]
            # Encode to integers
            gene_df = pd.DataFrame({'V': normalized_v, 'J': normalized_j})
            encoded_genes = self.gene_encoder.transform(gene_df[["V", "J"]])
            v_encoded = encoded_genes[:, 0].astype(int)
            j_encoded = encoded_genes[:, 1].astype(int)
            # Map unknown values (-1) to padding indices
            v_encoded[v_encoded == -1] = 64  # V gene padding index
            j_encoded[j_encoded == -1] = 14  # J gene padding index
        else:
            # Use 'NA' (index 0) for all sequences if genes not provided
            v_encoded = np.zeros(len(sequences), dtype=int)
            j_encoded = np.zeros(len(sequences), dtype=int)
        
        # Clean and prepare sequences
        cleaned_sequences = [seq.strip().upper() for seq in sequences]
        spaced_sequences = [self.insert_whitespace(seq) for seq in cleaned_sequences]
        
        # Accumulate embeddings on GPU
        gpu_embeddings = []
        
        # Progress tracking
        total_batches = (len(spaced_sequences) + batch_size - 1) // batch_size
        if self.verbose:
            print(f"  Processing {len(spaced_sequences):,} sequences in {total_batches:,} batches...")
        
        # Use tqdm for progress bar if verbose
        batch_range = range(0, len(spaced_sequences), batch_size)
        if self.verbose:
            batch_range = tqdm(batch_range, desc="  Embedding", unit="batch", 
                              total=total_batches, mininterval=10)  # Update every 10 seconds
        
        with torch.no_grad():
            for batch_idx, i in enumerate(batch_range, 1):
                batch_seqs = spaced_sequences[i:i+batch_size]
                batch_v = v_encoded[i:i+batch_size]
                batch_j = j_encoded[i:i+batch_size]
                
                # Tokenize sequences
                encoded = self.tokenizer(
                    batch_seqs,
                    padding="max_length",
                    max_length=self.max_len,
                    return_tensors="pt",
                    truncation=True
                )
                
                # Move to device
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                
                # Prepare V/J gene IDs for each token position
                # Format: [CLS] + [V_gene for each AA] + [SEP] + [padding with 64/14]
                batch_v_ids = []
                batch_j_ids = []
                for seq, v_idx, j_idx in zip(batch_seqs, batch_v, batch_j):
                    seq_len = len(seq.split()) + 2  # +2 for CLS and SEP
                    plength = len(seq.split())
                    v_ids = [0] + [int(v_idx)] * plength + [0] + [64] * (self.max_len - seq_len)
                    j_ids = [0] + [int(j_idx)] * plength + [0] + [14] * (self.max_len - seq_len)
                    batch_v_ids.append(v_ids[:self.max_len])
                    batch_j_ids.append(j_ids[:self.max_len])
                
                v_gene_tensor = torch.tensor(batch_v_ids, dtype=torch.long, device=self.device)
                j_gene_tensor = torch.tensor(batch_j_ids, dtype=torch.long, device=self.device)
                
                # Get model outputs
                if self.use_fp16:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            input_ids=encoded['input_ids'],
                            attention_mask=encoded['attention_mask'],
                            vgenes=v_gene_tensor,
                            jgenes=j_gene_tensor,
                            output_hidden_states=True
                        )
                else:
                    outputs = self.model(
                        input_ids=encoded['input_ids'],
                        attention_mask=encoded['attention_mask'],
                        vgenes=v_gene_tensor,
                        jgenes=j_gene_tensor,
                        output_hidden_states=True
                    )
                
                # Extract pooled embeddings (from pooler)
                if outputs.pooler_output is not None:
                    batch_embeddings = outputs.pooler_output
                else:
                    # Fallback: mean pooling over sequence tokens (excluding CLS and padding)
                    hidden_states = outputs.last_hidden_state
                    attention_mask = encoded['attention_mask']
                    seq_mask = attention_mask.clone()
                    seq_mask[:, 0] = 0  # Exclude CLS token
                    seq_mask = seq_mask.unsqueeze(-1)
                    masked_hidden = hidden_states * seq_mask
                    seq_lengths = seq_mask.squeeze(-1).sum(dim=1, keepdim=True)
                    seq_lengths = torch.clamp(seq_lengths, min=1.0)
                    batch_embeddings = masked_hidden.sum(dim=1) / seq_lengths
                
                gpu_embeddings.append(batch_embeddings)
        
        # Concatenate and transfer to CPU
        if gpu_embeddings:
            all_embeddings = torch.cat(gpu_embeddings, dim=0)
            return all_embeddings.cpu().numpy().astype(np.float32)
        else:
            return np.array([])

