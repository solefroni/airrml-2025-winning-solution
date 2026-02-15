#!/usr/bin/env python3
"""
Graph-Based T1D Classification
==============================

Classifies T1D samples using graph neural networks on TCR repertoire graphs.
Each sample is represented as a graph built using KNN in CVC embedding space.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import BatchNorm
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    precision_recall_curve, average_precision_score
)
from tqdm import tqdm
import json
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from cvc_embedder import CVCEmbedder
from graph_builder import GraphBuilder
from cache_utils import (
    load_cached_embeddings, save_cached_embeddings,
    load_cached_graph, save_cached_graph,
    get_sequence_hash, get_graph_cache_key,
    set_cache_dir
)

# Paths
# =============================================================================
# CONFIGURATION - Update these paths for your environment
# =============================================================================
import os
CODE_DIR = Path(__file__).parent.resolve()
DATASET_NUM = 8
DATA_DIR = Path(os.environ.get('DS8_TRAIN_DATA', '../input'))
DOWNSAMPLE_CACHE_DIR = Path(os.environ.get('DS8_DOWNSAMPLE_CACHE', '../cache/downsample'))
GRAPH_CACHE_DIR = Path(os.environ.get('DS8_GRAPH_CACHE', '../cache/graphs'))
RESULTS_DIR = CODE_DIR.parent / 'output' / 'results'
LOGS_DIR = CODE_DIR.parent / 'output' / 'logs'

# Create directories
GRAPH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
(GRAPH_CACHE_DIR / "embeddings").mkdir(parents=True, exist_ok=True)
(GRAPH_CACHE_DIR / "graphs").mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
# No test split - only train/val split (test will be evaluated separately later)
VAL_SIZE = 0.2  # 20% of data for validation
RANDOM_STATE = 42
K_VALUES = [30]  # Focus on k=30 to reproduce 0.8109 results with fixed random seeds
BATCH_SIZE = 8
NUM_EPOCHS = 300  # Increased for extended training
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 75  # Increased patience for better convergence
# Require CUDA - no CPU fallback
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required but not available. Please run on a GPU node.")
DEVICE = torch.device('cuda')

# CVC only for DS8 (GCN k=30)
EMBEDDER_TYPE = 'cvc'


def log_progress(message, level="INFO"):
    """Log progress with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}", flush=True)


# Set all random seeds for reproducibility
import random
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_STATE)
    torch.cuda.manual_seed_all(RANDOM_STATE)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Graph Neural Network Models
class GCNClassifier(nn.Module):
    """Graph Convolutional Network for graph classification."""
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, num_classes=2, dropout=0.5):
        super(GCNClassifier, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(BatchNorm(hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(BatchNorm(hidden_dim))
        
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.bn_final = BatchNorm(hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.dropout = dropout
    
    def forward(self, x, edge_index, batch):
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        x = self.bn_final(x)
        x = F.relu(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification
        x = self.classifier(x)
        return x


class GINClassifier(nn.Module):
    """Graph Isomorphism Network for graph classification."""
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, num_classes=2, dropout=0.5):
        super(GINClassifier, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First layer
        nn1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINConv(nn1, train_eps=True))
        self.bns.append(BatchNorm(hidden_dim))
        
        # Middle layers
        for _ in range(num_layers - 2):
            nn_mid = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(nn_mid, train_eps=True))
            self.bns.append(BatchNorm(hidden_dim))
        
        # Final layer
        nn_final = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINConv(nn_final, train_eps=True))
        self.bn_final = BatchNorm(hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.dropout = dropout
    
    def forward(self, x, edge_index, batch):
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        x = self.bn_final(x)
        x = F.relu(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification
        x = self.classifier(x)
        return x


class GraphSAGEClassifier(nn.Module):
    """GraphSAGE for graph classification."""
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, num_classes=2, dropout=0.5):
        super(GraphSAGEClassifier, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        self.bns.append(BatchNorm(hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.bns.append(BatchNorm(hidden_dim))
        
        self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.bn_final = BatchNorm(hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.dropout = dropout
    
    def forward(self, x, edge_index, batch):
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        x = self.bn_final(x)
        x = F.relu(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification
        x = self.classifier(x)
        return x


class GATClassifier(nn.Module):
    """Graph Attention Network for graph classification."""
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, num_classes=2, dropout=0.5, heads=4):
        super(GATClassifier, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout))
        self.bns.append(BatchNorm(hidden_dim * heads))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
            self.bns.append(BatchNorm(hidden_dim * heads))
        
        self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout))
        self.bn_final = BatchNorm(hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.dropout = dropout
    
    def forward(self, x, edge_index, batch):
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        x = self.bn_final(x)
        x = F.relu(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification
        x = self.classifier(x)
        return x


def load_metadata():
    """Load metadata."""
    log_progress("Loading metadata...")
    metadata_path = DATA_DIR / "metadata.csv"
    df = pd.read_csv(metadata_path)
    log_progress(f"Loaded {len(df)} samples from metadata")
    log_progress(f"Positive samples: {df['label_positive'].sum()}, Negative: {(~df['label_positive']).sum()}")
    return df


def load_repertoire_from_cache(rep_id, cache_dir):
    """Load downsampled repertoire from cache."""
    cache_file = cache_dir / f"{rep_id}.pkl"
    if not cache_file.exists():
        return None
    
    with open(cache_file, 'rb') as f:
        df = pickle.load(f)
    
    return df


# Caching functions are now imported from cache_utils


def build_graph_dataset(sample_ids, metadata_df, cache_dir, graph_builder, k_value):
    """Build graphs for all samples with batched embedding generation for efficiency."""
    import time
    phase_start = time.time()
    log_progress(f"Building graphs for {len(sample_ids)} samples (k={k_value})...")
    
    graphs = []
    labels = []
    cached_count = 0
    new_count = 0
    
    # First pass: collect all data and check cache
    samples_to_process = []
    sample_data = {}
    
    for rep_id in sample_ids:
        # Load repertoire from cache
        df = load_repertoire_from_cache(rep_id, cache_dir)
        if df is None or len(df) == 0:
            continue
        
        # Get sequences and counts
        sequences = df['junction_aa'].values.tolist()
        counts = df['templates'].values.astype(np.float32)
        
        # Get V/J genes if available
        v_genes = None
        j_genes = None
        if 'v_call' in df.columns and 'j_call' in df.columns:
            v_genes = df['v_call'].values.tolist()
            j_genes = df['j_call'].values.tolist()
        
        # Try to load from cache first
        cached_graph = load_cached_graph(rep_id, k_value, sequences, embedder_type=EMBEDDER_TYPE)
        
        if cached_graph is not None:
            graphs.append(cached_graph)
            cached_count += 1
        else:
            # Store for batch processing
            samples_to_process.append(rep_id)
            sample_data[rep_id] = {
                'sequences': sequences,
                'counts': counts,
                'v_genes': v_genes,
                'j_genes': j_genes
            }
        
        # Get label (store for all samples)
        label = metadata_df[metadata_df['repertoire_id'] == rep_id]['label_positive'].values[0]
        labels.append(int(label))
    
    # Second pass: check cache per-sample, then batch generate embeddings for uncached samples
    if samples_to_process:
        embedding_start = time.time()
        
        # First, check cache for each sample individually
        from cache_utils import load_cached_embeddings
        
        embedder_params = None
        cached_embeddings_dict = {}  # rep_id -> embeddings
        samples_to_embed = []
        
        log_progress(f"Checking cache for {len(samples_to_process)} samples...")
        for rep_id in samples_to_process:
            data = sample_data[rep_id]
            cache_key = data['sequences']
            if data['v_genes'] is not None and data['j_genes'] is not None:
                cache_key = (data['sequences'], tuple(data['v_genes']), tuple(data['j_genes']))
            
            cached_emb = load_cached_embeddings(cache_key, embedder_params)
            if cached_emb is not None:
                cached_embeddings_dict[rep_id] = cached_emb
            else:
                samples_to_embed.append(rep_id)
        
        if cached_embeddings_dict:
            log_progress(f"  Found {len(cached_embeddings_dict)} samples in cache")
        if samples_to_embed:
            log_progress(f"  Need to compute embeddings for {len(samples_to_embed)} samples")
        
        # INCREMENTAL CACHING: Process samples in smaller batches and save after each batch
        sample_batch_size = 100
        
        embedding_compute_start = time.time()
        all_computed_embeddings = {}  # rep_id -> embeddings (for samples we compute)
        
        if samples_to_embed:
            n_sample_batches = (len(samples_to_embed) + sample_batch_size - 1) // sample_batch_size
            log_progress(f"Processing {len(samples_to_embed)} samples in {n_sample_batches} batches of ~{sample_batch_size} samples each (incremental caching enabled)...")
            
            for batch_idx in range(n_sample_batches):
                batch_start = batch_idx * sample_batch_size
                batch_end = min((batch_idx + 1) * sample_batch_size, len(samples_to_embed))
                batch_samples = samples_to_embed[batch_start:batch_end]
                
                log_progress(f"  Batch {batch_idx+1}/{n_sample_batches}: processing {len(batch_samples)} samples...")
                
                # Collect sequences for this batch
                batch_sequences = []
                batch_v_genes = []
                batch_j_genes = []
                batch_sample_boundaries = []  # (start_idx, end_idx, rep_id) for each sample in batch
                current_idx = 0
                
                for rep_id in batch_samples:
                    data = sample_data[rep_id]
                    seqs = data['sequences']
                    batch_sequences.extend(seqs)
                    
                    if data['v_genes'] is not None and data['j_genes'] is not None:
                        batch_v_genes.extend(data['v_genes'])
                        batch_j_genes.extend(data['j_genes'])
                    else:
                        batch_v_genes.extend([None] * len(seqs))
                        batch_j_genes.extend([None] * len(seqs))
                    
                    batch_sample_boundaries.append((current_idx, current_idx + len(seqs), rep_id))
                    current_idx += len(seqs)
                
                # Generate embeddings for this batch
                if batch_v_genes and batch_j_genes and any(v is not None for v in batch_v_genes):
                    has_all_genes = all(v is not None and j is not None for v, j in zip(batch_v_genes, batch_j_genes))
                    if has_all_genes:
                        try:
                            batch_embeddings = graph_builder.embedder.embed(
                                batch_sequences, 
                                v_genes=batch_v_genes, 
                                j_genes=batch_j_genes
                            )
                        except TypeError:
                            # Fallback if embedder doesn't support V/J genes (e.g., CVC)
                            batch_embeddings = graph_builder.embedder.embed(batch_sequences)
                    else:
                        # Mixed: some have V/J genes, some don't
                        valid_indices = [i for i, (v, j) in enumerate(zip(batch_v_genes, batch_j_genes)) if v is not None and j is not None]
                        seqs_with_genes = [batch_sequences[i] for i in valid_indices]
                        v_with_genes = [batch_v_genes[i] for i in valid_indices]
                        j_with_genes = [batch_j_genes[i] for i in valid_indices]
                        seqs_without_genes = [batch_sequences[i] for i in range(len(batch_sequences)) if i not in valid_indices]
                        
                        try:
                            embeddings_with_genes = graph_builder.embedder.embed(
                                seqs_with_genes, v_genes=v_with_genes, j_genes=j_with_genes
                            ) if seqs_with_genes else np.array([])
                        except TypeError:
                            # Fallback if embedder doesn't support V/J genes (e.g., CVC)
                            embeddings_with_genes = graph_builder.embedder.embed(seqs_with_genes) if seqs_with_genes else np.array([])
                        embeddings_without_genes = graph_builder.embedder.embed(
                            seqs_without_genes
                        ) if seqs_without_genes else np.array([])
                        
                        embedding_dim = embeddings_with_genes.shape[1] if len(embeddings_with_genes) > 0 else embeddings_without_genes.shape[1]
                        batch_embeddings = np.zeros((len(batch_sequences), embedding_dim), dtype=np.float32)
                        if len(embeddings_with_genes) > 0:
                            batch_embeddings[valid_indices] = embeddings_with_genes
                        if len(embeddings_without_genes) > 0:
                            without_indices = [i for i in range(len(batch_sequences)) if i not in valid_indices]
                            batch_embeddings[without_indices] = embeddings_without_genes
                else:
                    batch_embeddings = graph_builder.embedder.embed(batch_sequences)
                
                # Split embeddings by sample and save to cache immediately
                from cache_utils import save_cached_embeddings
                for i, rep_id in enumerate(batch_samples):
                    start_idx, end_idx, _ = batch_sample_boundaries[i]
                    sample_embeddings = batch_embeddings[start_idx:end_idx]
                    all_computed_embeddings[rep_id] = sample_embeddings
                    
                    # Save to cache immediately (incremental caching)
                    data = sample_data[rep_id]
                    cache_key = data['sequences']
                    if data['v_genes'] is not None and data['j_genes'] is not None:
                        cache_key = (data['sequences'], tuple(data['v_genes']), tuple(data['j_genes']))
                    
                    save_cached_embeddings(cache_key, sample_embeddings, None)
                
                log_progress(f"    ✓ Saved {len(batch_samples)} samples to cache")
        
        embedding_compute_time = time.time() - embedding_compute_start
        if samples_to_embed:
            total_sequences = sum(len(sample_data[rep_id]['sequences']) for rep_id in samples_to_embed)
            log_progress(f"Generated embeddings for {len(samples_to_embed)} samples ({total_sequences:,} sequences) in {embedding_compute_time:.1f} seconds ({embedding_compute_time/60:.2f} minutes)")
            log_progress(f"  Embedding rate: {total_sequences/embedding_compute_time:.0f} sequences/second")
        
        log_progress(f"Building graphs from embeddings...")
        
        # Combine cached and newly computed embeddings, then build graphs
        graph_build_start = time.time()
        for rep_id in samples_to_process:
            data = sample_data[rep_id]
            
            # Get embeddings (from cache or newly computed)
            if rep_id in cached_embeddings_dict:
                sample_embeddings = cached_embeddings_dict[rep_id]
            elif rep_id in all_computed_embeddings:
                sample_embeddings = all_computed_embeddings[rep_id]
            else:
                log_progress(f"Warning: No embeddings found for {rep_id}", "ERROR")
                continue
            
            # Build graph using pre-computed embeddings
            try:
                graph = graph_builder.build_graph_from_embeddings(
                    data['sequences'], 
                    data['counts'], 
                    sample_embeddings
                )
                graph = graph_builder.add_centrality_features(graph)
                # Save to cache IMMEDIATELY (incremental caching for job recovery)
                save_cached_graph(rep_id, k_value, data['sequences'], graph, embedder_type=EMBEDDER_TYPE)
                graphs.append(graph)
                new_count += 1
                # Log every 10 graphs to show progress
                if new_count % 10 == 0:
                    log_progress(f"  Saved {new_count} new graphs to cache (recovery enabled)")
            except Exception as e:
                log_progress(f"Error building graph for {rep_id}: {e}", "ERROR")
                continue
        
        graph_build_time = time.time() - graph_build_start
        embedding_total_time = time.time() - embedding_start
        log_progress(f"Graph building completed in {graph_build_time:.1f} seconds ({graph_build_time/60:.2f} minutes)")
        log_progress(f"  Graph building rate: {new_count/graph_build_time:.2f} graphs/second")
        log_progress(f"Total embedding phase: {embedding_total_time:.1f} seconds ({embedding_total_time/60:.2f} minutes)")
    
    phase_time = time.time() - phase_start
    log_progress(f"Built {len(graphs)} graphs (cached: {cached_count}, new: {new_count}) in {phase_time:.1f} seconds ({phase_time/60:.2f} minutes)")
    return graphs, np.array(labels)


def train_model(model, train_loader, val_loader, device, num_epochs, patience):
    """Train a graph classification model."""
    import time
    training_start = time.time()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # Add learning rate scheduler to reduce LR when validation ROC-AUC plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=20, min_lr=1e-6
    )
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    best_val_roc_auc = 0.0
    patience_counter = 0
    best_model_state = None
    epoch_times = []
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        # Training
        model.train()
        train_loss = 0.0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation with detailed metrics
        model.eval()
        val_all_probs = []
        val_all_preds = []
        val_all_labels = []
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.batch)
                probs = F.softmax(out, dim=1)
                pred = out.argmax(dim=1)
                
                val_all_probs.extend(probs[:, 1].cpu().numpy())
                val_all_preds.extend(pred.cpu().numpy())
                val_all_labels.extend(data.y.cpu().numpy())
        
        val_all_probs = np.array(val_all_probs)
        val_all_preds = np.array(val_all_preds)
        val_all_labels = np.array(val_all_labels)
        
        # Calculate metrics
        val_acc = accuracy_score(val_all_labels, val_all_preds)
        val_balanced_acc = balanced_accuracy_score(val_all_labels, val_all_preds)
        
        # ROC-AUC and PR-AUC
        if len(np.unique(val_all_labels)) > 1:
            val_roc_auc = roc_auc_score(val_all_labels, val_all_probs)
            val_pr_auc = average_precision_score(val_all_labels, val_all_probs)
        else:
            val_roc_auc = 0.5
            val_pr_auc = 0.0
        
        # Precision, Recall, F1
        try:
            val_precision = precision_score(val_all_labels, val_all_preds, zero_division=0)
            val_recall = recall_score(val_all_labels, val_all_preds, zero_division=0)
            val_f1 = f1_score(val_all_labels, val_all_preds, zero_division=0)
        except:
            val_precision = 0.0
            val_recall = 0.0
            val_f1 = 0.0
        
        # Use ROC-AUC as primary metric for model selection (better for imbalanced data)
        # This is the metric we optimize for
        if val_roc_auc > best_val_roc_auc:
            best_val_roc_auc = val_roc_auc
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            log_progress(f"  → New best ROC-AUC: {best_val_roc_auc:.4f} (saving model)")
        else:
            patience_counter += 1
        
        # Update learning rate scheduler based on validation ROC-AUC
        scheduler.step(val_roc_auc)
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_epoch_time = np.mean(epoch_times[-10:]) if len(epoch_times) >= 10 else np.mean(epoch_times)
        remaining_epochs = num_epochs - (epoch + 1)
        estimated_remaining = avg_epoch_time * remaining_epochs if patience_counter < patience else 0
        
        # Log detailed metrics every epoch
        log_progress(
            f"Epoch {epoch+1}/{num_epochs}: "
            f"Train Loss={train_loss/len(train_loader):.4f} | "
            f"Val Acc={val_acc:.4f} | "
            f"Val Bal Acc={val_balanced_acc:.4f} | "
            f"Val ROC-AUC={val_roc_auc:.4f} | "
            f"Val PR-AUC={val_pr_auc:.4f} | "
            f"Val Precision={val_precision:.4f} | "
            f"Val Recall={val_recall:.4f} | "
            f"Val F1={val_f1:.4f} | "
            f"LR={current_lr:.6f} | "
            f"Epoch Time={epoch_time:.1f}s | Est. Remaining={estimated_remaining/60:.1f}min"
        )
        
        if patience_counter >= patience:
            log_progress(f"Early stopping at epoch {epoch+1} (best ROC-AUC: {best_val_roc_auc:.4f})")
            break
    
    # Load best model (selected based on ROC-AUC)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        log_progress(f"Loaded best model with ROC-AUC: {best_val_roc_auc:.4f}")
    
    total_training_time = time.time() - training_start
    avg_epoch_time = np.mean(epoch_times) if epoch_times else 0
    log_progress(f"Training summary: {len(epoch_times)} epochs in {total_training_time:.1f}s ({total_training_time/60:.2f} min), avg {avg_epoch_time:.2f}s/epoch")
    
    return model, best_val_roc_auc  # Return best ROC-AUC (optimization target)


def find_optimal_threshold(val_loader, model, device):
    """Find optimal threshold on validation set using F1 score."""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            probs = F.softmax(out, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    if len(np.unique(all_labels)) < 2:
        return 0.5  # Default threshold if only one class
    
    # Find threshold that maximizes F1 score
    precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    f1_scores = f1_scores[:-1]  # Remove last element (precision=1, recall=0)
    
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    return optimal_threshold


def evaluate_model(model, test_loader, device, threshold=0.5):
    """Evaluate model on test set with optional threshold."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            probs = F.softmax(out, dim=1)
            # Use custom threshold instead of argmax
            pred = (probs[:, 1] >= threshold).long()
            
            all_preds.extend(pred.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Handle edge cases for precision/recall when no positives predicted
    try:
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
    except:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'balanced_accuracy': balanced_accuracy_score(all_labels, all_preds),
        'roc_auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5,
        'pr_auc': average_precision_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'threshold_used': threshold,
    }
    
    return metrics, all_preds, all_probs, all_labels


def main():
    log_progress("="*60)
    log_progress("Graph-Based T1D Classification")
    log_progress("="*60)
    log_progress(f"Random seeds set to {RANDOM_STATE} for reproducibility")
    
    # Verify GPU is available and show GPU info
    if not torch.cuda.is_available():
        log_progress("ERROR: CUDA is required but not available. Please run on a GPU node.", "ERROR")
        raise RuntimeError("CUDA is required but not available. Please run on a GPU node.")
    
    log_progress(f"Using GPU: {torch.cuda.get_device_name(0)}")
    log_progress(f"CUDA Version: {torch.version.cuda}")
    log_progress(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Check cache
    cache_index_file = DOWNSAMPLE_CACHE_DIR / "cache_index.json"
    if not cache_index_file.exists():
        log_progress("ERROR: Downsampling cache not found. Run downsampling first.", "ERROR")
        return
    
    log_progress("Using cached downsampled data (50,000 templates per sample)")
    
    # Set cache directories (both old and new for loading, primary for saving)
    set_cache_dir(GRAPH_CACHE_DIR, OLD_CACHE_DIR, NEW_CACHE_DIR)
    
    # Log cache directory info
    old_count = len(list((OLD_CACHE_DIR / "embeddings").glob("*.pkl"))) if OLD_CACHE_DIR.exists() and (OLD_CACHE_DIR / "embeddings").exists() else 0
    new_count = len(list((NEW_CACHE_DIR / "embeddings").glob("*.pkl"))) if NEW_CACHE_DIR.exists() and (NEW_CACHE_DIR / "embeddings").exists() else 0
    
    log_progress(f"Primary cache directory (for saving): {GRAPH_CACHE_DIR}")
    if OLD_CACHE_DIR.exists():
        log_progress(f"  Old cache directory: {OLD_CACHE_DIR} ({old_count} cached embeddings)")
    if NEW_CACHE_DIR.exists() and NEW_CACHE_DIR != GRAPH_CACHE_DIR:
        log_progress(f"  New cache directory: {NEW_CACHE_DIR} ({new_count} cached embeddings)")
    log_progress(f"  Will check both directories when loading cached embeddings and graphs")
    
    # Load metadata
    metadata_df = load_metadata()
    
    # Split data - only train/val split (no test split)
    log_progress("Splitting data into train/val only (test will be evaluated separately)...")
    sample_ids = metadata_df['repertoire_id'].values
    y = metadata_df['label_positive'].values
    
    # Train/Val split (80/20) - using full dataset
    train_ids, val_ids, y_train, y_val = train_test_split(
        sample_ids, y, test_size=VAL_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    
    log_progress(f"Train: {len(train_ids)} samples ({y_train.sum()} positive)")
    log_progress(f"Val: {len(val_ids)} samples ({y_val.sum()} positive)")
    
    # Initialize CVC embedder only
    log_progress("Initializing CVC embedder...")
    try:
        batch_size = 16384 if DEVICE.type == 'cuda' else 256
        use_fp16 = DEVICE.type == 'cuda'
        embedder = CVCEmbedder(
            device=DEVICE,
            batch_size=batch_size,
            use_fp16=use_fp16,
            verbose=True
        )
        log_progress(f"CVC Embedder: batch_size={batch_size}, FP16={use_fp16}")
    except Exception as e:
        log_progress(f"ERROR: Could not initialize CVC embedder: {e}", "ERROR")
        log_progress("Please ensure CVC is installed from https://github.com/RomiGoldner/CVC", "ERROR")
        return
    
    # Results storage
    all_results = {}
    
    # Try different k values
    import time
    total_start_time = time.time()
    for k in K_VALUES:
        k_start_time = time.time()
        log_progress(f"\n{'='*60}")
        log_progress(f"Processing k={k}")
        log_progress(f"{'='*60}")
        
        # Initialize graph builder
        device_str = str(DEVICE) if isinstance(DEVICE, torch.device) else DEVICE
        graph_builder = GraphBuilder(
            embedder=embedder,
            k_neighbors=k,
            device=device_str,
            verbose=True
        )
        
        # Build graphs (only train and val, no test)
        graph_build_start = time.time()
        train_graphs, train_labels = build_graph_dataset(train_ids, metadata_df, DOWNSAMPLE_CACHE_DIR, graph_builder, k)
        val_graphs, val_labels = build_graph_dataset(val_ids, metadata_df, DOWNSAMPLE_CACHE_DIR, graph_builder, k)
        graph_build_total = time.time() - graph_build_start
        log_progress(f"Total graph building for k={k}: {graph_build_total:.1f} seconds ({graph_build_total/60:.2f} minutes)")
        
        if len(train_graphs) == 0:
            log_progress(f"ERROR: No graphs built for k={k}", "ERROR")
            continue
        
        # Add labels to graphs
        for i, graph in enumerate(train_graphs):
            graph.y = torch.tensor([train_labels[i]], dtype=torch.long)
        for i, graph in enumerate(val_graphs):
            graph.y = torch.tensor([val_labels[i]], dtype=torch.long)
        
        # Get input dimension
        input_dim = train_graphs[0].x.shape[1]
        log_progress(f"Node feature dimension: {input_dim}")
        
        # Create data loaders (only train and val)
        train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False)
        
        # Try different models - GCN for DS8 best model
        models_to_try = {
            'GCN': GCNClassifier,
        }
        
        k_results = {}
        
        for model_name, model_class in models_to_try.items():
            model_start_time = time.time()
            log_progress(f"\n--- Training {model_name} (k={k}) ---")
            
            # OG Configuration: Best performing setup (Job 32129)
            if model_name == 'GIN':
                model = model_class(input_dim=input_dim, num_classes=2, hidden_dim=256, num_layers=4, dropout=0.5)
            else:
                model = model_class(input_dim=input_dim, num_classes=2)
            
            # Train (optimizing for ROC-AUC)
            train_start = time.time()
            model, best_val_roc_auc = train_model(model, train_loader, val_loader, DEVICE, NUM_EPOCHS, EARLY_STOPPING_PATIENCE)
            train_time = time.time() - train_start
            
            # Save model checkpoint for feature ranking
            checkpoint_dir = CODE_DIR.parent / 'model' / 'checkpoints'
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"{model_name}_k{k}_dataset{DATASET_NUM}_best_model.pth"
            
            # Extract model hyperparameters
            if model_name == 'GIN':
                hidden_dim = 256
                num_layers = 4
            elif model_name == 'GCN':
                hidden_dim = 128
                num_layers = 3
            else:
                hidden_dim = 128
                num_layers = 3
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_name': model_name,
                'model_class': model_class.__name__,
                'input_dim': input_dim,
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'num_classes': 2,
                'k': k,
                'embedder_type': EMBEDDER_TYPE,
                'dataset_num': DATASET_NUM,
                'roc_auc': best_val_roc_auc,
                'best_val_acc': best_val_roc_auc
            }, checkpoint_path)
            log_progress(f"Saved model checkpoint to: {checkpoint_path}")
            log_progress(f"Training completed in {train_time:.1f} seconds ({train_time/60:.2f} minutes)")
            best_val_acc = best_val_roc_auc  # For backward compatibility in results
            
            # Find optimal threshold on validation set
            eval_start = time.time()
            optimal_threshold = find_optimal_threshold(val_loader, model, DEVICE)
            log_progress(f"  Optimal threshold (validation): {optimal_threshold:.4f}")
            
            # Evaluate on validation set with optimal threshold
            # Note: Test set will be evaluated separately later
            metrics, preds, probs, labels = evaluate_model(model, val_loader, DEVICE, threshold=optimal_threshold)
            eval_time = time.time() - eval_start
            log_progress(f"Evaluation completed in {eval_time:.1f} seconds ({eval_time/60:.2f} minutes)")
            
            log_progress(f"{model_name} (k={k}) Validation Results (test will be evaluated separately):")
            log_progress(f"  Threshold used: {metrics.get('threshold_used', 0.5):.4f}")
            log_progress(f"  Accuracy: {metrics['accuracy']:.4f}")
            log_progress(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
            log_progress(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            if 'pr_auc' in metrics:
                log_progress(f"  PR-AUC: {metrics['pr_auc']:.4f}")
            log_progress(f"  Precision: {metrics['precision']:.4f}")
            log_progress(f"  Recall: {metrics['recall']:.4f}")
            log_progress(f"  F1-Score: {metrics['f1']:.4f}")
            
            model_total_time = time.time() - model_start_time
            log_progress(f"Total {model_name} time for k={k}: {model_total_time:.1f} seconds ({model_total_time/60:.2f} minutes)")
            
            k_results[model_name] = {
                'metrics': metrics,
                'best_val_acc': best_val_acc,
                'k': k
            }
        
        k_total_time = time.time() - k_start_time
        log_progress(f"\n{'='*60}")
        log_progress(f"Completed k={k} in {k_total_time:.1f} seconds ({k_total_time/60:.2f} minutes)")
        log_progress(f"  Graph building: {graph_build_total:.1f}s ({graph_build_total/60:.2f} min)")
        # Note: train_time and eval_time are logged per model above
        log_progress(f"{'='*60}\n")
        
        all_results[f'k={k}'] = k_results
        
        # Save intermediate results after each k value
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = RESULTS_DIR / f"graph_classification_results_{timestamp}.json"
        
        # Convert numpy types to native Python types for JSON
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_results = convert_to_serializable(all_results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Display performance summary for this k value
        log_progress(f"\n{'='*60}")
        log_progress(f"PERFORMANCE SUMMARY FOR k={k}")
        log_progress(f"{'='*60}")
        for model_name, result in k_results.items():
            m = result['metrics']
            log_progress(f"{model_name}:")
            log_progress(f"  Balanced Accuracy: {m['balanced_accuracy']:.4f}")
            log_progress(f"  ROC-AUC: {m['roc_auc']:.4f}")
            log_progress(f"  F1-Score: {m['f1']:.4f}")
            log_progress(f"  Precision: {m['precision']:.4f}")
            log_progress(f"  Recall: {m['recall']:.4f}")
            log_progress(f"  Best Val Acc: {result['best_val_acc']:.4f}")
        
        # Find best model for this k
        best_model = max(k_results.items(), key=lambda x: x[1]['metrics']['balanced_accuracy'])
        log_progress(f"\nBest model for k={k}: {best_model[0]}")
        log_progress(f"  Balanced Accuracy: {best_model[1]['metrics']['balanced_accuracy']:.4f}")
        log_progress(f"  ROC-AUC: {best_model[1]['metrics']['roc_auc']:.4f}")
        log_progress(f"{'='*60}")
        log_progress(f"Results saved to: {results_file}")
    
    # Final summary across all k values
    log_progress(f"\n{'='*60}")
    log_progress("FINAL PERFORMANCE SUMMARY (ALL k VALUES)")
    log_progress(f"{'='*60}")
    
    for k_str, k_results in all_results.items():
        k_val = k_str.split('=')[1]
        best_model = max(k_results.items(), key=lambda x: x[1]['metrics']['balanced_accuracy'])
        log_progress(f"\n{k_str}:")
        log_progress(f"  Best: {best_model[0]} - Balanced Acc: {best_model[1]['metrics']['balanced_accuracy']:.4f}, ROC-AUC: {best_model[1]['metrics']['roc_auc']:.4f}")
    
    # Overall best
    all_best = []
    for k_str, k_results in all_results.items():
        best = max(k_results.items(), key=lambda x: x[1]['metrics']['balanced_accuracy'])
        all_best.append((k_str, best[0], best[1]['metrics']['balanced_accuracy'], best[1]['metrics']['roc_auc']))
    
    if all_best:
        overall_best = max(all_best, key=lambda x: x[2])
        log_progress(f"\n{'='*60}")
        log_progress(f"OVERALL BEST: {overall_best[0]}, {overall_best[1]}")
        log_progress(f"  Balanced Accuracy: {overall_best[2]:.4f}")
        log_progress(f"  ROC-AUC: {overall_best[3]:.4f}")
        log_progress(f"{'='*60}")
    
    total_time = time.time() - total_start_time
    log_progress("="*60)
    log_progress("Classification Complete")
    log_progress(f"Total runtime: {total_time:.1f} seconds ({total_time/60:.2f} minutes, {total_time/3600:.2f} hours)")
    log_progress("="*60)


if __name__ == "__main__":
    main()

