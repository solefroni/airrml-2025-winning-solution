#!/usr/bin/env python3
"""
Train GCN with Best Hyperparameters and Save Checkpoint
=========================================================
Retrains the GCN model using the best hyperparameters from Optuna
and saves a proper checkpoint for the ensemble.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import json
import pickle
import torch.nn.functional as F
import torch.nn as nn

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "samplegraphsT1D" / "scripts"))

from ensemble_predictor import GCNClassifier
from cvc_embedder import CVCEmbedder
from graph_builder import GraphBuilder
from cache_utils import set_cache_dir
import graph_classification as gc_module
from torch_geometric.data import DataLoader

# =============================================================================
# CONFIGURATION - Update these paths for your environment
# =============================================================================
import os
CODE_DIR = Path(__file__).parent.resolve()
DATASET_NUM = 8
DATA_DIR = Path(os.environ.get('DS8_TRAIN_DATA', '../input'))
DOWNSAMPLE_CACHE_DIR = Path(os.environ.get('DS8_DOWNSAMPLE_CACHE', '../cache/downsample'))
GRAPH_CACHE_DIR = Path(os.environ.get('DS8_GRAPH_CACHE', '../cache/graphs'))
if GRAPH_CACHE_DIR.exists() and (GRAPH_CACHE_DIR / "graphs").exists():
    pass  # Use existing cache
else:
    GRAPH_CACHE_DIR = NEW_CACHE_DIR

VAL_SIZE = 0.2
RANDOM_STATE = 42
K_VALUE = 30
EMBEDDER_TYPE = 'cvc'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15

# Load best hyperparameters from model config
MODEL_DIR = CODE_DIR.parent / 'model'
RESULTS_FILE = MODEL_DIR / "hyperparameter_results.json"
with open(RESULTS_FILE, 'r') as f:
    results = json.load(f)

BEST_PARAMS = results['best_params']
EXPECTED_AUC = results['best_value']

# Output checkpoint path
CHECKPOINT_DIR = MODEL_DIR
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = CHECKPOINT_DIR / f"gcn_model.pth"

print("="*70)
print("TRAINING GCN WITH BEST HYPERPARAMETERS")
print("="*70)
print(f"Device: {DEVICE}")
print(f"Dataset: {DATASET_NUM}")
print(f"K-value: {K_VALUE}")
print(f"Expected ROC-AUC: {EXPECTED_AUC:.6f}")
print(f"\nBest hyperparameters:")
for key, value in BEST_PARAMS.items():
    print(f"  {key}: {value}")
print()

# Load data
print("Loading data...")
metadata = pd.read_csv(DATA_DIR / "metadata.csv")
sample_ids = metadata['repertoire_id'].values
labels = metadata['label_positive'].astype(int).values

print(f"Total samples: {len(sample_ids)}")
print(f"Positive: {sum(labels)}, Negative: {len(labels) - sum(labels)}")

# Split using EXACT same method as hyperparameter tuning
train_ids, val_ids, y_train, y_val = train_test_split(
    sample_ids, labels, test_size=VAL_SIZE, stratify=labels, random_state=RANDOM_STATE
)

print(f"\nTrain: {len(train_ids)} samples ({sum(y_train)} positive)")
print(f"Val: {len(val_ids)} samples ({sum(y_val)} positive)")
print()

# Set up graph building
print("Setting up graph building...")
set_cache_dir(GRAPH_CACHE_DIR)
gc_module.EMBEDDER_TYPE = EMBEDDER_TYPE
gc_module.GRAPH_CACHE_DIR = GRAPH_CACHE_DIR

embedder = CVCEmbedder(device=DEVICE, verbose=False)
graph_builder = GraphBuilder(embedder=embedder, k_neighbors=K_VALUE, device=str(DEVICE), 
                             verbose=False, use_hnswlib=True)

# Build graphs
print(f"Building {len(train_ids)} training graphs...")
train_graphs, train_labels = gc_module.build_graph_dataset(
    train_ids, metadata, DOWNSAMPLE_CACHE_DIR, graph_builder, K_VALUE
)
print(f"Built: {len(train_graphs)} training graphs")

print(f"\nBuilding {len(val_ids)} validation graphs...")
val_graphs, val_labels = gc_module.build_graph_dataset(
    val_ids, metadata, DOWNSAMPLE_CACHE_DIR, graph_builder, K_VALUE
)
print(f"Built: {len(val_graphs)} validation graphs")

# Assign labels to graphs
for i, graph in enumerate(train_graphs):
    graph.y = torch.tensor([int(train_labels[i])], dtype=torch.long)
for i, graph in enumerate(val_graphs):
    graph.y = torch.tensor([int(val_labels[i])], dtype=torch.long)

# Get input dimension
input_dim = train_graphs[0].x.shape[1]
print(f"\nInput dimension: {input_dim}")

# Create model with best hyperparameters
print("\nCreating model...")
model = GCNClassifier(
    input_dim=input_dim,
    hidden_dim=BEST_PARAMS['hidden_dim'],
    num_layers=BEST_PARAMS['num_layers'],
    num_classes=2,
    dropout=BEST_PARAMS['dropout']
).to(DEVICE)

print(f"Model architecture:")
print(f"  input_dim: {input_dim}")
print(f"  hidden_dim: {BEST_PARAMS['hidden_dim']}")
print(f"  num_layers: {BEST_PARAMS['num_layers']}")
print(f"  dropout: {BEST_PARAMS['dropout']:.4f}")
print()

# Create data loaders
train_loader = DataLoader(train_graphs, batch_size=BEST_PARAMS['batch_size'], shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=BEST_PARAMS['batch_size'], shuffle=False)

# Training setup
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=BEST_PARAMS['learning_rate'],
    weight_decay=BEST_PARAMS['weight_decay']
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max',
    factor=BEST_PARAMS['lr_scheduler_factor'],
    patience=BEST_PARAMS['lr_scheduler_patience'],
    min_lr=1e-6
)
criterion = nn.CrossEntropyLoss()

# Training loop
print("Starting training...")
best_val_roc_auc = 0.0
patience_counter = 0
best_model_state = None
best_epoch = 0

for epoch in range(NUM_EPOCHS):
    # Training
    model.train()
    train_loss = 0.0
    for data in train_loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # Validation
    model.eval()
    val_all_probs = []
    val_all_labels = []
    
    with torch.no_grad():
        for data in val_loader:
            data = data.to(DEVICE)
            out = model(data.x, data.edge_index, data.batch)
            probs = F.softmax(out, dim=1)
            val_all_probs.extend(probs[:, 1].cpu().numpy())
            val_all_labels.extend(data.y.cpu().numpy())
    
    val_all_probs = np.array(val_all_probs)
    val_all_labels = np.array(val_all_labels)
    
    # Calculate ROC-AUC
    if len(np.unique(val_all_labels)) > 1:
        val_roc_auc = roc_auc_score(val_all_labels, val_all_probs)
    else:
        val_roc_auc = 0.5
    
    # Update best model
    if val_roc_auc > best_val_roc_auc:
        best_val_roc_auc = val_roc_auc
        patience_counter = 0
        best_model_state = model.state_dict().copy()
        best_epoch = epoch + 1
        print(f"  *** New best at epoch {epoch+1}: Val ROC-AUC={val_roc_auc:.6f} ***")
    else:
        patience_counter += 1
    
    scheduler.step(val_roc_auc)
    
    # Log progress
    if (epoch + 1) % 5 == 0 or patience_counter == 0:
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}: Train Loss={train_loss:.4f}, Val ROC-AUC={val_roc_auc:.6f}, Best={best_val_roc_auc:.6f}, Patience={patience_counter}/{EARLY_STOPPING_PATIENCE}")
    
    # Early stopping
    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

# Load best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)

print(f"\n{'='*70}")
print(f"TRAINING COMPLETE")
print(f"{'='*70}")
print(f"Best epoch: {best_epoch}")
print(f"Best validation ROC-AUC: {best_val_roc_auc:.6f}")
print(f"Expected ROC-AUC: {EXPECTED_AUC:.6f}")
print(f"Difference: {abs(best_val_roc_auc - EXPECTED_AUC):.6f}")
print()

# Save checkpoint
print(f"Saving checkpoint to: {CHECKPOINT_PATH}")
checkpoint = {
    'model_state_dict': model.state_dict(),
    'model_name': 'GCN',
    'model_class': 'GCNClassifier',
    'input_dim': input_dim,
    'hidden_dim': BEST_PARAMS['hidden_dim'],
    'num_layers': BEST_PARAMS['num_layers'],
    'dropout': BEST_PARAMS['dropout'],
    'num_classes': 2,
    'k': K_VALUE,
    'embedder_type': EMBEDDER_TYPE,
    'dataset_num': DATASET_NUM,
    'roc_auc': best_val_roc_auc,
    'best_epoch': best_epoch,
    'hyperparameters': BEST_PARAMS,
    'train_size': len(train_graphs),
    'val_size': len(val_graphs),
    'random_state': RANDOM_STATE
}

torch.save(checkpoint, CHECKPOINT_PATH)
print(f"✓ Checkpoint saved successfully")
print()

# Verify checkpoint can be loaded
print("Verifying checkpoint...")
loaded_checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
print(f"✓ Checkpoint loaded successfully")
print(f"  Model: {loaded_checkpoint['model_name']}")
print(f"  Input dim: {loaded_checkpoint['input_dim']}")
print(f"  Hidden dim: {loaded_checkpoint['hidden_dim']}")
print(f"  Num layers: {loaded_checkpoint['num_layers']}")
print(f"  ROC-AUC: {loaded_checkpoint['roc_auc']:.6f}")
print()

print("="*70)
print("SUCCESS: Model trained and checkpoint saved")
print("="*70)
print(f"\nCheckpoint path: {CHECKPOINT_PATH}")
print(f"Use this checkpoint in the ensemble model.")



