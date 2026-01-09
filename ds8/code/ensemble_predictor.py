#!/usr/bin/env python3
"""
Ensemble Predictor for T1D Classification
==========================================

Combines GCN (graph-based) and XGBoost (feature-based) predictions using stacking.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "samplegraphsT1D" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent / "simple"))

from cvc_embedder import CVCEmbedder
from graph_builder import GraphBuilder
from cache_utils import (
    load_cached_embeddings, save_cached_embeddings,
    load_cached_graph, save_cached_graph,
    get_sequence_hash, get_graph_cache_key,
    set_cache_dir
)
from graph_classification import build_graph_dataset
from extract_repertoire_features import extract_all_features


class GCNClassifier(nn.Module):
    """Graph Convolutional Network for graph classification."""
    
    def __init__(self, input_dim, hidden_dim=160, num_layers=3, num_classes=2, dropout=0.47):
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
        
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return x


class EnsemblePredictor:
    """Ensemble predictor combining GCN and XGBoost models."""
    
    def __init__(self, gcn_model_path=None, xgb_model_path=None, meta_model_path=None,
                 gcn_config=None, device='cuda'):
        """
        Initialize ensemble predictor.
        
        Args:
            gcn_model_path: Path to trained GCN model state dict
            xgb_model_path: Path to trained XGBoost model pickle
            meta_model_path: Path to trained meta-learner (stacking model)
            gcn_config: Dictionary with GCN configuration (input_dim, hidden_dim, etc.)
            device: Device to run GCN on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.gcn_model = None
        self.xgb_model = None
        self.meta_model = None
        self.selected_features = None
        self.gcn_config = gcn_config or {}
        
        # Load models if paths provided
        if gcn_model_path and Path(gcn_model_path).exists():
            self.load_gcn_model(gcn_model_path)
        
        if xgb_model_path and Path(xgb_model_path).exists():
            self.load_xgb_model(xgb_model_path)
        
        if meta_model_path and Path(meta_model_path).exists():
            self.load_meta_model(meta_model_path)
    
    def load_gcn_model(self, model_path):
        """Load trained GCN model."""
        print(f"Loading GCN model from {model_path}...")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract architecture from checkpoint (use saved architecture, not config)
        if isinstance(checkpoint, dict):
            # Get architecture parameters from checkpoint
            input_dim = checkpoint.get('input_dim', self.gcn_config.get('input_dim', 59))  # Default 59, not 768!
            hidden_dim = checkpoint.get('hidden_dim', self.gcn_config.get('hidden_dim', 160))
            num_layers = checkpoint.get('num_layers', self.gcn_config.get('num_layers', 3))
            dropout = checkpoint.get('dropout', self.gcn_config.get('dropout', 0.47))
            
            print(f"Model architecture from checkpoint: hidden_dim={hidden_dim}, num_layers={num_layers}, dropout={dropout}")
            
            # Create model with checkpoint architecture
            self.gcn_model = GCNClassifier(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_classes=2,
                dropout=dropout
            )
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                # Assume the dict itself is the state_dict
                state_dict = checkpoint
        else:
            # Fallback: use config if checkpoint is not a dict
            input_dim = self.gcn_config.get('input_dim', 59)  # Default 59, not 768!
            hidden_dim = self.gcn_config.get('hidden_dim', 160)
            num_layers = self.gcn_config.get('num_layers', 3)
            dropout = self.gcn_config.get('dropout', 0.47)
            
            self.gcn_model = GCNClassifier(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_classes=2,
                dropout=dropout
            )
            state_dict = checkpoint
        
        self.gcn_model.load_state_dict(state_dict)
        self.gcn_model.to(self.device)
        self.gcn_model.eval()
        print("GCN model loaded successfully")
    
    def load_xgb_model(self, model_path):
        """Load trained XGBoost model and selected features."""
        print(f"Loading XGBoost model from {model_path}...")
        
        model_dir = Path(model_path).parent
        
        # Load model
        with open(model_path, 'rb') as f:
            self.xgb_model = pickle.load(f)
        
        # Load selected features
        features_file = model_dir / "selected_features.json"
        if features_file.exists():
            with open(features_file, 'r') as f:
                self.selected_features = json.load(f)
            print(f"Loaded {len(self.selected_features)} selected features")
        else:
            print("Warning: selected_features.json not found")
        
        print("XGBoost model loaded successfully")
    
    def load_meta_model(self, model_path):
        """Load trained meta-learner (stacking model)."""
        print(f"Loading meta-learner from {model_path}...")
        
        with open(model_path, 'rb') as f:
            self.meta_model = pickle.load(f)
        
        print("Meta-learner loaded successfully")
    
    def predict_gcn(self, graph_loader):
        """Get GCN predictions from graph data."""
        if self.gcn_model is None:
            raise ValueError("GCN model not loaded")
        
        self.gcn_model.eval()
        probs = []
        
        with torch.no_grad():
            for data in graph_loader:
                data = data.to(self.device)
                out = self.gcn_model(data.x, data.edge_index, data.batch)
                prob = F.softmax(out, dim=1)[:, 1]
                probs.extend(prob.cpu().numpy())
        
        return np.array(probs)
    
    def predict_xgb(self, features_df):
        """Get XGBoost predictions from feature DataFrame."""
        if self.xgb_model is None:
            raise ValueError("XGBoost model not loaded")
        
        if self.selected_features is None:
            raise ValueError("Selected features not loaded")
        
        # Select and prepare features
        missing_features = [f for f in self.selected_features if f not in features_df.columns]
        if missing_features:
            print(f"Warning: {len(missing_features)} features missing. Filling with 0.")
            for feat in missing_features:
                features_df[feat] = 0.0
        
        X_selected = features_df[self.selected_features].fillna(0)
        probs = self.xgb_model.predict_proba(X_selected)[:, 1]
        
        return probs
    
    def predict_ensemble(self, graph_loader, features_df, return_individual=False):
        """
        Get ensemble predictions.
        
        Args:
            graph_loader: DataLoader with graph data
            features_df: DataFrame with repertoire features
            return_individual: If True, also return individual model predictions
        
        Returns:
            ensemble_probs: Ensemble probability predictions
            (gcn_probs, xgb_probs): Individual predictions if return_individual=True
        """
        # Get base model predictions
        gcn_probs = self.predict_gcn(graph_loader)
        xgb_probs = self.predict_xgb(features_df)
        
        # Ensure same length
        if len(gcn_probs) != len(xgb_probs):
            raise ValueError(f"GCN and XGBoost predictions have different lengths: {len(gcn_probs)} vs {len(xgb_probs)}")
        
        # Combine via meta-learner or simple average
        if self.meta_model is not None:
            # Use stacking meta-learner
            meta_features = np.column_stack([gcn_probs, xgb_probs])
            ensemble_probs = self.meta_model.predict_proba(meta_features)[:, 1]
        else:
            # Simple weighted average (default: equal weights)
            ensemble_probs = 0.5 * gcn_probs + 0.5 * xgb_probs
        
        if return_individual:
            return ensemble_probs, gcn_probs, xgb_probs
        return ensemble_probs
    
    def fit_meta_learner(self, gcn_train_probs, xgb_train_probs, y_train,
                         gcn_val_probs=None, xgb_val_probs=None, y_val=None,
                         meta_model_type='logistic'):
        """
        Train meta-learner on base model predictions.
        
        Args:
            gcn_train_probs: GCN predictions on training set
            xgb_train_probs: XGBoost predictions on training set
            y_train: Training labels
            gcn_val_probs: GCN predictions on validation set (optional, for evaluation)
            xgb_val_probs: XGBoost predictions on validation set (optional)
            y_val: Validation labels (optional)
            meta_model_type: Type of meta-learner ('logistic' or 'random_forest')
        
        Returns:
            Trained meta-learner
        """
        print("Training meta-learner...")
        
        # Create meta-features
        meta_train = np.column_stack([gcn_train_probs, xgb_train_probs])
        
        # Create meta-learner
        if meta_model_type == 'logistic':
            self.meta_model = LogisticRegression(random_state=42, max_iter=1000)
        elif meta_model_type == 'random_forest':
            self.meta_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            raise ValueError(f"Unknown meta_model_type: {meta_model_type}")
        
        # Train
        self.meta_model.fit(meta_train, y_train)
        
        # Evaluate if validation data provided
        if gcn_val_probs is not None and xgb_val_probs is not None and y_val is not None:
            meta_val = np.column_stack([gcn_val_probs, xgb_val_probs])
            val_probs = self.meta_model.predict_proba(meta_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_probs)
            print(f"Meta-learner validation ROC-AUC: {val_auc:.4f}")
        
        return self.meta_model
    
    def save_meta_model(self, save_path):
        """Save meta-learner to disk."""
        if self.meta_model is None:
            raise ValueError("Meta-learner not trained")
        
        with open(save_path, 'wb') as f:
            pickle.dump(self.meta_model, f)
        
        print(f"Meta-learner saved to {save_path}")

