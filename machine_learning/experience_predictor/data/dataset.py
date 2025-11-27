"""
PyTorch Dataset for experience prediction.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import logging

from .loader import (
    load_aal_atlas, 
    load_targets, 
    build_feature_matrix,
    compute_distance_matrix,
    build_adjacency_from_distance
)

logger = logging.getLogger(__name__)


class ExperienceDataset(Dataset):
    """
    Dataset for predicting subjective experience from spectral features.
    
    Supports both tabular (MLP) and graph (GNN) formats.
    """
    
    def __init__(self, 
                 X: np.ndarray,
                 y: np.ndarray,
                 edge_index: Optional[torch.Tensor] = None,
                 num_regions: int = 90,
                 num_bands: int = 6,
                 feature_names: Optional[List[str]] = None,
                 target_names: Optional[List[str]] = None,
                 use_graph: bool = False):
        """
        Args:
            X: Feature matrix [N, D] where D = num_regions * num_bands * num_conditions
            y: Target matrix [N, num_targets]
            edge_index: Graph connectivity [2, E] (optional, for GNN)
            num_regions: Number of brain regions (90 for AAL)
            num_bands: Number of frequency bands per condition
            feature_names: Names of features
            target_names: Names of target variables
            use_graph: If True, format data for GNN
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.edge_index = edge_index
        self.num_regions = num_regions
        self.num_bands = num_bands
        self.feature_names = feature_names
        self.target_names = target_names
        self.use_graph = use_graph
        
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.n_targets = y.shape[1]
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        if self.use_graph:
            return self._get_graph(idx)
        else:
            return self._get_tabular(idx)
    
    def _get_tabular(self, idx):
        """Return simple (features, targets) tuple for MLP."""
        return self.X[idx], self.y[idx]
    
    def _get_graph(self, idx):
        """
        Return PyG Data object for GNN.
        
        Node features: spectral power across bands for each region
        """
        x = self.X[idx]
        
        # Reshape to [num_regions, features_per_region]
        # Features are organized as: band1_region1...band1_region90, band2_region1...
        features_per_region = self.n_features // self.num_regions
        
        # Reorganize: for each region, collect all its band features
        node_features = []
        for region_idx in range(self.num_regions):
            region_feats = []
            for feat_idx in range(self.n_features):
                # Check if this feature belongs to this region
                if feat_idx % self.num_regions == region_idx:
                    region_feats.append(x[feat_idx].item())
            node_features.append(region_feats)
        
        node_x = torch.FloatTensor(node_features)
        
        data = Data(
            x=node_x,
            edge_index=self.edge_index,
            y=self.y[idx]
        )
        
        return data


def create_dataset_from_config(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], Optional[torch.Tensor]]:
    """
    Create dataset from configuration.
    
    Returns:
        Tuple of (X, y, feature_names, target_names, edge_index)
    """
    data_dir = config['paths']['data_dir']
    data_config = config['data']
    
    # Load features
    X, feature_names = build_feature_matrix(
        data_dir,
        use_baseline=data_config['use_baseline'],
        use_dmt=data_config['use_dmt'],
        use_difference=data_config['use_difference']
    )
    
    # Load targets
    targets_df, target_names = load_targets(data_dir)
    y = targets_df.values
    
    # Handle NaN in targets (if any)
    if np.isnan(y).any():
        logger.warning("NaN values found in targets, filling with column means")
        col_means = np.nanmean(y, axis=0)
        for j in range(y.shape[1]):
            y[np.isnan(y[:, j]), j] = col_means[j]
    
    # Build graph if needed
    edge_index = None
    if data_config.get('use_graph', False):
        aal = load_aal_atlas(data_dir)
        distances = compute_distance_matrix(aal)
        
        graph_config = data_config.get('graph', {})
        adj = build_adjacency_from_distance(
            distances,
            threshold=graph_config.get('distance_threshold', 50.0),
            k_neighbors=graph_config.get('k_neighbors', None)
        )
        
        # Convert to edge_index format
        edge_index = torch.LongTensor(np.array(np.nonzero(adj)))
    
    logger.info(f"Dataset created: X={X.shape}, y={y.shape}")
    
    return X, y, feature_names, target_names, edge_index


def create_cv_splits(X: np.ndarray, 
                     y: np.ndarray,
                     n_folds: int = 5,
                     random_state: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create cross-validation splits.
    
    Returns:
        List of (train_indices, val_indices) tuples
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    splits = list(kf.split(X))
    
    logger.info(f"Created {n_folds}-fold CV splits")
    for i, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"  Fold {i+1}: train={len(train_idx)}, val={len(val_idx)}")
    
    return splits


def normalize_data(X_train: np.ndarray, 
                   X_val: np.ndarray,
                   y_train: np.ndarray,
                   y_val: np.ndarray,
                   normalize_features: bool = True,
                   normalize_targets: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Normalize features and/or targets.
    
    Returns:
        Normalized data and scalers dict for inverse transform
    """
    scalers = {}
    
    if normalize_features:
        scaler_X = StandardScaler()
        X_train = scaler_X.fit_transform(X_train)
        X_val = scaler_X.transform(X_val)
        scalers['X'] = scaler_X
    
    if normalize_targets:
        scaler_y = StandardScaler()
        y_train = scaler_y.fit_transform(y_train)
        y_val = scaler_y.transform(y_val)
        scalers['y'] = scaler_y
    
    return X_train, X_val, y_train, y_val, scalers

