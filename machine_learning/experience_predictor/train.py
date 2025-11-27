#!/usr/bin/env python3
"""
Training script for multi-output experience prediction.

Supports two input types:
- spectral: Spectral power from spectral_sources/ (CSVs)
- graphs: Synchronization graphs from fwd-inv-stc/ (phases-*.pkl)

Uses k-fold cross-validation due to small sample size (N=29).
"""

import warnings
# Suppress PyTorch Geometric scatter warnings
warnings.filterwarnings('ignore', message='.*scatter.*')
warnings.filterwarnings('ignore', category=UserWarning, module='torch_geometric')

import argparse
import sys
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Data
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

sys.path.append(str(Path(__file__).parent))

from models import create_model_from_config
from utils.visualization import save_all_visualizations
from utils.logging_utils import (
    setup_tensorboard, log_epoch_metrics, log_per_target_metrics,
    log_final_results, save_results_json, create_experiment_summary
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True


def get_device(config: Dict[str, Any]) -> torch.device:
    """Get torch device from config."""
    device_name = config['device']
    if device_name == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device_name = 'cpu'
    return torch.device(device_name)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_spectral_data(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[torch.Tensor]]:
    """
    Load spectral power data from CSVs.
    
    Returns:
        Tuple of (X, y, target_names, edge_index)
    """
    from data.loader import load_aal_atlas, load_targets, load_spectral_data as _load_spectral
    from data.loader import build_feature_matrix, compute_distance_matrix, build_adjacency_from_distance
    
    spectral_dir = config['paths']['spectral_dir']
    spectral_config = config['spectral']
    
    # Load features
    X, feature_names = build_feature_matrix(
        spectral_dir,
        use_baseline=spectral_config['use_baseline'],
        use_dmt=spectral_config['use_dmt'],
        use_difference=spectral_config['use_difference']
    )
    
    # Load targets
    targets_df, target_names = load_targets(spectral_dir)
    y = targets_df.values
    
    # Handle NaN
    if np.isnan(y).any():
        col_means = np.nanmean(y, axis=0)
        for j in range(y.shape[1]):
            y[np.isnan(y[:, j]), j] = col_means[j]
    
    # Build graph if needed
    edge_index = None
    if spectral_config.get('build_graph', False):
        aal = load_aal_atlas(spectral_dir)
        distances = compute_distance_matrix(aal)
        graph_config = spectral_config.get('graph', {})
        adj = build_adjacency_from_distance(
            distances,
            threshold=graph_config.get('distance_threshold', 50.0),
            k_neighbors=graph_config.get('k_neighbors', None)
        )
        edge_index = torch.LongTensor(np.array(np.nonzero(adj)))
    
    logger.info(f"Loaded spectral data: X={X.shape}, y={y.shape}")
    
    return X, y, target_names, edge_index


def load_graph_data(config: Dict[str, Any], force_rebuild: bool = False, num_workers: int = None) -> Tuple[List[Data], np.ndarray, List[str], List[str]]:
    """
    Load graph data from phases-*.pkl files with caching.
    
    Args:
        config: Configuration dictionary
        force_rebuild: If True, rebuild from scratch ignoring cache
        num_workers: Number of parallel workers for loading
    
    Returns:
        Tuple of (graphs, y, target_names, subject_ids)
    """
    from data.graph_loader import load_all_dmt_subjects
    from data.loader import load_targets
    import hashlib
    
    phases_dir = config['paths']['phases_dir']
    spectral_dir = config['paths']['spectral_dir']
    graph_config = config['graphs']
    
    # Create cache directory in data/ folder
    # This ensures cache is shared across all experiments
    cache_dir = Path(__file__).parent / 'data' / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create cache key based on config
    cache_key = hashlib.md5(
        f"{graph_config.get('bands', 'all')}_{graph_config['epoch_aggregation']}_"
        f"{graph_config['fully_connected']}_{graph_config['edge_threshold']}_"
        f"{graph_config['band_combination']}_{graph_config.get('use_stc', False)}".encode()
    ).hexdigest()[:12]
    
    cache_file = cache_dir / f'graphs_cache_{cache_key}.pt'
    
    # Try to load from cache
    if cache_file.exists() and not force_rebuild:
        logger.info(f"  Loading from cache: {cache_file.name}")
        try:
            # weights_only=False needed for PyG Data objects (PyTorch 2.6+)
            cached = torch.load(cache_file, weights_only=False)
            graphs = cached['graphs']
            subject_ids = cached['subject_ids']
            logger.info(f"  ✓ Loaded {len(graphs)} graphs from cache")
        except Exception as e:
            logger.warning(f"  Cache load failed: {e}, rebuilding...")
            force_rebuild = True
    
    if not cache_file.exists() or force_rebuild:
        # Load graphs from scratch
        logger.info("  Building graphs from phases files (this may take a while)...")
        graphs, subject_ids = load_all_dmt_subjects(
            phases_dir=phases_dir,
            bands=graph_config.get('bands', None),
            aggregation=graph_config['epoch_aggregation'],
            fully_connected=graph_config['fully_connected'],
            edge_threshold=graph_config['edge_threshold'],
            combine_bands=graph_config['band_combination'],
            use_stc=graph_config.get('use_stc', False),
            num_workers=num_workers
        )
        
        # Save to cache
        logger.info(f"  Saving {len(graphs)} graphs to cache...")
        torch.save({'graphs': graphs, 'subject_ids': subject_ids}, cache_file)
        logger.info(f"  ✓ Cache saved: {cache_file.name}")
    else:
        # Already loaded from cache above
        pass
    
    # Load targets
    targets_df, target_names = load_targets(spectral_dir)
    y_full = targets_df.values
    
    # Match subjects - assume both are ordered the same way
    # The phases files have subjects S01, S02, S04, S05, etc. (skipping some)
    # Need to match them to target rows
    
    # Parse subject indices from subject_ids
    subject_indices = []
    for sid in subject_ids:
        try:
            idx = int(sid[1:]) - 1  # S01 -> 0, S02 -> 1, etc.
            if idx < y_full.shape[0]:
                subject_indices.append(idx)
            else:
                subject_indices.append(None)
        except:
            subject_indices.append(None)
    
    # Filter valid subjects
    valid_graphs = []
    valid_y = []
    valid_sids = []
    
    for i, (graph, sid) in enumerate(zip(graphs, subject_ids)):
        if subject_indices[i] is not None:
            valid_graphs.append(graph)
            valid_y.append(y_full[subject_indices[i]])
            valid_sids.append(sid)
    
    y = np.array(valid_y)
    
    # Handle NaN
    if np.isnan(y).any():
        col_means = np.nanmean(y, axis=0)
        for j in range(y.shape[1]):
            y[np.isnan(y[:, j]), j] = col_means[j]
    
    logger.info(f"Loaded graph data: {len(valid_graphs)} graphs, y={y.shape}")
    
    return valid_graphs, y, target_names, valid_sids


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                   target_names: List[str] = None) -> Dict[str, Any]:
    """Compute regression metrics."""
    metrics = {}
    
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['r2'] = r2_score(y_true, y_pred)
    
    # Mean Pearson correlation
    n_targets = y_true.shape[1]
    correlations = []
    for i in range(n_targets):
        if np.std(y_true[:, i]) > 1e-6 and np.std(y_pred[:, i]) > 1e-6:
            r, _ = pearsonr(y_true[:, i], y_pred[:, i])
            correlations.append(r if not np.isnan(r) else 0)
        else:
            correlations.append(0)
    
    metrics['mean_pearson'] = np.mean(correlations)
    metrics['std_pearson'] = np.std(correlations)
    
    # Per-target metrics
    if target_names is not None:
        metrics['per_target'] = {}
        for i, name in enumerate(target_names):
            if i < n_targets:
                if np.std(y_true[:, i]) > 1e-6 and np.std(y_pred[:, i]) > 1e-6:
                    r, p = pearsonr(y_true[:, i], y_pred[:, i])
                else:
                    r, p = 0, 1.0
                metrics['per_target'][name] = {
                    'mse': float(mean_squared_error(y_true[:, i], y_pred[:, i])),
                    'mae': float(mean_absolute_error(y_true[:, i], y_pred[:, i])),
                    'pearson_r': float(r) if not np.isnan(r) else 0,
                    'pearson_p': float(p) if not np.isnan(p) else 1.0
                }
    
    return metrics


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch_tabular(model, loader, optimizer, criterion, device):
    """Train epoch for tabular (MLP) model."""
    model.train()
    total_loss = 0
    n_batches = 0
    
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def train_epoch_graph(model, loader, optimizer, criterion, device):
    """Train epoch for graph (GNN) model."""
    model.train()
    total_loss = 0
    n_batches = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        
        # Handle target shape
        if hasattr(batch, 'y'):
            y = batch.y
            if y.dim() == 1:
                y = y.unsqueeze(0) if out.shape[0] == 1 else y.view(out.shape)
        else:
            continue
        
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_tabular(model, loader, criterion, device):
    """Evaluate tabular model."""
    model.eval()
    total_loss = 0
    n_batches = 0
    all_preds, all_labels = [], []
    
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        out = model(X)
        loss = criterion(out, y)
        total_loss += loss.item()
        n_batches += 1
        all_preds.append(out.cpu().numpy())
        all_labels.append(y.cpu().numpy())
    
    preds = np.concatenate(all_preds) if all_preds else np.array([])
    labels = np.concatenate(all_labels) if all_labels else np.array([])
    
    return total_loss / max(n_batches, 1), preds, labels


@torch.no_grad()
def evaluate_graph(model, loader, criterion, device):
    """Evaluate graph model."""
    model.eval()
    total_loss = 0
    n_batches = 0
    all_preds, all_labels = [], []
    
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        
        if hasattr(batch, 'y'):
            y = batch.y
            if y.dim() == 1:
                y = y.unsqueeze(0) if out.shape[0] == 1 else y.view(out.shape)
        else:
            continue
        
        loss = criterion(out, y)
        total_loss += loss.item()
        n_batches += 1
        all_preds.append(out.cpu().numpy())
        all_labels.append(y.cpu().numpy())
    
    preds = np.concatenate(all_preds) if all_preds else np.array([])
    labels = np.concatenate(all_labels) if all_labels else np.array([])
    
    return total_loss / max(n_batches, 1), preds, labels


# =============================================================================
# DATASET CREATION
# =============================================================================

class GraphDataset(torch.utils.data.Dataset):
    """Simple dataset for graphs with targets."""
    def __init__(self, graphs: List[Data], targets: np.ndarray):
        self.graphs = graphs
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        graph = self.graphs[idx].clone()
        graph.y = self.targets[idx]
        return graph


def create_fold_loaders_tabular(X_train, X_val, y_train, y_val, batch_size, normalize=True):
    """Create data loaders for tabular data."""
    scalers = {}
    
    if normalize:
        scaler_X = StandardScaler()
        X_train = scaler_X.fit_transform(X_train)
        X_val = scaler_X.transform(X_val)
        scalers['X'] = scaler_X
        
        scaler_y = StandardScaler()
        y_train = scaler_y.fit_transform(y_train)
        y_val = scaler_y.transform(y_val)
        scalers['y'] = scaler_y
    
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, scalers


def create_fold_loaders_graph(graphs_train, graphs_val, y_train, y_val, batch_size, normalize=True):
    """Create data loaders for graph data."""
    scalers = {}
    
    if normalize:
        scaler_y = StandardScaler()
        y_train = scaler_y.fit_transform(y_train)
        y_val = scaler_y.transform(y_val)
        scalers['y'] = scaler_y
    
    train_ds = GraphDataset(graphs_train, y_train)
    val_ds = GraphDataset(graphs_val, y_val)
    
    train_loader = PyGDataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = PyGDataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, scalers


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def train_fold(fold_idx, train_idx, val_idx, data, config, device, target_names, 
               writer=None, output_dir=None):
    """Train and evaluate one fold with full logging."""
    
    n_folds = config['data']['cv_folds']
    logger.info(f"\n{'='*60}")
    logger.info(f"FOLD {fold_idx + 1}/{n_folds} - Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")
    logger.info(f"{'='*60}")
    
    input_type = config['input_type']
    model_type = config['model']['type']
    batch_size = config['training']['batch_size']
    normalize = config['data']['normalize_features']
    
    if input_type == "spectral":
        X, y, edge_index = data['X'], data['y'], data.get('edge_index')
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        if model_type == "gnn" and edge_index is not None:
            # Build graphs from spectral features
            from data.dataset import ExperienceDataset
            train_ds = ExperienceDataset(X_train, y_train, edge_index, num_regions=90, use_graph=True)
            val_ds = ExperienceDataset(X_val, y_val, edge_index, num_regions=90, use_graph=True)
            
            # Normalize targets separately
            scaler_y = StandardScaler()
            y_train_norm = scaler_y.fit_transform(y_train)
            y_val_norm = scaler_y.transform(y_val)
            
            train_ds = ExperienceDataset(X_train, y_train_norm, edge_index, num_regions=90, use_graph=True)
            val_ds = ExperienceDataset(X_val, y_val_norm, edge_index, num_regions=90, use_graph=True)
            
            train_loader = PyGDataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = PyGDataLoader(val_ds, batch_size=batch_size, shuffle=False)
            scalers = {'y': scaler_y}
            is_graph = True
            
            sample = train_ds[0]
            num_node_features = sample.x.shape[1]
            input_dim = None
        else:
            train_loader, val_loader, scalers = create_fold_loaders_tabular(
                X_train, X_val, y_train, y_val, batch_size, normalize
            )
            is_graph = False
            input_dim = X_train.shape[1]
            num_node_features = None
    
    elif input_type == "graphs":
        graphs, y = data['graphs'], data['y']
        graphs_train = [graphs[i] for i in train_idx]
        graphs_val = [graphs[i] for i in val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        if len(graphs_train) == 0:
            raise ValueError(f"No training graphs available for fold {fold_idx}")
        
        train_loader, val_loader, scalers = create_fold_loaders_graph(
            graphs_train, graphs_val, y_train, y_val, batch_size, normalize
        )
        is_graph = True
        
        sample = graphs_train[0]
        num_node_features = sample.x.shape[1]
        num_edge_features = sample.edge_attr.shape[1] if hasattr(sample, 'edge_attr') and sample.edge_attr is not None else 1
        input_dim = None
    
    # Create model
    model = create_model_from_config(
        config, 
        input_dim=input_dim, 
        num_node_features=num_node_features,
        num_edge_features=num_edge_features if input_type == "graphs" else 1
    )
    model = model.to(device)
    
    # Loss
    loss_type = config['training']['loss']
    if loss_type == 'mse':
        criterion = nn.MSELoss()
    elif loss_type == 'l1':
        criterion = nn.L1Loss()
    elif loss_type == 'huber':
        criterion = nn.HuberLoss(delta=config['training']['huber_delta'])
    else:
        criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay'])
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        patience=int(config['training']['scheduler']['patience']),
        factor=float(config['training']['scheduler']['factor']),
        min_lr=float(config['training']['scheduler']['min_lr'])
    )
    
    # Select train/eval functions
    train_fn = train_epoch_graph if is_graph else train_epoch_tabular
    eval_fn = evaluate_graph if is_graph else evaluate_tabular
    
    # Training loop with history
    best_val_loss = float('inf')
    best_val_pearson = -float('inf')
    patience_counter = 0
    best_preds, best_labels = None, None
    best_model_state = None
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_pearson': [],
        'val_mse': [],
        'val_r2': []
    }
    
    for epoch in range(1, config['training']['num_epochs'] + 1):
        train_loss = train_fn(model, train_loader, optimizer, criterion, device)
        val_loss, val_preds, val_labels = eval_fn(model, val_loader, criterion, device)
        
        # Inverse transform
        if 'y' in scalers and len(val_preds) > 0:
            val_preds_orig = scalers['y'].inverse_transform(val_preds)
            val_labels_orig = scalers['y'].inverse_transform(val_labels)
        else:
            val_preds_orig = val_preds
            val_labels_orig = val_labels
        
        if len(val_preds_orig) > 0:
            metrics = compute_metrics(val_labels_orig, val_preds_orig, target_names)
        else:
            metrics = {'mean_pearson': 0, 'mse': 0, 'r2': 0}
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_pearson'].append(metrics.get('mean_pearson', 0))
        history['val_mse'].append(metrics.get('mse', 0))
        history['val_r2'].append(metrics.get('r2', 0))
        
        # Log to TensorBoard
        if writer is not None:
            current_lr = optimizer.param_groups[0]['lr']
            log_epoch_metrics(writer, epoch, train_loss, val_loss, metrics, current_lr, fold=fold_idx)
            
            # Log per-target metrics every 10 epochs
            if epoch % 10 == 0 and 'per_target' in metrics:
                log_per_target_metrics(writer, epoch, metrics['per_target'], fold=fold_idx)
        
        scheduler.step(val_loss)
        
        # Check for improvement
        improved = val_loss < best_val_loss - float(config['training']['early_stopping']['min_delta'])
        
        # Log every 5 epochs for clearer progress
        if epoch % 5 == 0 or epoch == 1:
            lr_current = optimizer.param_groups[0]['lr']
            improvement_marker = " *" if improved else ""
            logger.info(f"  Epoch {epoch:3d}/{config['training']['num_epochs']} | "
                       f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                       f"r: {metrics['mean_pearson']:.3f} | R²: {metrics.get('r2', 0):.3f} | "
                       f"LR: {lr_current:.2e}{improvement_marker}")
        
        # Early stopping based on validation loss
        if improved:
            best_val_loss = val_loss
            best_val_pearson = metrics.get('mean_pearson', 0)
            patience_counter = 0
            best_preds = val_preds_orig.copy() if len(val_preds_orig) > 0 else None
            best_labels = val_labels_orig.copy() if len(val_labels_orig) > 0 else None
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        if patience_counter >= int(config['training']['early_stopping']['patience']):
            logger.info(f"  >> Early stopping at epoch {epoch} (no improvement for {patience_counter} epochs)")
            break
    
    # Final metrics
    if best_preds is not None and best_labels is not None:
        final_metrics = compute_metrics(best_labels, best_preds, target_names)
    else:
        final_metrics = {'mse': 0, 'mae': 0, 'r2': 0, 'mean_pearson': 0, 'std_pearson': 0}
    
    logger.info(f"\n  >> Fold {fold_idx + 1} DONE: MSE={final_metrics['mse']:.4f}, MAE={final_metrics['mae']:.4f}, "
               f"R²={final_metrics['r2']:.4f}, r={final_metrics['mean_pearson']:.4f}")
    
    # Save fold visualizations
    if output_dir is not None and best_preds is not None:
        fold_dir = Path(output_dir) / f'fold_{fold_idx}'
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        save_all_visualizations(
            best_labels, best_preds, target_names, history,
            fold_dir, prefix=f'fold{fold_idx}'
        )
        
        # Save best model
        if best_model_state is not None:
            torch.save({
                'model_state_dict': best_model_state,
                'metrics': final_metrics,
                'fold': fold_idx
            }, fold_dir / 'best_model.pt')
    
    return {
        'fold': fold_idx,
        'metrics': final_metrics,
        'predictions': best_preds,
        'labels': best_labels,
        'val_indices': val_idx,
        'history': history
    }


def main(config_path: str):
    """Main training function with k-fold CV and full logging."""
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    set_seed(config['seed'])
    device = get_device(config)
    
    output_dir = Path(config['paths']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tensorboard_path = config['paths'].get('tensorboard')
    if tensorboard_path:
        tensorboard_dir = Path(tensorboard_path)
    else:
        tensorboard_dir = output_dir / 'runs'
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    
    input_type = config['input_type']
    
    logger.info("\n" + "="*60)
    logger.info(f"EXPERIENCE PREDICTOR - {input_type.upper()} INPUT")
    logger.info("="*60)
    logger.info(f"  Device:     {device}")
    logger.info(f"  Model:      {config['model']['type'].upper()}")
    if config['model']['type'] == 'gnn':
        logger.info(f"  GNN Type:   {config['model']['gnn']['conv_type']}")
    logger.info(f"  Epochs:     {config['training']['num_epochs']}")
    logger.info(f"  Batch Size: {config['training']['batch_size']}")
    logger.info(f"  LR:         {config['training']['learning_rate']}")
    logger.info(f"  CV Folds:   {config['data']['cv_folds']}")
    logger.info(f"  Seed:       {config['seed']}")
    logger.info(f"  Output:     {output_dir}")
    
    # Setup TensorBoard
    writer = None
    if config.get('logging', {}).get('tensorboard', True):
        writer = setup_tensorboard(tensorboard_dir)
    
    # Load data
    force_rebuild = config.get('force_rebuild_cache', False)
    num_workers = config.get('num_workers', 4)
    
    logger.info("\nLoading data...")
    if input_type == "spectral":
        X, y, target_names, edge_index = load_spectral_data(config)
        data = {'X': X, 'y': y, 'edge_index': edge_index}
        n_samples = X.shape[0]
        logger.info(f"  Spectral features: {X.shape[1]}")
    elif input_type == "graphs":
        logger.info(f"  Loading graphs from {config['paths']['phases_dir']}...")
        graphs, y, target_names, subject_ids = load_graph_data(
            config, force_rebuild=force_rebuild, num_workers=num_workers
        )
        data = {'graphs': graphs, 'y': y, 'subject_ids': subject_ids}
        n_samples = len(graphs)
        if graphs:
            logger.info(f"  Graph nodes: {graphs[0].x.shape[0]}, Node features: {graphs[0].x.shape[1]}")
    else:
        raise ValueError(f"Unknown input_type: {input_type}")
    
    # Cross-validation setup
    n_folds = int(config['data']['cv_folds'])
    
    logger.info(f"\nData loaded: {n_samples} samples, {len(target_names)} targets")
    logger.info(f"Starting {n_folds}-fold cross-validation...\n")
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=int(config['data']['random_state']))
    
    all_results = []
    all_predictions = []
    all_labels = []
    indices = np.arange(n_samples)
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
        result = train_fold(
            fold_idx, train_idx, val_idx, data, config, device, target_names,
            writer=writer, output_dir=output_dir
        )
        all_results.append(result)
        
        if result['predictions'] is not None:
            all_predictions.append(result['predictions'])
            all_labels.append(result['labels'])
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("CROSS-VALIDATION SUMMARY")
    logger.info("="*60)
    
    cv_metrics = {k: np.mean([r['metrics'][k] for r in all_results]) 
                  for k in ['mse', 'mae', 'r2', 'mean_pearson']}
    cv_std = {k: np.std([r['metrics'][k] for r in all_results]) 
              for k in ['mse', 'mae', 'r2', 'mean_pearson']}
    
    logger.info(f"MSE:          {cv_metrics['mse']:.4f} ± {cv_std['mse']:.4f}")
    logger.info(f"MAE:          {cv_metrics['mae']:.4f} ± {cv_std['mae']:.4f}")
    logger.info(f"R²:           {cv_metrics['r2']:.4f} ± {cv_std['r2']:.4f}")
    logger.info(f"Mean Pearson: {cv_metrics['mean_pearson']:.4f} ± {cv_std['mean_pearson']:.4f}")
    
    # Per-target summary
    logger.info("\nPer-Target Results:")
    per_target_results = {}
    for target_name in target_names:
        pearson_values = []
        for r in all_results:
            if 'per_target' in r['metrics'] and target_name in r['metrics']['per_target']:
                pearson_values.append(r['metrics']['per_target'][target_name]['pearson_r'])
        
        if pearson_values:
            mean_r = np.mean(pearson_values)
            std_r = np.std(pearson_values)
            per_target_results[target_name] = {'mean_r': mean_r, 'std_r': std_r}
            logger.info(f"  {target_name:25s}: r = {mean_r:.3f} ± {std_r:.3f}")
    
    # Find best fold
    best_fold_idx = max(range(len(all_results)), 
                       key=lambda i: all_results[i]['metrics'].get('mean_pearson', 0))
    best_fold = all_results[best_fold_idx]
    logger.info(f"\nBest Fold: {best_fold_idx + 1} (r = {best_fold['metrics']['mean_pearson']:.4f})")
    
    # Create final visualizations combining all folds
    if all_predictions:
        combined_preds = np.vstack(all_predictions)
        combined_labels = np.vstack(all_labels)
        
        save_all_visualizations(
            combined_labels, combined_preds, target_names,
            {}, output_dir, prefix='combined'
        )
    
    # Save results
    results = {
        'input_type': input_type,
        'model_type': config['model']['type'],
        'cv_metrics': cv_metrics,
        'cv_std': cv_std,
        'per_target': per_target_results,
        'n_folds': n_folds,
        'n_samples': n_samples,
        'best_fold': best_fold_idx,
        'best_fold_metrics': best_fold['metrics']
    }
    
    save_results_json(results, output_dir / 'cv_results.json')
    
    # Create experiment summary
    create_experiment_summary(all_results, target_names, output_dir)
    
    # Log final results to TensorBoard
    if writer is not None:
        log_final_results(writer, {'mean_pearson': cv_metrics['mean_pearson'],
                                   'mse': cv_metrics['mse'],
                                   'r2': cv_metrics['r2']})
        writer.close()
    
    logger.info(f"\n{'='*60}")
    logger.info("RESULTS SAVED:")
    logger.info(f"  - Results: {output_dir / 'cv_results.json'}")
    logger.info(f"  - Summary: {output_dir / 'experiment_summary.json'}")
    logger.info(f"  - Visualizations: {output_dir / 'visualizations'}")
    logger.info(f"  - TensorBoard: {tensorboard_dir}")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Experience Predictor')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of workers for data loading (default: auto)')
    parser.add_argument('--force-rebuild', action='store_true',
                       help='Force rebuild cache')
    args = parser.parse_args()
    
    # Load config and override with CLI args
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.workers is not None:
        config['num_workers'] = args.workers
    if args.force_rebuild:
        config['force_rebuild_cache'] = True
    
    # Save modified config to temp file and run
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        temp_config = f.name
    
    main(temp_config)
    
    # Cleanup
    Path(temp_config).unlink()
