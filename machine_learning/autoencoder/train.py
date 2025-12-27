#!/usr/bin/env python3
"""
Training script for Variational Autoencoder on EEG synchronization data.

This script handles:
- Loading and preprocessing graph datasets
- VAE model initialization and training
- β-VAE annealing schedule
- TensorBoard logging (losses, latent space, reconstructions, attention)
- Checkpointing and early stopping
"""

import argparse
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple
import logging
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from data import create_dataset_from_config
from models import create_vae_from_config
from utils import (
    setup_logging, log_metrics, 
    plot_training_curves, plot_latent_space, 
    plot_reconstructions, plot_kl_per_dimension,
    log_latent_to_tensorboard, save_latent_embeddings
)

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def normalize_graph_features(graphs, scaler=None, clip_percentile=99.5):
    """
    Normalize node features across all graphs using RobustScaler.
    
    Uses RobustScaler (median/IQR based) which is robust to outliers.
    Also clips extreme outliers before scaling.
    
    Args:
        graphs: List of PyG Data objects
        scaler: If provided, use this fitted scaler. Otherwise fit new one.
        clip_percentile: Percentile for clipping outliers (default 99.5)
        
    Returns:
        Normalized graphs and fitted scaler
    """
    # Collect all node features
    all_features = np.vstack([g.x.numpy() for g in graphs])
    
    if scaler is None:
        # Compute clipping bounds from training data
        lower = np.percentile(all_features, 100 - clip_percentile, axis=0)
        upper = np.percentile(all_features, clip_percentile, axis=0)
        
        # Clip outliers
        all_features_clipped = np.clip(all_features, lower, upper)
        
        # Fit RobustScaler (uses median and IQR, robust to remaining outliers)
        scaler = RobustScaler()
        scaler.fit(all_features_clipped)
        
        # Store clipping bounds in scaler for later use
        scaler.clip_lower_ = lower
        scaler.clip_upper_ = upper
    
    # Transform all graphs
    for g in graphs:
        features = g.x.numpy()
        # Clip using stored bounds
        features_clipped = np.clip(features, scaler.clip_lower_, scaler.clip_upper_)
        # Scale
        features_scaled = scaler.transform(features_clipped)
        g.x = torch.tensor(features_scaled, dtype=torch.float32)
    
    return graphs, scaler


def subsample_dataset(graphs, fraction=1.0, seed=42):
    """
    Randomly subsample a dataset.
    
    Args:
        graphs: List of graphs
        fraction: Fraction to keep (0.0-1.0)
        seed: Random seed
        
    Returns:
        Subsampled list
    """
    if fraction >= 1.0:
        return graphs
    
    random.seed(seed)
    n_keep = int(len(graphs) * fraction)
    return random.sample(graphs, n_keep)


def get_device(config: Dict[str, Any]) -> torch.device:
    """Get torch device from config."""
    device_name = config['device']
    if device_name == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device_name = 'cpu'
    return torch.device(device_name)


def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """Create optimizer from config."""
    train_config = config['training']
    optimizer_name = train_config['optimizer'].lower()
    lr = float(train_config['learning_rate'])
    weight_decay = float(train_config['weight_decay'])
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer


def create_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any]):
    """Create learning rate scheduler from config."""
    scheduler_config = config['training']['scheduler']
    
    scheduler_type = scheduler_config.get('type', None)
    if scheduler_type is None or scheduler_type == 'none':
        return None
    
    scheduler_type = scheduler_type.lower()
    
    if scheduler_type == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=int(scheduler_config['patience']),
            factor=float(scheduler_config['factor']),
            min_lr=float(scheduler_config['min_lr'])
        )
    elif scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config['training']['num_epochs']),
            eta_min=float(scheduler_config['min_lr'])
        )
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(scheduler_config.get('step_size', 30)),
            gamma=float(scheduler_config['factor'])
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
    
    return scheduler


def get_beta(epoch: int, config: Dict[str, Any]) -> float:
    """
    Get beta value for β-VAE with annealing.
    
    Args:
        epoch: Current epoch (1-indexed)
        config: Configuration dictionary
        
    Returns:
        Beta value
    """
    kl_config = config['loss']['kl']
    
    if not kl_config['annealing']['enabled']:
        return kl_config['weight']
    
    anneal_config = kl_config['annealing']
    start_beta = anneal_config['start']
    end_beta = anneal_config['end']
    anneal_epochs = anneal_config['epochs']
    anneal_type = anneal_config.get('type', 'linear')
    
    if epoch >= anneal_epochs:
        return end_beta
    
    progress = epoch / anneal_epochs
    
    if anneal_type == 'linear':
        return start_beta + (end_beta - start_beta) * progress
    elif anneal_type == 'cosine':
        return start_beta + (end_beta - start_beta) * (1 - np.cos(np.pi * progress)) / 2
    elif anneal_type == 'cyclical':
        # Cyclical annealing (4 cycles)
        num_cycles = 4
        cycle_progress = (progress * num_cycles) % 1.0
        return start_beta + (end_beta - start_beta) * cycle_progress
    else:
        return start_beta + (end_beta - start_beta) * progress


def train_epoch(model: nn.Module,
                loader: DataLoader,
                optimizer: optim.Optimizer,
                device: torch.device,
                config: Dict[str, Any],
                beta: float) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Returns:
        Dict with loss metrics
    """
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    total_recon_node = 0
    total_recon_edge = 0
    n_batches = 0
    
    free_bits = config['loss'].get('free_bits', 0.0)
    
    for batch_idx, data in enumerate(loader):
        data = data.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data, store_activations=False)
        
        # Compute loss
        losses = model.loss_function(output, data, beta=beta, free_bits=free_bits)
        loss = losses['loss']
        
        loss.backward()
        
        # Gradient clipping
        grad_clip_config = config['training'].get('gradient_clipping', {})
        if isinstance(grad_clip_config, dict) and grad_clip_config.get('enabled', False):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip_config['max_norm']
            )
        
        optimizer.step()
        
        total_loss += losses['loss'].item()
        total_recon_loss += losses['recon_loss'].item()
        total_kl_loss += losses['kl_loss'].item()
        total_recon_node += losses['recon_loss_node'].item()
        total_recon_edge += losses['recon_loss_edge'].item()
        n_batches += 1
    
    return {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon_loss / n_batches,
        'kl_loss': total_kl_loss / n_batches,
        'recon_node': total_recon_node / n_batches,
        'recon_edge': total_recon_edge / n_batches
    }


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             device: torch.device,
             config: Dict[str, Any],
             beta: float) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Evaluate model.
    
    Returns:
        Tuple of (metrics_dict, kl_per_dim)
    """
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    total_recon_node = 0
    total_recon_edge = 0
    n_batches = 0
    
    all_kl_per_dim = []
    
    free_bits = config['loss'].get('free_bits', 0.0)
    
    for data in loader:
        data = data.to(device)
        
        output = model(data, store_activations=False)
        losses = model.loss_function(output, data, beta=beta, free_bits=free_bits)
        
        total_loss += losses['loss'].item()
        total_recon_loss += losses['recon_loss'].item()
        total_kl_loss += losses['kl_loss'].item()
        total_recon_node += losses['recon_loss_node'].item()
        total_recon_edge += losses['recon_loss_edge'].item()
        all_kl_per_dim.append(losses['kl_per_dim'].cpu().numpy())
        n_batches += 1
    
    metrics = {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon_loss / n_batches,
        'kl_loss': total_kl_loss / n_batches,
        'recon_node': total_recon_node / n_batches,
        'recon_edge': total_recon_edge / n_batches
    }
    
    kl_per_dim = np.mean(all_kl_per_dim, axis=0)
    
    return metrics, kl_per_dim


def main(config_path: str, force_rebuild: bool = False,
         subsample_fraction: float = 1.0, normalize: bool = True,
         num_workers_override: int = None, dataset_workers_override: int = None):
    """Main training function.
    
    Args:
        config_path: Path to config YAML file
        force_rebuild: Force rebuild dataset cache
        subsample_fraction: Fraction of data to use (0.0-1.0)
        normalize: Whether to normalize features
        num_workers_override: Override num_workers for DataLoader
        dataset_workers_override: Override workers for dataset building
    """
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply worker overrides if provided (before logging setup)
    if num_workers_override is not None:
        config['num_workers'] = num_workers_override
    if dataset_workers_override is not None:
        config['dataset_workers'] = dataset_workers_override
    
    # Create output directories
    output_dir = Path(config['paths']['output_dir'])
    checkpoint_dir = Path(config['paths']['checkpoints'])
    tensorboard_dir = Path(config['paths']['tensorboard'])
    
    for dir_path in [output_dir, checkpoint_dir, tensorboard_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(config, output_dir / 'logs')
    logger.info("=" * 80)
    logger.info("Starting BrainStateVAE Training")
    logger.info("=" * 80)
    
    # Set seed
    set_seed(config['seed'])
    logger.info(f"Random seed set to {config['seed']}")
    
    # Get device
    device = get_device(config)
    logger.info(f"Using device: {device}")
    
    # ========================================================================
    # LOAD DATA
    # ========================================================================
    
    logger.info("Loading datasets...")
    
    dataset_workers = config.get('dataset_workers', None)
    logger.info(f"Using {dataset_workers} workers for dataset building")
    
    train_graphs, val_graphs, test_graphs = create_dataset_from_config(
        config, 
        force_rebuild=force_rebuild,
        num_workers=dataset_workers
    )
    
    logger.info(f"Original - Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")
    
    # Subsample if requested
    if subsample_fraction < 1.0:
        logger.info(f"Subsampling to {subsample_fraction*100:.0f}% of data...")
        train_graphs = subsample_dataset(train_graphs, subsample_fraction, config['seed'])
        val_graphs = subsample_dataset(val_graphs, subsample_fraction, config['seed'])
        test_graphs = subsample_dataset(test_graphs, subsample_fraction, config['seed'])
        logger.info(f"Subsampled - Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")
    
    # Normalize features
    if normalize:
        logger.info("Normalizing node features (RobustScaler + outlier clipping)...")
        # Fit on training data only
        train_graphs, scaler = normalize_graph_features(train_graphs, scaler=None)
        # Transform val/test with same scaler (including same clipping bounds)
        val_graphs, _ = normalize_graph_features(val_graphs, scaler=scaler)
        test_graphs, _ = normalize_graph_features(test_graphs, scaler=scaler)
        logger.info("  Features normalized (robust to outliers)")
    
    logger.info(f"Final - Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")
    
    # Create data loaders
    batch_size = config['training']['batch_size']
    num_workers = config.get('num_workers', 4)
    pin_memory = config.get('pin_memory', True)
    logger.info(f"DataLoader: batch_size={batch_size}, num_workers={num_workers}, pin_memory={pin_memory}")
    
    train_loader = DataLoader(
        train_graphs,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_graphs,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_graphs,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # ========================================================================
    # CREATE MODEL
    # ========================================================================
    
    # Infer feature dimensions from first graph
    sample_graph = train_graphs[0]
    num_node_features = sample_graph.x.shape[1]
    num_edge_features = sample_graph.edge_attr.shape[1] if hasattr(sample_graph, 'edge_attr') else 0
    num_graph_features = sample_graph.graph_attr.shape[0] if hasattr(sample_graph, 'graph_attr') else 0
    num_nodes = sample_graph.num_nodes
    
    logger.info(f"Node features: {num_node_features}")
    logger.info(f"Edge features: {num_edge_features}")
    logger.info(f"Graph features: {num_graph_features}")
    logger.info(f"Nodes per graph: {num_nodes}")
    
    model = create_vae_from_config(
        config,
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        num_graph_features=num_graph_features,
        num_nodes=num_nodes
    )
    model = model.to(device)
    
    # ========================================================================
    # TRAINING SETUP
    # ========================================================================
    
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    logger.info(f"Optimizer: {optimizer.__class__.__name__}")
    logger.info(f"Scheduler: {scheduler.__class__.__name__ if scheduler else None}")
    
    # TensorBoard
    writer = None
    if config['logging']['tensorboard']:
        writer = SummaryWriter(tensorboard_dir)
        logger.info(f"TensorBoard logging to: {tensorboard_dir}")
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    
    num_epochs = config['training']['num_epochs']
    early_stopping_patience = config['training']['early_stopping']['patience']
    early_stopping_delta = config['training']['early_stopping']['min_delta']
    save_frequency = config['logging']['save_frequency']
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'train_recon_loss': [],
        'train_kl_loss': [],
        'val_loss': [],
        'val_recon_loss': [],
        'val_kl_loss': [],
        'beta': [],
        'lr': []
    }
    
    class_names = config['data']['conditions']
    
    logger.info("Starting training...")
    
    for epoch in range(1, num_epochs + 1):
        # Get current beta
        beta = get_beta(epoch, config)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, config, beta
        )
        
        # Validate
        val_metrics, val_kl_per_dim = evaluate(
            model, val_loader, device, config, beta
        )
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_recon_loss'].append(train_metrics['recon_loss'])
        history['train_kl_loss'].append(train_metrics['kl_loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_recon_loss'].append(val_metrics['recon_loss'])
        history['val_kl_loss'].append(val_metrics['kl_loss'])
        history['beta'].append(beta)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Log to console (format used by eeg_viewer for real-time plotting)
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(
            f"Epoch {epoch:03d} [train] - Loss: {train_metrics['loss']:.4f} | "
            f"Recon: {train_metrics['recon_loss']:.4f} | KL: {train_metrics['kl_loss']:.4f}"
        )
        logger.info(
            f"Epoch {epoch:03d} [val]   - Loss: {val_metrics['loss']:.4f} | "
            f"Recon: {val_metrics['recon_loss']:.4f} | KL: {val_metrics['kl_loss']:.4f} | "
            f"β: {beta:.4f} | LR: {current_lr:.2e}"
        )
        
        # Log to TensorBoard
        if writer:
            # Losses
            writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            writer.add_scalar('Loss/train_recon', train_metrics['recon_loss'], epoch)
            writer.add_scalar('Loss/val_recon', val_metrics['recon_loss'], epoch)
            writer.add_scalar('Loss/train_kl', train_metrics['kl_loss'], epoch)
            writer.add_scalar('Loss/val_kl', val_metrics['kl_loss'], epoch)
            
            # Reconstruction components
            writer.add_scalar('Recon/train_node', train_metrics['recon_node'], epoch)
            writer.add_scalar('Recon/val_node', val_metrics['recon_node'], epoch)
            writer.add_scalar('Recon/train_edge', train_metrics['recon_edge'], epoch)
            writer.add_scalar('Recon/val_edge', val_metrics['recon_edge'], epoch)
            
            # Beta and LR
            writer.add_scalar('Schedule/beta', beta, epoch)
            writer.add_scalar('Schedule/learning_rate', optimizer.param_groups[0]['lr'], epoch)
            
            # KL per dimension
            if config['logging']['tensorboard_extras'].get('kl_per_dim', False):
                for i, kl_val in enumerate(val_kl_per_dim):
                    writer.add_scalar(f'KL_per_dim/dim_{i}', kl_val, epoch)
            
            # Latent space visualization (periodically)
            if epoch % save_frequency == 0:
                try:
                    log_latent_to_tensorboard(
                        writer, model, val_loader, device, epoch,
                        class_names, num_samples=500
                    )
                except Exception as e:
                    logger.warning(f"Failed to log latent space: {e}")
            
            # Save reconstructions every 10 epochs for visualization
            if epoch % 10 == 0 or epoch == 1:
                try:
                    recon_dir = output_dir / 'reconstructions'
                    recon_dir.mkdir(exist_ok=True)
                    
                    # Get batch for visualization
                    model.eval()
                    with torch.no_grad():
                        sample_batch = next(iter(val_loader))
                        sample_batch = sample_batch.to(device)
                        output = model(sample_batch)
                        
                        # Extract original and reconstructed node features
                        orig_nodes = sample_batch.x.cpu().numpy()
                        recon_nodes = output['x_recon'].cpu().numpy()
                        
                        # Limit size for storage
                        max_nodes = min(1000, orig_nodes.shape[0])
                        
                        # Save as compressed numpy
                        np.savez_compressed(
                            recon_dir / f'recon_epoch_{epoch:03d}.npz',
                            original=orig_nodes[:max_nodes],
                            reconstructed=recon_nodes[:max_nodes],
                            epoch=epoch
                        )
                        logger.info(f"Epoch {epoch:03d} [recon] - Saved {max_nodes} node features to {recon_dir}")
                except Exception as e:
                    logger.warning(f"Failed to save reconstructions: {e}")
        
        # Scheduler step
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
        
        # Save checkpoint
        if epoch % save_frequency == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'history': history,
                'config': config,
                'model_params': {
                    'num_node_features': num_node_features,
                    'num_edge_features': num_edge_features,
                    'num_graph_features': num_graph_features,
                    'num_nodes': num_nodes
                }
            }, checkpoint_path)
        
        # Save best model
        if val_metrics['loss'] < best_val_loss - early_stopping_delta:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            
            best_model_path = checkpoint_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'config': config,
                'model_params': {
                    'num_node_features': num_node_features,
                    'num_edge_features': num_edge_features,
                    'num_graph_features': num_graph_features,
                    'num_nodes': num_nodes
                }
            }, best_model_path)
            
            logger.info(f"✓ New best model saved! Val Loss: {val_metrics['loss']:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch} epochs")
            break
    
    # ========================================================================
    # FINAL EVALUATION
    # ========================================================================
    
    logger.info("=" * 80)
    logger.info("Training completed. Evaluating best model on test set...")
    
    # Load best model
    best_checkpoint = torch.load(checkpoint_dir / 'best_model.pt')
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    test_metrics, test_kl_per_dim = evaluate(
        model, test_loader, device, config, beta=1.0  # Full KL weight for final eval
    )
    
    logger.info(f"Test Loss:       {test_metrics['loss']:.4f}")
    logger.info(f"Test Recon Loss: {test_metrics['recon_loss']:.4f}")
    logger.info(f"Test KL Loss:    {test_metrics['kl_loss']:.4f}")
    
    # Save test results
    results = {
        'test_loss': float(test_metrics['loss']),
        'test_recon_loss': float(test_metrics['recon_loss']),
        'test_kl_loss': float(test_metrics['kl_loss']),
        'best_val_loss': float(best_val_loss),
        'best_epoch': int(best_checkpoint['epoch']),
        'latent_dim': config['model']['latent']['dim'],
        'num_gat_layers': config['model']['encoder']['num_gat_layers']
    }
    
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================
    
    viz_config = config['visualization']
    
    if viz_config['plot_training_curves']:
        plot_training_curves(
            history,
            output_dir / 'training_curves.png',
            title='BrainStateVAE Training Curves'
        )
        logger.info(f"✓ Saved training curves")
    
    # Plot latent space
    if viz_config['plot_latent_space']:
        logger.info("Generating latent space visualization...")
        
        # Collect latents from test set
        all_latents = []
        all_labels = []
        
        model.eval()
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                latent = model.get_latent(data, use_mean=True)
                all_latents.append(latent.cpu().numpy())
                all_labels.extend([data.y[i].item() for i in range(data.num_graphs)])
        
        latents = np.concatenate(all_latents, axis=0)
        labels = np.array(all_labels)
        
        plot_latent_space(
            latents, labels, class_names,
            output_dir / 'latent_space.png',
            method=viz_config.get('latent_method', 'tsne'),
            title='VAE Latent Space'
        )
        logger.info(f"✓ Saved latent space visualization")
    
    # Plot KL per dimension
    plot_kl_per_dimension(
        test_kl_per_dim,
        output_dir / 'kl_per_dimension.png',
        title='KL Divergence per Latent Dimension'
    )
    logger.info(f"✓ Saved KL per dimension plot")
    
    # ========================================================================
    # SAVE EMBEDDINGS FOR CLUSTERING
    # ========================================================================
    
    logger.info("=" * 80)
    logger.info("Saving embeddings for clustering analysis...")
    
    embeddings_dir = output_dir / 'embeddings'
    embeddings_dir.mkdir(exist_ok=True)
    
    # Save all splits
    for split_name, split_loader in [
        ('train', train_loader),
        ('val', val_loader),
        ('test', test_loader)
    ]:
        save_latent_embeddings(
            model, split_loader, device,
            embeddings_dir / f'{split_name}_embeddings.pkl',
            class_names
        )
        logger.info(f"✓ Saved {split_name} embeddings")
    
    # Log final metrics to TensorBoard
    if writer:
        writer.add_scalar('Test/loss', test_metrics['loss'], best_checkpoint['epoch'])
        writer.add_scalar('Test/recon_loss', test_metrics['recon_loss'], best_checkpoint['epoch'])
        writer.add_scalar('Test/kl_loss', test_metrics['kl_loss'], best_checkpoint['epoch'])
        writer.close()
    
    logger.info("=" * 80)
    logger.info("All done! Results saved to:")
    logger.info(f"  - Model: {checkpoint_dir / 'best_model.pt'}")
    logger.info(f"  - Results: {output_dir / 'test_results.json'}")
    logger.info(f"  - Visualizations: {output_dir}")
    logger.info(f"  - Embeddings: {embeddings_dir}")
    logger.info("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train BrainStateVAE model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--force-rebuild', action='store_true',
                       help='Force rebuild dataset even if cache exists')
    parser.add_argument('--subsample', type=float, default=1.0,
                       help='Fraction of dataset to use (0.0-1.0). Default: 1.0 (all)')
    parser.add_argument('--no-normalize', action='store_true',
                       help='Disable feature normalization')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of workers for data loading. Overrides config value.')
    parser.add_argument('--dataset-workers', type=int, default=None,
                       help='Number of workers for dataset building. Overrides config value.')
    
    args = parser.parse_args()
    
    main(args.config, args.force_rebuild, 
         subsample_fraction=args.subsample,
         normalize=not args.no_normalize,
         num_workers_override=args.workers,
         dataset_workers_override=args.dataset_workers)

