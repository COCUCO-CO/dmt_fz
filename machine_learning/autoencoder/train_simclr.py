#!/usr/bin/env python3
"""
Training script for SimCLR contrastive learning on EEG graph data.

This script handles:
- Loading and preprocessing graph datasets
- SimCLR model with graph augmentations
- NT-Xent contrastive loss
- TensorBoard logging (losses, embeddings, t-SNE)
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
from models.simclr_model import create_simclr_from_config
from utils import setup_logging, plot_training_curves

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
    """Normalize node features using RobustScaler."""
    all_features = np.vstack([g.x.numpy() for g in graphs])
    
    if scaler is None:
        lower = np.percentile(all_features, 100 - clip_percentile, axis=0)
        upper = np.percentile(all_features, clip_percentile, axis=0)
        all_features_clipped = np.clip(all_features, lower, upper)
        
        scaler = RobustScaler()
        scaler.fit(all_features_clipped)
        scaler.clip_lower_ = lower
        scaler.clip_upper_ = upper
    
    for g in graphs:
        features = g.x.numpy()
        features_clipped = np.clip(features, scaler.clip_lower_, scaler.clip_upper_)
        features_scaled = scaler.transform(features_clipped)
        g.x = torch.tensor(features_scaled, dtype=torch.float32)
    
    return graphs, scaler


def subsample_dataset(graphs, fraction=1.0, seed=42):
    """Randomly subsample a dataset."""
    if fraction >= 1.0:
        return graphs
    
    random.seed(seed)
    n_keep = int(len(graphs) * fraction)
    return random.sample(graphs, n_keep)


def get_device(config: Dict[str, Any]) -> torch.device:
    """Get torch device from config."""
    device_name = config.get('device', 'cuda')
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
    elif optimizer_name == 'lars':
        # LARS optimizer (used in original SimCLR)
        # Fallback to AdamW if lars not available
        try:
            from torch_optimizer import LARS
            optimizer = LARS(model.parameters(), lr=lr, weight_decay=weight_decay)
        except ImportError:
            logger.warning("LARS not available, using AdamW")
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer


def create_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any]):
    """Create learning rate scheduler."""
    scheduler_config = config['training'].get('scheduler', {})
    scheduler_type = scheduler_config.get('type', 'cosine')
    
    if scheduler_type is None or scheduler_type == 'none':
        return None
    
    scheduler_type = scheduler_type.lower()
    
    if scheduler_type == 'cosine':
        # Cosine annealing with warmup
        warmup_epochs = scheduler_config.get('warmup_epochs', 10)
        total_epochs = config['training']['num_epochs']
        
        # Ensure we have at least 1 epoch after warmup to avoid division by zero
        decay_epochs = max(1, total_epochs - warmup_epochs)
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Linear warmup
                return (epoch + 1) / warmup_epochs if warmup_epochs > 0 else 1.0
            else:
                # Cosine decay
                progress = (epoch - warmup_epochs) / decay_epochs
                return 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(scheduler_config.get('step_size', 30)),
            gamma=float(scheduler_config.get('factor', 0.5))
        )
    
    elif scheduler_type == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=int(scheduler_config.get('patience', 10)),
            factor=float(scheduler_config.get('factor', 0.5)),
            min_lr=float(scheduler_config.get('min_lr', 1e-6))
        )
    else:
        return None
    
    return scheduler


def train_epoch(model: nn.Module,
                loader: DataLoader,
                optimizer: optim.Optimizer,
                device: torch.device,
                config: Dict[str, Any]) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0
    
    for batch in loader:
        batch = batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with augmentation
        output = model(batch, augment=True)
        loss = output['loss']
        
        loss.backward()
        
        # Gradient clipping
        grad_clip_config = config['training'].get('gradient_clipping', {})
        if isinstance(grad_clip_config, dict) and grad_clip_config.get('enabled', False):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip_config['max_norm']
            )
        
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return {'loss': total_loss / n_batches}


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             device: torch.device) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()
    total_loss = 0
    n_batches = 0
    
    for batch in loader:
        batch = batch.to(device)
        output = model(batch, augment=True)
        total_loss += output['loss'].item()
        n_batches += 1
    
    return {'loss': total_loss / n_batches}


@torch.no_grad()
def collect_embeddings(model: nn.Module,
                       loader: DataLoader,
                       device: torch.device,
                       max_samples: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    """Collect embeddings and labels from the model."""
    model.eval()
    all_embeddings = []
    all_labels = []
    n_samples = 0
    
    for batch in loader:
        if n_samples >= max_samples:
            break
        
        batch = batch.to(device)
        embeddings = model.get_embeddings(batch)
        
        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.extend([batch.y[i].item() for i in range(batch.num_graphs)])
        n_samples += batch.num_graphs
    
    embeddings = np.concatenate(all_embeddings, axis=0)[:max_samples]
    labels = np.array(all_labels)[:max_samples]
    
    return embeddings, labels


def plot_latent_space_simclr(embeddings: np.ndarray,
                              labels: np.ndarray,
                              class_names: list,
                              save_path: Path,
                              method: str = 'tsne',
                              title: str = 'SimCLR Embedding Space'):
    """Plot t-SNE or UMAP of embeddings."""
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    
    # Reduce dimensions
    if method == 'tsne':
        perplexity = min(30, len(embeddings) - 1)
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    else:
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
        except ImportError:
            reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    
    coords = reducer.fit_transform(embeddings)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Colors for classes
    colors = plt.cm.Set1(np.linspace(0, 1, len(class_names)))
    
    for i, class_name in enumerate(class_names):
        mask = labels == i
        if mask.sum() > 0:
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=[colors[i]], label=class_name,
                alpha=0.6, s=20
            )
    
    ax.legend(loc='best')
    ax.set_title(title)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def log_embeddings_to_tensorboard(writer: SummaryWriter,
                                  model: nn.Module,
                                  loader: DataLoader,
                                  device: torch.device,
                                  epoch: int,
                                  class_names: list,
                                  num_samples: int = 500):
    """Log embeddings to TensorBoard projector."""
    embeddings, labels = collect_embeddings(model, loader, device, num_samples)
    
    # Create metadata for projector
    metadata = [class_names[l] for l in labels]
    
    writer.add_embedding(
        torch.tensor(embeddings),
        metadata=metadata,
        global_step=epoch,
        tag='embeddings'
    )


def save_embeddings_for_clustering(model: nn.Module,
                                    loader: DataLoader,
                                    device: torch.device,
                                    save_path: Path,
                                    class_names: list):
    """Save embeddings in format compatible with clustering pipeline."""
    import pickle
    
    model.eval()
    
    # Organize by condition
    embeddings_by_condition = {name: [] for name in class_names}
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            embeddings = model.get_embeddings(batch)
            
            for i in range(batch.num_graphs):
                label = batch.y[i].item()
                class_name = class_names[label]
                embeddings_by_condition[class_name].append(
                    embeddings[i].cpu().numpy()
                )
    
    # Convert to format expected by clustering.py
    # {condition: {band: [arrays]}}
    # Since SimCLR doesn't separate by band, use 'all' as band
    output = {}
    for condition, embs in embeddings_by_condition.items():
        output[condition] = {'all': embs}
    
    with open(save_path, 'wb') as f:
        pickle.dump(output, f)


def main(config_path: str, 
         force_rebuild: bool = False,
         subsample_fraction: float = 1.0, 
         normalize: bool = True,
         num_workers_override: int = None, 
         dataset_workers_override: int = None,
         max_files: int = None):
    """Main training function."""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply worker overrides
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
    logger.info("Starting BrainStateSimCLR Training")
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
    
    dataset_workers = config.get('dataset_workers', 4)
    train_graphs, val_graphs, test_graphs = create_dataset_from_config(
        config, 
        force_rebuild=force_rebuild,
        num_workers=dataset_workers,
        max_files=max_files
    )
    
    logger.info(f"Original - Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")
    
    # Subsample
    if subsample_fraction < 1.0:
        logger.info(f"Subsampling to {subsample_fraction*100:.0f}%...")
        train_graphs = subsample_dataset(train_graphs, subsample_fraction, config['seed'])
        val_graphs = subsample_dataset(val_graphs, subsample_fraction, config['seed'])
        test_graphs = subsample_dataset(test_graphs, subsample_fraction, config['seed'])
        logger.info(f"Subsampled - Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")
    
    # Normalize
    if normalize:
        logger.info("Normalizing features...")
        train_graphs, scaler = normalize_graph_features(train_graphs)
        val_graphs, _ = normalize_graph_features(val_graphs, scaler)
        test_graphs, _ = normalize_graph_features(test_graphs, scaler)
    
    # Create data loaders
    batch_size = config['training']['batch_size']
    num_workers = config.get('num_workers', 4)
    
    train_loader = DataLoader(
        train_graphs, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_graphs, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_graphs, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    # ========================================================================
    # CREATE MODEL
    # ========================================================================
    
    sample_graph = train_graphs[0]
    num_node_features = sample_graph.x.shape[1]
    num_edge_features = sample_graph.edge_attr.shape[1] if hasattr(sample_graph, 'edge_attr') else 0
    
    logger.info(f"Node features: {num_node_features}")
    logger.info(f"Edge features: {num_edge_features}")
    
    model = create_simclr_from_config(config, num_node_features, num_edge_features)
    model = model.to(device)
    
    # ========================================================================
    # TRAINING SETUP
    # ========================================================================
    
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    logger.info(f"Optimizer: {optimizer.__class__.__name__}")
    logger.info(f"Scheduler: {scheduler.__class__.__name__ if scheduler else 'None'}")
    
    # TensorBoard
    writer = None
    if config.get('logging', {}).get('tensorboard', True):
        writer = SummaryWriter(tensorboard_dir)
        logger.info(f"TensorBoard: {tensorboard_dir}")
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    
    num_epochs = config['training']['num_epochs']
    early_stopping_patience = config['training'].get('early_stopping', {}).get('patience', 30)
    early_stopping_delta = config['training'].get('early_stopping', {}).get('min_delta', 0.0001)
    save_frequency = config.get('logging', {}).get('save_frequency', 10)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    
    class_names = config['data']['conditions']
    
    logger.info("Starting training...")
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, config)
        
        # Validate
        val_metrics = evaluate(model, val_loader, device)
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Log
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(
            f"Epoch {epoch:03d} [train] - Loss: {train_metrics['loss']:.4f} | "
            f"[val] Loss: {val_metrics['loss']:.4f} | LR: {current_lr:.2e}"
        )
        
        # TensorBoard
        if writer:
            writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            writer.add_scalar('Schedule/learning_rate', current_lr, epoch)
            
            # Log embeddings periodically
            if epoch % save_frequency == 0:
                try:
                    log_embeddings_to_tensorboard(
                        writer, model, val_loader, device, epoch,
                        class_names, num_samples=500
                    )
                except Exception as e:
                    logger.warning(f"Failed to log embeddings: {e}")
        
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
                'model_type': 'simclr',
                'model_params': {
                    'num_node_features': num_node_features,
                    'num_edge_features': num_edge_features,
                    'num_nodes': sample_graph.num_nodes,
                }
            }, checkpoint_path)
        
        # Best model
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
                'model_type': 'simclr',
                'model_params': {
                    'num_node_features': num_node_features,
                    'num_edge_features': num_edge_features,
                    'num_nodes': sample_graph.num_nodes,
                }
            }, best_model_path)
            
            logger.info(f"✓ New best model! Val Loss: {val_metrics['loss']:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    # ========================================================================
    # FINAL EVALUATION
    # ========================================================================
    
    logger.info("=" * 80)
    logger.info("Training completed. Evaluating...")
    
    # Load best model
    best_checkpoint = torch.load(checkpoint_dir / 'best_model.pt')
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, device)
    logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
    
    # Save results
    results = {
        'test_loss': float(test_metrics['loss']),
        'best_val_loss': float(best_val_loss),
        'best_epoch': int(best_checkpoint['epoch']),
        'embedding_dim': config['model']['embedding']['dim'],
        'model_type': 'simclr'
    }
    
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================
    
    # Plot training curves
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(history['train_loss']) + 1)
        ax.plot(epochs, history['train_loss'], label='Train', color='#00ff88')
        ax.plot(epochs, history['val_loss'], label='Val', color='#f472b6')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Contrastive Loss')
        ax.set_title('SimCLR Training Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("✓ Saved training curves")
    except Exception as e:
        logger.warning(f"Failed to plot curves: {e}")
    
    # Plot latent space
    try:
        embeddings, labels = collect_embeddings(model, test_loader, device, max_samples=1500)
        plot_latent_space_simclr(
            embeddings, labels, class_names,
            output_dir / 'latent_space.png',
            method='tsne',
            title='SimCLR Embedding Space (t-SNE)'
        )
        logger.info("✓ Saved latent space visualization")
    except Exception as e:
        logger.warning(f"Failed to plot latent space: {e}")
    
    # ========================================================================
    # SAVE EMBEDDINGS
    # ========================================================================
    
    embeddings_dir = output_dir / 'embeddings'
    embeddings_dir.mkdir(exist_ok=True)
    
    for split_name, split_loader in [
        ('train', train_loader),
        ('val', val_loader),
        ('test', test_loader)
    ]:
        save_embeddings_for_clustering(
            model, split_loader, device,
            embeddings_dir / f'{split_name}_embeddings.pkl',
            class_names
        )
        logger.info(f"✓ Saved {split_name} embeddings")
    
    if writer:
        writer.add_scalar('Test/loss', test_metrics['loss'], best_checkpoint['epoch'])
        writer.close()
    
    logger.info("=" * 80)
    logger.info("Done!")
    logger.info(f"  Model: {checkpoint_dir / 'best_model.pt'}")
    logger.info(f"  Results: {output_dir / 'test_results.json'}")
    logger.info(f"  Embeddings: {embeddings_dir}")
    logger.info("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR model')
    parser.add_argument('--config', type=str, default='config/config_simclr.yaml',
                       help='Path to configuration file')
    parser.add_argument('--force-rebuild', action='store_true',
                       help='Force rebuild dataset')
    parser.add_argument('--subsample', type=float, default=1.0,
                       help='Fraction of dataset to use (0.0-1.0)')
    parser.add_argument('--no-normalize', action='store_true',
                       help='Disable feature normalization')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of data loading workers')
    parser.add_argument('--dataset-workers', type=int, default=None,
                       help='Number of dataset building workers')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Max files to load (random sample, for quick testing)')
    
    args = parser.parse_args()
    
    main(args.config, args.force_rebuild, 
         subsample_fraction=args.subsample,
         normalize=not args.no_normalize,
         num_workers_override=args.workers,
         dataset_workers_override=args.dataset_workers,
         max_files=args.max_files)

