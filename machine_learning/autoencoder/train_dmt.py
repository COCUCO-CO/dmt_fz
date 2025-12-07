#!/usr/bin/env python3
"""
Training script for DMT-Only VAE.

Key features:
- Subject-wise data splitting (no leakage)
- Graph convolutional decoder
- Extensive logging to tensorboard and disk
- Latent space quality metrics
"""

import argparse
import sys
import yaml
import json
import random
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, List
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score

# Add parent to path
sys.path.append(str(Path(__file__).parent))

from models.vae_model import BrainStateVAE
from data import create_dataset_from_config

# Setup logging
def setup_logging(output_dir: Path, name: str = 'train_dmt'):
    """Setup logging to file and console."""
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'{name}_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(name)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def normalize_graphs(graphs, scaler=None, clip_value=10.0):
    """Normalize node features using RobustScaler with clipping."""
    all_features = np.vstack([g.x.numpy() for g in graphs])
    
    # First clip extreme values before fitting scaler
    if scaler is None:
        # Calculate percentiles for clipping
        q01 = np.percentile(all_features, 1, axis=0)
        q99 = np.percentile(all_features, 99, axis=0)
        all_features_clipped = np.clip(all_features, q01, q99)
        
        scaler = RobustScaler()
        scaler.fit(all_features_clipped)
        scaler.clip_low = q01
        scaler.clip_high = q99
    
    for g in graphs:
        features = g.x.numpy()
        # Clip then scale
        features_clipped = np.clip(features, scaler.clip_low, scaler.clip_high)
        features_scaled = scaler.transform(features_clipped)
        # Final clip to prevent any remaining extremes
        features_scaled = np.clip(features_scaled, -clip_value, clip_value)
        g.x = torch.tensor(features_scaled, dtype=torch.float32)
    
    return graphs, scaler


def compute_loss(model, batch, config, beta=1.0):
    """Compute VAE loss (reconstruction + KL)."""
    # Forward pass
    output = model(batch)
    
    recon_x = output['x_recon']
    mu = output['mu']
    log_var = output['log_var']
    
    # Reconstruction loss (nodes)
    node_loss = nn.functional.mse_loss(recon_x, batch.x, reduction='mean')
    
    # Edge reconstruction loss (only if edge_attr_recon exists and is not None)
    edge_loss = torch.tensor(0.0, device=batch.x.device)
    edge_attr_recon = output.get('edge_attr_recon', None)
    if edge_attr_recon is not None and batch.edge_attr is not None:
        edge_loss = nn.functional.mse_loss(edge_attr_recon, batch.edge_attr, reduction='mean')
    
    # Total reconstruction loss
    node_weight = float(config['loss']['reconstruction']['node_weight'])
    edge_weight = float(config['loss']['reconstruction']['edge_weight'])
    recon_loss = node_weight * node_loss + edge_weight * edge_loss
    
    # KL divergence with free bits
    free_bits = float(config['loss'].get('free_bits', 0.0))
    kl_per_dim = 0.5 * (mu.pow(2) + log_var.exp() - log_var - 1)
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    kl_loss = kl_per_dim.sum(dim=1).mean()
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return {
        'total': total_loss,
        'recon': recon_loss,
        'node': node_loss,
        'edge': edge_loss,
        'kl': kl_loss
    }


def get_beta(epoch, config):
    """Get KL weight (beta) with annealing."""
    kl_config = config['loss']['kl']
    if not kl_config['annealing']['enabled']:
        return float(kl_config['weight'])
    
    anneal_epochs = int(kl_config['annealing']['epochs'])
    start = float(kl_config['annealing']['start'])
    end = float(kl_config['annealing']['end'])
    anneal_type = kl_config['annealing']['type']
    
    if epoch >= anneal_epochs:
        return end
    
    progress = epoch / anneal_epochs
    
    if anneal_type == 'linear':
        return start + (end - start) * progress
    elif anneal_type == 'cosine':
        return start + (end - start) * (1 - np.cos(np.pi * progress)) / 2
    else:
        return end * progress


def extract_latent(model, loader, device, graphs=None):
    """Extract latent representations for all samples."""
    model.eval()
    all_z = []
    all_subjects = []
    graph_idx = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            z = output['z']  # Already per-graph: (batch_size, latent_dim)
            
            all_z.append(z.cpu().numpy())
            
            # Get subjects from original graphs
            if graphs is not None:
                for _ in range(batch.num_graphs):
                    if graph_idx < len(graphs):
                        g = graphs[graph_idx]
                        if hasattr(g, 'subject_id'):
                            all_subjects.append(g.subject_id)
                        else:
                            all_subjects.append(f'S{graph_idx}')
                    graph_idx += 1
    
    return np.vstack(all_z), all_subjects


def compute_latent_quality(z: np.ndarray, subjects: List) -> Dict:
    """Compute latent space quality metrics."""
    metrics = {}
    
    # Convert subjects to numeric labels
    unique_subjects = list(set(subjects))
    subject_labels = np.array([unique_subjects.index(s) for s in subjects])
    
    # Silhouette score (by subject)
    if len(unique_subjects) > 1 and len(z) > len(unique_subjects):
        try:
            sil = silhouette_score(z, subject_labels)
            metrics['silhouette_by_subject'] = float(sil)
        except:
            metrics['silhouette_by_subject'] = 0.0
    else:
        metrics['silhouette_by_subject'] = 0.0
    
    # Variance explained per dimension
    var_per_dim = z.var(axis=0)
    metrics['var_per_dim'] = var_per_dim.tolist()
    metrics['total_variance'] = float(var_per_dim.sum())
    metrics['effective_dims'] = int((var_per_dim > 0.01).sum())
    
    return metrics


def train_epoch(model, loader, optimizer, config, beta, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        losses = compute_loss(model, batch, config, beta)
        losses['total'].backward()
        
        # Gradient clipping
        if config['training']['gradient_clipping']['enabled']:
            nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training']['gradient_clipping']['max_norm']
            )
        
        optimizer.step()
        
        total_loss += losses['total'].item()
        total_recon += losses['recon'].item()
        total_kl += losses['kl'].item()
    
    n_batches = len(loader)
    return {
        'loss': total_loss / n_batches,
        'recon': total_recon / n_batches,
        'kl': total_kl / n_batches
    }


def validate(model, loader, config, beta, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            losses = compute_loss(model, batch, config, beta)
            
            total_loss += losses['total'].item()
            total_recon += losses['recon'].item()
            total_kl += losses['kl'].item()
    
    n_batches = len(loader)
    return {
        'loss': total_loss / n_batches,
        'recon': total_recon / n_batches,
        'kl': total_kl / n_batches
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup output directory
    output_dir = Path(config['paths']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("=" * 70)
    logger.info("DMT-Only VAE Training")
    logger.info("=" * 70)
    
    # Set seed
    set_seed(config['seed'])
    logger.info(f"Random seed: {config['seed']}")
    
    # Device
    device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Load dataset
    logger.info("\nLoading DMT dataset...")
    train_graphs, val_graphs, test_graphs = create_dataset_from_config(config)
    
    logger.info(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")
    
    # Normalize
    logger.info("Normalizing features...")
    train_graphs, scaler = normalize_graphs(train_graphs)
    val_graphs, _ = normalize_graphs(val_graphs, scaler)
    test_graphs, _ = normalize_graphs(test_graphs, scaler)
    
    # Create data loaders
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size)
    test_loader = DataLoader(test_graphs, batch_size=batch_size)
    
    # Get data dimensions
    sample = train_graphs[0]
    num_node_features = sample.x.shape[1]
    num_edge_features = sample.edge_attr.shape[1] if sample.edge_attr is not None else 0
    num_graph_features = sample.graph_attr.shape[0] if hasattr(sample, 'graph_attr') and sample.graph_attr is not None else 0
    num_nodes = sample.x.shape[0]
    
    logger.info(f"Node features: {num_node_features}")
    logger.info(f"Edge features: {num_edge_features}")
    logger.info(f"Graph features: {num_graph_features}")
    logger.info(f"Nodes per graph: {num_nodes}")
    
    # Create model
    model = BrainStateVAE(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        num_graph_features=num_graph_features,
        num_nodes=num_nodes,
        config=config
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer (convert to float in case YAML loaded as string)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay'])
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs'],
        eta_min=float(config['training']['scheduler']['min_lr'])
    )
    
    # Tensorboard
    tb_dir = Path(config['paths']['tensorboard'])
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(tb_dir)
    
    # Checkpoints
    ckpt_dir = Path(config['paths']['checkpoints'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    history = {'train': [], 'val': []}
    
    logger.info("\nStarting training...")
    
    for epoch in range(1, config['training']['num_epochs'] + 1):
        beta = get_beta(epoch, config)
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, config, beta, device)
        
        # Validate
        val_metrics = validate(model, val_loader, config, beta, device)
        
        # Step scheduler
        scheduler.step()
        
        # Log
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)
        
        # Log all metrics to TensorBoard
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('Loss/train_recon', train_metrics['recon'], epoch)
        writer.add_scalar('Loss/val_recon', val_metrics['recon'], epoch)
        writer.add_scalar('Loss/train_kl', train_metrics['kl'], epoch)
        writer.add_scalar('Loss/val_kl', val_metrics['kl'], epoch)
        writer.add_scalar('Hyperparams/beta', beta, epoch)
        writer.add_scalar('Hyperparams/learning_rate', scheduler.get_last_lr()[0], epoch)
        
        # Log gradient norms
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        writer.add_scalar('Gradients/total_norm', total_norm, epoch)
        
        # Log generalization gap
        gap = abs(train_metrics['loss'] - val_metrics['loss'])
        writer.add_scalar('Metrics/generalization_gap', gap, epoch)
        
        # Log latent space statistics every 10 epochs
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                sample_batch = next(iter(val_loader)).to(device)
                output = model(sample_batch)
                mu = output['mu']
                log_var = output['log_var']
                z = output['z']
                
                # Latent statistics
                writer.add_scalar('Latent/mu_mean', mu.mean().item(), epoch)
                writer.add_scalar('Latent/mu_std', mu.std().item(), epoch)
                writer.add_scalar('Latent/logvar_mean', log_var.mean().item(), epoch)
                writer.add_scalar('Latent/z_std', z.std().item(), epoch)
                
                # Histograms
                writer.add_histogram('Latent/mu_dist', mu, epoch)
                writer.add_histogram('Latent/z_dist', z, epoch)
        
        logger.info(f"Epoch {epoch:3d} | Train: {train_metrics['loss']:.4f} | Val: {val_metrics['loss']:.4f} | β: {beta:.4f}")
        
        # Check best
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'config': config
            }, ckpt_dir / 'best_model.pt')
            
            logger.info(f"  ✓ New best model saved!")
        else:
            patience_counter += 1
        
        # Save checkpoint periodically
        if epoch % config['logging']['save_frequency'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_metrics['loss']
            }, ckpt_dir / f'checkpoint_epoch_{epoch:03d}.pt')
        
        # Early stopping
        if patience_counter >= config['training']['early_stopping']['patience']:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model
    logger.info("\nLoading best model for evaluation...")
    checkpoint = torch.load(ckpt_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation on test set
    test_metrics = validate(model, test_loader, config, get_beta(best_epoch, config), device)
    logger.info(f"\nTest Loss: {test_metrics['loss']:.4f}")
    
    # Extract latent representations
    logger.info("\nExtracting latent representations...")
    
    all_loaders = [
        ('train', train_loader, train_graphs),
        ('val', val_loader, val_graphs),
        ('test', test_loader, test_graphs)
    ]
    
    embeddings = {}
    for name, loader, graphs in all_loaders:
        z, subjects = extract_latent(model, loader, device, graphs)
        
        # Get subjects from graphs if not extracted properly
        if not subjects or len(subjects) != len(z):
            subjects = [g.subject_id if hasattr(g, 'subject_id') else f'S{i}' for i, g in enumerate(graphs)]
        
        embeddings[name] = {
            'z': z,
            'subjects': subjects
        }
    
    # Compute latent quality
    all_z = np.vstack([embeddings[k]['z'] for k in embeddings])
    all_subjects = []
    for k in embeddings:
        all_subjects.extend(embeddings[k]['subjects'])
    
    latent_metrics = compute_latent_quality(all_z, all_subjects)
    logger.info(f"Latent silhouette (by subject): {latent_metrics['silhouette_by_subject']:.4f}")
    logger.info(f"Effective dimensions: {latent_metrics['effective_dims']}")
    
    # Save embeddings
    emb_dir = output_dir / 'embeddings'
    emb_dir.mkdir(parents=True, exist_ok=True)
    
    for name, data in embeddings.items():
        with open(emb_dir / f'{name}_embeddings.pkl', 'wb') as f:
            pickle.dump({
                'latent_embeddings': data['z'],
                'subjects': data['subjects'],
                'conditions': ['DMT'] * len(data['subjects'])
            }, f)
    
    # Save results
    results = {
        'status': 'success',
        'best_epoch': best_epoch,
        'train_loss': history['train'][best_epoch-1]['loss'],
        'val_loss': best_val_loss,
        'test_loss': test_metrics['loss'],
        'recon_loss': test_metrics['recon'],
        'kl_loss': test_metrics['kl'],
        'latent_silhouette': latent_metrics['silhouette_by_subject'],
        'effective_dims': latent_metrics['effective_dims'],
        'total_epochs': len(history['train'])
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    writer.close()
    
    logger.info("\n" + "=" * 70)
    logger.info("Training complete!")
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Best val loss: {best_val_loss:.4f}")
    logger.info(f"Test loss: {test_metrics['loss']:.4f}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()

