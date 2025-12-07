#!/usr/bin/env python3
"""
Training script for Enhanced VAE with GAT Encoder/Decoder.

Supports:
- Training on specific subjects and conditions
- Multi-band, temporal window, and single-band modes
- GAT-based decoder with attention
- Attention weight extraction for clustering analysis

Usage:
    # Train on all data
    python train_enhanced.py --config config/config_enhanced.yaml
    
    # Train on specific subject and condition
    python train_enhanced.py --config config/config_enhanced.yaml --subject S01 --condition DMT
    
    # Train on multiple subjects
    python train_enhanced.py --config config/config_enhanced.yaml --subjects S01 S02 S03
    
    # Use CPU with multiple workers
    python train_enhanced.py --config config/config_enhanced.yaml --cpu --workers 12
"""

import argparse
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import logging
import json
import random
from datetime import datetime

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

from data.enhanced_dataset import create_dataset_from_config as create_enhanced_dataset
from models.enhanced_vae import create_enhanced_vae

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
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


def normalize_features(graphs, scaler=None):
    """Normalize node features using RobustScaler."""
    all_features = np.vstack([g.x.numpy() for g in graphs])
    
    if scaler is None:
        scaler = RobustScaler()
        scaler.fit(all_features)
    
    for g in graphs:
        features = g.x.numpy()
        features_scaled = scaler.transform(features)
        g.x = torch.tensor(features_scaled, dtype=torch.float32)
    
    return graphs, scaler


def get_beta(epoch: int, config: Dict[str, Any]) -> float:
    """Get beta value for β-VAE with annealing."""
    kl_config = config['loss']['kl']
    
    if not kl_config['annealing']['enabled']:
        return kl_config['weight']
    
    anneal = kl_config['annealing']
    start = anneal['start']
    end = anneal['end']
    epochs = anneal['epochs']
    
    if epoch >= epochs:
        return end
    
    progress = epoch / epochs
    
    if anneal.get('type', 'linear') == 'cosine':
        return start + (end - start) * (1 - np.cos(np.pi * progress)) / 2
    else:  # linear
        return start + (end - start) * progress


def train_epoch(model, loader, optimizer, device, config, beta):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_recon = 0
    total_kl = 0
    n_batches = 0
    
    free_bits = config['loss'].get('free_bits', 0.0)
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        output = model(data, store_activations=False)
        losses = model.loss_function(output, data, beta=beta, free_bits=free_bits)
        
        loss = losses['loss']
        loss.backward()
        
        # Gradient clipping
        clip_config = config['training'].get('gradient_clipping', {})
        if clip_config.get('enabled', False):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                clip_config['max_norm']
            )
        
        optimizer.step()
        
        total_loss += losses['loss'].item()
        total_recon += losses['recon_loss'].item()
        total_kl += losses['kl_loss'].item()
        n_batches += 1
    
    return {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon / n_batches,
        'kl_loss': total_kl / n_batches
    }


@torch.no_grad()
def evaluate(model, loader, device, config, beta):
    """Evaluate model."""
    model.eval()
    
    total_loss = 0
    total_recon = 0
    total_kl = 0
    n_batches = 0
    
    free_bits = config['loss'].get('free_bits', 0.0)
    
    for data in loader:
        data = data.to(device)
        
        output = model(data, store_activations=False)
        losses = model.loss_function(output, data, beta=beta, free_bits=free_bits)
        
        total_loss += losses['loss'].item()
        total_recon += losses['recon_loss'].item()
        total_kl += losses['kl_loss'].item()
        n_batches += 1
    
    return {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon / n_batches,
        'kl_loss': total_kl / n_batches
    }


def main():
    parser = argparse.ArgumentParser(description='Train Enhanced VAE')
    parser.add_argument('--config', type=str, default='config/config_enhanced.yaml',
                       help='Path to configuration file')
    parser.add_argument('--subject', type=str, default=None,
                       help='Single subject to train on (e.g., S01)')
    parser.add_argument('--subjects', type=str, nargs='+', default=None,
                       help='Multiple subjects to train on')
    parser.add_argument('--condition', type=str, default=None,
                       help='Single condition to train on (DMT, EC, EO)')
    parser.add_argument('--conditions', type=str, nargs='+', default=None,
                       help='Multiple conditions to train on')
    parser.add_argument('--force-rebuild', action='store_true',
                       help='Force rebuild dataset')
    parser.add_argument('--cpu', action='store_true',
                       help='Use CPU instead of GPU')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of data loader workers')
    parser.add_argument('--output-suffix', type=str, default=None,
                       help='Suffix for output directory')
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override device
    if args.cpu:
        config['device'] = 'cpu'
    
    if args.workers is not None:
        config['num_workers'] = args.workers
        config['dataset_workers'] = args.workers
    
    # Determine subjects and conditions
    subjects = None
    if args.subject:
        subjects = [args.subject]
    elif args.subjects:
        subjects = args.subjects
    
    conditions = None
    if args.condition:
        conditions = [args.condition]
    elif args.conditions:
        conditions = args.conditions
    
    # Create output directories with suffix
    suffix = args.output_suffix or ''
    if subjects:
        suffix = f"_{subjects[0]}" + (f"_{conditions[0]}" if conditions and len(conditions) == 1 else '')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"run_{timestamp}{suffix}"
    
    output_dir = Path(config['paths']['output_dir']) / run_name
    checkpoint_dir = output_dir / 'checkpoints'
    tensorboard_dir = output_dir / 'runs'
    
    for d in [output_dir, checkpoint_dir, tensorboard_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Save config copy
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Setup logging to file
    file_handler = logging.FileHandler(output_dir / 'train.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("=" * 80)
    logger.info("ENHANCED VAE TRAINING")
    logger.info("=" * 80)
    logger.info(f"Config: {config_path}")
    logger.info(f"Output: {output_dir}")
    if subjects:
        logger.info(f"Subjects: {subjects}")
    if conditions:
        logger.info(f"Conditions: {conditions}")
    
    # Set seed
    set_seed(config['seed'])
    
    # Device
    device_name = config['device']
    if device_name == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device_name = 'cpu'
    device = torch.device(device_name)
    logger.info(f"Device: {device}")
    
    # ==========================================================================
    # LOAD DATA
    # ==========================================================================
    
    logger.info("Loading dataset...")
    
    try:
        train_graphs, val_graphs, test_graphs = create_enhanced_dataset(
            config,
            subjects=subjects,
            conditions=conditions,
            force_rebuild=args.force_rebuild,
            num_workers=config.get('dataset_workers', 8)
        )
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}")
        raise
    
    logger.info(f"Dataset: Train={len(train_graphs)}, Val={len(val_graphs)}, Test={len(test_graphs)}")
    
    if len(train_graphs) == 0:
        logger.error("No training data! Check subjects/conditions.")
        return
    
    # Normalize features
    logger.info("Normalizing features...")
    train_graphs, scaler = normalize_features(train_graphs)
    val_graphs, _ = normalize_features(val_graphs, scaler)
    test_graphs, _ = normalize_features(test_graphs, scaler)
    
    # Create data loaders
    batch_size = config['training']['batch_size']
    num_workers = config.get('num_workers', 0)
    pin_memory = config.get('pin_memory', False) if num_workers > 0 else False
    
    logger.info(f"DataLoader: batch_size={batch_size}, num_workers={num_workers}")
    
    # Reduce batch size if dataset is small
    if len(train_graphs) < batch_size:
        batch_size = max(1, len(train_graphs) // 2)
        logger.info(f"Reduced batch size to {batch_size}")
    
    train_loader = DataLoader(
        train_graphs, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_graphs, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_graphs, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    # ==========================================================================
    # CREATE MODEL
    # ==========================================================================
    
    sample = train_graphs[0]
    num_node_features = sample.x.shape[1]
    num_edge_features = sample.edge_attr.shape[1] if hasattr(sample, 'edge_attr') else 0
    num_graph_features = sample.graph_attr.shape[0] if hasattr(sample, 'graph_attr') else 0
    num_nodes = sample.num_nodes
    
    logger.info(f"Node features: {num_node_features}")
    logger.info(f"Edge features: {num_edge_features}")
    logger.info(f"Graph features: {num_graph_features}")
    logger.info(f"Nodes: {num_nodes}")
    
    model = create_enhanced_vae(
        config,
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        num_graph_features=num_graph_features,
        num_nodes=num_nodes
    )
    model = model.to(device)
    
    # ==========================================================================
    # OPTIMIZER & SCHEDULER
    # ==========================================================================
    
    train_config = config['training']
    
    lr = float(train_config['learning_rate'])
    weight_decay = float(train_config['weight_decay'])
    
    if train_config['optimizer'].lower() == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    
    scheduler_config = train_config['scheduler']
    min_lr = float(scheduler_config.get('min_lr', 1e-6))
    if scheduler_config['type'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_config['num_epochs'],
            eta_min=min_lr
        )
    elif scheduler_config['type'] == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=scheduler_config['patience'],
            factor=float(scheduler_config['factor']),
            min_lr=min_lr
        )
    else:
        scheduler = None
    
    # TensorBoard
    writer = SummaryWriter(tensorboard_dir) if config['logging']['tensorboard'] else None
    
    # ==========================================================================
    # TRAINING LOOP
    # ==========================================================================
    
    num_epochs = train_config['num_epochs']
    early_stopping_patience = train_config['early_stopping']['patience']
    early_stopping_delta = train_config['early_stopping']['min_delta']
    save_frequency = config['logging']['save_frequency']
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    history = {
        'train_loss': [], 'train_recon': [], 'train_kl': [],
        'val_loss': [], 'val_recon': [], 'val_kl': [],
        'beta': [], 'lr': []
    }
    
    logger.info("Starting training...")
    
    pbar = tqdm(range(1, num_epochs + 1), desc="Training")
    
    for epoch in pbar:
        beta = get_beta(epoch, config)
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, config, beta)
        
        # Validate
        val_metrics = evaluate(model, val_loader, device, config, beta)
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_recon'].append(train_metrics['recon_loss'])
        history['train_kl'].append(train_metrics['kl_loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_recon'].append(val_metrics['recon_loss'])
        history['val_kl'].append(val_metrics['kl_loss'])
        history['beta'].append(beta)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Update progress bar
        pbar.set_postfix({
            'train': f"{train_metrics['loss']:.4f}",
            'val': f"{val_metrics['loss']:.4f}",
            'β': f"{beta:.3f}"
        })
        
        # Log to TensorBoard
        if writer:
            writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            writer.add_scalar('Loss/train_recon', train_metrics['recon_loss'], epoch)
            writer.add_scalar('Loss/val_recon', val_metrics['recon_loss'], epoch)
            writer.add_scalar('Loss/train_kl', train_metrics['kl_loss'], epoch)
            writer.add_scalar('Loss/val_kl', val_metrics['kl_loss'], epoch)
            writer.add_scalar('Schedule/beta', beta, epoch)
            writer.add_scalar('Schedule/lr', optimizer.param_groups[0]['lr'], epoch)
        
        # Scheduler step
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
        
        # Save checkpoint
        if epoch % save_frequency == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'config': config
            }, checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pt')
        
        # Best model
        if val_metrics['loss'] < best_val_loss - early_stopping_delta:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'config': config
            }, checkpoint_dir / 'best_model.pt')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    pbar.close()
    
    # ==========================================================================
    # FINAL EVALUATION
    # ==========================================================================
    
    logger.info("=" * 80)
    logger.info("Training complete. Loading best model...")
    
    best_ckpt = torch.load(checkpoint_dir / 'best_model.pt')
    model.load_state_dict(best_ckpt['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, device, config, beta=1.0)
    
    logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
    logger.info(f"Test Recon: {test_metrics['recon_loss']:.4f}")
    logger.info(f"Test KL: {test_metrics['kl_loss']:.4f}")
    
    # Save results
    results = {
        'test_loss': float(test_metrics['loss']),
        'test_recon_loss': float(test_metrics['recon_loss']),
        'test_kl_loss': float(test_metrics['kl_loss']),
        'best_val_loss': float(best_val_loss),
        'best_epoch': int(best_ckpt['epoch']),
        'subjects': subjects,
        'conditions': conditions,
        'num_train': len(train_graphs),
        'num_val': len(val_graphs),
        'num_test': len(test_graphs)
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    if writer:
        writer.close()
    
    logger.info("=" * 80)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Best model: {checkpoint_dir / 'best_model.pt'}")
    logger.info("=" * 80)
    
    print(f"\n✓ Training complete!")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Test loss: {test_metrics['loss']:.4f}")
    print(f"  Output: {output_dir}")


if __name__ == '__main__':
    main()

