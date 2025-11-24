#!/usr/bin/env python3
"""
Training script for Graph Attention Network on EEG synchronization data.

This script handles:
- Loading and preprocessing graph datasets
- Model initialization and training
- TensorBoard logging
- Checkpointing
- Early stopping
"""

import argparse
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from data import create_dataset_from_config
from models import create_model_from_config
from utils import setup_logging, log_metrics, plot_training_curves, plot_confusion_matrix
from utils.visualization import plot_graph_statistics

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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
    lr = train_config['learning_rate']
    weight_decay = train_config['weight_decay']
    
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
    
    if scheduler_config['type'] is None:
        return None
    
    scheduler_type = scheduler_config['type'].lower()
    
    if scheduler_type == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=scheduler_config['patience'],
            factor=scheduler_config['factor'],
            min_lr=scheduler_config['min_lr'],
            verbose=True
        )
    elif scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['num_epochs'],
            eta_min=scheduler_config['min_lr']
        )
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get('step_size', 30),
            gamma=scheduler_config['factor']
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
    
    return scheduler


def train_epoch(model: nn.Module,
                loader: DataLoader,
                optimizer: optim.Optimizer,
                criterion: nn.Module,
                device: torch.device,
                config: Dict[str, Any]) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, data in enumerate(loader):
        data = data.to(device)
        
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        
        loss.backward()
        
        # Gradient clipping
        if config['training']['gradient_clipping']['enabled']:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training']['gradient_clipping']['max_norm']
            )
        
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
        
        preds = out.argmax(dim=1).cpu().numpy()
        labels = data.y.cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)
    
    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model: nn.Module,
            loader: DataLoader,
            criterion: nn.Module,
            device: torch.device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate model.
    
    Returns:
        Tuple of (average_loss, accuracy, predictions, labels)
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for data in loader:
        data = data.to(device)
        out = model(data)
        loss = criterion(out, data.y)
        
        total_loss += loss.item() * data.num_graphs
        
        preds = out.argmax(dim=1).cpu().numpy()
        labels = data.y.cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)
    
    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


def main(config_path: str, force_rebuild: bool = False):
    """Main training function."""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directories
    output_dir = Path(config['paths']['output_dir'])
    checkpoint_dir = Path(config['paths']['checkpoints'])
    tensorboard_dir = Path(config['paths']['tensorboard'])
    
    for dir_path in [output_dir, checkpoint_dir, tensorboard_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(config, output_dir / 'logs')
    logger.info("=" * 80)
    logger.info("Starting BrainStateGAT Training")
    logger.info("=" * 80)
    
    # Set seed for reproducibility
    set_seed(config['seed'])
    logger.info(f"Random seed set to {config['seed']}")
    
    # Get device
    device = get_device(config)
    logger.info(f"Using device: {device}")
    
    # ========================================================================
    # LOAD DATA
    # ========================================================================
    
    logger.info("Loading datasets...")
    
    # Use dataset_workers from config or default to auto-detect
    dataset_workers = config.get('dataset_workers', None)  # None = auto-detect (max 8)
    
    train_graphs, val_graphs, test_graphs = create_dataset_from_config(
        config, 
        force_rebuild=force_rebuild,
        num_workers=dataset_workers
    )
    
    logger.info(f"Train: {len(train_graphs)} graphs")
    logger.info(f"Val:   {len(val_graphs)} graphs")
    logger.info(f"Test:  {len(test_graphs)} graphs")
    
    # Create data loaders
    batch_size = config['training']['batch_size']
    num_workers = config['num_workers']
    pin_memory = config['pin_memory']
    
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
    
    # Visualize dataset statistics
    if config['visualization']['plot_graph_statistics']:
        logger.info("Generating graph statistics plots...")
        plot_graph_statistics(
            train_graphs, val_graphs, test_graphs,
            output_dir / 'visualizations',
            class_names=config['data']['conditions']
        )
    
    # ========================================================================
    # CREATE MODEL
    # ========================================================================
    
    # Infer feature dimensions from first graph
    sample_graph = train_graphs[0]
    num_node_features = sample_graph.x.shape[1]
    num_edge_features = sample_graph.edge_attr.shape[1] if hasattr(sample_graph, 'edge_attr') else 0
    num_graph_features = sample_graph.graph_attr.shape[0] if hasattr(sample_graph, 'graph_attr') else 0
    num_classes = len(config['data']['conditions'])
    
    logger.info(f"Node features: {num_node_features}")
    logger.info(f"Edge features: {num_edge_features}")
    logger.info(f"Graph features: {num_graph_features}")
    logger.info(f"Classes: {num_classes}")
    
    model = create_model_from_config(
        config,
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        num_graph_features=num_graph_features,
        num_classes=num_classes
    )
    model = model.to(device)
    
    # ========================================================================
    # TRAINING SETUP
    # ========================================================================
    
    # Loss function
    class_weights = config['training']['class_weights']
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    label_smoothing = config['training']['label_smoothing']
    criterion = nn.NLLLoss(weight=class_weights, label_smoothing=label_smoothing)
    
    # Optimizer and scheduler
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
    best_val_acc = 0.0
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    logger.info("Starting training...")
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, config
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Log to console
        log_metrics(logger, epoch, 'train', {'loss': train_loss, 'acc': train_acc})
        log_metrics(logger, epoch, 'val', {'loss': val_loss, 'acc': val_acc})
        
        # Log to TensorBoard
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Scheduler step
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Save checkpoint
        if epoch % save_frequency == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'history': history
            }, checkpoint_path)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            
            best_model_path = checkpoint_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'config': config
            }, best_model_path)
            
            logger.info(f"✓ New best model saved! Val Acc: {val_acc:.4f}")
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
    
    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )
    
    # Compute detailed metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, test_preds, average='macro'
    )
    
    logger.info(f"Test Loss:      {test_loss:.4f}")
    logger.info(f"Test Accuracy:  {test_acc:.4f}")
    logger.info(f"Test Precision: {precision:.4f}")
    logger.info(f"Test Recall:    {recall:.4f}")
    logger.info(f"Test F1:        {f1:.4f}")
    
    # Save test results
    results = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'test_precision': precision,
        'test_recall': recall,
        'test_f1': f1,
        'best_val_acc': best_val_acc,
        'best_val_loss': best_val_loss,
        'best_epoch': best_checkpoint['epoch']
    }
    
    import json
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================
    
    if config['visualization']['plot_training_curves']:
        plot_training_curves(
            history,
            output_dir / 'training_curves.png',
            title='BrainStateGAT Training Curves'
        )
    
    if config['visualization']['plot_confusion_matrix']:
        plot_confusion_matrix(
            test_labels,
            test_preds,
            class_names=config['data']['conditions'],
            save_path=output_dir / 'confusion_matrix.png',
            normalize=True,
            title='Test Set Confusion Matrix'
        )
    
    # Close TensorBoard writer
    if writer:
        writer.close()
    
    logger.info("=" * 80)
    logger.info("All done! Results saved to:")
    logger.info(f"  - Model: {checkpoint_dir / 'best_model.pt'}")
    logger.info(f"  - Results: {output_dir / 'test_results.json'}")
    logger.info(f"  - Visualizations: {output_dir}")
    logger.info("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train BrainStateGAT model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--force-rebuild', action='store_true',
                       help='Force rebuild dataset even if cache exists')
    
    args = parser.parse_args()
    
    main(args.config, args.force_rebuild)

