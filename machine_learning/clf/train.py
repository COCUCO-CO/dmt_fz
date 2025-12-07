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
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    balanced_accuracy_score, confusion_matrix as sklearn_confusion_matrix,
    classification_report
)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from data import create_dataset_from_config
from models import create_model_from_config
from utils import setup_logging, log_metrics, plot_training_curves, plot_confusion_matrix
from utils import log_attention_to_tensorboard, plot_attention_distributions, plot_average_attention
from utils import log_embeddings_to_tensorboard, save_embeddings_to_file, compute_attention_matrix_per_class
from utils import generate_full_attention_analysis
from utils.visualization import plot_graph_statistics
from tests.test_dataset import DatasetValidator

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
        
        # Gradient clipping (supports both formats)
        grad_clip_config = config['training'].get('gradient_clipping', {})
        grad_clip_value = config['training'].get('gradient_clip', None)  # Simple format from hyperparam search
        
        if grad_clip_value is not None and grad_clip_value > 0:
            # Simple format: gradient_clip = 1.0 means clip to 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
        elif isinstance(grad_clip_config, dict) and grad_clip_config.get('enabled', False):
            # Nested format: gradient_clipping.enabled, gradient_clipping.max_norm
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip_config['max_norm']
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


def main(config_path: str, force_rebuild: bool = False, resume: bool = False):
    """Main training function.
    
    Args:
        config_path: Path to configuration file
        force_rebuild: Force rebuild dataset even if cache exists
        resume: Resume training from best checkpoint
    """
    
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
    
    # Validate dataset before training
    class_names = config['data']['conditions']
    validator = DatasetValidator(train_graphs, val_graphs, test_graphs, class_names)
    if not validator.validate_all():
        logger.error("Dataset validation FAILED! Check errors above.")
        if config.get('strict_validation', True):
            raise ValueError("Dataset validation failed. Set 'strict_validation: false' in config to skip.")
        else:
            logger.warning("Continuing despite validation errors (strict_validation=false)")
    
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
    
    # Note: NLLLoss doesn't support label_smoothing (only CrossEntropyLoss does)
    # Since our model outputs log_softmax, we use NLLLoss
    label_smoothing = config['training']['label_smoothing']
    if label_smoothing > 0:
        logger.warning(f"label_smoothing={label_smoothing} specified but NLLLoss doesn't support it. Ignoring.")
    
    criterion = nn.NLLLoss(weight=class_weights)
    
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
    # RESUME FROM CHECKPOINT (if requested)
    # ========================================================================
    
    start_epoch = 1
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    if resume:
        best_model_path = checkpoint_dir / 'best_model.pt'
        if best_model_path.exists():
            logger.info(f"Resuming from checkpoint: {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint.get('val_acc', 0.0)
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            logger.info(f"Resumed from epoch {checkpoint['epoch']} (best val acc: {best_val_acc:.4f})")
        else:
            logger.warning(f"No checkpoint found at {best_model_path}, starting from scratch")
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    
    num_epochs = config['training']['num_epochs']
    early_stopping_patience = config['training']['early_stopping']['patience']
    early_stopping_delta = config['training']['early_stopping']['min_delta']
    save_frequency = config['logging']['save_frequency']
    
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    logger.info("Starting training...")
    
    for epoch in range(start_epoch, num_epochs + 1):
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
        
        # Compute detailed validation metrics
        class_names = config['data']['conditions']
        val_balanced_acc = balanced_accuracy_score(val_labels, val_preds)
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
            val_labels, val_preds, average='macro', zero_division=0
        )
        val_precision_per_class, val_recall_per_class, val_f1_per_class, _ = precision_recall_fscore_support(
            val_labels, val_preds, average=None, zero_division=0
        )
        
        # Log to TensorBoard
        if writer:
            # Basic metrics
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
            
            # Detailed metrics (important for imbalanced datasets)
            writer.add_scalar('Metrics/val_balanced_accuracy', val_balanced_acc, epoch)
            writer.add_scalar('Metrics/val_f1_macro', val_f1, epoch)
            writer.add_scalar('Metrics/val_precision_macro', val_precision, epoch)
            writer.add_scalar('Metrics/val_recall_macro', val_recall, epoch)
            
            # Per-class metrics
            for i, cname in enumerate(class_names):
                if i < len(val_recall_per_class):
                    writer.add_scalar(f'Recall_PerClass/{cname}', val_recall_per_class[i], epoch)
                    writer.add_scalar(f'Precision_PerClass/{cname}', val_precision_per_class[i], epoch)
                    writer.add_scalar(f'F1_PerClass/{cname}', val_f1_per_class[i], epoch)
            
            # Log attention weights and embeddings periodically
            save_frequency = config['logging']['save_frequency']
            if epoch % save_frequency == 0:
                # Log attention weights (only for GATv2)
                if config['model']['architecture']['conv_type'] == 'gatv2':
                    try:
                        logger.info(f"Logging attention weights for epoch {epoch}...")
                        class_names = config['data']['conditions']
                        log_attention_to_tensorboard(
                            writer, model, val_loader, device, epoch, 
                            num_samples=5, class_names=class_names, save_dir=output_dir
                        )
                    except Exception as e:
                        logger.warning(f"Failed to log attention weights: {e}")
                    
                    # Log full per-class attention analysis (includes MST graphs)
                    if epoch % 10 == 0:
                        try:
                            logger.info(f"Generating per-class attention analysis for epoch {epoch}...")
                            attention_analysis_dir = output_dir / 'attention' / f'epoch_{epoch}'
                            generate_full_attention_analysis(
                                model, val_loader, device,
                                save_dir=attention_analysis_dir,
                                class_names=class_names,
                                writer=writer,
                                epoch=epoch
                            )
                        except Exception as e:
                            logger.warning(f"Failed to generate per-class attention analysis: {e}")
                
                # Log embeddings for visualization
                try:
                    logger.info(f"Logging embeddings for epoch {epoch}...")
                    class_names = config['data']['conditions']
                    log_embeddings_to_tensorboard(writer, model, val_loader, device, epoch, 
                                                   num_samples=300, class_names=class_names)
                except Exception as e:
                    logger.warning(f"Failed to log embeddings: {e}")
        
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
    
    class_names = config['data']['conditions']
    
    # Compute detailed metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, test_preds, average='macro', zero_division=0
    )
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        test_labels, test_preds, average=None, zero_division=0
    )
    balanced_acc = balanced_accuracy_score(test_labels, test_preds)
    conf_matrix = sklearn_confusion_matrix(test_labels, test_preds)
    
    # Log to console
    logger.info(f"Test Loss:              {test_loss:.4f}")
    logger.info(f"Test Accuracy:          {test_acc:.4f}")
    logger.info(f"Test Balanced Accuracy: {balanced_acc:.4f}")
    logger.info(f"Test Precision (macro): {precision:.4f}")
    logger.info(f"Test Recall (macro):    {recall:.4f}")
    logger.info(f"Test F1 (macro):        {f1:.4f}")
    
    logger.info("\nPer-class metrics:")
    for i, cname in enumerate(class_names):
        if i < len(recall_per_class):
            logger.info(f"  {cname:6s} - Precision: {precision_per_class[i]:.4f}, "
                       f"Recall: {recall_per_class[i]:.4f}, F1: {f1_per_class[i]:.4f}, "
                       f"Support: {support[i]}")
    
    logger.info(f"\nConfusion Matrix:\n{conf_matrix}")
    
    # Full classification report
    report = classification_report(test_labels, test_preds, target_names=class_names, zero_division=0)
    logger.info(f"\nClassification Report:\n{report}")
    
    # Save test results (comprehensive)
    results = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'test_balanced_accuracy': float(balanced_acc),
        'test_precision': float(precision),
        'test_recall': float(recall),
        'test_f1': float(f1),
        'per_class': {
            cname: {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1': float(f1_per_class[i]),
                'support': int(support[i])
            }
            for i, cname in enumerate(class_names) if i < len(recall_per_class)
        },
        'confusion_matrix': conf_matrix.tolist(),
        'best_val_acc': float(best_val_acc),
        'best_val_loss': float(best_val_loss),
        'best_epoch': int(best_checkpoint['epoch'])
    }
    
    import json
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save classification report to file
    with open(output_dir / 'classification_report.txt', 'w') as f:
        f.write(f"Classification Report\n{'='*50}\n\n")
        f.write(report)
        f.write(f"\n\nConfusion Matrix:\n{conf_matrix}\n")
        f.write(f"\nBalanced Accuracy: {balanced_acc:.4f}\n")
    
    # Log final metrics to TensorBoard
    if writer:
        writer.add_scalar('Test/loss', test_loss, best_checkpoint['epoch'])
        writer.add_scalar('Test/accuracy', test_acc, best_checkpoint['epoch'])
        writer.add_scalar('Test/balanced_accuracy', balanced_acc, best_checkpoint['epoch'])
        writer.add_scalar('Test/f1_macro', f1, best_checkpoint['epoch'])
        writer.add_scalar('Test/precision_macro', precision, best_checkpoint['epoch'])
        writer.add_scalar('Test/recall_macro', recall, best_checkpoint['epoch'])
        
        # Per-class final metrics
        for i, cname in enumerate(class_names):
            if i < len(recall_per_class):
                writer.add_scalar(f'Test_PerClass/{cname}_recall', recall_per_class[i], best_checkpoint['epoch'])
                writer.add_scalar(f'Test_PerClass/{cname}_precision', precision_per_class[i], best_checkpoint['epoch'])
                writer.add_scalar(f'Test_PerClass/{cname}_f1', f1_per_class[i], best_checkpoint['epoch'])
    
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
        cm_path = output_dir / 'confusion_matrix.png'
        plot_confusion_matrix(
            test_labels,
            test_preds,
            class_names=config['data']['conditions'],
            save_path=cm_path,
            normalize=True,
            title='Test Set Confusion Matrix'
        )
        
        # Log confusion matrix to TensorBoard
        if writer and cm_path.exists():
            import matplotlib.pyplot as plt
            cm_img = plt.imread(str(cm_path))
            if cm_img.ndim == 3 and cm_img.shape[2] == 4:  # RGBA
                cm_img = cm_img[:, :, :3]
            cm_tensor = torch.tensor(cm_img.transpose(2, 0, 1))
            writer.add_image('Test/confusion_matrix', cm_tensor, best_checkpoint['epoch'])
    
    # ========================================================================
    # ATTENTION VISUALIZATION (Final)
    # ========================================================================
    
    if config['model']['architecture']['conv_type'] == 'gatv2':
        logger.info("=" * 80)
        logger.info("Generating attention visualizations...")
        logger.info("=" * 80)
        
        attention_dir = output_dir / 'attention'
        attention_dir.mkdir(exist_ok=True)
        
        try:
            # Plot attention distributions per layer
            from utils import extract_attention_matrices
            attention_stats = extract_attention_matrices(model, test_loader, device, num_samples=100)
            plot_attention_distributions(attention_stats, save_path=attention_dir / 'attention_distributions.png')
            logger.info(f"✓ Saved attention distributions to {attention_dir / 'attention_distributions.png'}")
            
            # Plot average attention matrices (electrode names auto-loaded)
            num_nodes = test_graphs[0].num_nodes
            plot_average_attention(model, test_loader, device, num_nodes,
                                 save_dir=attention_dir)
            logger.info(f"✓ Saved average attention matrices to {attention_dir}/")
            
        except Exception as e:
            logger.warning(f"Failed to generate attention visualizations: {e}")
        
        # Generate full attention analysis: per-class averages, differences, distributions by class
        try:
            logger.info("Generating full attention analysis per class...")
            class_names = config['data']['conditions']
            
            generate_full_attention_analysis(
                model, test_loader, device, 
                save_dir=attention_dir / 'per_class_analysis',
                class_names=class_names,
                writer=writer,
                epoch=best_checkpoint['epoch']
            )
            logger.info(f"✓ Saved full attention analysis to {attention_dir / 'per_class_analysis'}/")
            
            # Also save raw matrices as pickle for later analysis
            import pickle
            num_nodes = test_graphs[0].num_nodes
            attention_per_class = compute_attention_matrix_per_class(
                model, test_loader, device, num_nodes, class_names
            )
            with open(attention_dir / 'attention_per_class.pkl', 'wb') as f:
                pickle.dump(attention_per_class, f)
            logger.info(f"✓ Saved per-class attention matrices to {attention_dir / 'attention_per_class.pkl'}")
            
        except Exception as e:
            logger.warning(f"Failed to compute per-class attention: {e}")
    
    # ========================================================================
    # SAVE EMBEDDINGS FOR ANALYSIS
    # ========================================================================
    
    logger.info("=" * 80)
    logger.info("Saving embeddings for statistical analysis...")
    logger.info("=" * 80)
    
    embeddings_dir = output_dir / 'embeddings'
    embeddings_dir.mkdir(exist_ok=True)
    
    try:
        class_names = config['data']['conditions']
        
        # Save train embeddings
        train_emb = save_embeddings_to_file(
            model, train_loader, device, 
            save_path=embeddings_dir / 'train_embeddings.pkl',
            num_samples=None,  # All samples
            class_names=class_names
        )
        logger.info(f"✓ Saved {len(train_emb['graph_embeddings'])} train embeddings")
        
        # Save test embeddings
        test_emb = save_embeddings_to_file(
            model, test_loader, device,
            save_path=embeddings_dir / 'test_embeddings.pkl',
            num_samples=None,
            class_names=class_names
        )
        logger.info(f"✓ Saved {len(test_emb['graph_embeddings'])} test embeddings")
        
    except Exception as e:
        logger.warning(f"Failed to save embeddings: {e}")
    
    # Close TensorBoard writer
    if writer:
        writer.close()
    
    logger.info("=" * 80)
    logger.info("All done! Results saved to:")
    logger.info(f"  - Model: {checkpoint_dir / 'best_model.pt'}")
    logger.info(f"  - Results: {output_dir / 'test_results.json'}")
    logger.info(f"  - Visualizations: {output_dir}")
    logger.info(f"  - Embeddings: {embeddings_dir}")
    if config['model']['architecture']['conv_type'] == 'gatv2':
        logger.info(f"  - Attention: {attention_dir}")
    logger.info("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train BrainStateGAT model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--force-rebuild', action='store_true',
                       help='Force rebuild dataset even if cache exists')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from best checkpoint')
    
    args = parser.parse_args()
    
    main(args.config, args.force_rebuild, args.resume)

