"""
Visualization utilities for results and analysis.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch

logger = logging.getLogger(__name__)


def plot_training_curves(history: Dict[str, List[float]],
                         save_path: Path,
                         title: str = "Training Curves"):
    """
    Plot training and validation curves.
    
    Args:
        history: Dictionary with keys like 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save the figure
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Loss Curves', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    if 'train_acc' in history:
        axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    if 'val_acc' in history:
        axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Accuracy Curves', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Training curves saved to {save_path}")


def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          class_names: List[str],
                          save_path: Path,
                          normalize: bool = True,
                          title: str = "Confusion Matrix"):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        save_path: Path to save the figure
        normalize: If True, normalize confusion matrix
        title: Plot title
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrix saved to {save_path}")


def save_attention_heatmap(attention_matrix: np.ndarray,
                           save_path: Path,
                           node_labels: Optional[List[str]] = None,
                           title: str = "Attention Heatmap"):
    """
    Plot and save attention weights as a heatmap.
    
    Args:
        attention_matrix: (N, N) attention matrix
        save_path: Path to save the figure
        node_labels: Optional node labels
        title: Plot title
    """
    plt.figure(figsize=(12, 10))
    
    if node_labels is not None and len(node_labels) <= 30:
        # Only show labels if not too many
        sns.heatmap(attention_matrix, cmap='viridis', square=True,
                   xticklabels=node_labels, yticklabels=node_labels,
                   cbar_kws={"shrink": 0.8})
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
    else:
        sns.heatmap(attention_matrix, cmap='viridis', square=True,
                   xticklabels=False, yticklabels=False,
                   cbar_kws={"shrink": 0.8})
    
    plt.xlabel('Target Node', fontsize=12)
    plt.ylabel('Source Node', fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Attention heatmap saved to {save_path}")


def plot_graph_statistics(train_graphs: List[Any],
                          val_graphs: List[Any],
                          test_graphs: List[Any],
                          save_dir: Path,
                          class_names: List[str]):
    """
    Plot various statistics about the graph dataset.
    
    Args:
        train_graphs: List of training graphs
        val_graphs: List of validation graphs
        test_graphs: List of test graphs
        save_dir: Directory to save plots
        class_names: Names of classes
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    all_graphs = train_graphs + val_graphs + test_graphs
    splits = ['train'] * len(train_graphs) + ['val'] * len(val_graphs) + ['test'] * len(test_graphs)
    
    # Extract statistics
    num_nodes = [g.num_nodes for g in all_graphs]
    num_edges = [g.edge_index.shape[1] // 2 for g in all_graphs]  # Undirected
    labels = [g.y.item() for g in all_graphs]
    
    # 1. Class distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, split_name in enumerate(['train', 'val', 'test']):
        split_labels = [l for l, s in zip(labels, splits) if s == split_name]
        axes[i].hist(split_labels, bins=len(class_names), edgecolor='black')
        axes[i].set_xlabel('Class')
        axes[i].set_ylabel('Count')
        axes[i].set_title(f'{split_name.capitalize()} Set Class Distribution')
        axes[i].set_xticks(range(len(class_names)))
        axes[i].set_xticklabels(class_names)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Graph size distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Number of nodes
    axes[0, 0].hist(num_nodes, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Number of Nodes')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Node Count Distribution')
    axes[0, 0].axvline(np.mean(num_nodes), color='red', linestyle='--', label=f'Mean: {np.mean(num_nodes):.1f}')
    axes[0, 0].legend()
    
    # Number of edges
    axes[0, 1].hist(num_edges, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Number of Edges')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Edge Count Distribution')
    axes[0, 1].axvline(np.mean(num_edges), color='red', linestyle='--', label=f'Mean: {np.mean(num_edges):.1f}')
    axes[0, 1].legend()
    
    # Nodes by class
    for i, class_name in enumerate(class_names):
        class_nodes = [n for n, l in zip(num_nodes, labels) if l == i]
        axes[1, 0].hist(class_nodes, bins=20, alpha=0.5, label=class_name, edgecolor='black')
    axes[1, 0].set_xlabel('Number of Nodes')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Node Count by Class')
    axes[1, 0].legend()
    
    # Edges by class
    for i, class_name in enumerate(class_names):
        class_edges = [e for e, l in zip(num_edges, labels) if l == i]
        axes[1, 1].hist(class_edges, bins=20, alpha=0.5, label=class_name, edgecolor='black')
    axes[1, 1].set_xlabel('Number of Edges')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Edge Count by Class')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'graph_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Graph statistics plots saved to {save_dir}")

