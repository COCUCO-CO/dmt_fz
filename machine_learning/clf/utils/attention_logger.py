"""
Attention logging utilities for GAT models.

This module provides functions to extract, visualize, and log attention weights
from Graph Attention Networks during training.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Batch

logger = logging.getLogger(__name__)


def extract_attention_matrices(model, data_loader, device, num_samples: int = 100):
    """
    Extract attention matrices from a batch of graphs.
    
    Args:
        model: GAT model
        data_loader: DataLoader with graphs
        device: torch device
        num_samples: Number of samples to process
        
    Returns:
        Dict with attention statistics per layer
    """
    model.eval()
    
    # Collect attention weights per layer
    layer_attentions = {i: [] for i in range(len(model.conv_layers))}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= num_samples // batch.num_graphs:
                break
            
            batch = batch.to(device)
            
            # Get attention from all layers
            all_attentions = model.get_all_attention_weights(batch)
            
            for layer_idx, (edge_index, alpha) in enumerate(all_attentions):
                # Average across attention heads if needed
                if alpha.dim() > 1:
                    alpha_mean = alpha.mean(dim=1)  # Average over heads
                else:
                    alpha_mean = alpha
                
                layer_attentions[layer_idx].append(alpha_mean.cpu().numpy())
    
    # Compute statistics
    attention_stats = {}
    for layer_idx, attentions in layer_attentions.items():
        if len(attentions) > 0:
            all_att = np.concatenate(attentions)
            attention_stats[f'layer_{layer_idx}'] = {
                'mean': float(np.mean(all_att)),
                'std': float(np.std(all_att)),
                'min': float(np.min(all_att)),
                'max': float(np.max(all_att)),
                'median': float(np.median(all_att)),
                'values': all_att  # Keep raw values for histograms
            }
    
    return attention_stats


def plot_attention_heatmap(edge_index: torch.Tensor, 
                          attention_weights: torch.Tensor,
                          num_nodes: int,
                          title: str = "Attention Weights",
                          save_path: Optional[Path] = None,
                          channel_names: Optional[List[str]] = None):
    """
    Plot attention weights as a heatmap (adjacency matrix style).
    
    Args:
        edge_index: Edge indices [2, num_edges]
        attention_weights: Attention values [num_edges] or [num_edges, num_heads]
        num_nodes: Number of nodes in graph
        title: Plot title
        save_path: Path to save figure
        channel_names: Optional list of node names
    """
    # Average across heads if multi-head
    if attention_weights.dim() > 1:
        attention_weights = attention_weights.mean(dim=1)
    
    # Convert to numpy
    edge_index = edge_index.cpu().numpy()
    attention_weights = attention_weights.cpu().numpy()
    
    # Create adjacency matrix
    adj_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        adj_matrix[src, dst] = attention_weights[i]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Use mask for zero values (no edges)
    mask = (adj_matrix == 0)
    
    sns.heatmap(adj_matrix, mask=mask, cmap='YlOrRd', 
                cbar_kws={'label': 'Attention Weight'},
                square=True, linewidths=0.1, linecolor='gray',
                ax=ax, vmin=0, vmax=1)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Target Node', fontsize=12)
    ax.set_ylabel('Source Node', fontsize=12)
    
    # Add channel names if provided
    if channel_names is not None and len(channel_names) == num_nodes:
        ax.set_xticks(np.arange(num_nodes) + 0.5)
        ax.set_yticks(np.arange(num_nodes) + 0.5)
        ax.set_xticklabels(channel_names, rotation=45, ha='right')
        ax.set_yticklabels(channel_names, rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved attention heatmap to {save_path}")
    
    plt.close()


def plot_attention_distributions(attention_stats: Dict,
                                 save_path: Optional[Path] = None):
    """
    Plot distribution of attention weights per layer.
    
    Args:
        attention_stats: Dict with attention statistics per layer
        save_path: Path to save figure
    """
    num_layers = len([k for k in attention_stats.keys() if k.startswith('layer_')])
    
    fig, axes = plt.subplots(1, num_layers, figsize=(5*num_layers, 4))
    if num_layers == 1:
        axes = [axes]
    
    for layer_idx in range(num_layers):
        layer_key = f'layer_{layer_idx}'
        if layer_key not in attention_stats:
            continue
        
        stats = attention_stats[layer_key]
        values = stats['values']
        
        ax = axes[layer_idx]
        
        # Histogram
        ax.hist(values, bins=50, edgecolor='black', alpha=0.7, color='coral')
        
        # Add mean line
        ax.axvline(stats['mean'], color='red', linestyle='--', linewidth=2,
                  label=f"Mean: {stats['mean']:.3f}")
        
        # Add median line
        ax.axvline(stats['median'], color='blue', linestyle='--', linewidth=2,
                  label=f"Median: {stats['median']:.3f}")
        
        ax.set_title(f'GAT Layer {layer_idx + 1}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Attention Weight', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add stats text
        stats_text = f"Min: {stats['min']:.3f}\nMax: {stats['max']:.3f}\nStd: {stats['std']:.3f}"
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=9, family='monospace')
    
    plt.suptitle('Attention Weight Distributions by Layer', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved attention distributions to {save_path}")
    
    plt.close()


def plot_average_attention(model, data_loader, device, num_nodes: int,
                          save_dir: Path, channel_names: Optional[List[str]] = None):
    """
    Plot average attention matrix per layer across multiple graphs.
    
    Args:
        model: GAT model
        data_loader: DataLoader with graphs
        device: torch device
        num_nodes: Number of nodes in each graph
        save_dir: Directory to save plots
        channel_names: Optional list of node names
    """
    model.eval()
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Accumulate attention matrices per layer
    layer_matrices = {i: [] for i in range(len(model.conv_layers))}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= 20:  # Process 20 batches
                break
            
            batch = batch.to(device)
            all_attentions = model.get_all_attention_weights(batch)
            
            for layer_idx, (edge_index, alpha) in enumerate(all_attentions):
                # Average across heads
                if alpha.dim() > 1:
                    alpha = alpha.mean(dim=1)
                
                # Convert to adjacency matrix for this batch
                for graph_idx in range(batch.num_graphs):
                    # Get edges for this graph
                    mask = (batch.batch[edge_index[0]] == graph_idx)
                    graph_edge_index = edge_index[:, mask]
                    graph_alpha = alpha[mask]
                    
                    # Create adjacency matrix
                    adj = torch.zeros((num_nodes, num_nodes), device=device)
                    for i in range(graph_edge_index.shape[1]):
                        src = graph_edge_index[0, i]
                        dst = graph_edge_index[1, i]
                        # Map to graph-local indices
                        src_local = src % num_nodes
                        dst_local = dst % num_nodes
                        adj[src_local, dst_local] = graph_alpha[i]
                    
                    layer_matrices[layer_idx].append(adj.cpu().numpy())
    
    # Compute average and plot
    for layer_idx, matrices in layer_matrices.items():
        if len(matrices) == 0:
            continue
        
        # Average across all graphs
        avg_matrix = np.mean(matrices, axis=0)
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        
        mask = (avg_matrix == 0)
        sns.heatmap(avg_matrix, mask=mask, cmap='YlOrRd',
                   cbar_kws={'label': 'Average Attention Weight'},
                   square=True, linewidths=0.1, linecolor='gray',
                   ax=ax, vmin=0, vmax=avg_matrix.max())
        
        ax.set_title(f'Average Attention Weights - Layer {layer_idx + 1}',
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Target Node', fontsize=12)
        ax.set_ylabel('Source Node', fontsize=12)
        
        if channel_names is not None:
            ax.set_xticks(np.arange(num_nodes) + 0.5)
            ax.set_yticks(np.arange(num_nodes) + 0.5)
            ax.set_xticklabels(channel_names, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(channel_names, rotation=0, fontsize=8)
        
        plt.tight_layout()
        
        save_path = save_dir / f'attention_layer_{layer_idx + 1}_average.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved average attention for layer {layer_idx + 1} to {save_path}")


def log_attention_to_tensorboard(writer, model, data_loader, device, epoch: int,
                                 num_samples: int = 5):
    """
    Log attention weights to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter
        model: GAT model
        data_loader: DataLoader with graphs
        device: torch device
        epoch: Current epoch number
        num_samples: Number of sample graphs to visualize
    """
    model.eval()
    
    # Extract attention statistics
    attention_stats = extract_attention_matrices(model, data_loader, device, num_samples=100)
    
    # Log scalar statistics per layer
    for layer_key, stats in attention_stats.items():
        if not layer_key.startswith('layer_'):
            continue
        
        layer_idx = int(layer_key.split('_')[1])
        writer.add_scalar(f'Attention/{layer_key}/mean', stats['mean'], epoch)
        writer.add_scalar(f'Attention/{layer_key}/std', stats['std'], epoch)
        writer.add_scalar(f'Attention/{layer_key}/max', stats['max'], epoch)
        writer.add_scalar(f'Attention/{layer_key}/min', stats['min'], epoch)
    
    # Log histograms
    for layer_key, stats in attention_stats.items():
        if not layer_key.startswith('layer_'):
            continue
        
        writer.add_histogram(f'Attention/{layer_key}/distribution',
                            stats['values'], epoch)
    
    # Log sample attention heatmaps as images
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= num_samples:
                break
            
            batch = batch.to(device)
            all_attentions = model.get_all_attention_weights(batch)
            
            # Take first graph from batch
            for layer_idx, (edge_index, alpha) in enumerate(all_attentions):
                # Get edges for first graph only
                mask = (batch.batch[edge_index[0]] == 0)
                graph_edge_index = edge_index[:, mask]
                graph_alpha = alpha[mask]
                
                # Average across heads
                if graph_alpha.dim() > 1:
                    graph_alpha = graph_alpha.mean(dim=1)
                
                # Create adjacency matrix
                num_nodes = int(batch.ptr[1] - batch.ptr[0])
                adj = torch.zeros((num_nodes, num_nodes))
                
                for i in range(graph_edge_index.shape[1]):
                    src = int(graph_edge_index[0, i])
                    dst = int(graph_edge_index[1, i])
                    adj[src, dst] = graph_alpha[i].item()
                
                # Convert to image and log
                fig, ax = plt.subplots(figsize=(8, 7))
                sns.heatmap(adj.numpy(), cmap='YlOrRd', square=True,
                           cbar_kws={'label': 'Attention'}, ax=ax)
                ax.set_title(f'Layer {layer_idx + 1} - Sample {batch_idx + 1}')
                
                # Convert figure to numpy array
                fig.canvas.draw()
                img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                img = img.transpose(2, 0, 1)  # CHW format for tensorboard
                
                writer.add_image(f'Attention/Layer_{layer_idx + 1}/Sample_{batch_idx + 1}',
                               img, epoch)
                
                plt.close(fig)
            
            break  # Only process first batch for images
    
    logger.info(f"Logged attention weights to TensorBoard for epoch {epoch}")

