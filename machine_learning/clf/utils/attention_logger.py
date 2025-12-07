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

# Use non-GUI backend to avoid tkinter threading issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Batch
import pickle

logger = logging.getLogger(__name__)

# Path to electrode info
EXTRA_PKL_PATH = Path('/media/storage_hdd/dmt_fz/fwd-inv-stc/extra.pkl')


def load_electrode_names() -> Optional[List[str]]:
    """
    Load electrode names from extra.pkl.
    
    Returns:
        List of electrode names or None if file not found
    """
    if not EXTRA_PKL_PATH.exists():
        logger.warning(f"Electrode info file not found: {EXTRA_PKL_PATH}")
        return None
    
    try:
        with open(EXTRA_PKL_PATH, 'rb') as f:
            data = pickle.load(f)
        ch_names = data[4]  # Channel names are at index 4
        return ch_names
    except Exception as e:
        logger.warning(f"Failed to load electrode names: {e}")
        return None


def format_electrode_labels(ch_names: List[str]) -> List[str]:
    """
    Format electrode names with their index for axis labels.
    
    Args:
        ch_names: List of electrode names
        
    Returns:
        List of formatted labels like "FP1-0", "FP2-1", etc.
    """
    return [f"{name}-{i}" for i, name in enumerate(ch_names)]


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
        channel_names: Optional list of electrode names (will auto-load if None)
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
    
    # Load electrode names if not provided
    if channel_names is None:
        channel_names = load_electrode_names()
    
    # Format labels with index: "Name-Index"
    if channel_names is not None and len(channel_names) >= num_nodes:
        axis_labels = format_electrode_labels(channel_names[:num_nodes])
    else:
        axis_labels = [str(i) for i in range(num_nodes)]
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Use mask for zero values (no edges)
    mask = (adj_matrix == 0)
    
    sns.heatmap(adj_matrix, mask=mask, cmap='YlOrRd', 
                cbar_kws={'label': 'Attention Weight'},
                square=True, linewidths=0.1, linecolor='gray',
                ax=ax, vmin=0, vmax=1,
                xticklabels=axis_labels,
                yticklabels=axis_labels)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Target Electrode', fontsize=12)
    ax.set_ylabel('Source Electrode', fontsize=12)
    
    # Rotate labels for readability
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
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
        channel_names: Optional list of electrode names (will auto-load if None)
    """
    model.eval()
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Detect large graphs and configure accordingly
    is_large_graph = num_nodes > 50
    if is_large_graph:
        max_batches = 8
        fig_size = (10, 8)
        fig_dpi = 100
        show_labels = False
        line_widths = 0
    else:
        max_batches = 20
        fig_size = (14, 12)
        fig_dpi = 300
        show_labels = True
        line_widths = 0.1
    
    # Load electrode names if not provided
    if channel_names is None:
        channel_names = load_electrode_names()
    
    # Format labels with index: "Name-Index"
    if show_labels and channel_names is not None and len(channel_names) >= num_nodes:
        axis_labels = format_electrode_labels(channel_names[:num_nodes])
    elif show_labels:
        axis_labels = [str(i) for i in range(num_nodes)]
    else:
        axis_labels = False
    
    # Accumulate attention matrices per layer
    layer_matrices = {i: [] for i in range(len(model.conv_layers))}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= max_batches:
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
        fig, ax = plt.subplots(figsize=fig_size)
        
        mask = (avg_matrix == 0)
        sns.heatmap(avg_matrix, mask=mask, cmap='YlOrRd',
                   cbar_kws={'label': 'Average Attention Weight'},
                   square=True, linewidths=line_widths, linecolor='gray',
                   ax=ax, vmin=0, vmax=avg_matrix.max(),
                   xticklabels=axis_labels,
                   yticklabels=axis_labels)
        
        ax.set_title(f'Average Attention Weights - Layer {layer_idx + 1}',
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Target Node', fontsize=12)
        ax.set_ylabel('Source Node', fontsize=12)
        
        # Rotate labels for readability
        if show_labels:
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.yticks(rotation=0, fontsize=8)
        
        plt.tight_layout()
        
        save_path = save_dir / f'attention_layer_{layer_idx + 1}_average.png'
        plt.savefig(save_path, dpi=fig_dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved average attention for layer {layer_idx + 1} to {save_path}")


def log_attention_to_tensorboard(writer, model, data_loader, device, epoch: int,
                                 num_samples: int = 5, class_names: List[str] = None,
                                 save_dir: Optional[Path] = None):
    """
    Log attention weights to TensorBoard with electrode names, one example per class.
    
    Generates 2 layers × N classes heatmaps with proper electrode labels.
    
    Args:
        writer: TensorBoard SummaryWriter
        model: GAT model
        data_loader: DataLoader with graphs
        device: torch device
        epoch: Current epoch number
        num_samples: Number of sample graphs to visualize (deprecated, now uses 1 per class)
        class_names: List of class names (default: ['DMT', 'EC', 'EO'])
        save_dir: Optional directory to save attention images as files
    """
    if class_names is None:
        class_names = ['DMT', 'EC', 'EO']
    
    model.eval()
    
    # Load electrode names
    ch_names = load_electrode_names()
    
    # Get number of nodes to detect large graphs
    first_batch = next(iter(data_loader))
    num_nodes_detect = first_batch.x.shape[0] // first_batch.num_graphs if first_batch.num_graphs > 0 else 24
    is_large_graph = num_nodes_detect > 50
    
    # Reduce samples for large graphs
    stats_samples = 30 if is_large_graph else 100
    
    # Extract attention statistics
    attention_stats = extract_attention_matrices(model, data_loader, device, num_samples=stats_samples)
    
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
    
    # Find one example per class
    class_examples = {name: None for name in class_names}
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            
            # Check each graph in batch
            for graph_idx in range(batch.num_graphs):
                label = batch.y[graph_idx].item()
                if label < len(class_names):
                    class_name = class_names[label]
                    if class_examples[class_name] is None:
                        # Store this graph's data
                        start_idx = batch.ptr[graph_idx].item()
                        end_idx = batch.ptr[graph_idx + 1].item()
                        num_nodes = end_idx - start_idx
                        
                        # Get attention for all layers
                        all_attentions = model.get_all_attention_weights(batch)
                        
                        layer_data = []
                        for layer_idx, (edge_index, alpha) in enumerate(all_attentions):
                            # Get edges for this graph
                            edge_mask = (edge_index[0] >= start_idx) & (edge_index[0] < end_idx)
                            graph_edges = edge_index[:, edge_mask] - start_idx
                            graph_alpha = alpha[edge_mask]
                            
                            # Average across heads
                            if graph_alpha.dim() > 1:
                                graph_alpha = graph_alpha.mean(dim=1)
                            
                            # Build attention matrix
                            att_matrix = np.zeros((num_nodes, num_nodes))
                            for i in range(graph_edges.shape[1]):
                                src = graph_edges[0, i].item()
                                dst = graph_edges[1, i].item()
                                att_matrix[src, dst] = graph_alpha[i].item()
                            
                            layer_data.append(att_matrix)
                        
                        class_examples[class_name] = {
                            'num_nodes': num_nodes,
                            'layers': layer_data
                        }
            
            # Check if we have all classes
            if all(v is not None for v in class_examples.values()):
                break
    
    # Create electrode labels
    num_nodes = next((v['num_nodes'] for v in class_examples.values() if v), 24)
    
    # Configure based on graph size
    is_large = num_nodes > 50
    if is_large:
        show_labels = False
        fig_size = (10, 8)
        fig_dpi = 100
        line_widths = 0  # Skip grid lines for large heatmaps
    else:
        show_labels = True
        fig_size = (12, 10)
        fig_dpi = 150
        line_widths = 0.1
    
    if show_labels and ch_names and len(ch_names) >= num_nodes:
        axis_labels = format_electrode_labels(ch_names[:num_nodes])
    elif show_labels:
        axis_labels = [str(i) for i in range(num_nodes)]
    else:
        axis_labels = False  # seaborn will hide labels
    
    # Create save directory if specified
    if save_dir:
        attention_save_dir = Path(save_dir) / 'attention' / f'epoch_{epoch}'
        attention_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Log one heatmap per class per layer
    for class_name, data in class_examples.items():
        if data is None:
            logger.warning(f"No example found for class {class_name}")
            continue
        
        for layer_idx, att_matrix in enumerate(data['layers']):
            # Create figure with electrode labels
            fig, ax = plt.subplots(figsize=fig_size)
            
            mask = (att_matrix == 0)
            sns.heatmap(att_matrix, mask=mask, cmap='YlOrRd', square=True,
                       cbar_kws={'label': 'Attention Weight'},
                       ax=ax, vmin=0, vmax=att_matrix.max() if att_matrix.max() > 0 else 1,
                       xticklabels=axis_labels,
                       yticklabels=axis_labels,
                       linewidths=line_widths, linecolor='gray')
            
            title = f'Attention - {class_name} - Layer {layer_idx + 1} (Epoch {epoch})'
            ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel('Target Node', fontsize=11)
            ax.set_ylabel('Source Node', fontsize=11)
            
            # Rotate labels for readability
            if show_labels:
                plt.xticks(rotation=45, ha='right', fontsize=7)
                plt.yticks(rotation=0, fontsize=7)
            
            plt.tight_layout()
            
            # Convert figure to numpy array for TensorBoard
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = img.transpose(2, 0, 1)  # CHW format for tensorboard
            
            # Log to TensorBoard with clear naming (individual example)
            tag = f'Attention_Example/{class_name}/Layer_{layer_idx + 1}'
            writer.add_image(tag, img, epoch)
            
            # Save as file if directory specified
            if save_dir:
                save_path = attention_save_dir / f'attention_example_{class_name}_layer_{layer_idx + 1}.png'
                plt.savefig(save_path, dpi=fig_dpi, bbox_inches='tight')
                logger.debug(f"Saved attention heatmap to {save_path}")
            
            plt.close(fig)
    
    # Log summary message
    num_classes_logged = sum(1 for v in class_examples.values() if v is not None)
    num_layers = len(next((v['layers'] for v in class_examples.values() if v), []))
    logger.info(f"Logged {num_classes_logged * num_layers} attention heatmaps to TensorBoard for epoch {epoch} "
                f"({num_classes_logged} classes × {num_layers} layers)")


def extract_embeddings_for_analysis(model, data_loader, device, 
                                     num_samples: int = None,
                                     class_names: List[str] = None) -> Dict[str, np.ndarray]:
    """
    Extract all embeddings from the model for statistical analysis.
    
    Args:
        model: GAT model with get_embeddings method
        data_loader: DataLoader with graphs
        device: torch device
        num_samples: Max samples to extract (None = all)
        class_names: List of class names for label mapping
        
    Returns:
        Dict with:
            - 'graph_GAT_embedding': Pre-MLP embeddings after GAT layers + pooling [num_graphs, pooled_dim]
            - 'graph_GAT_features_embedding': With graph-level features [num_graphs, mlp_input_dim]
            - 'graph_GAT_MLP_embedding': Post-MLP outputs (logits) [num_graphs, num_classes]
            - 'predictions': Predicted class indices [num_graphs]
            - 'labels': Ground truth labels [num_graphs]
            - 'conditions': String class names [num_graphs]
            - 'subjects': Subject IDs [num_graphs]
            - 'bands': Frequency bands [num_graphs]
            - 'epoch_indices': Epoch indices [num_graphs]
    """
    if class_names is None:
        class_names = ['DMT', 'EC', 'EO']
    
    model.eval()
    
    all_graph_embeddings = []
    all_graph_embeddings_with_features = []
    all_logits = []
    all_labels = []
    all_conditions = []
    all_subjects = []
    all_bands = []
    all_epoch_indices = []
    
    count = 0
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            
            # Get all embeddings
            embeddings_dict = model.get_embeddings(batch)
            
            all_graph_embeddings.append(embeddings_dict['graph_embeddings'].cpu().numpy())
            all_graph_embeddings_with_features.append(embeddings_dict['graph_embeddings_with_features'].cpu().numpy())
            all_logits.append(embeddings_dict['logits'].cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())
            
            # Extract metadata for each graph in batch
            for i in range(batch.num_graphs):
                label_idx = batch.y[i].item()
                all_conditions.append(class_names[label_idx] if label_idx < len(class_names) else f"Class_{label_idx}")
                
                # Handle subject_id - could be list, tensor, or single value
                if hasattr(batch, 'subject_id'):
                    try:
                        if isinstance(batch.subject_id, (list, tuple)):
                            subj = batch.subject_id[i]
                        elif hasattr(batch.subject_id, '__getitem__') and len(batch.subject_id) > 1:
                            subj = batch.subject_id[i]
                            if hasattr(subj, 'item'):
                                subj = subj.item()
                        else:
                            subj = batch.subject_id if not hasattr(batch.subject_id, 'item') else batch.subject_id.item()
                        all_subjects.append(str(subj))
                    except:
                        all_subjects.append('unknown')
                else:
                    all_subjects.append('unknown')
                
                # Handle band - could be list, tensor, or single value  
                if hasattr(batch, 'band'):
                    try:
                        if isinstance(batch.band, (list, tuple)):
                            band = batch.band[i]
                        elif hasattr(batch.band, '__getitem__') and hasattr(batch.band, '__len__') and len(batch.band) > 1:
                            band = batch.band[i]
                            if hasattr(band, 'item'):
                                band = band.item()
                        else:
                            band = batch.band if not hasattr(batch.band, 'item') else batch.band.item()
                        all_bands.append(str(band))
                    except:
                        all_bands.append('unknown')
                else:
                    all_bands.append('unknown')
                
                # Handle epoch_idx
                if hasattr(batch, 'epoch_idx'):
                    try:
                        if isinstance(batch.epoch_idx, (list, tuple)):
                            epoch_idx = batch.epoch_idx[i]
                        elif hasattr(batch.epoch_idx, '__getitem__') and hasattr(batch.epoch_idx, '__len__') and len(batch.epoch_idx) > 1:
                            epoch_idx = batch.epoch_idx[i]
                            if hasattr(epoch_idx, 'item'):
                                epoch_idx = epoch_idx.item()
                        else:
                            epoch_idx = batch.epoch_idx if not hasattr(batch.epoch_idx, 'item') else batch.epoch_idx.item()
                        all_epoch_indices.append(int(epoch_idx))
                    except:
                        all_epoch_indices.append(-1)
                else:
                    all_epoch_indices.append(-1)
            
            count += batch.num_graphs
            if num_samples and count >= num_samples:
                break
    
    graph_gat_embedding = np.vstack(all_graph_embeddings)
    graph_gat_mlp_embedding = np.vstack(all_logits)
    predictions = np.argmax(graph_gat_mlp_embedding, axis=1)
    
    return {
        'graph_GAT_embedding': graph_gat_embedding,
        'graph_GAT_features_embedding': np.vstack(all_graph_embeddings_with_features),
        'graph_GAT_MLP_embedding': graph_gat_mlp_embedding,
        'predictions': predictions,
        'labels': np.concatenate(all_labels),
        'conditions': np.array(all_conditions),
        'subjects': np.array(all_subjects),
        'bands': np.array(all_bands),
        'epoch_indices': np.array(all_epoch_indices)
    }


def log_embeddings_to_tensorboard(writer, model, data_loader, device, epoch: int,
                                   num_samples: int = 500, class_names: List[str] = None):
    """
    Log graph embeddings to TensorBoard for visualization (t-SNE/UMAP projector).
    Uses STRATIFIED sampling to ensure all classes are represented.
    
    Logs 3 types of embeddings:
      - GAT_pooled: After GAT layers + pooling (before graph features)
      - GAT_features: After concatenating graph-level features
      - GAT_MLP: After MLP (logits)
    
    Args:
        writer: TensorBoard SummaryWriter
        model: GAT model with get_embeddings method
        data_loader: DataLoader with graphs
        device: torch device
        epoch: Current epoch number
        num_samples: Number of samples to visualize (per class, to ensure balance)
        class_names: List of class names
    """
    if class_names is None:
        class_names = ['DMT', 'EC', 'EO']
    
    # Extract ALL embeddings first (no limit)
    emb_data = extract_embeddings_for_analysis(model, data_loader, device, num_samples=None, class_names=class_names)
    
    # STRATIFIED sampling: equal samples per class
    conditions = emb_data['conditions']
    samples_per_class = num_samples // len(class_names)
    
    # Find indices for each class
    selected_indices = []
    for class_name in class_names:
        class_indices = [i for i, c in enumerate(conditions) if c == class_name]
        if len(class_indices) > samples_per_class:
            # Random sample
            import random
            random.seed(42)  # Reproducible
            class_indices = random.sample(class_indices, samples_per_class)
        selected_indices.extend(class_indices)
    
    # Subsample all arrays
    selected_indices = sorted(selected_indices)
    
    emb_data_sampled = {
        'graph_GAT_embedding': emb_data['graph_GAT_embedding'][selected_indices],
        'graph_GAT_features_embedding': emb_data['graph_GAT_features_embedding'][selected_indices],
        'graph_GAT_MLP_embedding': emb_data['graph_GAT_MLP_embedding'][selected_indices],
        'conditions': emb_data['conditions'][selected_indices],
    }
    
    # Create metadata labels
    label_list = [str(cond) for cond in emb_data_sampled['conditions']]
    
    # Log class distribution
    from collections import Counter
    class_counts = Counter(label_list)
    logger.info(f"Embedding class distribution (stratified): {dict(class_counts)}")
    
    # Define embedding types to log
    embedding_types = [
        ('GAT_pooled', 'graph_GAT_embedding'),           # After pooling, before features
        ('GAT_features', 'graph_GAT_features_embedding'), # With graph features concatenated  
        ('GAT_MLP', 'graph_GAT_MLP_embedding'),           # After MLP (logits)
    ]
    
    logged_count = 0
    for tag_name, emb_key in embedding_types:
        if emb_key not in emb_data_sampled or emb_data_sampled[emb_key] is None:
            continue
            
        emb_tensor = torch.tensor(emb_data_sampled[emb_key], dtype=torch.float32)
        
        try:
            writer.add_embedding(
                mat=emb_tensor,
                metadata=label_list,
                global_step=epoch,
                tag=tag_name
            )
            logged_count += 1
        except Exception as e:
            logger.warning(f"Failed to log {tag_name} embeddings: {e}")
    
    writer.flush()
    logger.info(f"Logged {logged_count} embedding types ({len(label_list)} samples each) for epoch {epoch}")


def save_embeddings_to_file(model, data_loader, device, save_path: Path,
                            num_samples: int = None, class_names: List[str] = None):
    """
    Save all embeddings to a pickle file for later analysis.
    
    Args:
        model: GAT model with get_embeddings method
        data_loader: DataLoader with graphs
        device: torch device
        save_path: Path to save the embeddings
        num_samples: Max samples (None = all)
        class_names: List of class names
        
    Saves dict with:
        - graph_GAT_embedding: Embeddings after GAT layers + pooling
        - graph_GAT_features_embedding: With graph-level features added
        - graph_GAT_MLP_embedding: Post-MLP outputs (logits, before softmax)
        - predictions: Predicted class indices
        - labels: Ground truth labels
        - conditions: String class names
        - subjects: Subject IDs
        - bands: Frequency bands
    """
    import pickle
    
    if class_names is None:
        class_names = ['DMT', 'EC', 'EO']
    
    emb_data = extract_embeddings_for_analysis(model, data_loader, device, num_samples, class_names)
    
    # Add accuracy info
    correct = (emb_data['predictions'] == emb_data['labels']).sum()
    total = len(emb_data['labels'])
    emb_data['accuracy'] = correct / total
    emb_data['class_names'] = class_names
    
    with open(save_path, 'wb') as f:
        pickle.dump(emb_data, f)
    
    logger.info(f"Saved {total} embeddings to {save_path} (accuracy: {emb_data['accuracy']:.4f})")
    
    return emb_data


def compute_attention_matrix_per_class(model, data_loader, device, num_nodes: int,
                                        class_names: List[str] = None,
                                        max_batches: int = None) -> Dict[str, np.ndarray]:
    """
    Compute average attention matrices per class for comparison.
    
    Args:
        model: GAT model
        data_loader: DataLoader
        device: torch device
        num_nodes: Number of nodes per graph
        class_names: List of class names
        max_batches: Maximum batches to process (None = auto-detect based on num_nodes)
        
    Returns:
        Dict mapping class name to average attention matrix per layer
    """
    if class_names is None:
        class_names = ['DMT', 'EC', 'EO']
    
    model.eval()
    num_layers = len(model.conv_layers)
    
    # Auto-detect max_batches based on graph size to avoid slow processing
    if max_batches is None:
        if num_nodes > 80:
            max_batches = 10  # STC: 102 nodes - use fewer batches
        elif num_nodes > 40:
            max_batches = 20
        else:
            max_batches = None  # EEG: 24 nodes - use all
    
    # Initialize accumulators
    attention_sums = {name: {l: np.zeros((num_nodes, num_nodes)) for l in range(num_layers)} 
                      for name in class_names}
    counts = {name: 0 for name in class_names}
    
    batch_count = 0
    with torch.no_grad():
        for batch in data_loader:
            if max_batches is not None and batch_count >= max_batches:
                break
            batch_count += 1
            batch = batch.to(device)
            
            # Get attention weights
            all_attentions = model.get_all_attention_weights(batch)
            
            # Process each graph in batch
            for graph_idx in range(batch.num_graphs):
                label = batch.y[graph_idx].item()
                class_name = class_names[label]
                
                # Get node range for this graph
                start_idx = batch.ptr[graph_idx].item()
                end_idx = batch.ptr[graph_idx + 1].item()
                
                for layer_idx, (edge_index, alpha) in enumerate(all_attentions):
                    # Get edges for this graph
                    edge_mask = (edge_index[0] >= start_idx) & (edge_index[0] < end_idx)
                    graph_edges = edge_index[:, edge_mask] - start_idx
                    graph_alpha = alpha[edge_mask]
                    
                    # Average across heads
                    if graph_alpha.dim() > 1:
                        graph_alpha = graph_alpha.mean(dim=1)
                    
                    # Build attention matrix
                    att_matrix = np.zeros((num_nodes, num_nodes))
                    for i in range(graph_edges.shape[1]):
                        src = graph_edges[0, i].item()
                        dst = graph_edges[1, i].item()
                        att_matrix[src, dst] = graph_alpha[i].item()
                    
                    attention_sums[class_name][layer_idx] += att_matrix
                
                counts[class_name] += 1
    
    # Average
    attention_per_class = {}
    for class_name in class_names:
        if counts[class_name] > 0:
            attention_per_class[class_name] = {
                l: attention_sums[class_name][l] / counts[class_name]
                for l in range(num_layers)
            }
    
    logger.info(f"Computed average attention for {sum(counts.values())} graphs across {len(class_names)} classes")
    
    return attention_per_class


def plot_attention_per_class(attention_per_class: Dict[str, Dict[int, np.ndarray]],
                             save_dir: Path,
                             class_names: List[str] = None):
    """
    Plot average attention heatmaps for each class and their differences.
    
    Args:
        attention_per_class: Dict from compute_attention_matrix_per_class
        save_dir: Directory to save plots
        class_names: List of class names
    """
    if not attention_per_class:
        logger.warning("No attention data to plot")
        return
    
    if class_names is None:
        class_names = list(attention_per_class.keys())
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load electrode names
    ch_names = load_electrode_names()
    num_nodes = list(list(attention_per_class.values())[0].values())[0].shape[0]
    axis_labels = format_electrode_labels(ch_names[:num_nodes]) if ch_names else [str(i) for i in range(num_nodes)]
    
    num_layers = len(list(attention_per_class.values())[0])
    
    # 1. Plot average attention per class per layer
    for layer_idx in range(num_layers):
        fig, axes = plt.subplots(1, len(class_names), figsize=(7*len(class_names), 6))
        if len(class_names) == 1:
            axes = [axes]
        
        vmax = max(attention_per_class[c][layer_idx].max() for c in class_names if c in attention_per_class)
        
        for idx, class_name in enumerate(class_names):
            if class_name not in attention_per_class:
                continue
            
            att_matrix = attention_per_class[class_name][layer_idx]
            
            ax = axes[idx]
            sns.heatmap(att_matrix, cmap='YlOrRd', square=True, ax=ax,
                       vmin=0, vmax=vmax,
                       cbar_kws={'label': 'Attention', 'shrink': 0.8},
                       xticklabels=axis_labels, yticklabels=axis_labels)
            ax.set_title(f'{class_name} - Layer {layer_idx + 1}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Target', fontsize=10)
            ax.set_ylabel('Source', fontsize=10)
            ax.tick_params(axis='both', labelsize=6)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.suptitle(f'Average Attention by Class - Layer {layer_idx + 1}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_dir / f'attention_per_class_layer{layer_idx + 1}.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved per-class attention for layer {layer_idx + 1}")
    
    # 2. Plot differences between classes (for all pairs)
    from itertools import combinations
    class_pairs = list(combinations(class_names, 2))
    
    for layer_idx in range(num_layers):
        num_pairs = len(class_pairs)
        fig, axes = plt.subplots(1, num_pairs, figsize=(7*num_pairs, 6))
        if num_pairs == 1:
            axes = [axes]
        
        for pair_idx, (class_a, class_b) in enumerate(class_pairs):
            if class_a not in attention_per_class or class_b not in attention_per_class:
                continue
            
            diff_matrix = attention_per_class[class_a][layer_idx] - attention_per_class[class_b][layer_idx]
            
            ax = axes[pair_idx]
            vmax_diff = max(abs(diff_matrix.min()), abs(diff_matrix.max()))
            
            sns.heatmap(diff_matrix, cmap='RdBu_r', square=True, ax=ax,
                       center=0, vmin=-vmax_diff, vmax=vmax_diff,
                       cbar_kws={'label': f'{class_a} - {class_b}', 'shrink': 0.8},
                       xticklabels=axis_labels, yticklabels=axis_labels)
            ax.set_title(f'{class_a} vs {class_b}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Target', fontsize=10)
            ax.set_ylabel('Source', fontsize=10)
            ax.tick_params(axis='both', labelsize=6)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.suptitle(f'Attention Differences - Layer {layer_idx + 1}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_dir / f'attention_diff_layer{layer_idx + 1}.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved attention differences for layer {layer_idx + 1}")


def extract_attention_per_class(model, data_loader, device, class_names: List[str] = None,
                                max_batches: int = None) -> Dict:
    """
    Extract attention weight distributions per class for histogram plotting.
    
    Args:
        model: GAT model
        data_loader: DataLoader
        device: torch device
        class_names: List of class names
        max_batches: Maximum batches to process (None = auto-detect based on num_nodes)
        
    Returns:
        Dict with attention values per class per layer
    """
    if class_names is None:
        class_names = ['DMT', 'EC', 'EO']
    
    model.eval()
    num_layers = len(model.conv_layers)
    
    # Auto-detect max_batches based on first batch's graph size
    first_batch = next(iter(data_loader))
    num_nodes = first_batch.x.shape[0] // first_batch.num_graphs if first_batch.num_graphs > 0 else 24
    
    if max_batches is None:
        if num_nodes > 80:
            max_batches = 10  # STC: 102 nodes - use fewer batches
        elif num_nodes > 40:
            max_batches = 20
        else:
            max_batches = None  # EEG: 24 nodes - use all
    
    # Initialize storage
    attention_values = {name: {l: [] for l in range(num_layers)} for name in class_names}
    
    batch_count = 0
    with torch.no_grad():
        for batch in data_loader:
            if max_batches is not None and batch_count >= max_batches:
                break
            batch_count += 1
            batch = batch.to(device)
            all_attentions = model.get_all_attention_weights(batch)
            
            for graph_idx in range(batch.num_graphs):
                label = batch.y[graph_idx].item()
                if label >= len(class_names):
                    continue
                class_name = class_names[label]
                
                start_idx = batch.ptr[graph_idx].item()
                end_idx = batch.ptr[graph_idx + 1].item()
                
                for layer_idx, (edge_index, alpha) in enumerate(all_attentions):
                    edge_mask = (edge_index[0] >= start_idx) & (edge_index[0] < end_idx)
                    graph_alpha = alpha[edge_mask]
                    
                    if graph_alpha.dim() > 1:
                        graph_alpha = graph_alpha.mean(dim=1)
                    
                    attention_values[class_name][layer_idx].extend(graph_alpha.cpu().numpy().tolist())
    
    return attention_values


def plot_attention_distributions_by_class(attention_per_class: Dict,
                                          save_path: Optional[Path] = None,
                                          class_names: List[str] = None):
    """
    Plot attention weight distributions with a line/curve per class.
    
    Args:
        attention_per_class: Dict from extract_attention_per_class
        save_path: Path to save figure
        class_names: List of class names
    """
    if class_names is None:
        class_names = list(attention_per_class.keys())
    
    num_layers = len(list(attention_per_class.values())[0])
    
    # Define colors for each class
    colors = {'DMT': '#E74C3C', 'EC': '#3498DB', 'EO': '#2ECC71'}
    default_colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
    
    fig, axes = plt.subplots(1, num_layers, figsize=(6*num_layers, 5))
    if num_layers == 1:
        axes = [axes]
    
    for layer_idx in range(num_layers):
        ax = axes[layer_idx]
        
        for idx, class_name in enumerate(class_names):
            if class_name not in attention_per_class:
                continue
            
            values = np.array(attention_per_class[class_name][layer_idx])
            if len(values) == 0:
                continue
            
            color = colors.get(class_name, default_colors[idx % len(default_colors)])
            
            # Plot KDE (density estimate) for smoother visualization
            from scipy import stats
            try:
                kde = stats.gaussian_kde(values)
                x_range = np.linspace(values.min(), values.max(), 200)
                ax.plot(x_range, kde(x_range), color=color, linewidth=2.5, 
                       label=f'{class_name} (μ={values.mean():.3f})')
                ax.fill_between(x_range, kde(x_range), alpha=0.2, color=color)
            except Exception:
                # Fallback to histogram if KDE fails
                ax.hist(values, bins=50, alpha=0.5, color=color, 
                       label=f'{class_name} (μ={values.mean():.3f})', density=True)
        
        ax.set_title(f'GAT Layer {layer_idx + 1}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Attention Weight', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Attention Weight Distributions by Class', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved attention distributions by class to {save_path}")
    
    plt.close()


def compute_mst_from_attention(attention_matrix: np.ndarray) -> np.ndarray:
    """
    Compute Minimum Spanning Tree from attention matrix.
    Uses negative weights to find Maximum Spanning Tree (strongest connections).
    
    Args:
        attention_matrix: NxN attention weight matrix
        
    Returns:
        NxN adjacency matrix with only MST edges
    """
    import networkx as nx
    
    n = attention_matrix.shape[0]
    G = nx.Graph()
    
    # Add edges with negative weights (to get maximum spanning tree)
    for i in range(n):
        for j in range(i + 1, n):
            weight = attention_matrix[i, j] + attention_matrix[j, i]  # Symmetric
            if weight > 0:
                G.add_edge(i, j, weight=-weight)  # Negative for max spanning tree
    
    # Compute MST
    if G.number_of_edges() > 0:
        mst = nx.minimum_spanning_tree(G)
        
        # Create MST adjacency matrix with original weights
        mst_matrix = np.zeros_like(attention_matrix)
        for i, j in mst.edges():
            # Use average of both directions
            avg_weight = (attention_matrix[i, j] + attention_matrix[j, i]) / 2
            mst_matrix[i, j] = avg_weight
            mst_matrix[j, i] = avg_weight
        
        return mst_matrix
    else:
        return np.zeros_like(attention_matrix)


def get_standard_eeg_coordinates():
    """
    Return FIXED 2D coordinates for standard 10-20 EEG electrodes.
    These coordinates are normalized and consistent across all plots.
    Based on standard scalp positions.
    """
    # Standard 10-20 electrode positions (normalized to fit in [-0.12, 0.12] range)
    # Arranged to match typical EEG cap layout viewed from above (nose up)
    coords = {
        # Frontal pole
        'Fp1': (-0.03, 0.12), 'Fp2': (0.03, 0.12), 'Fpz': (0.0, 0.13),
        # Anterior frontal
        'AFz': (0.0, 0.10),
        # Frontal
        'F7': (-0.09, 0.06), 'F3': (-0.05, 0.07), 'Fz': (0.0, 0.07),
        'F4': (0.05, 0.07), 'F8': (0.09, 0.06),
        # Fronto-central
        'FC1': (-0.03, 0.04), 'FCz': (0.0, 0.04), 'FC2': (0.03, 0.04),
        # Temporal
        'T7': (-0.11, 0.0), 'T8': (0.11, 0.0),
        # Central
        'C3': (-0.06, 0.0), 'Cz': (0.0, 0.0), 'C4': (0.06, 0.0),
        # Centro-parietal
        'CP1': (-0.03, -0.04), 'CPz': (0.0, -0.04), 'CP2': (0.03, -0.04),
        # Parietal
        'P7': (-0.09, -0.06), 'P3': (-0.05, -0.07), 'Pz': (0.0, -0.07),
        'P4': (0.05, -0.07), 'P8': (0.09, -0.06),
        # Parieto-occipital
        'POz': (0.0, -0.10),
        # Occipital
        'O1': (-0.03, -0.12), 'O2': (0.03, -0.12), 'Oz': (0.0, -0.13),
        # Mastoid
        'M1': (-0.12, -0.02), 'M2': (0.12, -0.02),
    }
    return coords


def plot_attention_mst_graph(mst_matrix: np.ndarray, 
                              class_name: str,
                              layer_idx: int,
                              save_path: Path,
                              ch_names: List[str] = None,
                              eeg_coords_2d: Dict = None):
    """
    Plot the Minimum Spanning Tree of attention as an EEG graph visualization.
    Uses FIXED electrode positions for consistent layout across all plots.
    
    Args:
        mst_matrix: NxN MST adjacency matrix
        class_name: Name of the condition (DMT, EC, EO)
        layer_idx: GAT layer index (0-based)
        save_path: Path to save the figure
        ch_names: List of electrode names
        eeg_coords_2d: Dict mapping electrode names to 2D coordinates (ignored, uses standard coords)
    """
    import networkx as nx
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    
    n = mst_matrix.shape[0]
    
    # Load electrode names if not provided
    if ch_names is None:
        ch_names_loaded = load_electrode_names()
        if ch_names_loaded:
            ch_names = ch_names_loaded
        else:
            ch_names = [f'E{i}' for i in range(n)]
    
    # ALWAYS use standard fixed coordinates for consistency
    standard_coords = get_standard_eeg_coordinates()
    
    # Map channel names to coordinates
    eeg_coords_2d = {}
    for i, name in enumerate(ch_names):
        if name in standard_coords:
            eeg_coords_2d[name] = standard_coords[name]
        else:
            # Fallback: use circular layout for unknown electrodes
            angle = 2 * np.pi * i / n
            radius = 0.10
            eeg_coords_2d[name] = (radius * np.sin(angle), radius * np.cos(angle))
    
    # Build graph with MST edges only
    G = nx.Graph()
    for i in range(min(n, len(ch_names))):
        G.add_node(ch_names[i])
    
    edge_weights = {}
    for i in range(n):
        for j in range(i + 1, n):
            if mst_matrix[i, j] > 0:
                if i < len(ch_names) and j < len(ch_names):
                    G.add_edge(ch_names[i], ch_names[j])
                    edge_weights[(ch_names[i], ch_names[j])] = mst_matrix[i, j]
    
    if len(edge_weights) == 0:
        logger.warning(f"No edges in MST for {class_name} layer {layer_idx + 1}")
        return
    
    # Create figure - smaller for TensorBoard viewing
    fig = plt.figure(figsize=(10, 9))
    
    # Main graph area
    ax_graph = fig.add_axes([0.05, 0.12, 0.9, 0.78])
    
    pos = eeg_coords_2d
    all_weights = list(edge_weights.values())
    weight_min, weight_max = min(all_weights), max(all_weights)
    
    # Normalize weights for visualization
    if weight_max > weight_min:
        weight_range = weight_max - weight_min
    else:
        weight_range = 1.0
        weight_min = 0.0
    
    # Sort edges by weight to draw stronger connections on top
    sorted_edges = sorted(edge_weights.items(), key=lambda x: x[1])
    
    # Draw edges - same style as visualize_real_graph.py
    for (src, dst), weight in sorted_edges:
        normalized = (weight - weight_min) / weight_range if weight_range > 0 else 0.5
        alpha = 0.3 + normalized * 0.7
        width = 1.0 + normalized * 5.0
        color = cm.Reds(0.3 + normalized * 0.7)  # Avoid too light colors
        nx.draw_networkx_edges(G, pos, edgelist=[(src, dst)],
                              width=width, edge_color=[color], alpha=alpha, ax=ax_graph)
    
    # Draw nodes - sized for smaller figure
    node_names_list = [ch_names[i] for i in range(min(n, len(ch_names)))]
    
    # All nodes same color (light gray) to highlight edges
    nx.draw_networkx_nodes(G, pos, nodelist=node_names_list,
                          node_color='lightgray',
                          node_size=1200,
                          edgecolors='black', linewidths=2.0, 
                          alpha=0.95, ax=ax_graph)
    
    # Draw labels with white background
    for node_name in node_names_list:
        if node_name in pos:
            x, y = pos[node_name]
            ax_graph.text(x, y, node_name, fontsize=8, fontweight='bold',
                         ha='center', va='center', color='black',
                         bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                                  edgecolor='black', linewidth=0.6, alpha=0.90))
    
    # Set FIXED axis limits for consistent layout across all plots
    ax_graph.axis('equal')
    ax_graph.set_xlim(-0.15, 0.15)
    ax_graph.set_ylim(-0.16, 0.16)
    ax_graph.axis('off')
    
    # Add colorbar at the bottom
    cbar_ax = fig.add_axes([0.15, 0.04, 0.7, 0.02])
    norm = Normalize(vmin=weight_min, vmax=weight_max)
    sm = cm.ScalarMappable(norm=norm, cmap=cm.Reds)
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(f'Attention [{weight_min:.4f}, {weight_max:.4f}]', 
                   fontsize=9, fontweight='bold')
    cbar.ax.tick_params(labelsize=8)
    
    # Title
    fig.text(0.5, 0.96, f'Attention MST - {class_name} - Layer {layer_idx + 1}',
            fontsize=12, fontweight='bold', ha='center', va='top')
    fig.text(0.5, 0.92, f'({len(edge_weights)} edges)',
            fontsize=9, ha='center', va='top')
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Saved MST graph: {save_path.name}")


def generate_full_attention_analysis(model, data_loader, device, save_dir: Path,
                                     class_names: List[str] = None, writer=None, epoch: int = 0):
    """
    Generate complete attention analysis with per-class averages and differences.
    
    Generates and logs to TensorBoard:
      - Per-class average attention heatmaps for each GAT layer
      - Attention difference heatmaps between class pairs
      - Attention weight distributions by class (KDE curves)
      - Minimum Spanning Tree graphs for each class/layer (skipped for large graphs)
    
    Args:
        model: GAT model
        data_loader: DataLoader
        device: torch device
        save_dir: Directory to save plots
        class_names: List of class names
        writer: TensorBoard SummaryWriter (optional)
        epoch: Current epoch for TensorBoard logging
    """
    from itertools import combinations
    
    if class_names is None:
        class_names = ['DMT', 'EC', 'EO']
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Get number of nodes from first batch
    first_batch = next(iter(data_loader))
    num_nodes = first_batch.x.shape[0] // first_batch.num_graphs
    
    # Detect if we're working with large graphs (STC mode)
    is_large_graph = num_nodes > 50
    
    # Configure plotting parameters based on graph size
    if is_large_graph:
        fig_dpi = 100  # Lower DPI for faster rendering
        heatmap_figsize = (10, 8)  # Smaller figures
        show_labels = False  # Skip axis labels for large heatmaps
        skip_mst = True  # Skip MST graphs (too slow with 102 nodes)
        logger.info(f"Large graph detected ({num_nodes} nodes), using optimized settings")
    else:
        fig_dpi = 200
        heatmap_figsize = (12, 10)
        show_labels = True
        skip_mst = False
    
    logger.info(f"Generating full attention analysis for {len(class_names)} classes...")
    
    # Load electrode labels
    ch_names = load_electrode_names()
    if show_labels:
        axis_labels = format_electrode_labels(ch_names[:num_nodes]) if ch_names else [str(i) for i in range(num_nodes)]
    else:
        axis_labels = False  # seaborn will hide labels
    
    # 1. Compute average attention matrices per class (with batch limit for large graphs)
    attention_per_class = compute_attention_matrix_per_class(
        model, data_loader, device, num_nodes, class_names
    )
    
    # 2. Plot and log per-class attention heatmaps
    num_layers = len(list(attention_per_class.values())[0]) if attention_per_class else 0
    
    for layer_idx in range(num_layers):
        vmax = max(attention_per_class[c][layer_idx].max() for c in class_names if c in attention_per_class)
        
        for class_name in class_names:
            if class_name not in attention_per_class:
                continue
            
            att_matrix = attention_per_class[class_name][layer_idx]
            
            fig, ax = plt.subplots(figsize=heatmap_figsize)
            sns.heatmap(att_matrix, cmap='YlOrRd', square=True, ax=ax,
                       vmin=0, vmax=vmax,
                       cbar_kws={'label': 'Attention Weight'},
                       xticklabels=axis_labels, yticklabels=axis_labels)
            ax.set_title(f'Average Attention - {class_name} - Layer {layer_idx + 1}', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Target Node', fontsize=11)
            ax.set_ylabel('Source Node', fontsize=11)
            if show_labels:
                plt.xticks(rotation=45, ha='right', fontsize=7)
                plt.yticks(fontsize=7)
            plt.tight_layout()
            
            # Save file
            file_path = save_dir / f'attention_{class_name.lower()}_layer{layer_idx + 1}.png'
            plt.savefig(file_path, dpi=fig_dpi, bbox_inches='tight')
            
            # Log to TensorBoard
            if writer is not None:
                fig.canvas.draw()
                img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                img_tensor = torch.tensor(img.transpose(2, 0, 1))
                writer.add_image(f'Attention_Average/{class_name}/Layer_{layer_idx + 1}', img_tensor, epoch)
            
            plt.close()
    
    # 3. Plot and log attention differences between class pairs
    class_pairs = list(combinations(class_names, 2))
    
    for layer_idx in range(num_layers):
        for class_a, class_b in class_pairs:
            if class_a not in attention_per_class or class_b not in attention_per_class:
                continue
            
            diff_matrix = attention_per_class[class_a][layer_idx] - attention_per_class[class_b][layer_idx]
            vmax_diff = max(abs(diff_matrix.min()), abs(diff_matrix.max()))
            
            fig, ax = plt.subplots(figsize=heatmap_figsize)
            sns.heatmap(diff_matrix, cmap='RdBu_r', square=True, ax=ax,
                       center=0, vmin=-vmax_diff, vmax=vmax_diff,
                       cbar_kws={'label': f'{class_a} - {class_b}'},
                       xticklabels=axis_labels, yticklabels=axis_labels)
            ax.set_title(f'Attention Difference: {class_a} vs {class_b} - Layer {layer_idx + 1}', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Target Node', fontsize=11)
            ax.set_ylabel('Source Node', fontsize=11)
            if show_labels:
                plt.xticks(rotation=45, ha='right', fontsize=7)
                plt.yticks(fontsize=7)
            plt.tight_layout()
            
            # Save file
            file_path = save_dir / f'attention_diff_{class_a.lower()}_vs_{class_b.lower()}_layer{layer_idx + 1}.png'
            plt.savefig(file_path, dpi=fig_dpi, bbox_inches='tight')
            
            # Log to TensorBoard
            if writer is not None:
                fig.canvas.draw()
                img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                img_tensor = torch.tensor(img.transpose(2, 0, 1))
                writer.add_image(f'Attention_Differences/{class_a}_vs_{class_b}/Layer_{layer_idx + 1}', img_tensor, epoch)
            
            plt.close()
    
    # 4. Extract attention distributions per class and plot (with batch limit for large graphs)
    attention_dist_per_class = extract_attention_per_class(model, data_loader, device, class_names)
    
    # Define colors for each class
    colors = {'DMT': '#E74C3C', 'EC': '#3498DB', 'EO': '#2ECC71'}
    default_colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
    
    fig, axes = plt.subplots(1, num_layers, figsize=(6*num_layers, 5))
    if num_layers == 1:
        axes = [axes]
    
    for layer_idx in range(num_layers):
        ax = axes[layer_idx]
        
        for idx, class_name in enumerate(class_names):
            if class_name not in attention_dist_per_class:
                continue
            
            values = np.array(attention_dist_per_class[class_name][layer_idx])
            if len(values) == 0:
                continue
            
            color = colors.get(class_name, default_colors[idx % len(default_colors)])
            
            # Plot KDE
            try:
                from scipy import stats
                kde = stats.gaussian_kde(values)
                x_range = np.linspace(values.min(), values.max(), 200)
                ax.plot(x_range, kde(x_range), color=color, linewidth=2.5, 
                       label=f'{class_name} (μ={values.mean():.3f})')
                ax.fill_between(x_range, kde(x_range), alpha=0.2, color=color)
            except Exception:
                ax.hist(values, bins=50, alpha=0.5, color=color, 
                       label=f'{class_name} (μ={values.mean():.3f})', density=True)
        
        ax.set_title(f'GAT Layer {layer_idx + 1}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Attention Weight', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Attention Weight Distributions by Class', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save distributions plot
    dist_path = save_dir / 'attention_distributions_by_class.png'
    plt.savefig(dist_path, dpi=fig_dpi, bbox_inches='tight')
    
    # Log to TensorBoard
    if writer is not None:
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img_tensor = torch.tensor(img.transpose(2, 0, 1))
        writer.add_image('Attention_Distributions/By_Class', img_tensor, epoch)
    
    plt.close()
    
    # 5. Generate Minimum Spanning Tree graphs for each class/layer
    # Skip MST for large graphs (STC mode) as it's very slow with 100+ nodes
    if skip_mst:
        logger.info(f"Skipping MST graphs for large graph ({num_nodes} nodes)")
    else:
        logger.info("Generating Minimum Spanning Tree graphs...")
        mst_dir = save_dir / 'mst_graphs'
        mst_dir.mkdir(parents=True, exist_ok=True)
        
        # Load electrode coordinates for graph visualization
        try:
            with open(EXTRA_PKL_PATH, 'rb') as f:
                extra_data = pickle.load(f)
            eeg_coords_2d = extra_data[6]
        except Exception as e:
            logger.warning(f"Could not load electrode coordinates: {e}")
            eeg_coords_2d = None
        
        for layer_idx in range(num_layers):
            for class_name in class_names:
                if class_name not in attention_per_class:
                    continue
                
                att_matrix = attention_per_class[class_name][layer_idx]
                
                # Compute MST
                mst_matrix = compute_mst_from_attention(att_matrix)
                
                # Save MST graph visualization
                mst_path = mst_dir / f'mst_{class_name.lower()}_layer{layer_idx + 1}.png'
                try:
                    plot_attention_mst_graph(
                        mst_matrix, 
                        class_name, 
                        layer_idx,
                        mst_path,
                        ch_names=ch_names[:num_nodes] if ch_names else None,
                        eeg_coords_2d=eeg_coords_2d
                    )
                    
                    # Log to TensorBoard
                    if writer is not None:
                        img = plt.imread(str(mst_path))
                        if img.ndim == 3 and img.shape[2] == 4:  # RGBA
                            img = img[:, :, :3]
                        img_tensor = torch.tensor(img.transpose(2, 0, 1))
                        writer.add_image(f'Attention_MST/{class_name}/Layer_{layer_idx + 1}', img_tensor, epoch)
                except Exception as e:
                    logger.warning(f"Failed to generate MST for {class_name} layer {layer_idx + 1}: {e}")
    
    if writer is not None:
        writer.flush()
    
    logger.info(f"Full attention analysis saved to {save_dir} and logged to TensorBoard")

