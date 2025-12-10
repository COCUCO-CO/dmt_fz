"""
Visualization utilities for VAE.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
from typing import Dict, List, Optional, Any
import pickle

import torch
from torch_geometric.loader import DataLoader
from sklearn.manifold import TSNE
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
from sklearn.decomposition import PCA

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def plot_training_curves(history: Dict[str, List[float]], 
                         save_path: Path,
                         title: str = "VAE Training Curves"):
    """
    Plot training and validation curves.
    
    Args:
        history: Dict with 'train_loss', 'val_loss', 'train_recon', etc.
        save_path: Path to save figure
        title: Plot title
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Total loss
    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Train', alpha=0.8)
    ax.plot(history['val_loss'], label='Val', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss (Recon + β·KL)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Reconstruction loss
    ax = axes[0, 1]
    ax.plot(history['train_recon_loss'], label='Train', alpha=0.8)
    ax.plot(history['val_recon_loss'], label='Val', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reconstruction Loss')
    ax.set_title('Reconstruction Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # KL divergence
    ax = axes[1, 0]
    ax.plot(history['train_kl_loss'], label='Train KL', alpha=0.8)
    ax.plot(history['val_kl_loss'], label='Val KL', alpha=0.8)
    if 'beta' in history:
        ax2 = ax.twinx()
        ax2.plot(history['beta'], 'g--', label='β', alpha=0.5)
        ax2.set_ylabel('β (KL weight)', color='g')
        ax2.legend(loc='upper left')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('KL Divergence')
    ax.set_title('KL Divergence Loss')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Learning rate
    ax = axes[1, 1]
    if 'lr' in history:
        ax.plot(history['lr'], label='Learning Rate', alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Learning rate not tracked', 
                ha='center', va='center', transform=ax.transAxes)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_latent_space(latent_vectors: np.ndarray,
                      labels: np.ndarray,
                      class_names: List[str],
                      save_path: Path,
                      method: str = 'tsne',
                      title: str = "Latent Space"):
    """
    Visualize latent space using dimensionality reduction.
    
    Args:
        latent_vectors: [N, latent_dim] array
        labels: [N] array of class labels
        class_names: Names of classes
        save_path: Path to save figure
        method: 'tsne', 'pca', or 'umap'
        title: Plot title
    """
    # Reduce dimensionality
    n_samples = latent_vectors.shape[0]
    if method == 'tsne':
        # Perplexity must be less than n_samples
        perplexity = min(30, max(5, n_samples - 1))
        if n_samples < 5:
            # Too few samples for t-SNE, fallback to PCA
            reducer = PCA(n_components=2)
            method = 'pca'
        else:
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embedding = reducer.fit_transform(latent_vectors)
    elif method == 'pca':
        reducer = PCA(n_components=2)
        embedding = reducer.fit_transform(latent_vectors)
    elif method == 'umap' and HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding = reducer.fit_transform(latent_vectors)
    else:
        # Fallback to PCA
        reducer = PCA(n_components=2)
        embedding = reducer.fit_transform(latent_vectors)
        method = 'pca'
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color palette
    colors = plt.cm.Set1(np.linspace(0, 1, len(class_names)))
    
    for i, class_name in enumerate(class_names):
        mask = labels == i
        ax.scatter(
            embedding[mask, 0], 
            embedding[mask, 1],
            c=[colors[i]],
            label=class_name,
            alpha=0.6,
            s=30
        )
    
    ax.set_xlabel(f'{method.upper()} 1')
    ax.set_ylabel(f'{method.upper()} 2')
    ax.set_title(f'{title} ({method.upper()})')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_reconstructions(original: np.ndarray,
                         reconstructed: np.ndarray,
                         save_path: Path,
                         num_samples: int = 5,
                         feature_names: List[str] = None):
    """
    Compare original and reconstructed node features.
    
    Args:
        original: [N, num_features] original features
        reconstructed: [N, num_features] reconstructed features
        save_path: Path to save figure
        num_samples: Number of samples to show
        feature_names: Names of features
    """
    num_features = original.shape[1]
    
    if feature_names is None:
        feature_names = [f'F{i}' for i in range(num_features)]
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    indices = np.random.choice(len(original), num_samples, replace=False)
    
    for ax_idx, idx in enumerate(indices):
        ax = axes[ax_idx]
        x = np.arange(num_features)
        
        ax.bar(x - 0.2, original[idx], 0.4, label='Original', alpha=0.7)
        ax.bar(x + 0.2, reconstructed[idx], 0.4, label='Reconstructed', alpha=0.7)
        
        ax.set_xlabel('Feature')
        ax.set_ylabel('Value')
        ax.set_title(f'Sample {idx}')
        ax.set_xticks(x)
        ax.set_xticklabels(feature_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Reconstruction Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_kl_per_dimension(kl_per_dim: np.ndarray,
                          save_path: Path,
                          title: str = "KL Divergence per Latent Dimension"):
    """
    Plot KL divergence per latent dimension.
    
    Args:
        kl_per_dim: [latent_dim] array
        save_path: Path to save figure
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar plot
    ax = axes[0]
    x = np.arange(len(kl_per_dim))
    colors = plt.cm.RdYlGn_r(kl_per_dim / (kl_per_dim.max() + 1e-8))
    ax.bar(x, kl_per_dim, color=colors)
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('KL Divergence')
    ax.set_title('KL per Dimension')
    ax.grid(True, alpha=0.3)
    
    # Sorted bar plot
    ax = axes[1]
    sorted_indices = np.argsort(kl_per_dim)[::-1]
    sorted_kl = kl_per_dim[sorted_indices]
    colors = plt.cm.RdYlGn_r(sorted_kl / (sorted_kl.max() + 1e-8))
    ax.bar(range(len(sorted_kl)), sorted_kl, color=colors)
    ax.set_xlabel('Dimension Rank')
    ax.set_ylabel('KL Divergence')
    ax.set_title('KL per Dimension (Sorted)')
    ax.grid(True, alpha=0.3)
    
    # Add dimension labels for top 10
    for i, dim_idx in enumerate(sorted_indices[:10]):
        ax.annotate(f'd{dim_idx}', (i, sorted_kl[i]), 
                   textcoords="offset points", xytext=(0, 5),
                   ha='center', fontsize=8)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def log_latent_to_tensorboard(writer,
                               model,
                               loader: DataLoader,
                               device: torch.device,
                               epoch: int,
                               class_names: List[str],
                               num_samples: int = 500):
    """
    Log latent space visualization to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter
        model: VAE model
        loader: DataLoader
        device: Device
        epoch: Current epoch
        class_names: Class names
        num_samples: Max samples to visualize
    """
    model.eval()
    
    all_latents = []
    all_labels = []
    all_metadata = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            latent = model.get_latent(data, use_mean=True)
            
            all_latents.append(latent.cpu())
            all_labels.extend([data.y[i].item() for i in range(data.num_graphs)])
            
            # Metadata for each sample
            for i in range(data.num_graphs):
                label_idx = data.y[i].item()
                label_name = class_names[label_idx] if label_idx < len(class_names) else f"Class {label_idx}"
                
                subject = data.subject_id[i] if hasattr(data, 'subject_id') else "unknown"
                if isinstance(subject, torch.Tensor):
                    subject = subject.item() if subject.dim() == 0 else "batch"
                
                all_metadata.append(f"{label_name}_{subject}")
            
            if len(all_labels) >= num_samples:
                break
    
    # Stack and limit
    latents = torch.cat(all_latents, dim=0)[:num_samples]
    labels = all_labels[:num_samples]
    metadata = all_metadata[:num_samples]
    
    # Add embeddings to TensorBoard
    writer.add_embedding(
        latents,
        metadata=metadata,
        tag=f"latent_space/epoch_{epoch}",
        global_step=epoch
    )


def save_latent_embeddings(model,
                           loader: DataLoader,
                           device: torch.device,
                           save_path: Path,
                           class_names: List[str],
                           num_samples: int = None):
    """
    Save latent embeddings to pickle file.
    
    Args:
        model: VAE model
        loader: DataLoader
        device: Device
        save_path: Path to save file
        class_names: Class names
        num_samples: Max samples (None = all)
        
    Returns:
        Dict with embeddings and metadata
    """
    model.eval()
    
    all_latents = []
    all_labels = []
    all_subjects = []
    all_conditions = []
    all_bands = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            latent = model.get_latent(data, use_mean=True)
            
            all_latents.append(latent.cpu().numpy())
            
            for i in range(data.num_graphs):
                all_labels.append(data.y[i].item())
                
                # Helper to safely get batched attributes
                def safe_get_attr(attr_name, default='unknown'):
                    if not hasattr(data, attr_name):
                        return default
                    attr = getattr(data, attr_name)
                    if attr is None:
                        return default
                    if isinstance(attr, (list, tuple)):
                        return attr[i] if i < len(attr) else default
                    if isinstance(attr, torch.Tensor):
                        return attr[i].item() if attr.dim() > 0 else attr.item()
                    if data.num_graphs == 1:
                        return attr
                    return default
                
                all_subjects.append(safe_get_attr('subject_id'))
                all_conditions.append(safe_get_attr('condition', class_names[data.y[i].item()]))
                all_bands.append(safe_get_attr('band'))
            
            if num_samples is not None and len(all_labels) >= num_samples:
                break
    
    # Concatenate
    latents = np.concatenate(all_latents, axis=0)
    
    if num_samples is not None:
        latents = latents[:num_samples]
        all_labels = all_labels[:num_samples]
        all_subjects = all_subjects[:num_samples]
        all_conditions = all_conditions[:num_samples]
        all_bands = all_bands[:num_samples]
    
    result = {
        'latent_embeddings': latents,
        'labels': np.array(all_labels),
        'class_names': class_names,
        'subjects': all_subjects,
        'conditions': all_conditions,
        'bands': all_bands
    }
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(result, f)
    
    return result


def plot_attention_heatmap(attention_matrix: np.ndarray,
                           save_path: Path,
                           title: str = "GAT Attention Weights",
                           node_labels: List[str] = None):
    """
    Plot attention weight heatmap.
    
    Args:
        attention_matrix: [num_nodes, num_nodes] attention weights
        save_path: Path to save figure
        title: Plot title
        node_labels: Labels for nodes
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(attention_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax, label='Attention Weight')
    
    if node_labels is not None:
        ax.set_xticks(range(len(node_labels)))
        ax.set_yticks(range(len(node_labels)))
        ax.set_xticklabels(node_labels, rotation=90, fontsize=6)
        ax.set_yticklabels(node_labels, fontsize=6)
    
    ax.set_xlabel('Target Node')
    ax.set_ylabel('Source Node')
    ax.set_title(title)
    
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

