#!/usr/bin/env python3
"""
Visualize attention weights from trained GAT model.

This script extracts and visualizes attention weights to understand
which brain connections the model considers most important.
"""

import argparse
import sys
import yaml
from pathlib import Path
import logging

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch_geometric.loader import DataLoader

sys.path.append(str(Path(__file__).parent.parent))

from data import create_dataset_from_config
from models import create_model_from_config
from utils.visualization import save_attention_heatmap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_attention_weights(model, data_loader, device, num_samples=10):
    """
    Extract attention weights from model.
    
    Args:
        model: Trained GAT model
        data_loader: DataLoader with graphs
        device: torch device
        num_samples: Number of samples to extract
        
    Returns:
        List of (edge_index, attention_weights, labels) tuples
    """
    model.eval()
    attention_data = []
    
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if i >= num_samples:
                break
            
            data = data.to(device)
            
            # Get attention weights
            edge_index_att, alpha = model.get_attention_weights(data)
            
            attention_data.append({
                'edge_index': edge_index_att.cpu().numpy(),
                'attention': alpha.cpu().numpy(),
                'label': data.y.cpu().numpy(),
                'subject_id': data.subject_id[0] if hasattr(data, 'subject_id') else None,
                'condition': data.condition[0] if hasattr(data, 'condition') else None,
                'band': data.band[0] if hasattr(data, 'band') else None,
                'num_nodes': data.num_nodes
            })
    
    return attention_data


def create_attention_matrix(edge_index, attention, num_nodes):
    """
    Convert edge-based attention to matrix form.
    
    Args:
        edge_index: (2, num_edges) edge indices
        attention: (num_edges, num_heads) attention weights
        num_nodes: Number of nodes
        
    Returns:
        (num_nodes, num_nodes) attention matrix
    """
    # Average across attention heads
    if len(attention.shape) > 1:
        attention_avg = attention.mean(axis=1)
    else:
        attention_avg = attention
    
    # Create matrix
    att_matrix = np.zeros((num_nodes, num_nodes))
    
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i]
        dst = edge_index[1, i]
        att_matrix[src, dst] = attention_avg[i]
    
    return att_matrix


def plot_attention_comparison(att_data_by_class, class_names, save_dir):
    """
    Plot attention patterns for each class.
    
    Args:
        att_data_by_class: Dictionary mapping class index to attention data
        class_names: List of class names
        save_dir: Directory to save plots
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Average attention by class
    avg_attention = {}
    
    for class_idx, att_list in att_data_by_class.items():
        matrices = []
        for att_data in att_list:
            att_matrix = create_attention_matrix(
                att_data['edge_index'],
                att_data['attention'],
                att_data['num_nodes']
            )
            matrices.append(att_matrix)
        
        # Average across samples
        avg_attention[class_idx] = np.mean(matrices, axis=0)
    
    # Plot
    n_classes = len(class_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 5))
    
    if n_classes == 1:
        axes = [axes]
    
    for i, class_name in enumerate(class_names):
        ax = axes[i]
        
        if i in avg_attention:
            im = ax.imshow(avg_attention[i], cmap='viridis', aspect='auto')
            ax.set_title(f'{class_name}\nAverage Attention', fontsize=14)
            ax.set_xlabel('Target Node')
            ax.set_ylabel('Source Node')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.set_title(f'{class_name}\n(No data)', fontsize=14)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'attention_by_class.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Attention comparison saved to: {save_dir / 'attention_by_class.png'}")


def plot_attention_distribution(att_data_list, class_names, save_dir):
    """
    Plot distribution of attention weights.
    
    Args:
        att_data_list: List of attention data
        class_names: List of class names
        save_dir: Directory to save plots
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Collect all attention values by class
    attention_by_class = {i: [] for i in range(len(class_names))}
    
    for att_data in att_data_list:
        class_idx = att_data['label'][0]
        attention_by_class[class_idx].extend(att_data['attention'].flatten())
    
    # Distribution plot
    for class_idx, class_name in enumerate(class_names):
        if len(attention_by_class[class_idx]) > 0:
            axes[0].hist(attention_by_class[class_idx], bins=50, alpha=0.6,
                        label=class_name, density=True)
    
    axes[0].set_xlabel('Attention Weight', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('Distribution of Attention Weights', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    data_for_box = [attention_by_class[i] for i in range(len(class_names))]
    axes[1].boxplot(data_for_box, labels=class_names)
    axes[1].set_ylabel('Attention Weight', fontsize=12)
    axes[1].set_title('Attention Weight Distribution by Class', fontsize=14)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'attention_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Attention distribution saved to: {save_dir / 'attention_distribution.png'}")


def main(checkpoint_path: str, config_path: str = None, num_samples: int = 20):
    """Main visualization function."""
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get config
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = checkpoint.get('config')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    logger.info("Loading test dataset...")
    num_workers = config.get('dataset_workers', None)
    _, _, test_graphs = create_dataset_from_config(config, num_workers=num_workers)
    
    test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)
    
    # Create model
    sample_graph = test_graphs[0]
    num_node_features = sample_graph.x.shape[1]
    num_edge_features = sample_graph.edge_attr.shape[1] if hasattr(sample_graph, 'edge_attr') else 0
    num_graph_features = sample_graph.graph_attr.shape[0] if hasattr(sample_graph, 'graph_attr') else 0
    num_classes = len(config['data']['conditions'])
    
    model = create_model_from_config(
        config,
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        num_graph_features=num_graph_features,
        num_classes=num_classes
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Extract attention
    logger.info(f"Extracting attention weights from {num_samples} samples...")
    attention_data = extract_attention_weights(model, test_loader, device, num_samples)
    
    # Group by class
    att_by_class = {i: [] for i in range(num_classes)}
    for att_data in attention_data:
        class_idx = att_data['label'][0]
        att_by_class[class_idx].append(att_data)
    
    # Output directory
    output_dir = Path(config['paths']['output_dir']) / 'attention_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualizations
    class_names = config['data']['conditions']
    
    logger.info("Generating visualizations...")
    plot_attention_comparison(att_by_class, class_names, output_dir)
    plot_attention_distribution(attention_data, class_names, output_dir)
    
    # Save individual examples
    logger.info("Saving individual attention matrices...")
    examples_dir = output_dir / 'examples'
    examples_dir.mkdir(exist_ok=True)
    
    for i, att_data in enumerate(attention_data[:10]):  # Save first 10
        att_matrix = create_attention_matrix(
            att_data['edge_index'],
            att_data['attention'],
            att_data['num_nodes']
        )
        
        class_name = class_names[att_data['label'][0]]
        save_path = examples_dir / f'attention_example_{i:02d}_{class_name}.png'
        
        save_attention_heatmap(
            att_matrix,
            save_path,
            title=f'Attention: {class_name} (Sample {i+1})'
        )
    
    logger.info(f"\nAttention analysis complete! Results saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize attention weights')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--num-samples', type=int, default=20,
                       help='Number of samples to visualize')
    
    args = parser.parse_args()
    main(args.checkpoint, args.config, args.num_samples)

