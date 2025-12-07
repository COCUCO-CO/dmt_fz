#!/usr/bin/env python3
"""
extract_attention_states.py - Extract GAT attention matrices for state clustering analysis

This script:
1. Loads a trained GAT model
2. For each test subject, runs epochs IN ORDER (not shuffled)
3. Extracts attention matrices from all GAT layers for each epoch
4. Saves matrices as tensors + optional sample images
5. Organizes output by subject/condition for clustering analysis

The extracted attention matrices represent learned connectivity patterns
that can be clustered to find recurring "brain states".

Usage:
    python extract_attention_states.py --config path/to/config.yaml
    python extract_attention_states.py --config config.yaml --subject S01 --num-images 50
    python extract_attention_states.py --config config.yaml --cpu --workers 24
"""

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import logging
import multiprocessing as mp

import numpy as np
import torch
import yaml
from tqdm import tqdm

# Use non-GUI backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data import create_dataset_from_config
from models import create_model_from_config
from torch_geometric.loader import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def load_model_and_config(config_path: str, checkpoint_path: Optional[str] = None, force_cpu: bool = False):
    """
    Load trained model and configuration.
    
    Args:
        config_path: Path to config.yaml
        checkpoint_path: Path to model checkpoint (default: best_model.pt in checkpoints dir)
        force_cpu: Force CPU even if CUDA is available
        
    Returns:
        Tuple of (model, config, device)
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Device
    if force_cpu:
        device_name = 'cpu'
        logger.info("Forcing CPU mode")
    else:
        device_name = config['device']
        if device_name == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU")
            device_name = 'cpu'
    device = torch.device(device_name)
    
    # Load dataset to infer dimensions
    logger.info("Loading dataset to infer model dimensions...")
    train_graphs, val_graphs, test_graphs = create_dataset_from_config(config, force_rebuild=False)
    
    sample_graph = train_graphs[0]
    num_node_features = sample_graph.x.shape[1]
    num_edge_features = sample_graph.edge_attr.shape[1] if hasattr(sample_graph, 'edge_attr') else 0
    num_graph_features = sample_graph.graph_attr.shape[0] if hasattr(sample_graph, 'graph_attr') else 0
    num_classes = len(config['data']['conditions'])
    
    # Create model
    model = create_model_from_config(
        config,
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        num_graph_features=num_graph_features,
        num_classes=num_classes
    )
    
    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = Path(config['paths']['checkpoints']) / 'best_model.pt'
    
    logger.info(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    val_acc = checkpoint.get('val_acc', None)
    val_acc_str = f"{val_acc:.4f}" if isinstance(val_acc, (int, float)) else "?"
    logger.info(f"Model loaded from epoch {checkpoint.get('epoch', '?')} "
                f"with val_acc={val_acc_str}")
    
    return model, config, device, (train_graphs, val_graphs, test_graphs)


def get_test_subjects_info(test_graphs: List, config: Dict) -> Dict:
    """
    Get information about subjects in test set.
    
    Returns:
        Dict mapping subject_id to dict with conditions, num_epochs, etc.
    """
    subjects = defaultdict(lambda: {'conditions': set(), 'epochs': []})
    
    for graph in test_graphs:
        subj = graph.subject_id
        subjects[subj]['conditions'].add(graph.condition)
        subjects[subj]['epochs'].append({
            'condition': graph.condition,
            'band': graph.band,
            'epoch_idx': graph.epoch_idx,
            'label': graph.y.item()
        })
    
    # Sort epochs by (condition, band, epoch_idx)
    for subj in subjects:
        subjects[subj]['epochs'].sort(key=lambda x: (x['condition'], x['band'], x['epoch_idx']))
        subjects[subj]['conditions'] = list(subjects[subj]['conditions'])
        subjects[subj]['num_epochs'] = len(subjects[subj]['epochs'])
    
    return dict(subjects)


def extract_attention_for_graph(model, data, device, num_nodes: int):
    """
    Extract attention matrices from all GAT layers for a single graph.
    
    Args:
        model: GAT model
        data: Single PyTorch Geometric Data object
        device: torch device
        num_nodes: Number of nodes in graph
        
    Returns:
        Dict with attention matrices per layer:
        {
            'layer_0': {
                'raw': (edge_index, alpha) with alpha shape [num_edges, num_heads],
                'matrix_mean': [num_nodes, num_nodes] averaged over heads,
                'matrix_per_head': [num_heads, num_nodes, num_nodes]
            },
            'layer_1': {...}
        }
    """
    model.eval()
    
    with torch.no_grad():
        # Create batch from single graph (required for model)
        from torch_geometric.data import Batch
        batch = Batch.from_data_list([data]).to(device)
        
        all_attentions = model.get_all_attention_weights(batch)
        
        result = {}
        for layer_idx, (edge_index, alpha) in enumerate(all_attentions):
            # alpha shape: [num_edges, num_heads] or [num_edges]
            edge_index = edge_index.cpu()
            alpha = alpha.cpu()
            
            if alpha.dim() == 1:
                alpha = alpha.unsqueeze(1)
            
            num_heads = alpha.shape[1]
            
            # Build attention matrices
            matrix_per_head = torch.zeros(num_heads, num_nodes, num_nodes)
            
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[0, i].item(), edge_index[1, i].item()
                for h in range(num_heads):
                    matrix_per_head[h, src, dst] = alpha[i, h]
            
            # Average over heads
            matrix_mean = matrix_per_head.mean(dim=0)
            
            result[f'layer_{layer_idx}'] = {
                'matrix_mean': matrix_mean.numpy(),
                'matrix_per_head': matrix_per_head.numpy(),
                'num_heads': num_heads
            }
        
        return result


def save_attention_image(attention_matrix: np.ndarray, 
                        save_path: Path,
                        title: str = "",
                        figsize: Tuple[int, int] = (4, 4),
                        dpi: int = 64):
    """
    Save attention matrix as image.
    
    Args:
        attention_matrix: NxN attention matrix
        save_path: Path to save image
        title: Optional title
        figsize: Figure size in inches
        dpi: DPI (256x256 at 4 inches = 64 dpi)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot without labels for clean images
    sns.heatmap(attention_matrix, cmap='YlOrRd', ax=ax, 
                cbar=False, square=True,
                xticklabels=False, yticklabels=False)
    
    if title:
        ax.set_title(title, fontsize=8)
    
    ax.axis('off')
    plt.tight_layout(pad=0)
    
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def extract_attention_batch(
    model,
    graphs_batch: List,
    device: torch.device,
    num_nodes: int
) -> List[Dict]:
    """
    Extract attention for a batch of graphs efficiently.
    
    Args:
        model: GAT model
        graphs_batch: List of graph Data objects
        device: torch device
        num_nodes: Number of nodes per graph
        
    Returns:
        List of attention data dicts for each graph
    """
    from torch_geometric.data import Batch
    
    model.eval()
    results = []
    
    with torch.no_grad():
        # Create batch
        batch = Batch.from_data_list(graphs_batch).to(device)
        
        # Get all attention weights
        all_attentions = model.get_all_attention_weights(batch)
        
        # Process each graph in batch
        for graph_idx in range(batch.num_graphs):
            graph_result = {}
            
            # Get node range for this graph
            start_idx = batch.ptr[graph_idx].item()
            end_idx = batch.ptr[graph_idx + 1].item()
            
            for layer_idx, (edge_index, alpha) in enumerate(all_attentions):
                edge_index = edge_index.cpu()
                alpha = alpha.cpu()
                
                if alpha.dim() == 1:
                    alpha = alpha.unsqueeze(1)
                
                num_heads = alpha.shape[1]
                
                # Get edges for this graph
                edge_mask = (edge_index[0] >= start_idx) & (edge_index[0] < end_idx)
                graph_edges = edge_index[:, edge_mask] - start_idx
                graph_alpha = alpha[edge_mask]
                
                # Build attention matrices
                matrix_per_head = np.zeros((num_heads, num_nodes, num_nodes), dtype=np.float32)
                
                for i in range(graph_edges.shape[1]):
                    src, dst = graph_edges[0, i].item(), graph_edges[1, i].item()
                    for h in range(num_heads):
                        matrix_per_head[h, src, dst] = graph_alpha[i, h].item()
                
                # Average over heads
                matrix_mean = matrix_per_head.mean(axis=0)
                
                graph_result[f'layer_{layer_idx}'] = {
                    'matrix_mean': matrix_mean,
                    'matrix_per_head': matrix_per_head,
                    'num_heads': num_heads
                }
            
            results.append(graph_result)
    
    return results


def extract_attention_states(
    model,
    test_graphs: List,
    device: torch.device,
    output_dir: Path,
    subject_filter: Optional[str] = None,
    condition_filter: Optional[str] = None,
    num_sample_images: int = 50,
    image_size: int = 256,
    save_per_head: bool = False,
    batch_size: int = 32,
    num_threads: int = 1
):
    """
    Extract attention states for test subjects and save organized by subject/condition.
    
    Args:
        model: Trained GAT model
        test_graphs: List of test graphs
        device: torch device
        output_dir: Output directory
        subject_filter: Optional - extract only this subject
        condition_filter: Optional - extract only this condition
        num_sample_images: Number of sample images to save (first N epochs)
        image_size: Image size in pixels (square)
        save_per_head: If True, also save per-head attention matrices
        batch_size: Batch size for processing (higher = faster but more memory)
        num_threads: Number of threads for CPU operations
        
    Returns:
        Dict with extraction statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set number of threads for CPU parallelization
    if num_threads > 1:
        torch.set_num_threads(num_threads)
        logger.info(f"Using {num_threads} CPU threads")
    
    # Group graphs by subject and condition
    subject_cond_graphs = defaultdict(lambda: defaultdict(list))
    
    for graph in test_graphs:
        subject_cond_graphs[graph.subject_id][graph.condition].append(graph)
    
    # Filter subjects if specified
    subjects_to_process = list(subject_cond_graphs.keys())
    if subject_filter:
        subjects_to_process = [s for s in subjects_to_process if s == subject_filter]
        if not subjects_to_process:
            logger.warning(f"Subject {subject_filter} not found in test set!")
            logger.info(f"Available subjects: {list(subject_cond_graphs.keys())}")
            return {}
    
    # Get number of nodes from first graph
    num_nodes = test_graphs[0].num_nodes
    num_layers = len(model.conv_layers)
    
    logger.info(f"Extracting attention states...")
    logger.info(f"  Subjects: {len(subjects_to_process)}")
    logger.info(f"  Nodes: {num_nodes}")
    logger.info(f"  GAT layers: {num_layers}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Output: {output_dir}")
    
    stats = {
        'subjects': [],
        'total_epochs': 0,
        'epochs_per_subject_condition': {}
    }
    
    # Calculate DPI for desired image size
    figsize = 4  # inches
    dpi = image_size // figsize
    
    for subject_id in tqdm(subjects_to_process, desc="Subjects"):
        subject_dir = output_dir / subject_id
        subject_dir.mkdir(exist_ok=True)
        
        stats['subjects'].append(subject_id)
        stats['epochs_per_subject_condition'][subject_id] = {}
        
        conditions = list(subject_cond_graphs[subject_id].keys())
        
        # Filter conditions if specified
        if condition_filter:
            conditions = [c for c in conditions if c == condition_filter]
        
        for condition in conditions:
            cond_dir = subject_dir / condition
            cond_dir.mkdir(exist_ok=True)
            
            # Sort graphs by (band, epoch_idx) to maintain temporal order
            graphs = subject_cond_graphs[subject_id][condition]
            graphs.sort(key=lambda g: (g.band, g.epoch_idx))
            
            logger.info(f"  {subject_id}/{condition}: {len(graphs)} epochs")
            
            # Create layer-specific directories
            layer_dirs = {}
            for l in range(num_layers):
                layer_dir = cond_dir / f'layer{l}'
                layer_dir.mkdir(exist_ok=True)
                (layer_dir / 'tensors').mkdir(exist_ok=True)
                layer_dirs[l] = layer_dir
            
            # Storage for all attention matrices in this condition
            all_attention_matrices = {
                f'layer_{l}': {
                    'mean': [],  # List of [N, N] matrices
                    'per_head': [] if save_per_head else None
                }
                for l in range(num_layers)
            }
            
            metadata = []
            
            # Process in batches for efficiency
            num_batches = (len(graphs) + batch_size - 1) // batch_size
            
            for batch_idx in tqdm(range(num_batches), desc=f"  {condition}", leave=False):
                start = batch_idx * batch_size
                end = min(start + batch_size, len(graphs))
                batch_graphs = graphs[start:end]
                
                # Extract attention for batch
                batch_results = extract_attention_batch(model, batch_graphs, device, num_nodes)
                
                # Process each result
                for local_offset, attention_data in enumerate(batch_results):
                    epoch_local_idx = start + local_offset
                    graph = batch_graphs[local_offset]
                    
                    # Store matrices
                    for layer_key in attention_data:
                        all_attention_matrices[layer_key]['mean'].append(
                            attention_data[layer_key]['matrix_mean']
                        )
                        if save_per_head:
                            all_attention_matrices[layer_key]['per_head'].append(
                                attention_data[layer_key]['matrix_per_head']
                            )
                    
                    # Metadata for this epoch
                    metadata.append({
                        'local_idx': epoch_local_idx,
                        'original_epoch_idx': graph.epoch_idx,
                        'band': graph.band,
                        'label': graph.y.item()
                    })
                    
                    # Save sample images for first N epochs (in layer-specific folders)
                    if epoch_local_idx < num_sample_images:
                        for layer_idx in range(num_layers):
                            layer_key = f'layer_{layer_idx}'
                            images_dir = layer_dirs[layer_idx] / 'images'
                            if not images_dir.exists():
                                images_dir.mkdir(exist_ok=True)
                            
                            img_path = images_dir / f'epoch{epoch_local_idx:04d}.png'
                            
                            save_attention_image(
                                attention_data[layer_key]['matrix_mean'],
                                img_path,
                                title=f"E{epoch_local_idx} {graph.band}",
                                figsize=(figsize, figsize),
                                dpi=dpi
                            )
            
            # Convert to numpy arrays and save in layer-specific folders
            for layer_key in all_attention_matrices:
                layer_idx = int(layer_key.split('_')[1])
                tensors_dir = layer_dirs[layer_idx] / 'tensors'
                
                # Stack all epochs: shape [num_epochs, N, N]
                mean_matrices = np.stack(all_attention_matrices[layer_key]['mean'], axis=0)
                
                # Save as single .npy file for efficiency
                np.save(tensors_dir / 'attention_mean.npy', mean_matrices)
                
                if save_per_head and all_attention_matrices[layer_key]['per_head']:
                    per_head_matrices = np.stack(all_attention_matrices[layer_key]['per_head'], axis=0)
                    np.save(tensors_dir / 'attention_per_head.npy', per_head_matrices)
            
            # Save metadata
            with open(cond_dir / 'metadata.pkl', 'wb') as f:
                pickle.dump({
                    'subject_id': subject_id,
                    'condition': condition,
                    'num_epochs': len(graphs),
                    'num_nodes': num_nodes,
                    'num_layers': num_layers,
                    'num_heads': attention_data['layer_0']['num_heads'],
                    'epochs': metadata
                }, f)
            
            stats['epochs_per_subject_condition'][subject_id][condition] = len(graphs)
            stats['total_epochs'] += len(graphs)
    
    # Save global summary
    with open(output_dir / 'extraction_summary.pkl', 'wb') as f:
        pickle.dump(stats, f)
    
    logger.info(f"\nExtraction complete!")
    logger.info(f"  Total epochs: {stats['total_epochs']}")
    logger.info(f"  Output: {output_dir}")
    
    return stats


def print_test_subjects_summary(test_graphs: List, config: Dict):
    """Print summary of test subjects for user selection."""
    subjects_info = get_test_subjects_info(test_graphs, config)
    
    print("\n" + "=" * 60)
    print("TEST SET SUBJECTS")
    print("=" * 60)
    
    for subj, info in sorted(subjects_info.items()):
        conditions_str = ", ".join(sorted(info['conditions']))
        print(f"  {subj}: {info['num_epochs']} epochs ({conditions_str})")
    
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Extract GAT attention matrices for state clustering analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config.yaml')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (default: best_model.pt)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: output_dir/attention_states)')
    parser.add_argument('--subject', type=str, default=None,
                       help='Extract only this subject (default: all test subjects)')
    parser.add_argument('--condition', type=str, default=None,
                       help='Extract only this condition (default: all)')
    parser.add_argument('--num-images', type=int, default=50,
                       help='Number of sample images to save (default: 50)')
    parser.add_argument('--image-size', type=int, default=256,
                       help='Image size in pixels (default: 256)')
    parser.add_argument('--save-per-head', action='store_true',
                       help='Also save per-head attention matrices')
    parser.add_argument('--list-subjects', action='store_true',
                       help='Only list test subjects, do not extract')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU mode (useful when GPU is busy)')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of CPU threads for parallel processing (default: 1)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for extraction (default: 32, higher = faster)')
    
    args = parser.parse_args()
    
    # Load model and data
    model, config, device, (train_graphs, val_graphs, test_graphs) = load_model_and_config(
        args.config, args.checkpoint, force_cpu=args.cpu
    )
    
    # Print subjects info
    print_test_subjects_summary(test_graphs, config)
    
    if args.list_subjects:
        return
    
    # Verify GATv2 model
    if config['model']['architecture']['conv_type'] != 'gatv2':
        logger.error("This script requires a GATv2 model (conv_type: gatv2)")
        sys.exit(1)
    
    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config['paths']['output_dir']) / 'attention_states'
    
    # Extract attention states
    stats = extract_attention_states(
        model=model,
        test_graphs=test_graphs,
        device=device,
        output_dir=output_dir,
        subject_filter=args.subject,
        condition_filter=args.condition,
        num_sample_images=args.num_images,
        image_size=args.image_size,
        save_per_head=args.save_per_head,
        batch_size=args.batch_size,
        num_threads=args.workers
    )
    
    print("\nDone! Use attention_clustering.py to cluster the extracted states.")


if __name__ == '__main__':
    main()

