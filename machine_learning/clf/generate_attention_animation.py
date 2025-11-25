#!/usr/bin/env python3
"""
Generate attention animation for a specific subject.

This script loads a trained model and generates an animation showing
how attention patterns evolve across epochs for each condition.

Usage:
    # Full attention matrix animation
    python generate_attention_animation.py --checkpoint best_model.pt --subject sub-01 --fps 5
    
    # MST-based animation
    python generate_attention_animation.py --checkpoint best_model.pt --subject sub-01 --mst --fps 3
    
    # Specific layer
    python generate_attention_animation.py --checkpoint best_model.pt --subject sub-01 --layer 1

Output:
    attention_animation_{subject}_{condition}_layer{N}.mp4
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

import numpy as np
import torch
import yaml
import pickle
from collections import defaultdict
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.animation as animation
import networkx as nx

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from models import create_model_from_config
from data import create_dataset_from_config
from utils.attention_logger import (
    load_electrode_names, 
    get_standard_eeg_coordinates,
    compute_mst_from_attention
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model_and_config(checkpoint_path: Path, config_path: Path = None):
    """Load trained model and config."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Try to find config
    if config_path is None:
        # Look in same directory as checkpoint
        possible_configs = [
            checkpoint_path.parent / 'config.yaml',
            checkpoint_path.parent.parent / 'config.yaml',
            Path('config/config.yaml')
        ]
        for cfg in possible_configs:
            if cfg.exists():
                config_path = cfg
                break
    
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Config file not found. Tried: {possible_configs}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded config from {config_path}")
    
    # Get model dimensions from a sample graph
    # We need to create model with correct dimensions
    node_features = config.get('model_node_features', 8)
    edge_features = config.get('model_edge_features', 1)
    graph_features = config.get('model_graph_features', 8)
    num_classes = len(config['data']['conditions'])
    
    # Try to get from checkpoint
    if 'config' in checkpoint:
        node_features = checkpoint['config'].get('node_features', node_features)
        edge_features = checkpoint['config'].get('edge_features', edge_features)
        graph_features = checkpoint['config'].get('graph_features', graph_features)
    
    model = create_model_from_config(config, node_features, edge_features, graph_features, num_classes)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    logger.info(f"Loaded model from {checkpoint_path}")
    
    return model, config


def get_subject_epochs(graphs: List, subject_id: str, class_names: List[str]) -> Dict[str, List]:
    """
    Get all epochs for a specific subject, organized by condition.
    
    Returns:
        Dict mapping condition name to list of (epoch_idx, graph) tuples, sorted by epoch_idx
    """
    subject_data = defaultdict(list)
    
    for g in graphs:
        if hasattr(g, 'subject_id') and str(g.subject_id) == str(subject_id):
            condition_idx = g.y.item()
            condition_name = class_names[condition_idx] if condition_idx < len(class_names) else f"Class_{condition_idx}"
            epoch_idx = g.epoch_idx if hasattr(g, 'epoch_idx') else 0
            subject_data[condition_name].append((epoch_idx, g))
    
    # Sort by epoch index
    for cond in subject_data:
        subject_data[cond] = sorted(subject_data[cond], key=lambda x: x[0])
    
    return subject_data


def extract_attention_matrix(model, graph, device, layer_idx: int = 0) -> np.ndarray:
    """
    Extract attention matrix for a single graph at a specific layer.
    
    Returns:
        NxN attention matrix
    """
    model.eval()
    graph = graph.to(device)
    
    with torch.no_grad():
        all_attentions = model.get_all_attention_weights(graph)
    
    if layer_idx >= len(all_attentions):
        layer_idx = len(all_attentions) - 1
    
    edge_index, alpha = all_attentions[layer_idx]
    
    # Average across heads if multi-head
    if alpha.dim() > 1:
        alpha = alpha.mean(dim=1)
    
    # Build adjacency matrix
    num_nodes = graph.num_nodes
    adj = np.zeros((num_nodes, num_nodes))
    
    edge_index = edge_index.cpu().numpy()
    alpha = alpha.cpu().numpy()
    
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        adj[src, dst] = alpha[i]
    
    return adj


def create_attention_frame(attention_matrix: np.ndarray,
                          ch_names: List[str],
                          coords: Dict[str, Tuple[float, float]],
                          title: str,
                          use_mst: bool = False,
                          vmin: float = None,
                          vmax: float = None) -> plt.Figure:
    """
    Create a single frame showing the attention graph.
    
    Args:
        attention_matrix: NxN attention matrix
        ch_names: List of electrode names
        coords: Dict mapping electrode names to (x, y) coordinates
        title: Frame title
        use_mst: If True, show MST instead of full graph
        vmin, vmax: Color scale limits (for consistency across frames)
    
    Returns:
        matplotlib Figure
    """
    n = attention_matrix.shape[0]
    
    if use_mst:
        # Compute MST
        matrix = compute_mst_from_attention(attention_matrix)
    else:
        matrix = attention_matrix
    
    # Build graph
    G = nx.DiGraph() if not use_mst else nx.Graph()
    for i in range(min(n, len(ch_names))):
        G.add_node(ch_names[i])
    
    edge_weights = {}
    for i in range(n):
        for j in range(n):
            if matrix[i, j] > 0 and i < len(ch_names) and j < len(ch_names):
                if use_mst:
                    if i < j:  # Avoid duplicate edges for undirected
                        G.add_edge(ch_names[i], ch_names[j])
                        edge_weights[(ch_names[i], ch_names[j])] = matrix[i, j]
                else:
                    G.add_edge(ch_names[i], ch_names[j])
                    edge_weights[(ch_names[i], ch_names[j])] = matrix[i, j]
    
    if len(edge_weights) == 0:
        # Create empty figure with message
        fig, ax = plt.subplots(figsize=(10, 9))
        ax.text(0.5, 0.5, 'No attention edges', ha='center', va='center', fontsize=14)
        ax.set_title(title)
        ax.axis('off')
        return fig
    
    # Create figure
    fig = plt.figure(figsize=(10, 9))
    ax_graph = fig.add_axes([0.05, 0.12, 0.9, 0.78])
    
    # Get position dict (only for nodes we have)
    pos = {name: coords[name] for name in ch_names if name in coords}
    
    all_weights = list(edge_weights.values())
    if vmin is None:
        vmin = min(all_weights)
    if vmax is None:
        vmax = max(all_weights)
    
    weight_range = vmax - vmin if vmax > vmin else 1.0
    
    # Sort edges by weight to draw stronger on top
    sorted_edges = sorted(edge_weights.items(), key=lambda x: x[1])
    
    # Draw edges
    for (src, dst), weight in sorted_edges:
        if src in pos and dst in pos:
            normalized = (weight - vmin) / weight_range if weight_range > 0 else 0.5
            alpha = 0.2 + normalized * 0.8
            width = 0.5 + normalized * 4.0
            color = cm.Reds(0.3 + normalized * 0.7)
            
            nx.draw_networkx_edges(G, pos, edgelist=[(src, dst)],
                                  width=width, edge_color=[color], alpha=alpha, 
                                  ax=ax_graph, arrows=not use_mst,
                                  arrowsize=10 if not use_mst else 0,
                                  connectionstyle="arc3,rad=0.1" if not use_mst else None)
    
    # Draw nodes
    node_list = [name for name in ch_names if name in pos]
    nx.draw_networkx_nodes(G, pos, nodelist=node_list,
                          node_color='lightgray', node_size=800,
                          edgecolors='black', linewidths=1.5,
                          alpha=0.95, ax=ax_graph)
    
    # Draw labels
    for node_name in node_list:
        if node_name in pos:
            x, y = pos[node_name]
            ax_graph.text(x, y, node_name, fontsize=7, fontweight='bold',
                         ha='center', va='center', color='black',
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                  edgecolor='black', linewidth=0.5, alpha=0.9))
    
    # Fixed axis limits
    ax_graph.axis('equal')
    ax_graph.set_xlim(-0.15, 0.15)
    ax_graph.set_ylim(-0.16, 0.16)
    ax_graph.axis('off')
    
    # Colorbar
    cbar_ax = fig.add_axes([0.15, 0.04, 0.7, 0.02])
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(norm=norm, cmap=cm.Reds)
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(f'Attention Weight', fontsize=9, fontweight='bold')
    cbar.ax.tick_params(labelsize=8)
    
    # Title
    fig.text(0.5, 0.96, title, fontsize=12, fontweight='bold', ha='center', va='top')
    
    return fig


def generate_animation(model, subject_data: Dict[str, List],
                      device, ch_names: List[str],
                      coords: Dict[str, Tuple[float, float]],
                      output_dir: Path,
                      subject_id: str,
                      layer_idx: int = 0,
                      fps: int = 5,
                      use_mst: bool = False,
                      transition_frames: int = 3):
    """
    Generate animation for each condition.
    
    Args:
        model: Trained GAT model
        subject_data: Dict from get_subject_epochs()
        device: torch device
        ch_names: Electrode names
        coords: Electrode coordinates
        output_dir: Where to save videos
        subject_id: Subject ID for filename
        layer_idx: Which GAT layer to visualize
        fps: Frames per second
        use_mst: Use MST instead of full attention
        transition_frames: Number of interpolation frames between epochs
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for condition, epoch_data in subject_data.items():
        if not epoch_data:
            continue
        
        logger.info(f"Generating animation for {condition} ({len(epoch_data)} epochs)")
        
        # Extract all attention matrices first
        attention_matrices = []
        epoch_indices = []
        
        for epoch_idx, graph in tqdm(epoch_data, desc=f"Extracting attention"):
            att_matrix = extract_attention_matrix(model, graph, device, layer_idx)
            attention_matrices.append(att_matrix)
            epoch_indices.append(epoch_idx)
        
        if not attention_matrices:
            continue
        
        # Compute global min/max for consistent color scale
        all_values = np.concatenate([m.flatten() for m in attention_matrices])
        all_values = all_values[all_values > 0]  # Only positive values
        if len(all_values) == 0:
            continue
        vmin, vmax = np.percentile(all_values, [1, 99])
        
        # Create interpolated frames for smooth transitions
        frames_data = []
        for i, (att_matrix, epoch_idx) in enumerate(zip(attention_matrices, epoch_indices)):
            # Add main frame
            frames_data.append((att_matrix, epoch_idx, True))  # True = real frame
            
            # Add transition frames to next epoch
            if i < len(attention_matrices) - 1 and transition_frames > 0:
                next_matrix = attention_matrices[i + 1]
                for t in range(1, transition_frames + 1):
                    alpha = t / (transition_frames + 1)
                    interp_matrix = (1 - alpha) * att_matrix + alpha * next_matrix
                    frames_data.append((interp_matrix, None, False))  # False = transition
        
        # Create figure for animation
        fig = plt.figure(figsize=(10, 9))
        
        def update(frame_idx):
            fig.clf()
            att_matrix, epoch_idx, is_real = frames_data[frame_idx]
            
            graph_type = "MST" if use_mst else "Attention"
            if is_real:
                title = f"{graph_type} - {condition} - Layer {layer_idx + 1} - Epoch {epoch_idx}"
            else:
                title = f"{graph_type} - {condition} - Layer {layer_idx + 1} (transition)"
            
            # Recreate the plot
            if use_mst:
                matrix = compute_mst_from_attention(att_matrix)
            else:
                matrix = att_matrix
            
            n = matrix.shape[0]
            G = nx.Graph() if use_mst else nx.DiGraph()
            for i in range(min(n, len(ch_names))):
                G.add_node(ch_names[i])
            
            edge_weights = {}
            for i in range(n):
                for j in range(n):
                    if matrix[i, j] > 0 and i < len(ch_names) and j < len(ch_names):
                        if use_mst:
                            if i < j:
                                G.add_edge(ch_names[i], ch_names[j])
                                edge_weights[(ch_names[i], ch_names[j])] = matrix[i, j]
                        else:
                            G.add_edge(ch_names[i], ch_names[j])
                            edge_weights[(ch_names[i], ch_names[j])] = matrix[i, j]
            
            ax_graph = fig.add_axes([0.05, 0.12, 0.9, 0.78])
            pos = {name: coords[name] for name in ch_names if name in coords}
            
            weight_range = vmax - vmin if vmax > vmin else 1.0
            sorted_edges = sorted(edge_weights.items(), key=lambda x: x[1])
            
            for (src, dst), weight in sorted_edges:
                if src in pos and dst in pos:
                    normalized = (weight - vmin) / weight_range
                    normalized = max(0, min(1, normalized))
                    alpha = 0.2 + normalized * 0.8
                    width = 0.5 + normalized * 4.0
                    color = cm.Reds(0.3 + normalized * 0.7)
                    nx.draw_networkx_edges(G, pos, edgelist=[(src, dst)],
                                          width=width, edge_color=[color], alpha=alpha,
                                          ax=ax_graph, arrows=not use_mst,
                                          arrowsize=8 if not use_mst else 0)
            
            node_list = [name for name in ch_names if name in pos]
            nx.draw_networkx_nodes(G, pos, nodelist=node_list,
                                  node_color='lightgray', node_size=800,
                                  edgecolors='black', linewidths=1.5,
                                  alpha=0.95, ax=ax_graph)
            
            for node_name in node_list:
                if node_name in pos:
                    x, y = pos[node_name]
                    ax_graph.text(x, y, node_name, fontsize=7, fontweight='bold',
                                 ha='center', va='center', color='black',
                                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                          edgecolor='black', linewidth=0.5, alpha=0.9))
            
            ax_graph.axis('equal')
            ax_graph.set_xlim(-0.15, 0.15)
            ax_graph.set_ylim(-0.16, 0.16)
            ax_graph.axis('off')
            
            # Colorbar
            cbar_ax = fig.add_axes([0.15, 0.04, 0.7, 0.02])
            norm = Normalize(vmin=vmin, vmax=vmax)
            sm = cm.ScalarMappable(norm=norm, cmap=cm.Reds)
            cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
            cbar.set_label(f'Attention [{vmin:.4f}, {vmax:.4f}]', fontsize=9)
            cbar.ax.tick_params(labelsize=8)
            
            fig.text(0.5, 0.96, title, fontsize=12, fontweight='bold', ha='center', va='top')
            
            return []
        
        # Create animation
        total_frames = len(frames_data)
        logger.info(f"Creating animation with {total_frames} frames ({len(attention_matrices)} epochs + transitions)")
        
        anim = animation.FuncAnimation(fig, update, frames=total_frames,
                                       interval=1000/fps, blit=False)
        
        # Save
        graph_type = "mst" if use_mst else "attention"
        output_path = output_dir / f"{graph_type}_animation_{subject_id}_{condition}_layer{layer_idx + 1}.mp4"
        
        writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='GAT-EEG'),
                                        bitrate=2000)
        anim.save(str(output_path), writer=writer, dpi=150)
        plt.close(fig)
        
        logger.info(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate attention animation for a subject',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full attention animation at 5 fps
  python generate_attention_animation.py --checkpoint best_model.pt --subject sub-01 --fps 5
  
  # MST animation at 3 fps
  python generate_attention_animation.py --checkpoint best_model.pt --subject sub-01 --mst --fps 3
  
  # Specific layer with more transition frames
  python generate_attention_animation.py --checkpoint best_model.pt --subject sub-01 --layer 2 --transitions 5
        """
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pt file)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config.yaml (auto-detected if not provided)')
    parser.add_argument('--subject', type=str, required=True,
                       help='Subject ID to process (e.g., sub-01)')
    parser.add_argument('--layer', type=int, default=0,
                       help='GAT layer index to visualize (0-based, default: 0)')
    parser.add_argument('--fps', type=int, default=5,
                       help='Frames per second (default: 5)')
    parser.add_argument('--mst', action='store_true',
                       help='Use MST instead of full attention matrix')
    parser.add_argument('--transitions', type=int, default=3,
                       help='Number of transition frames between epochs (default: 3)')
    parser.add_argument('--output', type=str, default='output/animations',
                       help='Output directory (default: output/animations)')
    parser.add_argument('--conditions', type=str, nargs='+', default=None,
                       help='Conditions to process (default: all)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')
    parser.add_argument('--list-subjects', action='store_true',
                       help='List available subjects and exit')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model and config
    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config) if args.config else None
    
    model, config = load_model_and_config(checkpoint_path, config_path)
    model = model.to(device)
    
    # Load dataset
    logger.info("Loading dataset...")
    train_graphs, val_graphs, test_graphs = create_dataset_from_config(config)
    all_graphs = train_graphs + val_graphs + test_graphs
    
    class_names = config['data']['conditions']
    
    # List subjects if requested
    if args.list_subjects:
        subjects = set()
        for g in all_graphs:
            if hasattr(g, 'subject_id'):
                subjects.add(str(g.subject_id))
        print("\nAvailable subjects:")
        for s in sorted(subjects):
            # Count epochs per condition
            counts = defaultdict(int)
            for g in all_graphs:
                if hasattr(g, 'subject_id') and str(g.subject_id) == s:
                    cond = class_names[g.y.item()]
                    counts[cond] += 1
            count_str = ", ".join([f"{c}: {n}" for c, n in counts.items()])
            print(f"  {s}: {count_str}")
        return
    
    # Get subject epochs
    subject_data = get_subject_epochs(all_graphs, args.subject, class_names)
    
    if not subject_data:
        logger.error(f"No data found for subject '{args.subject}'")
        logger.info("Use --list-subjects to see available subjects")
        return
    
    # Filter conditions if specified
    if args.conditions:
        subject_data = {k: v for k, v in subject_data.items() if k in args.conditions}
    
    for cond, epochs in subject_data.items():
        logger.info(f"Found {len(epochs)} epochs for condition {cond}")
    
    # Load electrode info
    ch_names = load_electrode_names()
    if ch_names is None:
        ch_names = [f'E{i}' for i in range(24)]
    
    coords = get_standard_eeg_coordinates()
    
    # Generate animations
    output_dir = Path(args.output)
    generate_animation(
        model=model,
        subject_data=subject_data,
        device=device,
        ch_names=ch_names,
        coords=coords,
        output_dir=output_dir,
        subject_id=args.subject,
        layer_idx=args.layer,
        fps=args.fps,
        use_mst=args.mst,
        transition_frames=args.transitions
    )
    
    logger.info(f"\nAnimations saved to: {output_dir}")


if __name__ == '__main__':
    main()

