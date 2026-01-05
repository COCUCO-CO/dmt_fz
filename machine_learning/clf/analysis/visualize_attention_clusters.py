#!/usr/bin/env python3
"""
visualize_attention_clusters.py - Visualize attention clusters as EEG graphs

This script:
1. Loads clustering results from attention_clustering.py
2. Computes cluster centroids (average attention pattern per cluster)
3. Visualizes each cluster as a graph over EEG electrode positions
4. Uses transition matrices to show directed connections between clusters
5. Generates animated video of cluster transitions

Usage:
    python visualize_attention_clusters.py --input-dir path/to/attention_states
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

import numpy as np
from tqdm import tqdm

# Use non-GUI backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# Standard 10-20 EEG electrode positions (2D projection, nose up)
EEG_POSITIONS_24 = {
    # Frontal
    'Fp1': (-0.03, 0.095), 'Fp2': (0.03, 0.095),
    'F7': (-0.08, 0.05), 'F3': (-0.045, 0.055), 'Fz': (0.0, 0.06),
    'F4': (0.045, 0.055), 'F8': (0.08, 0.05),
    # Temporal/Central
    'T7': (-0.095, 0.0), 'C3': (-0.05, 0.0), 'Cz': (0.0, 0.0),
    'C4': (0.05, 0.0), 'T8': (0.095, 0.0),
    # Parietal
    'P7': (-0.08, -0.05), 'P3': (-0.045, -0.055), 'Pz': (0.0, -0.06),
    'P4': (0.045, -0.055), 'P8': (0.08, -0.05),
    # Occipital
    'O1': (-0.03, -0.095), 'O2': (0.03, -0.095),
    # Additional (CP, FC, etc. - approximate positions)
    'FC1': (-0.025, 0.03), 'FC2': (0.025, 0.03),
    'CP1': (-0.025, -0.03), 'CP2': (0.025, -0.03),
    'FCz': (0.0, 0.035), 'CPz': (0.0, -0.035),
}

# Fallback: Generate circular layout for N electrodes
def get_circular_positions(n_nodes: int) -> Dict[int, Tuple[float, float]]:
    """Generate circular electrode positions."""
    positions = {}
    for i in range(n_nodes):
        angle = 2 * np.pi * i / n_nodes - np.pi / 2  # Start from top
        x = 0.08 * np.cos(angle)
        y = 0.08 * np.sin(angle)
        positions[i] = (x, y)
    return positions


def load_electrode_names() -> Optional[List[str]]:
    """Load electrode names from extra.pkl."""
    extra_path = Path('/media/storage_hdd/dmt_fz/fwd-inv-stc/extra.pkl')
    if extra_path.exists():
        try:
            with open(extra_path, 'rb') as f:
                data = pickle.load(f)
            return data[4]  # Channel names at index 4
        except Exception as e:
            logger.warning(f"Could not load electrode names: {e}")
    return None


def get_electrode_positions(n_nodes: int, ch_names: Optional[List[str]] = None) -> Dict[int, Tuple[float, float]]:
    """
    Get 2D positions for electrodes.
    
    Args:
        n_nodes: Number of nodes/electrodes
        ch_names: Optional list of electrode names
        
    Returns:
        Dict mapping node index to (x, y) position
    """
    positions = {}
    
    if ch_names and len(ch_names) >= n_nodes:
        # Try to match electrode names to known positions
        for i, name in enumerate(ch_names[:n_nodes]):
            # Try exact match first
            if name in EEG_POSITIONS_24:
                positions[i] = EEG_POSITIONS_24[name]
            else:
                # Try without numbers/suffixes
                base_name = ''.join(c for c in name if not c.isdigit())
                if base_name in EEG_POSITIONS_24:
                    positions[i] = EEG_POSITIONS_24[base_name]
    
    # Fill missing with circular layout
    if len(positions) < n_nodes:
        circular = get_circular_positions(n_nodes)
        for i in range(n_nodes):
            if i not in positions:
                positions[i] = circular[i]
    
    return positions


def load_clustering_results(clustering_dir: Path) -> Dict:
    """Load clustering results."""
    results_path = clustering_dir / 'clustering_results.pkl'
    if not results_path.exists():
        raise FileNotFoundError(f"Clustering results not found: {results_path}")
    
    with open(results_path, 'rb') as f:
        return pickle.load(f)


def load_attention_matrices(attention_dir: Path, layer: str = 'layer0') -> Tuple[np.ndarray, List[Dict]]:
    """
    Load all attention matrices and metadata.
    
    Returns:
        Tuple of (matrices [n_epochs, n_nodes, n_nodes], metadata_list)
    """
    all_matrices = []
    all_metadata = []
    
    # Find all subject directories
    exclude_dirs = {'clustering', 'plots', 'analysis'}
    subject_dirs = sorted([d for d in attention_dir.iterdir() 
                          if d.is_dir() and d.name not in exclude_dirs])
    
    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        
        for cond_dir in subject_dir.iterdir():
            if not cond_dir.is_dir():
                continue
            
            condition = cond_dir.name
            
            # Look for layer directory (new format)
            layer_dir = cond_dir / layer
            if layer_dir.exists():
                tensors_dir = layer_dir / 'tensors'
                att_file = tensors_dir / 'attention_mean.npy'
            else:
                # Old format
                tensors_dir = cond_dir / 'tensors'
                att_file = tensors_dir / f'attention_{layer}_mean.npy'
            
            if not att_file.exists():
                continue
            
            matrices = np.load(att_file)
            n_epochs = matrices.shape[0]
            
            # Load metadata
            metadata_path = cond_dir / 'metadata.pkl'
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    meta = pickle.load(f)
                epochs_meta = meta.get('epochs', [])
            else:
                epochs_meta = [{'band': 'unknown'}] * n_epochs
            
            for i in range(n_epochs):
                all_matrices.append(matrices[i])
                all_metadata.append({
                    'subject_id': subject_id,
                    'condition': condition,
                    'epoch_idx': i,
                    'band': epochs_meta[i].get('band', 'unknown') if i < len(epochs_meta) else 'unknown'
                })
    
    return np.array(all_matrices), all_metadata


def compute_cluster_centroids(
    matrices: np.ndarray,
    labels: np.ndarray,
    n_clusters: int
) -> Dict[int, np.ndarray]:
    """
    Compute average attention matrix for each cluster.
    
    Returns:
        Dict mapping cluster_id to mean attention matrix
    """
    centroids = {}
    for k in range(n_clusters):
        mask = labels == k
        if mask.sum() > 0:
            centroids[k] = matrices[mask].mean(axis=0)
        else:
            centroids[k] = np.zeros_like(matrices[0])
    return centroids


def plot_attention_graph(
    attention_matrix: np.ndarray,
    positions: Dict[int, Tuple[float, float]],
    ax: plt.Axes,
    title: str = "",
    ch_names: Optional[List[str]] = None,
    edge_threshold_percentile: float = 75,
    node_size: int = 800,
    cmap: str = 'Reds',
    show_colorbar: bool = True
):
    """
    Plot attention matrix as a graph over EEG electrode positions.
    
    Args:
        attention_matrix: NxN attention weights
        positions: Dict of node positions
        ax: Matplotlib axis
        title: Plot title
        ch_names: Optional electrode names
        edge_threshold_percentile: Only show edges above this percentile
        node_size: Size of electrode nodes
        cmap: Colormap for edges
        show_colorbar: Whether to show colorbar
    """
    n_nodes = attention_matrix.shape[0]
    
    # Get edge threshold
    edge_values = attention_matrix.flatten()
    edge_values = edge_values[edge_values > 0]
    if len(edge_values) > 0:
        threshold = np.percentile(edge_values, edge_threshold_percentile)
    else:
        threshold = 0
    
    # Draw head outline
    head_circle = plt.Circle((0, 0), 0.105, fill=False, color='gray', linewidth=2)
    ax.add_patch(head_circle)
    
    # Draw nose indicator
    nose_x = [0, 0.015, 0]
    nose_y = [0.105, 0.115, 0.105]
    ax.plot(nose_x, nose_y, 'gray', linewidth=2)
    
    # Draw ears
    ear_left = plt.Circle((-0.11, 0), 0.015, fill=False, color='gray', linewidth=1.5)
    ear_right = plt.Circle((0.11, 0), 0.015, fill=False, color='gray', linewidth=1.5)
    ax.add_patch(ear_left)
    ax.add_patch(ear_right)
    
    # Prepare edges
    edges = []
    edge_weights = []
    
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j and attention_matrix[i, j] > threshold:
                edges.append((positions[i], positions[j]))
                edge_weights.append(attention_matrix[i, j])
    
    if edges:
        # Normalize weights for visualization
        weights_arr = np.array(edge_weights)
        norm = Normalize(vmin=weights_arr.min(), vmax=weights_arr.max())
        colormap = cm.get_cmap(cmap)
        
        # Sort by weight to draw stronger connections on top
        sorted_indices = np.argsort(edge_weights)
        
        for idx in sorted_indices:
            start, end = edges[idx]
            weight = edge_weights[idx]
            color = colormap(norm(weight))
            alpha = 0.3 + 0.7 * norm(weight)
            linewidth = 0.5 + 3 * norm(weight)
            
            # Draw arrow for directed edge
            ax.annotate('', xy=end, xytext=start,
                       arrowprops=dict(arrowstyle='->', color=color, 
                                      alpha=alpha, lw=linewidth,
                                      connectionstyle='arc3,rad=0.1'))
    
    # Draw nodes
    node_x = [positions[i][0] for i in range(n_nodes)]
    node_y = [positions[i][1] for i in range(n_nodes)]
    
    # Node importance based on incoming + outgoing attention
    node_importance = attention_matrix.sum(axis=0) + attention_matrix.sum(axis=1)
    node_importance = (node_importance - node_importance.min()) / (node_importance.max() - node_importance.min() + 1e-8)
    
    node_colors = [cm.Blues(0.3 + 0.7 * imp) for imp in node_importance]
    
    ax.scatter(node_x, node_y, s=node_size, c=node_colors, 
               edgecolors='black', linewidths=2, zorder=10)
    
    # Add electrode labels
    if ch_names:
        for i in range(min(n_nodes, len(ch_names))):
            ax.annotate(ch_names[i], positions[i], ha='center', va='center',
                       fontsize=7, fontweight='bold', zorder=11)
    else:
        for i in range(n_nodes):
            ax.annotate(str(i), positions[i], ha='center', va='center',
                       fontsize=7, fontweight='bold', zorder=11)
    
    ax.set_xlim(-0.14, 0.14)
    ax.set_ylim(-0.14, 0.14)
    ax.set_aspect('equal')
    ax.axis('off')
    
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    
    # Add colorbar
    if show_colorbar and edges:
        sm = cm.ScalarMappable(norm=norm, cmap=colormap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('Attention', fontsize=9)


def plot_all_cluster_centroids(
    centroids: Dict[int, np.ndarray],
    positions: Dict[int, Tuple[float, float]],
    output_dir: Path,
    ch_names: Optional[List[str]] = None,
    condition_distribution: Optional[Dict] = None
):
    """
    Plot all cluster centroids in a grid.
    """
    n_clusters = len(centroids)
    n_cols = min(4, n_clusters)
    n_rows = (n_clusters + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_clusters == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for k in range(n_clusters):
        row, col = k // n_cols, k % n_cols
        ax = axes[row, col]
        
        # Build title with condition info
        title = f"State {k}"
        if condition_distribution:
            for cond, dist in condition_distribution.items():
                if k < len(dist):
                    title += f"\n{cond}: {dist[k]:.1%}"
        
        plot_attention_graph(
            centroids[k], positions, ax,
            title=title,
            ch_names=ch_names,
            edge_threshold_percentile=70,
            show_colorbar=False
        )
    
    # Hide empty subplots
    for k in range(n_clusters, n_rows * n_cols):
        row, col = k // n_cols, k % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle('Attention State Centroids (EEG Graph View)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'cluster_centroids_graph.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved cluster centroids: {output_dir / 'cluster_centroids_graph.png'}")


def plot_individual_clusters(
    centroids: Dict[int, np.ndarray],
    positions: Dict[int, Tuple[float, float]],
    output_dir: Path,
    ch_names: Optional[List[str]] = None
):
    """Save each cluster as individual high-res image."""
    clusters_dir = output_dir / 'cluster_graphs'
    clusters_dir.mkdir(exist_ok=True)
    
    for k, centroid in centroids.items():
        fig, ax = plt.subplots(figsize=(10, 10))
        
        plot_attention_graph(
            centroid, positions, ax,
            title=f"Attention State {k}",
            ch_names=ch_names,
            edge_threshold_percentile=60,
            node_size=1500,
            show_colorbar=True
        )
        
        plt.savefig(clusters_dir / f'state_{k}.png', dpi=200, bbox_inches='tight', 
                   facecolor='white')
        plt.close()
    
    logger.info(f"Saved individual cluster graphs to: {clusters_dir}")


def plot_transition_graph(
    transition_matrices: Dict[str, np.ndarray],
    n_clusters: int,
    output_dir: Path
):
    """
    Plot cluster transition graph with directed edges.
    """
    n_conditions = len(transition_matrices)
    fig, axes = plt.subplots(1, n_conditions, figsize=(8 * n_conditions, 8))
    
    if n_conditions == 1:
        axes = [axes]
    
    for idx, (condition, trans_mat) in enumerate(transition_matrices.items()):
        ax = axes[idx]
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes
        for i in range(n_clusters):
            G.add_node(i)
        
        # Add edges with weights
        for i in range(n_clusters):
            for j in range(n_clusters):
                if trans_mat[i, j] > 0.05:  # Only significant transitions
                    G.add_edge(i, j, weight=trans_mat[i, j])
        
        # Layout
        pos = nx.circular_layout(G)
        
        # Draw nodes
        node_colors = [cm.Set3(i / n_clusters) for i in range(n_clusters)]
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                              node_size=2000, edgecolors='black', linewidths=2)
        
        # Draw node labels
        labels = {i: f'S{i}' for i in range(n_clusters)}
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=14, font_weight='bold')
        
        # Draw edges with varying width/alpha
        edges = G.edges(data=True)
        for u, v, data in edges:
            weight = data['weight']
            
            # Self-loops
            if u == v:
                # Draw self-loop as arc
                loop_pos = (pos[u][0], pos[u][1] + 0.15)
                circle = mpatches.FancyBboxPatch(
                    (pos[u][0] - 0.05, pos[u][1] + 0.05), 0.1, 0.1,
                    boxstyle="round,pad=0.02", 
                    facecolor='none', edgecolor=cm.Reds(weight),
                    linewidth=1 + 5 * weight, alpha=0.3 + 0.7 * weight
                )
                ax.add_patch(circle)
            else:
                # Regular edge
                nx.draw_networkx_edges(
                    G, pos, edgelist=[(u, v)], ax=ax,
                    edge_color=[cm.Reds(weight)],
                    width=1 + 5 * weight,
                    alpha=0.3 + 0.7 * weight,
                    arrows=True, arrowsize=20,
                    connectionstyle='arc3,rad=0.1'
                )
        
        ax.set_title(f'State Transitions - {condition}', fontsize=14, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'transition_graph.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved transition graph: {output_dir / 'transition_graph.png'}")


def create_transition_video(
    matrices: np.ndarray,
    labels: np.ndarray,
    metadata: List[Dict],
    positions: Dict[int, Tuple[float, float]],
    output_dir: Path,
    ch_names: Optional[List[str]] = None,
    fps: int = 4,
    max_frames: int = 100
):
    """
    Create animated video showing attention patterns transitioning between states.
    """
    try:
        from matplotlib.animation import FuncAnimation, FFMpegWriter
    except ImportError:
        logger.warning("FFMpegWriter not available, trying alternative")
        try:
            from matplotlib.animation import FuncAnimation, PillowWriter as VideoWriter
            writer_class = 'pillow'
        except:
            logger.error("No video writer available. Install ffmpeg or pillow.")
            return
    else:
        writer_class = 'ffmpeg'
    
    # Select epochs to animate (in temporal order)
    n_frames = min(len(matrices), max_frames)
    
    # Sort by subject/condition/epoch to maintain temporal order
    sorted_indices = sorted(range(len(metadata)), 
                           key=lambda i: (metadata[i]['subject_id'], 
                                         metadata[i]['condition'],
                                         metadata[i]['epoch_idx']))[:n_frames]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    def init():
        ax.clear()
        return []
    
    def update(frame_idx):
        ax.clear()
        
        idx = sorted_indices[frame_idx]
        matrix = matrices[idx]
        label = labels[idx]
        meta = metadata[idx]
        
        title = f"Epoch {frame_idx + 1}/{n_frames} | State {label}\n"
        title += f"{meta['subject_id']} - {meta['condition']}"
        
        plot_attention_graph(
            matrix, positions, ax,
            title=title,
            ch_names=ch_names,
            edge_threshold_percentile=70,
            show_colorbar=False
        )
        
        # Add state indicator
        state_colors = [cm.Set3(i / max(labels) + 1) for i in range(max(labels) + 1)]
        indicator = plt.Circle((0.11, 0.11), 0.015, 
                               color=state_colors[label], 
                               edgecolor='black', linewidth=2)
        ax.add_patch(indicator)
        ax.text(0.11, 0.11, str(label), ha='center', va='center', 
               fontsize=10, fontweight='bold')
        
        return []
    
    logger.info(f"Creating animation with {n_frames} frames...")
    anim = FuncAnimation(fig, update, init_func=init, 
                        frames=n_frames, interval=1000/fps, blit=False)
    
    video_path = output_dir / 'attention_states_animation.mp4'
    
    if writer_class == 'ffmpeg':
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='attention_clustering'))
        anim.save(str(video_path), writer=writer)
    else:
        # Fallback to GIF
        video_path = output_dir / 'attention_states_animation.gif'
        anim.save(str(video_path), writer='pillow', fps=fps)
    
    plt.close()
    
    logger.info(f"Saved animation: {video_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize attention clusters as EEG graphs'
    )
    
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Directory with extracted attention states')
    parser.add_argument('--layer', type=str, default='layer0',
                       help='Which layer to visualize (default: layer0)')
    parser.add_argument('--edge-threshold', type=float, default=70,
                       help='Percentile threshold for showing edges (default: 70)')
    parser.add_argument('--fps', type=int, default=4,
                       help='Frames per second for video (default: 4)')
    parser.add_argument('--max-frames', type=int, default=100,
                       help='Maximum frames in video (default: 100)')
    parser.add_argument('--no-video', action='store_true',
                       help='Skip video generation')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    clustering_dir = input_dir / 'clustering'
    output_dir = clustering_dir / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load clustering results
    logger.info("Loading clustering results...")
    try:
        results = load_clustering_results(clustering_dir)
    except FileNotFoundError:
        logger.error(f"Clustering results not found. Run attention_clustering.py first.")
        sys.exit(1)
    
    labels = results['labels']
    n_clusters = results['n_clusters']
    temporal_analysis = results.get('temporal_analysis', {})
    metadata = results.get('metadata', [])
    
    logger.info(f"Found {n_clusters} clusters, {len(labels)} epochs")
    
    # Load attention matrices
    logger.info("Loading attention matrices...")
    matrices, matrix_metadata = load_attention_matrices(input_dir, layer=args.layer)
    
    if len(matrices) != len(labels):
        logger.warning(f"Matrix count ({len(matrices)}) != label count ({len(labels)})")
        # Use minimum
        n_samples = min(len(matrices), len(labels))
        matrices = matrices[:n_samples]
        labels = labels[:n_samples]
        metadata = metadata[:n_samples] if metadata else matrix_metadata[:n_samples]
    
    n_nodes = matrices.shape[1]
    logger.info(f"Loaded {len(matrices)} attention matrices ({n_nodes} nodes)")
    
    # Get electrode positions
    ch_names = load_electrode_names()
    if ch_names:
        ch_names = ch_names[:n_nodes]
    positions = get_electrode_positions(n_nodes, ch_names)
    
    # Compute cluster centroids
    logger.info("Computing cluster centroids...")
    centroids = compute_cluster_centroids(matrices, labels, n_clusters)
    
    # Plot all centroids in grid
    logger.info("Generating centroid visualizations...")
    plot_all_cluster_centroids(
        centroids, positions, output_dir, ch_names,
        condition_distribution=temporal_analysis.get('state_distribution')
    )
    
    # Plot individual high-res clusters
    plot_individual_clusters(centroids, positions, output_dir, ch_names)
    
    # Plot transition graph
    if temporal_analysis.get('transition_matrix'):
        logger.info("Generating transition graph...")
        plot_transition_graph(
            temporal_analysis['transition_matrix'],
            n_clusters, output_dir
        )
    
    # Create animation
    if not args.no_video:
        logger.info("Generating animation...")
        create_transition_video(
            matrices, labels, metadata or matrix_metadata,
            positions, output_dir, ch_names,
            fps=args.fps, max_frames=args.max_frames
        )
    
    print(f"\n{'='*60}")
    print("VISUALIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"  - cluster_centroids_graph.png (all states overview)")
    print(f"  - cluster_graphs/state_*.png (individual high-res)")
    print(f"  - transition_graph.png (state transitions)")
    if not args.no_video:
        print(f"  - attention_states_animation.mp4 (animated)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()













