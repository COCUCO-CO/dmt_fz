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
import os
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from multiprocessing import Pool, cpu_count
from functools import partial

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


def render_single_frame(args):
    """
    Render a single frame to disk. Designed for multiprocessing.
    
    Args:
        args: Tuple of (frame_idx, frame_data, render_params)
    """
    frame_idx, frame_data, params = args
    att_matrix, epoch_idx, is_real = frame_data
    
    # Unpack params
    ch_names = params['ch_names']
    coords = params['coords']
    vmin = params['vmin']
    vmax = params['vmax']
    use_mst = params['use_mst']
    condition = params['condition']
    layer_idx = params['layer_idx']
    temp_dir = params['temp_dir']
    dpi = params['dpi']
    edge_percentile = params['edge_percentile']
    figsize = params['figsize']
    total_frames = params['total_frames']
    total_epochs = params['total_epochs']
    current_epoch_num = params.get('epoch_numbers', {}).get(frame_idx, 0)
    
    # Import here to avoid issues with multiprocessing
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    import networkx as nx
    
    graph_type = "MST" if use_mst else "Attention"
    if is_real:
        title = f"{graph_type} - {condition} - Layer {layer_idx + 1} - Epoch {epoch_idx}"
    else:
        title = f"{graph_type} - {condition} - Layer {layer_idx + 1} (transition)"
    
    # Process matrix
    if use_mst:
        from utils.attention_logger import compute_mst_from_attention
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
            # Skip self-loops (i == j)
            if i == j:
                continue
            if matrix[i, j] > 0 and i < len(ch_names) and j < len(ch_names):
                if use_mst:
                    if i < j:
                        G.add_edge(ch_names[i], ch_names[j])
                        edge_weights[(ch_names[i], ch_names[j])] = matrix[i, j]
                else:
                    G.add_edge(ch_names[i], ch_names[j])
                    edge_weights[(ch_names[i], ch_names[j])] = matrix[i, j]
    
    # Filter edges by percentile if requested (speed optimization)
    if edge_percentile < 100 and edge_weights:
        threshold = np.percentile(list(edge_weights.values()), 100 - edge_percentile)
        edge_weights = {k: v for k, v in edge_weights.items() if v >= threshold}
    
    fig = plt.figure(figsize=figsize)
    # Main graph area (left portion, leave space for mini-map on right)
    ax_graph = fig.add_axes([0.02, 0.12, 0.72, 0.78])
    pos = {name: coords[name] for name in ch_names if name in coords}
    
    # Calculate attention focus using TOP-K weighted center of mass
    top_k = params.get('top_k_nodes', 3)  # Use top 3 nodes by default
    
    n_nodes = min(n, len(ch_names))
    
    # OUTGOING attention (node that attends TO others) = "Hub" / broadcaster
    outgoing_attention = np.zeros(n_nodes)
    # INCOMING attention (node that receives attention FROM others) = "Sink" / aggregator
    incoming_attention = np.zeros(n_nodes)
    
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j and i < att_matrix.shape[0] and j < att_matrix.shape[1]:
                outgoing_attention[i] += att_matrix[i, j]  # attention FROM node i TO j
                incoming_attention[j] += att_matrix[i, j]  # attention TO node j FROM i
    
    total_attention = outgoing_attention.sum()
    
    # === OUTGOING (Hub) calculation ===
    if total_attention > 0:
        top_indices_out = np.argsort(outgoing_attention)[-top_k:][::-1]
        top_attention_out = outgoing_attention[top_indices_out]
        top_total_out = top_attention_out.sum()
        
        if top_total_out > 0:
            center_x_out, center_y_out = 0.0, 0.0
            for idx, att in zip(top_indices_out, top_attention_out):
                name = ch_names[idx] if idx < len(ch_names) else None
                if name and name in coords:
                    weight = att / top_total_out
                    center_x_out += coords[name][0] * weight
                    center_y_out += coords[name][1] * weight
        else:
            center_x_out, center_y_out = 0.0, 0.0
    else:
        center_x_out, center_y_out = 0.0, 0.0
        top_indices_out = [0]
    
    # === INCOMING (Sink) calculation ===
    if total_attention > 0:
        top_indices_in = np.argsort(incoming_attention)[-top_k:][::-1]
        top_attention_in = incoming_attention[top_indices_in]
        top_total_in = top_attention_in.sum()
        
        if top_total_in > 0:
            center_x_in, center_y_in = 0.0, 0.0
            for idx, att in zip(top_indices_in, top_attention_in):
                name = ch_names[idx] if idx < len(ch_names) else None
                if name and name in coords:
                    weight = att / top_total_in
                    center_x_in += coords[name][0] * weight
                    center_y_in += coords[name][1] * weight
        else:
            center_x_in, center_y_in = 0.0, 0.0
    else:
        center_x_in, center_y_in = 0.0, 0.0
        top_indices_in = [0]
    
    weight_range = vmax - vmin if vmax > vmin else 1.0
    sorted_edges = sorted(edge_weights.items(), key=lambda x: x[1])
    
    # Auto-filter for dense graphs: only show top edges if too many
    num_edges = len(sorted_edges)
    if num_edges > 100:
        # Keep only top 30% of edges for readability
        keep_count = max(int(num_edges * 0.3), 50)
        sorted_edges = sorted_edges[-keep_count:]
        # Recalculate vmin/vmax for visible edges only
        visible_weights = [w for _, w in sorted_edges]
        if visible_weights:
            vmin_vis = min(visible_weights)
            vmax_vis = max(visible_weights)
            weight_range = vmax_vis - vmin_vis if vmax_vis > vmin_vis else 1.0
            vmin = vmin_vis
    
    for (src, dst), weight in sorted_edges:
        if src in pos and dst in pos:
            normalized = (weight - vmin) / weight_range
            normalized = max(0, min(1, normalized))
            # Refined visual parameters for cleaner look
            alpha = 0.08 + normalized * 0.7  # More transparent base
            width = 0.3 + normalized * 2.5   # Thinner lines
            # Better color gradient: light orange to dark red
            color = cm.YlOrRd(0.2 + normalized * 0.8)
            nx.draw_networkx_edges(G, pos, edgelist=[(src, dst)],
                                  width=width, edge_color=[color], alpha=alpha,
                                  ax=ax_graph, arrows=not use_mst,
                                  arrowsize=6 if not use_mst else 0,
                                  connectionstyle="arc3,rad=0.05" if not use_mst else None)
    
    node_list = [name for name in ch_names if name in pos]
    nx.draw_networkx_nodes(G, pos, nodelist=node_list,
                          node_color='white', node_size=700,
                          edgecolors='#333333', linewidths=1.5,
                          alpha=0.98, ax=ax_graph)
    
    for node_name in node_list:
        if node_name in pos:
            x, y = pos[node_name]
            ax_graph.text(x, y, node_name, fontsize=7, fontweight='bold',
                         ha='center', va='center', color='#222222')
    
    ax_graph.axis('equal')
    ax_graph.set_xlim(-0.15, 0.15)
    ax_graph.set_ylim(-0.16, 0.16)
    ax_graph.axis('off')
    
    # === Mini-map 1: HUB Tracking (outgoing attention) - top-right ===
    ax_hub = fig.add_axes([0.73, 0.56, 0.24, 0.34])
    ax_hub.set_xlim(-0.16, 0.16)
    ax_hub.set_ylim(-0.17, 0.17)
    
    # Draw head outline
    theta = np.linspace(0, 2*np.pi, 100)
    head_r = 0.14
    ax_hub.plot(head_r * np.cos(theta), head_r * np.sin(theta), 
                color='#dddddd', linewidth=1.2, zorder=0)
    ax_hub.plot([0, 0], [head_r, head_r + 0.015], color='#dddddd', linewidth=1.2, zorder=0)
    
    # Draw electrodes - highlight top-K hubs in red
    top_k_set_out = set(top_indices_out)
    colors_hub = ['#ff3333', '#ff7777', '#ffaaaa']
    
    for i, name in enumerate(ch_names):
        if name in coords:
            x, y = coords[name]
            if i in top_k_set_out:
                rank = list(top_indices_out).index(i)
                if rank < len(colors_hub):
                    ax_hub.scatter(x, y, s=[70, 45, 30][rank], c=colors_hub[rank], 
                                  edgecolors=['darkred', '#cc5555', '#aa7777'][rank],
                                  linewidths=1.2, zorder=10-rank, alpha=0.9)
            else:
                ax_hub.scatter(x, y, s=15, c='#e8e8e8', edgecolors='#bbbbbb', 
                               linewidths=0.4, zorder=1, alpha=0.6)
    
    # Draw hub trail
    all_centers_out = params.get('all_centers_out', [])
    if all_centers_out and frame_idx > 0:
        trail_length = min(50, frame_idx)
        recent = all_centers_out[max(0, frame_idx - trail_length):frame_idx]
        if len(recent) > 1:
            for i, (cx, cy) in enumerate(recent):
                prog = (i + 1) / len(recent)
                ax_hub.scatter(cx, cy, s=5 + 12*prog, c='#4488ff', alpha=0.1 + 0.3*prog, 
                              zorder=2, edgecolors='none')
            for i in range(len(recent) - 1):
                prog = (i + 1) / len(recent)
                ax_hub.plot([recent[i][0], recent[i+1][0]], [recent[i][1], recent[i+1][1]], 
                           color='#4488ff', alpha=0.05 + 0.2*prog, linewidth=1, zorder=1)
    
    ax_hub.scatter(center_x_out, center_y_out, s=140, c='#2266dd', edgecolors='white',
                  linewidths=2, zorder=20, marker='D')
    ax_hub.set_title('Hub (broadcaster)', fontsize=8, fontweight='bold', pad=2, color='#cc0000')
    ax_hub.axis('off')
    
    # Hub info text
    if total_attention > 0:
        hub_names = [ch_names[i] if i < len(ch_names) else "?" for i in top_indices_out[:3]]
        hub_pcts = [outgoing_attention[i] / total_attention * 100 for i in top_indices_out[:3]]
        hub_text = " | ".join([f"{n}:{p:.1f}%" for n, p in zip(hub_names, hub_pcts)])
    else:
        hub_text = "—"
    ax_hub.text(0.5, -0.08, hub_text, transform=ax_hub.transAxes, ha='center', 
               fontsize=7, color='#666666', fontfamily='monospace')
    
    # === Mini-map 2: SINK Tracking (incoming attention) - middle-right ===
    ax_sink = fig.add_axes([0.73, 0.15, 0.24, 0.34])
    ax_sink.set_xlim(-0.16, 0.16)
    ax_sink.set_ylim(-0.17, 0.17)
    
    # Draw head outline
    ax_sink.plot(head_r * np.cos(theta), head_r * np.sin(theta), 
                color='#dddddd', linewidth=1.2, zorder=0)
    ax_sink.plot([0, 0], [head_r, head_r + 0.015], color='#dddddd', linewidth=1.2, zorder=0)
    
    # Draw electrodes - highlight top-K sinks in green
    top_k_set_in = set(top_indices_in)
    colors_sink = ['#33aa33', '#77cc77', '#aaddaa']
    
    for i, name in enumerate(ch_names):
        if name in coords:
            x, y = coords[name]
            if i in top_k_set_in:
                rank = list(top_indices_in).index(i)
                if rank < len(colors_sink):
                    ax_sink.scatter(x, y, s=[70, 45, 30][rank], c=colors_sink[rank], 
                                   edgecolors=['darkgreen', '#55aa55', '#77aa77'][rank],
                                   linewidths=1.2, zorder=10-rank, alpha=0.9)
            else:
                ax_sink.scatter(x, y, s=15, c='#e8e8e8', edgecolors='#bbbbbb', 
                               linewidths=0.4, zorder=1, alpha=0.6)
    
    # Draw sink trail
    all_centers_in = params.get('all_centers_in', [])
    if all_centers_in and frame_idx > 0:
        trail_length = min(50, frame_idx)
        recent = all_centers_in[max(0, frame_idx - trail_length):frame_idx]
        if len(recent) > 1:
            for i, (cx, cy) in enumerate(recent):
                prog = (i + 1) / len(recent)
                ax_sink.scatter(cx, cy, s=5 + 12*prog, c='#44bb44', alpha=0.1 + 0.3*prog, 
                               zorder=2, edgecolors='none')
            for i in range(len(recent) - 1):
                prog = (i + 1) / len(recent)
                ax_sink.plot([recent[i][0], recent[i+1][0]], [recent[i][1], recent[i+1][1]], 
                            color='#44bb44', alpha=0.05 + 0.2*prog, linewidth=1, zorder=1)
    
    ax_sink.scatter(center_x_in, center_y_in, s=140, c='#228822', edgecolors='white',
                   linewidths=2, zorder=20, marker='D')
    ax_sink.set_title('Sink (aggregator)', fontsize=8, fontweight='bold', pad=2, color='#228822')
    ax_sink.axis('off')
    
    # Sink info text
    if total_attention > 0:
        sink_names = [ch_names[i] if i < len(ch_names) else "?" for i in top_indices_in[:3]]
        sink_pcts = [incoming_attention[i] / total_attention * 100 for i in top_indices_in[:3]]
        sink_text = " | ".join([f"{n}:{p:.1f}%" for n, p in zip(sink_names, sink_pcts)])
    else:
        sink_text = "—"
    ax_sink.text(0.5, -0.08, sink_text, transform=ax_sink.transAxes, ha='center', 
                fontsize=7, color='#666666', fontfamily='monospace')
    
    # === Colorbar (moved up to avoid overlap) ===
    cbar_ax = fig.add_axes([0.08, 0.08, 0.55, 0.015])
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(norm=norm, cmap=cm.Reds)
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(f'Attention Weight', fontsize=8, labelpad=2)
    cbar.ax.tick_params(labelsize=7)
    
    # Title
    fig.text(0.38, 0.96, title, fontsize=12, fontweight='bold', ha='center', va='top')
    
    # === Progress bar (at the very bottom) ===
    progress = (frame_idx + 1) / total_frames
    progress_ax = fig.add_axes([0.08, 0.025, 0.55, 0.018])
    progress_ax.set_xlim(0, 1)
    progress_ax.set_ylim(0, 1)
    progress_ax.axis('off')
    
    # Background bar (gray)
    progress_ax.add_patch(plt.Rectangle((0, 0), 1, 1, 
                          facecolor='#e8e8e8', edgecolor='#aaaaaa', 
                          linewidth=1, zorder=1))
    
    # Progress fill
    progress_ax.add_patch(plt.Rectangle((0, 0), progress, 1, 
                          facecolor='#4facfe', edgecolor='none', 
                          zorder=2))
    # Highlight line at top
    progress_ax.plot([0, progress], [0.8, 0.8], color='white', 
                     linewidth=1.2, alpha=0.5, zorder=3)
    
    # Border
    progress_ax.add_patch(plt.Rectangle((0, 0), 1, 1, 
                          facecolor='none', edgecolor='#888888', 
                          linewidth=1, zorder=4))
    
    # Progress text (to the right of the bar)
    epoch_display = epoch_idx if is_real else "~"
    progress_text = f"Epoch {epoch_display}/{total_epochs} • {progress*100:.0f}%"
    fig.text(0.64, 0.034, progress_text, fontsize=8, ha='left', va='center',
             color='#555555', fontweight='medium')
    
    # Save frame
    frame_path = os.path.join(temp_dir, f"frame_{frame_idx:06d}.png")
    fig.savefig(frame_path, dpi=dpi, facecolor='white', edgecolor='none')
    plt.close(fig)
    
    return frame_idx


def generate_animation(model, subject_data: Dict[str, List],
                      device, ch_names: List[str],
                      coords: Dict[str, Tuple[float, float]],
                      output_dir: Path,
                      subject_id: str,
                      layer_idx: int = 0,
                      fps: int = 5,
                      use_mst: bool = False,
                      transition_frames: int = 3,
                      n_workers: int = 1,
                      dpi: int = 150,
                      edge_percentile: float = 100,
                      bitrate: int = 2000,
                      figsize: Tuple[float, float] = (12, 10)):
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
        n_workers: Number of parallel workers (default: 1, sequential)
        dpi: Output DPI (default: 150, use 100 for faster rendering)
        edge_percentile: Only show top N% of edges (default: 100 = all)
        bitrate: Video bitrate (default: 2000)
        figsize: Figure size tuple (default: (10, 9))
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
        
        total_frames = len(frames_data)
        logger.info(f"Creating animation with {total_frames} frames ({len(attention_matrices)} epochs + transitions)")
        
        # Create temp directory for frames
        temp_dir = tempfile.mkdtemp(prefix="anim_frames_")
        
        try:
            # Build epoch number mapping for progress display
            epoch_numbers = {}
            real_epoch_count = 0
            for i, (_, epoch_idx, is_real) in enumerate(frames_data):
                if is_real:
                    real_epoch_count += 1
                epoch_numbers[i] = real_epoch_count
            
            # Pre-calculate all attention focus positions (top-K weighted centers)
            logger.info("Pre-calculating hub and sink positions...")
            top_k = 3
            all_centers_out = []  # Hub (outgoing)
            all_centers_in = []   # Sink (incoming)
            n_nodes = len(ch_names)
            
            for att_matrix, _, _ in frames_data:
                outgoing_attention = np.zeros(n_nodes)
                incoming_attention = np.zeros(n_nodes)
                
                for i in range(min(n_nodes, att_matrix.shape[0])):
                    for j in range(min(n_nodes, att_matrix.shape[1])):
                        if i != j:
                            outgoing_attention[i] += att_matrix[i, j]
                            incoming_attention[j] += att_matrix[i, j]
                
                total_att = outgoing_attention.sum()
                
                # Outgoing (Hub) center
                if total_att > 0:
                    top_idx = np.argsort(outgoing_attention)[-top_k:][::-1]
                    top_att = outgoing_attention[top_idx]
                    top_total = top_att.sum()
                    if top_total > 0:
                        cx, cy = 0.0, 0.0
                        for idx, att in zip(top_idx, top_att):
                            name = ch_names[idx] if idx < len(ch_names) else None
                            if name and name in coords:
                                cx += coords[name][0] * (att / top_total)
                                cy += coords[name][1] * (att / top_total)
                        all_centers_out.append((cx, cy))
                    else:
                        all_centers_out.append((0.0, 0.0))
                else:
                    all_centers_out.append((0.0, 0.0))
                
                # Incoming (Sink) center
                if total_att > 0:
                    top_idx = np.argsort(incoming_attention)[-top_k:][::-1]
                    top_att = incoming_attention[top_idx]
                    top_total = top_att.sum()
                    if top_total > 0:
                        cx, cy = 0.0, 0.0
                        for idx, att in zip(top_idx, top_att):
                            name = ch_names[idx] if idx < len(ch_names) else None
                            if name and name in coords:
                                cx += coords[name][0] * (att / top_total)
                                cy += coords[name][1] * (att / top_total)
                        all_centers_in.append((cx, cy))
                    else:
                        all_centers_in.append((0.0, 0.0))
                else:
                    all_centers_in.append((0.0, 0.0))
            
            # Prepare render parameters (shared across all frames)
            render_params = {
                'ch_names': ch_names,
                'coords': coords,
                'vmin': vmin,
                'vmax': vmax,
                'use_mst': use_mst,
                'condition': condition,
                'layer_idx': layer_idx,
                'temp_dir': temp_dir,
                'dpi': dpi,
                'edge_percentile': edge_percentile,
                'figsize': figsize,
                'total_frames': total_frames,
                'total_epochs': len(attention_matrices),
                'epoch_numbers': epoch_numbers,
                'all_centers_out': all_centers_out,
                'all_centers_in': all_centers_in,
                'top_k_nodes': top_k,
            }
            
            # Prepare args for each frame
            frame_args = [(i, frames_data[i], render_params) for i in range(total_frames)]
            
            # Render frames (parallel or sequential)
            if n_workers > 1:
                logger.info(f"Rendering frames in parallel with {n_workers} workers...")
                with Pool(processes=n_workers) as pool:
                    list(tqdm(pool.imap(render_single_frame, frame_args), 
                             total=total_frames, desc="Rendering frames"))
            else:
                logger.info("Rendering frames sequentially...")
                for args in tqdm(frame_args, desc="Rendering frames"):
                    render_single_frame(args)
            
            # Combine frames with ffmpeg
            graph_type = "mst" if use_mst else "attention"
            output_path = output_dir / f"{graph_type}_animation_{subject_id}_{condition}_layer{layer_idx + 1}.mp4"
            
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-framerate', str(fps),
                '-i', os.path.join(temp_dir, 'frame_%06d.png'),
                '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',  # ensure even dimensions for h264
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-b:v', f'{bitrate}k',
                '-preset', 'fast',  # faster encoding
                str(output_path)
            ]
            
            logger.info("Combining frames with ffmpeg...")
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"ffmpeg error: {result.stderr}")
                raise RuntimeError("ffmpeg failed")
            
            logger.info(f"Saved: {output_path}")
            
        finally:
            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)


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
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of parallel workers for rendering (default: 1)')
    parser.add_argument('--dpi', type=int, default=150,
                       help='Output DPI (default: 150, use 100 for faster)')
    parser.add_argument('--edge-percentile', type=float, default=100,
                       help='Only show top N%% of edges (default: 100; auto-filters to 30%% if >100 edges)')
    parser.add_argument('--bitrate', type=int, default=2000,
                       help='Video bitrate in kbps (default: 2000)')
    
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
        transition_frames=args.transitions,
        n_workers=args.workers,
        dpi=args.dpi,
        edge_percentile=args.edge_percentile,
        bitrate=args.bitrate
    )
    
    logger.info(f"\nAnimations saved to: {output_dir}")


if __name__ == '__main__':
    main()

