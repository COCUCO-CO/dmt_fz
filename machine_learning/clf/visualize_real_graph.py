#!/usr/bin/env python3
"""
Visualize a REAL graph from the dataset with actual feature values.
"""

import sys
import yaml
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import Normalize
import matplotlib.cm as cm

sys.path.append(str(Path(__file__).parent))
from data import create_dataset_from_config

EXTRA_PKL = Path('/media/storage_hdd/dmt_fz/fwd-inv-stc/extra.pkl')

def load_electrode_info():
    """Load electrode names and 2D coordinates from extra.pkl"""
    with open(EXTRA_PKL, 'rb') as f:
        data = pickle.load(f)
    ch_names = data[4]
    mapping = data[5]
    eeg_coords_2d = data[6]
    return ch_names, mapping, eeg_coords_2d


def visualize_real_graph(graph, save_path: Path, ch_names, eeg_coords_2d):
    """Visualize a real graph with maximum size and proper spacing."""
    
    # LAYOUT APROVECHANDO TODO EL ESPACIO
    fig = plt.figure(figsize=(28, 18))
    
    # Grid: grafo GIGANTE a la izquierda, boxes a la derecha
    gs_main = fig.add_gridspec(3, 2, 
                               height_ratios=[8.5, 0.4, 1.7],  # Más espacio para histograma
                               width_ratios=[2.2, 1],
                               hspace=0.18, 
                               wspace=0.40,
                               left=0.04, 
                               right=0.97, 
                               top=0.94,
                               bottom=0.03)
    
    # Extract data
    num_nodes = graph.num_nodes
    edge_index = graph.edge_index.numpy()
    edge_attr = graph.edge_attr.numpy().flatten()
    node_features = graph.x.numpy()
    graph_features = graph.graph_attr.numpy()
    
    condition = graph.condition
    band = graph.band
    subject_id = graph.subject_id
    epoch_idx = graph.epoch_idx
    
    # ========================================================================
    # ROW 0: GRAPH (left) - MÁXIMO TAMAÑO
    # ========================================================================
    ax_graph = fig.add_subplot(gs_main[0, 0])
    
    # Build graph
    G = nx.Graph()
    for i in range(num_nodes):
        G.add_node(ch_names[i])
    
    edge_weights = {}
    for i in range(edge_index.shape[1]):
        src_idx, dst_idx = edge_index[0, i], edge_index[1, i]
        if src_idx < dst_idx:
            G.add_edge(ch_names[src_idx], ch_names[dst_idx])
            edge_weights[(ch_names[src_idx], ch_names[dst_idx])] = edge_attr[i]
    
    pos = eeg_coords_2d
    all_sync_values = list(edge_weights.values())
    sorted_edges = sorted(edge_weights.items(), key=lambda x: x[1])
    
    # Draw edges - Más finas y claras
    for (src, dst), sync_val in sorted_edges:
        alpha = 0.06 + sync_val * 0.45
        width = 0.3 + sync_val * 3.5
        color = cm.Reds(sync_val)
        nx.draw_networkx_edges(G, pos, edgelist=[(src, dst)],
                              width=width, edge_color=[color], alpha=alpha, ax=ax_graph)
    
    # Draw nodes - MÁS GRANDES
    node_colors = node_features[:, 0]
    node_names_list = [ch_names[i] for i in range(num_nodes)]
    phase_min, phase_max = node_colors.min(), node_colors.max()
    phase_range = phase_max - phase_min
    padding = phase_range * 0.1
    vmin, vmax = phase_min - padding, phase_max + padding
    
    nodes = nx.draw_networkx_nodes(G, pos, nodelist=node_names_list,
                                   node_color=node_colors, node_size=2000,
                                   cmap='RdBu_r', vmin=vmin, vmax=vmax,
                                   edgecolors='black', linewidths=4.5, alpha=0.95, ax=ax_graph)
    
    # Labels - proporcionales al tamaño de nodos
    for node_name in node_names_list:
        x, y = pos[node_name]
        ax_graph.text(x, y, node_name, fontsize=11, fontweight='bold',
                     ha='center', va='center', color='black',
                     bbox=dict(boxstyle='round,pad=0.38', facecolor='white',
                              edgecolor='black', linewidth=0.8, alpha=0.90))
    
    # NODOS MÁS SEPARADOS - aumentar límites
    ax_graph.axis('equal')
    ax_graph.set_xlim(-0.145, 0.145)
    ax_graph.set_ylim(-0.135, 0.175)
    ax_graph.axis('off')
    
    # ========================================================================
    # RIGHT SIDE: 3 BOXES STACKED
    # ========================================================================
    gs_right = gs_main[0, 1].subgridspec(3, 1, hspace=0.50)
    
    # BOX 1: NODE FEATURES
    ax_node = fig.add_subplot(gs_right[0])
    ax_node.axis('off')
    
    example_idx = ch_names.index('Cz') if 'Cz' in ch_names else 0
    example_name = ch_names[example_idx]
    
    text = f"NODE FEATURES - Example: {example_name}\n" + "="*40 + "\n"
    feat_names = ['Phase Mean', 'Phase Std', 'Amp Mean', 'Amp Std', 
                  'Entropy', 'Kurtosis', 'CV', 'Range']
    
    for i, name in enumerate(feat_names):
        if i < node_features.shape[1]:
            text += f"{name:12s}: {node_features[example_idx, i]:8.4f}\n"
    
    text += "-"*40 + f"\n{node_features.shape[1]} features × {num_nodes} electrodes"
    
    ax_node.text(0.05, 0.95, text, transform=ax_node.transAxes,
                fontsize=14, va='top', family='monospace',
                bbox=dict(boxstyle='round,pad=0.7', fc='lightblue', ec='darkblue', lw=2.8, alpha=0.95))
    
    # BOX 2: GRAPH-LEVEL FEATURES
    ax_global = fig.add_subplot(gs_right[1])
    ax_global.axis('off')
    
    text = "GRAPH-LEVEL FEATURES\n" + "="*40 + "\n"
    names = ['Kuramoto Mean', 'Kuramoto Std', 'Sync Mean', 'Sync Std',
             'Density', 'Avg Degree', 'Max Degree', 'Min Degree']
    
    for i, name in enumerate(names):
        if i < len(graph_features):
            text += f"{name:15s}: {graph_features[i]:8.4f}\n"
    
    text += "-"*40 + f"\nClass: {condition}\nLabel: {graph.y.item()}"
    
    ax_global.text(0.05, 0.95, text, transform=ax_global.transAxes,
                  fontsize=14, va='top', family='monospace',
                  bbox=dict(boxstyle='round,pad=0.7', fc='lightgreen', ec='darkgreen', lw=2.8, alpha=0.95))
    
    # BOX 3: DIMENSIONS SUMMARY
    ax_summary = fig.add_subplot(gs_right[2])
    ax_summary.axis('off')
    
    total_n = num_nodes * node_features.shape[1]
    total_e = len(edge_weights)
    total_g = len(graph_features)
    
    text = "DIMENSIONS SUMMARY\n" + "="*40 + "\n"
    text += f"Nodes:           {num_nodes}\n"
    text += f"Edges:           {len(edge_weights)}\n"
    text += f"Node feat dim:   {node_features.shape[1]}\n"
    text += f"Edge feat dim:   1\n"
    text += f"Graph feat dim:  {len(graph_features)}\n"
    text += "\nTOTAL FEATURES:\n"
    text += f"  Nodes:  {total_n:4d}  ({num_nodes}×{node_features.shape[1]})\n"
    text += f"  Edges:  {total_e:4d}\n"
    text += f"  Graph:  {total_g:4d}\n"
    text += f"  " + "─"*28 + "\n"
    text += f"  TOTAL:  {total_n+total_e+total_g:4d}"
    
    ax_summary.text(0.05, 0.95, text, transform=ax_summary.transAxes,
                   fontsize=14, va='top', family='monospace',
                   bbox=dict(boxstyle='round,pad=0.7', fc='wheat', ec='orange', lw=2.8, alpha=0.95))
    
    # ========================================================================
    # ROW 1: COLORBARS - CASI AL BORDE INFERIOR
    # ========================================================================
    ax_cbars = fig.add_subplot(gs_main[1, 0])
    ax_cbars.axis('off')
    
    sync_min, sync_max = min(all_sync_values), max(all_sync_values)
    
    # Colorbars casi al final de la imagen - más grandes y legibles
    cbar_ax1 = fig.add_axes([0.065, 0.12, 0.22, 0.022])
    cbar_n = plt.colorbar(nodes, cax=cbar_ax1, orientation='horizontal')
    cbar_n.set_label(f'Node Color: Mean Phase [{phase_min:.3f}, {phase_max:.3f}] rad', fontsize=12, fontweight='bold')
    cbar_n.ax.tick_params(labelsize=10)
    
    cbar_ax2 = fig.add_axes([0.305, 0.12, 0.22, 0.022])
    cbar_e = plt.colorbar(cm.ScalarMappable(norm=Normalize(sync_min, sync_max), cmap=cm.Reds),
                         cax=cbar_ax2, orientation='horizontal')
    cbar_e.set_label(f'Edge Color/Thickness: Synchronization [{sync_min:.3f}, {sync_max:.3f}]', fontsize=12, fontweight='bold')
    cbar_e.ax.tick_params(labelsize=10)
    
    # ========================================================================
    # ROW 2: HISTOGRAM (bottom right)
    # ========================================================================
    ax_hist = fig.add_subplot(gs_main[2, 1])
    
    ax_hist.hist(all_sync_values, bins=40, ec='black', alpha=0.78, color='crimson', lw=1.5)
    ax_hist.set_xlabel('Synchronization Value', fontsize=14, fontweight='bold')
    ax_hist.set_ylabel('Count', fontsize=14, fontweight='bold')
    ax_hist.set_title('Edge Features Distribution (Sync Values)', fontsize=15, fontweight='bold', pad=16)
    ax_hist.grid(True, alpha=0.35, ls='--', lw=1.0)
    ax_hist.tick_params(labelsize=12)
    
    stats = f"{len(all_sync_values)} edges\nμ={np.mean(all_sync_values):.3f}\nσ={np.std(all_sync_values):.3f}"
    ax_hist.text(0.97, 0.97, stats, transform=ax_hist.transAxes,
                fontsize=13, va='top', ha='right', family='monospace',
                bbox=dict(boxstyle='round,pad=0.55', fc='white', ec='red', lw=2.4, alpha=0.95))
    
    # ========================================================================
    # TÍTULOS - PERFECTAMENTE ALINEADOS
    # ========================================================================
    # Título principal
    fig.text(0.5, 0.980, 'Graph Neural Network - EEG Data Structure (Fully Connected)',
            fontsize=17, fontweight='bold', ha='center', va='top')
    
    # Subtítulo - justo debajo del título principal
    fig.text(0.5, 0.955, f'EEG Graph: {condition} | Band: {band} | Subject: {subject_id} | Epoch: {epoch_idx}',
            fontsize=14, fontweight='bold', ha='center', va='top')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {save_path.name}")


if __name__ == '__main__':
    print("="*80)
    print("Creating Real EEG Graph Visualizations")
    print("="*80)
    print("\nLoading electrode information...")
    
    ch_names, mapping, eeg_coords_2d = load_electrode_info()
    print(f"✓ Loaded {len(ch_names)} electrodes")
    
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("\nLoading graphs...")
    train_graphs, _, _ = create_dataset_from_config(config)
    print(f"✓ Loaded {len(train_graphs)} graphs")
    
    output_dir = Path('/media/storage_hdd/dmt_fz/machine_learning/clf/output/analysis')
    conditions = config['data']['conditions']
    
    print(f"\nGenerating visualizations:")
    
    for cond_idx, cond_name in enumerate(conditions):
        graph = None
        for g in train_graphs:
            if g.y.item() == cond_idx:
                graph = g
                break
        
        if graph is not None:
            save_path = output_dir / f'graph_real_example_{cond_name}.png'
            print(f"  {cond_name}...", end=' ')
            visualize_real_graph(graph, save_path, ch_names, eeg_coords_2d)
    
    print("\n" + "="*80)
    print("Complete!")
    print("="*80)
