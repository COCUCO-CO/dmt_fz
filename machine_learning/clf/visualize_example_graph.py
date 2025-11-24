#!/usr/bin/env python3
"""
Visualize example EEG graph with all features clearly labeled.

Creates a detailed diagram showing:
- Nodes (electrodes) with labels
- Node features (phase/amplitude stats)
- Edge features (synchronization values)
- Graph-level features (Kuramoto parameters)
"""

import sys
import yaml
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import networkx as nx

sys.path.append(str(Path(__file__).parent))
from data import create_dataset_from_config

# Standard 10-20 EEG electrode positions (approximate 2D projection)
ELECTRODE_POSITIONS = {
    'Fp1': (-0.3, 0.9), 'Fp2': (0.3, 0.9),
    'F7': (-0.7, 0.6), 'F3': (-0.3, 0.6), 'Fz': (0.0, 0.6), 'F4': (0.3, 0.6), 'F8': (0.7, 0.6),
    'T3': (-0.9, 0.0), 'C3': (-0.3, 0.0), 'Cz': (0.0, 0.0), 'C4': (0.3, 0.0), 'T4': (0.9, 0.0),
    'T5': (-0.7, -0.6), 'P3': (-0.3, -0.6), 'Pz': (0.0, -0.6), 'P4': (0.3, -0.6), 'T6': (0.7, -0.6),
    'O1': (-0.3, -0.9), 'O2': (0.3, -0.9),
    # Additional channels
    'Fpz': (0.0, 0.95), 'AFz': (0.0, 0.75), 'CPz': (0.0, -0.3), 'POz': (0.0, -0.75), 'Oz': (0.0, -0.95)
}


def create_example_visualization(save_path: Path):
    """
    Create a comprehensive visualization of an example EEG graph.
    """
    fig = plt.figure(figsize=(20, 14))
    
    # Create grid for subplots
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # ========================================================================
    # 1. MAIN GRAPH VISUALIZATION (top left, large)
    # ========================================================================
    
    ax_main = fig.add_subplot(gs[0:2, 0:2])
    
    # Create a simplified example with 8 electrodes for clarity
    electrodes = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4']
    pos = {elec: ELECTRODE_POSITIONS[elec] for elec in electrodes}
    
    # Create graph
    G = nx.Graph()
    G.add_nodes_from(electrodes)
    
    # Add example edges with synchronization values
    edges_with_sync = [
        ('Fp1', 'Fp2', 0.65),
        ('Fp1', 'F3', 0.82),
        ('Fp2', 'F4', 0.78),
        ('F3', 'F4', 0.55),
        ('F3', 'C3', 0.91),
        ('F4', 'C4', 0.88),
        ('C3', 'C4', 0.72),
        ('C3', 'P3', 0.85),
        ('C4', 'P4', 0.87),
        ('P3', 'P4', 0.68),
    ]
    
    for e1, e2, sync in edges_with_sync:
        G.add_edge(e1, e2, weight=sync)
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, 
        node_color='lightblue', 
        node_size=1000,
        edgecolors='darkblue',
        linewidths=3,
        ax=ax_main
    )
    
    # Draw edges with varying thickness based on synchronization
    for (e1, e2, data) in G.edges(data=True):
        sync = data['weight']
        width = sync * 5  # Scale for visibility
        alpha = 0.3 + sync * 0.5  # More sync = more visible
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(e1, e2)],
            width=width,
            alpha=alpha,
            edge_color='red',
            ax=ax_main
        )
    
    # Draw labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=12,
        font_weight='bold',
        ax=ax_main
    )
    
    # Add edge labels (synchronization values) for a few edges
    edge_labels_subset = {
        ('Fp1', 'F3'): '0.82',
        ('F3', 'C3'): '0.91',
        ('C3', 'P3'): '0.85',
    }
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels_subset,
        font_size=10,
        font_color='red',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
        ax=ax_main
    )
    
    ax_main.set_title('EEG Graph Example: 24 Electrodes (Simplified to 8 for clarity)\n'
                     'Fully Connected = All electrodes connected to all others',
                     fontsize=16, fontweight='bold', pad=20)
    ax_main.axis('off')
    
    # Add legend for edges
    ax_main.text(0.02, 0.98, 'Edge Features:\n• Thickness ∝ Synchronization\n• Color intensity ∝ Strength',
                transform=ax_main.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ========================================================================
    # 2. NODE FEATURES EXPLANATION (top right)
    # ========================================================================
    
    ax_node = fig.add_subplot(gs[0, 2])
    ax_node.axis('off')
    
    # Create example node with features
    node_info = """
NODE FEATURES (per electrode)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Example: Electrode "C3"
┌─────────────────────────────┐
│  Phase Statistics:          │
│  • Mean phase: 1.23 rad     │
│  • Std phase:  0.87 rad     │
│                              │
│  Amplitude Statistics:       │
│  • Mean amp:   0.45 µV      │
│  • Std amp:    0.12 µV      │
│                              │
│  Temporal Complexity:        │
│  • Entropy:    2.34         │
│  • Kurtosis:   1.87         │
│  • CV:         0.56         │
│  • Range:      4.21         │
└─────────────────────────────┘

Total: 8 features per node
(6 basic + 4 complexity)
"""
    
    ax_node.text(0.1, 0.95, node_info,
                transform=ax_node.transAxes,
                fontsize=11,
                verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
    
    # ========================================================================
    # 3. EDGE FEATURES EXPLANATION (middle right)
    # ========================================================================
    
    ax_edge = fig.add_subplot(gs[1, 2])
    ax_edge.axis('off')
    
    edge_info = """
EDGE FEATURES (per connection)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Example: Edge "C3 → P3"
┌─────────────────────────────┐
│  Synchronization Value:     │
│  • sync = 0.85              │
│                              │
│  Interpretation:            │
│  • 0.0 = No synchrony       │
│  • 1.0 = Perfect synchrony  │
│                              │
│  Computed from:             │
│  • Phase difference         │
│  • Between two electrodes   │
│  • Over time window         │
└─────────────────────────────┘

Total: 1 feature per edge
(synchronization strength)
"""
    
    ax_edge.text(0.1, 0.95, edge_info,
                transform=ax_edge.transAxes,
                fontsize=11,
                verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.9))
    
    # ========================================================================
    # 4. GRAPH-LEVEL FEATURES (bottom right)
    # ========================================================================
    
    ax_graph = fig.add_subplot(gs[2, 2])
    ax_graph.axis('off')
    
    graph_info = """
GRAPH FEATURES (per epoch)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Global Properties:
┌─────────────────────────────┐
│  Kuramoto Parameters:       │
│  • Mean (coherence): 0.12   │
│  • Std (metastability): 0.06│
│                              │
│  Synchronization Stats:     │
│  • Mean sync: 0.49          │
│  • Std sync:  0.19          │
│                              │
│  Topology:                   │
│  • Density: 1.0 (full)      │
│  • Avg degree: 23.0         │
│  • Max degree: 23           │
│  • Min degree: 23           │
└─────────────────────────────┘

Total: 8 global features
"""
    
    ax_graph.text(0.1, 0.95, graph_info,
                 transform=ax_graph.transAxes,
                 fontsize=11,
                 verticalalignment='top',
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
    
    # ========================================================================
    # 5. GRAPH STRUCTURE DIAGRAM (bottom left)
    # ========================================================================
    
    ax_struct = fig.add_subplot(gs[2, 0:2])
    ax_struct.axis('off')
    
    # Create schematic diagram
    y_pos = 0.8
    x_start = 0.05
    
    # Title
    ax_struct.text(0.5, 0.95, 'Complete Graph Structure',
                  transform=ax_struct.transAxes,
                  fontsize=14, fontweight='bold',
                  ha='center')
    
    # Input section
    ax_struct.add_patch(FancyBboxPatch(
        (0.05, 0.6), 0.15, 0.25,
        boxstyle="round,pad=0.01",
        facecolor='wheat', edgecolor='black', linewidth=2,
        transform=ax_struct.transAxes
    ))
    ax_struct.text(0.125, 0.725, 'INPUT\nEEG\nEpoch',
                  transform=ax_struct.transAxes,
                  fontsize=10, ha='center', va='center', fontweight='bold')
    
    # Arrow
    ax_struct.annotate('', xy=(0.25, 0.725), xytext=(0.21, 0.725),
                      arrowprops=dict(arrowstyle='->', lw=2, color='black'),
                      transform=ax_struct.transAxes)
    
    # Graph box
    ax_struct.add_patch(FancyBboxPatch(
        (0.25, 0.55), 0.50, 0.35,
        boxstyle="round,pad=0.01",
        facecolor='lightblue', edgecolor='darkblue', linewidth=3,
        transform=ax_struct.transAxes
    ))
    
    # Graph components
    ax_struct.text(0.50, 0.82, '⚫ 24 NODES (electrodes)',
                  transform=ax_struct.transAxes,
                  fontsize=11, ha='center', fontweight='bold')
    ax_struct.text(0.50, 0.75, '8 features each',
                  transform=ax_struct.transAxes,
                  fontsize=9, ha='center', style='italic')
    
    ax_struct.text(0.50, 0.68, '━━ 552 EDGES (connections)',
                  transform=ax_struct.transAxes,
                  fontsize=11, ha='center', fontweight='bold')
    ax_struct.text(0.50, 0.61, '1 feature each (sync value)',
                  transform=ax_struct.transAxes,
                  fontsize=9, ha='center', style='italic')
    
    # Arrow
    ax_struct.annotate('', xy=(0.81, 0.725), xytext=(0.76, 0.725),
                      arrowprops=dict(arrowstyle='->', lw=2, color='black'),
                      transform=ax_struct.transAxes)
    
    # Output section
    ax_struct.add_patch(FancyBboxPatch(
        (0.81, 0.6), 0.15, 0.25,
        boxstyle="round,pad=0.01",
        facecolor='lightgreen', edgecolor='darkgreen', linewidth=2,
        transform=ax_struct.transAxes
    ))
    ax_struct.text(0.885, 0.725, 'OUTPUT\nDMT\nEC\nEO',
                  transform=ax_struct.transAxes,
                  fontsize=10, ha='center', va='center', fontweight='bold')
    
    # Add graph features box at bottom
    ax_struct.text(0.50, 0.45, '+ 8 GLOBAL FEATURES (Kuramoto, topology)',
                  transform=ax_struct.transAxes,
                  fontsize=10, ha='center', fontweight='bold',
                  bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Add statistics
    ax_struct.text(0.50, 0.25, 'Dataset Statistics',
                  transform=ax_struct.transAxes,
                  fontsize=12, ha='center', fontweight='bold')
    
    stats_text = """
    Total Graphs: 65,305
    • DMT: 25,830 (40%)  • EC: 21,185 (32%)  • EO: 18,290 (28%)
    
    Per Graph:
    • Nodes: 24  • Edges: 552  • Node features: 192 (24×8)  • Edge features: 552 (552×1)
    """
    
    ax_struct.text(0.50, 0.10, stats_text,
                  transform=ax_struct.transAxes,
                  fontsize=10, ha='center', va='center',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ========================================================================
    # SAVE
    # ========================================================================
    
    plt.suptitle('EEG Graph Neural Network: Data Structure Explanation',
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Visualization saved to: {save_path}")


def main():
    """Main function."""
    
    print("="*80)
    print("Creating EEG Graph Visualization")
    print("="*80)
    
    output_dir = Path('/media/storage_hdd/dmt_fz/machine_learning/clf/output/analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = output_dir / 'graph_structure_explanation.png'
    
    create_example_visualization(save_path)
    
    print("\nVisualization complete!")
    print(f"Open: {save_path}")
    print("="*80)


if __name__ == '__main__':
    main()

