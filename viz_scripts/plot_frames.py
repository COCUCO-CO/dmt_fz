"""
Frame generation functions for static visualizations.
Includes basic modes (all, eeg, stc) and advanced visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import colorsys

# Import shared resources from plot_utils
from plot_utils import (
    plot_mat, plot_graph, plot_kuramoto_order, plot_osc,
    node_colors, label_names, stc_coords_2d,
    ch_names, mapping, eeg_coords_2d,
    stc_corte, eeg_corte, ch_order
)

# Import global state (these are set by load_subject_data)
import plot_utils


# Quality presets: (dpi, figsize_scale, jpeg_quality)
QUALITY_PRESETS = {
    "high": {"dpi": 96, "scale": 1.0, "optimize": False},      # ~1MB
    "medium": {"dpi": 72, "scale": 0.8, "optimize": True},     # ~500KB
    "low": {"dpi": 48, "scale": 0.7, "optimize": True},        # ~250KB
}


def get_savefig_kwargs(fmt, quality="high"):
    """Get savefig parameters based on format and quality."""
    preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["high"])
    kwargs = {"dpi": preset["dpi"], "format": fmt}
    
    if fmt == "png" and preset["optimize"]:
        kwargs["pil_kwargs"] = {"optimize": True, "compress_level": 9}
    
    return kwargs, preset["scale"]


def plot_eeg_only(epoch, fmt="png", quality="high", samples=[0, 400, 799]):
    """Generate frames showing only EEG channels."""
    save_kwargs, scale = get_savefig_kwargs(fmt, quality)
    eeg_syncro_mat = plot_utils.subject_syncro["syncros_eeg"][plot_utils.band][0][epoch]
    
    for sample in samples:
        fig = plt.figure(figsize=(24*scale, 10*scale))
        grid_size = (5, 3)
        
        ax0 = plt.subplot2grid(grid_size, (0, 0), colspan=3)
        plot_kuramoto_order(plot_utils.eeg_kuramoto_mat, ax0, epoch, "r", plot_utils.rej)
        ax0.set_title(f"EEG Kuramoto Order Parameter - Epoch {epoch+1}", fontsize=16*scale)
        
        ax1 = plt.subplot2grid(grid_size, (1, 0), rowspan=4)
        elec_colors = ["k" for _ in ch_names]
        plot_mat(eeg_syncro_mat, ax1, ch_order=ch_order, corte=eeg_corte, cmap="Reds",
                 threshold=plot_utils.eeg_threshold, signal_names=ch_names, fontsize=10*scale, colors=elec_colors)
        ax1.set_title("EEG Connectivity Matrix", fontsize=14*scale)
        
        ax2 = plt.subplot2grid(grid_size, (1, 1), rowspan=4)
        edgecolors = ["k" for _ in ch_names]
        linewidths = [1 for _ in ch_names]
        plot_graph(eeg_syncro_mat, ax2, threshold=plot_utils.eeg_threshold, pos=eeg_coords_2d,
                   edge_cmap="Reds", edge_width=2, relabel=mapping, node_color="r",
                   node_size=750*scale, font_size=12*scale, edgecolors=edgecolors, linewidths=linewidths)
        ax2.set_title("EEG Network Graph", fontsize=14*scale)
        
        ax3 = plt.subplot2grid(grid_size, (1, 2), projection="polar", rowspan=4)
        plot_osc(plot_utils.eeg_phase_mat, plot_utils.eeg_kuramoto_mat[epoch], ax3, 
                 sample=sample, color="r", jitter=True, labels=ch_names)
        
        plt.suptitle(f"SUJETO: {plot_utils.subject} | CONDICIÓN: {plot_utils.cond} | BANDA: {plot_utils.band} | Época {epoch+1:03d}/{plot_utils.num_epochs:03d} | EEG", 
                     fontsize=20*scale, fontweight='bold', y=0.97)
        plt.subplots_adjust(left=0.05, top=0.93, right=0.95, bottom=0.05, hspace=0.35, wspace=0.35)
        
        filename = f"{plot_utils.subject}_{plot_utils.cond}_{plot_utils.band}_eeg_{(1000+epoch)*1000+sample}"
        output_file = plot_utils.frames_path / f"{filename}.{fmt}"
        print(f"[EEG] {output_file}")
        fig.savefig(output_file, **save_kwargs)
        plt.close()


def plot_stc_only(epoch, fmt="png", quality="high", samples=[0, 400, 799]):
    """Generate frames showing only source space (brain network)."""
    save_kwargs, scale = get_savefig_kwargs(fmt, quality)
    stc_syncro_mat = plot_utils.subject_syncro["syncros_stc"][plot_utils.band][0][epoch]
    
    for sample in samples:
        fig = plt.figure(figsize=(24*scale, 10*scale))
        grid_size = (5, 3)
        
        ax0 = plt.subplot2grid(grid_size, (0, 0), colspan=3)
        plot_kuramoto_order(plot_utils.stc_kuramoto_mat, ax0, epoch, "b", plot_utils.rej)
        ax0.set_title(f"Source Space Kuramoto Order Parameter - Epoch {epoch+1}", fontsize=16*scale)
        
        ax1 = plt.subplot2grid(grid_size, (1, 0), rowspan=4)
        plot_mat(stc_syncro_mat, ax1, corte=stc_corte, xticks=False, colors=node_colors,
                 cmap="Dynamic", threshold=plot_utils.stc_threshold, signal_names=label_names)
        ax1.set_title("Source Connectivity Matrix (100 parcels)", fontsize=14*scale)
        
        ax2 = plt.subplot2grid(grid_size, (1, 1), rowspan=4)
        plot_graph(stc_syncro_mat, ax2, threshold=plot_utils.stc_threshold, pos=stc_coords_2d,
                   edge_cmap="Dynamic", font_size=8*scale, node_color=node_colors, edge_width=2)
        ax2.set_title("Source Network Graph", fontsize=14*scale)
        
        ax3 = plt.subplot2grid(grid_size, (1, 2), projection="polar", rowspan=4)
        plot_osc(plot_utils.stc_phase_mat, plot_utils.stc_kuramoto_mat[epoch], ax3, 
                 sample=sample, color=node_colors, jitter=True, labels=True)
        
        plt.suptitle(f"SUJETO: {plot_utils.subject} | CONDICIÓN: {plot_utils.cond} | BANDA: {plot_utils.band} | Época {epoch+1:03d}/{plot_utils.num_epochs:03d} | STC", 
                     fontsize=20*scale, fontweight='bold', y=0.97)
        plt.subplots_adjust(left=0.05, top=0.93, right=0.95, bottom=0.05, hspace=0.35, wspace=0.35)
        
        filename = f"{plot_utils.subject}_{plot_utils.cond}_{plot_utils.band}_stc_{(1000+epoch)*1000+sample}"
        output_file = plot_utils.frames_path / f"{filename}.{fmt}"
        print(f"[STC] {output_file}")
        fig.savefig(output_file, **save_kwargs)
        plt.close()


def plot_all(epoch, fmt="png", quality="high", samples=[0, 400, 799]):
    """Generate frames showing both EEG and STC (original full view)."""
    save_kwargs, scale = get_savefig_kwargs(fmt, quality)
    eeg_syncro_mat = plot_utils.subject_syncro["syncros_eeg"][plot_utils.band][0][epoch]
    stc_syncro_mat = plot_utils.subject_syncro["syncros_stc"][plot_utils.band][0][epoch]
    
    for sample in samples:
        fig = plt.figure(figsize=(24*scale, 20*scale))
        grid_size = (10, 3)
        
        ax0 = plt.subplot2grid(grid_size, (0, 0), colspan=3)
        plot_kuramoto_order(plot_utils.eeg_kuramoto_mat, ax0, epoch, "r", plot_utils.rej)
        
        ax1 = plt.subplot2grid(grid_size, (1, 0), rowspan=4)
        elec_colors = ["k" for _ in ch_names]
        plot_mat(eeg_syncro_mat, ax1, ch_order=ch_order, corte=eeg_corte, cmap="Reds",
                 threshold=plot_utils.eeg_threshold, signal_names=ch_names, fontsize=10*scale, colors=elec_colors)
        
        ax2 = plt.subplot2grid(grid_size, (1, 1), rowspan=4)
        edgecolors = ["k" for _ in ch_names]
        linewidths = [1 for _ in ch_names]
        plot_graph(eeg_syncro_mat, ax2, threshold=plot_utils.eeg_threshold, pos=eeg_coords_2d,
                   edge_cmap="Reds", edge_width=2, relabel=mapping, node_color="r",
                   node_size=750*scale, font_size=12*scale, edgecolors=edgecolors, linewidths=linewidths)
        
        ax3 = plt.subplot2grid(grid_size, (1, 2), projection="polar", rowspan=4)
        plot_osc(plot_utils.eeg_phase_mat, plot_utils.eeg_kuramoto_mat[epoch], ax3, 
                 sample=sample, color="r", jitter=True, labels=ch_names)
        
        ax4 = plt.subplot2grid(grid_size, (5, 0), colspan=3)
        plot_kuramoto_order(plot_utils.stc_kuramoto_mat, ax4, epoch, "b", plot_utils.rej)
        
        ax5 = plt.subplot2grid(grid_size, (6, 0), rowspan=4)
        plot_mat(stc_syncro_mat, ax5, corte=stc_corte, xticks=False, colors=node_colors,
                 cmap="Dynamic", threshold=plot_utils.stc_threshold, signal_names=label_names)
        
        ax6 = plt.subplot2grid(grid_size, (6, 1), rowspan=4)
        plot_graph(stc_syncro_mat, ax6, threshold=plot_utils.stc_threshold, pos=stc_coords_2d,
                   edge_cmap="Dynamic", font_size=8*scale, node_color=node_colors, edge_width=2)
        
        ax7 = plt.subplot2grid(grid_size, (6, 2), projection="polar", rowspan=4)
        plot_osc(plot_utils.stc_phase_mat, plot_utils.stc_kuramoto_mat[epoch], ax7, 
                 sample=sample, color=node_colors, jitter=True, labels=True)
        
        plt.suptitle(f"SUJETO: {plot_utils.subject} | CONDICIÓN: {plot_utils.cond} | BANDA: {plot_utils.band} | Época {epoch+1:03d}/{plot_utils.num_epochs:03d}", 
                     fontsize=20*scale, fontweight='bold', y=0.98)
        plt.subplots_adjust(left=0.05, top=0.96, right=0.95, bottom=0.05, hspace=0.35, wspace=0.35)
        
        filename = f"{plot_utils.subject}_{plot_utils.cond}_{plot_utils.band}_{(1000+epoch)*1000+sample}"
        output_file = plot_utils.frames_path / f"{filename}.{fmt}"
        print(str(output_file))
        
        fig.savefig(output_file, **save_kwargs)
        plt.close()


# ============================================================================
# ADVANCED VISUALIZATION HELPERS
# ============================================================================

def get_node_colors_by_sync(syncro_mat, base_colors, sync_values):
    """Color nodes based on their synchronization level."""
    n_nodes = min(len(sync_values), len(base_colors))
    colors = []
    sync_norm = (sync_values[:n_nodes] - sync_values[:n_nodes].min()) / (sync_values[:n_nodes].max() - sync_values[:n_nodes].min() + 1e-8)
    
    for i in range(n_nodes):
        try:
            base = base_colors[i][:3] if isinstance(base_colors[i], (list, tuple, np.ndarray)) else (0.5, 0.5, 0.8)
        except (IndexError, TypeError):
            base = (0.5, 0.5, 0.8)
        h, s, v = colorsys.rgb_to_hsv(base[0], base[1], base[2])
        s_new = 0.3 + 0.7 * sync_norm[i]
        v_new = 0.4 + 0.6 * sync_norm[i]
        r, g, b = colorsys.hsv_to_rgb(h, s_new, v_new)
        colors.append((r, g, b, 0.7 + 0.3 * sync_norm[i]))
    return colors


def draw_gradient_edges(ax, pos, syncro_mat, threshold_pct=80, max_edges=150, show_all=True):
    """Draw edges with gradient colors based on connection strength.
    
    Parameters:
    -----------
    show_all : bool
        If True, show ALL connections with weak ones attenuated (transparent).
        If False, use threshold to filter out weak connections.
    """
    pos = np.asarray(pos)
    n_nodes = min(syncro_mat.shape[0], len(pos))
    syncro_mat = syncro_mat[:n_nodes, :n_nodes]
    
    edges, weights = [], []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if syncro_mat[i, j] > 0:  # Include all non-zero connections
                edges.append((i, j))
                weights.append(syncro_mat[i, j])
    if not edges:
        return
    
    weights = np.array(weights)
    weights_norm = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
    
    # Sort by weight (weakest first so strongest are drawn on top)
    sorted_idx = np.argsort(weights)
    edges = [edges[i] for i in sorted_idx]
    weights_norm = weights_norm[sorted_idx]
    
    segments, colors, linewidths, alphas = [], [], [], []
    cmap = plt.cm.plasma
    for (i, j), w in zip(edges, weights_norm):
        segments.append([pos[i], pos[j]])
        colors.append(cmap(w))
        # Linewidth: thin for weak, thick for strong
        linewidths.append(0.3 + 2.5 * w)
        # Alpha: very transparent for weak (0.01), opaque for strong (0.8)
        alphas.append(0.01 + 0.79 * (w ** 3))  # Cubed to attenuate weak connections more aggressively
    
    if segments:
        # Draw each segment individually to allow per-edge alpha
        for seg, col, lw, alpha in zip(segments, colors, linewidths, alphas):
            ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], 
                   color=col, linewidth=lw, alpha=alpha, zorder=1)


def draw_glow_nodes(ax, pos, colors, sizes, sync_values):
    """Draw nodes with glow effect for highly synchronized ones."""
    pos = np.asarray(pos)
    n_nodes = min(len(pos), len(colors), len(sizes), len(sync_values))
    sync_norm = (sync_values[:n_nodes] - sync_values[:n_nodes].min()) / (sync_values[:n_nodes].max() - sync_values[:n_nodes].min() + 1e-8)
    
    for glow_layer in [3, 2, 1]:
        for i in range(n_nodes):
            if sync_norm[i] > 0.7:
                ax.scatter(pos[i, 0], pos[i, 1], s=sizes[i] * (1 + glow_layer * 0.3), 
                          c=[colors[i][:3]], alpha=0.1 / glow_layer, edgecolors='none', zorder=2)
    ax.scatter(pos[:n_nodes, 0], pos[:n_nodes, 1], s=sizes[:n_nodes], c=colors[:n_nodes], 
              edgecolors='black', linewidths=1.0, zorder=3)


def draw_phase_ring(ax, phases, colors, kuramoto_val, show_legend=True):
    """Draw oscillators on a ring with smooth gradient background."""
    n_osc = len(phases)
    theta_bg = np.linspace(0, 2 * np.pi, 100)
    ax.fill_between(theta_bg, 0.85, 1.15, alpha=0.1, color='gray')
    
    circle = plt.Circle((0, 0), 1.0, fill=False, color='#888888', 
                        linewidth=1.5, linestyle='--', transform=ax.transData._b)
    ax.add_patch(circle)
    
    mean_phase = np.angle(np.exp(1j * phases).mean())
    r_order = np.abs(np.exp(1j * phases).mean())
    
    ax.annotate('', xy=(mean_phase, r_order), xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', color='#CC9900', lw=4), zorder=10)
    
    phase_diff = np.abs(np.angle(np.exp(1j * (phases - mean_phase))))
    coherence = 1 - phase_diff / np.pi
    radii = 1 + np.random.uniform(-0.08, 0.08, n_osc)
    sizes = 300 + 400 * coherence
    
    for i in range(n_osc):
        color = colors[i] if i < len(colors) else 'blue'
        ax.scatter(phases[i], radii[i], s=sizes[i], c=[color], 
                  alpha=0.6 + 0.4 * coherence[i], edgecolors='black', linewidths=0.5, zorder=5)
    
    ax.set_title(f"Phase Coherence: r = {r_order:.2f}", fontsize=13, fontweight='bold', color='#006600', pad=10)
    ax.set_ylim(0, 1.4)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['polar'].set_visible(False)
    
    if show_legend:
        ax.text(0.5, -0.08, '● = Oscillators  |  → = Mean phase (length = r)', 
               transform=ax.transAxes, ha='center', va='top', fontsize=9, color='#666666', style='italic')


def plot_advanced(epoch, fmt="png", quality="high", only_points=True, samples=[0, 200, 400, 600, 799]):
    """Generate high-quality frames with modern visualization techniques.
    
    Parameters:
    -----------
    only_points : bool
        If True, show only scatter points in timeline (no connecting lines)
    """
    save_kwargs, scale = get_savefig_kwargs(fmt, quality)
    save_kwargs['facecolor'] = 'white'
    save_kwargs['edgecolor'] = 'none'
    # Advanced uses higher base DPI
    if quality == "high":
        save_kwargs['dpi'] = 120
    elif quality == "medium":
        save_kwargs['dpi'] = 90
    else:
        save_kwargs['dpi'] = 60
    
    eeg_syncro_mat = plot_utils.subject_syncro["syncros_eeg"][plot_utils.band][0][epoch].copy()
    stc_syncro_mat = plot_utils.subject_syncro["syncros_stc"][plot_utils.band][0][epoch].copy()
    
    # Calculate coherence (mean) and metastability (std) for EEG and STC (only valid epochs)
    eeg_kuramoto_all = np.asarray(plot_utils.eeg_kuramoto_mat).mean(axis=1)
    stc_kuramoto_all = np.asarray(plot_utils.stc_kuramoto_mat).mean(axis=1)
    valid_indices = [i for i in range(len(eeg_kuramoto_all)) if i not in plot_utils.rej]
    eeg_coherence = np.mean(eeg_kuramoto_all[valid_indices])  # Global mean (coherence)
    eeg_metastability = np.std(eeg_kuramoto_all[valid_indices])  # Global std (metastability)
    stc_coherence = np.mean(stc_kuramoto_all[valid_indices])
    stc_metastability = np.std(stc_kuramoto_all[valid_indices])
    
    for sample in samples:
        fig = plt.figure(figsize=(24*scale, 22*scale), facecolor='white')
        fig.patch.set_facecolor('white')
        grid_size = (10, 3)
        
        # EEG Timeline
        ax_eeg_timeline = plt.subplot2grid(grid_size, (0, 0), colspan=3)
        ax_eeg_timeline.set_facecolor('white')
        
        full_epochs_count = len(plot_utils.eeg_kuramoto_mat)  # 210 epochs total
        x_timeline = np.arange(full_epochs_count) + 0.5
        eeg_kuramoto_means = np.asarray(plot_utils.eeg_kuramoto_mat).mean(axis=1)
        y_eeg = np.array([np.nan if i in plot_utils.rej else eeg_kuramoto_means[i] for i in range(full_epochs_count)])
        valid_mask = ~np.isnan(y_eeg)
        
        # Metastability band (std interval) - draw first so it's behind points
        ax_eeg_timeline.fill_between(
            [0, full_epochs_count], 
            eeg_coherence - eeg_metastability, 
            eeg_coherence + eeg_metastability,
            color='#3366cc', alpha=0.15, zorder=1, label=f'Metastability (σ={eeg_metastability:.3f})'
        )
        
        # Coherence line (mean) - dashed
        ax_eeg_timeline.axhline(y=eeg_coherence, color='#3366cc', linewidth=2, linestyle='--', 
                                zorder=2, label=f'Coherence (μ={eeg_coherence:.3f})')
        
        # Bad epochs as filled gray spans (draw before points)
        for i in plot_utils.rej:
            ax_eeg_timeline.axvspan(i, i + 1, color='grey', alpha=0.2, linewidth=0, zorder=0)
        
        # Only scatter points (no connecting lines) - 40% smaller
        ax_eeg_timeline.scatter(x_timeline[valid_mask], y_eeg[valid_mask], s=12, color='#cc0000', zorder=5)
        
        right_epochs = np.argwhere(valid_mask).flatten()
        if epoch < len(right_epochs):
            ax_eeg_timeline.axvline(x=right_epochs[epoch] + 0.5, color='black', linewidth=2, linestyle='--')
        
        ax_eeg_timeline.set_xlim(0, full_epochs_count)
        # Calculate Y limits to use full vertical space with small margin
        y_eeg_valid = y_eeg[valid_mask]
        eeg_y_min = min(y_eeg_valid.min(), eeg_coherence - eeg_metastability)
        eeg_y_max = max(y_eeg_valid.max(), eeg_coherence + eeg_metastability)
        eeg_y_margin = (eeg_y_max - eeg_y_min) * 0.05
        ax_eeg_timeline.set_ylim(eeg_y_min - eeg_y_margin, eeg_y_max + eeg_y_margin)
        ax_eeg_timeline.set_ylabel('Kuramoto Order', color='black', fontsize=10)
        ax_eeg_timeline.tick_params(colors='black', labelsize=8)
        ax_eeg_timeline.set_title('EEG Synchronization', color='black', fontsize=12, fontweight='bold')
        ax_eeg_timeline.legend(loc='upper right', fontsize=8, framealpha=0.9)
        for spine in ax_eeg_timeline.spines.values():
            spine.set_color('#cccccc')
        
        # EEG Matrix
        ax_eeg_mat = plt.subplot2grid(grid_size, (1, 0), rowspan=4)
        ax_eeg_mat.set_facecolor('white')
        plot_mat(eeg_syncro_mat, ax_eeg_mat, ch_order=ch_order, corte=eeg_corte, cmap="Reds",
                 threshold=plot_utils.eeg_threshold, signal_names=ch_names, fontsize=8, colors=["black"]*len(ch_names))
        ax_eeg_mat.set_title('EEG Connectivity Matrix', color='black', fontsize=11, fontweight='bold')
        
        # EEG Graph
        ax_eeg_graph = plt.subplot2grid(grid_size, (1, 1), rowspan=4)
        ax_eeg_graph.set_facecolor('white')
        eeg_pos_array = np.array(list(eeg_coords_2d.values())) if isinstance(eeg_coords_2d, dict) else np.asarray(eeg_coords_2d)
        n_eeg = min(len(ch_names), len(eeg_pos_array), eeg_syncro_mat.shape[0])
        eeg_syncro_limited = eeg_syncro_mat[:n_eeg, :n_eeg]
        eeg_sync_values = eeg_syncro_limited.mean(axis=1)
        eeg_node_colors = get_node_colors_by_sync(eeg_syncro_limited, [(1.0, 0.42, 0.42)] * n_eeg, eeg_sync_values)
        # Uniform node sizes for EEG graph
        eeg_sizes = np.full(n_eeg, 500)  # All nodes same size
        
        draw_gradient_edges(ax_eeg_graph, eeg_pos_array[:n_eeg], eeg_syncro_limited, threshold_pct=plot_utils.eeg_threshold)
        draw_glow_nodes(ax_eeg_graph, eeg_pos_array[:n_eeg], eeg_node_colors, eeg_sizes, eeg_sync_values)
        for i in range(n_eeg):
            ax_eeg_graph.annotate(ch_names[i], eeg_pos_array[i], ha='center', va='center', fontsize=7, color='black', fontweight='bold', zorder=10)
        ax_eeg_graph.set_xlim(eeg_pos_array[:n_eeg, 0].min() - 0.02, eeg_pos_array[:n_eeg, 0].max() + 0.02)
        ax_eeg_graph.set_ylim(eeg_pos_array[:n_eeg, 1].min() - 0.02, eeg_pos_array[:n_eeg, 1].max() + 0.02)
        ax_eeg_graph.set_aspect('equal')
        ax_eeg_graph.axis('off')
        ax_eeg_graph.set_title('EEG Network', color='black', fontsize=11, fontweight='bold')
        
        # EEG Phase Ring
        ax_eeg_phase = plt.subplot2grid(grid_size, (1, 2), rowspan=4, projection='polar')
        ax_eeg_phase.set_facecolor('white')
        eeg_phases_current = plot_utils.eeg_phase_mat[:n_eeg, sample]
        draw_phase_ring(ax_eeg_phase, eeg_phases_current, eeg_node_colors, 
                       plot_utils.eeg_kuramoto_mat[epoch][sample] if sample < len(plot_utils.eeg_kuramoto_mat[epoch]) else 0)
        
        # STC Timeline
        ax_stc_timeline = plt.subplot2grid(grid_size, (5, 0), colspan=3)
        ax_stc_timeline.set_facecolor('white')
        stc_kuramoto_means = np.asarray(plot_utils.stc_kuramoto_mat).mean(axis=1)
        y_stc = np.array([np.nan if i in plot_utils.rej else stc_kuramoto_means[i] for i in range(full_epochs_count)])
        
        # Metastability band (std interval) - draw first so it's behind points
        ax_stc_timeline.fill_between(
            [0, full_epochs_count], 
            stc_coherence - stc_metastability, 
            stc_coherence + stc_metastability,
            color='#3366cc', alpha=0.15, zorder=1, label=f'Metastability (σ={stc_metastability:.3f})'
        )
        
        # Coherence line (mean) - dashed
        ax_stc_timeline.axhline(y=stc_coherence, color='#3366cc', linewidth=2, linestyle='--', 
                                zorder=2, label=f'Coherence (μ={stc_coherence:.3f})')
        
        # Bad epochs as filled gray spans (draw before points)
        for i in plot_utils.rej:
            ax_stc_timeline.axvspan(i, i + 1, color='grey', alpha=0.2, linewidth=0, zorder=0)
        
        # Only scatter points (no connecting lines) - 40% smaller
        ax_stc_timeline.scatter(x_timeline[valid_mask], y_stc[valid_mask], s=12, color='#009688', zorder=5)
        
        if epoch < len(right_epochs):
            ax_stc_timeline.axvline(x=right_epochs[epoch] + 0.5, color='black', linewidth=2, linestyle='--')
        ax_stc_timeline.set_xlim(0, full_epochs_count)
        # Calculate Y limits to use full vertical space with small margin
        y_stc_valid = y_stc[valid_mask]
        stc_y_min = min(y_stc_valid.min(), stc_coherence - stc_metastability)
        stc_y_max = max(y_stc_valid.max(), stc_coherence + stc_metastability)
        stc_y_margin = (stc_y_max - stc_y_min) * 0.05
        ax_stc_timeline.set_ylim(stc_y_min - stc_y_margin, stc_y_max + stc_y_margin)
        ax_stc_timeline.set_ylabel('Kuramoto Order', color='black', fontsize=10)
        ax_stc_timeline.tick_params(colors='black', labelsize=8)
        ax_stc_timeline.set_title('Source Space Synchronization', color='black', fontsize=12, fontweight='bold')
        ax_stc_timeline.legend(loc='upper right', fontsize=8, framealpha=0.9)
        for spine in ax_stc_timeline.spines.values():
            spine.set_color('#cccccc')
        
        # STC Matrix
        ax_stc_mat = plt.subplot2grid(grid_size, (6, 0), rowspan=4)
        ax_stc_mat.set_facecolor('white')
        plot_mat(stc_syncro_mat, ax_stc_mat, corte=stc_corte, xticks=False, colors=node_colors,
                 cmap="Dynamic", threshold=plot_utils.stc_threshold, signal_names=label_names, fontsize=5)
        ax_stc_mat.set_title('Source Connectivity Matrix', color='black', fontsize=11, fontweight='bold')
        
        # STC Graph
        ax_stc_graph = plt.subplot2grid(grid_size, (6, 1), rowspan=4)
        ax_stc_graph.set_facecolor('white')
        n_stc = min(len(node_colors), len(stc_coords_2d), stc_syncro_mat.shape[0])
        stc_syncro_limited = stc_syncro_mat[:n_stc, :n_stc]
        stc_sync_values = stc_syncro_limited.mean(axis=1)
        stc_node_colors_dynamic = get_node_colors_by_sync(stc_syncro_limited, node_colors[:n_stc], stc_sync_values)
        # Uniform node sizes for STC graph
        stc_sizes = np.full(n_stc, 120)  # All nodes same size
        draw_gradient_edges(ax_stc_graph, stc_coords_2d[:n_stc], stc_syncro_limited, threshold_pct=plot_utils.stc_threshold, max_edges=200)
        draw_glow_nodes(ax_stc_graph, stc_coords_2d[:n_stc], stc_node_colors_dynamic, stc_sizes, stc_sync_values)
        # Add node numbers like in original visualization
        for i in range(n_stc):
            ax_stc_graph.annotate(str(i), stc_coords_2d[i], ha='center', va='center', fontsize=5, color='black', fontweight='bold', zorder=10)
        ax_stc_graph.set_xlim(stc_coords_2d[:n_stc, 0].min() - 0.01, stc_coords_2d[:n_stc, 0].max() + 0.01)
        ax_stc_graph.set_ylim(stc_coords_2d[:n_stc, 1].min() - 0.01, stc_coords_2d[:n_stc, 1].max() + 0.01)
        ax_stc_graph.set_aspect('equal')
        ax_stc_graph.axis('off')
        ax_stc_graph.set_title(f'Source Network ({n_stc} parcels)', color='black', fontsize=11, fontweight='bold')
        
        # STC Phase Ring
        ax_stc_phase = plt.subplot2grid(grid_size, (6, 2), rowspan=4, projection='polar')
        ax_stc_phase.set_facecolor('white')
        stc_phases_current = plot_utils.stc_phase_mat[:n_stc, sample]
        draw_phase_ring(ax_stc_phase, stc_phases_current, stc_node_colors_dynamic,
                       plot_utils.stc_kuramoto_mat[epoch][sample] if sample < len(plot_utils.stc_kuramoto_mat[epoch]) else 0)
        
        # Colorbar for edge strength (plasma colormap)
        cbar_ax = fig.add_axes([0.35, 0.01, 0.30, 0.012])  # [left, bottom, width, height]
        cbar_ax.set_facecolor('white')
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        cbar_ax.imshow(gradient, aspect='auto', cmap='plasma', extent=[0, 1, 0, 1])
        cbar_ax.set_yticks([])
        cbar_ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        cbar_ax.set_xticklabels(['0.0', '0.25', '0.5', '0.75', '1.0'], fontsize=9, color='black')
        cbar_ax.set_xlabel('Connection Strength (PLV)', fontsize=10, color='black', labelpad=2)
        for spine in cbar_ax.spines.values():
            spine.set_color('#cccccc')
        
        fig.suptitle(f"🧠 {plot_utils.subject} | {plot_utils.cond} | {plot_utils.band} Band | Epoch {epoch+1:03d}/{plot_utils.num_epochs:03d}",
                     fontsize=18, fontweight='bold', color='black', y=0.99)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.96, bottom=0.04, hspace=0.35, wspace=0.25)
        
        filename = f"{plot_utils.subject}_{plot_utils.cond}_{plot_utils.band}_adv_{(1000+epoch)*1000+sample}"
        output_file = plot_utils.frames_path / f"{filename}.{fmt}"
        print(f"[ADVANCED] {output_file}")
        fig.savefig(output_file, **save_kwargs)
        plt.close()
