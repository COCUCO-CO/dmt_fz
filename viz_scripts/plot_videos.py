"""
Video generation functions for dynamic visualizations.
Includes smooth video, simple network video, 3D brain video, and advanced video.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import colorsys

# Import shared resources from plot_utils
from plot_utils import (
    load_file, RESULTS_DIR,
    plot_mat, plot_graph, plot_kuramoto_order, plot_osc,
    node_colors, label_names, stc_coords_2d, stc_coords_3d,
    ch_names, mapping, eeg_coords_2d,
    stc_corte, eeg_corte, ch_order,
    rejected_epochs
)

# Import global state
import plot_utils


def generate_smooth_video(
    subject_id="S01",
    condition="DMT",
    band="Alpha",
    output_file="network_dynamics.mp4",
    fps=30,
    interpolation_factor=10,
    epoch_duration=2.0,
    threshold_eeg=60,
    threshold_stc=80,
    dpi=120,
    max_epochs=None,
):
    """Generate a smooth video of network synchronization dynamics."""
    print(f"[VIDEO] Generating smooth video for {subject_id}-{condition}, band={band}")
    print(f"[VIDEO] Settings: fps={fps}, interpolation={interpolation_factor}x, dpi={dpi}")
    
    cond_dir = RESULTS_DIR / condition
    key = f"{subject_id}-{condition[:2].upper()}"
    
    try:
        rej = [x - 1 for x in rejected_epochs[key]]
    except KeyError:
        rej = []
    
    subject_syncro = load_file(cond_dir / f"syncro-{subject_id}-{condition}.pkl")
    subject_phases = load_file(cond_dir / f"phases-{subject_id}-{condition}.pkl")
    
    print(f"[VIDEO] Loaded data, rejected epochs: {len(rej)}")
    
    # Keep FULL kuramoto data for timeline display
    eeg_kuramoto_full = subject_syncro["kuramoto_eeg"][band][0]
    stc_kuramoto_full = subject_syncro["kuramoto_stc"][band][0]
    
    eeg_phases = subject_phases["phases_eeg"][band]
    stc_phases = subject_phases["phases_stc"][band]
    eeg_kuramoto = subject_syncro["kuramoto_eeg"][band][0]
    stc_kuramoto = subject_syncro["kuramoto_stc"][band][0]
    eeg_syncros = subject_syncro["syncros_eeg"][band][0]
    stc_syncros = subject_syncro["syncros_stc"][band][0]
    
    num_epochs = len(eeg_phases)
    samples_per_epoch = eeg_phases[0].shape[1]
    
    if max_epochs is not None:
        num_epochs = min(num_epochs, max_epochs)
        eeg_phases = eeg_phases[:num_epochs]
        stc_phases = stc_phases[:num_epochs]
        eeg_kuramoto = eeg_kuramoto[:num_epochs]
        stc_kuramoto = stc_kuramoto[:num_epochs]
        eeg_syncros = eeg_syncros[:num_epochs]
        stc_syncros = stc_syncros[:num_epochs]
    
    print(f"[VIDEO] Epochs: {num_epochs}, samples per epoch: {samples_per_epoch}")
    
    def interpolate_data(data_list, factor=10):
        interpolated = []
        for epoch_data in data_list:
            if len(epoch_data.shape) == 1:
                x_orig = np.arange(len(epoch_data))
                x_new = np.linspace(0, len(epoch_data) - 1, len(epoch_data) * factor)
                f = interp1d(x_orig, epoch_data, kind='cubic', fill_value='extrapolate')
                interpolated.append(f(x_new))
            elif len(epoch_data.shape) == 2:
                interpolated_channels = []
                for channel in epoch_data:
                    x_orig = np.arange(len(channel))
                    x_new = np.linspace(0, len(channel) - 1, len(channel) * factor)
                    f = interp1d(x_orig, channel, kind='cubic', fill_value='extrapolate')
                    interpolated_channels.append(f(x_new))
                interpolated.append(np.array(interpolated_channels))
        return interpolated
    
    print(f"[VIDEO] Interpolating data ({interpolation_factor}x)...")
    eeg_phases_smooth = interpolate_data(eeg_phases, interpolation_factor)
    stc_phases_smooth = interpolate_data(stc_phases, interpolation_factor)
    eeg_kuramoto_smooth = interpolate_data(eeg_kuramoto, interpolation_factor)
    stc_kuramoto_smooth = interpolate_data(stc_kuramoto, interpolation_factor)
    
    frames_per_epoch = int(epoch_duration * fps)
    sample_step = (samples_per_epoch * interpolation_factor) // frames_per_epoch
    
    frame_list = []
    for epoch_idx in range(num_epochs):
        if epoch_idx in rej:
            continue
        for frame in range(frames_per_epoch):
            sample_idx = frame * sample_step
            if sample_idx < len(eeg_phases_smooth[epoch_idx][0]):
                frame_list.append((epoch_idx, sample_idx))
    
    total_frames = len(frame_list)
    print(f"[VIDEO] Total frames to render: {total_frames} (~{total_frames/fps:.1f} seconds)")
    
    fig = plt.figure(figsize=(24, 20))
    fig.patch.set_facecolor('white')
    grid_size = (10, 3)
    
    ax0 = plt.subplot2grid(grid_size, (0, 0), colspan=3)
    ax1 = plt.subplot2grid(grid_size, (1, 0), rowspan=4)
    ax2 = plt.subplot2grid(grid_size, (1, 1), rowspan=4)
    ax3 = plt.subplot2grid(grid_size, (1, 2), projection="polar", rowspan=4)
    ax4 = plt.subplot2grid(grid_size, (5, 0), colspan=3)
    ax5 = plt.subplot2grid(grid_size, (6, 0), rowspan=4)
    ax6 = plt.subplot2grid(grid_size, (6, 1), rowspan=4)
    ax7 = plt.subplot2grid(grid_size, (6, 2), projection="polar", rowspan=4)
    
    axes = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7]
    
    def update_frame(frame_num):
        if frame_num % 30 == 0:
            print(f"[VIDEO] Rendering frame {frame_num}/{total_frames} ({frame_num*100//total_frames}%)")
        
        epoch_idx, sample_idx = frame_list[frame_num]
        
        for ax in axes:
            ax.clear()
        
        eeg_phase_mat = eeg_phases_smooth[epoch_idx]
        stc_phase_mat = stc_phases_smooth[epoch_idx]
        eeg_kuramoto_mat = eeg_kuramoto_smooth[epoch_idx]
        stc_kuramoto_mat = stc_kuramoto_smooth[epoch_idx]
        eeg_syncro_mat = eeg_syncros[epoch_idx]
        stc_syncro_mat = stc_syncros[epoch_idx]
        
        plot_kuramoto_order(eeg_kuramoto_full, ax0, epoch_idx, "r", rej, shade_reject=True)
        ax0.set_title(f"EEG Kuramoto - Epoch {epoch_idx+1:03d}/{num_epochs:03d}", fontsize=16)
        
        elec_colors = ["k" for _ in ch_names]
        plot_mat(eeg_syncro_mat, ax1, ch_order=ch_order, corte=eeg_corte, cmap="Reds",
                 threshold=threshold_eeg, signal_names=ch_names, fontsize=10, colors=elec_colors)
        ax1.set_title("EEG Connectivity Matrix", fontsize=14)
        
        edgecolors = ["k" for _ in ch_names]
        linewidths = [1 for _ in ch_names]
        plot_graph(eeg_syncro_mat, ax2, threshold=threshold_eeg, pos=eeg_coords_2d,
                   edge_cmap="Reds", edge_width=2, relabel=mapping, node_color="r",
                   node_size=750, font_size=12, edgecolors=edgecolors, linewidths=linewidths)
        ax2.set_title("EEG Network Graph", fontsize=14)
        
        plot_osc(eeg_phase_mat, eeg_kuramoto_mat, ax3, sample=sample_idx, 
                 color="r", jitter=True, labels=ch_names)
        
        plot_kuramoto_order(stc_kuramoto_full, ax4, epoch_idx, "b", rej, shade_reject=True)
        ax4.set_title(f"STC Kuramoto - Epoch {epoch_idx+1:03d}/{num_epochs:03d}", fontsize=16)
        
        plot_mat(stc_syncro_mat, ax5, corte=stc_corte, xticks=False, colors=node_colors,
                 cmap="Dynamic", threshold=threshold_stc, signal_names=label_names)
        ax5.set_title("Source Connectivity Matrix (100 parcels)", fontsize=14)
        
        plot_graph(stc_syncro_mat, ax6, threshold=threshold_stc, pos=stc_coords_2d,
                   edge_cmap="Dynamic", font_size=8, node_color=node_colors, edge_width=2)
        ax6.set_title("Source Network Graph", fontsize=14)
        
        plot_osc(stc_phase_mat, stc_kuramoto_mat, ax7, sample=sample_idx, 
                 color=node_colors, jitter=True, labels=True)
        
        time_sec = sample_idx / (samples_per_epoch * interpolation_factor) * 2.0
        fig.suptitle(f"{subject_id} | {condition} | {band} | Epoch {epoch_idx+1:03d}/{num_epochs:03d}",
                     fontsize=20, fontweight='bold')
        
        return axes
    
    print(f"[VIDEO] Creating animation...")
    anim = animation.FuncAnimation(fig, update_frame, frames=total_frames, interval=1000/fps, blit=False, repeat=False)
    
    output_path = plot_utils.frames_path / output_file
    print(f"[VIDEO] Saving video to {output_path}...")
    
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, bitrate=5000, codec='h264')
    
    anim.save(str(output_path), writer=writer, dpi=dpi)
    plt.close(fig)
    
    print(f"[VIDEO] ✓ Video saved successfully!")
    print(f"[VIDEO] Duration: {total_frames/fps:.1f}s, Size: {output_path.stat().st_size / 1e6:.1f} MB")
    
    return output_path


def generate_simple_network_video(
    subject_id="S01",
    condition="DMT",
    band="Alpha",
    output_file="network_simple.mp4",
    fps=30,
    interpolation_factor=5,
    view="graph",
    dpi=120,
    threshold_stc=80,
    epoch_duration=2.0,
    max_epochs=None,
):
    """Generate a focused video showing only one aspect of the network."""
    print(f"[VIDEO] Generating {view} video for {subject_id}-{condition}, band={band}")
    
    cond_dir = RESULTS_DIR / condition
    key = f"{subject_id}-{condition[:2].upper()}"
    
    try:
        rej = [x - 1 for x in rejected_epochs[key]]
    except KeyError:
        rej = []
    
    subject_syncro = load_file(cond_dir / f"syncro-{subject_id}-{condition}.pkl")
    subject_phases = load_file(cond_dir / f"phases-{subject_id}-{condition}.pkl")
    
    stc_phases = subject_phases["phases_stc"][band]
    stc_kuramoto = subject_syncro["kuramoto_stc"][band][0]
    stc_syncros = subject_syncro["syncros_stc"][band][0]
    
    stc_phases = [stc_phases[i] for i in range(len(stc_phases)) if i not in rej]
    stc_kuramoto = [stc_kuramoto[i] for i in range(len(stc_kuramoto)) if i not in rej]
    stc_syncros = [stc_syncros[i] for i in range(len(stc_syncros)) if i not in rej]
    
    num_epochs = len(stc_phases)
    
    if max_epochs is not None:
        num_epochs = min(num_epochs, max_epochs)
        stc_phases = stc_phases[:num_epochs]
        stc_kuramoto = stc_kuramoto[:num_epochs]
        stc_syncros = stc_syncros[:num_epochs]
    
    samples_per_epoch = stc_phases[0].shape[1]
    print(f"[VIDEO] Valid epochs: {num_epochs}, samples: {samples_per_epoch}")
    
    def interpolate_phases(phases_list, factor):
        interpolated = []
        for phase_mat in phases_list:
            interp_mat = []
            for channel_phases in phase_mat:
                x = np.arange(len(channel_phases))
                x_new = np.linspace(0, len(channel_phases) - 1, len(channel_phases) * factor)
                f = interp1d(x, channel_phases, kind='cubic', fill_value='extrapolate')
                interp_mat.append(f(x_new))
            interpolated.append(np.array(interp_mat))
        return interpolated
    
    print(f"[VIDEO] Interpolating phases ({interpolation_factor}x)...")
    stc_phases_smooth = interpolate_phases(stc_phases, interpolation_factor)
    
    if view == "all":
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        ax_graph, ax_matrix, ax_osc = axes
        ax_osc = plt.subplot(1, 3, 3, projection='polar')
    elif view == "graph":
        fig, ax_graph = plt.subplots(figsize=(12, 12))
        ax_matrix, ax_osc = None, None
    elif view == "matrix":
        fig, ax_matrix = plt.subplots(figsize=(12, 12))
        ax_graph, ax_osc = None, None
    elif view == "oscillators":
        fig = plt.figure(figsize=(12, 12))
        ax_osc = plt.subplot(111, projection='polar')
        ax_graph, ax_matrix = None, None
    else:
        raise ValueError(f"Invalid view: {view}")
    
    fig.patch.set_facecolor('white')
    
    frames_per_epoch = int(epoch_duration * fps)
    frames_total = len(stc_phases_smooth) * frames_per_epoch
    
    print(f"[VIDEO] Total frames: {frames_total} (~{frames_total/fps:.1f}s)")
    
    def update(frame_num):
        if frame_num % 30 == 0:
            print(f"[VIDEO] Frame {frame_num}/{frames_total} ({frame_num*100//frames_total}%)")
        
        current_epoch = frame_num // frames_per_epoch
        frame_in_epoch = frame_num % frames_per_epoch
        
        total_samples_in_epoch = samples_per_epoch * interpolation_factor
        local_sample = int((frame_in_epoch / frames_per_epoch) * total_samples_in_epoch)
        local_sample = min(local_sample, total_samples_in_epoch - 1)
        
        phase_mat = stc_phases_smooth[current_epoch]
        syncro_mat = stc_syncros[current_epoch]
        kuramoto_epoch = stc_kuramoto[current_epoch]
        
        if ax_graph:
            ax_graph.clear()
        if ax_matrix:
            ax_matrix.clear()
        if ax_osc:
            ax_osc.clear()
        
        if ax_graph:
            plot_graph(syncro_mat, ax_graph, threshold=threshold_stc, pos=stc_coords_2d,
                       edge_cmap="Dynamic", font_size=8, node_color=node_colors, edge_width=2)
            r_value = np.abs(np.exp(1j * phase_mat[:, min(local_sample, phase_mat.shape[1]-1)]).mean())
            ax_graph.set_title(f"Network Synchronization (r = {r_value:.3f})", fontsize=16)
        
        if ax_matrix:
            plot_mat(syncro_mat, ax_matrix, corte=stc_corte, xticks=False, colors=node_colors,
                     cmap="Dynamic", threshold=threshold_stc, signal_names=label_names)
            ax_matrix.set_title("Connectivity Matrix (100 parcels)", fontsize=16)
        
        if ax_osc:
            safe_sample = min(local_sample, len(phase_mat[0]) - 1)
            plot_osc(phase_mat, kuramoto_epoch, ax_osc, sample=safe_sample,
                     color=node_colors, jitter=True, labels=False)
        
        fig.suptitle(f"{subject_id} | {condition} | {band} | Epoch {current_epoch+1:03d}/{len(stc_phases):03d}",
                     fontsize=20, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        return [ax_graph or ax_matrix or ax_osc]
    
    print(f"[VIDEO] Creating animation...")
    anim = animation.FuncAnimation(fig, update, frames=frames_total, interval=1000/fps, blit=False, repeat=False)
    
    output_path = plot_utils.frames_path / output_file
    print(f"[VIDEO] Saving to {output_path}...")
    
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, bitrate=8000, codec='h264')
    
    anim.save(str(output_path), writer=writer, dpi=dpi)
    plt.close(fig)
    
    file_size = output_path.stat().st_size / 1e6
    print(f"[VIDEO] ✓ Complete! Duration: {frames_total/fps:.1f}s, Size: {file_size:.1f} MB")
    
    return output_path


def generate_3d_brain_video(
    subject_id="S01",
    condition="DMT",
    band="Alpha",
    output_file="brain_3d.mp4",
    fps=30,
    interpolation_factor=10,
    dpi=120,
    threshold_stc=85,
    epoch_duration=2.0,
    rotate_camera=True,
    show_edges=True,
    edge_threshold=90,
    max_epochs=None,
):
    """Generate a 3D brain video showing parcels with synchronization-based coloring."""
    import matplotlib
    matplotlib.use('Agg')
    
    print(f"[VIDEO 3D] Generating 3D brain video for {subject_id}-{condition}, band={band}")
    
    cond_dir = RESULTS_DIR / condition
    key = f"{subject_id}-{condition[:2].upper()}"
    
    try:
        rej = [x - 1 for x in rejected_epochs[key]]
    except KeyError:
        rej = []
    
    subject_syncro = load_file(cond_dir / f"syncro-{subject_id}-{condition}.pkl")
    subject_phases = load_file(cond_dir / f"phases-{subject_id}-{condition}.pkl")
    
    stc_phases = subject_phases["phases_stc"][band]
    stc_kuramoto = subject_syncro["kuramoto_stc"][band][0]
    stc_syncros = subject_syncro["syncros_stc"][band][0]
    
    stc_phases = [stc_phases[i] for i in range(len(stc_phases)) if i not in rej]
    stc_kuramoto = [stc_kuramoto[i] for i in range(len(stc_kuramoto)) if i not in rej]
    stc_syncros = [stc_syncros[i] for i in range(len(stc_syncros)) if i not in rej]
    
    num_epochs = len(stc_phases)
    
    if max_epochs is not None:
        num_epochs = min(num_epochs, max_epochs)
        stc_phases = stc_phases[:num_epochs]
        stc_kuramoto = stc_kuramoto[:num_epochs]
        stc_syncros = stc_syncros[:num_epochs]
    
    samples_per_epoch = stc_phases[0].shape[1]
    print(f"[VIDEO 3D] Valid epochs: {num_epochs}, samples: {samples_per_epoch}")
    
    print(f"[VIDEO 3D] Interpolating phases ({interpolation_factor}x)...")
    stc_phases_smooth = []
    for phase_mat in stc_phases:
        x_old = np.arange(phase_mat.shape[1])
        x_new = np.linspace(0, phase_mat.shape[1] - 1, phase_mat.shape[1] * interpolation_factor)
        phase_smooth = np.zeros((phase_mat.shape[0], len(x_new)))
        for i in range(phase_mat.shape[0]):
            unwrapped = np.unwrap(phase_mat[i])
            f = interp1d(x_old, unwrapped, kind='cubic')
            phase_smooth[i] = np.angle(np.exp(1j * f(x_new)))
        stc_phases_smooth.append(phase_smooth)
    
    frames_per_epoch = int(epoch_duration * fps)
    frames_total = len(stc_phases_smooth) * frames_per_epoch
    
    print(f"[VIDEO 3D] Total frames: {frames_total} (~{frames_total/fps:.1f}s)")
    
    print(f"[VIDEO 3D] Pre-calculating frame data...")
    frame_data_cache = []
    num_parcels = len(stc_coords_3d)
    
    for frame_num in range(frames_total):
        current_epoch = frame_num // frames_per_epoch
        frame_in_epoch = frame_num % frames_per_epoch
        
        total_samples_in_epoch = samples_per_epoch * interpolation_factor
        local_sample = int((frame_in_epoch / frames_per_epoch) * total_samples_in_epoch)
        local_sample = min(local_sample, total_samples_in_epoch - 1)
        
        phase_mat = stc_phases_smooth[current_epoch][:num_parcels, :]
        syncro_mat = stc_syncros[current_epoch][:num_parcels, :num_parcels]
        
        node_sync = syncro_mat.mean(axis=1)
        node_sync_norm = (node_sync - node_sync.min()) / (node_sync.max() - node_sync.min() + 1e-8)
        
        alphas = 0.5 + 0.5 * node_sync_norm
        sizes = 50 + 150 * node_sync_norm
        
        colors_rgba = np.zeros((num_parcels, 4))
        for i in range(num_parcels):
            color = node_colors[i]
            brightness = 0.4 + 0.6 * node_sync_norm[i]
            colors_rgba[i] = [color[0] * brightness, color[1] * brightness, color[2] * brightness, alphas[i]]
        
        edge_colors_rgba = np.array([[0, 0, 0, a] for a in alphas])
        
        segments = []
        edge_colors = []
        if show_edges:
            threshold_val = np.percentile(syncro_mat, edge_threshold)
            strong_connections = np.where(syncro_mat > threshold_val)
            for i, j in zip(strong_connections[0], strong_connections[1]):
                if i < j and i < num_parcels and j < num_parcels:
                    segments.append([stc_coords_3d[i], stc_coords_3d[j]])
                    strength = (syncro_mat[i, j] - threshold_val) / (syncro_mat.max() - threshold_val + 1e-8)
                    edge_colors.append([0.2, 0.5, 0.8, 0.2 + 0.6 * strength])
        
        if rotate_camera:
            angle = (frame_num / frames_total) * 360
            elev = 10 + 5 * np.sin(np.radians(angle * 2))
            azim = 45 + angle
        else:
            elev, azim = 10, 45
        
        r_value = np.abs(np.exp(1j * phase_mat[:, min(local_sample, phase_mat.shape[1]-1)]).mean())
        time_sec = local_sample / (samples_per_epoch * interpolation_factor) * 2.0
        epoch_time = current_epoch * 2.0 + time_sec
        
        frame_data_cache.append({
            'sizes': sizes, 'colors': colors_rgba, 'edge_colors_node': edge_colors_rgba,
            'segments': segments, 'edge_colors': edge_colors,
            'elev': elev, 'azim': azim, 'r_value': r_value,
            'epoch': current_epoch, 'epoch_time': epoch_time, 'num_epochs': num_epochs
        })
        
        if frame_num % 100 == 0:
            print(f"[VIDEO 3D] Pre-calculated {frame_num}/{frames_total} frames...")
    
    print(f"[VIDEO 3D] Pre-calculation complete! Starting fast render...")
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    coords = stc_coords_3d[:num_parcels]
    
    scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                        c=node_colors[:num_parcels], s=100, alpha=0.8, edgecolors='k', linewidths=0.5)
    
    line_collection = None
    if show_edges:
        line_collection = Line3DCollection([], colors='gray', linewidths=1, alpha=0.3)
        ax.add_collection3d(line_collection)
    
    ax.set_xlim(coords[:, 0].min() - 0.01, coords[:, 0].max() + 0.01)
    ax.set_ylim(coords[:, 1].min() - 0.01, coords[:, 1].max() + 0.01)
    ax.set_zlim(coords[:, 2].min() - 0.01, coords[:, 2].max() + 0.01)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.view_init(elev=10, azim=45)
    
    def update(frame_num):
        if frame_num % 30 == 0:
            print(f"[VIDEO 3D] Rendering frame {frame_num}/{frames_total} ({frame_num*100//frames_total}%)")
        
        data = frame_data_cache[frame_num]
        
        scatter._sizes = data['sizes']
        scatter._facecolors = data['colors']
        scatter._edgecolors = data['edge_colors_node']
        
        if show_edges and line_collection is not None:
            line_collection.set_segments(data['segments'])
            line_collection.set_colors(data['edge_colors'])
        
        ax.view_init(elev=data['elev'], azim=data['azim'])
        
        ax.set_title(
            f"{subject_id} - {condition} - {band} Band\n"
            f"Epoch {data['epoch']+1}/{data['num_epochs']} | "
            f"Time: {data['epoch_time']:.2f}s | "
            f"Kuramoto Order: r = {data['r_value']:.3f}",
            fontsize=14, fontweight='bold', pad=20
        )
        
        return scatter, line_collection
    
    print("[VIDEO 3D] Creating animation...")
    anim = animation.FuncAnimation(fig, update, frames=frames_total, interval=1000/fps, blit=False)
    
    output_path = plot_utils.frames_path / output_file
    print(f"[VIDEO 3D] Saving to {output_path}...")
    
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, bitrate=10000, codec='h264')
    
    anim.save(str(output_path), writer=writer, dpi=dpi)
    plt.close(fig)
    
    file_size = output_path.stat().st_size / 1e6
    print(f"[VIDEO 3D] ✓ Complete! Duration: {frames_total/fps:.1f}s, Size: {file_size:.1f} MB")
    
    return output_path


def generate_advanced_video(
    subject_id="S01",
    condition="DMT",
    band="Alpha",
    output_file="advanced_network.mp4",
    fps=30,
    interpolation_factor=5,
    dpi=120,
    max_epochs=None,
    epoch_duration=2.0,
):
    """Generate a video using the advanced visualization style with full EEG + STC layout."""
    print(f"[VIDEO ADV] Generating advanced video for {subject_id}-{condition}, band={band}")
    
    cond_dir = RESULTS_DIR / condition
    key = f"{subject_id}-{condition[:2].upper()}"
    
    try:
        rej = [x - 1 for x in rejected_epochs[key]]
    except KeyError:
        rej = []
    
    subject_syncro_local = load_file(cond_dir / f"syncro-{subject_id}-{condition}.pkl")
    subject_phases_local = load_file(cond_dir / f"phases-{subject_id}-{condition}.pkl")
    
    # Load ALL kuramoto data (before filtering) for timeline
    eeg_kuramoto_all = subject_syncro_local["kuramoto_eeg"][band][0]
    stc_kuramoto_all = subject_syncro_local["kuramoto_stc"][band][0]
    full_epochs_count = len(rej) + len(stc_kuramoto_all)
    
    # Calculate timeline y values with NaN for rejected - EEG
    eeg_kuramoto_means = np.asarray(eeg_kuramoto_all).mean(axis=1).tolist()
    y_eeg_timeline = np.array([eeg_kuramoto_means.pop(0) if i not in rej else np.nan for i in range(full_epochs_count)])
    
    # Calculate timeline y values with NaN for rejected - STC
    stc_kuramoto_means = np.asarray(stc_kuramoto_all).mean(axis=1).tolist()
    y_stc_timeline = np.array([stc_kuramoto_means.pop(0) if i not in rej else np.nan for i in range(full_epochs_count)])
    
    x_timeline = np.arange(full_epochs_count) + 0.5
    valid_mask = ~np.isnan(y_stc_timeline)
    valid_epoch_indices = np.argwhere(valid_mask).flatten()
    
    # Filter data for animation - EEG
    eeg_phases = subject_phases_local["phases_eeg"][band]
    eeg_syncros = subject_syncro_local["syncros_eeg"][band][0]
    eeg_phases = [eeg_phases[i] for i in range(len(eeg_phases)) if i not in rej]
    eeg_syncros = [eeg_syncros[i] for i in range(len(eeg_syncros)) if i not in rej]
    
    # Filter data for animation - STC
    stc_phases = subject_phases_local["phases_stc"][band]
    stc_syncros = subject_syncro_local["syncros_stc"][band][0]
    stc_phases = [stc_phases[i] for i in range(len(stc_phases)) if i not in rej]
    stc_syncros = [stc_syncros[i] for i in range(len(stc_syncros)) if i not in rej]
    
    num_epochs_total = len(stc_phases)
    
    if max_epochs is not None:
        num_epochs_total = min(num_epochs_total, max_epochs)
        eeg_phases = eeg_phases[:num_epochs_total]
        eeg_syncros = eeg_syncros[:num_epochs_total]
        stc_phases = stc_phases[:num_epochs_total]
        stc_syncros = stc_syncros[:num_epochs_total]
    
    samples_per_epoch = stc_phases[0].shape[1]
    print(f"[VIDEO ADV] Epochs: {num_epochs_total}, samples: {samples_per_epoch}")
    
    print(f"[VIDEO ADV] Interpolating phases ({interpolation_factor}x)...")
    
    # Interpolate EEG phases
    eeg_phases_smooth = []
    for phase_mat in eeg_phases:
        x_old = np.arange(phase_mat.shape[1])
        x_new = np.linspace(0, phase_mat.shape[1] - 1, phase_mat.shape[1] * interpolation_factor)
        phase_smooth = np.zeros((phase_mat.shape[0], len(x_new)))
        for i in range(phase_mat.shape[0]):
            unwrapped = np.unwrap(phase_mat[i])
            f = interp1d(x_old, unwrapped, kind='cubic')
            phase_smooth[i] = np.angle(np.exp(1j * f(x_new)))
        eeg_phases_smooth.append(phase_smooth)
    
    # Interpolate STC phases
    stc_phases_smooth = []
    for phase_mat in stc_phases:
        x_old = np.arange(phase_mat.shape[1])
        x_new = np.linspace(0, phase_mat.shape[1] - 1, phase_mat.shape[1] * interpolation_factor)
        phase_smooth = np.zeros((phase_mat.shape[0], len(x_new)))
        for i in range(phase_mat.shape[0]):
            unwrapped = np.unwrap(phase_mat[i])
            f = interp1d(x_old, unwrapped, kind='cubic')
            phase_smooth[i] = np.angle(np.exp(1j * f(x_new)))
        stc_phases_smooth.append(phase_smooth)
    
    frames_per_epoch = int(epoch_duration * fps)
    frames_total = num_epochs_total * frames_per_epoch
    print(f"[VIDEO ADV] Total frames: {frames_total} (~{frames_total/fps:.1f}s)")
    
    n_eeg = min(len(ch_names), eeg_syncros[0].shape[0])
    n_stc = min(len(node_colors), len(stc_coords_2d), stc_syncros[0].shape[0])
    
    # Get EEG coordinates
    if isinstance(eeg_coords_2d, dict):
        eeg_pos_array = np.array(list(eeg_coords_2d.values()))
    else:
        eeg_pos_array = np.asarray(eeg_coords_2d)
    
    def get_colors_by_sync(sync_values, base_colors):
        n = min(len(sync_values), len(base_colors))
        sync_norm = (sync_values[:n] - sync_values[:n].min()) / (sync_values[:n].max() - sync_values[:n].min() + 1e-8)
        colors = []
        for i in range(n):
            try:
                base = base_colors[i][:3]
            except:
                base = (0.5, 0.5, 0.8)
            h, s, v = colorsys.rgb_to_hsv(base[0], base[1], base[2])
            s_new = 0.3 + 0.7 * sync_norm[i]
            v_new = 0.4 + 0.6 * sync_norm[i]
            r, g, b = colorsys.hsv_to_rgb(h, s_new, v_new)
            colors.append((r, g, b, 0.7 + 0.3 * sync_norm[i]))
        return colors
    
    # Create figure with full EEG + STC layout (10x3 grid like frames)
    fig = plt.figure(figsize=(24, 20), facecolor='white')
    
    def update(frame_num):
        if frame_num % 30 == 0:
            print(f"[VIDEO ADV] Frame {frame_num}/{frames_total} ({frame_num*100//frames_total}%)")
        
        fig.clear()
        
        current_epoch = frame_num // frames_per_epoch
        frame_in_epoch = frame_num % frames_per_epoch
        
        total_samples = samples_per_epoch * interpolation_factor
        local_sample = int((frame_in_epoch / frames_per_epoch) * total_samples)
        local_sample = min(local_sample, total_samples - 1)
        
        eeg_phase_mat = eeg_phases_smooth[current_epoch][:n_eeg, :]
        eeg_syncro_mat = eeg_syncros[current_epoch][:n_eeg, :n_eeg]
        stc_phase_mat = stc_phases_smooth[current_epoch][:n_stc, :]
        stc_syncro_mat = stc_syncros[current_epoch][:n_stc, :n_stc]
        
        grid_size = (10, 3)
        
        # ===== EEG TIMELINE =====
        ax_eeg_timeline = plt.subplot2grid(grid_size, (0, 0), colspan=3)
        ax_eeg_timeline.set_facecolor('white')
        # Line and scatter only (no fill)
        ax_eeg_timeline.plot(x_timeline, y_eeg_timeline, color='#cc0000', linewidth=1.5)
        ax_eeg_timeline.scatter(x_timeline[valid_mask], y_eeg_timeline[valid_mask], s=20, color='#cc0000', zorder=5)
        if current_epoch < len(valid_epoch_indices):
            ax_eeg_timeline.axvline(x=valid_epoch_indices[current_epoch] + 0.5, color='black', linewidth=2, linestyle='--')
        for i in rej:
            ax_eeg_timeline.axvline(x=i+0.5, color='gray', alpha=0.3, linewidth=1)
        ax_eeg_timeline.set_xlim(0, full_epochs_count)
        ax_eeg_timeline.set_ylim(0, None)
        ax_eeg_timeline.set_ylabel('Kuramoto Order', color='black', fontsize=10)
        ax_eeg_timeline.tick_params(colors='black', labelsize=8)
        ax_eeg_timeline.set_title('EEG Synchronization', color='black', fontsize=12, fontweight='bold')
        for spine in ax_eeg_timeline.spines.values():
            spine.set_color('#cccccc')
        
        # ===== EEG MATRIX =====
        ax_eeg_mat = plt.subplot2grid(grid_size, (1, 0), rowspan=4)
        ax_eeg_mat.set_facecolor('white')
        elec_colors = ["black" for _ in ch_names]
        plot_mat(eeg_syncro_mat.copy(), ax_eeg_mat, ch_order=ch_order, corte=eeg_corte, cmap="Reds",
                 threshold=60, signal_names=ch_names, fontsize=7, colors=elec_colors)
        ax_eeg_mat.set_title('EEG Connectivity Matrix', color='black', fontsize=10, fontweight='bold')
        
        # ===== EEG GRAPH =====
        ax_eeg_graph = plt.subplot2grid(grid_size, (1, 1), rowspan=4)
        ax_eeg_graph.set_facecolor('white')
        eeg_sync_values = eeg_syncro_mat.mean(axis=1)
        eeg_node_colors = get_colors_by_sync(eeg_sync_values, [(1.0, 0.42, 0.42)] * n_eeg)
        eeg_sizes = 150 + 400 * (eeg_sync_values - eeg_sync_values.min()) / (eeg_sync_values.max() - eeg_sync_values.min() + 1e-8)
        
        threshold_eeg = np.percentile(eeg_syncro_mat, 60)
        for i in range(n_eeg):
            for j in range(i+1, n_eeg):
                if eeg_syncro_mat[i,j] > threshold_eeg:
                    w = (eeg_syncro_mat[i,j] - threshold_eeg) / (eeg_syncro_mat.max() - threshold_eeg + 1e-8)
                    ax_eeg_graph.plot([eeg_pos_array[i,0], eeg_pos_array[j,0]], 
                                     [eeg_pos_array[i,1], eeg_pos_array[j,1]], 
                                     color=plt.cm.plasma(w), linewidth=0.5+2*w, alpha=0.6)
        
        ax_eeg_graph.scatter(eeg_pos_array[:n_eeg, 0], eeg_pos_array[:n_eeg, 1], s=eeg_sizes, c=eeg_node_colors, edgecolors='black', linewidths=1, zorder=5)
        for i in range(n_eeg):
            ax_eeg_graph.annotate(ch_names[i], eeg_pos_array[i], ha='center', va='center', fontsize=6, color='black', fontweight='bold', zorder=10)
        ax_eeg_graph.set_xlim(eeg_pos_array[:n_eeg, 0].min() - 0.02, eeg_pos_array[:n_eeg, 0].max() + 0.02)
        ax_eeg_graph.set_ylim(eeg_pos_array[:n_eeg, 1].min() - 0.02, eeg_pos_array[:n_eeg, 1].max() + 0.02)
        ax_eeg_graph.set_aspect('equal')
        ax_eeg_graph.axis('off')
        eeg_r = np.abs(np.exp(1j * eeg_phase_mat[:, local_sample]).mean())
        ax_eeg_graph.set_title(f'EEG Network (r={eeg_r:.2f})', color='black', fontsize=10, fontweight='bold')
        
        # ===== EEG PHASE =====
        ax_eeg_phase = plt.subplot2grid(grid_size, (1, 2), rowspan=4, projection='polar')
        ax_eeg_phase.set_facecolor('white')
        eeg_phases_current = eeg_phase_mat[:, local_sample]
        eeg_mean_phase = np.angle(np.exp(1j * eeg_phases_current).mean())
        ax_eeg_phase.fill_between(np.linspace(0, 2*np.pi, 100), 0.85, 1.15, alpha=0.1, color='gray')
        circle = plt.Circle((0, 0), 1.0, fill=False, color='#888888', linewidth=1.5, linestyle='--', transform=ax_eeg_phase.transData._b)
        ax_eeg_phase.add_patch(circle)
        ax_eeg_phase.annotate('', xy=(eeg_mean_phase, eeg_r), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='#CC9900', lw=4), zorder=10)
        phase_diff = np.abs(np.angle(np.exp(1j * (eeg_phases_current - eeg_mean_phase))))
        coherence = 1 - phase_diff / np.pi
        radii = 1 + np.random.uniform(-0.08, 0.08, n_eeg)
        for i in range(n_eeg):
            ax_eeg_phase.scatter(eeg_phases_current[i], radii[i], s=200+300*coherence[i], c=[eeg_node_colors[i]], alpha=0.6+0.4*coherence[i], edgecolors='black', linewidths=0.5, zorder=5)
        ax_eeg_phase.set_ylim(0, 1.4)
        ax_eeg_phase.set_yticks([])
        ax_eeg_phase.set_xticks([])
        ax_eeg_phase.spines['polar'].set_visible(False)
        ax_eeg_phase.set_title(f'Phase Coherence: r = {eeg_r:.2f}', color='#006600', fontsize=10, fontweight='bold')
        
        # ===== STC TIMELINE =====
        ax_stc_timeline = plt.subplot2grid(grid_size, (5, 0), colspan=3)
        ax_stc_timeline.set_facecolor('white')
        # Line and scatter only (no fill)
        ax_stc_timeline.plot(x_timeline, y_stc_timeline, color='#009688', linewidth=1.5)
        ax_stc_timeline.scatter(x_timeline[valid_mask], y_stc_timeline[valid_mask], s=20, color='#009688', zorder=5)
        if current_epoch < len(valid_epoch_indices):
            ax_stc_timeline.axvline(x=valid_epoch_indices[current_epoch] + 0.5, color='black', linewidth=2, linestyle='--')
        for i in rej:
            ax_stc_timeline.axvline(x=i+0.5, color='gray', alpha=0.3, linewidth=1)
        ax_stc_timeline.set_xlim(0, full_epochs_count)
        ax_stc_timeline.set_ylim(0, None)
        ax_stc_timeline.set_ylabel('Kuramoto Order', color='black', fontsize=10)
        ax_stc_timeline.tick_params(colors='black', labelsize=8)
        ax_stc_timeline.set_title('Source Space Synchronization', color='black', fontsize=12, fontweight='bold')
        for spine in ax_stc_timeline.spines.values():
            spine.set_color('#cccccc')
        
        # ===== STC MATRIX =====
        ax_stc_mat = plt.subplot2grid(grid_size, (6, 0), rowspan=4)
        ax_stc_mat.set_facecolor('white')
        plot_mat(stc_syncro_mat.copy(), ax_stc_mat, corte=stc_corte, xticks=False, colors=node_colors,
                 cmap="Dynamic", threshold=80, signal_names=label_names, fontsize=4)
        ax_stc_mat.set_title('Source Connectivity Matrix', color='black', fontsize=10, fontweight='bold')
        
        # ===== STC GRAPH =====
        ax_stc_graph = plt.subplot2grid(grid_size, (6, 1), rowspan=4)
        ax_stc_graph.set_facecolor('white')
        stc_sync_values = stc_syncro_mat.mean(axis=1)
        stc_node_colors_dyn = get_colors_by_sync(stc_sync_values, node_colors[:n_stc])
        stc_sizes = 50 + 150 * (stc_sync_values - stc_sync_values.min()) / (stc_sync_values.max() - stc_sync_values.min() + 1e-8)
        
        threshold_stc = np.percentile(stc_syncro_mat, 85)
        edges, weights = [], []
        for i in range(n_stc):
            for j in range(i+1, n_stc):
                if stc_syncro_mat[i,j] > threshold_stc:
                    edges.append((i,j))
                    weights.append(stc_syncro_mat[i,j])
        if edges:
            weights = np.array(weights)
            w_norm = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
            segments = [[stc_coords_2d[i], stc_coords_2d[j]] for i,j in edges]
            colors_edge = [plt.cm.plasma(w) for w in w_norm]
            lc = LineCollection(segments, colors=colors_edge, linewidths=0.5 + 2*w_norm, alpha=0.6)
            ax_stc_graph.add_collection(lc)
        
        ax_stc_graph.scatter(stc_coords_2d[:n_stc, 0], stc_coords_2d[:n_stc, 1], s=stc_sizes, c=stc_node_colors_dyn, edgecolors='black', linewidths=0.5, zorder=5)
        ax_stc_graph.set_xlim(stc_coords_2d[:n_stc, 0].min() - 0.01, stc_coords_2d[:n_stc, 0].max() + 0.01)
        ax_stc_graph.set_ylim(stc_coords_2d[:n_stc, 1].min() - 0.01, stc_coords_2d[:n_stc, 1].max() + 0.01)
        ax_stc_graph.set_aspect('equal')
        ax_stc_graph.axis('off')
        stc_r = np.abs(np.exp(1j * stc_phase_mat[:, local_sample]).mean())
        ax_stc_graph.set_title(f'Source Network ({n_stc} parcels)', color='black', fontsize=10, fontweight='bold')
        
        # ===== STC PHASE =====
        ax_stc_phase = plt.subplot2grid(grid_size, (6, 2), rowspan=4, projection='polar')
        ax_stc_phase.set_facecolor('white')
        stc_phases_current = stc_phase_mat[:, local_sample]
        stc_mean_phase = np.angle(np.exp(1j * stc_phases_current).mean())
        ax_stc_phase.fill_between(np.linspace(0, 2*np.pi, 100), 0.85, 1.15, alpha=0.1, color='gray')
        circle = plt.Circle((0, 0), 1.0, fill=False, color='#888888', linewidth=1.5, linestyle='--', transform=ax_stc_phase.transData._b)
        ax_stc_phase.add_patch(circle)
        ax_stc_phase.annotate('', xy=(stc_mean_phase, stc_r), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='#CC9900', lw=4), zorder=10)
        phase_diff = np.abs(np.angle(np.exp(1j * (stc_phases_current - stc_mean_phase))))
        coherence = 1 - phase_diff / np.pi
        radii = 1 + np.random.uniform(-0.08, 0.08, n_stc)
        for i in range(n_stc):
            ax_stc_phase.scatter(stc_phases_current[i], radii[i], s=100+200*coherence[i], c=[stc_node_colors_dyn[i]], alpha=0.6+0.4*coherence[i], edgecolors='black', linewidths=0.3, zorder=5)
        ax_stc_phase.set_ylim(0, 1.4)
        ax_stc_phase.set_yticks([])
        ax_stc_phase.set_xticks([])
        ax_stc_phase.spines['polar'].set_visible(False)
        ax_stc_phase.set_title(f'Phase Coherence: r = {stc_r:.2f}', color='#006600', fontsize=10, fontweight='bold')
        
        # ===== COLORBAR =====
        cbar_ax = fig.add_axes([0.35, 0.01, 0.30, 0.012])
        cbar_ax.set_facecolor('white')
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        cbar_ax.imshow(gradient, aspect='auto', cmap='plasma', extent=[0, 1, 0, 1])
        cbar_ax.set_yticks([])
        cbar_ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        cbar_ax.set_xticklabels(['0.0', '0.25', '0.5', '0.75', '1.0'], fontsize=9, color='black')
        cbar_ax.set_xlabel('Connection Strength (PLV)', fontsize=10, color='black', labelpad=2)
        for spine in cbar_ax.spines.values():
            spine.set_color('#cccccc')
        
        # Main title
        time_sec = (current_epoch * epoch_duration) + (frame_in_epoch / frames_per_epoch * epoch_duration)
        fig.suptitle(f"🧠 {subject_id} | {condition} | {band} Band | Epoch {current_epoch+1:03d}/{num_epochs_total:03d}",
                    fontsize=18, fontweight='bold', color='black', y=0.99)
        
        plt.subplots_adjust(left=0.05, right=0.95, top=0.96, bottom=0.03, hspace=0.35, wspace=0.25)
        
        return []
    
    print(f"[VIDEO ADV] Creating animation...")
    anim = animation.FuncAnimation(fig, update, frames=frames_total, interval=1000/fps, blit=False)
    
    output_path = plot_utils.frames_path / output_file
    print(f"[VIDEO ADV] Saving to {output_path}...")
    
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, bitrate=10000, codec='h264')
    
    anim.save(str(output_path), writer=writer, dpi=dpi)
    plt.close(fig)
    
    file_size = output_path.stat().st_size / 1e6
    print(f"[VIDEO ADV] ✓ Complete! Duration: {frames_total/fps:.1f}s, Size: {file_size:.1f} MB")
    
    return output_path

