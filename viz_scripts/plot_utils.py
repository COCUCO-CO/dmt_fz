"""
Utility functions and shared resources for plotting.
Contains common plotting primitives, data loading, and configuration.
"""

import numpy as np
import networkx as nx
from random import uniform as rand

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Circle
from PIL import Image

import pickle
from pathlib import Path
import sys

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pymatreader import read_mat

# Add parent directory to path to import from pipeline
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
from paths import BASE_DIR, RESULTS_DIR, EEG_CLEAN_DIR as EEG_DIR, FRAMES_DIR, VISUALIZATIONS_DIR, ensure_dir

# Initialize scaler
scaler = MinMaxScaler()

# ============================================================================
# FILE I/O
# ============================================================================

def save_file(data, folder, file):
    folder = ensure_dir(folder)
    with open(Path(folder) / f"{file}.pkl", 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_file(file):
    with open(Path(file), 'rb') as handle:
        return pickle.load(handle)


# ============================================================================
# NETWORK COLOR MAPS
# ============================================================================

network_cmap = {
    'DAN': plt.cm.Greens,
    'DMN': plt.cm.Reds,
    'FPN': plt.cm.Oranges,
    'LN ': plt.cm.Greys,
    'SMN': plt.cm.Blues,
    'SVA': plt.cm.RdPu,
    'VN ': plt.cm.Purples
}


# ============================================================================
# LOAD STATIC DATA (rejected epochs, coordinates, etc.)
# ============================================================================

rejected_subjects = [2, 5, 8, 16, 23, 31]
subjects = np.array([x for x in range(35) if x not in rejected_subjects])

# Load rejected channels
bad_channels = read_mat(str(EEG_DIR / "channels_removed.mat"))
_subject = bad_channels["channels_removed"][0]
_channels = bad_channels["channels_removed"][1]
rejected_channels = {k[:6].replace("_", "-"): v for k, v in zip(_subject, _channels)}

# Load rejected epochs
bad_epochs = read_mat(str(EEG_DIR / "rejected_epochs.mat"))
_subject = bad_epochs["rejected_epochs"][0]
_epochs = bad_epochs["rejected_epochs"][1]
rejected_epochs = {k[:6].replace("_", "-"): v.tolist() for k, v in zip(_subject, _epochs)}

# Load coordinate data
[node_colors, label_names, label_names_short, stc_coords_3d,
 ch_names, mapping, eeg_coords_2d] = load_file(RESULTS_DIR / "extra.pkl")

stc_coords_2d = stc_coords_3d[:, :2]
label_network = [x[:6] for x in label_names]

# Calculate network boundaries for matrix visualization
stc_corte = []
for i in range(1, len(label_network)):
    if label_network[i] != label_network[i - 1]:
        stc_corte.append(i)

eeg_corte = [8, 12, 14, 16, 21]

ch_order = ['Fp1', 'AFz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'C3', 'Cz', 'C4', 'CPz',
            'T7', 'M1', 'T8', 'M2', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'O2']


# ============================================================================
# GLOBAL STATE (set by load_subject_data)
# ============================================================================

frames_path = None
band = None
cond = None
subject = None
rej = None
subject_syncro = None
subject_phases = None
threshold = 60
eeg_threshold = 60
stc_threshold = 80
num_epochs = None
eeg_phase_mat = None
stc_phase_mat = None
eeg_kuramoto_mat = None
stc_kuramoto_mat = None


def load_subject_data(subject_id, condition, frequency_band, mode="all"):
    """Load data for a specific subject, condition, and band."""
    global frames_path, band, cond, subject, rej
    global subject_syncro, subject_phases, num_epochs
    global eeg_phase_mat, stc_phase_mat, eeg_kuramoto_mat, stc_kuramoto_mat
    
    # Create mode-specific output folder with subject, condition, and band
    subfolder = f"{subject_id}_{condition}_{frequency_band}"
    frames_path = ensure_dir(VISUALIZATIONS_DIR / "plot" / mode / subfolder)
    band = frequency_band
    cond = condition
    subject = subject_id
    
    print(f"[OUTPUT] Saving to: {frames_path}")
    
    # Build key for rejected epochs lookup
    cond_key_map = {"DMT": "DM", "EC": "EC", "EO": "EO"}
    key = f"{subject}-{cond_key_map.get(cond, cond[:2])}"
    
    try:
        rej = [x - 1 for x in rejected_epochs[key]]
    except KeyError:
        print(f"[WARNING] No rejected epochs found for {key}, using empty list")
        rej = []
    
    cond_dir = RESULTS_DIR / cond
    subject_syncro = load_file(cond_dir / f"syncro-{subject}-{cond}.pkl")
    subject_phases = load_file(cond_dir / f"phases-{subject}-{cond}.pkl")
    
    num_epochs = len(subject_phases["phases_eeg"][band])
    
    eeg_phase_mat = subject_phases["phases_eeg"][band][0]
    stc_phase_mat = subject_phases["phases_stc"][band][0]
    eeg_kuramoto_mat = subject_syncro["kuramoto_eeg"][band][0]
    stc_kuramoto_mat = subject_syncro["kuramoto_stc"][band][0]
    
    print(f"[DATA] Loaded {subject}-{cond}, band={band}, epochs={num_epochs}, rejected={len(rej)}")


# ============================================================================
# BASIC PLOTTING FUNCTIONS
# ============================================================================

def plot_mat(syncro_mat, ax, signal_names, ch_order=None, corte=None, xticks=True, 
             colors="k", cmap="Blues", threshold=None, fontsize=5):
    """Plot synchronization matrix as heatmap."""
    
    if ch_order is not None:
        df_syncro = pd.DataFrame(syncro_mat, columns=signal_names, index=signal_names)
        df_syncro = df_syncro[ch_order]
        df_syncro = df_syncro.T[ch_order]
        syncro_mat = df_syncro.values
    else:
        ch_order = signal_names

    if threshold is not None:
        syncro_mat[syncro_mat < np.percentile(syncro_mat, threshold)] = 0

    size = syncro_mat.shape[0]
    if signal_names is not None and len(signal_names) == size:
        local_label_network = [str(name)[:6] for name in signal_names]
    else:
        local_label_network = label_network[:size]
        if len(local_label_network) < size:
            local_label_network += ["UNK"] * (size - len(local_label_network))

    if cmap == "Dynamic":
        display_mat = np.zeros((size, size, 4))
        for i in range(size):
            for j in range(size):
                if local_label_network[i][-3:] == local_label_network[j][-3:]:
                    cmap_local = network_cmap.get(local_label_network[i][-3:], plt.cm.Blues)
                elif local_label_network[i][:3] == local_label_network[j][:3]:
                    cmap_local = plt.cm.Blues
                else:
                    cmap_local = plt.cm.PuBu
                display_mat[i, j, :] = cmap_local(syncro_mat[i, j])
        ax.imshow(display_mat)
    else:
        ax.imshow(syncro_mat, cmap=cmap)

    ticks = np.arange(syncro_mat.shape[1])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    xticks_labels = (ticks if xticks == False else ch_order)
    if len(xticks_labels) == len(ticks):
        ax.set_xticklabels(xticks_labels, fontsize=fontsize)
    else:
        ax.set_xticklabels(ticks, fontsize=fontsize)

    if len(ch_order) == len(ticks):
        ax.set_yticklabels(ch_order, fontsize=fontsize)
    else:
        ax.set_yticklabels(ticks, fontsize=fontsize)

    for xtick, color in zip(ax.get_xticklabels(), colors):
        xtick.set_color(color)

    for ytick, color in zip(ax.get_yticklabels(), colors):
        ytick.set_color(color)

    if corte is not None:
        for cut in corte:
            ax.axhline(y=cut - 0.5, color="black", linestyle="--", linewidth=1)
            ax.axvline(x=cut - 0.5, color="black", linestyle="--", linewidth=1)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")


def plot_graph(syncro_mat, ax, pos, threshold=None, relabel=None,
               edge_cmap=plt.cm.Blues, edge_width=1,
               node_color="b", node_size=250, font_size=8,
               edgecolors="k", linewidths=1):
    """Plot synchronization graph with nodes and edges."""
    
    if threshold is not None:
        syncro_mat[syncro_mat < np.percentile(syncro_mat, threshold)] = 0

    if isinstance(pos, dict):
        pos_len = len(pos)
    else:
        pos_array = np.asarray(pos)
        pos_len = pos_array.shape[0]

    size = min(syncro_mat.shape[0], pos_len)
    syncro_mat = syncro_mat[:size, :size]

    local_label_network = label_network[:size]
    if len(local_label_network) < size:
        local_label_network += ["UNK"] * (size - len(local_label_network))

    if isinstance(node_color, (list, tuple, np.ndarray)):
        nc_array = np.asarray(node_color)
        if nc_array.shape[0] < size:
            pad = ["b"] * (size - nc_array.shape[0])
            node_color = list(nc_array) + pad
        else:
            node_color = list(nc_array[:size])

    if isinstance(pos, dict):
        pos_dict = {i: pos[i] for i in list(pos)[:size]}
    else:
        pos_array = np.asarray(pos)[:size]
        pos_dict = {i: tuple(pos_array[i]) for i in range(size)}

    syncro_graph = nx.from_numpy_array(syncro_mat)
    if relabel is not None:
        syncro_graph = nx.relabel_nodes(syncro_graph, {i: relabel.get(i, i) for i in range(size)})

    edge_dict = nx.get_edge_attributes(syncro_graph, 'weight')
    sorted_dict = dict(sorted(edge_dict.items(), key=lambda item: item[1]))
    edges, weights = zip(*sorted_dict.items())
    weights = scaler.fit_transform(np.asarray(weights).reshape(-1, 1))

    if edge_cmap == "Dynamic":
        colors = []
        for edge, weight in zip(edges, weights):
            if local_label_network[edge[0]][-3:] == local_label_network[edge[1]][-3:]:
                cmap_local = network_cmap.get(local_label_network[edge[0]][-3:], plt.cm.Blues)
            elif local_label_network[edge[0]][:3] == local_label_network[edge[1]][:3]:
                cmap_local = plt.cm.Blues
            else:
                cmap_local = plt.cm.PuBu
            colors.append(cmap_local(weight))
    else:
        colors = [mpl.colormaps[edge_cmap](x) for x in weights]

    nx.draw_networkx_nodes(syncro_graph, ax=ax, pos=pos_dict, node_size=node_size,
                           node_color=node_color, edgecolors=edgecolors)
    nx.draw_networkx_labels(syncro_graph, ax=ax, pos=pos_dict)
    nx.draw_networkx_edges(syncro_graph, ax=ax, pos=pos_dict, edgelist=edges,
                           edge_color=colors, width=edge_width)
    ax.axis('off')


def plot_kuramoto_order(kuramoto_epochs, ax, epoch, color, rej, shade_reject=True):
    """Plot Kuramoto order parameter timeline across epochs."""
    num_epochs = len(kuramoto_epochs)
    
    x_s = [x + 0.5 for x in range(num_epochs)]
    avr = np.asarray(kuramoto_epochs).mean(axis=1)
    y_s = np.array([np.nan if i in rej else avr[i] for i in range(num_epochs)])
    right_epochs = np.argwhere(~np.isnan(y_s))
    
    ax.plot(x_s, y_s, color=color)
    ax.scatter(x_s, y_s, s=10, color=color)
    plt.axvline(x=right_epochs[epoch], color="black")
    
    if shade_reject:
        for i in rej:
            plt.axvspan(i, i + 1, color='grey', alpha=0.2, linewidth=0)


def plot_osc(phase_mat, kuramoto_mat, ax, color="r", sample=500, labels=None, jitter=False):
    """Plot oscillators on polar plot showing phase distribution."""
    num_osc = phase_mat.shape[0]
    radii = [1 + rand(-0.1, 0.1) for _ in range(num_osc)] if jitter else [1 for _ in range(num_osc)]

    if labels is True:
        text = [str(x) for x in range(num_osc)]
    elif labels is None:
        text = ["" for _ in range(num_osc)]
    else:
        text = list(labels)
        if len(text) < num_osc:
            text += [""] * (num_osc - len(text))
        else:
            text = text[:num_osc]

    size = 750
    theta = phase_mat[:, sample]

    if not isinstance(color, (list, tuple, np.ndarray)):
        color_list = [color for _ in range(num_osc)]
    else:
        color_list = list(color)
        if len(color_list) < num_osc:
            color_list += ["r"] * (num_osc - len(color_list))
        else:
            color_list = color_list[:num_osc]

    for i in range(num_osc):
        current_color = color_list[i]
        ax.scatter(theta[i], radii[i], color=current_color, s=size, edgecolors="k", zorder=i + 1)
        if text[i]:
            ax.annotate(text[i], (theta[i], radii[i]), ha='center', va='center', 
                       color="black", size=12, zorder=i + 1)

    r_current = np.abs(np.exp(1j * theta).mean())
    title = str(r_current)[:5]
    ax.annotate('r = %s' % title, (0, 0), ha='center', va='center', color="black", size=24)

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylim(0.0, 1.3)
    ax.axis('off')

    circle = Circle((0.0, 0.0), 1.0, transform=ax.transData._b, edgecolor="grey", 
                    facecolor="white", zorder=0)
    ax.add_artist(circle)


def add_logo(fig, path, x_frac=0.5, y_frac=0.5, scale=1, alpha=1):
    """Add a logo image to the figure."""
    im = Image.open(path)
    fig.set_dpi(96)
    im.thumbnail((int(im.size[0] * scale), int(im.size[1] * scale)))
    img_x, img_y = im.size[0], im.size[1]
    x_offset = int((fig.bbox.xmax * x_frac - img_x / 2))
    y_offset = int((fig.bbox.ymax * y_frac - img_y / 2))
    fig.figimage(im, xo=x_offset, yo=y_offset, origin='upper', zorder=10, alpha=alpha)
    return fig

