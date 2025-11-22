import numpy as np

import networkx as nx

from random import uniform as rand

import matplotlib.pyplot as plt
import matplotlib as mpl
#cmap = mpl.colormaps['viridis']
from matplotlib.patches import Circle

from PIL import Image

try:
    from IPython import get_ipython
    ipy = get_ipython()
    if ipy is not None:
        ipy.run_line_magic('matplotlib', 'inline')
except ImportError:
    ipy = None

#from mne.viz import circular_layout
#from mne_connectivity.viz import plot_connectivity_circle

# import imageio
# import cv2
# import moviepy.editor as mp
# import glob

#%%

import pickle
from pathlib import Path
import sys

# Add parent directory to path to import from pipeline
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
from paths import BASE_DIR, RESULTS_DIR, EEG_CLEAN_DIR as EEG_DIR, FRAMES_DIR, VISUALIZATIONS_DIR, ensure_dir


def save_file(data, folder, file):
    folder = ensure_dir(folder)
    with open(Path(folder) / f"{file}.pkl", 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_file(file):
    with open(Path(file), 'rb') as handle:
        return pickle.load(handle)


#%%

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

#%%

import pandas as pd

network_cmap={'DAN':plt.cm.Greens,
              'DMN':plt.cm.Reds,
              'FPN':plt.cm.Oranges,
              'LN ':plt.cm.Greys,
              'SMN':plt.cm.Blues,
              'SVA':plt.cm.RdPu,
              'VN ':plt.cm.Purples}

def plot_mat(syncro_mat, ax, signal_names, ch_order=None, corte=None, xticks=True, colors="k", cmap="Blues", threshold=None, fontsize=5):

  if ch_order != None:
    df_syncro = pd.DataFrame(syncro_mat, columns=signal_names, index=signal_names)
    df_syncro = df_syncro[ch_order]
    df_syncro = df_syncro.T[ch_order]
    syncro_mat = df_syncro.values
  else:
    ch_order = signal_names

  #np.fill_diagonal(syncro_mat, 0)
  if threshold != None:
    syncro_mat[syncro_mat < np.percentile(syncro_mat,threshold)] = 0

  size = syncro_mat.shape[0]
  if signal_names is not None and len(signal_names) == size:
    local_label_network = [str(name)[:6] for name in signal_names]
  else:
    local_label_network = label_network[:size]
    if len(local_label_network) < size:
      local_label_network += ["UNK"] * (size - len(local_label_network))

  if cmap=="Dynamic":
    display_mat = np.zeros((size,size,4))
    for i in range(size):
      for j in range(size):
        # Same Network
        if local_label_network[i][-3:] == local_label_network[j][-3:]:
          cmap = network_cmap.get(local_label_network[i][-3:], plt.cm.Blues)
        # Same Hemisphere
        elif local_label_network[i][:3] == local_label_network[j][:3]:
          cmap = plt.cm.Blues
        # Different network and different hemispheres
        else:
          cmap = plt.cm.PuBu
        display_mat[i,j,:] = cmap(syncro_mat[i,j])
    ax.imshow(display_mat)
  else:
    ax.imshow(syncro_mat, cmap=cmap)

  ticks = np.arange(syncro_mat.shape[1])
  ax.set_xticks(ticks)
  ax.set_yticks(ticks)

  xticks = (ticks if xticks == False else ch_order)
  if len(xticks) == len(ticks):
      ax.set_xticklabels(xticks, fontsize=fontsize)
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
      
  if corte!=None:
      for cut in corte:
          ax.axhline(y=cut-0.5, color="black", linestyle="--", linewidth=1)
          ax.axvline(x=cut-0.5, color="black", linestyle="--", linewidth=1)

  plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

  #return syncro_mat


def plot_graph(syncro_mat, ax, pos, threshold=None, relabel=None,
               edge_cmap=plt.cm.Blues, edge_width=1,
               node_color="b",node_size=250, font_size=8,
               edgecolors="k", linewidths=1):

  # raw_syncro_graph = nx.from_numpy_array(syncro_mat)
  # if relabel != None:
  #   raw_syncro_graph = nx.relabel_nodes(raw_syncro_graph, relabel)
  # return raw_syncro_graph
  #np.fill_diagonal(syncro_mat, 0)
  
  if threshold != None:
    syncro_mat[syncro_mat < np.percentile(syncro_mat,threshold)] = 0

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
  if relabel != None:
    syncro_graph = nx.relabel_nodes(syncro_graph, {i: relabel.get(i, i) for i in range(size)})

  edge_dict = nx.get_edge_attributes(syncro_graph,'weight')
  sorted_dict = dict(sorted(edge_dict.items(), key=lambda item: item[1])) # = {k: v for k, v in sorted(edge_dict.items(), key=lambda item: item[1])}
  edges, weights = zip(*sorted_dict.items())
  weights = scaler.fit_transform(np.asarray(weights).reshape(-1,1))
  
  if edge_cmap == "Dynamic":
      colors = []
      for edge, weight in zip(edges, weights):
        # Same Network
        if local_label_network[edge[0]][-3:] == local_label_network[edge[1]][-3:]:
          cmap = network_cmap.get(local_label_network[edge[0]][-3:], plt.cm.Blues)
        # Same Hemisphere
        elif local_label_network[edge[0]][:3] == local_label_network[edge[1]][:3]:
          cmap = plt.cm.Blues
        # Different network and different hemispheres
        else:
          cmap = plt.cm.PuBu
        colors.append(cmap(weight))
  else:
      colors = [mpl.colormaps[edge_cmap](x) for x in weights]
    
  nx.draw_networkx_nodes(syncro_graph, ax=ax, pos=pos_dict, node_size=node_size,
                         node_color=node_color, edgecolors=edgecolors)
  
  nx.draw_networkx_labels(syncro_graph, ax=ax, pos=pos_dict)

  nx.draw_networkx_edges(syncro_graph, ax=ax, pos=pos_dict, edgelist=edges,
                         edge_color=colors, width=edge_width)
  
  ax.axis('off')

  # nx.draw(syncro_graph, ax=ax,
  #         edgelist=edges, edge_color=weights, edge_cmap=edge_cmap, width=edge_width,
  #         pos=pos, with_labels=True, font_size=font_size ,
  #         node_size=node_size, node_color=node_color,
  #         edgecolors=edgecolors, linewidths=linewidths) #, cmap=plt.cm.Reds, node_shape='o',




# def chord_plot(syncro_mat, ax, label_names, node_order, n_lines, node_colors, colormap="Blues"):
#     node_angles = circular_layout(label_names, node_order, start_pos=90)
#                                   #group_boundaries=[0, len(label_names) // 2])
                                  
#     plot_connectivity_circle(syncro_mat, label_names, facecolor='white', textcolor='black',
#                              linewidth=1.5, n_lines=n_lines, colormap=colormap,
#                              colorbar=False, #colorbar_size=0.9, vmin=0, vmax=1, fontsize_colorbar=18,
#                              #fontsize_names=30, fontsize_title=20, title="",
#                              node_angles=node_angles, node_colors=node_colors, node_linewidth=0.5, node_width=4, ax=ax,show=False)


def plot_kuramoto_order(kuramoto_epochs, ax, epoch, color, rej, shade_reject=True):
    num_epochs = len(kuramoto_epochs)
    
    # kuramoto_epochs already contains ALL epochs (including rejected ones)
    x_s = [x+0.5 for x in range(num_epochs)]
    avr = np.asarray(kuramoto_epochs).mean(axis=1).tolist()
    
    # Plot all epochs (y_s directly maps to x_s by index)
    y_s = avr
    
    ax.plot(x_s, y_s, color=color)
    ax.scatter(x_s, y_s, s=10, color=color)
    # Mark current epoch with vertical line
    plt.axvline(x=epoch + 0.5, color="black")
    
    if shade_reject == True:
        for i in rej:
            plt.axvspan(i, i+1, color='grey', alpha=0.2, linewidth=0)
            

def plot_osc(phase_mat, kuramoto_mat, ax, color="r", sample=500, labels=None, jitter=False):
  num_osc = phase_mat.shape[0]
  radii = [1+rand(-0.1,0.1) for _ in range(num_osc)] if jitter else [1 for _ in range(num_osc)]

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
      # Fixed: Removed kuramoto_mat[i] line - it mixed oscillator index with time index (no semantic meaning)
      ax.scatter(theta[i], radii[i], color=current_color, s=size, edgecolors="k", zorder=i+1)
      if text[i]:
          ax.annotate(text[i], (theta[i], radii[i]), ha='center', va='center', color="black", size=12, zorder=i+1)
  
  # Fixed: Calculate order parameter for current sample instead of indexing
  r_current = np.abs(np.exp(1j * theta).mean())
  title = str(r_current)[:5]
  ax.annotate('r = %s' %title, (0,0), ha='center', va='center', color="black", size=24)  
  
  ax.set_yticks([])
  ax.set_xticks([])
  ax.set_ylim(0.0, 1.3)
  ax.axis('off')
  #ax.spines['polar'].set_color('none')

  circle = Circle((0.0, 0.0), 1.0, transform=ax.transData._b, edgecolor="grey", facecolor="white", zorder=0) #alpha=0.5
  
  ax.add_artist(circle)


def add_logo(fig, path, x_frac=0.5, y_frac=0.5, scale=1, alpha=1):
    im = Image.open(path)
    fig.set_dpi(96)
    im.thumbnail((int(im.size[0] * scale), int(im.size[1] * scale)))#, Image.ANTIALIAS)
    img_x, img_y = im.size[0], im.size[1]
    x_offset = int((fig.bbox.xmax * x_frac - img_x/2))
    y_offset = int((fig.bbox.ymax * y_frac - img_y/2))
    fig.figimage(im, xo=x_offset, yo=y_offset, origin='upper', zorder=10, alpha=alpha)
    return fig


#%%

from pymatreader import read_mat


rejected_subjects = [2, 5, 8, 16, 23, 31]
subjects = np.array([x for x in range(35) if x not in rejected_subjects])

folder = EEG_DIR

bad_channels = read_mat(str(folder / "channels_removed.mat"))
subject = bad_channels["channels_removed"][0]
channels = bad_channels["channels_removed"][1]
rejected_channels = {k[:6].replace("_","-"): v for k, v in zip(subject, channels)}

bad_epochs = read_mat(str(folder / "rejected_epochs.mat"))
subject = bad_epochs["rejected_epochs"][0]
epochs = bad_epochs["rejected_epochs"][1]
rejected_epochs = {k[:6].replace("_","-"): v.tolist() for k, v in zip(subject, epochs)}


#%%

[node_colors, label_names, \
 label_names_short, stc_coords_3d, \
 ch_names, mapping, eeg_coords_2d] = load_file(RESULTS_DIR / "extra.pkl")

stc_coords_2d = stc_coords_3d[:,:2]

label_network = [x[:6] for x in label_names]

stc_corte = []
for i in range(1,len(label_network)):
  if label_network[i] != label_network[i-1]:
    stc_corte.append(i)

eeg_corte = [8,12,14,16,21]

ch_order = ['Fp1', 'AFz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'C3', 'Cz', 'C4', 'CPz',
            'T7', 'M1', 'T8', 'M2', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'O2']


        
#%%

# rejected_subjects = [2, 5, 8, 16, 23, 31]
# subjects = np.array([x for x in range(35) if x not in rejected_subjects])

#subj = str(subjects[subject]+1)
 #"S"+("0"+subj)[-2:]+"-DM"

frames_path = ensure_dir(VISUALIZATIONS_DIR / "plot")
band = "Alpha"
cond = "DMT"
subject = "S01"

key = subject+"-DM"
rej = [x-1 for x in rejected_epochs[key]]

cond_dir = RESULTS_DIR / cond
subject_syncro = load_file(cond_dir / f"syncro-{subject}-{cond}.pkl")
subject_phases = load_file(cond_dir / f"phases-{subject}-{cond}.pkl")

threshold = 60
eeg_threshold = 60
stc_threshold = 80
num_epochs = len(subject_phases["phases_eeg"][band])

eeg_phase_mat = subject_phases["phases_eeg"][band][0]
stc_phase_mat = subject_phases["phases_stc"][band][0]
eeg_kuramoto_mat = subject_syncro["kuramoto_eeg"][band][0]
stc_kuramoto_mat = subject_syncro["kuramoto_stc"][band][0]

def plot_eeg_only(epoch, samples=[0, 400, 799]):
    """Generate frames showing only EEG channels."""
    eeg_syncro_mat = subject_syncro["syncros_eeg"][band][0][epoch]
    
    for sample in samples:
        fig = plt.figure(figsize=(24,10))
        grid_size = (5,3)
        
        # EEG Kuramoto Timeline
        ax0 = plt.subplot2grid(grid_size, (0, 0), colspan=3)
        plot_kuramoto_order(eeg_kuramoto_mat, ax0, epoch, "r", rej)
        ax0.set_title(f"EEG Kuramoto Order Parameter - Epoch {epoch+1}", fontsize=16)
        
        # EEG Matrix
        ax1 = plt.subplot2grid(grid_size, (1, 0), rowspan=4)
        elec_colors = ["k" for _ in ch_names]
        plot_mat(eeg_syncro_mat, ax1, ch_order=ch_order, corte=eeg_corte, cmap="Reds",
                 threshold=eeg_threshold, signal_names=ch_names, fontsize=10, colors=elec_colors)
        ax1.set_title("EEG Connectivity Matrix", fontsize=14)
        
        # EEG Graph
        ax2 = plt.subplot2grid(grid_size, (1, 1), rowspan=4)
        edgecolors = ["k" for _ in ch_names]
        linewidths = [1 for _ in ch_names]
        plot_graph(eeg_syncro_mat, ax2, threshold=eeg_threshold, pos=eeg_coords_2d,
                   edge_cmap="Reds", edge_width=2, relabel=mapping, node_color="r",
                   node_size=750, font_size=12, edgecolors=edgecolors, linewidths=linewidths)
        ax2.set_title("EEG Network Graph", fontsize=14)
        
        # EEG Oscillators
        ax3 = plt.subplot2grid(grid_size, (1, 2), projection="polar", rowspan=4)
        plot_osc(eeg_phase_mat, eeg_kuramoto_mat[epoch], ax3, sample=sample, color="r", jitter=True, labels=ch_names)
        
        plt.suptitle(f"SUJETO: {subject} | CONDICIÓN: {cond} | BANDA: {band} | Época {epoch+1}/{num_epochs} | EEG Only", 
                     fontsize=20, fontweight='bold', y=0.97)
        plt.subplots_adjust(left=0.05, top=0.93, right=0.95, bottom=0.05, hspace=0.35, wspace=0.35)
        
        filename = f"eeg_{(1000+epoch)*1000+sample}"
        output_file = frames_path / f"{filename}.png"
        print(f"[EEG] {output_file}")
        fig.savefig(output_file, dpi=96)
        plt.close()


def plot_stc_only(epoch, samples=[0, 400, 799]):
    """Generate frames showing only source space (brain network)."""
    stc_syncro_mat = subject_syncro["syncros_stc"][band][0][epoch]
    
    for sample in samples:
        fig = plt.figure(figsize=(24,10))
        grid_size = (5,3)
        
        # STC Kuramoto Timeline
        ax0 = plt.subplot2grid(grid_size, (0, 0), colspan=3)
        plot_kuramoto_order(stc_kuramoto_mat, ax0, epoch, "b", rej)
        ax0.set_title(f"Source Space Kuramoto Order Parameter - Epoch {epoch+1}", fontsize=16)
        
        # STC Matrix
        ax1 = plt.subplot2grid(grid_size, (1, 0), rowspan=4)
        plot_mat(stc_syncro_mat, ax1, corte=stc_corte, xticks=False, colors=node_colors,
                 cmap="Dynamic", threshold=stc_threshold, signal_names=label_names)
        ax1.set_title("Source Connectivity Matrix (100 parcels)", fontsize=14)
        
        # STC Graph
        ax2 = plt.subplot2grid(grid_size, (1, 1), rowspan=4)
        plot_graph(stc_syncro_mat, ax2, threshold=stc_threshold, pos=stc_coords_2d,
                   edge_cmap="Dynamic", font_size=8, node_color=node_colors, edge_width=2)
        ax2.set_title("Source Network Graph", fontsize=14)
        
        # STC Oscillators
        ax3 = plt.subplot2grid(grid_size, (1, 2), projection="polar", rowspan=4)
        plot_osc(stc_phase_mat, stc_kuramoto_mat[epoch], ax3, sample=sample, color=node_colors, jitter=True, labels=True)
        
        plt.suptitle(f"SUJETO: {subject} | CONDICIÓN: {cond} | BANDA: {band} | Época {epoch+1}/{num_epochs} | Source Space", 
                     fontsize=20, fontweight='bold', y=0.97)
        plt.subplots_adjust(left=0.05, top=0.93, right=0.95, bottom=0.05, hspace=0.35, wspace=0.35)
        
        filename = f"stc_{(1000+epoch)*1000+sample}"
        output_file = frames_path / f"{filename}.png"
        print(f"[STC] {output_file}")
        fig.savefig(output_file, dpi=96)
        plt.close()


def plot_all(epoch, samples=[0, 400, 799]):
    """Generate frames showing both EEG and STC (original full view)."""
    eeg_syncro_mat = subject_syncro["syncros_eeg"][band][0][epoch]
    stc_syncro_mat = subject_syncro["syncros_stc"][band][0][epoch]
    
    for sample in samples:
    
        fig = plt.figure(figsize=(24,20))
        grid_size = (10,3)
        
        #---
        ax0 = plt.subplot2grid(grid_size, (0, 0), colspan=3)
    
        plot_kuramoto_order(eeg_kuramoto_mat, ax0, epoch, "r", rej)
        
        #---
        ax1 = plt.subplot2grid(grid_size, (1, 0), rowspan=4)
        
        # Fixed: elec_colors should use rejected_channels, not rejected_epochs (rej)
        # For now, use black for all channels (epochs ≠ channels)
        elec_colors = ["k" for _ in ch_names]
        
        plot_mat(eeg_syncro_mat, ax1, ch_order=ch_order, corte=eeg_corte, cmap="Reds",
                                  threshold=eeg_threshold, signal_names=ch_names, fontsize=10, colors=elec_colors)
        
        #---
        ax2 = plt.subplot2grid(grid_size, (1, 1), rowspan=4)
        
        # Fixed: edgecolors/linewidths should use rejected_channels, not rejected_epochs
        # For now, use uniform styling (epochs ≠ channels)
        edgecolors = ["k" for _ in ch_names]
        linewidths = [1 for _ in ch_names]
        
        plot_graph(eeg_syncro_mat, ax2, threshold=eeg_threshold, pos=eeg_coords_2d,
                   edge_cmap="Reds", edge_width=2, relabel=mapping, node_color="r",
                   node_size=750, font_size=12, edgecolors=edgecolors, linewidths=linewidths)
        
        #---
        ax3 = plt.subplot2grid(grid_size, (1, 2), projection="polar", rowspan=4)
        
        plot_osc(eeg_phase_mat, eeg_kuramoto_mat[epoch], ax3, sample=sample, color="r", jitter=True, labels=ch_names)
        
        #ch_order=['Fz','Fp1','F3','F7','CPz','C3','T7','M1','P7','P3','O1','Pz']+['POz','O2','P4','P8','M2','T8','C4','Cz','F8','F4','Fp2','AFz']
        #chord_plot(eeg_syncro_mat, ax3, ch_names, ch_order, n_lines=75, node_colors="r", colormap="Reds")
        
        #---
        ax4 = plt.subplot2grid(grid_size, (5, 0), colspan=3)
        
        plot_kuramoto_order(stc_kuramoto_mat, ax4, epoch, "b", rej)
        
        #---
        ax5 = plt.subplot2grid(grid_size, (6, 0), rowspan=4)
        
        plot_mat(stc_syncro_mat, ax5, corte=stc_corte, xticks=False, colors=node_colors,
                 cmap="Dynamic", threshold=stc_threshold, signal_names=label_names)
        
        #---
        ax6 = plt.subplot2grid(grid_size, (6, 1), rowspan=4)
        
        #fig = add_logo(fig, "C:\\Users\\xochipilli\\Desktop\\Cerebro.jpg", x_frac=0.518, y_frac=0.18, scale=0.36, alpha=0.1)
        plot_graph(stc_syncro_mat, ax6, threshold=stc_threshold, pos=stc_coords_2d,
                   edge_cmap="Dynamic", font_size=8, node_color=node_colors, edge_width=2)
        
        #---
        ax7 = plt.subplot2grid(grid_size, (6, 2), projection="polar", rowspan=4)
        
        plot_osc(stc_phase_mat, stc_kuramoto_mat[epoch], ax7, sample=sample, color=node_colors, jitter=True, labels=True)
        
        #chord_plot(stc_syncro_mat, ax7, label_names, node_order, n_lines=300, node_colors=node_colors, colormap="Blues")
        
        #---
        
        # Add clear title with subject, condition, band
        plt.suptitle(f"SUJETO: {subject} | CONDICIÓN: {cond} | BANDA: {band} | Época {epoch+1}/{num_epochs} | Sample {sample}", 
                     fontsize=20, fontweight='bold', y=0.98)
        
        #plt.tight_layout()
        plt.subplots_adjust(left = 0.05, top = 0.96, right = 0.95, bottom = 0.05, hspace = 0.35, wspace = 0.35)
    
        filename = str((1000+epoch)*1000+sample)
        output_file = frames_path / f"{filename}.png"
        print(str(output_file))
    
        fig.savefig(output_file, dpi=96)#bbox_inches='tight')
        plt.close()
        #plt.show()

# Generate frames for testing (uncomment to run)
# Adjust subject, condition, and band at line 365-376 before running
# for epoch in range(10):  # Generate first 10 epochs for testing
#     plot_all(epoch)

# To generate all epochs:
# for epoch in range(num_epochs):
#     plot_all(epoch)

#%%

# COMMENTED OUT - Old video generation method (saves frames to disk)
# Use generate_video.py with the new smooth video functions instead
# 
# import os
# import cv2
# import numpy as np
# from tqdm import tqdm
# 
# width = 2304
# height = 1920
# fc = cv2.VideoWriter_fourcc(*"mp4v")
# video = cv2.VideoWriter(str(frames_path / "output2.mp4"), fc, 30, (width, height))
# 
# for image in tqdm(os.listdir(frames_path)[:300]):
#     image = cv2.imread(str(frames_path / image))
#     #image = cv2.resize(image, (width/2, height/2))
#     video.write(image)
# 
# video.release()


#%%
# from multiprocessing import Pool

# if __name__ == '__main__':
#     with Pool(10) as p:
#         p.map(plot_all, range(210))


#%%
# ============================================================================
# MODERN VIDEO GENERATION WITH SMOOTH TRANSITIONS
# ============================================================================

import matplotlib.animation as animation
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

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
):
    """
    Generate a smooth video of network synchronization dynamics.
    
    Parameters
    ----------
    subject_id : str
        Subject identifier (e.g., "S01")
    condition : str
        Condition to visualize ("DMT", "EC", "EO")
    band : str
        Frequency band ("Delta", "Theta", "Alpha", "Beta", "Gamma")
    output_file : str or Path
        Output video filename
    fps : int
        Frames per second (30 recommended for smooth playback)
    interpolation_factor : int
        How many interpolated frames between actual samples (10 = 10x smoother)
    epoch_duration : float
        Duration in seconds to show each epoch
    threshold_eeg : float
        Percentile threshold for EEG connectivity visualization
    threshold_stc : float
        Percentile threshold for STC connectivity visualization
    dpi : int
        Resolution (120 recommended for good quality/file size balance)
        
    Returns
    -------
    Path
        Path to generated video file
    """
    print(f"[VIDEO] Generating smooth video for {subject_id}-{condition}, band={band}")
    print(f"[VIDEO] Settings: fps={fps}, interpolation={interpolation_factor}x, dpi={dpi}")
    
    # Load data
    cond_dir = RESULTS_DIR / condition
    key = f"{subject_id}-{condition[:2].upper()}"
    
    try:
        rej = [x - 1 for x in rejected_epochs[key]]
    except KeyError:
        rej = []
    
    subject_syncro = load_file(cond_dir / f"syncro-{subject_id}-{condition}.pkl")
    subject_phases = load_file(cond_dir / f"phases-{subject_id}-{condition}.pkl")
    
    print(f"[VIDEO] Loaded data, rejected epochs: {len(rej)}")
    
    # Extract data for the specified band
    eeg_phases = subject_phases["phases_eeg"][band]
    stc_phases = subject_phases["phases_stc"][band]
    eeg_kuramoto = subject_syncro["kuramoto_eeg"][band][0]
    stc_kuramoto = subject_syncro["kuramoto_stc"][band][0]
    eeg_syncros = subject_syncro["syncros_eeg"][band][0]
    stc_syncros = subject_syncro["syncros_stc"][band][0]
    
    num_epochs = len(eeg_phases)
    samples_per_epoch = eeg_phases[0].shape[1]
    
    print(f"[VIDEO] Epochs: {num_epochs}, samples per epoch: {samples_per_epoch}")
    
    # Create interpolated time series for smooth transitions
    def interpolate_data(data_list, factor=10):
        """Interpolate between samples for smooth animation."""
        interpolated = []
        for epoch_data in data_list:
            if len(epoch_data.shape) == 1:
                # 1D array (e.g., kuramoto values)
                x_orig = np.arange(len(epoch_data))
                x_new = np.linspace(0, len(epoch_data) - 1, len(epoch_data) * factor)
                f = interp1d(x_orig, epoch_data, kind='cubic', fill_value='extrapolate')
                interpolated.append(f(x_new))
            elif len(epoch_data.shape) == 2:
                # 2D array (e.g., phase matrix)
                interpolated_channels = []
                for channel in epoch_data:
                    x_orig = np.arange(len(channel))
                    x_new = np.linspace(0, len(channel) - 1, len(channel) * factor)
                    f = interp1d(x_orig, channel, kind='cubic', fill_value='extrapolate')
                    interpolated_channels.append(f(x_new))
                interpolated.append(np.array(interpolated_channels))
        return interpolated
    
    # Interpolate for smooth transitions
    print(f"[VIDEO] Interpolating data ({interpolation_factor}x)...")
    eeg_phases_smooth = interpolate_data(eeg_phases, interpolation_factor)
    stc_phases_smooth = interpolate_data(stc_phases, interpolation_factor)
    eeg_kuramoto_smooth = interpolate_data(eeg_kuramoto, interpolation_factor)
    stc_kuramoto_smooth = interpolate_data(stc_kuramoto, interpolation_factor)
    
    # Calculate total frames
    frames_per_epoch = int(epoch_duration * fps)
    sample_step = (samples_per_epoch * interpolation_factor) // frames_per_epoch
    
    # Build frame list (epoch, sample_index pairs)
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
    
    # Create figure
    fig = plt.figure(figsize=(24, 20))
    fig.patch.set_facecolor('white')
    grid_size = (10, 3)
    
    # Pre-create all axes
    ax0 = plt.subplot2grid(grid_size, (0, 0), colspan=3)  # EEG Kuramoto timeline
    ax1 = plt.subplot2grid(grid_size, (1, 0), rowspan=4)  # EEG matrix
    ax2 = plt.subplot2grid(grid_size, (1, 1), rowspan=4)  # EEG graph
    ax3 = plt.subplot2grid(grid_size, (1, 2), projection="polar", rowspan=4)  # EEG oscillators
    ax4 = plt.subplot2grid(grid_size, (5, 0), colspan=3)  # STC Kuramoto timeline
    ax5 = plt.subplot2grid(grid_size, (6, 0), rowspan=4)  # STC matrix
    ax6 = plt.subplot2grid(grid_size, (6, 1), rowspan=4)  # STC graph
    ax7 = plt.subplot2grid(grid_size, (6, 2), projection="polar", rowspan=4)  # STC oscillators
    
    axes = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7]
    
    def update_frame(frame_num):
        """Update function for each frame."""
        if frame_num % 30 == 0:
            print(f"[VIDEO] Rendering frame {frame_num}/{total_frames} ({frame_num*100//total_frames}%)")
        
        epoch_idx, sample_idx = frame_list[frame_num]
        
        # Clear all axes
        for ax in axes:
            ax.clear()
        
        # Get current epoch data
        eeg_phase_mat = eeg_phases_smooth[epoch_idx]
        stc_phase_mat = stc_phases_smooth[epoch_idx]
        eeg_kuramoto_mat = eeg_kuramoto_smooth[epoch_idx]
        stc_kuramoto_mat = stc_kuramoto_smooth[epoch_idx]
        eeg_syncro_mat = eeg_syncros[epoch_idx]
        stc_syncro_mat = stc_syncros[epoch_idx]
        
        # --- EEG Kuramoto Timeline (ax0) ---
        plot_kuramoto_order(eeg_kuramoto, ax0, epoch_idx, "r", rej, shade_reject=True)
        ax0.set_title(f"EEG Kuramoto - Epoch {epoch_idx+1}/{num_epochs}", fontsize=16)
        
        # --- EEG Matrix (ax1) ---
        # Fixed: Use uniform colors (epochs ≠ channels)
        elec_colors = ["k" for _ in ch_names]
        plot_mat(eeg_syncro_mat, ax1, ch_order=ch_order, corte=eeg_corte, cmap="Reds",
                 threshold=threshold_eeg, signal_names=ch_names, fontsize=10, colors=elec_colors)
        ax1.set_title("EEG Connectivity Matrix", fontsize=14)
        
        # --- EEG Graph (ax2) ---
        # Fixed: Use uniform styling (epochs ≠ channels)
        edgecolors = ["k" for _ in ch_names]
        linewidths = [1 for _ in ch_names]
        plot_graph(eeg_syncro_mat, ax2, threshold=threshold_eeg, pos=eeg_coords_2d,
                   edge_cmap="Reds", edge_width=2, relabel=mapping, node_color="r",
                   node_size=750, font_size=12, edgecolors=edgecolors, linewidths=linewidths)
        ax2.set_title("EEG Network Graph", fontsize=14)
        
        # --- EEG Oscillators (ax3) ---
        plot_osc(eeg_phase_mat, eeg_kuramoto_mat, ax3, sample=sample_idx, 
                 color="r", jitter=True, labels=ch_names)
        
        # --- STC Kuramoto Timeline (ax4) ---
        plot_kuramoto_order(stc_kuramoto, ax4, epoch_idx, "b", rej, shade_reject=True)
        ax4.set_title(f"STC Kuramoto - Epoch {epoch_idx+1}/{num_epochs}", fontsize=16)
        
        # --- STC Matrix (ax5) ---
        plot_mat(stc_syncro_mat, ax5, corte=stc_corte, xticks=False, colors=node_colors,
                 cmap="Dynamic", threshold=threshold_stc, signal_names=label_names)
        ax5.set_title("Source Connectivity Matrix (100 parcels)", fontsize=14)
        
        # --- STC Graph (ax6) ---
        plot_graph(stc_syncro_mat, ax6, threshold=threshold_stc, pos=stc_coords_2d,
                   edge_cmap="Dynamic", font_size=8, node_color=node_colors, edge_width=2)
        ax6.set_title("Source Network Graph", fontsize=14)
        
        # --- STC Oscillators (ax7) ---
        plot_osc(stc_phase_mat, stc_kuramoto_mat, ax7, sample=sample_idx, 
                 color=node_colors, jitter=True, labels=True)
        
        # Overall title
        time_sec = sample_idx / (samples_per_epoch * interpolation_factor) * 2.0  # 2 sec per epoch
        fig.suptitle(f"{subject_id} - {condition} - {band} band | Epoch {epoch_idx+1} | Time: {time_sec:.2f}s",
                     fontsize=20, fontweight='bold')
        
        return axes
    
    # Create animation
    print(f"[VIDEO] Creating animation...")
    anim = animation.FuncAnimation(
        fig,
        update_frame,
        frames=total_frames,
        interval=1000/fps,  # milliseconds per frame
        blit=False,
        repeat=False
    )
    
    # Save video
    output_path = frames_path / output_file
    print(f"[VIDEO] Saving video to {output_path}...")
    print(f"[VIDEO] This may take several minutes depending on the number of frames...")
    
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
    view="graph",  # "graph", "matrix", or "oscillators"
    dpi=120,
    threshold_stc=80,
    epoch_duration=2.0,
):
    """
    Generate a focused video showing only one aspect of the network.
    
    Parameters
    ----------
    view : str
        Which visualization to show:
        - "graph": Network graph with nodes and edges
        - "matrix": Connectivity matrix heatmap
        - "oscillators": Phase oscillators on unit circle
        - "all": Side-by-side comparison
    """
    print(f"[VIDEO] Generating {view} video for {subject_id}-{condition}, band={band}")
    
    # Load data
    cond_dir = RESULTS_DIR / condition
    key = f"{subject_id}-{condition[:2].upper()}"
    
    try:
        rej = [x - 1 for x in rejected_epochs[key]]
    except KeyError:
        rej = []
    
    subject_syncro = load_file(cond_dir / f"syncro-{subject_id}-{condition}.pkl")
    subject_phases = load_file(cond_dir / f"phases-{subject_id}-{condition}.pkl")
    
    # Extract data
    stc_phases = subject_phases["phases_stc"][band]
    stc_kuramoto = subject_syncro["kuramoto_stc"][band][0]
    stc_syncros = subject_syncro["syncros_stc"][band][0]
    
    # Fixed: Filter rejected epochs from ALL arrays to maintain alignment
    stc_phases = [stc_phases[i] for i in range(len(stc_phases)) if i not in rej]
    stc_kuramoto = [stc_kuramoto[i] for i in range(len(stc_kuramoto)) if i not in rej]
    stc_syncros = [stc_syncros[i] for i in range(len(stc_syncros)) if i not in rej]
    
    num_epochs = len(stc_phases)
    samples_per_epoch = stc_phases[0].shape[1]
    
    print(f"[VIDEO] Valid epochs: {num_epochs}, samples: {samples_per_epoch}")
    
    # Interpolate phase data
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
    
    # Setup figure based on view
    if view == "all":
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        fig.patch.set_facecolor('white')
        ax_graph, ax_matrix, ax_osc = axes
        ax_osc = plt.subplot(1, 3, 3, projection='polar')
    elif view == "graph":
        fig, ax_graph = plt.subplots(figsize=(12, 12))
        fig.patch.set_facecolor('white')
        ax_matrix, ax_osc = None, None
    elif view == "matrix":
        fig, ax_matrix = plt.subplots(figsize=(12, 12))
        fig.patch.set_facecolor('white')
        ax_graph, ax_osc = None, None
    elif view == "oscillators":
        fig = plt.figure(figsize=(12, 12))
        fig.patch.set_facecolor('white')
        ax_osc = plt.subplot(111, projection='polar')
        ax_graph, ax_matrix = None, None
    else:
        raise ValueError(f"Invalid view: {view}. Use 'graph', 'matrix', 'oscillators', or 'all'")
    
    # Calculate frames: each epoch gets epoch_duration seconds at fps rate
    frames_per_epoch = int(epoch_duration * fps)
    frames_total = len(stc_phases_smooth) * frames_per_epoch
    
    print(f"[VIDEO] Total frames: {frames_total} (~{frames_total/fps:.1f}s)")
    
    def update(frame_num):
        
        if frame_num % 30 == 0:
            print(f"[VIDEO] Frame {frame_num}/{frames_total} ({frame_num*100//frames_total}%)")
        
        # Determine current epoch and sample within that epoch
        current_epoch = frame_num // frames_per_epoch
        frame_in_epoch = frame_num % frames_per_epoch
        
        # Map frame to sample index in interpolated data
        # Each epoch has samples_per_epoch * interpolation_factor samples
        total_samples_in_epoch = samples_per_epoch * interpolation_factor
        local_sample = int((frame_in_epoch / frames_per_epoch) * total_samples_in_epoch)
        local_sample = min(local_sample, total_samples_in_epoch - 1)
        
        # Get data for current frame
        phase_mat = stc_phases_smooth[current_epoch]
        syncro_mat = stc_syncros[current_epoch]
        kuramoto_epoch = stc_kuramoto[current_epoch]
        
        # Clear axes
        if ax_graph:
            ax_graph.clear()
        if ax_matrix:
            ax_matrix.clear()
        if ax_osc:
            ax_osc.clear()
        
        # Plot graph view
        if ax_graph:
            plot_graph(syncro_mat, ax_graph, threshold=threshold_stc, pos=stc_coords_2d,
                       edge_cmap="Dynamic", font_size=8, node_color=node_colors, edge_width=2)
            r_value = np.abs(np.exp(1j * phase_mat[:, min(local_sample, phase_mat.shape[1]-1)]).mean())
            ax_graph.set_title(f"Network Synchronization (r = {r_value:.3f})", fontsize=16)
        
        # Plot matrix view
        if ax_matrix:
            plot_mat(syncro_mat, ax_matrix, corte=stc_corte, xticks=False, colors=node_colors,
                     cmap="Dynamic", threshold=threshold_stc, signal_names=label_names)
            ax_matrix.set_title("Connectivity Matrix (100 parcels)", fontsize=16)
        
        # Plot oscillators view
        if ax_osc:
            safe_sample = min(local_sample, len(phase_mat[0]) - 1)
            plot_osc(phase_mat, kuramoto_epoch, ax_osc, sample=safe_sample,
                     color=node_colors, jitter=True, labels=False)
        
        # Overall title
        time_sec = local_sample / (samples_per_epoch * interpolation_factor) * 2.0
        epoch_time = current_epoch * 2.0 + time_sec
        fig.suptitle(
            f"{subject_id} - {condition} - {band} | Epoch {current_epoch+1}/{len(stc_phases)} | Time: {epoch_time:.2f}s",
            fontsize=20, fontweight='bold'
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        return axes if view == "all" else [ax_graph or ax_matrix or ax_osc]
    
    # Create animation
    print(f"[VIDEO] Creating animation...")
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=frames_total,
        interval=1000/fps,
        blit=False,
        repeat=False
    )
    
    # Save
    output_path = frames_path / output_file
    print(f"[VIDEO] Saving to {output_path}...")
    
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, bitrate=8000, codec='h264')
    
    anim.save(str(output_path), writer=writer, dpi=dpi)
    plt.close(fig)
    
    file_size = output_path.stat().st_size / 1e6
    print(f"[VIDEO] ✓ Complete! Duration: {frames_total/fps:.1f}s, Size: {file_size:.1f} MB")
    
    return output_path


# Example usage:
# output = generate_simple_network_video(
#     subject_id="S01",
#     condition="DMT",
#     band="Alpha",
#     output_file="network_alpha_smooth.mp4",
#     fps=30,
#     interpolation_factor=10,
#     view="graph",  # or "matrix", "oscillators", "all"
#     dpi=120
# )


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
):
    """
    Generate a 3D brain video showing parcels with synchronization-based coloring.
    OPTIMIZED VERSION: Pre-calculates all data for fast rendering.
    
    Parameters
    ----------
    rotate_camera : bool
        If True, slowly rotate the camera around the brain
    show_edges : bool
        If True, show edges between highly synchronized parcels
    edge_threshold : float
        Percentile threshold for showing edges (higher = fewer edges)
    """
    import matplotlib
    matplotlib.use('Agg')  # Use fast non-GUI backend
    
    print(f"[VIDEO 3D] Generating 3D brain video for {subject_id}-{condition}, band={band}")
    
    # Load data
    cond_dir = RESULTS_DIR / condition
    key = f"{subject_id}-{condition[:2].upper()}"
    
    try:
        rej = [x - 1 for x in rejected_epochs[key]]
    except KeyError:
        rej = []
    
    subject_syncro = load_file(cond_dir / f"syncro-{subject_id}-{condition}.pkl")
    subject_phases = load_file(cond_dir / f"phases-{subject_id}-{condition}.pkl")
    
    # Extract data
    stc_phases = subject_phases["phases_stc"][band]
    stc_kuramoto = subject_syncro["kuramoto_stc"][band][0]
    stc_syncros = subject_syncro["syncros_stc"][band][0]
    
    # Filter out rejected epochs
    stc_phases = [stc_phases[i] for i in range(len(stc_phases)) if i not in rej]
    stc_kuramoto = [stc_kuramoto[i] for i in range(len(stc_kuramoto)) if i not in rej]
    stc_syncros = [stc_syncros[i] for i in range(len(stc_syncros)) if i not in rej]
    
    num_epochs = len(stc_phases)
    samples_per_epoch = stc_phases[0].shape[1]
    
    print(f"[VIDEO 3D] Valid epochs: {num_epochs}, samples: {samples_per_epoch}")
    
    # Interpolate for smooth transitions
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
    
    # Calculate frames
    frames_per_epoch = int(epoch_duration * fps)
    frames_total = len(stc_phases_smooth) * frames_per_epoch
    
    print(f"[VIDEO 3D] Total frames: {frames_total} (~{frames_total/fps:.1f}s)")
    
    # PRE-CALCULATE all frame data for fast rendering
    print(f"[VIDEO 3D] Pre-calculating frame data (this speeds up rendering)...")
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
        
        # Pre-calculate node synchronization
        node_sync = syncro_mat.mean(axis=1)
        node_sync_norm = (node_sync - node_sync.min()) / (node_sync.max() - node_sync.min() + 1e-8)
        
        # Pre-calculate sizes and colors
        alphas = 0.5 + 0.5 * node_sync_norm
        sizes = 50 + 150 * node_sync_norm
        
        colors_rgba = np.zeros((num_parcels, 4))
        for i in range(num_parcels):
            color = node_colors[i]
            brightness = 0.4 + 0.6 * node_sync_norm[i]
            colors_rgba[i] = [color[0] * brightness, color[1] * brightness, 
                             color[2] * brightness, alphas[i]]
        
        edge_colors_rgba = np.array([[0, 0, 0, a] for a in alphas])
        
        # Pre-calculate edges if needed
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
        
        # Pre-calculate camera angle
        if rotate_camera:
            angle = (frame_num / frames_total) * 360
            elev = 10 + 5 * np.sin(np.radians(angle * 2))
            azim = 45 + angle
        else:
            elev, azim = 10, 45
        
        # Pre-calculate title values
        r_value = np.abs(np.exp(1j * phase_mat[:, min(local_sample, phase_mat.shape[1]-1)]).mean())
        time_sec = local_sample / (samples_per_epoch * interpolation_factor) * 2.0
        epoch_time = current_epoch * 2.0 + time_sec
        
        frame_data_cache.append({
            'sizes': sizes,
            'colors': colors_rgba,
            'edge_colors_node': edge_colors_rgba,
            'segments': segments,
            'edge_colors': edge_colors,
            'elev': elev,
            'azim': azim,
            'r_value': r_value,
            'epoch': current_epoch,
            'epoch_time': epoch_time,
            'num_epochs': num_epochs
        })
        
        if frame_num % 100 == 0:
            print(f"[VIDEO 3D] Pre-calculated {frame_num}/{frames_total} frames...")
    
    print(f"[VIDEO 3D] Pre-calculation complete! Starting fast render...")
    
    # Create figure with 3D axis
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get 3D coordinates
    coords = stc_coords_3d[:num_parcels]
    
    # Initialize scatter plot
    scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                        c=node_colors[:num_parcels], s=100, alpha=0.8, edgecolors='k', linewidths=0.5)
    
    # Initialize edge collection if needed
    line_collection = None
    if show_edges:
        line_collection = Line3DCollection([], colors='gray', linewidths=1, alpha=0.3)
        ax.add_collection3d(line_collection)
    
    # Set up camera
    ax.set_xlim(coords[:, 0].min() - 0.01, coords[:, 0].max() + 0.01)
    ax.set_ylim(coords[:, 1].min() - 0.01, coords[:, 1].max() + 0.01)
    ax.set_zlim(coords[:, 2].min() - 0.01, coords[:, 2].max() + 0.01)
    
    # Remove axes for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Initial view
    ax.view_init(elev=10, azim=45)
    
    def update(frame_num):
        # OPTIMIZED: Just retrieve pre-calculated data
        if frame_num % 30 == 0:
            print(f"[VIDEO 3D] Rendering frame {frame_num}/{frames_total} ({frame_num*100//frames_total}%)")
        
        data = frame_data_cache[frame_num]
        
        # Update scatter plot (fast operations)
        scatter._sizes = data['sizes']
        scatter._facecolors = data['colors']
        scatter._edgecolors = data['edge_colors_node']
        
        # Update edges if enabled
        if show_edges and line_collection is not None:
            line_collection.set_segments(data['segments'])
            line_collection.set_colors(data['edge_colors'])
        
        # Rotate camera
        ax.view_init(elev=data['elev'], azim=data['azim'])
        
        # Update title
        ax.set_title(
            f"{subject_id} - {condition} - {band} Band\n"
            f"Epoch {data['epoch']+1}/{data['num_epochs']} | "
            f"Time: {data['epoch_time']:.2f}s | "
            f"Kuramoto Order: r = {data['r_value']:.3f}",
            fontsize=14, fontweight='bold', pad=20
        )
        
        return scatter, line_collection
    
    print("[VIDEO 3D] Creating animation...")
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=frames_total,
        interval=1000/fps,
        blit=False
    )
    
    # Save
    output_path = frames_path / output_file
    print(f"[VIDEO 3D] Saving to {output_path}...")
    
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, bitrate=10000, codec='h264')
    
    anim.save(str(output_path), writer=writer, dpi=dpi)
    plt.close(fig)
    
    file_size = output_path.stat().st_size / 1e6
    print(f"[VIDEO 3D] ✓ Complete! Duration: {frames_total/fps:.1f}s, Size: {file_size:.1f} MB")
    
    return output_path