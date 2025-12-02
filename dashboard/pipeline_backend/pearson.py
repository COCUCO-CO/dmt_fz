#!/usr/bin/env python3
"""
Pearson correlation analysis between EEG metrics and questionnaires.
"""

import os
import pickle
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid tkinter warnings
import matplotlib.pyplot as plt

# BASE_DIR is the project root (dmt_fz/), not dashboard/
# From dashboard/pipeline_backend/ we need to go up TWO levels
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # Go up to dmt_fz/

# Check for custom output directory from environment
_custom_output = os.environ.get('PIPELINE_OUTPUT_DIR')
if _custom_output:
    RESULTS_DIR = Path(_custom_output)
    PEARSON_RESULTS_DIR = Path(_custom_output) / "pearson_results"
    print(f"[PATHS] Using custom output dir: {RESULTS_DIR}")
else:
    RESULTS_DIR = BASE_DIR / "fwd-inv-stc"
    PEARSON_RESULTS_DIR = BASE_DIR / "pearson_results"

EEG_DIR = BASE_DIR / "EEG_CLEAN"
SPECTRAL_DIR = BASE_DIR / "spectral_sources"
print(f"[PATHS] EEG_DIR: {EEG_DIR}")
PEARSON_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_file(data, folder, file):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / f"{file}.pkl", 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_file(file):
    with open(Path(file), 'rb') as handle:
        return pickle.load(handle)


def _sanitize_filename_part(part):
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(part))


def save_figure(fig, *parts, suffix="png", dpi=300, also_svg=True):
    filename = "_".join(_sanitize_filename_part(part) for part in parts if part is not None and str(part) != "")
    if not filename:
        filename = "figure"
    output_path = PEARSON_RESULTS_DIR / f"{filename}.{suffix}"
    print(f"[SAVE] {output_path}")
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if also_svg:
        svg_path = PEARSON_RESULTS_DIR / f"{filename}.svg"
        print(f"[SAVE] {svg_path}")
        fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def cond_path(cond):
    """Return Path to condition folder (handles legacy trailing backslash)."""
    return RESULTS_DIR / cond.replace("\\", "")



#%%

[node_colors, label_names, \
 label_names_short, stc_coords_3d, \
 ch_names, mapping, eeg_coords_2d] = load_file(RESULTS_DIR / "extra.pkl")

label_network = [x[:6] for x in label_names]

print(
    "[INFO] Loaded metadata:",
    f"{len(label_names)} labels,",
    f"{len(ch_names)} EEG channels,",
    f"coordinates shape {stc_coords_3d.shape if hasattr(stc_coords_3d, 'shape') else 'n/a'}",
)


#%%
import pandas as pd

df_labels = pd.DataFrame()
df_labels["label"] = [x[:6] for x in label_names]
df_labels["hemi"] = [x[:2] for x in df_labels["label"]]
df_labels["net"] = [x[3:] for x in df_labels["label"]]

def order_parameter_filter(phase, df_labels, hemi="both", net="all"):
    df = pd.concat([pd.DataFrame(phase), df_labels], axis=1)
    if hemi != "both":
        df = df[df["hemi"]==hemi]
    if net != "all":
        df = df[df["net"]==net]
    df = df.drop(columns=["hemi","label","net"])
    
    r = np.abs(np.exp(1j*df).mean(axis=0))
    return r

def network_filter(phase, df_labels, hemi="both", net="all"):
    df = pd.concat([pd.DataFrame(phase), df_labels], axis=1)
    df = (df[df["hemi"]==hemi] if hemi != "both" else df)
    df = (df[df["net"]==net] if net != "all" else df)
    df = df.drop(columns=["hemi","label","net"])
    return df

def order_parameter(phase):
    r = np.abs(np.exp(1j*phase).mean(axis=0))
    return r

def reject_outliers(data, m=2):
    """Reject outliers beyond m standard deviations."""
    # Ensure data is numeric array
    data = np.asarray(data, dtype=float)
    # Remove NaN/Inf values
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return data
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def reject_outliers_paired(data1, data2, m=2):
    """Reject outliers while maintaining pairing between two datasets."""
    # Ensure both are numeric arrays
    data1 = np.asarray(data1, dtype=float)
    data2 = np.asarray(data2, dtype=float)
    
    # Check if arrays have same length
    if len(data1) != len(data2):
        raise ValueError(f"Arrays must have same length for paired analysis: {len(data1)} vs {len(data2)}")
    
    # Remove NaN/Inf values from both arrays (keep pairs only where both are valid)
    valid_mask = np.isfinite(data1) & np.isfinite(data2)
    data1 = data1[valid_mask]
    data2 = data2[valid_mask]
    
    if len(data1) == 0:
        return data1, data2
    
    # Identify outliers in each dataset
    mask1 = np.abs(data1 - np.mean(data1)) < m * np.std(data1)
    mask2 = np.abs(data2 - np.mean(data2)) < m * np.std(data2)
    
    # Keep only pairs where BOTH are not outliers
    valid_pairs = mask1 & mask2
    
    return data1[valid_pairs], data2[valid_pairs]

def reject_outliers_per_subject(data, dataset, m=2):
    """Reject outliers per subject based on dataset statistics."""
    # Ensure both are numeric arrays
    data = np.asarray(data, dtype=float)
    dataset = np.asarray(dataset, dtype=float)
    # Remove NaN/Inf values
    data = data[np.isfinite(data)]
    dataset = dataset[np.isfinite(dataset)]
    if len(data) == 0 or len(dataset) == 0:
        return data
    return data[abs(data - np.mean(dataset)) < m * np.std(dataset)]

#%%

from pymatreader import read_mat
import numpy as np

rejected_subjects = [2, 5, 8, 16, 23, 31]
subjects = np.array([x for x in range(35) if x not in rejected_subjects])

print(f"[INFO] Total subjects available: {len(subjects)} (excluded: {rejected_subjects})")

folder = EEG_DIR

bad_epochs = read_mat(str(folder / "rejected_epochs.mat"))
subject = bad_epochs["rejected_epochs"][0]
epochs = bad_epochs["rejected_epochs"][1]
rejected_epochs = {k[:6].replace("_","-"): v.tolist() for k, v in zip(subject, epochs)}

print(f"[INFO] Loaded rejected epochs for {len(rejected_epochs)} subjects from {folder / 'rejected_epochs.mat'}")


def summarize_metric_subjects(metric_dict, metric_name):
    print(f"[INFO] {metric_name} subject counts by condition/band/net:")
    for cond_key in sorted(metric_dict.keys()):
        for band_key in band_list:
            nets = metric_dict.get(cond_key, {}).get(band_key, {})
            if not nets:
                print(f"  - {cond_key} / {band_key}: no data")
                continue
            counts = {net.strip(): int(np.sum(~np.isnan(values))) for net, values in nets.items()}
            print(
                f"  - {cond_key} / {band_key}: "
                f"min={min(counts.values(), default=0)}, "
                f"max={max(counts.values(), default=0)}, "
                f"counts={counts}"
            )


#%%
import os
from tqdm import tqdm
from copy import deepcopy

band_list = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
hemi_list = ["RH", "LH", "both"]
net_list = ["FPN", "DMN", "DAN", "LN " , "SVA", "SMN", "VN "]
cond_list = ["DMT", "EC", "EO"]

r_dict = {}
for cond in deepcopy(cond_list):
    r_dict[cond] = {}
    for band in deepcopy(band_list):
        r_dict[cond][band] = {}
        for hemi in deepcopy(hemi_list):
            r_dict[cond][band][hemi] = {net:[] for net in deepcopy(net_list)}
            


# path = RESULTS_DIR

# for cond in cond_list:
#     print(cond)
#     files_names = sorted([x for x in os.listdir(path+cond) if x.startswith('order-')])
#     for file in tqdm(files_names):
#         data_dict = load_file(path+cond+"\\"+file)
#         for band in band_list:
#             for hemi in hemi_list:
#                 for net in net_list:                   
#                     data_r = pd.DataFrame(data_dict[cond][band][hemi][net])
#                     print(data_r.shape)
#                     r_dict[cond][band][hemi][net].append(data_r.mean(axis=1).tolist())
                        
# save_file(r_dict, path, "r_kuramoto_nets_epochs_mean")


#%%

path = RESULTS_DIR
r_dict = load_file(path / "r_kuramoto_nets_epochs_mean.pkl")

available_conditions = sorted(r_dict.keys())
print(f"[INFO] Loaded Kuramoto metrics from {path / 'r_kuramoto_nets_epochs_mean.pkl'}")
print(f"[INFO] Conditions in dataset: {available_conditions}")


#%%

folder = SPECTRAL_DIR
labels = list(pd.read_csv(folder / "target_labels.txt", header=None)[0])
targets = pd.read_csv(folder / "target.csv", header=None, names=labels)

print(f"[INFO] Loaded targets: {len(labels)} variables, {len(targets)} observations from {folder}")

#%%
from copy import deepcopy
from scipy import stats 
from scipy.stats import norm
from decimal import Decimal
import numpy as np

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

band_list = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
#band_list = ["Gamma","Beta","Alpha","Theta","Delta"]
hemi_list = ["RH", "both", "LH"]
net_list = ["FPN", "DMN", "DAN", "LN " , "SVA", "SMN", "VN "]
cond_list = ["DMT\\", "EC\\", "EO\\"]
path = RESULTS_DIR

scale = 7
hemi = "both"


r_dict2 = {}
for cond in deepcopy(cond_list):
    r_dict2[cond] = {}
    for band in deepcopy(band_list):
        r_dict2[cond][band] = {net:[] for net in deepcopy(net_list)}    

def flatten_and_extract_numbers(data):
    """Recursively extract numeric values from nested structures."""
    if isinstance(data, (int, float, np.number)):
        return [float(data)]
    elif isinstance(data, (list, tuple, np.ndarray)):
        result = []
        for item in data:
            result.extend(flatten_and_extract_numbers(item))
        return result
    elif isinstance(data, dict):
        # Skip dicts, they're not numeric data
        return []
    else:
        # Try to convert to float
        try:
            return [float(data)]
        except (TypeError, ValueError):
            return []

for cond in cond_list:
    for i, band in enumerate(band_list):
        for j, net in enumerate(net_list):
            cond_str = cond.replace("\\","")
            cond_data = r_dict[cond_str][band][hemi][net]
            
            # Flatten and clean data
            flat_numbers = flatten_and_extract_numbers(cond_data)
            if not flat_numbers:
                print(f"[WARN] No numeric data for {cond_str}/{band}/{net}, filling with NaN")
                r_dict2[cond][band][net] = [np.nan] * len(cond_data)
                continue
            
            cond_array = np.asarray(flat_numbers, dtype=float)
            cond_clean = []
            for subj in cond_data:
                subj_numbers = flatten_and_extract_numbers(subj)
                if subj_numbers:
                    subj_array = np.asarray(subj_numbers, dtype=float)
                    cleaned = reject_outliers_per_subject(subj_array, cond_array)
                    cond_clean.append(cleaned)
                else:
                    cond_clean.append(np.array([]))
            
            cond_means = [np.mean(x) if np.asarray(x).size else np.nan for x in cond_clean]
            r_dict2[cond][band][net] = cond_means

summarize_metric_subjects(r_dict2, "Coherence")

# Detect how many subjects are actually in the processed data
n_processed_subjects = None
for cond in cond_list:
    for band in band_list:
        for net in net_list:
            cond_str = cond.replace("\\","")
            data = r_dict2[cond][band][net]
            if data and len(data) > 0:
                n_processed_subjects = len(data)
                break
        if n_processed_subjects is not None:
            break
    if n_processed_subjects is not None:
        break

if n_processed_subjects is None:
    print("[ERROR] Could not determine number of processed subjects")
    n_processed_subjects = len(subjects)
elif n_processed_subjects != len(targets):
    print(f"[INFO] Processed subjects ({n_processed_subjects}) != target subjects ({len(targets)})")
    print(f"[INFO] Adjusting targets to match first {n_processed_subjects} subjects")
    targets = targets.iloc[:n_processed_subjects]
            
#%%

r_dict3 = {}
for cond in deepcopy(cond_list):
    r_dict3[cond] = {}
    for band in deepcopy(band_list):
        r_dict3[cond][band] = {net:[] for net in deepcopy(net_list)}    

for cond in cond_list:
    for i, band in enumerate(band_list):
        for j, net in enumerate(net_list):
            cond_str = cond.replace("\\","")
            cond_data = r_dict[cond_str][band][hemi][net]
            
            # Flatten and clean data
            flat_numbers = flatten_and_extract_numbers(cond_data)
            if not flat_numbers:
                print(f"[WARN] No numeric data for {cond_str}/{band}/{net}, filling with NaN")
                r_dict3[cond][band][net] = [np.nan] * len(cond_data)
                continue
            
            cond_array = np.asarray(flat_numbers, dtype=float)
            cond_clean = []
            for subj in cond_data:
                subj_numbers = flatten_and_extract_numbers(subj)
                if subj_numbers:
                    subj_array = np.asarray(subj_numbers, dtype=float)
                    cleaned = reject_outliers_per_subject(subj_array, cond_array)
                    cond_clean.append(cleaned)
                else:
                    cond_clean.append(np.array([]))
            
            cond_vars = [np.var(x) if np.asarray(x).size else np.nan for x in cond_clean]
            r_dict3[cond][band][net] = cond_vars

summarize_metric_subjects(r_dict3, "Metastability")

#%%

from matplotlib import cm
from matplotlib.colors import to_rgb, to_rgba #,rgb2hex
#from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.stats import pearsonr
from statsmodels.stats.multitest import fdrcorrection

import numpy as np
#from sklearn.preprocessing import MinMaxScaler
from itertools import product

# Simple rgb2gray without skimage dependency
def rgb2gray(rgb):
    """Convert RGB to grayscale using standard luminance formula."""
    return 0.2989 * rgb[0] + 0.5870 * rgb[1] + 0.1140 * rgb[2]

# cmap = plt.get_cmap('plasma')
def to_gray(x):
  rgba = cm.twilight_shifted(abs(x))
  rgb = to_rgb(rgba) # RGBA to RGB
  gray = rgb2gray(rgb) # RGB to Gray (no skimage needed)
  return cm.gray(gray)

def r_to_color(r):
    return cm.plasma(abs(r))

labels = list(pd.read_csv(folder / "target_labels.txt", header=None)[0])[:-3]

n_sources = len(net_list)
n_quest = len(labels)

def PlotCorr(ax, dictio, band, cond, fdr=True, alpha=0.05, pvalue=0.05, ticks=False):
  rpearson_matrix = np.zeros((n_sources,n_quest))
  pvalues_matrix = np.ones((n_sources,n_quest))
  corr_matrix = np.zeros((4,n_sources,n_quest)) #RGBA

  for roi, score in product(range(n_sources), range(n_quest)):
    array1 = np.asarray(dictio[cond][band][net_list[roi]])
    array2 = targets.iloc[:,score].to_numpy()
    nas = np.logical_or(np.isnan(array1), np.isnan(array2))
    r, p = pearsonr(array1[~nas], array2[~nas])
    rpearson_matrix[roi,score] = r
    pvalues_matrix[roi,score] = p
    corr_matrix[:, roi,score] = r_to_color(r)

  #max_value = np.max(rpearson_matrix)
  #min_value = np.min(rpearson_matrix)

  # scaler = MinMaxScaler()
  # scaler.fit(rpearson_matrix)
  # rpearson_matrix = scaler.transform(rpearson_matrix)
  # for roi, score in product(range(n_sources), range(n_quest)):
  #  r = rpearson_matrix[roi,score]
  #  corr_matrix[:, roi,score] = to_rgba(cm.plasma(r))

  if fdr==False:
    #
    for roi, score in product(range(n_sources), range(n_quest)):
      if pvalues_matrix[roi,score] > pvalue:
        corr_matrix[:, roi,score] = to_gray(rpearson_matrix[roi,score])
  
  if fdr==True:
    p_corrected = fdrcorrection(pvalues_matrix.flatten(),alpha=alpha)[0].reshape(n_sources,n_quest)
    for roi, score in product(range(n_sources), range(n_quest)):
      if p_corrected[roi,score] != True:
        corr_matrix[:,roi,score] = to_gray(rpearson_matrix[roi,score])


  ax.imshow(np.transpose(corr_matrix))
  ax.set_xticks(np.arange(n_sources), net_list, fontsize=16, rotation='vertical')
  if ticks != False:
      ax.set_yticks(np.arange(n_quest), labels, fontsize=16)
  else:
      ax.set_yticks([])
      
  for i in range(n_sources):
      for j in range(n_quest):
          ax.text(i, j, f"{rpearson_matrix[i, j]:.2f}",
                  ha="center", va="center", color="white", fontsize=9)

  ax.set_title(band+" - "+cond.replace("\\",""), fontsize=16)
  #norm = plt.Normalize(radii[0], radii[-1])
  # m = plt.cm.ScalarMappable(cmap=cmap) #norm
  # m.set_array([min_value, max_value])
  # divider = make_axes_locatable(ax)
  # cax = divider.append_axes("right", size="1%", pad=0.07)
  # plt.colorbar(m, cax=cax).set_label('r Pearson', size=15)


for fdr in [True]:
    for dictio in [r_dict2, r_dict3]:
        for cond in ["DMT\\", "EC\\"]:
            cond_label = cond.replace("\\","")
            dict_label = "Coherence" if dictio is r_dict2 else "Metastability"
            for band in band_list:
                print(f"[INFO] Generating {dict_label} heatmap for condition {cond_label}, band {band}, FDR {'ON' if fdr else 'OFF'}")
                fig, ax = plt.subplots(figsize=(12, 12))
                PlotCorr(ax, dictio, band, cond, fdr=fdr, ticks=True)
                plt.suptitle(f"{band} - {cond_label} - {dict_label} (FDR {'ON' if fdr else 'OFF'})", fontsize=18)
                plt.tight_layout()
                save_figure(
                    fig,
                    "heatmap",
                    dict_label,
                    cond_label,
                    band,
                    f"fdr_{'on' if fdr else 'off'}",
                )

# Generate consolidated heatmaps with all bands as columns
def PlotCorrAllBands(axs, dictio, cond, fdr=True, alpha=0.05, pvalue=0.05):
    """Plot correlation heatmap with all bands as columns.
    
    FDR correction is applied PER BAND to be consistent with individual heatmaps.
    """
    rpearson_all = np.zeros((n_sources, n_quest, len(band_list)))
    pvalues_all = np.ones((n_sources, n_quest, len(band_list)))
    p_corrected_all = np.zeros((n_sources, n_quest, len(band_list)), dtype=bool)
    
    # Calculate correlations for all bands
    for band_idx, band in enumerate(band_list):
        for roi, score in product(range(n_sources), range(n_quest)):
            array1 = np.asarray(dictio[cond][band][net_list[roi]])
            array2 = targets.iloc[:,score].to_numpy()
            nas = np.logical_or(np.isnan(array1), np.isnan(array2))
            if np.sum(~nas) > 2:
                r, p = pearsonr(array1[~nas], array2[~nas])
                rpearson_all[roi, score, band_idx] = r
                pvalues_all[roi, score, band_idx] = p
        
        # Apply FDR correction PER BAND (consistent with individual heatmaps)
        if fdr:
            p_band = pvalues_all[:, :, band_idx].flatten()
            p_corrected_band = fdrcorrection(p_band, alpha=alpha)[0].reshape(n_sources, n_quest)
            p_corrected_all[:, :, band_idx] = p_corrected_band
    
    # Plot each band in its column
    for band_idx, band in enumerate(band_list):
        ax = axs[band_idx]
        corr_matrix = np.zeros((4, n_sources, n_quest))
        
        for roi, score in product(range(n_sources), range(n_quest)):
            r = rpearson_all[roi, score, band_idx]
            if fdr:
                is_sig = p_corrected_all[roi, score, band_idx]
            else:
                is_sig = pvalues_all[roi, score, band_idx] <= pvalue
            
            if is_sig:
                corr_matrix[:, roi, score] = r_to_color(r)
            else:
                corr_matrix[:, roi, score] = to_gray(r)
        
        ax.imshow(np.transpose(corr_matrix))
        ax.set_xticks(np.arange(n_sources), net_list, fontsize=12, rotation='vertical')
        if band_idx == 0:
            ax.set_yticks(np.arange(n_quest), labels, fontsize=10)
        else:
            ax.set_yticks([])
        
        for i in range(n_sources):
            for j in range(n_quest):
                ax.text(i, j, f"{rpearson_all[i, j, band_idx]:.2f}",
                        ha="center", va="center", color="white", fontsize=7)
        
        cond_clean = cond.replace("\\", "")
        ax.set_title(f"{band} - {cond_clean}", fontsize=14)

print("[INFO] Generating consolidated heatmaps with all bands...")
for fdr in [True]:
    for dictio in [r_dict2, r_dict3]:
        for cond in ["DMT\\", "EC\\"]:
            cond_label = cond.replace("\\","")
            dict_label = "Coherence" if dictio is r_dict2 else "Metastability"
            print(f"[INFO] Generating consolidated {dict_label} heatmap for {cond_label}, FDR {'ON' if fdr else 'OFF'}")
            
            fig, axs = plt.subplots(1, len(band_list), figsize=(len(band_list) * 6, 14))
            PlotCorrAllBands(axs, dictio, cond, fdr=fdr)
            plt.suptitle(f"All Bands - {dict_label} - {cond_label} (FDR {'ON' if fdr else 'OFF'})", fontsize=18)
            plt.tight_layout()
            save_figure(
                fig,
                "heatmap_all_bands",
                dict_label,
                cond_label,
                f"fdr_{'on' if fdr else 'off'}",
            )


#%%

labels = list(pd.read_csv(folder / "target_labels.txt", header=None)[0])

n_sources = len(net_list)
#n_quest = len(labels)
n_bands = len(band_list)

def PlotCorr2(ax, dictio, cond, score, fdr=True, alpha=0.05, pvalue=0.05, ticks=False):
  rpearson_matrix = np.zeros((n_sources,n_bands))
  pvalues_matrix = np.ones((n_sources,n_bands))
  corr_matrix = np.zeros((4,n_sources,n_bands)) #RGBA

  for roi, band in product(range(n_sources), range(n_bands)):
    array1 = np.asarray(dictio[cond][band_list[band]][net_list[roi]])
    array2 = targets.iloc[:,score].to_numpy()
    nas = np.logical_or(np.isnan(array1), np.isnan(array2))
    r, p = pearsonr(array1[~nas], array2[~nas])
    rpearson_matrix[roi,band] = r
    pvalues_matrix[roi,band] = p
    corr_matrix[:,roi,band] = r_to_color(r)

  if fdr==False:
    #
    for roi, band in product(range(n_sources), range(n_bands)):
      if pvalues_matrix[roi,band] > pvalue:
        corr_matrix[:,roi,band] = to_gray(rpearson_matrix[roi,band])
  
  if fdr==True:
    p_corrected = fdrcorrection(pvalues_matrix.flatten(),alpha=alpha)[0].reshape(n_sources,n_bands)
    for roi, band in product(range(n_sources), range(n_bands)):
      if p_corrected[roi,band] != True:
        corr_matrix[:,roi,band] = to_gray(rpearson_matrix[roi,band])


  ax.imshow(np.transpose(corr_matrix))
  ax.set_xticks(np.arange(n_sources), net_list, fontsize=16, rotation='vertical')
  if ticks != False:
      ax.set_yticks(np.arange(n_bands), band_list, fontsize=16)
  else:
      ax.set_yticks([])
      
  for i in range(n_sources):
      for j in range(n_bands):
          ax.text(i, j, str(rpearson_matrix[i, j])[:5],
                  ha="center", va="center", color="white", fontsize=16)

  ax.set_title(cond.replace("\\","")+ " - "+labels[score], fontsize=16)
  #norm = plt.Normalize(radii[0], radii[-1])
  # m = plt.cm.ScalarMappable(cmap=cmap) #norm
  # m.set_array([min_value, max_value])
  # divider = make_axes_locatable(ax)
  # cax = divider.append_axes("right", size="1%", pad=0.07)
  # plt.colorbar(m, cax=cax).set_label('r Pearson', size=15)

fig, axs = plt.subplots(figsize=(10, 30))
cond = "DMT\\"
score_idx = 0
cond_clean = cond.replace("\\", "")
print(f"[INFO] Generating band vs net heatmap for condition {cond_clean}, score index {score_idx}")
PlotCorr2(axs, r_dict3, cond, score=score_idx, fdr=False, ticks=True)
save_figure(
    fig,
    "heatmap_band_vs_net",
    "metastability",
    cond_clean,
    f"score_{score_idx}",
    labels[score_idx],
    "fdr_off",
)

#%%

labels = list(pd.read_csv(folder / "target_labels.txt", header=None)[0])

n_sources = len(net_list)
#n_quest = len(labels)
n_bands = len(band_list)

def CalcCorr(dictio, cond, score, fdr=True, alpha=0.05, pvalue=0.05, ticks=True):
  rpearson_matrix = np.zeros((n_sources,n_bands))
  pvalues_matrix = np.ones((n_sources,n_bands))

  for roi, band in product(range(n_sources), range(n_bands)):
    array1 = np.asarray(dictio[cond][band_list[band]][net_list[roi]])
    array2 = targets.iloc[:,score].to_numpy()
    nas = np.logical_or(np.isnan(array1), np.isnan(array2))
    r, p = pearsonr(array1[~nas], array2[~nas])
    rpearson_matrix[roi,band] = r
    pvalues_matrix[roi,band] = p
    
  return rpearson_matrix, pvalues_matrix.flatten()

fdr = True

score = 15
print(f"[INFO] Building correlation matrices for score {score} (FDR {'ON' if fdr else 'OFF'})")
rs, ps_segments, titles = [], [], []
metadata = []
for r_dict in [r_dict2, r_dict3]:
    sub = "Coherence" if r_dict is r_dict2 else "Metastability"
    for cond in ["EC\\", "DMT\\"]:
        try:
            r, p = CalcCorr(r_dict, cond, score=score, fdr=fdr)
        except ValueError as exc:
            print(f"[WARN] Skipping cond={cond} metric={sub}: {exc}")
            continue
        rs.append(r)
        ps_segments.append(p)
        titles.append(cond.replace("\\", " - ") + sub)
        metadata.append((sub, cond.replace("\\", ""), score))

if rs:
    ps_concat = np.concatenate(ps_segments)
    p_corrected = fdrcorrection(ps_concat, alpha=0.05)[0]
    n_panels = len(rs)
    fig, ax = plt.subplots(1, n_panels, figsize=(10 * n_panels, 8))
    axs = np.atleast_1d(ax)
    offset = 0
    for idx, (r_mat, p_seg, title) in enumerate(zip(rs, ps_segments, titles)):
        axis = axs[idx]
        seg_len = p_seg.size
        if seg_len != n_sources * n_bands:
            print(f"[WARN] Expected {n_sources * n_bands} p-values, got {seg_len} for {title}. Skipping plot.")
            axis.axis("off")
            offset += seg_len
            continue
        this_p = p_corrected[offset:offset + seg_len].reshape(n_sources, n_bands)
        offset += seg_len
        corr_mat = np.zeros((4, n_sources, n_bands))
        for roi, band in product(range(n_sources), range(n_bands)):
            if this_p[roi, band]:
                corr_mat[:, roi, band] = r_to_color(r_mat[roi, band])
            else:
                corr_mat[:, roi, band] = to_gray(r_mat[roi, band])

        axis.imshow(np.transpose(corr_mat))
        axis.set_xticks(np.arange(n_sources), net_list, fontsize=12, rotation='vertical')
        axis.set_yticks(np.arange(n_bands), band_list, fontsize=12)
        for j in range(n_sources):
            for k in range(n_bands):
                axis.text(j, k, str(r_mat[j, k])[:5], ha="center", va="center", color="white", fontsize=10)
        axis.set_title(title, fontsize=18)

    plt.suptitle(labels[score], fontsize=22)
    plt.tight_layout()
    save_figure(
        fig,
        "corr_matrix_summary",
        f"score_{score}",
        labels[score],
        f"fdr_{'on' if fdr else 'off'}",
        "metrics_coherence_metastability",
        "conds_EC_DMT",
    )
else:
    print(f"[WARN] No correlation matrices generated for score {score}.")

#%%

labels = list(pd.read_csv(folder / "target_labels.txt", header=None)[0])

n_sources = len(net_list)
#n_quest = len(labels)
n_bands = len(band_list)

def CalcCorr(dictio, cond, score, fdr=True, alpha=0.05, pvalue=0.05, ticks=True):
  rpearson_matrix = np.zeros((n_sources,n_bands))
  pvalues_matrix = np.ones((n_sources,n_bands))

  for roi, band in product(range(n_sources), range(n_bands)):
    array1 = np.asarray(dictio[cond][band_list[band]][net_list[roi]])
    array2 = targets.iloc[:,score].to_numpy()
    nas = np.logical_or(np.isnan(array1), np.isnan(array2))
    r, p = pearsonr(array1[~nas], array2[~nas])
    rpearson_matrix[roi,band] = r
    pvalues_matrix[roi,band] = p
    
  return rpearson_matrix, pvalues_matrix.flatten()

fdr = True

score_indices = [0, 15]
print(f"[INFO] Building combined correlation matrices for scores {score_indices} (FDR {'ON' if fdr else 'OFF'})")
rs, ps_segments, titles = [], [], []
for score in score_indices:
    for r_dict in [r_dict2, r_dict3]:
        sub = "Coherence" if r_dict is r_dict2 else "Metastability"
        for cond in ["EC\\", "DMT\\"]:
            try:
                r, p = CalcCorr(r_dict, cond, score=score, fdr=fdr)
            except ValueError as exc:
                print(f"[WARN] Skipping cond={cond} metric={sub} score={score}: {exc}")
                continue
            rs.append(r)
            ps_segments.append(p)
            titles.append(cond.replace("\\", " - ") + sub + f"_score={score}")

if rs:
    ps_concat = np.concatenate(ps_segments)
    p_corrected = fdrcorrection(ps_concat, alpha=0.05)[0]
    n_panels = len(rs)
    fig, ax = plt.subplots(1, n_panels, figsize=(10 * n_panels, 8))
    axs = np.atleast_1d(ax)
    offset = 0
    for idx, (r_mat, p_seg, title) in enumerate(zip(rs, ps_segments, titles)):
        axis = axs[idx]
        seg_len = p_seg.size
        if seg_len != n_sources * n_bands:
            print(f"[WARN] Expected {n_sources * n_bands} p-values, got {seg_len} for {title}. Skipping plot.")
            axis.axis("off")
            offset += seg_len
            continue
        this_p = p_corrected[offset:offset + seg_len].reshape(n_sources, n_bands)
        offset += seg_len
        corr_mat = np.zeros((4, n_sources, n_bands))
        for roi, band in product(range(n_sources), range(n_bands)):
            if this_p[roi, band]:
                corr_mat[:, roi, band] = r_to_color(r_mat[roi, band])
            else:
                corr_mat[:, roi, band] = to_gray(r_mat[roi, band])

        axis.imshow(np.transpose(corr_mat))
        axis.set_xticks(np.arange(n_sources), net_list, fontsize=12, rotation='vertical')
        axis.set_yticks(np.arange(n_bands), band_list, fontsize=12)
        for j in range(n_sources):
            for k in range(n_bands):
                axis.text(j, k, str(r_mat[j, k])[:5], ha="center", va="center", color="white", fontsize=10)
        axis.set_title(title, fontsize=18)

    composite_label = "_".join(_sanitize_filename_part(labels[idx]) for idx in score_indices)
    plt.suptitle(", ".join(labels[idx] for idx in score_indices), fontsize=22)
    plt.tight_layout()
    save_figure(
        fig,
        "corr_matrix_summary_scores",
        composite_label,
        f"fdr_{'on' if fdr else 'off'}",
        "metrics_coherence_metastability",
        "conds_EC_DMT",
    )
else:
    print("[WARN] No combined correlation matrices generated for requested scores.")


#%%
from scipy import stats

# Generate scatter plots for ALL significant correlations (FDR-corrected PER BAND, same as heatmaps)
print("[INFO] Generating scatter plots for all significant correlations...")

alpha = 0.05
labels_scatter = list(pd.read_csv(folder / "target_labels.txt", header=None)[0])[:-3]

# Network name mapping for axis labels
net_names = {
    "FPN": "Frontoparietal Network",
    "DMN": "Default Mode Network", 
    "DAN": "Dorsal Attention Network",
    "LN ": "Limbic Network",
    "SVA": "Salience/Ventral Attention",
    "SMN": "Somatomotor Network",
    "VN ": "Visual Network"
}

for dictio, metric_name in [(r_dict2, "Coherence"), (r_dict3, "Metastability")]:
    for cond in ["DMT\\", "EC\\"]:
        cond_clean = cond.replace("\\", "")
        total_significant = 0
        
        # Apply FDR correction PER BAND (consistent with heatmaps)
        for band in band_list:
            # Collect p-values for this band only
            band_pvalues = []
            band_combinations = []
            
            for roi in net_list:
                for score_idx in range(len(labels_scatter)):
                    array1 = np.asarray(dictio[cond][band][roi])
                    array2 = targets.iloc[:,score_idx].to_numpy()
                    nas = np.logical_or(np.isnan(array1), np.isnan(array2))
                    
                    if np.sum(~nas) > 2:
                        r, p = pearsonr(array1[~nas], array2[~nas])
                        band_pvalues.append(p)
                        band_combinations.append((roi, score_idx, r, p))
                    else:
                        band_pvalues.append(1.0)
                        band_combinations.append((roi, score_idx, np.nan, 1.0))
            
            # Apply FDR correction for this band
            rejected, pvalues_corrected = fdrcorrection(band_pvalues, alpha=alpha)
            
            n_significant = np.sum(rejected)
            total_significant += n_significant
            if n_significant > 0:
                print(f"[INFO] {metric_name} - {cond_clean} - {band}: {n_significant} significant correlations")
            
            # Generate scatter plots for significant correlations in this band
            for idx, (roi, score_idx, r_val, p_orig) in enumerate(band_combinations):
                if not rejected[idx]:
                    continue
                
                array1 = np.asarray(dictio[cond][band][roi])
                array2 = targets.iloc[:,score_idx].to_numpy()
                nas = np.logical_or(np.isnan(array1), np.isnan(array2))
                
                m, b, r, p, std_err = stats.linregress(array1[~nas], array2[~nas])
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(array1[~nas], array2[~nas], s=150, edgecolor="black", alpha=0.7)
                
                ax.axline((array1[~nas].mean(), array2[~nas].mean()),
                          slope=m, color="black", linestyle=(0, (5, 5)),
                          label=f"r = {r:.2f}, p(FDR) = {pvalues_corrected[idx]:.4f}")
                
                ax.set_title(f"{band} - {cond_clean}", fontsize=14)
                
                roi_name = net_names.get(roi, roi.strip())
                ax.set_xlabel(f"{roi_name} - {metric_name}", fontsize=12)
                ax.set_ylabel(labels_scatter[score_idx], fontsize=12)
                
                ax.spines['top'].set_color('none')
                ax.spines['right'].set_color('none')
                
                ax.legend(fontsize=10)
                
                print(f"  [SCATTER] {metric_name} | {cond_clean} | {band} | {roi.strip()} | {labels_scatter[score_idx]} | r={r:.2f}")
                save_figure(
                    fig,
                    "scatter",
                    metric_name.lower(),
                    cond_clean,
                    band,
                    roi.strip(),
                    f"score_{score_idx}",
                    labels_scatter[score_idx],
                )
        
        print(f"[INFO] {metric_name} - {cond_clean}: Total {total_significant} significant correlations across all bands")

#%%
from scipy import stats 
from scipy.stats import norm
from decimal import Decimal

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

band_list = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
#band_list = ["Gamma","Beta","Alpha","Theta","Delta"]
hemi_list = ["RH", "both", "LH"]
net_list = ["FPN", "DMN", "DAN", "LN " , "SVA", "SMN", "VN "]
cond_list = ["DMT\\", "EC\\", "EO\\"]
path = RESULTS_DIR

scale = 4
hemi = "both"

# Generate histograms for all condition pairs
from itertools import combinations
condition_pairs = list(combinations(cond_list, 2))
print(f"[INFO] Generating coherence histograms for condition pairs: {condition_pairs} (hemi={hemi}) with FDR correction")

for cond_pair in condition_pairs:
    print(f"[INFO] Processing pair {cond_pair[0]} vs {cond_pair[1]}")
    fig, axs = plt.subplots(figsize=(7*scale,5*scale), ncols=7, nrows=5)

    # FIRST PASS: Calculate all p-values for FDR correction
    print("[INFO] Calculating all p-values for FDR correction...")
    pvalues_matrix = np.ones((len(band_list), len(net_list)))
    data_cache = {}

    #for i, hemi in enumerate(hemi_list):
    for i, band in enumerate(band_list):
        for j, net in enumerate(net_list):
            # Remove backslash from cond_list to match r_dict keys
            cond1_str = cond_pair[0].replace("\\","")
            cond2_str = cond_pair[1].replace("\\","")
            
            # Use r_dict2 (coherence data) with backslash keys
            try:
                cond1 = r_dict2[cond_pair[0]][band][net]
                cond2 = r_dict2[cond_pair[1]][band][net]
            except (KeyError, TypeError) as e:
                print(f"[WARN] Skipping histogram {band}/{net}: {cond_pair[0]} or {cond_pair[1]} data not complete")
                continue
            
            # Flatten and clean data
            cond1_flat = flatten_and_extract_numbers(cond1)
            cond2_flat = flatten_and_extract_numbers(cond2)
            
            # Convert to arrays
            cond1_array = np.asarray(cond1_flat, dtype=float) if cond1_flat else np.array([])
            cond2_array = np.asarray(cond2_flat, dtype=float) if cond2_flat else np.array([])
            
            # Try paired analysis if lengths match, otherwise use independent
            use_paired = False
            if len(cond1_array) > 0 and len(cond2_array) > 0:
                if len(cond1_array) == len(cond2_array):
                    # Same length: paired analysis
                    cond1, cond2 = reject_outliers_paired(cond1_array, cond2_array)
                    use_paired = True
                else:
                    # Different lengths: independent analysis
                    cond1 = reject_outliers(cond1_array)
                    cond2 = reject_outliers(cond2_array)
                    use_paired = False
            else:
                cond1 = cond1_array
                cond2 = cond2_array
            
            # Store data for second pass
            data_cache[(i, j)] = (cond1, cond2, cond1_str, cond2_str)
            
            # Calculate p-value for FDR correction using appropriate t-test
            if len(cond1) > 1 and len(cond2) > 1:
                if use_paired:
                    statistic, pvalue = stats.ttest_rel(cond1, cond2)  # Paired t-test
                else:
                    statistic, pvalue = stats.ttest_ind(cond1, cond2)  # Independent t-test
                pvalues_matrix[i, j] = pvalue

    # Apply FDR correction (Benjamini-Hochberg)
    pvalues_flat = pvalues_matrix.flatten()
    rejected, pvalues_corrected = fdrcorrection(pvalues_flat, alpha=0.05)
    pvalues_corrected = pvalues_corrected.reshape(len(band_list), len(net_list))
    rejected = rejected.reshape(len(band_list), len(net_list))

    n_significant = rejected.sum()
    n_total = len(pvalues_flat)
    print(f"[INFO] FDR correction (α=0.05): {n_significant}/{n_total} comparisons significant ({100*n_significant/n_total:.1f}%)")

    # SECOND PASS: Plot with FDR-corrected significance
    for i, band in enumerate(band_list):
        for j, net in enumerate(net_list):
            if (i, j) not in data_cache:
                continue
            
            cond1, cond2, cond1_str, cond2_str = data_cache[(i, j)]
            
            axs[i][j].hist(cond1, bins=50, alpha=0.5, density=True)
            axs[i][j].hist(cond2, bins=50, alpha=0.5, density=True)
            
            xmin, xmax = axs[i][j].get_xlim()
            x = np.linspace(xmin, xmax, 100)
            
            mu1, std1 = norm.fit(cond1)
            mu2, std2 = norm.fit(cond2)
            p1 = norm.pdf(x, mu1, std1)
            p2 = norm.pdf(x, mu2, std2)
            
            axs[i][j].plot(x, p1, colors[0], linewidth=2, label=cond1_str)
            axs[i][j].plot(x, p2, colors[1], linewidth=2, label=cond2_str)
            
            axs[i][j].axvline(x=mu1, linestyle="--", linewidth=2, color=colors[0])
            axs[i][j].axvline(x=mu2, linestyle="--", linewidth=2, color=colors[1])
            
            # Use FDR-corrected p-value
            pvalue_corrected = pvalues_corrected[i, j]
            is_significant = rejected[i, j]
            
            # Format p-value with FDR-corrected significance indicators and colors
            if is_significant:
                if pvalue_corrected < 0.001:
                    pvalue_str = f'p < 0.001 ***'
                    txt_color = 'green'
                    txt_weight = 'bold'
                elif pvalue_corrected < 0.01:
                    pvalue_str = f'p = {pvalue_corrected:.4f} **'
                    txt_color = 'darkgreen'
                    txt_weight = 'bold'
                else:  # < 0.05
                    pvalue_str = f'p = {pvalue_corrected:.4f} *'
                    txt_color = 'orange'
                    txt_weight = 'bold'
            else:
                pvalue_str = f'p = {pvalue_corrected:.4f} n.s.'
                txt_color = 'gray'
                txt_weight = 'normal'
            
            axs[i][j].text(0.05, 0.95, pvalue_str, transform=axs[i][j].transAxes,
                           fontsize=12, ha="left", va="top", color=txt_color, weight=txt_weight)
            axs[i][j].legend()

    for ax, col in zip(axs[0], net_list):
        ax.set_title(col, size=24)
    for ax, row in zip(axs[:,0], band_list):
        ax.set_ylabel(row, size=24) #rotation=0, size='large')


    title = "Kuramoto r Order Parameter - "
    fig.suptitle(title+cond1_str+" vs "+cond2_str, size=36)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    save_figure(
        fig,
        "histogram_coherence",
        cond1_str,
        "vs",
        cond2_str,
        "hemi_both",
        "bands_all",
        "nets_all",
    )

#%%

# r_dict = {}
# for cond in deepcopy(cond_list):
#     r_dict[cond] = {}
#     for band in deepcopy(band_list):
#         r_dict[cond][band] = {}
#         for hemi in deepcopy(hemi_list):
#             r_dict[cond][band][hemi] = {}
#             for (net1,net2) in combs(deepcopy(net_list), 2):         
#                 r_dict[cond][band][hemi][net1] = r_dict[cond][band][hemi].get(net1,{})
#                 r_dict[cond][band][hemi][net1][net2] = []

# path = "E:\\FedeZ\\DMT\\fwd-inv-stc\\"

# for cond in cond_list:
#     print(cond)
#     files_names = sorted([x for x in os.listdir(path+cond) if x.startswith('order_all')])
#     for file in tqdm(files_names):
#         data_dict = load_file(path+cond+"\\"+file)
        
#         for band in band_list:
#             for hemi in hemi_list:
#                 for (net1,net2) in combs(net_list, 2):
#                     num_epochs = len(data_dict[cond][band][hemi][net1])
#                     for epoch in range(num_epochs):
#                         data_r1 = pd.DataFrame(data_dict[cond][band][hemi][net1][epoch])
#                         data_r2 = pd.DataFrame(data_dict[cond][band][hemi][net2][epoch])
#                         data_r = order_parameter(pd.concat([data_r1, data_r2]))
#                         r_dict[cond][band][hemi][net1][net2].append(data_r.mean().to_list())
                        
# save_file(r_dict, path, "r_kuramoto_nets_all_mean")


#%%

# Load r_kuramoto_nets_all_mean into a different variable to avoid overwriting r_dict2
r_dict_all_mean = load_file(path / "r_kuramoto_nets_all_mean.pkl")


#%%
from itertools import combinations_with_replacement as comb
from itertools import combinations

hemi="both"

# Generate all pairwise comparisons between conditions
condition_pairs = list(combinations(cond_list, 2))
print(f"[INFO] Will generate network-pair histograms for condition pairs: {condition_pairs}")

for cond_pair in condition_pairs:
    for band in band_list:
        print(f"[INFO] Generating network-pair histograms for band {band} ({cond_pair[0]} vs {cond_pair[1]}, hemi={hemi}) with FDR correction")
        fig, axs = plt.subplots(figsize=(21*scale,5*scale), ncols=7, nrows=7)
        
        # FIRST PASS: Calculate all p-values for FDR correction
        n_pairs = sum(1 for i in range(len(net_list)) for j in range(len(net_list)) if i <= j)
        pvalues_list = []
        data_cache_pairs = {}
        pair_indices = []
        
        for i, net1 in enumerate(net_list):
            for j, net2 in enumerate(net_list):
                if i <= j:
                    cond1_str = cond_pair[0].replace("\\","")
                    cond2_str = cond_pair[1].replace("\\","")
                cond1 = r_dict_all_mean[cond1_str][band][hemi][net1][net2]
                cond2 = r_dict_all_mean[cond2_str][band][hemi][net1][net2]
                
                # Flatten and clean data
                cond1_flat = flatten_and_extract_numbers(cond1)
                cond2_flat = flatten_and_extract_numbers(cond2)
                
                # Convert to arrays
                cond1_array = np.asarray(cond1_flat, dtype=float) if cond1_flat else np.array([])
                cond2_array = np.asarray(cond2_flat, dtype=float) if cond2_flat else np.array([])
                
                # Try paired analysis if lengths match, otherwise use independent
                use_paired = False
                if len(cond1_array) > 0 and len(cond2_array) > 0:
                    if len(cond1_array) == len(cond2_array):
                        # Same length: paired analysis
                        cond1, cond2 = reject_outliers_paired(cond1_array, cond2_array)
                        use_paired = True
                    else:
                        # Different lengths: independent analysis
                        cond1 = reject_outliers(cond1_array)
                        cond2 = reject_outliers(cond2_array)
                        use_paired = False
                else:
                    cond1 = cond1_array
                    cond2 = cond2_array
                
                # Store data for second pass
                data_cache_pairs[(i, j)] = (cond1, cond2, cond1_str, cond2_str)
                
                # Calculate p-value using appropriate t-test
                if len(cond1) > 1 and len(cond2) > 1:
                    if use_paired:
                        statistic, pvalue = stats.ttest_rel(cond1, cond2)  # Paired t-test
                    else:
                        statistic, pvalue = stats.ttest_ind(cond1, cond2)  # Independent t-test
                    pvalues_list.append(pvalue)
                else:
                    pvalues_list.append(1.0)
                pair_indices.append((i, j))
    
    # Apply FDR correction
    if pvalues_list:
        rejected_pairs, pvalues_corrected_pairs = fdrcorrection(pvalues_list, alpha=0.05)
        n_significant_pairs = rejected_pairs.sum()
        print(f"  [{band}] FDR correction: {n_significant_pairs}/{len(pvalues_list)} pairs significant")
    else:
        rejected_pairs = []
        pvalues_corrected_pairs = []
    
    # SECOND PASS: Plot with FDR-corrected significance
    pair_idx = 0
    for i, net1 in enumerate(net_list):
        for j, net2 in enumerate(net_list):
            if i <= j:
                if (i, j) not in data_cache_pairs:
                    axs[i][j].remove()
                    continue
                
                cond1, cond2, cond1_str, cond2_str = data_cache_pairs[(i, j)]
                
                axs[i][j].hist(cond1, bins=50, alpha=0.5, density=True)
                axs[i][j].hist(cond2, bins=50, alpha=0.5, density=True)
                
                xmin, xmax = axs[i][j].get_xlim()
                x = np.linspace(xmin, xmax, 100)
                
                mu1, std1 = norm.fit(cond1)
                mu2, std2 = norm.fit(cond2)
                p1 = norm.pdf(x, mu1, std1)
                p2 = norm.pdf(x, mu2, std2)
                
                axs[i][j].plot(x, p1, colors[0], linewidth=2)
                axs[i][j].plot(x, p2, colors[1], linewidth=2)
                
                axs[i][j].axvline(x=mu1, linestyle="--", linewidth=2, color=colors[0])
                axs[i][j].axvline(x=mu2, linestyle="--", linewidth=2, color=colors[1])
                
                # Use FDR-corrected p-value
                if pair_idx < len(pvalues_corrected_pairs):
                    pvalue_corrected = pvalues_corrected_pairs[pair_idx]
                    is_significant = rejected_pairs[pair_idx]
                    
                    if is_significant:
                        if pvalue_corrected < 0.001:
                            pvalue_str = f'p < 0.001 ***'
                            txt_color = 'green'
                            txt_weight = 'bold'
                        elif pvalue_corrected < 0.01:
                            pvalue_str = f'p = {pvalue_corrected:.4f} **'
                            txt_color = 'darkgreen'
                            txt_weight = 'bold'
                        else:
                            pvalue_str = f'p = {pvalue_corrected:.4f} *'
                            txt_color = 'orange'
                            txt_weight = 'bold'
                    else:
                        pvalue_str = f'p = {pvalue_corrected:.4f} n.s.'
                        txt_color = 'gray'
                        txt_weight = 'normal'
                    
                    axs[i][j].text(0.05, 0.95, pvalue_str, transform=axs[i][j].transAxes,
                                   fontsize=12, ha="left", va="top", color=txt_color, weight=txt_weight)
                pair_idx += 1
            else:
                axs[i][j].remove() 
        
        for ax, col in zip(axs[0], net_list):
            ax.set_title(col, size=24)
        for ax, row in zip(axs[:,0], net_list):
            ax.set_ylabel(row, size=24) #rotation=0, size='large')
        
        title = "Kuramoto r Order Parameter - "+band+" - "
        fig.suptitle(title+cond1_str+" vs "+cond2_str, size=36)
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        save_figure(
            fig,
            "histogram_coherence_pairs",
            band,
            cond1_str,
            "vs",
            cond2_str,
            "hemi_both",
        )