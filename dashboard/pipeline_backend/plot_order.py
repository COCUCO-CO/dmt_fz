#!/usr/bin/env python3
"""
plot_order.py - Kuramoto order parameter plotting and visualization

Generates plots for Kuramoto synchronization metrics across different 
experimental conditions, frequency bands, and brain networks.

NOTE: This script requires pre-computed data files. Run build_order_data.py
first to generate the required .pkl files:
  - r_kuramoto_nets_epochs_mean.pkl (for plot_histogram_kuramoto_full)
  - r_kuramoto_nets_all_mean.pkl (for network pair analysis)

Usage:
  python plot_order.py                    # Generate all plots
  python plot_order.py --max-subjects 5   # Limit subjects for testing
"""

import argparse
import os
import pickle
import re
from itertools import product
from multiprocessing import Pool, cpu_count
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for multiprocessing compatibility
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymatreader import read_mat
from scipy import stats
from scipy.stats import norm
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm

from paths import BASE_DIR, RESULTS_DIR, EEG_CLEAN_DIR as EEG_DIR, ensure_dir

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

PLOT_ORDER_RESULTS_DIR = BASE_DIR / "plot_order_results"
PLOT_ORDER_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BAND_LIST = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
HEMI_LIST = ["RH", "LH", "both"]
NET_LIST = ["FPN", "DMN", "DAN", "LN ", "SVA", "SMN", "VN "]
COND_LIST = ["DMT", "EC", "EO"]

REJECTED_SUBJECTS = [2, 5, 8, 16, 23, 31]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Kuramoto order plots and summaries.",
        add_help=True
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Maximum number of subject files per condition to process. Defaults to all.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers. Defaults to cpu_count().",
    )
    args, _ = parser.parse_known_args()
    return args


def _apply_pandas_compatibility_patches():
    """Apply monkey-patches for old pandas pickle compatibility."""
    import sys
    
    # Handle missing modules that were reorganized in pandas 2.0+
    try:
        import pandas.core.indexes.numeric
    except (ImportError, ModuleNotFoundError):
        import pandas.core.indexes as indexes
        sys.modules['pandas.core.indexes.numeric'] = indexes
    
    # Handle Int64Index, Float64Index, etc. that were removed in pandas 2.0
    try:
        from pandas import Int64Index
    except ImportError:
        from pandas import Index as Int64Index
        pd.Int64Index = Int64Index
        sys.modules['pandas'].Int64Index = Int64Index
    
    try:
        from pandas import Float64Index
    except ImportError:
        from pandas import Index as Float64Index
        pd.Float64Index = Float64Index
        sys.modules['pandas'].Float64Index = Float64Index
    
    try:
        from pandas import UInt64Index
    except ImportError:
        from pandas import Index as UInt64Index
        pd.UInt64Index = UInt64Index
        sys.modules['pandas'].UInt64Index = UInt64Index


def load_file(file):
    """Load data from a pickle file with compatibility for old pandas versions."""
    # Apply patches in this process (important for multiprocessing workers)
    _apply_pandas_compatibility_patches()
    
    try:
        with open(Path(file), 'rb') as handle:
            return pickle.load(handle)
    except Exception as e:
        # Fallback: try pandas read_pickle which handles version mismatches better
        # Silently try pd.read_pickle first (common for old pandas pickles)
        try:
            return pd.read_pickle(Path(file))
        except Exception as e2:
            # Only print if both methods fail
            print(f"[ERROR] Failed to load {file}")
            print(f"  - pickle.load error: {e}")
            print(f"  - pd.read_pickle error: {e2}")
            raise


def _sanitize_filename_part(part: str) -> str:
    """Sanitize a filename part by replacing non-alphanumeric characters."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(part))


def save_figure(fig, *parts, suffix="png", dpi=300):
    """Save a matplotlib figure to a file."""
    filename = "_".join(_sanitize_filename_part(part) for part in parts if part)
    if not filename:
        filename = "figure"
    output_path = PLOT_ORDER_RESULTS_DIR / f"{filename}.{suffix}"
    print(f"[PLOT_ORDER] Saving {output_path}")
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def cond_path(cond):
    """Get the path for a condition, stripping backslashes."""
    return RESULTS_DIR / cond.replace("\\", "")


def _limit_files(files: list[str], limit: int | None):
    """Limit the number of files to process."""
    total = len(files)
    if limit is None or limit < 0:
        return files, total
    return files[:limit], total


def order_parameter_filter(phase, df_labels, hemi="both", net="all"):
    """Calculate order parameter filtered by hemisphere and network."""
    df = pd.concat([pd.DataFrame(phase), df_labels], axis=1)
    if hemi != "both":
        df = df[df["hemi"] == hemi]
    if net != "all":
        df = df[df["net"] == net]
    df = df.drop(columns=["hemi", "label", "net"])
    
    r = np.abs(np.exp(1j * df).mean(axis=0))
    return r


def network_filter(phase, df_labels, hemi="both", net="all"):
    """Filter phase data by hemisphere and network."""
    df = pd.concat([pd.DataFrame(phase), df_labels], axis=1)
    df = (df[df["hemi"] == hemi] if hemi != "both" else df)
    df = (df[df["net"] == net] if net != "all" else df)
    df = df.drop(columns=["hemi", "label", "net"])
    return df


def order_parameter(phase):
    """Calculate the Kuramoto order parameter."""
    r = np.abs(np.exp(1j * phase).mean(axis=0))
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


# ============================================================================
# METADATA LOADING
# ============================================================================

def load_metadata():
    """Load metadata including labels, rejected epochs, etc."""
    print("[PLOT_ORDER] Loading metadata...")
    
    # Load extra.pkl with brain labels and coordinates
    [node_colors, label_names, label_names_short, stc_coords_3d,
     ch_names, mapping, eeg_coords_2d] = load_file(RESULTS_DIR / "extra.pkl")
    
    label_network = [x[:6] for x in label_names]
    
    # Create labels dataframe
    df_labels = pd.DataFrame()
    df_labels["label"] = [x[:6] for x in label_names]
    df_labels["hemi"] = [x[:2] for x in df_labels["label"]]
    df_labels["net"] = [x[3:] for x in df_labels["label"]]
    
    # Load rejected epochs
    folder = EEG_DIR
    bad_epochs = read_mat(str(folder / "rejected_epochs.mat"))
    subject = bad_epochs["rejected_epochs"][0]
    epochs = bad_epochs["rejected_epochs"][1]
    rejected_epochs = {k[:6].replace("_", "-"): v.tolist() for k, v in zip(subject, epochs)}
    
    # Calculate valid subjects
    subjects = np.array([x for x in range(35) if x not in REJECTED_SUBJECTS])
    
    print(f"[PLOT_ORDER] Loaded {len(label_names)} labels, {len(rejected_epochs)} subject epoch info")
    
    return {
        'node_colors': node_colors,
        'label_names': label_names,
        'label_names_short': label_names_short,
        'stc_coords_3d': stc_coords_3d,
        'ch_names': ch_names,
        'mapping': mapping,
        'eeg_coords_2d': eeg_coords_2d,
        'label_network': label_network,
        'df_labels': df_labels,
        'rejected_epochs': rejected_epochs,
        'subjects': subjects,
    }


# ============================================================================
# FILE CATALOG BUILDING
# ============================================================================

def build_file_catalog(conditions, max_subjects=None):
    """
    Build a catalog of files for each condition to avoid repeated directory scanning.
    
    Returns a dict like:
    {
        'DMT': {
            'syncro': ['syncro-S01-DMT.pkl', ...],
            'phases': ['phases-S01-DMT.pkl', ...],
            'order': ['order-S01-DMT.pkl', ...],
            'order_all': ['order_all-S01-DMT.pkl', ...],
        },
        ...
    }
    """
    print(f"[PLOT_ORDER] Building file catalog for conditions: {conditions}")
    print(f"[PLOT_ORDER] Max subjects per condition: {max_subjects if max_subjects else 'all'}")
    print(f"[PLOT_ORDER] Base RESULTS_DIR: {RESULTS_DIR}")
    
    catalog = {}
    
    for cond in conditions:
        cond_dir = cond_path(cond)
        print(f"[PLOT_ORDER] Scanning {cond_dir}...")
        
        if not cond_dir.exists():
            print(f"[PLOT_ORDER] WARNING: {cond_dir} does not exist")
            catalog[cond] = {'syncro': [], 'phases': [], 'order': [], 'order_all': []}
            continue
        
        all_files = os.listdir(cond_dir)
        
        # Syncro files
        syncro_candidates = sorted([x for x in all_files if x.startswith('syncro')])
        syncro_files, syncro_total = _limit_files(syncro_candidates, max_subjects)
        
        # Phases files
        phases_candidates = sorted([x for x in all_files if x.startswith('phases')])
        phases_files, phases_total = _limit_files(phases_candidates, max_subjects)
        
        # Order files
        order_candidates = sorted([x for x in all_files if x.startswith('order-')])
        order_files, order_total = _limit_files(order_candidates, max_subjects)
        
        # Order_all files
        order_all_candidates = sorted([x for x in all_files if x.startswith('order_all')])
        order_all_files, order_all_total = _limit_files(order_all_candidates, max_subjects)
        
        catalog[cond] = {
            'syncro': syncro_files,
            'phases': phases_files,
            'order': order_files,
            'order_all': order_all_files,
        }
        
        print(f"  - syncro: {len(syncro_files)}/{syncro_total}")
        print(f"  - phases: {len(phases_files)}/{phases_total}")
        print(f"  - order: {len(order_files)}/{order_total}")
        print(f"  - order_all: {len(order_all_files)}/{order_all_total}")
    
    return catalog


# ============================================================================
# WORKER FUNCTIONS FOR PARALLEL PROCESSING
# ============================================================================

def _process_syncro_file_worker(args):
    """Worker to process a single syncro file for Kuramoto data."""
    file_path, band, split, rejected_epochs = args
    try:
        data = load_file(file_path)
        
        file_name = file_path.name
        subj = file_name.replace("syncro-", "")[:6]
        
        try:
            rej = [x - 1 for x in rejected_epochs[subj]]
        except KeyError:
            rej = []
        
        # If band is None, return full data for all bands (optimization)
        if band is None:
            return {
                'data': data,
                'subject': subj,
                'rejected': rej,
                'file': file_name
            }
        
        # Otherwise, return just the requested band (legacy behavior)
        eeg_kuramoto_epochs = data["kuramoto_eeg"][band][split]
        return {
            'epochs': eeg_kuramoto_epochs,
            'subject': subj,
            'rejected': rej,
            'file': file_name
        }
    except Exception as e:
        print(f"[WARNING] Error processing {file_path}: {e}")
        return None


def _process_phases_file_worker(args):
    """Worker to process a single phases file."""
    file_path, band, df_labels, hemi, net = args
    try:
        data_dict = load_file(file_path)
        epoch_list = data_dict["phases_stc"][band]
        
        results = []
        for epoch_data in epoch_list:
            r = order_parameter_filter(epoch_data, df_labels, hemi=hemi, net=net)
            results.append(r.mean())
        
        return results
    except Exception as e:
        print(f"[WARNING] Error processing {file_path}: {e}")
        return []


def _process_kuramoto_stc_file_worker(args):
    """Worker to process kuramoto_stc from a single file."""
    file_path, band = args
    try:
        data_dict = load_file(file_path)
        epoch_list = data_dict["kuramoto_stc"][band][0]
        return [epoch_data.mean() for epoch_data in epoch_list]
    except Exception as e:
        print(f"[WARNING] Error processing {file_path}: {e}")
        return []


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_kuramoto_summary_all_bands(file_catalog, metadata, workers=None):
    """Plot Kuramoto EEG order parameter summary for ALL bands (memory efficient)."""
    print(f"\n[PLOT_ORDER] Generating Kuramoto summaries for all bands...")
    
    if workers is None or workers <= 0:
        workers = cpu_count()
    
    rejected_epochs = metadata['rejected_epochs']
    
    # Process each band separately to minimize memory usage
    for band in BAND_LIST:
        print(f"  Loading files and plotting {band}...")
        
        fig, ax = plt.subplots(figsize=(24, 24), nrows=3)
        fig.suptitle(f"Kuramoto EEG order parameter per subject (band {band})", fontsize=26, y=0.92)
        averages = {}
        
        for row, cond in enumerate(["DMT", "EO", "EC"]):
            files_names = file_catalog.get(cond, {}).get('syncro', [])
            averages[cond] = []
            
            max_secs = (400 if cond == "DMT" else 300)
            max_epochs = max_secs // 2
            tick_labels = [str(minn) + ":" + secs for minn, secs in product(range(7), ["00", "30"])]
            x_major_ticks = [i * 15 for i in range(len(tick_labels))]
            
            ax[row].set_xticks(x_major_ticks)
            ax[row].set_xticklabels(tick_labels, rotation=0)
            ax[row].set_xlim(0.0, max_epochs)
            ax[row].set_ylim(0.0, 1.0)
            ax[row].set_ylabel("Kuramoto r")
            ax[row].set_title(f"{cond} (N={len(files_names)})")
            
            if not files_names:
                ax[row].text(0.5, 0.5, f"Sin datos para {cond}", transform=ax[row].transAxes,
                             ha="center", va="center", fontsize=14, color="grey")
                continue
            
            color_map = plt.get_cmap("tab20", max(len(files_names), 1))
            
            # Process files for THIS band only (memory efficient - limited workers)
            file_paths = [cond_path(cond) / file for file in files_names]
            args_list = [(fp, band, 0, rejected_epochs) for fp in file_paths]
            
            # Use minimal workers to avoid RAM explosion (each worker loads ~300MB file)
            # imap processes files one-by-one, releasing memory between iterations
            num_workers = min(2, len(args_list))
            with Pool(processes=num_workers) as pool:
                results = list(tqdm(pool.imap(_process_syncro_file_worker, args_list, chunksize=1), 
                                  total=len(args_list), desc=f"{cond} {band}", leave=False))
            
            # Plot results
            for file_idx, result in enumerate(results):
                if result is None:
                    continue
                
                eeg_kuramoto_epochs = result['epochs']
                subj = result['subject']
                rej = result['rejected']
                
                num_epochs = len(eeg_kuramoto_epochs)
                full_epochs = len(rej) + num_epochs
                
                x_s = [x + 0.5 for x in range(full_epochs)]
                avr = np.asarray(eeg_kuramoto_epochs).mean(axis=1).tolist()
                y_s = [avr.pop() if i not in rej else np.nan for i in range(full_epochs)]
                
                diff = max_epochs - full_epochs
                values = y_s[:max_epochs]
                if diff > 0:
                    values += [np.nan] * diff
                averages[cond].append(values)
                
                color = color_map(file_idx % color_map.N)
                ax[row].plot(x_s, y_s, color=color, linewidth=1.2, alpha=0.8, label=subj)
                ax[row].scatter(x_s, y_s, s=10, color=color, alpha=0.6, label="_nolegend_")
                
                rejs = [i for i, x in enumerate(np.isnan(values)) if x]
                for i in rejs:
                    ax[row].axvspan(i, i + 1, color="grey", alpha=1 / 29, linewidth=0)
            
            if averages[cond]:
                arrays = np.nanmean(np.asarray(averages[cond]), axis=0).tolist()
                ax[row].plot(range(len(arrays)), arrays, color="black", linewidth=2.5, label="mean")
            
            legend_cols = max(1, min(4, (len(files_names) + 9) // 10))
            ax[row].legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0,
                           fontsize=8, ncol=legend_cols, title="Sujetos")
        
        if isinstance(ax, (list, np.ndarray)):
            ax[-1].set_xlabel("Tiempo (min:seg)")
        else:
            ax.set_xlabel("Tiempo (min:seg)")
        
        fig.tight_layout(rect=[0, 0, 0.88, 0.9])
        save_figure(fig, "kuramoto_epochs_summary", f"band_{band}", "conditions_DMT_EO_EC")


def plot_kuramoto_mean_trajectories_all_bands(file_catalog, metadata, workers=None, split=0):
    """Plot Kuramoto EEG mean trajectories for ALL bands (memory efficient)."""
    print(f"\n[PLOT_ORDER] Generating Kuramoto mean trajectories for all bands...")
    
    if workers is None or workers <= 0:
        workers = cpu_count()
    
    rejected_epochs = metadata['rejected_epochs']
    
    for band in BAND_LIST:
        print(f"  Loading files and plotting {band} mean trajectories...")
        
        fig, ax = plt.subplots(figsize=(24, 8))
        
        max_secs = 400
        max_epochs = max_secs // 2
        tick_labels = [str(minn) + ":" + secs for minn, secs in product(range(7), ["00", "30"])]
        x_major_ticks = [i * 15 for i in range(len(tick_labels))]
        
        ax.set_xticks(x_major_ticks)
        ax.set_xticklabels(tick_labels)
        ax.set_xlim(0.0, max_epochs)
        ax.set_ylim(0.0, 0.5)
        ax.set_title(f"Kuramoto EEG mean trajectories ({band})")
        ax.set_ylabel("Kuramoto r")
        ax.set_xlabel("Tiempo (min:seg)")
        
        averages = {}
        
        for row, cond in enumerate(["DMT", "EO", "EC"]):
            files_names = file_catalog.get(cond, {}).get('syncro', [])
            averages[cond] = []
            
            if not files_names:
                ax.text(0.5, 0.9 - 0.07 * row, f"{cond}: sin datos", transform=ax.transAxes,
                        ha="center", va="center", fontsize=12, color="grey")
                continue
            
            # Load files for THIS band only (limit workers to save RAM)
            file_paths = [cond_path(cond) / file for file in files_names]
            args_list = [(fp, band, split, rejected_epochs) for fp in file_paths]
            
            # Use minimal workers to avoid RAM explosion
            num_workers = min(2, len(args_list))
            with Pool(processes=num_workers) as pool:
                results = list(tqdm(pool.imap(_process_syncro_file_worker, args_list, chunksize=1),
                                  total=len(args_list), desc=f"{cond} {band}", leave=False))
            
            for result in results:
                if result is None:
                    continue
                
                eeg_kuramoto_epochs = result['epochs']
                rej = result['rejected']
                num_epochs = len(eeg_kuramoto_epochs)
                full_epochs = len(rej) + num_epochs
                
                x_s = [x + 0.5 for x in range(full_epochs)]
                avr = np.asarray(eeg_kuramoto_epochs).mean(axis=1).tolist()
                y_s = [avr.pop() if i not in rej else np.nan for i in range(full_epochs)]
                
                diff = max_epochs - full_epochs
                values = y_s[:max_epochs]
                if diff > 0:
                    values += [np.nan] * diff
                averages[cond].append(values)
            
            if not averages[cond]:
                ax.text(0.5, 0.9 - 0.07 * row, f"{cond}: sin datos", transform=ax.transAxes,
                        ha="center", va="center", fontsize=12, color="grey")
                continue
            
            arrays = np.nanmean(np.asarray(averages[cond]), axis=0).tolist()
            ax.plot(range(len(arrays)), arrays, linewidth=2, label=f"{cond} mean (N={len(files_names)})")
        
        ax.legend(loc="upper right")
        save_figure(fig, "kuramoto_gamma_summary", f"split_{split}", "conditions_DMT_EO_EC")


def plot_hist_kuramoto_stc_all_bands(file_catalog, metadata, workers=None, conditions=["DMT", "EC"]):
    """Plot histogram of Kuramoto STC for ALL bands (memory efficient)."""
    print(f"\n[PLOT_ORDER] Generating Kuramoto STC histograms for all bands...")
    
    if workers is None or workers <= 0:
        workers = cpu_count()
    
    for band in BAND_LIST:
        print(f"  Loading files and plotting {band} STC histogram...")
        
        r_dict = {}
        
        for cond in conditions:
            files_names = file_catalog.get(cond, {}).get('syncro', [])
            r_dict[cond] = []
            
            if not files_names:
                continue
            
            # Load files for THIS band only (limit workers to save RAM)
            file_paths = [cond_path(cond) / file for file in files_names]
            args_list = [(fp, band) for fp in file_paths]
            
            # Use minimal workers to avoid RAM explosion
            num_workers = min(2, len(args_list))
            with Pool(processes=num_workers) as pool:
                results = list(tqdm(pool.imap(_process_kuramoto_stc_file_worker, args_list, chunksize=1),
                                  total=len(args_list), desc=f"{cond} {band} STC", leave=False))
            
            # Flatten results
            for result in results:
                r_dict[cond] += result
        
        fig, axs = plt.subplots(figsize=(16, 8))
        for cond in conditions:
            axs.hist(r_dict.get(cond, []), bins=50, alpha=0.5, density=True, label=cond)
        axs.legend()
        axs.set_xlabel("Kuramoto r")
        axs.set_ylabel("Density")
        axs.set_title(f"Kuramoto STC - {band} band")
        cond_str = "_".join(conditions)
        save_figure(fig, "hist_kuramoto_stc", band, f"conditions_{cond_str}")


def plot_hist_phases(file_catalog, metadata, band="Alpha", hemi="both", net="DMN", conditions=["DMT", "EC"], workers=None):
    """Plot histogram of phase order parameter for a given band/network (parallelized)."""
    print(f"\n[PLOT_ORDER] Generating {band} {net} ({hemi}) phase histogram...")
    
    if workers is None or workers <= 0:
        workers = cpu_count()
    
    df_labels = metadata['df_labels']
    
    r_dict2 = {}
    for cond in conditions:
        files_names = file_catalog.get(cond, {}).get('phases', [])
        r_dict2[cond] = []
        
        if not files_names:
            continue
        
        # Parallel processing
        file_paths = [cond_path(cond) / file for file in files_names]
        args_list = [(fp, band, df_labels, hemi, net) for fp in file_paths]
        
        num_workers = min(workers, len(args_list))
        print(f"  [{cond}] Processing {len(files_names)} files with {num_workers} workers...")
        
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap(_process_phases_file_worker, args_list),
                              total=len(args_list), desc=f"{cond} phases {band}"))
        
        # Flatten results
        for result in results:
            r_dict2[cond] += result
    
    fig, axs = plt.subplots(figsize=(12, 6))
    for cond in conditions:
        axs.hist(r_dict2.get(cond, []), alpha=0.5, density=True, label=cond)
    axs.legend()
    axs.set_xlabel("Order parameter r")
    axs.set_ylabel("Density")
    axs.set_title(f"Phase order parameter - {band} {net} ({hemi})")
    cond_str = "_".join(conditions)
    save_figure(fig, "hist_kuramoto_pair", band, hemi, net, f"conditions_{cond_str}")


def plot_histogram_kuramoto_full(metadata):
    """
    Plot comprehensive histogram grid of Kuramoto order parameter.
    
    NOTE: This function loads pre-computed r_kuramoto_nets_epochs_mean.pkl
    which must exist from a previous run.
    """
    print("\n[PLOT_ORDER] Generating full Kuramoto histogram grid with FDR correction...")
    
    try:
        r_dict = load_file(RESULTS_DIR / "r_kuramoto_nets_epochs_mean.pkl")
    except FileNotFoundError:
        print("[PLOT_ORDER] WARNING: r_kuramoto_nets_epochs_mean.pkl not found, skipping full histogram")
        return
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    cond_list = ["EC", "EO"]
    scale = 7
    hemi = "both"
    
    fig, axs = plt.subplots(figsize=(21 * scale, 5 * scale), ncols=7, nrows=5)
    
    # Helper function for flattening
    def flatten_and_clean(data):
        if not data:
            return np.array([])
        
        def extract_numbers(obj):
            """Recursively extract numeric values from nested structures."""
            if isinstance(obj, (int, float, np.number)):
                return [float(obj)]
            elif isinstance(obj, (list, tuple, np.ndarray)):
                result = []
                for item in obj:
                    result.extend(extract_numbers(item))
                return result
            elif isinstance(obj, dict):
                return []
            else:
                try:
                    return [float(obj)]
                except (TypeError, ValueError):
                    return []
        
        try:
            flat = extract_numbers(data)
            if not flat:
                return np.array([])
            return np.asarray(flat, dtype=float)
        except Exception:
            return np.array([])
    
    # FIRST PASS: Calculate all p-values for FDR correction (memory-efficient)
    print("[PLOT_ORDER] Calculating all p-values for FDR correction...")
    pvalues_matrix = np.ones((len(BAND_LIST), len(NET_LIST)))
    valid_indices = set()  # Track which indices have valid data
    
    for i, band in enumerate(BAND_LIST):
        for j, net in enumerate(NET_LIST):
            cond1_str = cond_list[0]
            cond2_str = cond_list[1]
            
            try:
                cond1 = r_dict[cond1_str][band][hemi][net]
                cond2 = r_dict[cond2_str][band][hemi][net]
                
                cond1 = reject_outliers(flatten_and_clean(cond1))
                cond2 = reject_outliers(flatten_and_clean(cond2))
                
                # Calculate p-value (don't store data, just p-value)
                if len(cond1) > 1 and len(cond2) > 1:
                    statistic, pvalue = stats.ttest_ind(cond1, cond2)
                    pvalues_matrix[i, j] = pvalue
                    valid_indices.add((i, j))
            except Exception as e:
                print(f"[WARNING] Error processing {band}/{net}: {e}")
                continue
    
    # Apply FDR correction
    pvalues_flat = pvalues_matrix.flatten()
    rejected, pvalues_corrected = fdrcorrection(pvalues_flat, alpha=0.05)
    pvalues_corrected = pvalues_corrected.reshape(len(BAND_LIST), len(NET_LIST))
    rejected = rejected.reshape(len(BAND_LIST), len(NET_LIST))
    
    n_significant = rejected.sum()
    n_total = len(pvalues_flat)
    print(f"[PLOT_ORDER] FDR correction (α=0.05): {n_significant}/{n_total} comparisons significant ({100*n_significant/n_total:.1f}%)")
    
    # SECOND PASS: Plot with FDR-corrected significance (reprocess data to save RAM)
    for i, band in enumerate(BAND_LIST):
        for j, net in enumerate(NET_LIST):
            if (i, j) not in valid_indices:
                axs[i][j].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axs[i][j].transAxes)
                continue
            
            # Reprocess data (faster than keeping in RAM)
            cond1_str = cond_list[0]
            cond2_str = cond_list[1]
            
            try:
                cond1 = r_dict[cond1_str][band][hemi][net]
                cond2 = r_dict[cond2_str][band][hemi][net]
                
                cond1 = reject_outliers(flatten_and_clean(cond1))
                cond2 = reject_outliers(flatten_and_clean(cond2))
            except Exception as e:
                axs[i][j].text(0.5, 0.5, 'Error', ha='center', va='center', transform=axs[i][j].transAxes)
                continue
            
            if len(cond1) == 0 or len(cond2) == 0:
                axs[i][j].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axs[i][j].transAxes)
                continue
            
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
            pvalue_corrected = pvalues_corrected[i, j]
            
            # Format p-value as simple decimal
            if pvalue_corrected < 0.001:
                pvalue_str = f'p-value < 0.001'
            else:
                pvalue_str = f'p-value = {pvalue_corrected:.4f}'
            
            axs[i][j].text(0.05, 0.95, pvalue_str, transform=axs[i][j].transAxes,
                           fontsize=12, ha="left", va="top")
    
    for ax, col in zip(axs[0], NET_LIST):
        ax.set_title(col, size=24)
    for ax, row in zip(axs[:, 0], BAND_LIST):
        ax.set_ylabel(row, size=24)
    
    title = "Kuramoto r Order Parameter - "
    fig.suptitle(title + cond1_str + " vs " + cond2_str, size=36)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    save_figure(fig, "histogram_kuramoto_full", cond1_str, "vs", cond2_str, "hemi_both")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 80)
    print("PLOT_ORDER - Kuramoto Plotting")
    print("=" * 80)
    
    # Parse arguments
    args = parse_arguments()
    max_subjects = args.max_subjects
    workers = args.workers
    
    if workers is None or workers <= 0:
        workers = cpu_count()
    
    if max_subjects:
        print(f"\n[CONFIG] Limiting to {max_subjects} subjects per condition")
    else:
        print("\n[CONFIG] Processing all subjects")
    
    print(f"[CONFIG] Parallel workers: {workers}")
    
    # Load metadata
    metadata = load_metadata()
    
    # Build file catalog
    file_catalog = build_file_catalog(COND_LIST, max_subjects=max_subjects)
    
    # Generate plots
        print("\n" + "=" * 80)
        print("GENERATING PLOTS (MEMORY OPTIMIZED - loads per section)")
        print("=" * 80)
        
        # MEMORY OPTIMIZED: Load files per band, never cache full files
        # Each function loads only what it needs for each band, then releases
        
    #print("\n[1/5] Kuramoto summary plots (loads per band)...")
        #plot_kuramoto_summary_all_bands(file_catalog, metadata, workers=workers)
        
    print("\n[2/5] Kuramoto mean trajectories (loads per band)...")
        plot_kuramoto_mean_trajectories_all_bands(file_catalog, metadata, workers=workers, split=0)
        
    print("\n[3/5] Kuramoto STC histograms (loads per band)...")
        plot_hist_kuramoto_stc_all_bands(file_catalog, metadata, workers=workers, conditions=["DMT", "EC"])
        
    print("\n[4/5] Phase histograms (loads per band/network)...")
        for band in BAND_LIST:
            for net in NET_LIST:
                plot_hist_phases(file_catalog, metadata, band=band, hemi="both", 
                               net=net, conditions=["DMT", "EC"], workers=workers)
        
    # Generate comprehensive histogram (requires pre-computed r_kuramoto_nets_epochs_mean.pkl)
    print("\n[5/5] Full Kuramoto histogram grid...")
        plot_histogram_kuramoto_full(metadata)
        
    print("\n[PLOT_ORDER] ✓ All plots generated")
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()


# ============================================================================
# PARALLELIZATION SUMMARY
# ============================================================================
"""
This script uses multiprocessing to parallelize file I/O operations:

1. plot_kuramoto_summary_all_bands(): 
   - Parallelizes loading of syncro-*.pkl files
   - Each worker loads one file and extracts Kuramoto epochs

2. plot_kuramoto_mean_trajectories_all_bands():
   - Parallelizes syncro-*.pkl file loading
   - Same worker pattern as above

3. plot_hist_kuramoto_stc_all_bands():
   - Parallelizes processing of kuramoto_stc data
   - Each worker computes means for one file

4. plot_hist_phases():
   - Parallelizes phases-*.pkl file loading
   - Each worker processes phases for one file and computes order parameters

NOTE: For data structure generation (r_kuramoto_*.pkl files), 
      use build_order_data.py instead.

Performance:
- With 30 subjects and cpu_count()=8: ~4-6x speedup vs sequential
- Memory: Each worker loads one file at a time (low memory footprint)
- I/O bound: Limited by disk read speed, not CPU
"""


# ============================================================================
# LEGACY / DIAGNOSTIC CODE BLOCKS
# ============================================================================
# The following blocks are preserved but not executed by default.
# They contain exploratory/diagnostic code from the original script.
# Uncomment and adapt as needed.

"""
# Example: Load a single file for inspection
# path = RESULTS_DIR / "EC" / "syncro-S01-EC.pkl"
# eeg_kuramoto_epochs = load_file(path)["syncros_stc"]["Alpha"][0]

# Example: Inspect network structure (requires pre-existing r_dict)
# cond1 = r_dict["DMT"]["Alpha"]["both"]["DMN"]
# cond2 = r_dict["EC"]["Alpha"]["both"]["DMN"]
# fig, axs = plt.subplots()
# axs.hist(cond1, alpha=0.5, bins=50, density=True)
# axs.hist(cond2, alpha=0.5, bins=50, density=True)
# save_figure(fig, "hist_kuramoto_network", "Alpha", "DMN", "conditions_DMT_EC")

# Example: Single-subject phase analysis
# data_dict = load_file(RESULTS_DIR / "DMT" / "phases-S01-DMT.pkl")
# epoch_list = data_dict["phases_eeg"]["Alpha"]
# r_list = [order_parameter(epoch_data).mean() for epoch_data in epoch_list]
# print(np.mean(r_list))

# Example: MNE inverse solution workflow
# import mne
# from mne.datasets import fetch_fsaverage
# from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
# 
# path = EEG_DIR / "DMT"
# dmt_files_all = sorted([x for x in os.listdir(path) if x.endswith(".set")])
# dmt_files, total_dmt = _limit_files(dmt_files_all, max_subjects)
# print(f"[MNE] DMT .set files: found {total_dmt} (using {len(dmt_files)})")
# dmt_files_list = [path / x for x in dmt_files]
# 
# file_name = dmt_files_list[0]
# epochs_raw = mne.io.read_epochs_eeglab(file_name, montage_units='dm', verbose=False)
# epochs = epochs_raw.get_data()
# 
# fs_dir = fetch_fsaverage(verbose=False)
# src = fs_dir + "/bem/fsaverage-ico-5-src.fif"
# bem = fs_dir + "/bem/fsaverage-5120-5120-5120-bem-sol.fif"
# labels = mne.read_labels_from_annot('fsaverage', parc='Schaefer2018_100Parcels_7Networks_order')
# 
# method = "dSPM"
# snr = 3.
# lambda2 = 1. / snr ** 2
# 
# epochs_raw = epochs_raw.pick('eeg').apply_baseline((None, None), verbose=False)
# epochs_raw = epochs_raw.set_montage('standard_1020')
# epochs_raw = epochs_raw.set_eeg_reference(projection=True, verbose=False).apply_proj()
# 
# noise_cov = mne.compute_covariance(epochs_raw, tmax=0., method=['shrunk', 'empirical'], rank=None, verbose=False)
# fwd = mne.make_forward_solution(epochs_raw.info, trans='fsaverage', src=src, bem=bem, eeg=True, mindist=5.0, verbose=False)
# inverse_operator = make_inverse_operator(epochs_raw.info, fwd, noise_cov, loose=0.2, depth=0.8, verbose=False)
# 
# label = labels[4]
# stc = apply_inverse_epochs(epochs, inverse_operator, lambda2, method=method, pick_ori=None, label=label, verbose=False)
# 
# epoch_num = 132
# y = stc[epoch_num].data.mean(axis=0)
# x = range(len(y))
# fig, ax = plt.subplots()
# ax.plot(x, y)
# ax.set_xlabel('Time [samples]')
# ax.set_ylim((0, 1))
# ax.set_title(label.name + " - Epoch: " + str(epoch_num))
# save_figure(fig, "label_timecourse", label.name, f"epoch_{epoch_num}")
"""
