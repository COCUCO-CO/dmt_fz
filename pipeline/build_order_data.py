#!/usr/bin/env python3
"""
build_order_data.py - Build Kuramoto order parameter data structures

Generates pickle files with aggregated Kuramoto synchronization metrics
that are used by plot_order.py for visualization.

Output files:
- r_kuramoto_nets_epochs_mean.pkl: Mean order parameters per epoch/band/network
- r_kuramoto_nets_all_mean.pkl: Order parameters for network pair combinations
"""

import argparse
import os
import pickle
import re
import sys
from copy import deepcopy
from itertools import combinations_with_replacement as combs
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from pymatreader import read_mat
from tqdm import tqdm

from paths import BASE_DIR, RESULTS_DIR, EEG_CLEAN_DIR as EEG_DIR, ensure_dir

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

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
        description="Build Kuramoto order parameter data structures.",
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
    parser.add_argument(
        "--build-epochs-mean",
        action="store_true",
        help="Build r_kuramoto_nets_epochs_mean.pkl",
    )
    parser.add_argument(
        "--build-all-mean",
        action="store_true",
        help="Build r_kuramoto_nets_all_mean.pkl",
    )
    parser.add_argument(
        "--build-all",
        action="store_true",
        help="Build all data files (equivalent to --build-epochs-mean --build-all-mean)",
    )
    args, _ = parser.parse_known_args()
    return args


def save_file(data, folder, file):
    """Save data to a pickle file."""
    folder = ensure_dir(folder)
    with open(folder / f"{file}.pkl", 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _apply_pandas_compatibility_patches():
    """Apply monkey-patches for old pandas pickle compatibility."""
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
        try:
            return pd.read_pickle(Path(file))
        except Exception as e2:
            print(f"[ERROR] Failed to load {file}")
            print(f"  - pickle.load error: {e}")
            print(f"  - pd.read_pickle error: {e2}")
            raise


def cond_path(cond):
    """Get the path for a condition, stripping backslashes."""
    return RESULTS_DIR / cond.replace("\\", "")


def _limit_files(files: list[str], limit: int | None):
    """Limit the number of files to process."""
    total = len(files)
    if limit is None or limit < 0:
        return files, total
    return files[:limit], total


def order_parameter(phase):
    """Calculate the Kuramoto order parameter."""
    r = np.abs(np.exp(1j * phase).mean(axis=0))
    return r


# ============================================================================
# METADATA LOADING
# ============================================================================

def load_metadata():
    """Load metadata including labels, rejected epochs, etc."""
    print("[BUILD_ORDER_DATA] Loading metadata...")
    
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
    
    print(f"[BUILD_ORDER_DATA] Loaded {len(label_names)} labels, {len(rejected_epochs)} subject epoch info")
    
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
    """
    print(f"[BUILD_ORDER_DATA] Building file catalog for conditions: {conditions}")
    print(f"[BUILD_ORDER_DATA] Max subjects per condition: {max_subjects if max_subjects else 'all'}")
    
    catalog = {}
    
    for cond in conditions:
        cond_dir = cond_path(cond)
        print(f"[BUILD_ORDER_DATA] Scanning {cond_dir}...")
        
        if not cond_dir.exists():
            print(f"[BUILD_ORDER_DATA] WARNING: {cond_dir} does not exist")
            catalog[cond] = {'syncro': [], 'phases': [], 'order': [], 'order_all': []}
            continue
        
        all_files = os.listdir(cond_dir)
        
        # Order files (main input for data building)
        order_candidates = sorted([x for x in all_files if x.startswith('order-')])
        order_files, order_total = _limit_files(order_candidates, max_subjects)
        
        # Order_all files
        order_all_candidates = sorted([x for x in all_files if x.startswith('order_all')])
        order_all_files, order_all_total = _limit_files(order_all_candidates, max_subjects)
        
        catalog[cond] = {
            'order': order_files,
            'order_all': order_all_files,
        }
        
        print(f"  - order: {len(order_files)}/{order_total}")
        print(f"  - order_all: {len(order_all_files)}/{order_all_total}")
    
    return catalog


# ============================================================================
# DATA BUILDING FUNCTIONS
# ============================================================================

def build_r_kuramoto_nets_epochs_mean(file_catalog, metadata):
    """
    Build r_kuramoto_nets_epochs_mean structure.
    
    This function processes 'order' files and computes mean order parameters
    across epochs for all conditions, bands, hemispheres, and networks.
    
    Output: RESULTS_DIR / r_kuramoto_nets_epochs_mean.pkl
    """
    print("\n[BUILD_ORDER_DATA] Building r_kuramoto_nets_epochs_mean...")
    
    r_dict = {}
    for cond in deepcopy(COND_LIST):
        r_dict[cond] = {}
        for band in deepcopy(BAND_LIST):
            r_dict[cond][band] = {}
            for hemi in deepcopy(HEMI_LIST):
                r_dict[cond][band][hemi] = {net: [] for net in deepcopy(NET_LIST)}
    
    for cond in COND_LIST:
        print(f"  Processing {cond}...")
        files_names = file_catalog.get(cond, {}).get('order', [])
        
        if not files_names:
            print(f"  [WARN] No order files found for {cond}")
            continue
        
        for file in tqdm(files_names, desc=f"{cond} order mean"):
            file_path = cond_path(cond) / file
            data_dict = load_file(file_path)
            for band in BAND_LIST:
                for hemi in HEMI_LIST:
                    for net in NET_LIST:
                        data_r = pd.DataFrame(data_dict[cond][band][hemi][net])
                        r_dict[cond][band][hemi][net].append(data_r.mean(axis=1).tolist())
    
    save_file(r_dict, RESULTS_DIR, "r_kuramoto_nets_epochs_mean")
    print(f"[BUILD_ORDER_DATA] Saved {RESULTS_DIR / 'r_kuramoto_nets_epochs_mean.pkl'}")


def _process_order_file_worker(args):
    """Worker function for parallel processing of order files."""
    cond, file_path = args
    data_dict = load_file(file_path)
    result = {}
    
    for band in BAND_LIST:
        for hemi in HEMI_LIST:
            for (net1, net2) in combs(NET_LIST, 2):
                key = (band, hemi, net1, net2)
                epochs_data = []
                net1_epochs = data_dict[cond][band][hemi][net1]
                net2_epochs = data_dict[cond][band][hemi][net2]
                num_epochs = len(net1_epochs)
                for epoch in range(num_epochs):
                    data_r1 = pd.DataFrame(net1_epochs[epoch])
                    data_r2 = pd.DataFrame(net2_epochs[epoch])
                    data_r = order_parameter(pd.concat([data_r1, data_r2]))
                    epochs_data.append(float(data_r.mean()))
                result[key] = epochs_data
    return cond, result


def build_kuramoto_all_mean(file_catalog, workers=None):
    """
    Build r_kuramoto_nets_all_mean using parallel processing.
    
    This processes 'order' files across multiple workers.
    
    Output: RESULTS_DIR / r_kuramoto_nets_all_mean.pkl
    """
    print(f"\n[BUILD_ORDER_DATA] Generating r_kuramoto_nets_all_mean (workers={workers})")
    
    arg_list = []
    for cond in COND_LIST:
        files_selected = file_catalog.get(cond, {}).get('order', [])
        if not files_selected:
            print(f"[BUILD_ORDER_DATA] No order files found for condition {cond}")
            continue
        cond_files = [cond_path(cond) / file for file in files_selected]
        arg_list.extend((cond, file_path) for file_path in cond_files)
    
    if not arg_list:
        print("[BUILD_ORDER_DATA] No files to process.")
        return
    
    if workers is None or workers <= 0:
        workers = cpu_count()
    workers = min(workers, len(arg_list))
    print(f"[BUILD_ORDER_DATA] Using {workers} worker(s) for Pool across {len(arg_list)} file(s)")
    
    r_dict = {cond: {band: {hemi: {net1: {net2: [] for net2 in NET_LIST} for net1 in NET_LIST}
                            for hemi in HEMI_LIST} for band in BAND_LIST} for cond in COND_LIST}
    
    with Pool(processes=workers) as pool:
        for cond, partial in tqdm(pool.imap_unordered(_process_order_file_worker, arg_list),
                                   total=len(arg_list), desc="Processing order"):
            for (band, hemi, net1, net2), epochs_data in partial.items():
                r_dict[cond][band][hemi][net1][net2].append(epochs_data)
    
    save_file(r_dict, RESULTS_DIR, "r_kuramoto_nets_all_mean")
    print(f"[BUILD_ORDER_DATA] Saved {RESULTS_DIR / 'r_kuramoto_nets_all_mean.pkl'}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 80)
    print("BUILD_ORDER_DATA - Kuramoto Data Structure Builder")
    print("=" * 80)
    
    # Parse arguments
    args = parse_arguments()
    max_subjects = args.max_subjects
    workers = args.workers
    
    # Determine what to build
    build_epochs_mean = args.build_epochs_mean or args.build_all
    build_all_mean = args.build_all_mean or args.build_all
    
    # If no specific build flag, show help
    if not build_epochs_mean and not build_all_mean:
        print("\n[BUILD_ORDER_DATA] No build target specified.")
        print("\nUsage:")
        print("  python build_order_data.py --build-epochs-mean  # Build r_kuramoto_nets_epochs_mean.pkl")
        print("  python build_order_data.py --build-all-mean     # Build r_kuramoto_nets_all_mean.pkl")
        print("  python build_order_data.py --build-all          # Build all data files")
        print("\nOptional:")
        print("  --max-subjects N  # Limit subjects per condition")
        print("  --workers N       # Number of parallel workers")
        return
    
    if workers is None or workers <= 0:
        workers = cpu_count()
    
    print(f"\n[CONFIG] Max subjects per condition: {max_subjects if max_subjects else 'all'}")
    print(f"[CONFIG] Parallel workers: {workers}")
    print(f"[CONFIG] Build epochs mean: {build_epochs_mean}")
    print(f"[CONFIG] Build all mean: {build_all_mean}")
    
    # Load metadata
    metadata = load_metadata()
    
    # Build file catalog
    file_catalog = build_file_catalog(COND_LIST, max_subjects=max_subjects)
    
    # Build data structures
    print("\n" + "=" * 80)
    print("BUILDING DATA STRUCTURES")
    print("=" * 80)
    
    if build_epochs_mean:
        build_r_kuramoto_nets_epochs_mean(file_catalog, metadata)
    
    if build_all_mean:
        build_kuramoto_all_mean(file_catalog, workers=workers)
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

