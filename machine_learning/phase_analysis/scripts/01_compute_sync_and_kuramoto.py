#!/usr/bin/env python3
"""
01_compute_sync_and_kuramoto.py

Step 1 of Phase Analysis Pipeline:
- Load pre-computed sync matrices and Kuramoto from phases-*.pkl files
- Concatenate multi-band sync vectors (fingerprint del estado cerebral)
- Save for clustering

Input format (phases-*.pkl):
    {
        "syncros_eeg": {"Delta": [array(24,24), ...], ...},
        "syncros_stc": {"Delta": [array(100,100), ...], ...},
        "kuramoto_eeg": {"Delta": [array(T), ...], ...},
        "kuramoto_stc": {"Delta": [array(T), ...], ...},
    }

Output:
    output/sync_matrices/
        sync_data.pkl         # Multi-band sync vectors + metadata
        kuramoto_data.pkl     # Kuramoto order parameter per epoch
    
Usage:
    python scripts/01_compute_sync_and_kuramoto.py
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import logging

import numpy as np
from tqdm import tqdm
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def vectorize_sync_matrix(sync_matrix: np.ndarray) -> np.ndarray:
    """
    Extract upper triangle of sync matrix as feature vector.
    
    Args:
        sync_matrix: [n_channels, n_channels] symmetric matrix
        
    Returns:
        [n_features] vector (upper triangle without diagonal)
    """
    n = sync_matrix.shape[0]
    indices = np.triu_indices(n, k=1)
    return sync_matrix[indices]


def load_phases_file(filepath: Path) -> Optional[Dict]:
    """Load phases-*.pkl file."""
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning(f"Error loading {filepath}: {e}")
        return None


def find_phase_files(phases_dir: Path, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Find all phases-*.pkl files.
    
    Returns:
        List of dicts with filepath, subject, condition
    """
    conditions = config['data']['conditions']
    subjects_filter = config['data'].get('subjects')
    
    files = []
    
    for condition in conditions:
        condition_dir = phases_dir / condition
        if not condition_dir.exists():
            logger.warning(f"Condition directory not found: {condition_dir}")
            continue
        
        # Find phases-*.pkl files
        for filepath in sorted(condition_dir.glob("phases-*.pkl")):
            # Extract subject ID from filename: phases-S01-DMT.pkl -> S01
            filename = filepath.stem  # phases-S01-DMT
            parts = filename.split('-')
            if len(parts) >= 2:
                subject_id = parts[1]  # S01
            else:
                subject_id = "unknown"
            
            # Filter by subject if specified
            if subjects_filter and subject_id not in subjects_filter:
                continue
            
            files.append({
                'filepath': filepath,
                'subject': subject_id,
                'condition': condition
            })
    
    return files


def process_phases_file(
    file_info: Dict[str, Any],
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Process a phases-*.pkl file and extract multi-band sync vectors.
    
    Returns:
        List of dicts with sync_vector, kuramoto, metadata per epoch
    """
    filepath = file_info['filepath']
    subject = file_info['subject']
    condition = file_info['condition']
    
    data = load_phases_file(filepath)
    if data is None:
        return []
    
    bands = config['data']['bands']
    use_stc = config['data'].get('use_stc', False)
    max_epochs = config['data'].get('max_epochs_per_file')
    
    # Select EEG or source space data
    suffix = '_stc' if use_stc else '_eeg'
    syncros_key = f'syncros{suffix}'
    kuramoto_key = f'kuramoto{suffix}'
    
    if syncros_key not in data:
        logger.warning(f"Key '{syncros_key}' not found in {filepath}")
        return []
    
    syncros = data[syncros_key]
    kuramoto = data.get(kuramoto_key, {})
    
    # Determine number of epochs (from first band)
    first_band = bands[0]
    if first_band not in syncros:
        logger.warning(f"Band '{first_band}' not found in {filepath}")
        return []
    
    n_epochs = len(syncros[first_band])
    if max_epochs:
        n_epochs = min(n_epochs, max_epochs)
    
    # Check all bands have same number of epochs
    for band in bands:
        if band not in syncros:
            logger.warning(f"Band '{band}' missing in {filepath}")
            return []
        if len(syncros[band]) < n_epochs:
            n_epochs = len(syncros[band])
    
    results = []
    
    for epoch_idx in range(n_epochs):
        try:
            # Collect sync vectors and kuramoto for each band
            band_sync_vectors = []
            band_kuramoto = []
            
            for band in bands:
                # Sync matrix for this epoch/band
                sync_matrix = syncros[band][epoch_idx]
                sync_vector = vectorize_sync_matrix(sync_matrix)
                band_sync_vectors.append(sync_vector)
                
                # Kuramoto (mean of timeseries)
                if band in kuramoto and epoch_idx < len(kuramoto[band]):
                    k_ts = kuramoto[band][epoch_idx]
                    k_mean = float(np.mean(k_ts))
                else:
                    k_mean = 0.0
                band_kuramoto.append(k_mean)
            
            # Concatenate multi-band features
            multiband_vector = np.concatenate(band_sync_vectors)
            kuramoto_mean = np.mean(band_kuramoto)
            
            results.append({
                'sync_vector': multiband_vector,
                'kuramoto_mean': kuramoto_mean,
                'kuramoto_per_band': dict(zip(bands, band_kuramoto)),
                'epoch_idx': epoch_idx,
                'subject': subject,
                'condition': condition
            })
            
        except Exception as e:
            logger.warning(f"Error processing epoch {epoch_idx} in {filepath}: {e}")
            continue
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Load pre-computed sync matrices and Kuramoto')
    parser.add_argument('--config', type=str, 
                       default='config/config.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    
    # Change to script directory for relative paths
    script_dir = Path(__file__).parent.parent
    config_path = script_dir / args.config
    
    logger.info("=" * 70)
    logger.info("PHASE ANALYSIS - Step 1: Load Multi-Band Sync & Kuramoto")
    logger.info("=" * 70)
    
    # Load config
    config = load_config(config_path)
    bands = config['data']['bands']
    use_stc = config['data'].get('use_stc', False)
    
    logger.info(f"Config: {config_path}")
    logger.info(f"Mode: {'Source Space (STC)' if use_stc else 'EEG Electrodes'}")
    logger.info(f"Bands: {bands} (will be concatenated)")
    
    # Setup paths
    phases_dir = Path(config['paths']['phases_dir'])
    output_dir = Path(config['paths']['output_dir']) / 'sync_matrices'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Phases dir: {phases_dir}")
    logger.info(f"Output dir: {output_dir}")
    
    # Find all phase files
    files = find_phase_files(phases_dir, config)
    logger.info(f"Found {len(files)} phase files")
    
    if len(files) == 0:
        logger.error("No files found! Check paths and configuration.")
        sys.exit(1)
    
    # Process all files
    all_data = []
    
    for file_info in tqdm(files, desc="Processing files"):
        results = process_phases_file(file_info, config)
        all_data.extend(results)
    
    logger.info(f"Processed {len(all_data)} epochs total")
    
    if len(all_data) == 0:
        logger.error("No epochs processed! Check data files.")
        sys.exit(1)
    
    # Organize data for saving
    sync_vectors = np.array([d['sync_vector'] for d in all_data])
    n_features_per_band = sync_vectors.shape[1] // len(bands)
    
    metadata = {
        'subjects': [d['subject'] for d in all_data],
        'conditions': [d['condition'] for d in all_data],
        'epoch_indices': [d['epoch_idx'] for d in all_data],
    }
    
    kuramoto_means = np.array([d['kuramoto_mean'] for d in all_data])
    kuramoto_per_band = {
        band: np.array([d['kuramoto_per_band'][band] for d in all_data])
        for band in bands
    }
    
    # Infer n_channels from feature count
    # n_features = n_channels * (n_channels - 1) / 2
    # Solving: n^2 - n - 2*features = 0
    n_features = n_features_per_band
    n_channels = int((1 + np.sqrt(1 + 8 * n_features)) / 2)
    
    # Save sync data
    sync_data = {
        'sync_vectors': sync_vectors,
        'metadata': metadata,
        'n_channels': n_channels,
        'n_epochs': len(all_data),
        'n_features_per_band': n_features_per_band,
        'n_features_total': sync_vectors.shape[1],
        'conditions': config['data']['conditions'],
        'bands': bands,
        'multiband': True,
        'use_stc': use_stc,
    }
    
    sync_path = output_dir / 'sync_data.pkl'
    with open(sync_path, 'wb') as f:
        pickle.dump(sync_data, f)
    logger.info(f"Saved sync data to {sync_path}")
    
    # Save Kuramoto data
    kuramoto_data = {
        'kuramoto_means': kuramoto_means,
        'kuramoto_per_band': kuramoto_per_band,
        'metadata': metadata,
        'n_epochs': len(all_data),
        'bands': bands,
    }
    
    kuramoto_path = output_dir / 'kuramoto_data.pkl'
    with open(kuramoto_path, 'wb') as f:
        pickle.dump(kuramoto_data, f)
    logger.info(f"Saved Kuramoto data to {kuramoto_path}")
    
    # Print summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total epochs: {len(all_data)}")
    logger.info(f"Channels: {n_channels}")
    logger.info(f"Features per band: {n_features_per_band}")
    logger.info(f"Total features (multi-band): {sync_vectors.shape[1]}")
    logger.info(f"  = {len(bands)} bands × {n_features_per_band} features")
    
    # Per condition
    logger.info("")
    logger.info("[By Condition]")
    for condition in config['data']['conditions']:
        mask = np.array(metadata['conditions']) == condition
        n = mask.sum()
        if n > 0:
            k_mean = kuramoto_means[mask].mean()
            k_std = kuramoto_means[mask].std()
            logger.info(f"  {condition}: {n} epochs, Kuramoto = {k_mean:.3f} ± {k_std:.3f}")
    
    # Kuramoto per band
    logger.info("")
    logger.info("[Kuramoto by Band]")
    for band in bands:
        k_mean = kuramoto_per_band[band].mean()
        k_std = kuramoto_per_band[band].std()
        logger.info(f"  {band}: {k_mean:.3f} ± {k_std:.3f}")
    
    # Subjects
    unique_subjects = sorted(set(metadata['subjects']))
    logger.info("")
    logger.info(f"[Subjects: {len(unique_subjects)}]")
    logger.info(f"  {', '.join(unique_subjects[:10])}{'...' if len(unique_subjects) > 10 else ''}")
    
    logger.info("")
    logger.info("Step 1 complete! Next: python scripts/02_cluster_sync_matrices.py")


if __name__ == '__main__':
    main()
