"""
Data loading utilities for spectral sources.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def load_aal_atlas(data_dir: str) -> pd.DataFrame:
    """
    Load AAL90 atlas with region labels and coordinates.
    
    Returns:
        DataFrame with columns: Index, Label, Abr, x, y, z
    """
    path = Path(data_dir) / "AAL90.csv"
    df = pd.read_csv(path, delimiter=";", index_col=0)
    logger.info(f"Loaded AAL90 atlas with {len(df)} regions")
    return df


def load_targets(data_dir: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load target variables (experience scores).
    
    Returns:
        Tuple of (targets DataFrame, list of target names)
    """
    data_dir = Path(data_dir)
    
    # Load labels
    labels_path = data_dir / "target_labels.txt"
    labels = pd.read_csv(labels_path, header=None)[0].tolist()
    
    # Load targets
    targets_path = data_dir / "target.csv"
    targets = pd.read_csv(targets_path, header=None, names=labels)
    
    # Drop last row if empty
    if targets.iloc[-1].isna().all():
        targets = targets.iloc[:-1]
    
    logger.info(f"Loaded {len(targets)} subjects with {len(labels)} target variables")
    return targets, labels


def load_spectral_data(data_dir: str, 
                       bands: List[str],
                       aal_labels: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """
    Load spectral power data for specified frequency bands.
    
    Args:
        data_dir: Path to spectral_sources directory
        bands: List of band names (e.g., ["alpha", "beta", "DMT_alpha"])
        aal_labels: Optional list of AAL region names for column headers
    
    Returns:
        Dict mapping band name to DataFrame (subjects × regions)
    """
    data_dir = Path(data_dir)
    data = {}
    
    for band in bands:
        path = data_dir / f"{band}.csv"
        if not path.exists():
            logger.warning(f"Band file not found: {path}")
            continue
        
        df = pd.read_csv(path, header=None)
        
        # Drop last row if empty
        if df.iloc[-1].isna().all():
            df = df.iloc[:-1]
        
        # Set column names if provided
        if aal_labels is not None and len(aal_labels) == df.shape[1]:
            df.columns = aal_labels
        
        data[band] = df
        logger.info(f"Loaded {band}: {df.shape[0]} subjects × {df.shape[1]} regions")
    
    return data


def build_feature_matrix(data_dir: str,
                        use_baseline: bool = True,
                        use_dmt: bool = True,
                        use_difference: bool = True) -> Tuple[np.ndarray, List[str]]:
    """
    Build combined feature matrix from spectral data.
    
    Args:
        data_dir: Path to spectral_sources directory
        use_baseline: Include baseline (pre-DMT) bands
        use_dmt: Include during-DMT bands
        use_difference: Include DMT - Baseline difference
    
    Returns:
        Tuple of (feature matrix [N, D], feature names)
    """
    data_dir = Path(data_dir)
    
    # Load AAL labels
    aal = load_aal_atlas(data_dir)
    aal_labels = aal["Label"].tolist()
    
    baseline_bands = ["alpha", "beta", "delta", "theta", "gamma1", "gamma2"]
    dmt_bands = ["DMT_alpha", "DMT_beta", "DMT_delta", "DMT_theta", "DMT_gamma1", "DMT_gamma2"]
    
    features = []
    feature_names = []
    
    if use_baseline:
        baseline_data = load_spectral_data(data_dir, baseline_bands, aal_labels)
        for band_name, df in baseline_data.items():
            features.append(df.values)
            feature_names.extend([f"{band_name}_{region}" for region in aal_labels])
    
    if use_dmt:
        dmt_data = load_spectral_data(data_dir, dmt_bands, aal_labels)
        for band_name, df in dmt_data.items():
            features.append(df.values)
            feature_names.extend([f"{band_name}_{region}" for region in aal_labels])
    
    if use_difference and use_baseline and use_dmt:
        baseline_data = load_spectral_data(data_dir, baseline_bands, aal_labels)
        dmt_data = load_spectral_data(data_dir, dmt_bands, aal_labels)
        
        for base_band, dmt_band in zip(baseline_bands, dmt_bands):
            if base_band in baseline_data and dmt_band in dmt_data:
                diff = dmt_data[dmt_band].values - baseline_data[base_band].values
                features.append(diff)
                band_short = base_band.replace("_", "")
                feature_names.extend([f"diff_{band_short}_{region}" for region in aal_labels])
    
    X = np.concatenate(features, axis=1)
    logger.info(f"Built feature matrix: {X.shape[0]} subjects × {X.shape[1]} features")
    
    return X, feature_names


def compute_distance_matrix(aal_df: pd.DataFrame) -> np.ndarray:
    """
    Compute pairwise Euclidean distance matrix between AAL regions.
    
    Args:
        aal_df: DataFrame with x, y, z coordinates
    
    Returns:
        Distance matrix [90, 90]
    """
    coords = aal_df[['x', 'y', 'z']].values
    
    # Compute pairwise distances
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=2))
    
    return distances


def build_adjacency_from_distance(distance_matrix: np.ndarray,
                                  threshold: float = 50.0,
                                  k_neighbors: Optional[int] = None) -> np.ndarray:
    """
    Build adjacency matrix from distance matrix.
    
    Args:
        distance_matrix: Pairwise distances [N, N]
        threshold: Distance threshold for connectivity
        k_neighbors: If set, connect to k nearest neighbors instead
    
    Returns:
        Binary adjacency matrix [N, N]
    """
    n = distance_matrix.shape[0]
    
    if k_neighbors is not None:
        # K-nearest neighbors
        adj = np.zeros((n, n))
        for i in range(n):
            # Get indices of k nearest (excluding self)
            distances_i = distance_matrix[i].copy()
            distances_i[i] = np.inf
            nearest = np.argsort(distances_i)[:k_neighbors]
            adj[i, nearest] = 1
        # Make symmetric
        adj = np.maximum(adj, adj.T)
    else:
        # Distance threshold
        adj = (distance_matrix < threshold).astype(float)
        np.fill_diagonal(adj, 0)
    
    logger.info(f"Built adjacency matrix: {int(adj.sum())} edges")
    return adj

