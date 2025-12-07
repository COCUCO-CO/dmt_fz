"""
Enhanced Dataset Builder for VAE with multiple input variations.

Supports:
- Multi-band simultaneous processing (all bands as channels)
- Temporal windows (sequences of K consecutive epochs)
- Direct phase matrices (24×T instead of statistics)
- Temporal differences (Δ synchronization between epochs)
- Source-localized data (68 regions)

Author: Assistant
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import multiprocessing as mp
from functools import partial

import numpy as np
import torch
from torch_geometric.data import Data
from scipy.io import loadmat
from scipy import stats
from tqdm import tqdm

logger = logging.getLogger(__name__)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_phases_file(filepath: Path) -> Optional[Dict]:
    """Load a phases .mat or .pkl file."""
    try:
        if filepath.suffix == '.mat':
            data = loadmat(str(filepath), squeeze_me=True)
            # Extract phases array
            if 'phases' in data:
                return {'phases': data['phases']}
            elif 'phase' in data:
                return {'phases': data['phase']}
            return None
        elif filepath.suffix == '.pkl':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        return None
    except Exception as e:
        logger.warning(f"Failed to load {filepath}: {e}")
        return None


def compute_sync_matrix(phases: np.ndarray) -> np.ndarray:
    """
    Compute synchronization matrix from phase data using PLV.
    
    Args:
        phases: [n_channels, n_timepoints] phase array
        
    Returns:
        [n_channels, n_channels] synchronization matrix
    """
    n_channels = phases.shape[0]
    sync_matrix = np.zeros((n_channels, n_channels))
    
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            # Phase Locking Value
            phase_diff = phases[i] - phases[j]
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            sync_matrix[i, j] = plv
            sync_matrix[j, i] = plv
    
    return sync_matrix


def compute_kuramoto(phases: np.ndarray) -> float:
    """Compute Kuramoto order parameter."""
    n_channels, n_time = phases.shape
    # Complex order parameter
    z = np.mean(np.exp(1j * phases), axis=0)
    r = np.abs(z)
    return float(np.mean(r))


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_phase_statistics(phases: np.ndarray) -> np.ndarray:
    """
    Extract statistical features from phase time series.
    
    Args:
        phases: [n_channels, n_timepoints]
        
    Returns:
        [n_channels, n_features] feature array
    """
    n_channels = phases.shape[0]
    features = []
    
    for ch in range(n_channels):
        ch_phases = phases[ch]
        
        # Circular statistics
        mean_phase = np.angle(np.mean(np.exp(1j * ch_phases)))
        phase_std = np.std(ch_phases)
        
        # Phase velocity
        phase_velocity = np.diff(ch_phases)
        mean_velocity = np.mean(phase_velocity)
        std_velocity = np.std(phase_velocity)
        
        # Phase acceleration
        phase_accel = np.diff(phase_velocity)
        mean_accel = np.mean(phase_accel) if len(phase_accel) > 0 else 0
        
        # Circular variance
        r = np.abs(np.mean(np.exp(1j * ch_phases)))
        circular_var = 1 - r
        
        features.append([
            mean_phase, phase_std, mean_velocity, std_velocity,
            mean_accel, circular_var, r
        ])
    
    return np.array(features)


def extract_direct_phase_features(phases: np.ndarray, 
                                   target_length: int = 64) -> np.ndarray:
    """
    Extract direct phase matrix features (resampled to fixed length).
    
    Args:
        phases: [n_channels, n_timepoints]
        target_length: Fixed temporal dimension
        
    Returns:
        [n_channels, target_length] resampled phases
    """
    n_channels, n_time = phases.shape
    
    if n_time == target_length:
        return phases
    
    # Resample to target length
    indices = np.linspace(0, n_time - 1, target_length).astype(int)
    return phases[:, indices]


def compute_temporal_complexity(phases: np.ndarray) -> np.ndarray:
    """
    Compute temporal complexity features per channel.
    
    Returns:
        [n_channels, n_features] complexity features
    """
    n_channels = phases.shape[0]
    features = []
    
    for ch in range(n_channels):
        ch_phases = phases[ch]
        
        # Approximate entropy (simplified)
        diff = np.abs(np.diff(ch_phases))
        entropy_approx = np.std(diff) / (np.mean(diff) + 1e-8)
        
        # Hurst exponent approximation (R/S analysis simplified)
        n = len(ch_phases)
        if n > 10:
            half = n // 2
            r1 = np.max(ch_phases[:half]) - np.min(ch_phases[:half])
            r2 = np.max(ch_phases[half:]) - np.min(ch_phases[half:])
            s1 = np.std(ch_phases[:half])
            s2 = np.std(ch_phases[half:])
            hurst = np.log((r1/s1 + r2/s2) / 2 + 1e-8) / np.log(2)
        else:
            hurst = 0.5
        
        # Sample entropy approximation
        sample_entropy = -np.log(np.mean(np.abs(np.diff(diff))) + 1e-8)
        
        features.append([entropy_approx, hurst, sample_entropy])
    
    return np.array(features)


# =============================================================================
# GRAPH BUILDING
# =============================================================================

def build_enhanced_graph(
    phases: np.ndarray,
    condition: str,
    band: str,
    subject_id: str,
    epoch_idx: int,
    config: Dict[str, Any],
    prev_sync_matrix: Optional[np.ndarray] = None
) -> Data:
    """
    Build enhanced graph with multiple feature types.
    
    Args:
        phases: [n_channels, n_timepoints] phase data
        condition: Condition name (DMT, EC, EO)
        band: Frequency band name
        subject_id: Subject identifier
        epoch_idx: Epoch index
        config: Configuration dictionary
        prev_sync_matrix: Previous epoch's sync matrix (for temporal diff)
        
    Returns:
        PyG Data object
    """
    n_channels = phases.shape[0]
    node_config = config['data']['node_features']
    graph_config = config['data']['graph']
    
    # Compute synchronization matrix
    sync_matrix = compute_sync_matrix(phases)
    
    # ==========================================================================
    # NODE FEATURES
    # ==========================================================================
    
    node_features_list = []
    
    # 1. Phase statistics
    if node_config.get('use_phase_stats', True):
        phase_stats = extract_phase_statistics(phases)
        node_features_list.append(phase_stats)
    
    # 2. Direct phase matrix (flattened or as separate feature)
    if node_config.get('use_direct_phases', False):
        target_len = node_config.get('phase_target_length', 64)
        direct_phases = extract_direct_phase_features(phases, target_len)
        node_features_list.append(direct_phases)
    
    # 3. Temporal complexity
    if node_config.get('use_temporal_complexity', True):
        complexity = compute_temporal_complexity(phases)
        node_features_list.append(complexity)
    
    # 4. Local connectivity features
    if node_config.get('use_local_connectivity', True):
        # Node degree (sum of sync weights)
        degree = sync_matrix.sum(axis=1, keepdims=True)
        # Clustering coefficient approximation
        clustering = (sync_matrix ** 2).sum(axis=1, keepdims=True) / (degree + 1e-8)
        node_features_list.append(np.hstack([degree, clustering]))
    
    # Concatenate node features
    x = np.hstack(node_features_list) if node_features_list else np.zeros((n_channels, 1))
    
    # ==========================================================================
    # EDGE FEATURES
    # ==========================================================================
    
    # Build edges based on config
    if graph_config.get('fully_connected', True):
        # All pairs
        src, dst = [], []
        for i in range(n_channels):
            for j in range(n_channels):
                if i != j or graph_config.get('self_loops', False):
                    src.append(i)
                    dst.append(j)
        edge_index = np.array([src, dst])
    else:
        # Threshold-based
        threshold = graph_config.get('edge_threshold', 0.3)
        mask = sync_matrix > threshold
        np.fill_diagonal(mask, graph_config.get('self_loops', False))
        edge_index = np.array(np.where(mask))
    
    # Edge features
    edge_attr_list = []
    
    # Synchronization weight
    edge_weights = sync_matrix[edge_index[0], edge_index[1]]
    edge_attr_list.append(edge_weights.reshape(-1, 1))
    
    # Temporal difference - always add if enabled to maintain consistent dimensions
    if config['data'].get('use_temporal_diff', False):
        if prev_sync_matrix is not None:
            diff_matrix = sync_matrix - prev_sync_matrix
            edge_diff = diff_matrix[edge_index[0], edge_index[1]]
        else:
            # First epoch: use zeros to maintain consistent edge_attr shape
            edge_diff = np.zeros_like(edge_weights)
        edge_attr_list.append(edge_diff.reshape(-1, 1))
    
    edge_attr = np.hstack(edge_attr_list) if edge_attr_list else edge_weights.reshape(-1, 1)
    
    # ==========================================================================
    # GRAPH-LEVEL FEATURES
    # ==========================================================================
    
    graph_features = []
    graph_config_features = config['data'].get('graph_features', {})
    
    if graph_config_features.get('use_kuramoto', True):
        kuramoto = compute_kuramoto(phases)
        graph_features.append(kuramoto)
    
    if graph_config_features.get('use_global_sync', True):
        global_sync = np.mean(sync_matrix[np.triu_indices(n_channels, k=1)])
        graph_features.append(global_sync)
    
    if graph_config_features.get('use_topology', True):
        # Mean degree
        mean_degree = np.mean(sync_matrix.sum(axis=1))
        # Sync variance
        sync_var = np.var(sync_matrix[np.triu_indices(n_channels, k=1)])
        graph_features.extend([mean_degree, sync_var])
    
    graph_attr = np.array(graph_features) if graph_features else None
    
    # ==========================================================================
    # CREATE DATA OBJECT
    # ==========================================================================
    
    # Label based on condition
    condition_to_label = {'DMT': 0, 'EC': 1, 'EO': 2}
    label = condition_to_label.get(condition, 0)
    
    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        y=torch.tensor([label], dtype=torch.long),
        num_nodes=n_channels
    )
    
    if graph_attr is not None:
        data.graph_attr = torch.tensor(graph_attr, dtype=torch.float32)
    
    # Metadata (stored as simple types to avoid batching issues)
    data.subject_id = subject_id
    data.condition = condition
    data.band = band
    data.epoch_idx = epoch_idx
    
    return data


# =============================================================================
# MULTI-BAND PROCESSING
# =============================================================================

def build_multiband_graph(
    all_phases: Dict[str, np.ndarray],
    condition: str,
    subject_id: str,
    epoch_idx: int,
    config: Dict[str, Any]
) -> Data:
    """
    Build graph with all frequency bands as multi-channel features.
    
    Args:
        all_phases: Dict mapping band name to [n_channels, n_timepoints]
        condition: Condition name
        subject_id: Subject identifier
        epoch_idx: Epoch index
        config: Configuration
        
    Returns:
        PyG Data with multi-band features
    """
    bands = config['data']['bands']
    n_channels = list(all_phases.values())[0].shape[0]
    
    # Compute features for each band
    all_node_features = []
    all_sync_matrices = []
    
    for band in bands:
        if band not in all_phases:
            continue
        phases = all_phases[band]
        
        # Phase statistics
        phase_stats = extract_phase_statistics(phases)
        all_node_features.append(phase_stats)
        
        # Sync matrix
        sync_matrix = compute_sync_matrix(phases)
        all_sync_matrices.append(sync_matrix)
    
    # Stack features: [n_channels, n_bands * n_features_per_band]
    x = np.hstack(all_node_features)
    
    # Average sync matrix across bands (or could keep separate)
    avg_sync = np.mean(all_sync_matrices, axis=0)
    
    # Build edges
    graph_config = config['data']['graph']
    if graph_config.get('fully_connected', True):
        src, dst = [], []
        for i in range(n_channels):
            for j in range(n_channels):
                if i != j:
                    src.append(i)
                    dst.append(j)
        edge_index = np.array([src, dst])
    else:
        threshold = graph_config.get('edge_threshold', 0.3)
        mask = avg_sync > threshold
        np.fill_diagonal(mask, False)
        edge_index = np.array(np.where(mask))
    
    # Multi-band edge features: one weight per band
    edge_attr_list = []
    for sync_matrix in all_sync_matrices:
        weights = sync_matrix[edge_index[0], edge_index[1]]
        edge_attr_list.append(weights.reshape(-1, 1))
    edge_attr = np.hstack(edge_attr_list)
    
    # Graph features: Kuramoto per band
    graph_features = []
    for band in bands:
        if band in all_phases:
            kuramoto = compute_kuramoto(all_phases[band])
            graph_features.append(kuramoto)
    
    # Label
    condition_to_label = {'DMT': 0, 'EC': 1, 'EO': 2}
    label = condition_to_label.get(condition, 0)
    
    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        y=torch.tensor([label], dtype=torch.long),
        num_nodes=n_channels
    )
    
    data.graph_attr = torch.tensor(graph_features, dtype=torch.float32)
    data.subject_id = subject_id
    data.condition = condition
    data.band = 'multiband'
    data.epoch_idx = epoch_idx
    
    return data


# =============================================================================
# TEMPORAL WINDOWS
# =============================================================================

def build_temporal_window_graphs(
    epochs_data: List[Tuple[np.ndarray, Dict]],
    window_size: int,
    stride: int,
    config: Dict[str, Any]
) -> List[Data]:
    """
    Build graphs from temporal windows of consecutive epochs.
    
    Args:
        epochs_data: List of (phases, metadata) tuples
        window_size: Number of epochs in window
        stride: Step between windows
        config: Configuration
        
    Returns:
        List of Data objects representing temporal windows
    """
    graphs = []
    n_epochs = len(epochs_data)
    
    for start_idx in range(0, n_epochs - window_size + 1, stride):
        window = epochs_data[start_idx:start_idx + window_size]
        
        # Stack phases from all epochs in window
        phases_list = [w[0] for w in window]
        metadata = window[0][1]  # Use first epoch's metadata
        
        # Compute features across the window
        n_channels = phases_list[0].shape[0]
        
        # Aggregate node features across window
        all_features = []
        all_sync = []
        
        for phases in phases_list:
            stats = extract_phase_statistics(phases)
            all_features.append(stats)
            sync = compute_sync_matrix(phases)
            all_sync.append(sync)
        
        # Temporal features: mean, std, trend across window
        stacked_features = np.stack(all_features, axis=0)  # [window, channels, features]
        
        mean_features = np.mean(stacked_features, axis=0)
        std_features = np.std(stacked_features, axis=0)
        
        # Trend (simple linear fit coefficient)
        x_time = np.arange(window_size)
        trend_features = np.zeros_like(mean_features)
        for ch in range(n_channels):
            for f in range(mean_features.shape[1]):
                slope, _, _, _, _ = stats.linregress(x_time, stacked_features[:, ch, f])
                trend_features[ch, f] = slope
        
        x = np.hstack([mean_features, std_features, trend_features])
        
        # Edge features: mean sync and sync change
        stacked_sync = np.stack(all_sync, axis=0)
        mean_sync = np.mean(stacked_sync, axis=0)
        sync_change = stacked_sync[-1] - stacked_sync[0]
        
        # Build edges
        graph_config = config['data']['graph']
        src, dst = [], []
        for i in range(n_channels):
            for j in range(n_channels):
                if i != j:
                    src.append(i)
                    dst.append(j)
        edge_index = np.array([src, dst])
        
        edge_weights = mean_sync[edge_index[0], edge_index[1]]
        edge_changes = sync_change[edge_index[0], edge_index[1]]
        edge_attr = np.stack([edge_weights, edge_changes], axis=1)
        
        # Label
        condition = metadata['condition']
        condition_to_label = {'DMT': 0, 'EC': 1, 'EO': 2}
        label = condition_to_label.get(condition, 0)
        
        data = Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
            y=torch.tensor([label], dtype=torch.long),
            num_nodes=n_channels
        )
        
        # Graph-level features
        kuramatos = [compute_kuramoto(p) for p in phases_list]
        data.graph_attr = torch.tensor([
            np.mean(kuramatos),
            np.std(kuramatos),
            kuramatos[-1] - kuramatos[0]
        ], dtype=torch.float32)
        
        data.subject_id = metadata['subject_id']
        data.condition = condition
        data.band = metadata['band']
        data.epoch_idx = start_idx
        data.window_size = window_size
        
        graphs.append(data)
    
    return graphs


# =============================================================================
# DATASET CREATION
# =============================================================================

def process_subject_condition(
    subject_id: str,
    condition: str,
    phases_dir: Path,
    config: Dict[str, Any],
    use_stc: bool = False
) -> List[Data]:
    """
    Process all epochs for a subject/condition combination.
    
    Args:
        subject_id: Subject ID (e.g., 'S01')
        condition: Condition name
        phases_dir: Directory containing phase files
        config: Configuration
        use_stc: Use source-localized data
        
    Returns:
        List of Data objects
    """
    graphs = []
    bands = config['data']['bands']
    processing_mode = config['data'].get('processing_mode', 'single_band')
    
    # Find the phases file for this subject/condition
    # Format: phases-S01-DMT.pkl
    phases_file = phases_dir / condition / f"phases-{subject_id}-{condition}.pkl"
    
    if not phases_file.exists():
        logger.warning(f"File not found: {phases_file}")
        return []
    
    # Load the phases file
    data = load_phases_file(phases_file)
    if data is None:
        logger.warning(f"Failed to load: {phases_file}")
        return []
    
    # Determine which data key to use (EEG or STC)
    if use_stc:
        phases_key = 'phases_stc'
        syncros_key = 'syncros_stc'
        kuramoto_key = 'kuramoto_stc'
    else:
        phases_key = 'phases_eeg'
        syncros_key = 'syncros_eeg'
        kuramoto_key = 'kuramoto_eeg'
    
    if phases_key not in data:
        logger.warning(f"Key '{phases_key}' not found in {phases_file}")
        return []
    
    phases_data = data[phases_key]  # Dict: band -> list of epochs
    syncros_data = data.get(syncros_key, {})
    kuramoto_data = data.get(kuramoto_key, {})
    
    # Collect band data
    band_data = {}
    for band in bands:
        if band in phases_data:
            band_data[band] = phases_data[band]
    
    if not band_data:
        logger.warning(f"No matching bands found for {subject_id}/{condition}")
        return []
    
    # Get number of epochs (use minimum across bands)
    n_epochs = min(len(epochs) for epochs in band_data.values())
    
    logger.debug(f"  {subject_id}/{condition}: {n_epochs} epochs")
    
    # ==========================================================================
    # PROCESSING MODES
    # ==========================================================================
    
    if processing_mode == 'multiband':
        # All bands at once
        for epoch_idx in range(n_epochs):
            all_phases = {band: band_data[band][epoch_idx] for band in band_data}
            graph = build_multiband_graph(
                all_phases, condition, subject_id, epoch_idx, config
            )
            graphs.append(graph)
    
    elif processing_mode == 'temporal_window':
        # Temporal windows
        window_size = config['data'].get('window_size', 5)
        stride = config['data'].get('window_stride', 1)
        
        for band in bands:
            if band not in band_data:
                continue
            
            epochs = band_data[band]
            epochs_with_meta = [
                (epochs[i], {'subject_id': subject_id, 'condition': condition, 'band': band})
                for i in range(len(epochs))
            ]
            
            window_graphs = build_temporal_window_graphs(
                epochs_with_meta, window_size, stride, config
            )
            graphs.extend(window_graphs)
    
    else:  # single_band (default)
        # Process each band separately
        for band in bands:
            if band not in band_data:
                continue
            
            prev_sync = None
            for epoch_idx in range(n_epochs):
                phases = band_data[band][epoch_idx]
                
                graph = build_enhanced_graph(
                    phases, condition, band, subject_id, epoch_idx,
                    config, prev_sync_matrix=prev_sync
                )
                graphs.append(graph)
                
                # Store for temporal diff
                if config['data'].get('use_temporal_diff', False):
                    prev_sync = compute_sync_matrix(phases)
    
    return graphs


def create_enhanced_dataset(
    config: Dict[str, Any],
    subjects: Optional[List[str]] = None,
    conditions: Optional[List[str]] = None,
    num_workers: int = 8
) -> List[Data]:
    """
    Create enhanced dataset with all specified variations.
    
    Args:
        config: Configuration dictionary
        subjects: Optional list of subjects to include
        conditions: Optional list of conditions to include
        num_workers: Number of parallel workers
        
    Returns:
        List of Data objects
    """
    phases_dir = Path(config['paths']['phases_dir'])
    use_stc = config['data'].get('use_stc', False)
    
    if conditions is None:
        conditions = config['data']['conditions']
    
    # Find all subjects from phases-*.pkl files
    if subjects is None:
        subjects = set()
        for condition in conditions:
            cond_dir = phases_dir / condition
            if cond_dir.exists():
                # Look for phases-S*-*.pkl files
                for f in cond_dir.glob("phases-S*-*.pkl"):
                    # Extract subject ID from filename like "phases-S01-DMT.pkl"
                    parts = f.stem.split('-')
                    if len(parts) >= 2 and parts[1].startswith('S'):
                        subjects.add(parts[1])
        subjects = sorted(subjects)
    
    if not subjects:
        raise ValueError(f"No subjects found in {phases_dir}")
    
    logger.info(f"Processing {len(subjects)} subjects, {len(conditions)} conditions")
    
    # Process all subject/condition combinations
    all_graphs = []
    
    tasks = [(subj, cond) for subj in subjects for cond in conditions]
    
    for subject_id, condition in tqdm(tasks, desc="Processing"):
        graphs = process_subject_condition(
            subject_id, condition, phases_dir, config, use_stc
        )
        all_graphs.extend(graphs)
    
    logger.info(f"Created {len(all_graphs)} graphs")
    
    return all_graphs


def split_dataset(
    graphs: List[Data],
    config: Dict[str, Any],
    random_state: int = 42
) -> Tuple[List[Data], List[Data], List[Data]]:
    """
    Split dataset into train/val/test sets.
    
    Args:
        graphs: List of all graphs
        config: Configuration
        random_state: Random seed
        
    Returns:
        Tuple of (train, val, test) lists
    """
    split_config = config['data']['split']
    train_ratio = split_config['train_ratio']
    val_ratio = split_config['val_ratio']
    
    np.random.seed(random_state)
    
    # Check number of unique subjects
    unique_subjects = set(g.subject_id for g in graphs)
    
    # If only 1 subject or group_by_subject is False, split by epochs
    if len(unique_subjects) <= 1 or not split_config.get('group_by_subject', True):
        # Random split by epochs
        indices = np.random.permutation(len(graphs))
        n_train = int(len(graphs) * train_ratio)
        n_val = int(len(graphs) * val_ratio)
        
        train = [graphs[i] for i in indices[:n_train]]
        val = [graphs[i] for i in indices[n_train:n_train + n_val]]
        test = [graphs[i] for i in indices[n_train + n_val:]]
        
        logger.info(f"Split by epochs (single subject mode)")
    else:
        # Group by subject to avoid data leakage
        subject_graphs = defaultdict(list)
        for g in graphs:
            subject_graphs[g.subject_id].append(g)
        
        subjects = list(subject_graphs.keys())
        np.random.shuffle(subjects)
        
        n_train = int(len(subjects) * train_ratio)
        n_val = int(len(subjects) * val_ratio)
        
        # Ensure at least 1 subject in each split if possible
        if n_train == 0 and len(subjects) >= 1:
            n_train = 1
        if n_val == 0 and len(subjects) >= 2:
            n_val = 1
        
        train_subjects = subjects[:n_train]
        val_subjects = subjects[n_train:n_train + n_val]
        test_subjects = subjects[n_train + n_val:]
        
        train = [g for s in train_subjects for g in subject_graphs[s]]
        val = [g for s in val_subjects for g in subject_graphs[s]]
        test = [g for s in test_subjects for g in subject_graphs[s]]
        
        logger.info(f"Split by subject: train={len(train_subjects)}, val={len(val_subjects)}, test={len(test_subjects)}")
    
    return train, val, test


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def create_dataset_from_config(
    config: Dict[str, Any],
    subjects: Optional[List[str]] = None,
    conditions: Optional[List[str]] = None,
    force_rebuild: bool = False,
    num_workers: int = 8
) -> Tuple[List[Data], List[Data], List[Data]]:
    """
    Create and split dataset from configuration.
    
    Args:
        config: Configuration dictionary
        subjects: Optional specific subjects
        conditions: Optional specific conditions
        force_rebuild: Force rebuild even if cache exists
        num_workers: Number of workers
        
    Returns:
        Tuple of (train, val, test) graph lists
    """
    # Check cache
    cache_dir = Path(config['paths'].get('dataset_cache', 'cache'))
    cache_file = cache_dir / 'enhanced_dataset.pkl'
    
    if cache_file.exists() and not force_rebuild and subjects is None and conditions is None:
        logger.info(f"Loading cached dataset from {cache_file}")
        with open(cache_file, 'rb') as f:
            cached = pickle.load(f)
        return cached['train'], cached['val'], cached['test']
    
    # Create dataset
    graphs = create_enhanced_dataset(config, subjects, conditions, num_workers)
    
    if len(graphs) == 0:
        raise ValueError("No graphs created! Check data paths and configuration.")
    
    # Split
    train, val, test = split_dataset(graphs, config)
    
    # Cache (only if not filtering)
    if subjects is None and conditions is None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump({'train': train, 'val': val, 'test': test}, f)
        logger.info(f"Cached dataset to {cache_file}")
    
    logger.info(f"Dataset: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    
    return train, val, test

