"""
Dataset builder for converting EEG synchronization data to graph format.

This module handles loading phases-*.pkl files and converting each epoch
to a PyTorch Geometric graph with appropriate node/edge features.
"""

import os
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm
from scipy.stats import entropy, kurtosis
from sklearn.model_selection import train_test_split, GroupShuffleSplit

warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = logging.getLogger(__name__)


class EEGGraphDataset(Dataset):
    """
    PyTorch Geometric Dataset for EEG synchronization graphs.
    
    Each graph represents one epoch with:
    - Nodes: EEG channels or brain parcels
    - Edges: Synchronization strength between nodes
    - Node features: Phase/amplitude statistics, network membership
    - Edge features: Synchronization value
    - Graph features: Global Kuramoto parameter, topology metrics
    """
    
    def __init__(self, 
                 root: str,
                 graphs: List[Data],
                 transform=None,
                 pre_transform=None):
        """
        Args:
            root: Root directory for dataset
            graphs: List of PyTorch Geometric Data objects
            transform: Optional transform to apply on-the-fly
            pre_transform: Optional transform to apply once during processing
        """
        self.graphs = graphs
        super().__init__(root, transform, pre_transform)
    
    def len(self) -> int:
        return len(self.graphs)
    
    def get(self, idx: int) -> Data:
        data = self.graphs[idx]
        if self.transform:
            data = self.transform(data)
        return data


def load_phases_file(filepath: str) -> Dict[str, Any]:
    """
    Load a phases-*.pkl file with backward compatibility.
    
    Args:
        filepath: Path to phases-*.pkl file
        
    Returns:
        Dictionary containing phases, amplitudes, syncros, kuramoto data
    """
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        # Try pandas compatibility fix
        try:
            import pandas.core.indexes.numeric as ni
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            return data
        except:
            raise


def compute_temporal_complexity(signal: np.ndarray) -> Dict[str, float]:
    """
    Compute temporal complexity features from a time series.
    
    Args:
        signal: 1D array of temporal signal
        
    Returns:
        Dictionary of complexity features
    """
    features = {}
    
    # Entropy (discretize into 10 bins)
    hist, _ = np.histogram(signal, bins=10, density=True)
    hist = hist[hist > 0]  # Remove zeros
    features['entropy'] = entropy(hist)
    
    # Kurtosis (tailedness)
    features['kurtosis'] = kurtosis(signal)
    
    # Coefficient of variation
    if np.abs(signal.mean()) > 1e-6:
        features['cv'] = signal.std() / np.abs(signal.mean())
    else:
        features['cv'] = 0.0
    
    # Range
    features['range'] = signal.max() - signal.min()
    
    return features


def compute_graph_topology(edge_index: torch.Tensor, num_nodes: int) -> Dict[str, float]:
    """
    Compute graph topology features.
    
    Args:
        edge_index: Edge indices [2, num_edges]
        num_nodes: Number of nodes
        
    Returns:
        Dictionary of topology features
    """
    features = {}
    
    # Density
    num_edges = edge_index.shape[1]
    max_edges = num_nodes * (num_nodes - 1)
    features['density'] = num_edges / max_edges if max_edges > 0 else 0.0
    
    # Average degree
    degrees = torch.bincount(edge_index[0], minlength=num_nodes)
    features['avg_degree'] = degrees.float().mean().item()
    features['max_degree'] = degrees.float().max().item()
    features['min_degree'] = degrees.float().min().item()
    
    return features


def build_graph_from_epoch(
    syncro_matrix: np.ndarray,
    phase_array: np.ndarray,
    amplitude_array: np.ndarray,
    kuramoto_series: np.ndarray,
    label: int,
    subject_id: str,
    condition: str,
    band: str,
    epoch_idx: int,
    config: Dict[str, Any],
    network_labels: Optional[pd.DataFrame] = None
) -> Data:
    """
    Convert one EEG epoch to a PyTorch Geometric graph.
    
    Args:
        syncro_matrix: (N, N) synchronization matrix
        phase_array: (N, T) instantaneous phases
        amplitude_array: (N, T) amplitude envelopes
        kuramoto_series: (T,) Kuramoto order parameter time series
        label: Class label (0=DMT, 1=EC, 2=EO)
        subject_id: Subject identifier
        condition: Condition name
        band: Frequency band name
        epoch_idx: Epoch index
        config: Configuration dictionary
        network_labels: DataFrame with network membership (optional)
        
    Returns:
        PyTorch Geometric Data object
    """
    N = syncro_matrix.shape[0]  # Number of nodes
    T = phase_array.shape[1]    # Number of time points
    
    graph_config = config['data']['graph']
    node_config = config['data']['node_features']
    graph_feat_config = config['data']['graph_features']
    
    # ========================================================================
    # 1. EDGE CONSTRUCTION
    # ========================================================================
    
    # Check if fully connected mode
    fully_connected = graph_config.get('fully_connected', False)
    
    edge_index = []
    edge_attr = []
    
    if fully_connected:
        # FULLY CONNECTED: All nodes connected to all others
        for i in range(N):
            for j in range(N):
                if i != j:  # No self-loops (unless config says so)
                    edge_index.append([i, j])
                    edge_attr.append(syncro_matrix[i, j])
        
        logger.debug(f"Fully connected graph: {N} nodes, {len(edge_index)} edges")
    else:
        # THRESHOLD-BASED: Only strong connections
        threshold = graph_config['edge_threshold']
        if graph_config['edge_threshold_percentile'] is not None:
            threshold = np.percentile(syncro_matrix, graph_config['edge_threshold_percentile'])
        
        # Extract edges from upper triangle of synchronization matrix
        for i in range(N):
            for j in range(i+1, N):
                sync_value = syncro_matrix[i, j]
                if sync_value > threshold:
                    # Add edge in both directions (undirected)
                    edge_index.append([i, j])
                    edge_index.append([j, i])
                    edge_attr.extend([sync_value, sync_value])
        
        # Handle empty graphs
        if len(edge_index) == 0:
            logger.warning(f"Empty graph for {subject_id} {condition} {band} epoch {epoch_idx}")
            # Add at least one self-loop to avoid errors
            edge_index = [[0, 0]]
            edge_attr = [0.0]
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(1)
    
    # ========================================================================
    # 2. NODE FEATURES
    # ========================================================================
    
    node_features_list = []
    
    # Phase statistics
    if node_config['use_phase_stats']:
        phase_mean = phase_array.mean(axis=1)
        phase_std = phase_array.std(axis=1)
        node_features_list.extend([phase_mean, phase_std])
    
    # Amplitude statistics
    if node_config['use_amplitude_stats']:
        amp_mean = amplitude_array.mean(axis=1)
        amp_std = amplitude_array.std(axis=1)
        node_features_list.extend([amp_mean, amp_std])
    
    # Temporal complexity features
    if node_config['use_temporal_complexity']:
        complexity_features = []
        for i in range(N):
            comp = compute_temporal_complexity(phase_array[i])
            complexity_features.append([
                comp['entropy'],
                comp['kurtosis'],
                comp['cv'],
                comp['range']
            ])
        complexity_features = np.array(complexity_features).T
        for feat in complexity_features:
            node_features_list.append(feat)
    
    # Network membership (one-hot encoding)
    if node_config['use_network_label'] and network_labels is not None:
        # TODO: Implement network one-hot encoding
        # This requires parsing network labels from Schaefer atlas
        pass
    
    # Stack all node features
    node_features = np.column_stack(node_features_list)
    x = torch.tensor(node_features, dtype=torch.float)
    
    # ========================================================================
    # 3. GRAPH-LEVEL FEATURES
    # ========================================================================
    
    graph_features_list = []
    
    # Kuramoto statistics
    if graph_feat_config['use_kuramoto']:
        kuramoto_mean = kuramoto_series.mean()
        kuramoto_std = kuramoto_series.std()
        graph_features_list.extend([kuramoto_mean, kuramoto_std])
    
    # Global synchronization statistics
    if graph_feat_config['use_global_sync']:
        sync_mean = syncro_matrix.mean()
        sync_std = syncro_matrix.std()
        graph_features_list.extend([sync_mean, sync_std])
    
    # Topology features
    if graph_feat_config['use_topology']:
        topology = compute_graph_topology(edge_index, N)
        graph_features_list.extend([
            topology['density'],
            topology['avg_degree'],
            topology['max_degree'],
            topology['min_degree']
        ])
    
    graph_attr = torch.tensor(graph_features_list, dtype=torch.float)
    
    # ========================================================================
    # 4. CREATE DATA OBJECT
    # ========================================================================
    
    y = torch.tensor([label], dtype=torch.long)
    
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        graph_attr=graph_attr,
        num_nodes=N
    )
    
    # Store metadata
    data.subject_id = subject_id
    data.condition = condition
    data.band = band
    data.epoch_idx = epoch_idx
    
    return data


def process_single_file(args):
    """
    Process a single phases-*.pkl file. Designed for multiprocessing.
    
    Returns numpy arrays instead of torch tensors to avoid multiprocessing issues.
    
    Args:
        args: Tuple of (file_path, condition, cond_idx, bands, suffix, config)
        
    Returns:
        List of graph data as dictionaries (to be converted to Data objects later)
    """
    file_path, condition, cond_idx, bands, suffix, config = args
    
    graph_dicts = []
    subject_id = file_path.name.replace("phases-", "").replace(".pkl", "")
    
    # Load file
    try:
        data = load_phases_file(file_path)
    except Exception as e:
        # Use print instead of logger in multiprocessing
        print(f"Failed to load {file_path.name}: {e}")
        return graph_dicts
    
    # Process each band
    for band in bands:
        try:
            syncros = data[f"syncros{suffix}"][band]
            phases = data[f"phases{suffix}"][band]
            amplitudes = data[f"amplitudes{suffix}"][band]
            kuramoto = data[f"kuramoto{suffix}"][band]
        except KeyError as e:
            print(f"Missing key in {file_path.name}: {e}")
            continue
        
        # Each epoch becomes a graph
        num_epochs = len(syncros)
        for epoch_idx in range(num_epochs):
            try:
                # Build graph but keep as numpy/dict (not torch tensors)
                graph_dict = build_graph_dict(
                    syncro_matrix=syncros[epoch_idx],
                    phase_array=phases[epoch_idx],
                    amplitude_array=amplitudes[epoch_idx],
                    kuramoto_series=kuramoto[epoch_idx],
                    label=cond_idx,
                    subject_id=subject_id,
                    condition=condition,
                    band=band,
                    epoch_idx=epoch_idx,
                    config=config
                )
                graph_dicts.append(graph_dict)
            except Exception as e:
                print(f"Error creating graph for {subject_id} {band} epoch {epoch_idx}: {e}")
                continue
    
    return graph_dicts


def build_graph_dict(
    syncro_matrix: np.ndarray,
    phase_array: np.ndarray,
    amplitude_array: np.ndarray,
    kuramoto_series: np.ndarray,
    label: int,
    subject_id: str,
    condition: str,
    band: str,
    epoch_idx: int,
    config: Dict[str, Any],
    network_labels: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Build graph as dictionary (numpy arrays) instead of torch Data object.
    This avoids multiprocessing issues with torch tensors.
    
    Returns:
        Dictionary with numpy arrays (to be converted to Data object later)
    """
    N = syncro_matrix.shape[0]
    T = phase_array.shape[1]
    
    graph_config = config['data']['graph']
    node_config = config['data']['node_features']
    graph_feat_config = config['data']['graph_features']
    
    # ========================================================================
    # 1. EDGE CONSTRUCTION
    # ========================================================================
    
    fully_connected = graph_config.get('fully_connected', False)
    
    edge_index = []
    edge_attr = []
    
    if fully_connected:
        for i in range(N):
            for j in range(N):
                if i != j:
                    edge_index.append([i, j])
                    edge_attr.append(syncro_matrix[i, j])
    else:
        threshold = graph_config['edge_threshold']
        if graph_config['edge_threshold_percentile'] is not None:
            threshold = np.percentile(syncro_matrix, graph_config['edge_threshold_percentile'])
        
        for i in range(N):
            for j in range(i+1, N):
                sync_value = syncro_matrix[i, j]
                if sync_value > threshold:
                    edge_index.append([i, j])
                    edge_index.append([j, i])
                    edge_attr.extend([sync_value, sync_value])
        
        if len(edge_index) == 0:
            edge_index = [[0, 0]]
            edge_attr = [0.0]
    
    edge_index = np.array(edge_index, dtype=np.int64).T
    edge_attr = np.array(edge_attr, dtype=np.float32).reshape(-1, 1)
    
    # ========================================================================
    # 2. NODE FEATURES
    # ========================================================================
    
    node_features_list = []
    
    if node_config['use_phase_stats']:
        phase_mean = phase_array.mean(axis=1)
        phase_std = phase_array.std(axis=1)
        node_features_list.extend([phase_mean, phase_std])
    
    if node_config['use_amplitude_stats']:
        amp_mean = amplitude_array.mean(axis=1)
        amp_std = amplitude_array.std(axis=1)
        node_features_list.extend([amp_mean, amp_std])
    
    if node_config['use_temporal_complexity']:
        complexity_features = []
        for i in range(N):
            comp = compute_temporal_complexity(phase_array[i])
            complexity_features.append([
                comp['entropy'],
                comp['kurtosis'],
                comp['cv'],
                comp['range']
            ])
        complexity_features = np.array(complexity_features).T
        for feat in complexity_features:
            node_features_list.append(feat)
    
    node_features = np.column_stack(node_features_list).astype(np.float32)
    
    # ========================================================================
    # 3. GRAPH-LEVEL FEATURES
    # ========================================================================
    
    graph_features_list = []
    
    if graph_feat_config['use_kuramoto']:
        kuramoto_mean = kuramoto_series.mean()
        kuramoto_std = kuramoto_series.std()
        graph_features_list.extend([kuramoto_mean, kuramoto_std])
    
    if graph_feat_config['use_global_sync']:
        sync_mean = syncro_matrix.mean()
        sync_std = syncro_matrix.std()
        graph_features_list.extend([sync_mean, sync_std])
    
    if graph_feat_config['use_topology']:
        num_edges = edge_index.shape[1]
        max_edges = N * (N - 1)
        density = num_edges / max_edges if max_edges > 0 else 0.0
        degrees = np.bincount(edge_index[0], minlength=N)
        graph_features_list.extend([
            density,
            degrees.mean(),
            degrees.max(),
            degrees.min()
        ])
    
    graph_attr = np.array(graph_features_list, dtype=np.float32)
    
    # ========================================================================
    # 4. RETURN AS DICT (not torch Data)
    # ========================================================================
    
    return {
        'x': node_features,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'y': label,
        'graph_attr': graph_attr,
        'num_nodes': N,
        'subject_id': subject_id,
        'condition': condition,
        'band': band,
        'epoch_idx': epoch_idx
    }


def create_dataset_from_config(config: Dict[str, Any], 
                                force_rebuild: bool = False,
                                num_workers: Optional[int] = None) -> Tuple[List[Data], List[Data], List[Data]]:
    """
    Create train/val/test datasets from configuration.
    
    Args:
        config: Configuration dictionary
        force_rebuild: If True, rebuild dataset even if cache exists
        num_workers: Number of parallel workers (None = auto-detect)
        
    Returns:
        Tuple of (train_graphs, val_graphs, test_graphs)
    """
    phases_dir = Path(config['paths']['phases_dir'])
    cache_dir = Path(config['paths']['dataset_cache'])
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    conditions = config['data']['conditions']
    bands = config['data']['bands']
    use_stc = config['data']['use_stc']
    
    # Generate cache filename (use_stc determines if we use source space or electrodes)
    source_type = "stc" if use_stc else "eeg"
    cache_key = f"dataset_{'_'.join(conditions)}_{'_'.join(bands)}_{source_type}.pkl"
    cache_file = cache_dir / cache_key
    
    # Try to load from cache
    if cache_file.exists() and not force_rebuild:
        logger.info(f"Loading dataset from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            cached = pickle.load(f)
        return cached['train'], cached['val'], cached['test']
    
    logger.info("Building dataset from scratch with multiprocessing...")
    
    # Determine number of workers
    if num_workers is None:
        # Auto-detect: conservative default to avoid memory issues
        num_workers = min(cpu_count() // 2, 8)
        logger.info(f"Auto-detected {num_workers} workers (conservative default)")
    else:
        # Use explicit value from config
        logger.info(f"Using {num_workers} workers (from config)")
    
    logger.info(f"Starting parallel processing...")
    
    # ========================================================================
    # PREPARE FILE LIST FOR PARALLEL PROCESSING
    # ========================================================================
    
    suffix = "_stc" if use_stc else "_eeg"
    file_args = []
    
    for cond_idx, condition in enumerate(conditions):
        cond_path = phases_dir / condition
        
        if not cond_path.exists():
            logger.warning(f"Directory not found: {cond_path}")
            continue
        
        # Get all phases files
        phase_files = sorted([cond_path / f for f in os.listdir(cond_path) if f.startswith("phases-")])
        
        # Prepare arguments for each file
        for phase_file in phase_files:
            file_args.append((phase_file, condition, cond_idx, bands, suffix, config))
    
    logger.info(f"Found {len(file_args)} files to process across {len(conditions)} conditions")
    
    # ========================================================================
    # PARALLEL PROCESSING
    # ========================================================================
    
    all_graph_dicts = []
    
    with Pool(processes=num_workers) as pool:
        # Process files in parallel with progress bar
        results = list(tqdm(
            pool.imap(process_single_file, file_args),
            total=len(file_args),
            desc="Processing files"
        ))
        
        # Flatten results (lists of dicts)
        for graph_dicts_from_file in results:
            all_graph_dicts.extend(graph_dicts_from_file)
    
    logger.info(f"Total graphs processed: {len(all_graph_dicts)}")
    
    # ========================================================================
    # CONVERT TO TORCH DATA OBJECTS (after multiprocessing)
    # ========================================================================
    
    logger.info("Converting to PyTorch Geometric Data objects...")
    all_graphs = []
    
    for graph_dict in tqdm(all_graph_dicts, desc="Converting to torch"):
        # Now convert numpy arrays to torch tensors
        data = Data(
            x=torch.from_numpy(graph_dict['x']),
            edge_index=torch.from_numpy(graph_dict['edge_index']),
            edge_attr=torch.from_numpy(graph_dict['edge_attr']),
            y=torch.tensor([graph_dict['y']], dtype=torch.long),
            graph_attr=torch.from_numpy(graph_dict['graph_attr']),
            num_nodes=graph_dict['num_nodes']
        )
        
        # Store metadata
        data.subject_id = graph_dict['subject_id']
        data.condition = graph_dict['condition']
        data.band = graph_dict['band']
        data.epoch_idx = graph_dict['epoch_idx']
        
        all_graphs.append(data)
    
    logger.info(f"Total graphs created: {len(all_graphs)}")
    
    if len(all_graphs) == 0:
        raise ValueError("No graphs were created! Check your data paths and configuration.")
    
    # ========================================================================
    # SPLIT INTO TRAIN/VAL/TEST
    # ========================================================================
    
    split_config = config['data']['split']
    
    # Extract labels for stratification
    labels = [g.y.item() for g in all_graphs]
    
    # Split ratios
    train_ratio = split_config['train_ratio']
    val_ratio = split_config['val_ratio']
    test_ratio = split_config['test_ratio']
    
    # Check if we should split by subject (IMPORTANT to avoid data leakage!)
    group_by_subject = split_config.get('group_by_subject', False)
    
    if group_by_subject:
        # Extract subject IDs and group graphs by subject
        from collections import defaultdict
        subject_graphs = defaultdict(list)
        
        for g in all_graphs:
            if hasattr(g, 'subject_id'):
                subject_graphs[g.subject_id].append(g)
            else:
                subject_graphs[f"unknown_{len(subject_graphs)}"].append(g)
        
        unique_subjects = list(subject_graphs.keys())
        logger.info(f"Splitting by subject: {len(unique_subjects)} unique subjects")
        
        # Calculate class distribution per subject for stratification
        # Use the DOMINANT class of each subject for stratification
        subject_dominant_class = {}
        for subj, graphs in subject_graphs.items():
            class_counts = defaultdict(int)
            for g in graphs:
                class_counts[g.y.item()] += 1
            # Dominant class is the one with most graphs
            subject_dominant_class[subj] = max(class_counts.keys(), key=lambda k: class_counts[k])
        
        # STRATIFIED split by subject's dominant class  
        subjects_array = np.array(unique_subjects)
        subject_labels = np.array([subject_dominant_class[s] for s in unique_subjects])
        
        # First split: train vs (val+test)
        train_subjects_arr, temp_subjects_arr = train_test_split(
            subjects_array,
            train_size=train_ratio,
            stratify=subject_labels if split_config['stratify'] else None,
            random_state=split_config['random_state']
        )
        
        # Second split: val vs test
        temp_labels = np.array([subject_dominant_class[s] for s in temp_subjects_arr])
        val_size = val_ratio / (val_ratio + test_ratio)
        
        val_subjects_arr, test_subjects_arr = train_test_split(
            temp_subjects_arr,
            train_size=val_size,
            stratify=temp_labels if split_config['stratify'] else None,
            random_state=split_config['random_state']
        )
        
        train_subjects = set(train_subjects_arr)
        val_subjects = set(val_subjects_arr)
        test_subjects = set(test_subjects_arr)
        
        # Collect graphs for each split
        train_graphs = [g for s in train_subjects for g in subject_graphs[s]]
        val_graphs = [g for s in val_subjects for g in subject_graphs[s]]
        test_graphs = [g for s in test_subjects for g in subject_graphs[s]]
        
        # SHUFFLE graphs within each split (important for training!)
        import random
        random.seed(split_config['random_state'])
        random.shuffle(train_graphs)
        random.shuffle(val_graphs)
        random.shuffle(test_graphs)
        
        # Log class distribution per split
        for split_name, split_graphs in [('train', train_graphs), ('val', val_graphs), ('test', test_graphs)]:
            class_counts = defaultdict(int)
            for g in split_graphs:
                class_counts[g.y.item()] += 1
            logger.info(f"  {split_name} class distribution: {dict(class_counts)}")
        
        logger.info(f"Subject split: train={len(train_subjects)} subjects, "
                   f"val={len(val_subjects)} subjects, test={len(test_subjects)} subjects")
        logger.info(f"NO DATA LEAKAGE: Each subject appears in only ONE split")
        
        # Log detailed subject assignment per split
        conditions = config['data'].get('conditions', ['DMT', 'EC', 'EO'])
        logger.info("\n" + "=" * 60)
        logger.info("SUBJECT ASSIGNMENT PER SPLIT")
        logger.info("=" * 60)
        
        for split_name, split_subjs in [('TRAIN', train_subjects), ('VAL', val_subjects), ('TEST', test_subjects)]:
            logger.info(f"\n{split_name} subjects ({len(split_subjs)}):")
            for subj in sorted(split_subjs):
                graphs = subject_graphs[subj]
                subj_class_counts = defaultdict(int)
                for g in graphs:
                    subj_class_counts[g.y.item()] += 1
                
                # Format: subject_id: N graphs (DMT: X, EC: Y)
                class_str = ", ".join([f"{conditions[k]}: {v}" for k, v in sorted(subj_class_counts.items())])
                logger.info(f"  {subj}: {len(graphs)} graphs ({class_str})")
        
        logger.info("=" * 60 + "\n")
    else:
        # Random split (original behavior - may have data leakage!)
        logger.warning("group_by_subject=False: Subjects may appear in multiple splits (potential data leakage)")
        
        # First split: train vs (val+test)
        train_graphs, temp_graphs = train_test_split(
            all_graphs,
            train_size=train_ratio,
            stratify=labels if split_config['stratify'] else None,
            random_state=split_config['random_state']
        )
        
        # Second split: val vs test
        temp_labels = [g.y.item() for g in temp_graphs]
        val_size = val_ratio / (val_ratio + test_ratio)
        
        val_graphs, test_graphs = train_test_split(
            temp_graphs,
            train_size=val_size,
            stratify=temp_labels if split_config['stratify'] else None,
            random_state=split_config['random_state']
        )
    
    logger.info(f"Dataset split: train={len(train_graphs)}, val={len(val_graphs)}, test={len(test_graphs)}")
    
    # ========================================================================
    # CACHE RESULTS
    # ========================================================================
    
    with open(cache_file, 'wb') as f:
        pickle.dump({
            'train': train_graphs,
            'val': val_graphs,
            'test': test_graphs
        }, f)
    
    logger.info(f"Dataset cached to: {cache_file}")
    
    return train_graphs, val_graphs, test_graphs

