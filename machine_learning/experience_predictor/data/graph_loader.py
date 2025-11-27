"""
Graph data loader for loading synchronization graphs from fwd-inv-stc.

Loads phases-*.pkl files and builds graphs per epoch, then aggregates
by subject for experience prediction.

Supports parallel loading with multiple workers.
"""

import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import torch
from torch_geometric.data import Data
from scipy.stats import entropy, kurtosis
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)
logger = logging.getLogger(__name__)


def load_phases_file(filepath: str) -> Dict[str, Any]:
    """Load a phases-*.pkl file."""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        raise


def compute_temporal_features(signal: np.ndarray) -> np.ndarray:
    """Compute temporal complexity features from a time series."""
    features = []
    
    # Mean and std
    features.append(float(np.mean(signal)))
    features.append(float(np.std(signal)))
    
    # Entropy (discretize into 10 bins)
    try:
        hist, _ = np.histogram(signal, bins=10, density=True)
        hist = hist[hist > 0]
        features.append(float(entropy(hist)) if len(hist) > 0 else 0.0)
    except:
        features.append(0.0)
    
    # Kurtosis
    try:
        features.append(float(kurtosis(signal)))
    except:
        features.append(0.0)
    
    return np.array(features, dtype=np.float32)


def build_graph_from_epoch(
    syncro_matrix: np.ndarray,
    phase_array: np.ndarray,
    amplitude_array: np.ndarray,
    kuramoto_series: np.ndarray,
    fully_connected: bool = True,
    edge_threshold: float = 0.3
) -> Data:
    """
    Convert one EEG epoch to a PyTorch Geometric graph.
    
    Args:
        syncro_matrix: (N, N) synchronization matrix
        phase_array: (N, T) instantaneous phases
        amplitude_array: (N, T) amplitude envelopes
        kuramoto_series: (T,) Kuramoto order parameter
        fully_connected: If True, all nodes connected
        edge_threshold: Threshold for edge creation
    
    Returns:
        PyTorch Geometric Data object
    """
    N = syncro_matrix.shape[0]
    
    # Build edges
    edge_index = []
    edge_attr = []
    
    if fully_connected:
        for i in range(N):
            for j in range(N):
                if i != j:
                    edge_index.append([i, j])
                    edge_attr.append(float(syncro_matrix[i, j]))
    else:
        for i in range(N):
            for j in range(i+1, N):
                if syncro_matrix[i, j] > edge_threshold:
                    edge_index.append([i, j])
                    edge_index.append([j, i])
                    edge_attr.extend([float(syncro_matrix[i, j]), float(syncro_matrix[i, j])])
        
        if len(edge_index) == 0:
            edge_index = [[0, 0]]
            edge_attr = [0.0]
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(1)
    
    # Node features: phase/amplitude statistics
    node_features = []
    for i in range(N):
        feats = []
        # Phase stats
        feats.extend([float(np.mean(phase_array[i])), float(np.std(phase_array[i]))])
        # Amplitude stats  
        feats.extend([float(np.mean(amplitude_array[i])), float(np.std(amplitude_array[i]))])
        # Temporal complexity
        feats.extend(compute_temporal_features(phase_array[i]).tolist())
        node_features.append(feats)
    
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Graph-level features
    graph_attr = torch.tensor([
        float(np.mean(kuramoto_series)),
        float(np.std(kuramoto_series)),
        float(np.mean(syncro_matrix)),
        float(np.std(syncro_matrix))
    ], dtype=torch.float)
    
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        graph_attr=graph_attr,
        num_nodes=N
    )


def load_subject_graphs(
    phases_dir: str,
    subject_id: str,
    condition: str = "DMT",
    bands: List[str] = None,
    fully_connected: bool = True,
    edge_threshold: float = 0.3,
    use_stc: bool = False
) -> Dict[str, List[Data]]:
    """
    Load all graphs for a single subject.
    
    Args:
        phases_dir: Path to fwd-inv-stc directory
        subject_id: Subject ID (e.g., "S01")
        condition: Condition to load ("DMT", "EC", "EO")
        bands: List of bands to load (default: all)
        fully_connected: If True, create fully connected graphs
        edge_threshold: Edge threshold for sparse graphs
        use_stc: If True, use source space data (102 parcels), else EEG (24 channels)
    
    Returns:
        Dict mapping band name to list of graphs (one per epoch)
    """
    phases_dir = Path(phases_dir) / condition
    phases_file = phases_dir / f"phases-{subject_id}-{condition}.pkl"
    
    if not phases_file.exists():
        logger.warning(f"File not found: {phases_file}")
        return {}
    
    data = load_phases_file(str(phases_file))
    
    # Determine suffix based on data type
    suffix = "_stc" if use_stc else "_eeg"
    
    # Check if keys exist
    syncros_key = f"syncros{suffix}"
    phases_key = f"phases{suffix}"
    amplitudes_key = f"amplitudes{suffix}"
    kuramoto_key = f"kuramoto{suffix}"
    
    if syncros_key not in data:
        logger.warning(f"Key {syncros_key} not found in {phases_file}")
        return {}
    
    if bands is None:
        bands = list(data[syncros_key].keys())
    
    subject_graphs = {}
    
    for band in bands:
        try:
            syncros = data[syncros_key][band]
            phases = data[phases_key][band]
            amplitudes = data[amplitudes_key][band]
            kuramoto = data[kuramoto_key][band]
        except KeyError as e:
            logger.warning(f"Band {band} not found: {e}")
            continue
        
        graphs = []
        num_epochs = len(syncros)
        
        for epoch_idx in range(num_epochs):
            try:
                graph = build_graph_from_epoch(
                    syncro_matrix=syncros[epoch_idx],
                    phase_array=phases[epoch_idx],
                    amplitude_array=amplitudes[epoch_idx],
                    kuramoto_series=kuramoto[epoch_idx],
                    fully_connected=fully_connected,
                    edge_threshold=edge_threshold
                )
                graphs.append(graph)
            except Exception as e:
                logger.warning(f"Error building graph for {subject_id} {band} epoch {epoch_idx}: {e}")
                continue
        
        if graphs:
            subject_graphs[band] = graphs
    
    return subject_graphs


def aggregate_subject_graphs(
    graphs: List[Data],
    method: str = "mean"
) -> Optional[Data]:
    """
    Aggregate multiple epoch graphs into a single subject-level graph.
    
    Args:
        graphs: List of graphs for the same subject
        method: Aggregation method ("mean", "max", "concat")
    
    Returns:
        Single aggregated graph or None if no graphs
    """
    if len(graphs) == 0:
        logger.warning("No graphs to aggregate")
        return None
    
    if len(graphs) == 1:
        return graphs[0]
    
    # Use first graph as template for structure
    template = graphs[0]
    
    # Stack node features
    x_stack = torch.stack([g.x for g in graphs], dim=0)  # [num_epochs, N, F]
    
    if method == "mean":
        x_agg = x_stack.mean(dim=0)
    elif method == "max":
        x_agg = x_stack.max(dim=0)[0]
    elif method == "concat":
        # Concatenate along feature dimension
        x_agg = x_stack.transpose(0, 1).reshape(template.num_nodes, -1)
    else:
        x_agg = x_stack.mean(dim=0)
    
    # Aggregate edge attributes
    edge_attr_stack = torch.stack([g.edge_attr for g in graphs], dim=0)
    edge_attr_agg = edge_attr_stack.mean(dim=0)
    
    # Aggregate graph attributes
    graph_attr_stack = torch.stack([g.graph_attr for g in graphs], dim=0)
    graph_attr_agg = graph_attr_stack.mean(dim=0)
    
    return Data(
        x=x_agg,
        edge_index=template.edge_index,
        edge_attr=edge_attr_agg,
        graph_attr=graph_attr_agg,
        num_nodes=template.num_nodes
    )


def _process_single_subject(args: Tuple) -> Optional[Tuple[str, Data]]:
    """
    Process a single subject (for parallel loading).
    
    Args:
        args: Tuple of (subject_id, phases_dir, bands, fully_connected, 
              edge_threshold, use_stc, aggregation, combine_bands)
    
    Returns:
        Tuple of (subject_id, combined_graph) or None if failed
    """
    (subject_id, phases_dir, bands, fully_connected, 
     edge_threshold, use_stc, aggregation, combine_bands) = args
    
    try:
        subject_graphs = load_subject_graphs(
            phases_dir=phases_dir,
            subject_id=subject_id,
            condition="DMT",
            bands=bands,
            fully_connected=fully_connected,
            edge_threshold=edge_threshold,
            use_stc=use_stc
        )
        
        if not subject_graphs:
            return None
        
        # Aggregate epochs per band
        aggregated_per_band = {}
        for band, graphs in subject_graphs.items():
            try:
                agg_graph = aggregate_subject_graphs(graphs, method=aggregation)
                if agg_graph is not None:
                    aggregated_per_band[band] = agg_graph
            except Exception:
                continue
        
        if not aggregated_per_band:
            return None
        
        # Combine bands
        if combine_bands == "concat":
            band_list = sorted(aggregated_per_band.keys())
            x_concat = torch.cat([aggregated_per_band[b].x for b in band_list], dim=1)
            graph_attr_concat = torch.cat([aggregated_per_band[b].graph_attr for b in band_list])
            template = aggregated_per_band[band_list[0]]
            
            combined = Data(
                x=x_concat,
                edge_index=template.edge_index,
                edge_attr=template.edge_attr,
                graph_attr=graph_attr_concat,
                num_nodes=template.num_nodes
            )
            
        elif combine_bands == "mean":
            band_list = sorted(aggregated_per_band.keys())
            x_stack = torch.stack([aggregated_per_band[b].x for b in band_list])
            x_mean = x_stack.mean(dim=0)
            template = aggregated_per_band[band_list[0]]
            
            combined = Data(
                x=x_mean,
                edge_index=template.edge_index,
                edge_attr=template.edge_attr,
                graph_attr=template.graph_attr,
                num_nodes=template.num_nodes
            )
            
        elif combine_bands == "separate":
            combined = aggregated_per_band
        else:
            return None
        
        return (subject_id, combined)
        
    except Exception as e:
        return None


def load_all_dmt_subjects(
    phases_dir: str,
    bands: List[str] = None,
    aggregation: str = "mean",
    fully_connected: bool = True,
    edge_threshold: float = 0.3,
    combine_bands: str = "concat",
    use_stc: bool = False,
    num_workers: int = None
) -> Tuple[List[Data], List[str]]:
    """
    Load and aggregate graphs for all DMT subjects.
    
    Args:
        phases_dir: Path to fwd-inv-stc directory
        bands: Bands to use (default: all)
        aggregation: How to aggregate epochs ("mean", "max")
        fully_connected: Create fully connected graphs
        edge_threshold: Edge threshold for sparse graphs
        combine_bands: How to combine bands ("concat", "mean", "separate")
        use_stc: If True, use source space (102 parcels), else EEG (24 channels)
        num_workers: Number of parallel workers (default: cpu_count)
    
    Returns:
        Tuple of (list of subject graphs, list of subject IDs)
    """
    phases_dir = Path(phases_dir)
    dmt_dir = phases_dir / "DMT"
    
    # Find all subjects
    subject_files = sorted(dmt_dir.glob("phases-S*-DMT.pkl"))
    subject_ids = [f.stem.split("-")[1] for f in subject_files]
    
    logger.info(f"Found {len(subject_ids)} DMT subjects")
    
    # Determine number of workers
    if num_workers is None:
        num_workers = min(cpu_count(), 8)  # Cap at 8 to avoid memory issues
    
    # Prepare arguments for parallel processing
    args_list = [
        (sid, str(phases_dir), bands, fully_connected, 
         edge_threshold, use_stc, aggregation, combine_bands)
        for sid in subject_ids
    ]
    
    all_graphs = []
    valid_subject_ids = []
    
    if num_workers > 1:
        logger.info(f"  Loading {len(subject_ids)} subjects with {num_workers} workers...")
        with Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(_process_single_subject, args_list),
                total=len(args_list),
                desc="  Loading subjects",
                ncols=80
            ))
    else:
        logger.info(f"  Loading {len(subject_ids)} subjects sequentially...")
        results = []
        for args in tqdm(args_list, desc="  Loading subjects", ncols=80):
            results.append(_process_single_subject(args))
    
    # Collect results
    for result in results:
        if result is not None:
            subject_id, graph = result
            all_graphs.append(graph)
            valid_subject_ids.append(subject_id)
    
    logger.info(f"Loaded {len(all_graphs)} subject graphs")
    
    return all_graphs, valid_subject_ids


def match_subjects_to_targets(
    subject_ids: List[str],
    targets: np.ndarray,
    target_subject_order: List[str] = None
) -> Tuple[np.ndarray, List[int]]:
    """
    Match loaded subjects to target array.
    
    Args:
        subject_ids: List of subject IDs from loaded graphs
        targets: Target array [N_targets, num_targets]
        target_subject_order: Order of subjects in target array
    
    Returns:
        Tuple of (matched targets, valid indices)
    """
    if target_subject_order is None:
        # Assume targets are in order S01, S02, S04, S05, ... (matching phases files)
        target_subject_order = [f"S{i:02d}" for i in range(1, targets.shape[0] + 1)]
    
    matched_targets = []
    valid_indices = []
    
    for i, sid in enumerate(subject_ids):
        if sid in target_subject_order:
            target_idx = target_subject_order.index(sid)
            if target_idx < targets.shape[0]:
                matched_targets.append(targets[target_idx])
                valid_indices.append(i)
    
    return np.array(matched_targets), valid_indices
