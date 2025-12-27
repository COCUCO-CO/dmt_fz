"""
Kuramoto Proxy Calculation Module.

Computes synchronization metrics from reconstructed edge features (PLV)
to compare with original Kuramoto order parameter.

The VAE reconstructs edge_attr (PLV values), from which we can derive
global synchronization metrics analogous to Kuramoto.

Mathematical background:
- Original Kuramoto: r(t) = |1/N * Σ e^(iφ_j)| computed from phases
- PLV: |<e^(i(φ_i - φ_j))>| measures pairwise phase coherence
- Kuramoto proxy: derived from mean PLV or spectral properties of sync matrix
"""

import numpy as np
import torch
from typing import Dict, Union


def compute_kuramoto_proxy_from_edges(
    edge_attr: Union[np.ndarray, torch.Tensor],
    edge_index: Union[np.ndarray, torch.Tensor],
    num_nodes: int,
    method: str = 'mean_plv'
) -> float:
    """
    Compute Kuramoto-like synchronization proxy from edge features (PLV).
    
    Since we only have PLV (pairwise sync), not raw phases, we approximate
    global synchronization using different methods.
    
    Args:
        edge_attr: Edge features [num_edges, num_edge_features] or [num_edges]
                   First column is assumed to be PLV/sync weight
        edge_index: Edge indices [2, num_edges]
        num_nodes: Number of nodes in the graph
        method: Calculation method:
            - 'mean_plv': Mean of all PLV values (simple, fast)
            - 'spectral': Largest eigenvalue of sync matrix / num_nodes
            - 'weighted_mean': Mean weighted by degree
            
    Returns:
        Kuramoto proxy value in [0, 1]
    """
    # Convert to numpy if tensor
    if isinstance(edge_attr, torch.Tensor):
        edge_attr = edge_attr.detach().cpu().numpy()
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.detach().cpu().numpy()
    
    # Get PLV values (first column if multi-dimensional)
    if edge_attr.ndim > 1:
        plv_values = edge_attr[:, 0]
    else:
        plv_values = edge_attr
    
    # Ensure values are in valid range
    plv_values = np.clip(plv_values, 0, 1)
    
    if method == 'mean_plv':
        # Simple mean - equivalent to global_sync in original data
        return float(np.mean(plv_values))
    
    elif method == 'spectral':
        # Build sync matrix and compute largest eigenvalue
        sync_matrix = build_sync_matrix(edge_index, plv_values, num_nodes)
        # Largest eigenvalue normalized by N gives coherence measure
        try:
            eigenvalues = np.linalg.eigvalsh(sync_matrix)
            max_eigenvalue = np.max(eigenvalues)
            # Normalize: for fully coherent (all 1s), max eigenvalue ≈ N
            return float(max_eigenvalue / num_nodes)
        except np.linalg.LinAlgError:
            return float(np.mean(plv_values))  # Fallback
    
    elif method == 'weighted_mean':
        # Weight PLV by node connectivity
        sync_matrix = build_sync_matrix(edge_index, plv_values, num_nodes)
        degrees = sync_matrix.sum(axis=1)
        total_degree = degrees.sum()
        if total_degree > 0:
            weighted_sum = np.sum(sync_matrix * degrees[:, None])
            return float(weighted_sum / (total_degree * num_nodes))
        return float(np.mean(plv_values))
    
    else:
        raise ValueError(f"Unknown method: {method}")


def build_sync_matrix(
    edge_index: np.ndarray,
    edge_weights: np.ndarray,
    num_nodes: int
) -> np.ndarray:
    """
    Build synchronization matrix from sparse edge representation.
    
    Args:
        edge_index: [2, num_edges] source and target indices
        edge_weights: [num_edges] edge weights (PLV values)
        num_nodes: Number of nodes
        
    Returns:
        [num_nodes, num_nodes] symmetric sync matrix
    """
    sync_matrix = np.zeros((num_nodes, num_nodes))
    
    src = edge_index[0]
    dst = edge_index[1]
    
    for i, (s, d) in enumerate(zip(src, dst)):
        if s < num_nodes and d < num_nodes:
            weight = edge_weights[i] if i < len(edge_weights) else 0
            sync_matrix[s, d] = weight
            sync_matrix[d, s] = weight  # Symmetric
    
    return sync_matrix


def compute_kuramoto_comparison(
    original_graph_attr: Union[np.ndarray, torch.Tensor],
    reconstructed_edge_attr: Union[np.ndarray, torch.Tensor],
    edge_index: Union[np.ndarray, torch.Tensor],
    num_nodes: int,
    method: str = 'mean_plv'
) -> Dict[str, float]:
    """
    Compare original Kuramoto with reconstructed proxy.
    
    Args:
        original_graph_attr: Original graph attributes [4]:
            [kuramoto, global_sync, mean_degree, sync_variance]
        reconstructed_edge_attr: Reconstructed edge features
        edge_index: Edge indices
        num_nodes: Number of nodes
        method: Proxy calculation method
        
    Returns:
        Dictionary with:
            - kuramoto_original: Original Kuramoto value
            - kuramoto_proxy: Computed proxy from reconstruction
            - global_sync_original: Original global sync (mean PLV)
            - global_sync_proxy: Reconstructed global sync
            - absolute_error: |original - proxy|
            - relative_error: |original - proxy| / original
    """
    # Convert to numpy
    if isinstance(original_graph_attr, torch.Tensor):
        original_graph_attr = original_graph_attr.detach().cpu().numpy()
    
    # Extract original values
    kuramoto_original = float(original_graph_attr[0])
    global_sync_original = float(original_graph_attr[1]) if len(original_graph_attr) > 1 else kuramoto_original
    
    # Compute proxy
    kuramoto_proxy = compute_kuramoto_proxy_from_edges(
        reconstructed_edge_attr, edge_index, num_nodes, method
    )
    
    # Compute global sync proxy (always mean_plv for fair comparison)
    global_sync_proxy = compute_kuramoto_proxy_from_edges(
        reconstructed_edge_attr, edge_index, num_nodes, 'mean_plv'
    )
    
    # Compute errors
    abs_error = abs(kuramoto_original - kuramoto_proxy)
    rel_error = abs_error / kuramoto_original if kuramoto_original > 0 else 0.0
    
    return {
        'kuramoto_original': kuramoto_original,
        'kuramoto_proxy': kuramoto_proxy,
        'global_sync_original': global_sync_original,
        'global_sync_proxy': global_sync_proxy,
        'absolute_error': abs_error,
        'relative_error': rel_error
    }


def batch_kuramoto_comparison(
    model_output: Dict[str, torch.Tensor],
    batch_data,
    method: str = 'mean_plv'
) -> Dict[str, np.ndarray]:
    """
    Compute Kuramoto comparison for a batch of graphs.
    
    Args:
        model_output: VAE output with 'edge_attr_recon'
        batch_data: PyG Batch object with original data
        method: Proxy calculation method
        
    Returns:
        Dictionary with arrays for each metric across the batch
    """
    results = {
        'kuramoto_original': [],
        'kuramoto_proxy': [],
        'global_sync_original': [],
        'global_sync_proxy': [],
        'absolute_error': [],
        'relative_error': []
    }
    
    if 'edge_attr_recon' not in model_output:
        return {k: np.array([]) for k in results}
    
    edge_attr_recon = model_output['edge_attr_recon']
    
    # For batched data, we need to split by graph
    if hasattr(batch_data, 'batch') and batch_data.batch is not None:
        batch_idx = batch_data.batch.cpu().numpy()
        num_graphs = batch_idx.max() + 1
        
        # Get edge batch assignment
        edge_batch = batch_idx[batch_data.edge_index[0].cpu().numpy()]
        
        for g in range(num_graphs):
            # Get edges for this graph
            edge_mask = edge_batch == g
            graph_edge_attr = edge_attr_recon[edge_mask]
            
            # Get edge_index for this graph (need to reindex)
            graph_edge_index = batch_data.edge_index[:, edge_mask].cpu().numpy()
            node_offset = (batch_idx < g).sum()
            graph_edge_index = graph_edge_index - node_offset
            
            # Get num_nodes for this graph
            num_nodes = (batch_idx == g).sum()
            
            # Get original graph_attr
            if hasattr(batch_data, 'graph_attr') and batch_data.graph_attr is not None:
                # graph_attr is [batch_size, num_features] or concatenated
                if batch_data.graph_attr.dim() == 2:
                    graph_attr = batch_data.graph_attr[g]
                else:
                    # Need to figure out indexing
                    graph_attr = batch_data.graph_attr[g * 4:(g + 1) * 4]
            else:
                continue
            
            comparison = compute_kuramoto_comparison(
                graph_attr, graph_edge_attr, graph_edge_index, num_nodes, method
            )
            
            for key in results:
                results[key].append(comparison[key])
    else:
        # Single graph
        comparison = compute_kuramoto_comparison(
            batch_data.graph_attr,
            edge_attr_recon,
            batch_data.edge_index,
            batch_data.num_nodes,
            method
        )
        for key in results:
            results[key].append(comparison[key])
    
    return {k: np.array(v) for k, v in results.items()}

