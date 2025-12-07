#!/usr/bin/env python3
"""
attention_clustering.py - Cluster GAT attention matrices to find brain states

This script:
1. Loads extracted attention matrices from extract_attention_states.py
2. Flattens/vectorizes the attention matrices
3. Applies dimensionality reduction (PCA, t-SNE, UMAP)
4. Performs clustering (K-Means, GMM, etc.)
5. Analyzes temporal dynamics of state transitions
6. Generates visualizations

The goal is to find recurring "attention states" that might represent
distinct brain connectivity patterns during DMT vs baseline conditions.

Usage:
    python attention_clustering.py --input-dir path/to/attention_states
    python attention_clustering.py --input-dir attention_states --subject S01 --method kmeans --n-clusters 5
"""

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

# Use non-GUI backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def load_attention_data(input_dir: Path, subject: Optional[str] = None):
    """
    Load extracted attention matrices.
    
    Supports both old format (tensors/ in condition dir) and new format (layer0/, layer1/ subdirs).
    
    Args:
        input_dir: Directory with extracted attention states
        subject: Optional subject filter
        
    Returns:
        Dict with attention data organized by subject/condition
    """
    input_dir = Path(input_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    data = {}
    
    # List subjects (exclude output directories)
    exclude_dirs = {'plots', 'clustering', 'analysis'}
    subject_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir() and d.name not in exclude_dirs])
    
    if subject:
        subject_dirs = [d for d in subject_dirs if d.name == subject]
    
    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        data[subject_id] = {}
        
        # List conditions
        for cond_dir in subject_dir.iterdir():
            if not cond_dir.is_dir():
                continue
            
            condition = cond_dir.name
            
            cond_data = {
                'attention': {},
                'metadata': None
            }
            
            # Check for NEW format: layer0/, layer1/ subdirectories
            layer_dirs = sorted([d for d in cond_dir.iterdir() if d.is_dir() and d.name.startswith('layer')])
            
            if layer_dirs:
                # NEW FORMAT: layer-specific directories
                for layer_dir in layer_dirs:
                    layer_name = layer_dir.name  # e.g., 'layer0'
                    tensors_dir = layer_dir / 'tensors'
                    
                    if tensors_dir.exists():
                        # Look for attention_mean.npy (new format)
                        att_file = tensors_dir / 'attention_mean.npy'
                        if att_file.exists():
                            cond_data['attention'][layer_name] = np.load(att_file)
            else:
                # OLD FORMAT: tensors/ directly in condition dir
                tensors_dir = cond_dir / 'tensors'
                if tensors_dir.exists():
                    attention_files = sorted(tensors_dir.glob('attention_layer*_mean.npy'))
                    for att_file in attention_files:
                        layer_name = att_file.stem.replace('attention_', '').replace('_mean', '')
                        cond_data['attention'][layer_name] = np.load(att_file)
            
            # Skip if no attention data found
            if not cond_data['attention']:
                continue
            
            # Load metadata
            metadata_path = cond_dir / 'metadata.pkl'
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    cond_data['metadata'] = pickle.load(f)
            
            data[subject_id][condition] = cond_data
    
    return data


def vectorize_attention_matrices(
    attention_data: Dict,
    layer: str = 'layer0',
    use_upper_triangle: bool = True,
    include_diagonal: bool = False
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Convert attention matrices to feature vectors.
    
    Args:
        attention_data: Data loaded by load_attention_data
        layer: Which layer to use ('layer0', 'layer1', or 'all')
        use_upper_triangle: Only use upper triangle (symmetric matrix)
        include_diagonal: Include diagonal elements
        
    Returns:
        Tuple of (feature_matrix [n_samples, n_features], metadata_list)
    """
    all_vectors = []
    all_metadata = []
    
    for subject_id, conditions in attention_data.items():
        for condition, cond_data in conditions.items():
            if layer == 'all':
                # Concatenate all layers
                layer_matrices = []
                for layer_name in sorted(cond_data['attention'].keys()):
                    layer_matrices.append(cond_data['attention'][layer_name])
                matrices = np.concatenate(layer_matrices, axis=2)  # Concat along last dim
            else:
                if layer not in cond_data['attention']:
                    # Try with alternative naming
                    layer_key = f"layer{layer[-1]}" if layer.startswith('layer') else layer
                    if layer_key not in cond_data['attention']:
                        logger.warning(f"Layer {layer} not found for {subject_id}/{condition}")
                        continue
                    layer = layer_key
                matrices = cond_data['attention'][layer]
            
            num_epochs = matrices.shape[0]
            num_nodes = matrices.shape[1]
            
            for epoch_idx in range(num_epochs):
                mat = matrices[epoch_idx]
                
                if use_upper_triangle:
                    # Get upper triangle indices
                    if include_diagonal:
                        indices = np.triu_indices(num_nodes, k=0)
                    else:
                        indices = np.triu_indices(num_nodes, k=1)
                    vector = mat[indices]
                else:
                    # Flatten entire matrix
                    vector = mat.flatten()
                
                all_vectors.append(vector)
                
                # Build metadata
                meta = {
                    'subject_id': subject_id,
                    'condition': condition,
                    'epoch_idx': epoch_idx,
                    'band': cond_data['metadata']['epochs'][epoch_idx]['band'] if cond_data['metadata'] else 'unknown'
                }
                all_metadata.append(meta)
    
    return np.array(all_vectors), all_metadata


def find_optimal_clusters(
    X: np.ndarray,
    min_k: int = 2,
    max_k: int = 15,
    n_components: int = 10,
    method: str = 'kmeans'
) -> Dict:
    """
    Find optimal number of clusters using silhouette and other metrics.
    
    Args:
        X: Feature matrix [n_samples, n_features]
        min_k, max_k: Range of clusters to try
        n_components: PCA components
        method: Clustering method ('kmeans', 'gmm')
        
    Returns:
        Dict with optimization results
    """
    # PCA reduction first
    pca = PCA(n_components=min(n_components, X.shape[1], X.shape[0] - 1))
    X_pca = pca.fit_transform(X)
    
    logger.info(f"PCA: {X.shape[1]} -> {X_pca.shape[1]} dimensions "
                f"(explained variance: {pca.explained_variance_ratio_.sum():.3f})")
    
    results = {
        'silhouette': [],
        'calinski': [],
        'inertia': [],  # For kmeans
        'bic': [],  # For GMM
        'k_values': list(range(min_k, max_k + 1)),
        'pca': pca,
        'X_pca': X_pca
    }
    
    for k in tqdm(range(min_k, max_k + 1), desc="Finding optimal k"):
        if method == 'kmeans':
            model = KMeans(n_clusters=k, n_init='auto', random_state=42)
            labels = model.fit_predict(X_pca)
            results['inertia'].append(model.inertia_)
        elif method == 'gmm':
            model = GaussianMixture(n_components=k, random_state=42, n_init=3)
            labels = model.fit_predict(X_pca)
            results['bic'].append(model.bic(X_pca))
        else:
            model = AgglomerativeClustering(n_clusters=k)
            labels = model.fit_predict(X_pca)
        
        # Metrics
        if len(set(labels)) > 1:
            results['silhouette'].append(silhouette_score(X_pca, labels))
            results['calinski'].append(calinski_harabasz_score(X_pca, labels))
        else:
            results['silhouette'].append(0)
            results['calinski'].append(0)
    
    # Find best k
    best_k_idx = np.argmax(results['silhouette'])
    results['best_k'] = results['k_values'][best_k_idx]
    results['best_silhouette'] = results['silhouette'][best_k_idx]
    
    logger.info(f"Best k={results['best_k']} (silhouette={results['best_silhouette']:.4f})")
    
    return results


def cluster_attention_states(
    X: np.ndarray,
    n_clusters: int,
    n_components: int = 10,
    method: str = 'kmeans',
    pca: Optional[PCA] = None
) -> Dict:
    """
    Perform clustering on attention matrices.
    
    Args:
        X: Feature matrix [n_samples, n_features]
        n_clusters: Number of clusters
        n_components: PCA components (ignored if pca provided)
        method: Clustering method
        pca: Optional pre-fitted PCA
        
    Returns:
        Dict with clustering results
    """
    # Reduce dimensions
    if pca is None:
        pca = PCA(n_components=min(n_components, X.shape[1], X.shape[0] - 1))
        X_pca = pca.fit_transform(X)
    else:
        X_pca = pca.transform(X)
    
    # Cluster
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
        labels = model.fit_predict(X_pca)
        centers = model.cluster_centers_
    elif method == 'gmm':
        model = GaussianMixture(n_components=n_clusters, random_state=42, n_init=3)
        labels = model.fit_predict(X_pca)
        centers = model.means_
    elif method == 'hierarchical':
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X_pca)
        # Compute pseudo-centers as cluster means
        centers = np.array([X_pca[labels == i].mean(axis=0) for i in range(n_clusters)])
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Metrics
    silhouette = silhouette_score(X_pca, labels) if len(set(labels)) > 1 else 0
    calinski = calinski_harabasz_score(X_pca, labels) if len(set(labels)) > 1 else 0
    
    return {
        'labels': labels,
        'centers': centers,
        'model': model,
        'pca': pca,
        'X_pca': X_pca,
        'n_clusters': n_clusters,
        'method': method,
        'silhouette': silhouette,
        'calinski': calinski
    }


def compute_tsne_embedding(X_pca: np.ndarray, perplexity: int = 30, random_state: int = 42):
    """Compute t-SNE embedding for visualization."""
    n_samples = X_pca.shape[0]
    perplexity = min(perplexity, n_samples // 4)
    
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    return tsne.fit_transform(X_pca)


def analyze_temporal_dynamics(
    labels: np.ndarray,
    metadata: List[Dict],
    n_clusters: int
) -> Dict:
    """
    Analyze temporal dynamics of cluster transitions.
    
    Args:
        labels: Cluster labels for each epoch
        metadata: Metadata for each epoch
        n_clusters: Number of clusters
        
    Returns:
        Dict with temporal analysis results
    """
    results = {
        'transition_matrix': {},  # Per condition
        'state_duration': {},
        'state_distribution': {}
    }
    
    # Group by subject and condition
    subject_cond_data = defaultdict(lambda: defaultdict(list))
    
    for idx, meta in enumerate(metadata):
        key = (meta['subject_id'], meta['condition'])
        subject_cond_data[key]['labels'].append(labels[idx])
        subject_cond_data[key]['epochs'].append(meta['epoch_idx'])
    
    for (subject, condition), data in subject_cond_data.items():
        # Sort by epoch index to ensure temporal order
        sorted_indices = np.argsort(data['epochs'])
        sorted_labels = np.array(data['labels'])[sorted_indices]
        
        # Transition matrix
        if condition not in results['transition_matrix']:
            results['transition_matrix'][condition] = np.zeros((n_clusters, n_clusters))
        
        for i in range(len(sorted_labels) - 1):
            from_state = sorted_labels[i]
            to_state = sorted_labels[i + 1]
            results['transition_matrix'][condition][from_state, to_state] += 1
        
        # State distribution
        if condition not in results['state_distribution']:
            results['state_distribution'][condition] = np.zeros(n_clusters)
        
        for label in sorted_labels:
            results['state_distribution'][condition][label] += 1
    
    # Normalize transition matrices to probabilities
    for condition in results['transition_matrix']:
        row_sums = results['transition_matrix'][condition].sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        results['transition_matrix'][condition] = results['transition_matrix'][condition] / row_sums
        
        # Normalize state distribution
        total = results['state_distribution'][condition].sum()
        if total > 0:
            results['state_distribution'][condition] /= total
    
    return results


def plot_clustering_results(
    clustering_results: Dict,
    metadata: List[Dict],
    output_dir: Path,
    temporal_analysis: Optional[Dict] = None
):
    """
    Generate visualization plots for clustering results.
    
    Args:
        clustering_results: Results from cluster_attention_states
        metadata: Epoch metadata
        output_dir: Directory to save plots
        temporal_analysis: Optional temporal analysis results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    labels = clustering_results['labels']
    X_pca = clustering_results['X_pca']
    n_clusters = clustering_results['n_clusters']
    
    # Extract condition labels
    conditions = [m['condition'] for m in metadata]
    unique_conditions = sorted(set(conditions))
    
    # Color maps
    cluster_cmap = plt.cm.get_cmap('tab10', n_clusters)
    condition_colors = {'DMT': '#E74C3C', 'EC': '#3498DB', 'EO': '#2ECC71'}
    
    # 1. t-SNE visualization colored by cluster
    logger.info("Computing t-SNE embedding...")
    X_tsne = compute_tsne_embedding(X_pca)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # By cluster
    ax = axes[0]
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                   c=[cluster_cmap(cluster_id)], label=f'State {cluster_id}',
                   alpha=0.6, s=30)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title(f'Attention States (k={n_clusters}, silhouette={clustering_results["silhouette"]:.3f})')
    ax.legend()
    
    # By condition
    ax = axes[1]
    for cond in unique_conditions:
        mask = np.array([m['condition'] == cond for m in metadata])
        color = condition_colors.get(cond, 'gray')
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                   c=color, label=cond, alpha=0.6, s=30)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('Colored by Condition')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'tsne_clusters.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. State distribution per condition
    fig, ax = plt.subplots(figsize=(10, 6))
    
    state_counts = np.zeros((len(unique_conditions), n_clusters))
    for idx, meta in enumerate(metadata):
        cond_idx = unique_conditions.index(meta['condition'])
        state_counts[cond_idx, labels[idx]] += 1
    
    # Normalize per condition
    state_counts_norm = state_counts / state_counts.sum(axis=1, keepdims=True)
    
    x = np.arange(n_clusters)
    width = 0.8 / len(unique_conditions)
    
    for i, cond in enumerate(unique_conditions):
        color = condition_colors.get(cond, 'gray')
        ax.bar(x + i * width, state_counts_norm[i], width, label=cond, color=color)
    
    ax.set_xlabel('State')
    ax.set_ylabel('Proportion')
    ax.set_title('State Distribution by Condition')
    ax.set_xticks(x + width * (len(unique_conditions) - 1) / 2)
    ax.set_xticklabels([f'S{i}' for i in range(n_clusters)])
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'state_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Transition matrices
    if temporal_analysis:
        n_conditions = len(temporal_analysis['transition_matrix'])
        fig, axes = plt.subplots(1, n_conditions, figsize=(6 * n_conditions, 5))
        if n_conditions == 1:
            axes = [axes]
        
        for idx, (cond, trans_mat) in enumerate(temporal_analysis['transition_matrix'].items()):
            ax = axes[idx]
            sns.heatmap(trans_mat, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                       cbar_kws={'label': 'Probability'})
            ax.set_xlabel('To State')
            ax.set_ylabel('From State')
            ax.set_title(f'Transition Matrix - {cond}')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'transition_matrices.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # 4. PCA explained variance
    fig, ax = plt.subplots(figsize=(8, 5))
    pca = clustering_results['pca']
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    
    ax.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
           pca.explained_variance_ratio_, alpha=0.7, label='Individual')
    ax.plot(range(1, len(cumsum) + 1), cumsum, 'ro-', label='Cumulative')
    ax.axhline(y=0.95, color='gray', linestyle='--', label='95% threshold')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title('PCA Explained Variance')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pca_variance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. State centroids visualization (if attention matrices can be reconstructed)
    # This requires the original matrix dimensions
    
    logger.info(f"Plots saved to: {output_dir}")


def plot_elbow_curves(optimization_results: Dict, output_dir: Path):
    """Plot elbow curves for cluster optimization."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    k_values = optimization_results['k_values']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Silhouette
    ax = axes[0]
    ax.plot(k_values, optimization_results['silhouette'], 'bo-')
    best_k = optimization_results['best_k']
    ax.axvline(x=best_k, color='red', linestyle='--', label=f'Best k={best_k}')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Score vs k')
    ax.legend()
    
    # Calinski-Harabasz
    ax = axes[1]
    ax.plot(k_values, optimization_results['calinski'], 'go-')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Calinski-Harabasz Index')
    ax.set_title('Calinski-Harabasz vs k')
    
    # Inertia (for kmeans) or BIC (for GMM)
    ax = axes[2]
    if optimization_results['inertia']:
        ax.plot(k_values, optimization_results['inertia'], 'ro-')
        ax.set_ylabel('Inertia')
        ax.set_title('Elbow Curve (Inertia)')
    elif optimization_results['bic']:
        ax.plot(k_values, optimization_results['bic'], 'mo-')
        ax.set_ylabel('BIC')
        ax.set_title('BIC vs k')
    ax.set_xlabel('Number of Clusters')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'elbow_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Cluster GAT attention matrices to find brain states',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Directory with extracted attention states')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: input_dir/clustering)')
    parser.add_argument('--subject', type=str, default=None,
                       help='Process only this subject')
    parser.add_argument('--layer', type=str, default='layer0',
                       choices=['layer0', 'layer1', 'all'],
                       help='Which layer to use (default: layer0)')
    parser.add_argument('--method', type=str, default='kmeans',
                       choices=['kmeans', 'gmm', 'hierarchical'],
                       help='Clustering method (default: kmeans)')
    parser.add_argument('--n-clusters', type=int, default=None,
                       help='Number of clusters (default: auto-optimize)')
    parser.add_argument('--min-k', type=int, default=2,
                       help='Min clusters for optimization (default: 2)')
    parser.add_argument('--max-k', type=int, default=15,
                       help='Max clusters for optimization (default: 15)')
    parser.add_argument('--n-components', type=int, default=10,
                       help='PCA components (default: 10)')
    parser.add_argument('--no-upper-triangle', action='store_true',
                       help='Use full matrix instead of upper triangle')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_dir / 'clustering'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info(f"Loading attention data from: {input_dir}")
    attention_data = load_attention_data(input_dir, subject=args.subject)
    
    if not attention_data:
        logger.error("No attention data found!")
        sys.exit(1)
    
    # Log data summary
    total_epochs = 0
    for subj, conditions in attention_data.items():
        for cond, data in conditions.items():
            if data['attention']:
                n_epochs = list(data['attention'].values())[0].shape[0]
                total_epochs += n_epochs
                logger.info(f"  {subj}/{cond}: {n_epochs} epochs")
    
    logger.info(f"Total epochs: {total_epochs}")
    
    # Vectorize attention matrices
    logger.info(f"Vectorizing attention matrices (layer={args.layer})...")
    X, metadata = vectorize_attention_matrices(
        attention_data,
        layer=args.layer,
        use_upper_triangle=not args.no_upper_triangle
    )
    
    logger.info(f"Feature matrix shape: {X.shape}")
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Find optimal clusters or use specified
    if args.n_clusters is None:
        logger.info("Optimizing number of clusters...")
        opt_results = find_optimal_clusters(
            X_scaled,
            min_k=args.min_k,
            max_k=args.max_k,
            n_components=args.n_components,
            method=args.method
        )
        n_clusters = opt_results['best_k']
        plot_elbow_curves(opt_results, output_dir)
    else:
        n_clusters = args.n_clusters
        opt_results = None
    
    # Perform clustering
    logger.info(f"Clustering with k={n_clusters} ({args.method})...")
    clustering_results = cluster_attention_states(
        X_scaled,
        n_clusters=n_clusters,
        n_components=args.n_components,
        method=args.method,
        pca=opt_results['pca'] if opt_results else None
    )
    
    # Temporal analysis
    logger.info("Analyzing temporal dynamics...")
    temporal_analysis = analyze_temporal_dynamics(
        clustering_results['labels'],
        metadata,
        n_clusters
    )
    
    # Generate plots
    logger.info("Generating visualizations...")
    plot_clustering_results(
        clustering_results,
        metadata,
        output_dir,
        temporal_analysis
    )
    
    # Save results
    results_to_save = {
        'labels': clustering_results['labels'],
        'n_clusters': n_clusters,
        'method': args.method,
        'layer': args.layer,
        'silhouette': clustering_results['silhouette'],
        'calinski': clustering_results['calinski'],
        'temporal_analysis': temporal_analysis,
        'metadata': metadata,
        'n_samples': len(metadata),
        'n_features': X.shape[1],
        'pca_variance_explained': clustering_results['pca'].explained_variance_ratio_.sum()
    }
    
    with open(output_dir / 'clustering_results.pkl', 'wb') as f:
        pickle.dump(results_to_save, f)
    
    # Print summary
    print("\n" + "=" * 60)
    print("CLUSTERING RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Epochs analyzed: {len(metadata)}")
    print(f"  Feature dimension: {X.shape[1]}")
    print(f"  PCA components: {args.n_components}")
    print(f"  Variance explained: {results_to_save['pca_variance_explained']:.3f}")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Silhouette score: {clustering_results['silhouette']:.4f}")
    print(f"  Calinski-Harabasz: {clustering_results['calinski']:.2f}")
    print("=" * 60)
    
    # State distribution per condition
    print("\nState Distribution per Condition:")
    for cond, dist in temporal_analysis['state_distribution'].items():
        dist_str = ", ".join([f"S{i}:{d:.1%}" for i, d in enumerate(dist)])
        print(f"  {cond}: {dist_str}")
    
    print(f"\nResults saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()

