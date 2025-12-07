#!/usr/bin/env python3
"""
02_cluster_sync_matrices.py

Step 2 of Phase Analysis Pipeline:
- Load vectorized sync matrices from Step 1
- Apply dimensionality reduction (UMAP/t-SNE)
- Perform clustering (HDBSCAN/KMeans)
- Visualize clusters colored by condition
- Analyze cluster composition

Output:
    output/clustering/
        clustering_results.pkl    # Cluster labels, embeddings
        embedding_2d.png          # 2D visualization
        cluster_composition.png   # Clusters vs conditions
        cluster_metrics.json      # Silhouette, etc.

Usage:
    python scripts/02_cluster_sync_matrices.py
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import logging

import numpy as np
import yaml

# Use non-GUI backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# UMAP and HDBSCAN (optional)
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    
try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

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


def reduce_dimensions(
    X: np.ndarray,
    config: Dict[str, Any],
    n_components: int = 2
) -> np.ndarray:
    """
    Apply dimensionality reduction.
    
    Args:
        X: [n_samples, n_features] data matrix
        config: Configuration dict
        n_components: Number of output dimensions
        
    Returns:
        [n_samples, n_components] reduced data
    """
    method = config['clustering']['reduction']['method']
    
    logger.info(f"Reducing to {n_components}D using {method.upper()}...")
    
    if method == 'umap':
        if not HAS_UMAP:
            raise ImportError("UMAP not installed. Run: pip install umap-learn")
        
        umap_config = config['clustering']['reduction']['umap']
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=umap_config['n_neighbors'],
            min_dist=umap_config['min_dist'],
            metric=umap_config['metric'],
            random_state=config['random_seed']
        )
        return reducer.fit_transform(X)
    
    elif method == 'tsne':
        tsne_config = config['clustering']['reduction']['tsne']
        # t-SNE works better with PCA pre-reduction for high-dim data
        if X.shape[1] > 50:
            logger.info("  Pre-reducing with PCA to 50 dims...")
            pca = PCA(n_components=50, random_state=config['random_seed'])
            X = pca.fit_transform(X)
        
        reducer = TSNE(
            n_components=n_components,
            perplexity=tsne_config['perplexity'],
            learning_rate=tsne_config['learning_rate'],
            random_state=config['random_seed']
        )
        return reducer.fit_transform(X)
    
    elif method == 'pca':
        reducer = PCA(n_components=n_components, random_state=config['random_seed'])
        return reducer.fit_transform(X)
    
    else:
        raise ValueError(f"Unknown reduction method: {method}")


def perform_clustering(
    X: np.ndarray,
    config: Dict[str, Any]
) -> np.ndarray:
    """
    Perform clustering on data.
    
    Args:
        X: [n_samples, n_features] data matrix
        config: Configuration dict
        
    Returns:
        [n_samples] cluster labels
    """
    algorithm = config['clustering']['algorithm']
    
    logger.info(f"Clustering with {algorithm.upper()}...")
    
    if algorithm == 'hdbscan':
        if not HAS_HDBSCAN:
            raise ImportError("HDBSCAN not installed. Run: pip install hdbscan")
        
        hdb_config = config['clustering']['hdbscan']
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=hdb_config['min_cluster_size'],
            min_samples=hdb_config['min_samples'],
            cluster_selection_epsilon=hdb_config['cluster_selection_epsilon']
        )
        return clusterer.fit_predict(X)
    
    elif algorithm == 'kmeans':
        km_config = config['clustering']['kmeans']
        clusterer = KMeans(
            n_clusters=km_config['n_clusters'],
            n_init=km_config['n_init'],
            random_state=config['random_seed']
        )
        return clusterer.fit_predict(X)
    
    elif algorithm == 'gmm':
        gmm_config = config['clustering']['gmm']
        clusterer = GaussianMixture(
            n_components=gmm_config['n_components'],
            covariance_type=gmm_config['covariance_type'],
            random_state=config['random_seed']
        )
        return clusterer.fit_predict(X)
    
    else:
        raise ValueError(f"Unknown clustering algorithm: {algorithm}")


def plot_embedding_2d(
    embedding: np.ndarray,
    labels: np.ndarray,
    conditions: np.ndarray,
    kuramoto: np.ndarray,
    output_dir: Path,
    config: Dict[str, Any]
):
    """
    Create 2D embedding visualizations.
    """
    viz_config = config['visualization']
    condition_colors = viz_config['cmap_conditions']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Color by cluster
    ax = axes[0]
    unique_labels = np.unique(labels)
    n_clusters = len([l for l in unique_labels if l >= 0])
    
    for label in unique_labels:
        mask = labels == label
        if label == -1:
            # Noise points (HDBSCAN)
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                      c='gray', alpha=0.3, s=10, label='Noise')
        else:
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                      alpha=viz_config['alpha'], s=20, label=f'Cluster {label}')
    
    ax.set_title(f'Clusters ({n_clusters} found)', fontsize=14, fontweight='bold')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(loc='best', fontsize=8)
    
    # 2. Color by condition
    ax = axes[1]
    for condition in np.unique(conditions):
        mask = conditions == condition
        color = condition_colors.get(condition, 'gray')
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                  c=color, alpha=viz_config['alpha'], s=20, label=condition)
    
    ax.set_title('By Condition', fontsize=14, fontweight='bold')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(loc='best')
    
    # 3. Color by Kuramoto
    ax = axes[2]
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1],
                        c=kuramoto, cmap='viridis', alpha=viz_config['alpha'], s=20)
    plt.colorbar(scatter, ax=ax, label='Kuramoto R')
    ax.set_title('By Kuramoto Order', fontsize=14, fontweight='bold')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'embedding_2d.png', dpi=viz_config['dpi'], bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved embedding plot to {output_dir / 'embedding_2d.png'}")


def plot_cluster_composition(
    labels: np.ndarray,
    conditions: np.ndarray,
    bands: np.ndarray,
    output_dir: Path,
    config: Dict[str, Any],
    is_multiband: bool = False
):
    """
    Analyze and visualize cluster composition.
    """
    viz_config = config['visualization']
    condition_colors = viz_config['cmap_conditions']
    
    unique_clusters = sorted([l for l in np.unique(labels) if l >= 0])
    unique_conditions = np.unique(conditions)
    
    if is_multiband:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax = axes[0]
    
    # 1. Cluster vs Condition
    composition = np.zeros((len(unique_clusters), len(unique_conditions)))
    
    for i, cluster in enumerate(unique_clusters):
        cluster_mask = labels == cluster
        for j, condition in enumerate(unique_conditions):
            condition_mask = conditions == condition
            composition[i, j] = (cluster_mask & condition_mask).sum()
    
    # Normalize rows (avoid div by zero)
    row_sums = composition.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    composition_norm = composition / row_sums
    
    x = np.arange(len(unique_clusters))
    width = 0.8 / len(unique_conditions)
    
    for j, condition in enumerate(unique_conditions):
        color = condition_colors.get(condition, 'gray')
        ax.bar(x + j * width, composition_norm[:, j], width, 
               label=condition, color=color, alpha=0.8)
    
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Proportion', fontsize=12)
    ax.set_title('Cluster Composition by Condition', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(unique_conditions) - 1) / 2)
    ax.set_xticklabels([f'C{c}' for c in unique_clusters])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if not is_multiband:
        # 2. Cluster vs Band (only for single-band mode)
        ax = axes[1]
        unique_bands = np.unique(bands)
        composition_band = np.zeros((len(unique_clusters), len(unique_bands)))
        
        for i, cluster in enumerate(unique_clusters):
            cluster_mask = labels == cluster
            for j, band in enumerate(unique_bands):
                band_mask = bands == band
                composition_band[i, j] = (cluster_mask & band_mask).sum()
        
        row_sums = composition_band.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        composition_band_norm = composition_band / row_sums
        
        im = ax.imshow(composition_band_norm, aspect='auto', cmap='YlOrRd')
        ax.set_xticks(np.arange(len(unique_bands)))
        ax.set_yticks(np.arange(len(unique_clusters)))
        ax.set_xticklabels(unique_bands, rotation=45, ha='right')
        ax.set_yticklabels([f'Cluster {c}' for c in unique_clusters])
        ax.set_xlabel('Frequency Band', fontsize=12)
        ax.set_ylabel('Cluster', fontsize=12)
        ax.set_title('Cluster Composition by Band', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Proportion')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cluster_composition.png', dpi=viz_config['dpi'], bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved composition plot to {output_dir / 'cluster_composition.png'}")


def plot_kuramoto_by_cluster(
    labels: np.ndarray,
    kuramoto: np.ndarray,
    conditions: np.ndarray,
    output_dir: Path,
    config: Dict[str, Any]
):
    """
    Analyze Kuramoto distribution per cluster.
    """
    viz_config = config['visualization']
    
    unique_clusters = sorted([l for l in np.unique(labels) if l >= 0])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Kuramoto distribution per cluster
    ax = axes[0]
    data_for_box = [kuramoto[labels == c] for c in unique_clusters]
    bp = ax.boxplot(data_for_box, labels=[f'C{c}' for c in unique_clusters], patch_artist=True)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Kuramoto R', fontsize=12)
    ax.set_title('Kuramoto Order by Cluster', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 2. Kuramoto by condition (for reference)
    ax = axes[1]
    condition_colors = viz_config['cmap_conditions']
    unique_conditions = np.unique(conditions)
    
    data_by_condition = [kuramoto[conditions == c] for c in unique_conditions]
    bp = ax.boxplot(data_by_condition, labels=unique_conditions, patch_artist=True)
    
    for patch, condition in zip(bp['boxes'], unique_conditions):
        patch.set_facecolor(condition_colors.get(condition, 'gray'))
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Condition', fontsize=12)
    ax.set_ylabel('Kuramoto R', fontsize=12)
    ax.set_title('Kuramoto Order by Condition', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'kuramoto_analysis.png', dpi=viz_config['dpi'], bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved Kuramoto analysis to {output_dir / 'kuramoto_analysis.png'}")


def main():
    parser = argparse.ArgumentParser(description='Cluster sync matrices')
    parser.add_argument('--config', type=str,
                       default='config/config.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    
    # Change to script directory for relative paths
    script_dir = Path(__file__).parent.parent
    config_path = script_dir / args.config
    
    logger.info("=" * 70)
    logger.info("PHASE ANALYSIS - Step 2: Cluster Sync Matrices")
    logger.info("=" * 70)
    
    # Load config
    config = load_config(config_path)
    
    # Check dependencies
    if config['clustering']['reduction']['method'] == 'umap' and not HAS_UMAP:
        logger.error("UMAP not installed. Run: pip install umap-learn")
        sys.exit(1)
    
    if config['clustering']['algorithm'] == 'hdbscan' and not HAS_HDBSCAN:
        logger.error("HDBSCAN not installed. Run: pip install hdbscan")
        sys.exit(1)
    
    # Setup paths
    base_output = Path(config['paths']['output_dir'])
    sync_data_path = base_output / 'sync_matrices' / 'sync_data.pkl'
    kuramoto_data_path = base_output / 'sync_matrices' / 'kuramoto_data.pkl'
    output_dir = base_output / 'clustering'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info(f"Loading sync data from {sync_data_path}")
    with open(sync_data_path, 'rb') as f:
        sync_data = pickle.load(f)
    
    with open(kuramoto_data_path, 'rb') as f:
        kuramoto_data = pickle.load(f)
    
    X = sync_data['sync_vectors']
    conditions = np.array(sync_data['metadata']['conditions'])
    subjects = np.array(sync_data['metadata']['subjects'])
    kuramoto = kuramoto_data['kuramoto_means']
    
    # Check if multiband format
    is_multiband = sync_data.get('multiband', False)
    bands_list = sync_data.get('bands', [])
    
    logger.info(f"Data shape: {X.shape}")
    logger.info(f"Conditions: {np.unique(conditions)}")
    
    if is_multiband:
        logger.info(f"Multi-band format: {bands_list}")
        logger.info(f"  Features per band: {sync_data.get('n_features_per_band', 'unknown')}")
        # Create dummy bands array (all epochs have all bands concatenated)
        bands = np.array(['multiband'] * len(conditions))
    else:
        bands = np.array(sync_data['metadata'].get('bands', ['unknown'] * len(conditions)))
        logger.info(f"Single-band format, bands: {np.unique(bands)}")
    
    # Standardize features
    logger.info("Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Dimensionality reduction for clustering
    n_components_cluster = config['clustering']['reduction']['n_components_clustering']
    X_reduced_cluster = reduce_dimensions(X_scaled, config, n_components=n_components_cluster)
    
    # Dimensionality reduction for visualization
    X_reduced_2d = reduce_dimensions(X_scaled, config, n_components=2)
    
    # Clustering
    labels = perform_clustering(X_reduced_cluster, config)
    
    # Compute metrics
    unique_labels = np.unique(labels)
    n_clusters = len([l for l in unique_labels if l >= 0])
    n_noise = (labels == -1).sum()
    
    logger.info(f"Found {n_clusters} clusters, {n_noise} noise points")
    
    # Compute clustering quality metrics (excluding noise)
    valid_mask = labels >= 0
    if valid_mask.sum() > 0 and n_clusters > 1:
        silhouette = silhouette_score(X_reduced_cluster[valid_mask], labels[valid_mask])
        calinski = calinski_harabasz_score(X_reduced_cluster[valid_mask], labels[valid_mask])
    else:
        silhouette = 0.0
        calinski = 0.0
    
    logger.info(f"Silhouette score: {silhouette:.4f}")
    logger.info(f"Calinski-Harabasz score: {calinski:.2f}")
    
    # Save results
    results = {
        'labels': labels,
        'embedding_2d': X_reduced_2d,
        'embedding_cluster': X_reduced_cluster,
        'conditions': conditions,
        'bands': bands,
        'subjects': subjects,
        'kuramoto': kuramoto,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'silhouette': silhouette,
        'calinski_harabasz': calinski,
    }
    
    with open(output_dir / 'clustering_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    logger.info(f"Saved clustering results to {output_dir / 'clustering_results.pkl'}")
    
    # Save metrics as JSON
    metrics = {
        'n_clusters': n_clusters,
        'n_noise': int(n_noise),
        'n_samples': len(labels),
        'silhouette_score': float(silhouette),
        'calinski_harabasz_score': float(calinski),
        'reduction_method': config['clustering']['reduction']['method'],
        'clustering_algorithm': config['clustering']['algorithm'],
    }
    
    with open(output_dir / 'cluster_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    
    plot_embedding_2d(X_reduced_2d, labels, conditions, kuramoto, output_dir, config)
    plot_cluster_composition(labels, conditions, bands, output_dir, config, is_multiband=is_multiband)
    plot_kuramoto_by_cluster(labels, kuramoto, conditions, output_dir, config)
    
    # Print cluster summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("CLUSTER SUMMARY")
    logger.info("=" * 70)
    
    for cluster in sorted([l for l in unique_labels if l >= 0]):
        mask = labels == cluster
        n = mask.sum()
        k_mean = kuramoto[mask].mean()
        
        # Condition breakdown
        cond_counts = {c: (mask & (conditions == c)).sum() for c in np.unique(conditions)}
        cond_str = ", ".join([f"{c}: {n}" for c, n in cond_counts.items()])
        
        logger.info(f"Cluster {cluster}: {n} samples, Kuramoto={k_mean:.3f}, [{cond_str}]")
    
    if n_noise > 0:
        logger.info(f"Noise: {n_noise} samples")
    
    logger.info("")
    logger.info("Step 2 complete! Next: python scripts/03_analyze_results.py")


if __name__ == '__main__':
    main()

