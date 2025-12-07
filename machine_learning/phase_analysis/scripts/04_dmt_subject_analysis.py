#!/usr/bin/env python3
"""
04_dmt_subject_analysis.py

DMT-only analysis with per-subject profiles:
- Filter to DMT condition only
- Cluster DMT epochs
- Compute per-subject "fingerprints" (cluster distribution)
- Compare subjects by fingerprint similarity
- Identify shared vs individual-specific brain states

Output:
    output/dmt_analysis/
        dmt_clustering.pkl           # Clustering results for DMT only
        subject_fingerprints.csv     # Cluster distribution per subject
        subject_similarity.png       # Heatmap of subject similarity
        subject_clusters.png         # Hierarchical clustering of subjects
        cluster_profiles.png         # Characteristic patterns per cluster
        temporal_trajectories.png    # How subjects traverse clusters over time

Usage:
    python scripts/04_dmt_subject_analysis.py
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import logging

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform

# Use non-GUI backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# UMAP and HDBSCAN
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


def convert_to_serializable(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def load_dmt_data(sync_path: Path, normalize_within_subject: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load sync data and filter to DMT condition only.
    
    Args:
        sync_path: Path to sync data
        normalize_within_subject: If True, z-score normalize within each subject
                                  to remove individual differences
    
    Returns:
        X: sync vectors for DMT epochs
        subjects: subject IDs
        epoch_indices: original epoch indices
        kuramoto: kuramoto values
    """
    logger.info(f"Loading data from {sync_path}")
    
    with open(sync_path, 'rb') as f:
        sync_data = pickle.load(f)
    
    X_all = sync_data['sync_vectors']
    subjects_all = np.array(sync_data['metadata']['subjects'])
    conditions_all = np.array(sync_data['metadata']['conditions'])
    epoch_indices_all = np.array(sync_data['metadata']['epoch_indices'])
    
    # Filter to DMT only
    dmt_mask = conditions_all == 'DMT'
    X = X_all[dmt_mask].copy()
    subjects = subjects_all[dmt_mask]
    epoch_indices = epoch_indices_all[dmt_mask]
    
    logger.info(f"DMT samples: {len(X)} (from {len(X_all)} total)")
    logger.info(f"Unique subjects: {len(np.unique(subjects))}")
    
    # Within-subject normalization to find shared states
    if normalize_within_subject:
        logger.info("Applying within-subject normalization...")
        unique_subjects = np.unique(subjects)
        for subj in unique_subjects:
            mask = subjects == subj
            subj_data = X[mask]
            # Z-score within subject
            mean = subj_data.mean(axis=0)
            std = subj_data.std(axis=0)
            std[std == 0] = 1  # Avoid division by zero
            X[mask] = (subj_data - mean) / std
    
    # Load kuramoto if available
    kuramoto_path = sync_path.parent / 'kuramoto_data.pkl'
    if kuramoto_path.exists():
        with open(kuramoto_path, 'rb') as f:
            kuramoto_data = pickle.load(f)
        kuramoto_all = kuramoto_data['kuramoto_means']
        kuramoto = kuramoto_all[dmt_mask]
    else:
        kuramoto = np.zeros(len(X))
    
    return X, subjects, epoch_indices, kuramoto


def cluster_dmt(
    X: np.ndarray,
    config: Dict[str, Any],
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Cluster DMT epochs using UMAP + HDBSCAN.
    
    Returns:
        labels: cluster labels
        embedding_10d: 10D UMAP embedding
        embedding_2d: 2D UMAP embedding for visualization
        metrics: clustering quality metrics
    """
    logger.info("Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # UMAP to 10D for clustering
    logger.info("Reducing to 10D using UMAP...")
    reducer_10d = umap.UMAP(
        n_components=10,
        n_neighbors=15,
        min_dist=0.1,
        metric='euclidean',
        random_state=random_state
    )
    embedding_10d = reducer_10d.fit_transform(X_scaled)
    
    # UMAP to 2D for visualization
    logger.info("Reducing to 2D using UMAP...")
    reducer_2d = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric='euclidean',
        random_state=random_state
    )
    embedding_2d = reducer_2d.fit_transform(X_scaled)
    
    # HDBSCAN clustering
    logger.info("Clustering with HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config.get('min_cluster_size', 30),
        min_samples=config.get('min_samples', 10),
        cluster_selection_epsilon=config.get('cluster_selection_epsilon', 0.0),
        cluster_selection_method='eom'
    )
    labels = clusterer.fit_predict(embedding_10d)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    logger.info(f"Found {n_clusters} clusters, {n_noise} noise points")
    
    # Compute metrics
    valid_mask = labels != -1
    if valid_mask.sum() > 1 and n_clusters > 1:
        silhouette = silhouette_score(embedding_10d[valid_mask], labels[valid_mask])
        calinski = calinski_harabasz_score(embedding_10d[valid_mask], labels[valid_mask])
    else:
        silhouette = 0.0
        calinski = 0.0
    
    logger.info(f"Silhouette score: {silhouette:.4f}")
    
    metrics = {
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'silhouette': silhouette,
        'calinski_harabasz': calinski
    }
    
    return labels, embedding_10d, embedding_2d, metrics


def compute_subject_fingerprints(
    labels: np.ndarray,
    subjects: np.ndarray,
    n_clusters: int
) -> pd.DataFrame:
    """
    Compute cluster distribution (fingerprint) for each subject.
    
    Returns:
        DataFrame with subjects as rows, clusters as columns, values = proportion
    """
    unique_subjects = sorted(set(subjects))
    
    fingerprints = {}
    for subj in unique_subjects:
        mask = subjects == subj
        subj_labels = labels[mask]
        
        # Count cluster occurrences (excluding noise)
        valid_labels = subj_labels[subj_labels != -1]
        counts = np.zeros(n_clusters)
        for label in valid_labels:
            if label < n_clusters:
                counts[label] += 1
        
        # Normalize to proportions
        if counts.sum() > 0:
            fingerprints[subj] = counts / counts.sum()
        else:
            fingerprints[subj] = counts
    
    df = pd.DataFrame(fingerprints).T
    df.columns = [f'C{i}' for i in range(n_clusters)]
    df.index.name = 'Subject'
    
    return df


def compute_transition_matrices(
    labels: np.ndarray,
    subjects: np.ndarray,
    epoch_indices: np.ndarray,
    n_clusters: int
) -> Dict[str, np.ndarray]:
    """
    Compute state transition matrices for each subject.
    
    Returns:
        Dict mapping subject -> transition matrix
    """
    unique_subjects = sorted(set(subjects))
    transition_matrices = {}
    
    for subj in unique_subjects:
        mask = subjects == subj
        subj_labels = labels[mask]
        subj_epochs = epoch_indices[mask]
        
        # Sort by epoch index
        sort_idx = np.argsort(subj_epochs)
        sorted_labels = subj_labels[sort_idx]
        
        # Build transition matrix
        trans_mat = np.zeros((n_clusters + 1, n_clusters + 1))  # +1 for noise
        for i in range(len(sorted_labels) - 1):
            from_state = sorted_labels[i] + 1  # Shift so noise (-1) becomes 0
            to_state = sorted_labels[i + 1] + 1
            if from_state >= 0 and to_state >= 0:
                trans_mat[from_state, to_state] += 1
        
        # Normalize rows
        row_sums = trans_mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        trans_mat = trans_mat / row_sums
        
        transition_matrices[subj] = trans_mat
    
    return transition_matrices


def compute_subject_similarity(fingerprints: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise similarity between subjects based on fingerprints.
    Uses correlation as similarity measure.
    """
    # Correlation matrix
    similarity = fingerprints.T.corr()
    return similarity


def plot_embedding_by_subject(
    embedding_2d: np.ndarray,
    subjects: np.ndarray,
    labels: np.ndarray,
    output_path: Path
):
    """Plot 2D embedding colored by subject and by cluster."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    unique_subjects = sorted(set(subjects))
    n_subjects = len(unique_subjects)
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, n_subjects)))
    
    # By subject
    ax = axes[0]
    for i, subj in enumerate(unique_subjects):
        mask = subjects == subj
        ax.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            c=[colors[i % 20]],
            label=subj if i < 10 else None,
            alpha=0.5,
            s=10
        )
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('DMT Epochs by Subject')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    
    # By cluster
    ax = axes[1]
    unique_labels = sorted(set(labels))
    n_clusters = len([l for l in unique_labels if l != -1])
    cluster_colors = plt.cm.Spectral(np.linspace(0, 1, n_clusters))
    
    # Plot noise first
    noise_mask = labels == -1
    if noise_mask.any():
        ax.scatter(
            embedding_2d[noise_mask, 0],
            embedding_2d[noise_mask, 1],
            c='lightgray',
            alpha=0.3,
            s=5,
            label='Noise'
        )
    
    # Plot clusters
    for i, label in enumerate([l for l in unique_labels if l != -1]):
        mask = labels == label
        ax.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            c=[cluster_colors[i]],
            alpha=0.6,
            s=10
        )
    
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title(f'DMT Epochs by Cluster (n={n_clusters})')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved embedding plot to {output_path}")


def plot_subject_fingerprints(
    fingerprints: pd.DataFrame,
    output_path: Path
):
    """Plot heatmap of subject fingerprints."""
    # Select top clusters by variance
    cluster_var = fingerprints.var()
    top_clusters = cluster_var.nlargest(min(30, len(cluster_var))).index
    fp_subset = fingerprints[top_clusters]
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    sns.heatmap(
        fp_subset,
        cmap='YlOrRd',
        ax=ax,
        xticklabels=True,
        yticklabels=True,
        cbar_kws={'label': 'Proportion of time'}
    )
    
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Subject')
    ax.set_title('Subject Fingerprints: Time Spent in Each Cluster')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved fingerprints plot to {output_path}")


def plot_subject_similarity(
    similarity: pd.DataFrame,
    fingerprints: pd.DataFrame,
    output_path: Path
):
    """Plot subject similarity matrix with hierarchical clustering."""
    # Hierarchical clustering of subjects
    linkage_matrix = linkage(pdist(fingerprints.values), method='ward')
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Dendrogram
    ax = axes[0]
    dendrogram(
        linkage_matrix,
        labels=fingerprints.index.tolist(),
        ax=ax,
        leaf_rotation=90
    )
    ax.set_title('Hierarchical Clustering of Subjects')
    ax.set_ylabel('Distance')
    
    # Reorder similarity matrix by dendrogram
    from scipy.cluster.hierarchy import leaves_list
    order = leaves_list(linkage_matrix)
    ordered_subjects = fingerprints.index[order]
    similarity_ordered = similarity.loc[ordered_subjects, ordered_subjects]
    
    # Similarity heatmap
    ax = axes[1]
    sns.heatmap(
        similarity_ordered,
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        square=True,
        xticklabels=True,
        yticklabels=True,
        cbar_kws={'label': 'Correlation'}
    )
    ax.set_title('Subject Similarity (Fingerprint Correlation)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved similarity plot to {output_path}")


def plot_temporal_trajectories(
    labels: np.ndarray,
    subjects: np.ndarray,
    epoch_indices: np.ndarray,
    embedding_2d: np.ndarray,
    output_path: Path,
    n_subjects_to_plot: int = 6
):
    """Plot temporal trajectories through cluster space for selected subjects."""
    unique_subjects = sorted(set(subjects))[:n_subjects_to_plot]
    
    n_cols = 3
    n_rows = (len(unique_subjects) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()
    
    for idx, subj in enumerate(unique_subjects):
        ax = axes[idx]
        mask = subjects == subj
        subj_embedding = embedding_2d[mask]
        subj_epochs = epoch_indices[mask]
        
        # Sort by epoch
        sort_idx = np.argsort(subj_epochs)
        sorted_embedding = subj_embedding[sort_idx]
        
        # Plot background (all DMT points)
        ax.scatter(
            embedding_2d[:, 0],
            embedding_2d[:, 1],
            c='lightgray',
            alpha=0.1,
            s=5
        )
        
        # Plot trajectory
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_embedding)))
        ax.scatter(
            sorted_embedding[:, 0],
            sorted_embedding[:, 1],
            c=colors,
            s=20,
            alpha=0.8
        )
        
        # Draw arrows for trajectory
        for i in range(0, len(sorted_embedding) - 1, max(1, len(sorted_embedding) // 20)):
            ax.annotate(
                '',
                xy=sorted_embedding[i + 1],
                xytext=sorted_embedding[i],
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.3, lw=0.5)
            )
        
        ax.set_title(f'{subj}')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
    
    # Hide unused axes
    for idx in range(len(unique_subjects), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Temporal Trajectories Through State Space (color = time)', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved trajectories plot to {output_path}")


def plot_cluster_kuramoto(
    labels: np.ndarray,
    kuramoto: np.ndarray,
    output_path: Path
):
    """Plot Kuramoto distribution per cluster."""
    valid_mask = labels != -1
    
    df = pd.DataFrame({
        'Cluster': labels[valid_mask],
        'Kuramoto': kuramoto[valid_mask]
    })
    
    # Sort clusters by mean Kuramoto
    cluster_means = df.groupby('Cluster')['Kuramoto'].mean().sort_values()
    cluster_order = cluster_means.index.tolist()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    sns.boxplot(
        data=df,
        x='Cluster',
        y='Kuramoto',
        order=cluster_order,
        ax=ax,
        palette='viridis'
    )
    
    ax.set_xlabel('Cluster (sorted by mean Kuramoto)')
    ax.set_ylabel('Kuramoto Order Parameter')
    ax.set_title('Synchronization Level by Cluster')
    ax.tick_params(axis='x', rotation=90)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved Kuramoto plot to {output_path}")


def identify_subject_groups(
    fingerprints: pd.DataFrame,
    n_groups: int = 3
) -> Dict[str, List[str]]:
    """
    Identify groups of subjects with similar fingerprints.
    
    Returns:
        Dict mapping group name -> list of subjects
    """
    # Hierarchical clustering
    linkage_matrix = linkage(pdist(fingerprints.values), method='ward')
    group_labels = fcluster(linkage_matrix, n_groups, criterion='maxclust')
    
    groups = defaultdict(list)
    for subj, group in zip(fingerprints.index, group_labels):
        groups[f'Group_{group}'].append(subj)
    
    return dict(groups)


def main():
    parser = argparse.ArgumentParser(description='DMT Subject Analysis')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--min-cluster-size', type=int, default=30)
    parser.add_argument('--n-subject-groups', type=int, default=3)
    parser.add_argument('--normalize-within-subject', action='store_true',
                        help='Apply within-subject normalization to find shared states')
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("PHASE ANALYSIS - Step 4: DMT Subject Analysis")
    logger.info("=" * 70)
    
    # Load config
    script_dir = Path(__file__).parent.parent
    config_path = script_dir / 'config' / args.config
    config = load_config(config_path)
    
    # Setup paths
    output_base = Path(config['paths']['output_dir'])
    sync_path = output_base / 'sync_matrices' / 'sync_data.pkl'
    output_dir = output_base / 'dmt_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load DMT data
    X, subjects, epoch_indices, kuramoto = load_dmt_data(
        sync_path, 
        normalize_within_subject=args.normalize_within_subject
    )
    
    # Cluster DMT epochs
    cluster_config = {
        'min_cluster_size': args.min_cluster_size,
        'min_samples': 10,
        'cluster_selection_epsilon': 0.0
    }
    labels, embedding_10d, embedding_2d, metrics = cluster_dmt(
        X, cluster_config, random_state=config['random_seed']
    )
    
    n_clusters = metrics['n_clusters']
    
    # Compute subject fingerprints
    logger.info("Computing subject fingerprints...")
    fingerprints = compute_subject_fingerprints(labels, subjects, n_clusters)
    
    # Compute transition matrices
    logger.info("Computing transition matrices...")
    transition_matrices = compute_transition_matrices(
        labels, subjects, epoch_indices, n_clusters
    )
    
    # Compute subject similarity
    logger.info("Computing subject similarity...")
    similarity = compute_subject_similarity(fingerprints)
    
    # Identify subject groups
    logger.info("Identifying subject groups...")
    subject_groups = identify_subject_groups(fingerprints, n_groups=args.n_subject_groups)
    
    # Save results
    logger.info("Saving results...")
    
    results = {
        'labels': labels,
        'embedding_10d': embedding_10d,
        'embedding_2d': embedding_2d,
        'subjects': subjects,
        'epoch_indices': epoch_indices,
        'kuramoto': kuramoto,
        'metrics': metrics,
        'fingerprints': fingerprints,
        'similarity': similarity,
        'transition_matrices': transition_matrices,
        'subject_groups': subject_groups
    }
    
    with open(output_dir / 'dmt_clustering.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Save fingerprints as CSV
    fingerprints.to_csv(output_dir / 'subject_fingerprints.csv')
    
    # Save summary as JSON
    summary = {
        'n_dmt_epochs': len(X),
        'n_subjects': len(np.unique(subjects)),
        'n_clusters': n_clusters,
        'n_noise': metrics['n_noise'],
        'silhouette': metrics['silhouette'],
        'subject_groups': subject_groups,
        'epochs_per_subject': {
            subj: int((subjects == subj).sum())
            for subj in sorted(set(subjects))
        }
    }
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(convert_to_serializable(summary), f, indent=2)
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    
    plot_embedding_by_subject(
        embedding_2d, subjects, labels,
        output_dir / 'embedding_by_subject.png'
    )
    
    plot_subject_fingerprints(
        fingerprints,
        output_dir / 'subject_fingerprints.png'
    )
    
    plot_subject_similarity(
        similarity, fingerprints,
        output_dir / 'subject_similarity.png'
    )
    
    plot_temporal_trajectories(
        labels, subjects, epoch_indices, embedding_2d,
        output_dir / 'temporal_trajectories.png'
    )
    
    if kuramoto.any():
        plot_cluster_kuramoto(
            labels, kuramoto,
            output_dir / 'cluster_kuramoto.png'
        )
    
    # Print summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("DMT ANALYSIS SUMMARY")
    logger.info("=" * 70)
    
    logger.info(f"\n[Data]")
    logger.info(f"  DMT epochs: {len(X)}")
    logger.info(f"  Subjects: {len(np.unique(subjects))}")
    
    logger.info(f"\n[Clustering]")
    logger.info(f"  Clusters: {n_clusters}")
    logger.info(f"  Noise: {metrics['n_noise']}")
    logger.info(f"  Silhouette: {metrics['silhouette']:.4f}")
    
    logger.info(f"\n[Subject Groups]")
    for group, members in subject_groups.items():
        logger.info(f"  {group}: {members}")
    
    # Find most similar and dissimilar subjects
    sim_values = similarity.values
    np.fill_diagonal(sim_values, np.nan)
    max_idx = np.nanargmax(sim_values)
    min_idx = np.nanargmin(sim_values)
    max_i, max_j = np.unravel_index(max_idx, sim_values.shape)
    min_i, min_j = np.unravel_index(min_idx, sim_values.shape)
    
    logger.info(f"\n[Subject Similarity]")
    logger.info(f"  Most similar: {similarity.index[max_i]} & {similarity.index[max_j]} (r={sim_values[max_i, max_j]:.3f})")
    logger.info(f"  Most different: {similarity.index[min_i]} & {similarity.index[min_j]} (r={sim_values[min_i, min_j]:.3f})")
    
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()

