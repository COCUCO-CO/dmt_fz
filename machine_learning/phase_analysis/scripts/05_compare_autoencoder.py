#!/usr/bin/env python3
"""
05_compare_autoencoder.py

Compare DMT clustering results between:
1. Phase synchronization analysis (from step 4)
2. Autoencoder latent space

This helps validate if both approaches capture similar brain states.

Output:
    output/comparison/
        ae_dmt_clustering.pkl       # Autoencoder DMT clustering
        ae_subject_fingerprints.csv
        comparison_summary.json
        fingerprint_correlation.png  # Correlation between methods
        embedding_comparison.png     # Side-by-side embeddings

Usage:
    python scripts/05_compare_autoencoder.py
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple
import logging

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

# Use non-GUI backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

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


def convert_to_serializable(obj):
    """Recursively convert numpy types to native Python types."""
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


def load_autoencoder_embeddings(ae_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load autoencoder embeddings and filter to DMT.
    
    Returns:
        embeddings: [n_samples, 64] latent vectors
        subjects: subject IDs (cleaned)
        bands: frequency bands
    """
    emb_dir = ae_dir / 'output' / 'embeddings'
    
    all_embeddings = []
    all_subjects = []
    all_conditions = []
    all_bands = []
    
    for split in ['train', 'val', 'test']:
        path = emb_dir / f'{split}_embeddings.pkl'
        if path.exists():
            with open(path, 'rb') as f:
                data = pickle.load(f)
            all_embeddings.append(data['latent_embeddings'])
            all_subjects.extend(data['subjects'])
            all_conditions.extend(data['conditions'])
            all_bands.extend(data['bands'])
    
    embeddings = np.vstack(all_embeddings)
    subjects = np.array(all_subjects)
    conditions = np.array(all_conditions)
    bands = np.array(all_bands)
    
    logger.info(f"Loaded {len(embeddings)} total samples")
    
    # Filter to DMT only
    dmt_mask = conditions == 'DMT'
    embeddings = embeddings[dmt_mask]
    subjects = subjects[dmt_mask]
    bands = bands[dmt_mask]
    
    # Clean subject IDs (remove condition suffix)
    subjects = np.array([s.split('-')[0] for s in subjects])
    
    logger.info(f"DMT samples: {len(embeddings)}")
    logger.info(f"Unique subjects: {len(np.unique(subjects))}")
    logger.info(f"Unique bands: {np.unique(bands)}")
    
    return embeddings, subjects, bands


def cluster_embeddings(
    X: np.ndarray,
    min_cluster_size: int = 100,
    normalize_within_subject: bool = False,
    subjects: np.ndarray = None,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Cluster autoencoder embeddings using UMAP + HDBSCAN.
    """
    X_processed = X.copy()
    
    # Within-subject normalization if requested
    if normalize_within_subject and subjects is not None:
        logger.info("Applying within-subject normalization...")
        for subj in np.unique(subjects):
            mask = subjects == subj
            subj_data = X_processed[mask]
            mean = subj_data.mean(axis=0)
            std = subj_data.std(axis=0)
            std[std == 0] = 1
            X_processed[mask] = (subj_data - mean) / std
    
    # Standardize
    logger.info("Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_processed)
    
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
        min_cluster_size=min_cluster_size,
        min_samples=10,
        cluster_selection_epsilon=0.0,
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
    else:
        silhouette = 0.0
    
    logger.info(f"Silhouette score: {silhouette:.4f}")
    
    metrics = {
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'silhouette': silhouette
    }
    
    return labels, embedding_10d, embedding_2d, metrics


def compute_fingerprints(labels: np.ndarray, subjects: np.ndarray, n_clusters: int) -> pd.DataFrame:
    """Compute cluster distribution per subject."""
    unique_subjects = sorted(set(subjects))
    
    fingerprints = {}
    for subj in unique_subjects:
        mask = subjects == subj
        subj_labels = labels[mask]
        
        valid_labels = subj_labels[subj_labels != -1]
        counts = np.zeros(n_clusters)
        for label in valid_labels:
            if label < n_clusters:
                counts[label] += 1
        
        if counts.sum() > 0:
            fingerprints[subj] = counts / counts.sum()
        else:
            fingerprints[subj] = counts
    
    df = pd.DataFrame(fingerprints).T
    df.columns = [f'C{i}' for i in range(n_clusters)]
    df.index.name = 'Subject'
    
    return df


def compare_fingerprints(fp_phase: pd.DataFrame, fp_ae: pd.DataFrame) -> pd.DataFrame:
    """
    Compare subject fingerprints between phase sync and autoencoder.
    
    Returns correlation matrix between subjects across methods.
    """
    # Get common subjects
    common_subjects = sorted(set(fp_phase.index) & set(fp_ae.index))
    logger.info(f"Common subjects: {len(common_subjects)}")
    
    # Compute correlation per subject between their two fingerprints
    correlations = {}
    for subj in common_subjects:
        phase_fp = fp_phase.loc[subj].values
        ae_fp = fp_ae.loc[subj].values
        
        # Can't correlate if different number of clusters
        # Instead, we compare the subject similarity matrices
        correlations[subj] = {
            'phase_fp': phase_fp,
            'ae_fp': ae_fp
        }
    
    return correlations, common_subjects


def plot_embedding_comparison(
    emb_phase: np.ndarray,
    emb_ae: np.ndarray,
    subjects_phase: np.ndarray,
    subjects_ae: np.ndarray,
    labels_phase: np.ndarray,
    labels_ae: np.ndarray,
    output_path: Path
):
    """Side-by-side embedding visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Phase sync - by subject
    ax = axes[0, 0]
    unique_subjects = sorted(set(subjects_phase))[:10]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, subj in enumerate(unique_subjects):
        mask = subjects_phase == subj
        ax.scatter(emb_phase[mask, 0], emb_phase[mask, 1], 
                   c=[colors[i]], label=subj, alpha=0.5, s=10)
    ax.set_title('Phase Sync - By Subject')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(fontsize=7, ncol=2)
    
    # Phase sync - by cluster
    ax = axes[0, 1]
    n_clusters = len(set(labels_phase)) - (1 if -1 in labels_phase else 0)
    for label in range(n_clusters):
        mask = labels_phase == label
        ax.scatter(emb_phase[mask, 0], emb_phase[mask, 1], alpha=0.5, s=10)
    noise_mask = labels_phase == -1
    if noise_mask.any():
        ax.scatter(emb_phase[noise_mask, 0], emb_phase[noise_mask, 1], 
                   c='lightgray', alpha=0.2, s=5)
    ax.set_title(f'Phase Sync - By Cluster (n={n_clusters})')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    
    # Autoencoder - by subject
    ax = axes[1, 0]
    unique_subjects = sorted(set(subjects_ae))[:10]
    for i, subj in enumerate(unique_subjects):
        mask = subjects_ae == subj
        ax.scatter(emb_ae[mask, 0], emb_ae[mask, 1], 
                   c=[colors[i]], label=subj, alpha=0.5, s=10)
    ax.set_title('Autoencoder - By Subject')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(fontsize=7, ncol=2)
    
    # Autoencoder - by cluster
    ax = axes[1, 1]
    n_clusters = len(set(labels_ae)) - (1 if -1 in labels_ae else 0)
    for label in range(n_clusters):
        mask = labels_ae == label
        ax.scatter(emb_ae[mask, 0], emb_ae[mask, 1], alpha=0.5, s=10)
    noise_mask = labels_ae == -1
    if noise_mask.any():
        ax.scatter(emb_ae[noise_mask, 0], emb_ae[noise_mask, 1], 
                   c='lightgray', alpha=0.2, s=5)
    ax.set_title(f'Autoencoder - By Cluster (n={n_clusters})')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved embedding comparison to {output_path}")


def plot_similarity_comparison(
    fp_phase: pd.DataFrame,
    fp_ae: pd.DataFrame,
    common_subjects: list,
    output_path: Path
):
    """Compare subject similarity matrices between methods."""
    # Compute similarity matrices
    sim_phase = fp_phase.loc[common_subjects].T.corr()
    sim_ae = fp_ae.loc[common_subjects].T.corr()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Phase sync similarity
    ax = axes[0]
    sns.heatmap(sim_phase, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                ax=ax, square=True, xticklabels=True, yticklabels=True,
                cbar_kws={'label': 'Correlation'})
    ax.set_title('Phase Sync\nSubject Similarity')
    ax.tick_params(axis='both', labelsize=7)
    
    # Autoencoder similarity
    ax = axes[1]
    sns.heatmap(sim_ae, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                ax=ax, square=True, xticklabels=True, yticklabels=True,
                cbar_kws={'label': 'Correlation'})
    ax.set_title('Autoencoder\nSubject Similarity')
    ax.tick_params(axis='both', labelsize=7)
    
    # Correlation between similarity matrices
    ax = axes[2]
    
    # Flatten upper triangles
    phase_flat = sim_phase.values[np.triu_indices(len(common_subjects), k=1)]
    ae_flat = sim_ae.values[np.triu_indices(len(common_subjects), k=1)]
    
    ax.scatter(phase_flat, ae_flat, alpha=0.6)
    
    # Fit line
    r, p = stats.pearsonr(phase_flat, ae_flat)
    z = np.polyfit(phase_flat, ae_flat, 1)
    line = np.poly1d(z)
    x_line = np.linspace(-1, 1, 100)
    ax.plot(x_line, line(x_line), 'r--', label=f'r={r:.3f}, p={p:.2e}')
    
    ax.set_xlabel('Phase Sync Similarity')
    ax.set_ylabel('Autoencoder Similarity')
    ax.set_title('Method Comparison')
    ax.legend()
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(0, color='gray', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved similarity comparison to {output_path}")
    
    return r, p


def plot_fingerprint_heatmaps(
    fp_phase: pd.DataFrame,
    fp_ae: pd.DataFrame,
    common_subjects: list,
    output_path: Path
):
    """Plot fingerprint heatmaps side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    
    # Phase sync
    ax = axes[0]
    fp_phase_common = fp_phase.loc[common_subjects]
    # Select top clusters by variance
    top_clusters = fp_phase_common.var().nlargest(min(15, len(fp_phase_common.columns))).index
    sns.heatmap(fp_phase_common[top_clusters], cmap='YlOrRd', ax=ax,
                xticklabels=True, yticklabels=True,
                cbar_kws={'label': 'Proportion'})
    ax.set_title('Phase Sync Fingerprints')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Subject')
    
    # Autoencoder
    ax = axes[1]
    fp_ae_common = fp_ae.loc[common_subjects]
    top_clusters = fp_ae_common.var().nlargest(min(15, len(fp_ae_common.columns))).index
    sns.heatmap(fp_ae_common[top_clusters], cmap='YlOrRd', ax=ax,
                xticklabels=True, yticklabels=True,
                cbar_kws={'label': 'Proportion'})
    ax.set_title('Autoencoder Fingerprints')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Subject')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved fingerprint heatmaps to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare Phase Sync vs Autoencoder')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--min-cluster-size', type=int, default=100)
    parser.add_argument('--normalize-within-subject', action='store_true')
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("PHASE ANALYSIS - Step 5: Compare with Autoencoder")
    logger.info("=" * 70)
    
    # Paths
    script_dir = Path(__file__).parent.parent
    config_path = script_dir / 'config' / args.config
    config = yaml.safe_load(open(config_path))
    
    output_base = Path(config['paths']['output_dir'])
    ae_dir = script_dir.parent / 'autoencoder'
    output_dir = output_base / 'comparison'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load phase sync results
    logger.info("\n[Loading Phase Sync Results]")
    phase_results_path = output_base / 'dmt_analysis' / 'dmt_clustering.pkl'
    with open(phase_results_path, 'rb') as f:
        phase_results = pickle.load(f)
    
    fp_phase = phase_results['fingerprints']
    emb_phase = phase_results['embedding_2d']
    labels_phase = phase_results['labels']
    subjects_phase = phase_results['subjects']
    
    logger.info(f"Phase sync: {len(emb_phase)} samples, {phase_results['metrics']['n_clusters']} clusters")
    
    # Load and cluster autoencoder embeddings
    logger.info("\n[Processing Autoencoder Embeddings]")
    embeddings_ae, subjects_ae, bands_ae = load_autoencoder_embeddings(ae_dir)
    
    labels_ae, emb_10d_ae, emb_2d_ae, metrics_ae = cluster_embeddings(
        embeddings_ae,
        min_cluster_size=args.min_cluster_size,
        normalize_within_subject=args.normalize_within_subject,
        subjects=subjects_ae,
        random_state=config['random_seed']
    )
    
    # Compute autoencoder fingerprints
    fp_ae = compute_fingerprints(labels_ae, subjects_ae, metrics_ae['n_clusters'])
    
    # Compare fingerprints
    logger.info("\n[Comparing Methods]")
    _, common_subjects = compare_fingerprints(fp_phase, fp_ae)
    
    # Generate visualizations
    logger.info("\n[Generating Visualizations]")
    
    plot_embedding_comparison(
        emb_phase, emb_2d_ae,
        subjects_phase, subjects_ae,
        labels_phase, labels_ae,
        output_dir / 'embedding_comparison.png'
    )
    
    r, p = plot_similarity_comparison(
        fp_phase, fp_ae, common_subjects,
        output_dir / 'similarity_comparison.png'
    )
    
    plot_fingerprint_heatmaps(
        fp_phase, fp_ae, common_subjects,
        output_dir / 'fingerprint_heatmaps.png'
    )
    
    # Save results
    ae_results = {
        'labels': labels_ae,
        'embedding_10d': emb_10d_ae,
        'embedding_2d': emb_2d_ae,
        'subjects': subjects_ae,
        'bands': bands_ae,
        'metrics': metrics_ae,
        'fingerprints': fp_ae
    }
    with open(output_dir / 'ae_dmt_clustering.pkl', 'wb') as f:
        pickle.dump(ae_results, f)
    
    fp_ae.to_csv(output_dir / 'ae_subject_fingerprints.csv')
    
    # Summary
    summary = {
        'phase_sync': {
            'n_samples': len(emb_phase),
            'n_clusters': phase_results['metrics']['n_clusters'],
            'n_noise': phase_results['metrics']['n_noise'],
            'silhouette': phase_results['metrics']['silhouette']
        },
        'autoencoder': {
            'n_samples': len(embeddings_ae),
            'n_clusters': metrics_ae['n_clusters'],
            'n_noise': metrics_ae['n_noise'],
            'silhouette': metrics_ae['silhouette']
        },
        'comparison': {
            'common_subjects': len(common_subjects),
            'similarity_correlation': r,
            'similarity_pvalue': p
        }
    }
    
    with open(output_dir / 'comparison_summary.json', 'w') as f:
        json.dump(convert_to_serializable(summary), f, indent=2)
    
    # Print summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 70)
    
    logger.info(f"\n[Phase Synchronization]")
    logger.info(f"  Samples: {len(emb_phase)}")
    logger.info(f"  Clusters: {phase_results['metrics']['n_clusters']}")
    logger.info(f"  Silhouette: {phase_results['metrics']['silhouette']:.4f}")
    
    logger.info(f"\n[Autoencoder]")
    logger.info(f"  Samples: {len(embeddings_ae)}")
    logger.info(f"  Clusters: {metrics_ae['n_clusters']}")
    logger.info(f"  Silhouette: {metrics_ae['silhouette']:.4f}")
    
    logger.info(f"\n[Method Comparison]")
    logger.info(f"  Common subjects: {len(common_subjects)}")
    logger.info(f"  Similarity matrix correlation: r={r:.4f}, p={p:.2e}")
    
    if r > 0.5 and p < 0.05:
        logger.info(f"  → SIGNIFICANT AGREEMENT between methods!")
    elif r > 0.3 and p < 0.05:
        logger.info(f"  → Moderate agreement between methods")
    else:
        logger.info(f"  → Methods capture different aspects of DMT experience")
    
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()






