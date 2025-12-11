#!/usr/bin/env python3
"""
06_compare_alpha.py

Compare DMT Alpha band analysis between:
1. Phase synchronization (Alpha band features only)
2. Autoencoder trained on Alpha only

Usage:
    python scripts/06_compare_alpha.py
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
from scipy.cluster.hierarchy import linkage, leaves_list

# Use non-GUI backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

import umap
import hdbscan

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def convert_to_serializable(obj):
    """Convert numpy types to native Python types."""
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


def load_phase_sync_alpha(sync_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load phase sync data and extract only Alpha band features.
    """
    logger.info(f"Loading phase sync data from {sync_path}")
    
    with open(sync_path, 'rb') as f:
        data = pickle.load(f)
    
    X_all = data['sync_vectors']
    subjects_all = np.array(data['metadata']['subjects'])
    conditions_all = np.array(data['metadata']['conditions'])
    
    # Check if multiband
    if data.get('multiband', False):
        bands = data['bands']
        n_features_per_band = data['n_features_per_band']
        
        # Find Alpha band index
        alpha_idx = bands.index('Alpha')
        start = alpha_idx * n_features_per_band
        end = start + n_features_per_band
        
        logger.info(f"Extracting Alpha features: columns {start}:{end}")
        X_alpha = X_all[:, start:end]
    else:
        X_alpha = X_all
    
    # Filter to DMT
    dmt_mask = conditions_all == 'DMT'
    X = X_alpha[dmt_mask]
    subjects = subjects_all[dmt_mask]
    
    logger.info(f"Phase sync DMT Alpha: {len(X)} samples, {X.shape[1]} features")
    
    return X, subjects


def load_autoencoder_alpha(ae_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load autoencoder embeddings (already Alpha-only).
    """
    emb_dir = ae_dir / 'embeddings'
    
    all_embeddings = []
    all_subjects = []
    all_conditions = []
    
    for split in ['train', 'val', 'test']:
        path = emb_dir / f'{split}_embeddings.pkl'
        if path.exists():
            with open(path, 'rb') as f:
                data = pickle.load(f)
            all_embeddings.append(data['latent_embeddings'])
            all_subjects.extend(data['subjects'])
            all_conditions.extend(data['conditions'])
    
    embeddings = np.vstack(all_embeddings)
    subjects = np.array(all_subjects)
    conditions = np.array(all_conditions)
    
    # Filter to DMT
    dmt_mask = conditions == 'DMT'
    embeddings = embeddings[dmt_mask]
    subjects = subjects[dmt_mask]
    
    # Clean subject IDs
    subjects = np.array([s.split('-')[0] for s in subjects])
    
    logger.info(f"Autoencoder DMT Alpha: {len(embeddings)} samples, {embeddings.shape[1]} dims")
    
    return embeddings, subjects


def cluster_data(
    X: np.ndarray,
    subjects: np.ndarray,
    min_cluster_size: int = 50,
    normalize_within_subject: bool = True,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Cluster data using UMAP + HDBSCAN."""
    X_proc = X.copy()
    
    # Within-subject normalization
    if normalize_within_subject:
        for subj in np.unique(subjects):
            mask = subjects == subj
            subj_data = X_proc[mask]
            mean = subj_data.mean(axis=0)
            std = subj_data.std(axis=0)
            std[std == 0] = 1
            X_proc[mask] = (subj_data - mean) / std
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_proc)
    
    # UMAP to 2D
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=random_state
    )
    embedding = reducer.fit_transform(X_scaled)
    
    # HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=5,
        cluster_selection_method='eom'
    )
    labels = clusterer.fit_predict(embedding)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    
    valid_mask = labels != -1
    if valid_mask.sum() > 1 and n_clusters > 1:
        silhouette = silhouette_score(embedding[valid_mask], labels[valid_mask])
    else:
        silhouette = 0.0
    
    metrics = {
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'silhouette': silhouette
    }
    
    return labels, embedding, metrics


def compute_fingerprints(labels: np.ndarray, subjects: np.ndarray) -> pd.DataFrame:
    """Compute cluster distribution per subject."""
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    unique_subjects = sorted(set(subjects))
    
    fingerprints = {}
    for subj in unique_subjects:
        mask = subjects == subj
        subj_labels = labels[mask]
        valid_labels = subj_labels[subj_labels != -1]
        
        counts = np.zeros(n_clusters)
        for label in valid_labels:
            if 0 <= label < n_clusters:
                counts[label] += 1
        
        if counts.sum() > 0:
            fingerprints[subj] = counts / counts.sum()
        else:
            fingerprints[subj] = counts
    
    df = pd.DataFrame(fingerprints).T
    df.columns = [f'C{i}' for i in range(n_clusters)]
    df.index.name = 'Subject'
    return df


def compute_similarity(fingerprints: pd.DataFrame) -> pd.DataFrame:
    """Compute subject similarity matrix."""
    return fingerprints.T.corr()


def plot_comparison(
    emb_phase: np.ndarray, labels_phase: np.ndarray, subjects_phase: np.ndarray,
    emb_ae: np.ndarray, labels_ae: np.ndarray, subjects_ae: np.ndarray,
    output_path: Path
):
    """Side-by-side embedding visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Get common subjects for consistent coloring
    common_subjects = sorted(set(subjects_phase) & set(subjects_ae))[:15]
    colors = plt.cm.tab20(np.linspace(0, 1, len(common_subjects)))
    subj_colors = {s: colors[i] for i, s in enumerate(common_subjects)}
    
    # Phase - by subject
    ax = axes[0, 0]
    for subj in common_subjects:
        mask = subjects_phase == subj
        ax.scatter(emb_phase[mask, 0], emb_phase[mask, 1], 
                   c=[subj_colors[subj]], label=subj, alpha=0.5, s=15)
    ax.set_title('Phase Sync Alpha - By Subject')
    ax.legend(fontsize=6, ncol=3, loc='upper right')
    
    # Phase - by cluster
    ax = axes[0, 1]
    n_clusters = len(set(labels_phase)) - (1 if -1 in labels_phase else 0)
    cluster_colors = plt.cm.Spectral(np.linspace(0, 1, max(n_clusters, 1)))
    for i in range(n_clusters):
        mask = labels_phase == i
        ax.scatter(emb_phase[mask, 0], emb_phase[mask, 1], 
                   c=[cluster_colors[i]], alpha=0.6, s=15)
    noise_mask = labels_phase == -1
    if noise_mask.any():
        ax.scatter(emb_phase[noise_mask, 0], emb_phase[noise_mask, 1],
                   c='lightgray', alpha=0.2, s=5)
    ax.set_title(f'Phase Sync Alpha - Clusters (n={n_clusters})')
    
    # AE - by subject
    ax = axes[1, 0]
    for subj in common_subjects:
        mask = subjects_ae == subj
        if mask.any():
            ax.scatter(emb_ae[mask, 0], emb_ae[mask, 1], 
                       c=[subj_colors[subj]], label=subj, alpha=0.5, s=15)
    ax.set_title('Autoencoder Alpha - By Subject')
    ax.legend(fontsize=6, ncol=3, loc='upper right')
    
    # AE - by cluster
    ax = axes[1, 1]
    n_clusters = len(set(labels_ae)) - (1 if -1 in labels_ae else 0)
    cluster_colors = plt.cm.Spectral(np.linspace(0, 1, max(n_clusters, 1)))
    for i in range(n_clusters):
        mask = labels_ae == i
        ax.scatter(emb_ae[mask, 0], emb_ae[mask, 1], 
                   c=[cluster_colors[i]], alpha=0.6, s=15)
    noise_mask = labels_ae == -1
    if noise_mask.any():
        ax.scatter(emb_ae[noise_mask, 0], emb_ae[noise_mask, 1],
                   c='lightgray', alpha=0.2, s=5)
    ax.set_title(f'Autoencoder Alpha - Clusters (n={n_clusters})')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_similarity_comparison(
    sim_phase: pd.DataFrame,
    sim_ae: pd.DataFrame,
    output_path: Path
) -> Tuple[float, float]:
    """Compare similarity matrices."""
    common = sorted(set(sim_phase.index) & set(sim_ae.index))
    
    sim_phase = sim_phase.loc[common, common]
    sim_ae = sim_ae.loc[common, common]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Phase similarity
    ax = axes[0]
    sns.heatmap(sim_phase, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                ax=ax, square=True, xticklabels=True, yticklabels=True)
    ax.set_title('Phase Sync Alpha\nSubject Similarity')
    ax.tick_params(labelsize=7)
    
    # AE similarity
    ax = axes[1]
    sns.heatmap(sim_ae, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                ax=ax, square=True, xticklabels=True, yticklabels=True)
    ax.set_title('Autoencoder Alpha\nSubject Similarity')
    ax.tick_params(labelsize=7)
    
    # Correlation between methods
    ax = axes[2]
    
    # Upper triangle values
    triu_idx = np.triu_indices(len(common), k=1)
    phase_flat = sim_phase.values[triu_idx]
    ae_flat = sim_ae.values[triu_idx]
    
    ax.scatter(phase_flat, ae_flat, alpha=0.6, s=40)
    
    r, p = stats.pearsonr(phase_flat, ae_flat)
    z = np.polyfit(phase_flat, ae_flat, 1)
    x_line = np.linspace(-1, 1, 100)
    ax.plot(x_line, np.poly1d(z)(x_line), 'r--', linewidth=2, 
            label=f'r={r:.3f}, p={p:.2e}')
    
    ax.set_xlabel('Phase Sync Similarity')
    ax.set_ylabel('Autoencoder Similarity')
    ax.set_title('Method Comparison')
    ax.legend(fontsize=10)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(0, color='gray', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")
    
    return r, p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min-cluster-size', type=int, default=50)
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("ALPHA BAND COMPARISON: Phase Sync vs Autoencoder")
    logger.info("=" * 70)
    
    # Paths
    base_dir = Path(__file__).parent.parent
    sync_path = base_dir / 'output' / 'sync_matrices' / 'sync_data.pkl'
    ae_dir = base_dir.parent / 'autoencoder' / 'output_alpha'
    output_dir = base_dir / 'output' / 'alpha_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("\n[Loading Data]")
    X_phase, subjects_phase = load_phase_sync_alpha(sync_path)
    X_ae, subjects_ae = load_autoencoder_alpha(ae_dir)
    
    # Cluster both
    logger.info("\n[Clustering Phase Sync Alpha]")
    labels_phase, emb_phase, metrics_phase = cluster_data(
        X_phase, subjects_phase, 
        min_cluster_size=args.min_cluster_size,
        normalize_within_subject=True
    )
    logger.info(f"  Clusters: {metrics_phase['n_clusters']}, Noise: {metrics_phase['n_noise']}, Silhouette: {metrics_phase['silhouette']:.3f}")
    
    logger.info("\n[Clustering Autoencoder Alpha]")
    labels_ae, emb_ae, metrics_ae = cluster_data(
        X_ae, subjects_ae,
        min_cluster_size=args.min_cluster_size,
        normalize_within_subject=True
    )
    logger.info(f"  Clusters: {metrics_ae['n_clusters']}, Noise: {metrics_ae['n_noise']}, Silhouette: {metrics_ae['silhouette']:.3f}")
    
    # Fingerprints
    logger.info("\n[Computing Fingerprints]")
    fp_phase = compute_fingerprints(labels_phase, subjects_phase)
    fp_ae = compute_fingerprints(labels_ae, subjects_ae)
    
    # Similarity matrices
    sim_phase = compute_similarity(fp_phase)
    sim_ae = compute_similarity(fp_ae)
    
    # Visualizations
    logger.info("\n[Generating Visualizations]")
    plot_comparison(
        emb_phase, labels_phase, subjects_phase,
        emb_ae, labels_ae, subjects_ae,
        output_dir / 'embedding_comparison.png'
    )
    
    r, p = plot_similarity_comparison(
        sim_phase, sim_ae,
        output_dir / 'similarity_comparison.png'
    )
    
    # Save results
    results = {
        'phase_sync': {
            'n_samples': len(X_phase),
            'n_clusters': metrics_phase['n_clusters'],
            'n_noise': metrics_phase['n_noise'],
            'silhouette': metrics_phase['silhouette']
        },
        'autoencoder': {
            'n_samples': len(X_ae),
            'n_clusters': metrics_ae['n_clusters'],
            'n_noise': metrics_ae['n_noise'],
            'silhouette': metrics_ae['silhouette']
        },
        'comparison': {
            'similarity_correlation': r,
            'similarity_pvalue': p
        }
    }
    
    with open(output_dir / 'comparison_results.json', 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    
    fp_phase.to_csv(output_dir / 'fingerprints_phase.csv')
    fp_ae.to_csv(output_dir / 'fingerprints_ae.csv')
    
    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("ALPHA COMPARISON SUMMARY")
    logger.info("=" * 70)
    logger.info(f"\n[Phase Sync Alpha]")
    logger.info(f"  Samples: {len(X_phase)}, Clusters: {metrics_phase['n_clusters']}, Silhouette: {metrics_phase['silhouette']:.3f}")
    logger.info(f"\n[Autoencoder Alpha]")
    logger.info(f"  Samples: {len(X_ae)}, Clusters: {metrics_ae['n_clusters']}, Silhouette: {metrics_ae['silhouette']:.3f}")
    logger.info(f"\n[Method Comparison]")
    logger.info(f"  Similarity correlation: r={r:.4f}, p={p:.2e}")
    
    if r > 0.5 and p < 0.05:
        logger.info("  → STRONG AGREEMENT: Both methods capture similar structure!")
    elif r > 0.3 and p < 0.05:
        logger.info("  → MODERATE AGREEMENT: Some overlap in what methods capture")
    elif r > 0 and p < 0.05:
        logger.info("  → WEAK AGREEMENT: Methods capture different but related aspects")
    else:
        logger.info("  → NO AGREEMENT: Methods capture different aspects")
    
    logger.info("")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()






