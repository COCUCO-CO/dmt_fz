#!/usr/bin/env python3
"""
03_analyze_results.py

Step 3 of Phase Analysis Pipeline:
- Deep analysis of clustering results
- Statistical tests: do clusters correspond to conditions?
- Kuramoto correlation analysis
- Temporal dynamics (within-subject cluster transitions)
- Per-band analysis

Output:
    output/visualizations/
        condition_separation.png      # How well clusters separate conditions
        kuramoto_correlation.png      # Kuramoto vs cluster/embedding position
        temporal_dynamics.png         # State transitions over time
        band_analysis.png             # Per-band cluster distribution
        summary_stats.json            # Statistical test results

Usage:
    python scripts/03_analyze_results.py
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict
import logging

import numpy as np
import yaml
from scipy import stats

# Use non-GUI backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import confusion_matrix

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


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


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def compute_condition_cluster_metrics(
    labels: np.ndarray,
    conditions: np.ndarray
) -> Dict[str, float]:
    """
    Compute metrics measuring how well clusters correspond to conditions.
    """
    # Convert conditions to numeric labels
    unique_conditions = np.unique(conditions)
    condition_labels = np.array([np.where(unique_conditions == c)[0][0] for c in conditions])
    
    # Filter out noise points
    valid_mask = labels >= 0
    labels_valid = labels[valid_mask]
    conditions_valid = condition_labels[valid_mask]
    
    if len(labels_valid) == 0 or len(np.unique(labels_valid)) < 2:
        return {'ari': 0.0, 'nmi': 0.0}
    
    # Adjusted Rand Index
    ari = adjusted_rand_score(conditions_valid, labels_valid)
    
    # Normalized Mutual Information
    nmi = normalized_mutual_info_score(conditions_valid, labels_valid)
    
    return {
        'adjusted_rand_index': float(ari),
        'normalized_mutual_info': float(nmi)
    }


def compute_kuramoto_correlation(
    embedding: np.ndarray,
    kuramoto: np.ndarray,
    labels: np.ndarray
) -> Dict[str, Any]:
    """
    Analyze correlation between Kuramoto and embedding/clusters.
    """
    results = {}
    
    # Correlation with embedding dimensions
    for dim in range(embedding.shape[1]):
        corr, pval = stats.spearmanr(embedding[:, dim], kuramoto)
        results[f'spearman_dim{dim}'] = {'correlation': float(corr), 'pvalue': float(pval)}
    
    # Kuramoto difference between clusters
    unique_clusters = sorted([l for l in np.unique(labels) if l >= 0])
    
    if len(unique_clusters) >= 2:
        cluster_kuramoto = {c: kuramoto[labels == c] for c in unique_clusters}
        
        # ANOVA test
        if len(unique_clusters) > 2:
            f_stat, p_val = stats.f_oneway(*[cluster_kuramoto[c] for c in unique_clusters])
            results['anova'] = {'f_statistic': float(f_stat), 'pvalue': float(p_val)}
        else:
            # t-test for 2 clusters
            t_stat, p_val = stats.ttest_ind(
                cluster_kuramoto[unique_clusters[0]],
                cluster_kuramoto[unique_clusters[1]]
            )
            results['ttest'] = {'t_statistic': float(t_stat), 'pvalue': float(p_val)}
        
        # Effect size (Cohen's d between extreme clusters)
        k_means = {c: np.mean(cluster_kuramoto[c]) for c in unique_clusters}
        sorted_clusters = sorted(k_means.keys(), key=lambda c: k_means[c])
        
        low_cluster = sorted_clusters[0]
        high_cluster = sorted_clusters[-1]
        
        mean_diff = k_means[high_cluster] - k_means[low_cluster]
        pooled_std = np.sqrt(
            (np.var(cluster_kuramoto[low_cluster]) + np.var(cluster_kuramoto[high_cluster])) / 2
        )
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        
        results['cohens_d_extreme'] = float(cohens_d)
        results['kuramoto_range'] = {
            'min_cluster': int(low_cluster),
            'min_mean': float(k_means[low_cluster]),
            'max_cluster': int(high_cluster),
            'max_mean': float(k_means[high_cluster])
        }
    
    return results


def analyze_temporal_dynamics(
    labels: np.ndarray,
    subjects: np.ndarray,
    conditions: np.ndarray,
    bands: np.ndarray,
    epoch_indices: np.ndarray
) -> Dict[str, Any]:
    """
    Analyze temporal dynamics of cluster transitions within subjects.
    """
    results = {}
    
    # Group by subject-condition-band
    groups = defaultdict(list)
    for i, (subj, cond, band, epoch) in enumerate(zip(subjects, conditions, bands, epoch_indices)):
        groups[(subj, cond, band)].append((epoch, labels[i]))
    
    # Compute transition matrices per condition
    unique_clusters = sorted([l for l in np.unique(labels) if l >= 0])
    n_clusters = len(unique_clusters)
    cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
    
    transitions_by_condition = defaultdict(lambda: np.zeros((n_clusters, n_clusters)))
    
    for (subj, cond, band), epoch_labels in groups.items():
        # Sort by epoch
        epoch_labels.sort(key=lambda x: x[0])
        
        for i in range(len(epoch_labels) - 1):
            label_from = epoch_labels[i][1]
            label_to = epoch_labels[i + 1][1]
            
            if label_from >= 0 and label_to >= 0:
                idx_from = cluster_to_idx[label_from]
                idx_to = cluster_to_idx[label_to]
                transitions_by_condition[cond][idx_from, idx_to] += 1
    
    # Normalize transition matrices
    for cond in transitions_by_condition:
        row_sums = transitions_by_condition[cond].sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transitions_by_condition[cond] = transitions_by_condition[cond] / row_sums
    
    results['transition_matrices'] = {
        cond: mat.tolist() for cond, mat in transitions_by_condition.items()
    }
    results['cluster_labels'] = unique_clusters
    
    # Compute stability (probability of staying in same cluster)
    stability_by_condition = {}
    for cond, mat in transitions_by_condition.items():
        stability = np.diag(mat).mean()
        stability_by_condition[cond] = float(stability)
    
    results['stability_by_condition'] = stability_by_condition
    
    return results


def plot_condition_separation(
    embedding: np.ndarray,
    labels: np.ndarray,
    conditions: np.ndarray,
    output_dir: Path,
    config: Dict[str, Any]
):
    """
    Detailed visualization of condition separation.
    """
    viz_config = config['visualization']
    condition_colors = viz_config['cmap_conditions']
    unique_conditions = np.unique(conditions)
    unique_clusters = sorted([l for l in np.unique(labels) if l >= 0])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Embedding colored by condition with cluster boundaries
    ax = axes[0, 0]
    for condition in unique_conditions:
        mask = conditions == condition
        color = condition_colors.get(condition, 'gray')
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                  c=color, alpha=0.5, s=15, label=condition)
    
    ax.set_title('Embedding by Condition', fontsize=12, fontweight='bold')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.legend()
    
    # 2. Confusion matrix: predicted cluster vs actual condition
    ax = axes[0, 1]
    
    valid_mask = labels >= 0
    labels_valid = labels[valid_mask]
    conditions_valid = conditions[valid_mask]
    
    # Create contingency table
    contingency = np.zeros((len(unique_clusters), len(unique_conditions)))
    for i, cluster in enumerate(unique_clusters):
        for j, condition in enumerate(unique_conditions):
            contingency[i, j] = ((labels_valid == cluster) & (conditions_valid == condition)).sum()
    
    # Normalize by cluster size
    contingency_norm = contingency / contingency.sum(axis=1, keepdims=True)
    
    sns.heatmap(contingency_norm, annot=True, fmt='.2f', cmap='YlOrRd',
               xticklabels=unique_conditions,
               yticklabels=[f'Cluster {c}' for c in unique_clusters],
               ax=ax)
    ax.set_title('Cluster-Condition Association', fontsize=12, fontweight='bold')
    ax.set_xlabel('Condition')
    ax.set_ylabel('Cluster')
    
    # 3. Cluster purity per condition
    ax = axes[1, 0]
    
    purity_data = []
    for condition in unique_conditions:
        cond_mask = conditions == condition
        cond_labels = labels[cond_mask & (labels >= 0)]
        
        if len(cond_labels) > 0:
            # Find most common cluster for this condition
            cluster_counts = np.bincount(cond_labels, minlength=max(unique_clusters)+1)
            purity = cluster_counts.max() / len(cond_labels)
            purity_data.append({'condition': condition, 'purity': purity})
    
    if purity_data:
        x = range(len(purity_data))
        bars = ax.bar(x, [d['purity'] for d in purity_data],
                     color=[condition_colors.get(d['condition'], 'gray') for d in purity_data])
        ax.set_xticks(x)
        ax.set_xticklabels([d['condition'] for d in purity_data])
        ax.set_ylabel('Cluster Purity')
        ax.set_title('Dominant Cluster Purity per Condition', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    
    # 4. Distribution of conditions within each cluster
    ax = axes[1, 1]
    
    x = np.arange(len(unique_clusters))
    width = 0.8 / len(unique_conditions)
    
    for j, condition in enumerate(unique_conditions):
        counts = contingency[:, j]
        color = condition_colors.get(condition, 'gray')
        ax.bar(x + j * width, counts, width, label=condition, color=color, alpha=0.8)
    
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Count')
    ax.set_title('Condition Distribution per Cluster', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width * (len(unique_conditions) - 1) / 2)
    ax.set_xticklabels([f'C{c}' for c in unique_clusters])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'condition_separation.png', dpi=viz_config['dpi'], bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved condition separation plot")


def plot_kuramoto_detailed(
    embedding: np.ndarray,
    kuramoto: np.ndarray,
    labels: np.ndarray,
    conditions: np.ndarray,
    output_dir: Path,
    config: Dict[str, Any]
):
    """
    Detailed Kuramoto analysis visualization.
    """
    viz_config = config['visualization']
    condition_colors = viz_config['cmap_conditions']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Scatter: Kuramoto vs embedding position (colored by condition)
    ax = axes[0, 0]
    
    # Use first embedding dimension
    for condition in np.unique(conditions):
        mask = conditions == condition
        color = condition_colors.get(condition, 'gray')
        ax.scatter(embedding[mask, 0], kuramoto[mask],
                  c=color, alpha=0.4, s=10, label=condition)
    
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('Kuramoto R')
    ax.set_title('Kuramoto vs Embedding Position', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Kuramoto distribution by condition (violin plot)
    ax = axes[0, 1]
    
    data_for_violin = []
    labels_for_violin = []
    for condition in np.unique(conditions):
        k_vals = kuramoto[conditions == condition]
        data_for_violin.extend(k_vals)
        labels_for_violin.extend([condition] * len(k_vals))
    
    import pandas as pd
    df = pd.DataFrame({'Kuramoto': data_for_violin, 'Condition': labels_for_violin})
    
    palette = {c: condition_colors.get(c, 'gray') for c in np.unique(conditions)}
    sns.violinplot(data=df, x='Condition', y='Kuramoto', palette=palette, ax=ax)
    ax.set_title('Kuramoto Distribution by Condition', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 3. Kuramoto percentiles by cluster
    ax = axes[1, 0]
    
    unique_clusters = sorted([l for l in np.unique(labels) if l >= 0])
    cluster_stats = []
    
    for cluster in unique_clusters:
        k_vals = kuramoto[labels == cluster]
        cluster_stats.append({
            'cluster': cluster,
            'mean': np.mean(k_vals),
            'std': np.std(k_vals),
            'p25': np.percentile(k_vals, 25),
            'p75': np.percentile(k_vals, 75)
        })
    
    x = np.arange(len(unique_clusters))
    means = [s['mean'] for s in cluster_stats]
    stds = [s['std'] for s in cluster_stats]
    
    ax.bar(x, means, yerr=stds, capsize=5, color=plt.cm.viridis(np.linspace(0, 1, len(unique_clusters))))
    ax.set_xticks(x)
    ax.set_xticklabels([f'C{c}' for c in unique_clusters])
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Kuramoto R (mean ± std)')
    ax.set_title('Kuramoto by Cluster', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 4. 2D embedding colored by Kuramoto (with contours)
    ax = axes[1, 1]
    
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1],
                        c=kuramoto, cmap='viridis', alpha=0.5, s=15)
    plt.colorbar(scatter, ax=ax, label='Kuramoto R')
    
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('Embedding Colored by Kuramoto', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'kuramoto_analysis_detailed.png', dpi=viz_config['dpi'], bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved detailed Kuramoto analysis")


def main():
    parser = argparse.ArgumentParser(description='Analyze clustering results')
    parser.add_argument('--config', type=str,
                       default='config/config.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    
    # Change to script directory for relative paths
    script_dir = Path(__file__).parent.parent
    config_path = script_dir / args.config
    
    logger.info("=" * 70)
    logger.info("PHASE ANALYSIS - Step 3: Analyze Results")
    logger.info("=" * 70)
    
    # Load config
    config = load_config(config_path)
    
    # Setup paths
    base_output = Path(config['paths']['output_dir'])
    clustering_path = base_output / 'clustering' / 'clustering_results.pkl'
    sync_path = base_output / 'sync_matrices' / 'sync_data.pkl'
    output_dir = base_output / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info(f"Loading clustering results from {clustering_path}")
    with open(clustering_path, 'rb') as f:
        results = pickle.load(f)
    
    with open(sync_path, 'rb') as f:
        sync_data = pickle.load(f)
    
    labels = results['labels']
    embedding = results['embedding_2d']
    conditions = results['conditions']
    bands = results['bands']
    subjects = results['subjects']
    kuramoto = results['kuramoto']
    epoch_indices = np.array(sync_data['metadata']['epoch_indices'])
    
    logger.info(f"Loaded {len(labels)} samples")
    logger.info(f"Clusters: {results['n_clusters']}, Noise: {results['n_noise']}")
    
    # Compute metrics
    logger.info("Computing condition-cluster metrics...")
    condition_metrics = compute_condition_cluster_metrics(labels, conditions)
    
    logger.info("Computing Kuramoto correlation...")
    kuramoto_metrics = compute_kuramoto_correlation(embedding, kuramoto, labels)
    
    logger.info("Analyzing temporal dynamics...")
    temporal_metrics = analyze_temporal_dynamics(labels, subjects, conditions, bands, epoch_indices)
    
    # Generate visualizations
    logger.info("Generating detailed visualizations...")
    plot_condition_separation(embedding, labels, conditions, output_dir, config)
    plot_kuramoto_detailed(embedding, kuramoto, labels, conditions, output_dir, config)
    
    # Compile summary statistics
    summary = {
        'n_samples': len(labels),
        'n_clusters': results['n_clusters'],
        'n_noise': results['n_noise'],
        'silhouette_score': results['silhouette'],
        'calinski_harabasz_score': results['calinski_harabasz'],
        'condition_cluster_metrics': condition_metrics,
        'kuramoto_metrics': kuramoto_metrics,
        'temporal_metrics': {
            'stability_by_condition': temporal_metrics['stability_by_condition']
        }
    }
    
    # Save summary (convert numpy types to native Python types)
    with open(output_dir / 'summary_stats.json', 'w') as f:
        json.dump(convert_to_serializable(summary), f, indent=2)
    
    logger.info(f"Saved summary to {output_dir / 'summary_stats.json'}")
    
    # Print summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("ANALYSIS SUMMARY")
    logger.info("=" * 70)
    
    logger.info(f"\n[Clustering Quality]")
    logger.info(f"  Silhouette Score: {results['silhouette']:.4f}")
    logger.info(f"  Calinski-Harabasz: {results['calinski_harabasz']:.2f}")
    
    logger.info(f"\n[Condition-Cluster Association]")
    logger.info(f"  Adjusted Rand Index: {condition_metrics['adjusted_rand_index']:.4f}")
    logger.info(f"  Normalized Mutual Info: {condition_metrics['normalized_mutual_info']:.4f}")
    
    if 'kuramoto_range' in kuramoto_metrics:
        kr = kuramoto_metrics['kuramoto_range']
        logger.info(f"\n[Kuramoto by Cluster]")
        logger.info(f"  Lowest: Cluster {kr['min_cluster']} (R={kr['min_mean']:.3f})")
        logger.info(f"  Highest: Cluster {kr['max_cluster']} (R={kr['max_mean']:.3f})")
        logger.info(f"  Cohen's d: {kuramoto_metrics['cohens_d_extreme']:.3f}")
    
    if 'anova' in kuramoto_metrics:
        logger.info(f"  ANOVA: F={kuramoto_metrics['anova']['f_statistic']:.2f}, p={kuramoto_metrics['anova']['pvalue']:.2e}")
    
    logger.info(f"\n[Temporal Stability]")
    for cond, stab in temporal_metrics['stability_by_condition'].items():
        logger.info(f"  {cond}: {stab:.3f} (prob of staying in same cluster)")
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("Phase 1 Analysis Complete!")
    logger.info(f"Results: {output_dir}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()


