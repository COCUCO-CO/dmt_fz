#!/usr/bin/env python3
"""
Analyze graph properties and compare DMT vs EC conditions.

This script performs statistical analysis and visualization of graph structures
to understand how brain states differ at the network level.
"""

import argparse
import sys
import yaml
import pickle
from pathlib import Path
from typing import List, Dict, Any
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from data import create_dataset_from_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_graph_features(graphs: List[Any], condition_name: str) -> pd.DataFrame:
    """
    Extract topological features from graphs.
    
    Args:
        graphs: List of graph Data objects
        condition_name: Name of condition
        
    Returns:
        DataFrame with graph features
    """
    features = []
    
    for g in tqdm(graphs, desc=f"Extracting features for {condition_name}"):
        num_nodes = g.num_nodes
        num_edges = g.edge_index.shape[1] // 2  # Undirected
        
        # Density
        max_edges = num_nodes * (num_nodes - 1) / 2
        density = num_edges / max_edges if max_edges > 0 else 0
        
        # Degree statistics
        degrees = np.bincount(g.edge_index[0].cpu().numpy(), minlength=num_nodes)
        
        # Edge weight statistics (synchronization strength)
        if hasattr(g, 'edge_attr') and g.edge_attr is not None:
            edge_weights = g.edge_attr.cpu().numpy().flatten()
            mean_sync = edge_weights.mean()
            std_sync = edge_weights.std()
            max_sync = edge_weights.max()
            min_sync = edge_weights.min()
        else:
            mean_sync = std_sync = max_sync = min_sync = 0
        
        # Kuramoto statistics (if available)
        if hasattr(g, 'graph_attr') and g.graph_attr is not None:
            graph_features = g.graph_attr.cpu().numpy()
            kuramoto_mean = graph_features[0] if len(graph_features) > 0 else 0
            kuramoto_std = graph_features[1] if len(graph_features) > 1 else 0
        else:
            kuramoto_mean = kuramoto_std = 0
        
        features.append({
            'condition': condition_name,
            'subject_id': g.subject_id,
            'band': g.band,
            'epoch_idx': g.epoch_idx,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'density': density,
            'mean_degree': degrees.mean(),
            'std_degree': degrees.std(),
            'max_degree': degrees.max(),
            'min_degree': degrees.min(),
            'mean_sync': mean_sync,
            'std_sync': std_sync,
            'max_sync': max_sync,
            'min_sync': min_sync,
            'kuramoto_mean': kuramoto_mean,
            'kuramoto_std': kuramoto_std
        })
    
    return pd.DataFrame(features)


def compare_conditions(df: pd.DataFrame, 
                       cond1: str, 
                       cond2: str,
                       features: List[str],
                       save_dir: Path):
    """
    Statistical comparison between two conditions.
    
    Args:
        df: DataFrame with graph features
        cond1: First condition name
        cond2: Second condition name
        features: List of feature names to compare
        save_dir: Directory to save results
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    logger.info(f"\nComparing {cond1} vs {cond2}:")
    logger.info("=" * 80)
    
    for feature in features:
        data1 = df[df['condition'] == cond1][feature].values
        data2 = df[df['condition'] == cond2][feature].values
        
        # T-test
        t_stat, t_pval = stats.ttest_ind(data1, data2)
        
        # Mann-Whitney U test (non-parametric)
        u_stat, u_pval = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        
        # Effect size (Cohen's d)
        mean1, mean2 = data1.mean(), data2.mean()
        std_pooled = np.sqrt((data1.std()**2 + data2.std()**2) / 2)
        cohens_d = (mean1 - mean2) / std_pooled if std_pooled > 0 else 0
        
        results.append({
            'feature': feature,
            f'{cond1}_mean': mean1,
            f'{cond1}_std': data1.std(),
            f'{cond2}_mean': mean2,
            f'{cond2}_std': data2.std(),
            't_statistic': t_stat,
            't_pvalue': t_pval,
            'u_statistic': u_stat,
            'u_pvalue': u_pval,
            'cohens_d': cohens_d,
            'significant': t_pval < 0.05
        })
        
        sig_marker = "***" if t_pval < 0.001 else ("**" if t_pval < 0.01 else ("*" if t_pval < 0.05 else ""))
        logger.info(f"{feature:20s}: {mean1:8.4f} vs {mean2:8.4f}  |  "
                   f"t={t_stat:7.3f}  p={t_pval:.4f} {sig_marker}  |  d={cohens_d:6.3f}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(save_dir / f'comparison_{cond1}_vs_{cond2}.csv', index=False)
    
    logger.info(f"\nResults saved to: {save_dir / f'comparison_{cond1}_vs_{cond2}.csv'}")
    
    return results_df


def plot_feature_distributions(df: pd.DataFrame,
                               features: List[str],
                               save_dir: Path):
    """
    Plot feature distributions by condition.
    
    Args:
        df: DataFrame with graph features
        features: List of feature names to plot
        save_dir: Directory to save plots
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    conditions = df['condition'].unique()
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        
        for condition in conditions:
            data = df[df['condition'] == condition][feature]
            ax.hist(data, bins=30, alpha=0.6, label=condition, edgecolor='black')
        
        ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Graph Feature Distributions by Condition', fontsize=16, y=1.00)
    plt.tight_layout()
    plt.savefig(save_dir / 'feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Feature distributions saved to: {save_dir / 'feature_distributions.png'}")


def plot_feature_boxplots(df: pd.DataFrame,
                          features: List[str],
                          save_dir: Path):
    """
    Plot feature boxplots by condition.
    
    Args:
        df: DataFrame with graph features
        features: List of feature names to plot
        save_dir: Directory to save plots
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        sns.boxplot(data=df, x='condition', y=feature, ax=ax, palette='Set2')
        ax.set_xlabel('Condition', fontsize=11)
        ax.set_ylabel(feature.replace('_', ' ').title(), fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Graph Feature Boxplots by Condition', fontsize=16, y=1.00)
    plt.tight_layout()
    plt.savefig(save_dir / 'feature_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Feature boxplots saved to: {save_dir / 'feature_boxplots.png'}")


def plot_band_comparison(df: pd.DataFrame,
                        feature: str,
                        save_dir: Path):
    """
    Plot feature values across frequency bands.
    
    Args:
        df: DataFrame with graph features
        feature: Feature name to plot
        save_dir: Directory to save plots
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    conditions = df['condition'].unique()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(bands))
    width = 0.8 / len(conditions)
    
    for i, condition in enumerate(conditions):
        means = []
        stds = []
        for band in bands:
            data = df[(df['condition'] == condition) & (df['band'] == band)][feature]
            means.append(data.mean())
            stds.append(data.std() / np.sqrt(len(data)))  # SEM
        
        ax.bar(x + i * width, means, width, yerr=stds, 
               label=condition, capsize=5, alpha=0.8)
    
    ax.set_xlabel('Frequency Band', fontsize=12)
    ax.set_ylabel(feature.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'{feature.replace("_", " ").title()} by Frequency Band', fontsize=14)
    ax.set_xticks(x + width * (len(conditions) - 1) / 2)
    ax.set_xticklabels(bands)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_dir / f'band_comparison_{feature}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Band comparison saved to: {save_dir / f'band_comparison_{feature}.png'}")


def main(config_path: str):
    """Main analysis function."""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    output_dir = Path(config['paths']['output_dir']) / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("Graph Analysis: DMT vs EC vs EO")
    logger.info("=" * 80)
    
    # Load datasets
    logger.info("Loading datasets...")
    
    # Get dataset_workers from config (or auto-detect if not specified)
    num_workers = config.get('dataset_workers', None)
    
    train_graphs, val_graphs, test_graphs = create_dataset_from_config(
        config,
        num_workers=num_workers
    )
    all_graphs = train_graphs + val_graphs + test_graphs
    
    # Extract features by condition
    logger.info("Extracting graph features...")
    
    dfs = []
    for condition in config['data']['conditions']:
        condition_graphs = [g for g in all_graphs if g.y.item() == config['data']['conditions'].index(condition)]
        df_cond = extract_graph_features(condition_graphs, condition)
        dfs.append(df_cond)
    
    df_all = pd.concat(dfs, ignore_index=True)
    df_all.to_csv(output_dir / 'graph_features_all.csv', index=False)
    logger.info(f"Features saved to: {output_dir / 'graph_features_all.csv'}")
    
    # Statistical comparisons
    features_to_compare = [
        'density', 'mean_degree', 'mean_sync', 'std_sync',
        'kuramoto_mean', 'kuramoto_std'
    ]
    
    conditions = config['data']['conditions']
    
    # DMT vs EC
    if 'DMT' in conditions and 'EC' in conditions:
        compare_conditions(df_all, 'DMT', 'EC', features_to_compare, output_dir)
    
    # DMT vs EO
    if 'DMT' in conditions and 'EO' in conditions:
        compare_conditions(df_all, 'DMT', 'EO', features_to_compare, output_dir)
    
    # EC vs EO
    if 'EC' in conditions and 'EO' in conditions:
        compare_conditions(df_all, 'EC', 'EO', features_to_compare, output_dir)
    
    # Visualizations
    logger.info("\nGenerating visualizations...")
    
    plot_feature_distributions(df_all, features_to_compare, output_dir)
    plot_feature_boxplots(df_all, features_to_compare, output_dir)
    
    # Band-specific analysis
    for feature in ['kuramoto_mean', 'mean_sync', 'density']:
        plot_band_comparison(df_all, feature, output_dir)
    
    logger.info("=" * 80)
    logger.info(f"Analysis complete! Results saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze graph properties')
    parser.add_argument('--config', type=str, 
                       default='config/config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    main(args.config)

