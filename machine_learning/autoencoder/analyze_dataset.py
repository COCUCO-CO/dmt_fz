#!/usr/bin/env python3
"""
Dataset Analysis Script for BrainStateVAE
Generates comprehensive statistical analysis and visualizations.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "clf"))

from data import create_dataset_from_config

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def setup_output_dir(base_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / f"analysis_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def extract_all_features(graphs, class_names=None):
    node_data, edge_data, graph_data = [], [], []
    
    for idx, g in enumerate(tqdm(graphs, desc="Extracting features")):
        label = g.y.item() if hasattr(g, 'y') else -1
        class_name = class_names[label] if class_names and label >= 0 else f"Class_{label}"
        subject = getattr(g, 'subject_id', 'unknown')
        if isinstance(subject, torch.Tensor):
            subject = subject.item() if subject.numel() == 1 else 'batch'
        
        x = g.x.numpy()
        num_nodes, num_node_features = x.shape
        
        for node_idx in range(num_nodes):
            node_row = {'graph_idx': idx, 'node_idx': node_idx, 'class': class_name, 'subject': subject}
            for feat_idx in range(num_node_features):
                node_row[f'feat_{feat_idx}'] = x[node_idx, feat_idx]
            node_data.append(node_row)
        
        if hasattr(g, 'edge_attr') and g.edge_attr is not None:
            edge_attr = g.edge_attr.numpy()
            edge_index = g.edge_index.numpy()
            for edge_idx in range(len(edge_attr)):
                edge_row = {'graph_idx': idx, 'class': class_name,
                           'src': edge_index[0, edge_idx], 'dst': edge_index[1, edge_idx]}
                if len(edge_attr.shape) > 1:
                    for feat_idx in range(edge_attr.shape[1]):
                        edge_row[f'edge_feat_{feat_idx}'] = edge_attr[edge_idx, feat_idx]
                else:
                    edge_row['edge_feat_0'] = edge_attr[edge_idx]
                edge_data.append(edge_row)
        
        graph_row = {'graph_idx': idx, 'class': class_name, 'label': label, 'subject': subject,
                    'num_nodes': num_nodes, 'num_edges': g.edge_index.shape[1],
                    'node_feat_mean': x.mean(), 'node_feat_std': x.std(),
                    'node_feat_min': x.min(), 'node_feat_max': x.max()}
        
        if hasattr(g, 'graph_features') and g.graph_features is not None:
            gf = g.graph_features.numpy().flatten()
            for feat_idx, val in enumerate(gf):
                graph_row[f'graph_feat_{feat_idx}'] = val
        graph_data.append(graph_row)
    
    return pd.DataFrame(node_data), pd.DataFrame(edge_data), pd.DataFrame(graph_data)


def compute_statistics(df, feature_cols, group_col='class'):
    stats_list = []
    for col in feature_cols:
        overall = {'feature': col, 'group': 'ALL', 'count': df[col].count(),
                  'mean': df[col].mean(), 'std': df[col].std(),
                  'min': df[col].min(), 'max': df[col].max(),
                  'median': df[col].median(), 'q25': df[col].quantile(0.25),
                  'q75': df[col].quantile(0.75),
                  'skewness': stats.skew(df[col].dropna()),
                  'kurtosis': stats.kurtosis(df[col].dropna()),
                  'pct_zeros': (df[col] == 0).mean() * 100}
        stats_list.append(overall)
        
        for group, gdf in df.groupby(group_col):
            gstats = {'feature': col, 'group': group, 'count': gdf[col].count(),
                     'mean': gdf[col].mean(), 'std': gdf[col].std(),
                     'min': gdf[col].min(), 'max': gdf[col].max(),
                     'median': gdf[col].median(), 'skewness': stats.skew(gdf[col].dropna())}
            stats_list.append(gstats)
    return pd.DataFrame(stats_list)


def plot_distributions(df, feature_cols, output_dir, prefix, group_col='class'):
    n = min(len(feature_cols), 16)
    cols = 4
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
    axes = axes.flatten()
    
    for idx, feat in enumerate(feature_cols[:n]):
        ax = axes[idx]
        for cls in df[group_col].unique():
            data = df[df[group_col] == cls][feat].dropna()
            ax.hist(data, bins=50, alpha=0.5, label=cls, density=True)
        ax.set_title(feat, fontsize=10)
        ax.legend(fontsize=8)
    
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'{prefix} - Feature Distributions by Class')
    plt.tight_layout()
    plt.savefig(output_dir / f'{prefix}_distributions.png', dpi=150)
    plt.close()
    
    # Boxplots
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
    axes = axes.flatten()
    for idx, feat in enumerate(feature_cols[:n]):
        sns.boxplot(data=df, x=group_col, y=feat, ax=axes[idx])
        axes[idx].set_title(feat, fontsize=10)
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)
    plt.suptitle(f'{prefix} - Boxplots by Class')
    plt.tight_layout()
    plt.savefig(output_dir / f'{prefix}_boxplots.png', dpi=150)
    plt.close()


def plot_correlation(df, feature_cols, output_dir, prefix):
    corr = df[feature_cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap='RdBu_r', center=0, vmin=-1, vmax=1, square=True)
    plt.title(f'{prefix} - Correlation Matrix')
    plt.tight_layout()
    plt.savefig(output_dir / f'{prefix}_correlation.png', dpi=150)
    plt.close()


def plot_pca(df, feature_cols, output_dir, prefix, group_col='class'):
    data = df[feature_cols].dropna()
    if len(data) < 10:
        return
    
    scaler = StandardScaler()
    X = scaler.fit_transform(data)
    pca = PCA()
    X_pca = pca.fit_transform(X)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].bar(range(1, min(11, len(pca.explained_variance_ratio_)+1)),
               pca.explained_variance_ratio_[:10] * 100)
    axes[0].set_xlabel('PC')
    axes[0].set_ylabel('Variance (%)')
    axes[0].set_title('Explained Variance')
    
    labels = df.loc[data.index, group_col].values
    for cls in np.unique(labels):
        mask = labels == cls
        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], alpha=0.5, label=cls, s=10)
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[1].legend()
    axes[1].set_title('PCA Projection')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{prefix}_pca.png', dpi=150)
    plt.close()


def plot_tsne(df, feature_cols, output_dir, prefix, group_col='class', n_samples=3000):
    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=42)
    
    data = df[feature_cols].dropna()
    if len(data) < 50:
        return
    
    print(f"  Running t-SNE on {len(data)} samples...")
    X = StandardScaler().fit_transform(data)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
    X_tsne = tsne.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    labels = df.loc[data.index, group_col].values
    for cls in np.unique(labels):
        mask = labels == cls
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], alpha=0.5, label=cls, s=10)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend()
    plt.title(f'{prefix} - t-SNE')
    plt.tight_layout()
    plt.savefig(output_dir / f'{prefix}_tsne.png', dpi=150)
    plt.close()


def plot_class_balance(graph_df, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    counts = graph_df['class'].value_counts()
    axes[0].bar(counts.index, counts.values)
    axes[0].set_ylabel('Count')
    axes[0].set_title('Samples per Class')
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 50, str(v), ha='center')
    
    subj_cls = graph_df.groupby(['subject', 'class']).size().unstack(fill_value=0)
    subj_cls.plot(kind='bar', stacked=True, ax=axes[1])
    axes[1].set_title('Samples per Subject')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'class_distribution.png', dpi=150)
    plt.close()


def generate_report(graph_df, node_stats, output_dir):
    lines = ["="*60, "DATASET ANALYSIS REPORT", "="*60]
    lines.append(f"\nGenerated: {datetime.now()}")
    lines.append(f"\nTotal graphs: {len(graph_df)}")
    lines.append(f"Classes: {', '.join(graph_df['class'].unique())}")
    lines.append(f"Subjects: {graph_df['subject'].nunique()}")
    lines.append(f"Nodes per graph: {graph_df['num_nodes'].iloc[0]}")
    lines.append(f"Edges per graph: {graph_df['num_edges'].mean():.1f} ± {graph_df['num_edges'].std():.1f}")
    
    lines.append("\n" + "-"*40 + "\nCLASS DISTRIBUTION\n" + "-"*40)
    for cls, cnt in graph_df['class'].value_counts().items():
        lines.append(f"  {cls}: {cnt} ({cnt/len(graph_df)*100:.1f}%)")
    
    lines.append("\n" + "-"*40 + "\nNODE FEATURES (ALL)\n" + "-"*40)
    all_stats = node_stats[node_stats['group'] == 'ALL']
    for _, r in all_stats.iterrows():
        lines.append(f"  {r['feature']}: [{r['min']:.4f}, {r['max']:.4f}] mean={r['mean']:.4f} std={r['std']:.4f}")
    
    lines.append("\n" + "-"*40 + "\nPOTENTIAL ISSUES\n" + "-"*40)
    issues = []
    for _, r in all_stats.iterrows():
        rng = r['max'] - r['min']
        if rng > 100:
            issues.append(f"  - {r['feature']}: Large range ({rng:.1f}) - NEEDS NORMALIZATION")
        if abs(r['skewness']) > 2:
            issues.append(f"  - {r['feature']}: High skewness ({r['skewness']:.2f})")
    lines.extend(issues if issues else ["  No major issues."])
    
    report = "\n".join(lines)
    with open(output_dir / 'report.txt', 'w') as f:
        f.write(report)
    print(report)


def main(config_path="config/config.yaml", output_dir=None):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if output_dir is None:
        output_dir = setup_output_dir(Path(config['paths']['output_dir']) / "dataset_analysis")
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output: {output_dir}")
    
    print("\nLoading dataset...")
    train, val, test = create_dataset_from_config(config)
    all_graphs = train + val + test
    print(f"Total graphs: {len(all_graphs)}")
    
    class_names = config['data'].get('conditions', ['DMT', 'EC', 'EO'])
    
    print("\nExtracting features...")
    node_df, edge_df, graph_df = extract_all_features(all_graphs, class_names)
    
    node_cols = [c for c in node_df.columns if c.startswith('feat_')]
    edge_cols = [c for c in edge_df.columns if c.startswith('edge_feat_')]
    graph_cols = [c for c in graph_df.columns if c.startswith('graph_feat_')]
    
    print(f"Node features: {len(node_cols)}")
    print(f"Edge features: {len(edge_cols)}")
    print(f"Graph features: {len(graph_cols)}")
    
    print("\nComputing statistics...")
    node_stats = compute_statistics(node_df, node_cols)
    node_stats.to_csv(output_dir / 'node_stats.csv', index=False)
    
    print("\nGenerating plots...")
    print("  - Class distribution")
    plot_class_balance(graph_df, output_dir)
    
    print("  - Node distributions")
    plot_distributions(node_df, node_cols, output_dir, 'node')
    
    print("  - Node correlations")
    plot_correlation(node_df, node_cols, output_dir, 'node')
    
    if edge_cols:
        print("  - Edge distributions")
        plot_distributions(edge_df, edge_cols, output_dir, 'edge')
    
    print("  - PCA")
    plot_pca(graph_df, graph_cols + ['node_feat_mean', 'node_feat_std'], output_dir, 'graph')
    
    print("  - t-SNE")
    plot_tsne(graph_df, graph_cols + ['node_feat_mean', 'node_feat_std'], output_dir, 'graph')
    
    print("\nGenerating report...")
    generate_report(graph_df, node_stats, output_dir)
    
    graph_df.to_csv(output_dir / 'graph_data.csv', index=False)
    
    print(f"\n{'='*60}")
    print(f"Done! Results in: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config.yaml')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    main(args.config, args.output)
