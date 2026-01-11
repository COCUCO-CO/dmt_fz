#!/usr/bin/env python3
"""
Complete Latent Space Analysis for DMT VAE

Includes:
1. Interpretable metrics extraction
2. PCA analysis of latent space
3. Kuramoto cross-validation
4. Correlation with interviews

Usage:
    python analyze_latent_complete.py --config config/config_dmt_delta.yaml
"""

import argparse
import pickle
from pathlib import Path
import logging
import json
import yaml

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import fdrcorrection
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def load_embeddings(output_dir: Path):
    """Load all embeddings from output directory."""
    emb_dir = output_dir / 'embeddings'
    
    all_embeddings = []
    all_subjects = []
    
    for split in ['train', 'val', 'test']:
        path = emb_dir / f'{split}_embeddings.pkl'
        if path.exists():
            with open(path, 'rb') as f:
                data = pickle.load(f)
            all_embeddings.append(data['latent_embeddings'])
            all_subjects.extend(data['subjects'])
    
    embeddings = np.vstack(all_embeddings)
    subjects = np.array([s.split('-')[0] if '-' in s else s for s in all_subjects])
    
    logger.info(f"Loaded {len(embeddings)} embeddings from {len(set(subjects))} subjects")
    return embeddings, subjects


def load_kuramoto_data(phases_dir: Path, band: str, condition: str = 'DMT'):
    """Load Kuramoto order parameter data for validation."""
    kuramoto_data = {}
    cond_dir = phases_dir / condition
    
    for phases_file in cond_dir.glob('phases-*.pkl'):
        subject_id = phases_file.stem.replace('phases-', '').split('-')[0]
        
        with open(phases_file, 'rb') as f:
            data = pickle.load(f)
        
        if 'kuramoto_eeg' in data and band in data['kuramoto_eeg']:
            kuramoto = np.array(data['kuramoto_eeg'][band])
            # Compute per-epoch mean and std
            kuramoto_mean = kuramoto.mean(axis=1)  # Mean over time
            kuramoto_std = kuramoto.std(axis=1)    # Std over time
            
            kuramoto_data[subject_id] = {
                'mean': kuramoto_mean.mean(),      # Subject mean coherence
                'std': kuramoto_mean.std(),        # Subject metastability
                'mean_of_std': kuramoto_std.mean() # Mean temporal variability
            }
    
    logger.info(f"Loaded Kuramoto data for {len(kuramoto_data)} subjects")
    return kuramoto_data


def compute_metrics(embeddings: np.ndarray, subjects: np.ndarray):
    """Compute interpretable per-subject metrics."""
    unique_subjects = sorted(set(subjects))
    global_centroid = embeddings.mean(axis=0)
    
    metrics = {
        'subject_id': [],
        'centroid_distance': [],
        'dispersion': [],
        'mean_magnitude': [],
        'total_variance': [],
        'trajectory_length': [],
        'n_epochs': [],
    }
    
    for subj in unique_subjects:
        mask = subjects == subj
        subj_emb = embeddings[mask]
        n_epochs = len(subj_emb)
        
        subj_centroid = subj_emb.mean(axis=0)
        
        metrics['subject_id'].append(subj)
        metrics['centroid_distance'].append(np.linalg.norm(subj_centroid - global_centroid))
        metrics['dispersion'].append(np.linalg.norm(subj_emb - subj_centroid, axis=1).mean())
        metrics['mean_magnitude'].append(np.linalg.norm(subj_emb, axis=1).mean())
        metrics['total_variance'].append(subj_emb.var(axis=0).sum())
        
        if n_epochs > 1:
            trajectory = np.linalg.norm(np.diff(subj_emb, axis=0), axis=1).sum()
        else:
            trajectory = 0.0
        metrics['trajectory_length'].append(trajectory)
        metrics['n_epochs'].append(n_epochs)
    
    return pd.DataFrame(metrics)


def perform_pca_analysis(embeddings: np.ndarray, subjects: np.ndarray, output_dir: Path):
    """Perform PCA analysis on latent space."""
    logger.info("\n=== PCA Analysis ===")
    
    # Standardize
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # PCA
    pca = PCA()
    pca_embeddings = pca.fit_transform(embeddings_scaled)
    
    # Explained variance
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    logger.info(f"Variance explained by first 5 PCs: {cumulative_var[:5]}")
    logger.info(f"PCs needed for 90% variance: {np.argmax(cumulative_var >= 0.9) + 1}")
    
    # Plot explained variance
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scree plot
    axes[0].bar(range(1, len(explained_var) + 1), explained_var, alpha=0.7, label='Individual')
    axes[0].plot(range(1, len(explained_var) + 1), cumulative_var, 'r-o', label='Cumulative')
    axes[0].axhline(y=0.9, color='g', linestyle='--', label='90% threshold')
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].set_title('PCA Scree Plot')
    axes[0].legend()
    
    # 2D projection colored by subject
    unique_subjects = sorted(set(subjects))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_subjects)))
    subj_to_color = {s: c for s, c in zip(unique_subjects, colors)}
    
    for subj in unique_subjects:
        mask = subjects == subj
        axes[1].scatter(pca_embeddings[mask, 0], pca_embeddings[mask, 1], 
                       c=[subj_to_color[subj]], alpha=0.3, s=10, label=subj)
    axes[1].set_xlabel(f'PC1 ({explained_var[0]:.1%})')
    axes[1].set_ylabel(f'PC2 ({explained_var[1]:.1%})')
    axes[1].set_title('Latent Space - First 2 PCs')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pca_analysis.png', dpi=150)
    plt.close()
    
    # Subject centroids in PC space
    subject_centroids_pca = []
    for subj in unique_subjects:
        mask = subjects == subj
        centroid = pca_embeddings[mask].mean(axis=0)
        subject_centroids_pca.append(centroid)
    
    subject_centroids_pca = np.array(subject_centroids_pca)
    
    # Plot subject centroids
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, subj in enumerate(unique_subjects):
        ax.scatter(subject_centroids_pca[i, 0], subject_centroids_pca[i, 1], 
                  s=100, c=[subj_to_color[subj]])
        ax.annotate(subj, (subject_centroids_pca[i, 0], subject_centroids_pca[i, 1]),
                   fontsize=8)
    ax.set_xlabel(f'PC1 ({explained_var[0]:.1%})')
    ax.set_ylabel(f'PC2 ({explained_var[1]:.1%})')
    ax.set_title('Subject Centroids in PCA Space')
    plt.tight_layout()
    plt.savefig(output_dir / 'pca_subject_centroids.png', dpi=150)
    plt.close()
    
    # Return PCA results for further analysis
    pca_results = {
        'explained_variance': explained_var,
        'cumulative_variance': cumulative_var,
        'n_components_90': int(np.argmax(cumulative_var >= 0.9) + 1),
        'pca_object': pca,
        'subject_centroids_pca': subject_centroids_pca,
        'unique_subjects': unique_subjects
    }
    
    return pca_results


def validate_with_kuramoto(metrics_df: pd.DataFrame, kuramoto_data: dict, output_dir: Path):
    """Cross-validate latent metrics with Kuramoto measures."""
    logger.info("\n=== Kuramoto Cross-Validation ===")
    
    # Match subjects
    matched_subjects = []
    kuramoto_coherence = []
    kuramoto_metastability = []
    
    for _, row in metrics_df.iterrows():
        subj = row['subject_id']
        if subj in kuramoto_data:
            matched_subjects.append(subj)
            kuramoto_coherence.append(kuramoto_data[subj]['mean'])
            kuramoto_metastability.append(kuramoto_data[subj]['std'])
    
    if len(matched_subjects) < 5:
        logger.warning("Not enough matched subjects for Kuramoto validation")
        return None
    
    # Get metrics for matched subjects
    metrics_matched = metrics_df[metrics_df['subject_id'].isin(matched_subjects)]
    metrics_matched = metrics_matched.set_index('subject_id').loc[matched_subjects]
    
    kuramoto_coherence = np.array(kuramoto_coherence)
    kuramoto_metastability = np.array(kuramoto_metastability)
    
    # Correlate each latent metric with Kuramoto
    metric_cols = ['centroid_distance', 'dispersion', 'mean_magnitude', 
                   'total_variance', 'trajectory_length']
    
    results = []
    for metric in metric_cols:
        x = metrics_matched[metric].values
        
        # Coherence correlation
        r_coh, p_coh = pearsonr(x, kuramoto_coherence)
        results.append({
            'latent_metric': metric,
            'kuramoto_metric': 'coherence (mean)',
            'correlation': r_coh,
            'pvalue': p_coh
        })
        
        # Metastability correlation
        r_meta, p_meta = pearsonr(x, kuramoto_metastability)
        results.append({
            'latent_metric': metric,
            'kuramoto_metric': 'metastability (std)',
            'correlation': r_meta,
            'pvalue': p_meta
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('pvalue')
    
    logger.info("\nKuramoto Validation Results:")
    for _, row in results_df.head(10).iterrows():
        sig = '*' if row['pvalue'] < 0.05 else ''
        logger.info(f"  {row['latent_metric']} vs {row['kuramoto_metric']}: "
                   f"r={row['correlation']:.3f}, p={row['pvalue']:.4f} {sig}")
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metric_cols):
        x = metrics_matched[metric].values
        
        ax = axes[i]
        ax.scatter(x, kuramoto_coherence, alpha=0.7, label='Coherence')
        ax.scatter(x, kuramoto_metastability, alpha=0.7, label='Metastability')
        
        # Add correlation text
        r_coh, _ = pearsonr(x, kuramoto_coherence)
        r_meta, _ = pearsonr(x, kuramoto_metastability)
        ax.set_xlabel(metric)
        ax.set_ylabel('Kuramoto')
        ax.set_title(f'{metric}\nr_coh={r_coh:.2f}, r_meta={r_meta:.2f}')
        ax.legend()
    
    axes[-1].axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / 'kuramoto_validation.png', dpi=150)
    plt.close()
    
    results_df.to_csv(output_dir / 'kuramoto_validation.csv', index=False)
    
    return results_df


def load_interviews(spectral_dir: Path):
    """Load interview data."""
    labels = list(pd.read_csv(spectral_dir / 'target_labels.txt', header=None)[0])
    targets = pd.read_csv(spectral_dir / 'target.csv', header=None, names=labels)
    return targets, labels


def match_subjects_to_interviews(ae_subjects: list, n_interview_subjects: int):
    """Match AE subjects with interview subjects."""
    rejected = [3, 6, 9, 17, 24, 32]
    valid_subjects = [i for i in range(1, 36) if i not in rejected]
    subj_to_row = {subj: idx for idx, subj in enumerate(valid_subjects)}
    
    matched_ae = []
    matched_rows = []
    
    for subj_id in ae_subjects:
        subj_num = int(subj_id[1:])
        if subj_num in subj_to_row:
            row_idx = subj_to_row[subj_num]
            if row_idx < n_interview_subjects:
                matched_ae.append(subj_id)
                matched_rows.append(row_idx)
    
    return matched_ae, matched_rows


def correlate_with_interviews(metrics_df: pd.DataFrame, pca_results: dict,
                             interview_df: pd.DataFrame, matched_subjects: list,
                             matched_rows: list, output_dir: Path):
    """Correlate metrics and PCA components with interviews."""
    logger.info("\n=== Interview Correlations ===")
    
    # Filter metrics
    metrics_matched = metrics_df[metrics_df['subject_id'].isin(matched_subjects)]
    metrics_matched = metrics_matched.set_index('subject_id').loc[matched_subjects]
    
    # Get PCA centroids for matched subjects
    subj_to_idx = {s: i for i, s in enumerate(pca_results['unique_subjects'])}
    pca_centroids = np.array([pca_results['subject_centroids_pca'][subj_to_idx[s]] 
                              for s in matched_subjects])
    
    # Add first 3 PCs as metrics
    for i in range(min(3, pca_centroids.shape[1])):
        metrics_matched[f'PC{i+1}'] = pca_centroids[:, i]
    
    # Interview data
    interview_matched = interview_df.iloc[matched_rows]
    
    # Metric columns
    metric_cols = ['centroid_distance', 'dispersion', 'mean_magnitude', 
                   'total_variance', 'trajectory_length', 'PC1', 'PC2', 'PC3']
    metric_cols = [c for c in metric_cols if c in metrics_matched.columns]
    interview_cols = interview_df.columns.tolist()
    
    # Compute correlations
    n_metrics = len(metric_cols)
    n_interviews = len(interview_cols)
    
    correlations = np.zeros((n_metrics, n_interviews))
    pvalues = np.zeros((n_metrics, n_interviews))
    
    for i, metric in enumerate(metric_cols):
        for j, interview in enumerate(interview_cols):
            x = metrics_matched[metric].values
            y = interview_matched[interview].values
            
            valid = ~(np.isnan(x) | np.isnan(y))
            if valid.sum() >= 5:
                r, p = pearsonr(x[valid], y[valid])
                correlations[i, j] = r
                pvalues[i, j] = p
            else:
                correlations[i, j] = 0
                pvalues[i, j] = 1.0
    
    # FDR correction
    flat_pvals = pvalues.flatten()
    rejected, corrected = fdrcorrection(flat_pvals, alpha=0.05)
    corrected_pvals = corrected.reshape(pvalues.shape)
    
    n_significant = rejected.sum()
    logger.info(f"Significant correlations (FDR < 0.05): {n_significant}/{len(flat_pvals)}")
    
    # Create results dataframe
    all_results = []
    for i, metric in enumerate(metric_cols):
        for j, interview in enumerate(interview_cols):
            all_results.append({
                'metric': metric,
                'interview': interview,
                'correlation': correlations[i, j],
                'pvalue_raw': pvalues[i, j],
                'pvalue_fdr': corrected_pvals[i, j],
                'significant': corrected_pvals[i, j] < 0.05
            })
    
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('pvalue_fdr')
    
    # Print top results
    logger.info("\nTop 10 correlations:")
    for _, row in results_df.head(10).iterrows():
        sig = '***' if row['pvalue_fdr'] < 0.05 else ('*' if row['pvalue_raw'] < 0.05 else '')
        logger.info(f"  {row['metric']} vs {row['interview']}: "
                   f"r={row['correlation']:.3f}, p_fdr={row['pvalue_fdr']:.4f} {sig}")
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(18, 10))
    
    sns.heatmap(correlations, 
                xticklabels=interview_cols,
                yticklabels=metric_cols,
                cmap='RdBu_r',
                center=0,
                vmin=-0.7, vmax=0.7,
                annot=True, fmt='.2f',
                ax=ax)
    
    # Mark significant cells
    for i in range(n_metrics):
        for j in range(n_interviews):
            if corrected_pvals[i, j] < 0.05:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, 
                                          edgecolor='gold', linewidth=3))
    
    ax.set_title(f'Correlaciones: Métricas Latentes + PCA vs Entrevistas\n'
                 f'({n_significant} significativas FDR < 0.05, recuadro dorado)')
    ax.set_xlabel('Variables de Entrevista')
    ax.set_ylabel('Métricas')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap_complete.png', dpi=150)
    plt.close()
    
    results_df.to_csv(output_dir / 'correlations_complete.csv', index=False)
    
    return results_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    output_base = Path(config['paths']['output_dir'])
    band = config['data']['bands'][0]
    phases_dir = Path(config['paths']['phases_dir'])
    spectral_dir = Path('/media/storage_hdd/dmt_fz/spectral_sources')
    
    if args.output_dir:
        analysis_dir = Path(args.output_dir)
    else:
        analysis_dir = output_base / 'complete_analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Analyzing {band} band embeddings")
    logger.info(f"Output: {analysis_dir}")
    
    # 1. Load embeddings
    logger.info("\n1. Loading embeddings...")
    embeddings, subjects = load_embeddings(output_base)
    
    # 2. Compute metrics
    logger.info("\n2. Computing interpretable metrics...")
    metrics_df = compute_metrics(embeddings, subjects)
    metrics_df.to_csv(analysis_dir / 'subject_metrics.csv', index=False)
    
    # 3. PCA Analysis
    logger.info("\n3. Performing PCA analysis...")
    pca_results = perform_pca_analysis(embeddings, subjects, analysis_dir)
    
    # 4. Kuramoto validation
    logger.info("\n4. Kuramoto cross-validation...")
    kuramoto_data = load_kuramoto_data(phases_dir, band)
    kuramoto_results = validate_with_kuramoto(metrics_df, kuramoto_data, analysis_dir)
    
    # 5. Interview correlations
    logger.info("\n5. Interview correlations...")
    interview_df, _ = load_interviews(spectral_dir)
    matched_subjects, matched_rows = match_subjects_to_interviews(
        metrics_df['subject_id'].tolist(), len(interview_df)
    )
    logger.info(f"   Matched {len(matched_subjects)} subjects")
    
    interview_results = correlate_with_interviews(
        metrics_df, pca_results, interview_df, 
        matched_subjects, matched_rows, analysis_dir
    )
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Band: {band}")
    logger.info(f"Subjects: {len(set(subjects))}")
    logger.info(f"Embeddings: {len(embeddings)}")
    logger.info(f"Latent dims: {embeddings.shape[1]}")
    logger.info(f"PCs for 90% variance: {pca_results['n_components_90']}")
    logger.info(f"Output: {analysis_dir}")
    
    # Save summary
    summary = {
        'band': band,
        'n_subjects': len(set(subjects)),
        'n_embeddings': len(embeddings),
        'latent_dim': embeddings.shape[1],
        'pca_components_90': pca_results['n_components_90'],
        'n_significant_interviews': int(interview_results['significant'].sum())
    }
    
    with open(analysis_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()













