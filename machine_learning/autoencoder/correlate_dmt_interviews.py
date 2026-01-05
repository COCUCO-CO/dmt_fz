#!/usr/bin/env python3
"""
Correlate DMT-Only VAE embeddings with subjective interview reports.

Uses the best model from hyperparameter search to extract interpretable
metrics and correlate with questionnaire responses.

Metrics derived from latent space:
1. centroid_distance: How unique is this subject's brain state
2. dispersion: Variability of brain states (metastability analog)
3. trajectory_length: Total distance traveled in latent space
4. total_variance: Overall variance of latent representations
5. mean_magnitude: Average activation level

Usage:
    python correlate_dmt_interviews.py --search-dir <path_to_search_results>
"""

import argparse
import pickle
from pathlib import Path
import logging
import json

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import fdrcorrection

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_embeddings(exp_dir: Path):
    """Load all embeddings from an experiment."""
    emb_dir = exp_dir / 'output' / 'embeddings'
    
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
    subjects = np.array(all_subjects)
    
    # Clean subject IDs (S01-DMT -> S01)
    subjects = np.array([s.split('-')[0] if '-' in s else s for s in subjects])
    
    logger.info(f"Loaded {len(embeddings)} embeddings from {len(set(subjects))} subjects")
    
    return embeddings, subjects


def compute_metrics(embeddings: np.ndarray, subjects: np.ndarray):
    """
    Compute interpretable per-subject metrics from latent embeddings.
    
    Analogous to Kuramoto coherence (mean) and metastability (variance).
    """
    unique_subjects = sorted(set(subjects))
    global_centroid = embeddings.mean(axis=0)
    
    metrics = {
        'subject_id': [],
        'centroid_distance': [],    # Uniqueness
        'dispersion': [],           # Metastability analog
        'mean_magnitude': [],       # Average activation
        'total_variance': [],       # Overall variance
        'trajectory_length': [],    # State transitions
        'n_epochs': [],             # Number of epochs
    }
    
    for subj in unique_subjects:
        mask = subjects == subj
        subj_emb = embeddings[mask]
        n_epochs = len(subj_emb)
        
        # Subject centroid
        subj_centroid = subj_emb.mean(axis=0)
        
        # 1. Distance from global centroid
        centroid_dist = np.linalg.norm(subj_centroid - global_centroid)
        
        # 2. Dispersion: mean distance from subject's own centroid
        distances = np.linalg.norm(subj_emb - subj_centroid, axis=1)
        dispersion = distances.mean()
        
        # 3. Mean magnitude of latent vectors
        magnitudes = np.linalg.norm(subj_emb, axis=1)
        mean_mag = magnitudes.mean()
        
        # 4. Total variance
        total_var = subj_emb.var(axis=0).sum()
        
        # 5. Trajectory length
        if n_epochs > 1:
            diffs = np.diff(subj_emb, axis=0)
            trajectory = np.linalg.norm(diffs, axis=1).sum()
        else:
            trajectory = 0.0
        
        metrics['subject_id'].append(subj)
        metrics['centroid_distance'].append(centroid_dist)
        metrics['dispersion'].append(dispersion)
        metrics['mean_magnitude'].append(mean_mag)
        metrics['total_variance'].append(total_var)
        metrics['trajectory_length'].append(trajectory)
        metrics['n_epochs'].append(n_epochs)
    
    return pd.DataFrame(metrics)


def load_interviews(spectral_dir: Path):
    """Load interview/questionnaire data."""
    labels = list(pd.read_csv(spectral_dir / 'target_labels.txt', header=None)[0])
    targets = pd.read_csv(spectral_dir / 'target.csv', header=None, names=labels)
    return targets, labels


def match_subjects(ae_subjects: list, n_interview_subjects: int):
    """Match AE subjects with interview subjects."""
    # Rejected subjects (from pearson.py)
    rejected = [3, 6, 9, 17, 24, 32]
    valid_subjects = [i for i in range(1, 36) if i not in rejected]
    
    subj_to_row = {subj: idx for idx, subj in enumerate(valid_subjects)}
    
    matched_ae = []
    matched_rows = []
    
    for subj_id in ae_subjects:
        subj_num = int(subj_id[1:])  # S01 -> 1
        if subj_num in subj_to_row:
            row_idx = subj_to_row[subj_num]
            if row_idx < n_interview_subjects:
                matched_ae.append(subj_id)
                matched_rows.append(row_idx)
    
    return matched_ae, matched_rows


def compute_correlations(metrics_df: pd.DataFrame, interview_df: pd.DataFrame, 
                         matched_subjects: list, matched_rows: list):
    """Compute correlations between metrics and interviews."""
    # Filter metrics to matched subjects
    metrics_matched = metrics_df[metrics_df['subject_id'].isin(matched_subjects)]
    metrics_matched = metrics_matched.set_index('subject_id').loc[matched_subjects]
    
    # Get interview data for matched subjects
    interview_matched = interview_df.iloc[matched_rows]
    
    # Metric columns (exclude subject_id and n_epochs)
    metric_cols = [c for c in metrics_matched.columns if c not in ['subject_id', 'n_epochs']]
    interview_cols = interview_df.columns.tolist()
    
    n_metrics = len(metric_cols)
    n_interviews = len(interview_cols)
    
    correlations = np.zeros((n_metrics, n_interviews))
    pvalues = np.zeros((n_metrics, n_interviews))
    
    for i, metric in enumerate(metric_cols):
        for j, interview in enumerate(interview_cols):
            x = metrics_matched[metric].values
            y = interview_matched[interview].values
            
            # Remove NaN
            valid = ~(np.isnan(x) | np.isnan(y))
            if valid.sum() >= 5:
                r, p = pearsonr(x[valid], y[valid])
                correlations[i, j] = r
                pvalues[i, j] = p
            else:
                correlations[i, j] = 0
                pvalues[i, j] = 1.0
    
    return correlations, pvalues, metric_cols, interview_cols


def plot_results(correlations, pvalues, metric_cols, interview_cols, output_dir):
    """Generate visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # FDR correction
    flat_pvals = pvalues.flatten()
    rejected, corrected = fdrcorrection(flat_pvals, alpha=0.05)
    corrected_pvals = corrected.reshape(pvalues.shape)
    
    # Count significant correlations
    n_significant = rejected.sum()
    logger.info(f"\nSignificant correlations (FDR < 0.05): {n_significant}/{len(flat_pvals)}")
    
    # 1. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(16, 8))
    
    mask = corrected_pvals >= 0.05
    
    sns.heatmap(correlations, 
                xticklabels=interview_cols,
                yticklabels=metric_cols,
                cmap='RdBu_r',
                center=0,
                vmin=-0.7, vmax=0.7,
                annot=True, fmt='.2f',
                mask=mask if n_significant > 0 else None,
                ax=ax)
    
    ax.set_title(f'Correlación: Métricas Latentes vs Reportes Subjetivos\n'
                 f'({n_significant} correlaciones significativas, FDR < 0.05)')
    ax.set_xlabel('Variables de Entrevista')
    ax.set_ylabel('Métricas del Espacio Latente')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png', dpi=150)
    plt.close()
    
    # 2. Top correlations bar chart
    corr_flat = correlations.flatten()
    pval_flat = corrected_pvals.flatten()
    
    indices = np.argsort(np.abs(corr_flat))[::-1][:20]  # Top 20
    
    top_corrs = []
    for idx in indices:
        i, j = np.unravel_index(idx, correlations.shape)
        top_corrs.append({
            'metric': metric_cols[i],
            'interview': interview_cols[j],
            'correlation': corr_flat[idx],
            'pvalue_fdr': pval_flat[idx],
            'significant': pval_flat[idx] < 0.05
        })
    
    top_df = pd.DataFrame(top_corrs)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['green' if sig else 'gray' for sig in top_df['significant']]
    
    labels = [f"{row['metric']} vs {row['interview'][:20]}..." 
              for _, row in top_df.iterrows()]
    
    bars = ax.barh(range(len(top_df)), top_df['correlation'], color=colors)
    ax.set_yticks(range(len(top_df)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Correlación (Pearson r)')
    ax.set_title('Top 20 Correlaciones (verde = significativo FDR < 0.05)')
    ax.axvline(0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'top_correlations.png', dpi=150)
    plt.close()
    
    # 3. Save detailed results
    all_corrs = []
    for i, metric in enumerate(metric_cols):
        for j, interview in enumerate(interview_cols):
            all_corrs.append({
                'metric': metric,
                'interview': interview,
                'correlation': correlations[i, j],
                'pvalue_raw': pvalues[i, j],
                'pvalue_fdr': corrected_pvals[i, j],
                'significant': corrected_pvals[i, j] < 0.05
            })
    
    results_df = pd.DataFrame(all_corrs)
    results_df = results_df.sort_values('pvalue_fdr')
    results_df.to_csv(output_dir / 'all_correlations.csv', index=False)
    
    # Print significant ones
    sig_df = results_df[results_df['significant']]
    if len(sig_df) > 0:
        logger.info("\n=== CORRELACIONES SIGNIFICATIVAS (FDR < 0.05) ===")
        for _, row in sig_df.iterrows():
            logger.info(f"  {row['metric']} vs {row['interview']}: r={row['correlation']:.3f}, p={row['pvalue_fdr']:.4f}")
    else:
        logger.info("\nNo significant correlations after FDR correction")
        logger.info("Top 5 strongest correlations (uncorrected):")
        for _, row in results_df.head(5).iterrows():
            logger.info(f"  {row['metric']} vs {row['interview']}: r={row['correlation']:.3f}, p_raw={row['pvalue_raw']:.4f}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--search-dir', type=str, 
                        default='/media/storage_hdd/dmt_fz/machine_learning/autoencoder/hyperparam_search_dmt/search_20251201_194555')
    parser.add_argument('--spectral-dir', type=str,
                        default='/media/storage_hdd/dmt_fz/spectral_sources')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()
    
    search_dir = Path(args.search_dir)
    spectral_dir = Path(args.spectral_dir)
    
    # Find best experiment
    best_exp_file = search_dir / 'best_experiment.json'
    if best_exp_file.exists():
        with open(best_exp_file, 'r') as f:
            best_info = json.load(f)
        exp_id = best_info['experiment_id']
    else:
        # Find by directory name pattern
        exp_dirs = list(search_dir.glob('exp_*'))
        exp_id = sorted(exp_dirs)[0].name
    
    exp_dir = search_dir / exp_id
    logger.info(f"Using best experiment: {exp_id}")
    
    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = search_dir / 'interview_correlation'
    
    # Load embeddings
    logger.info("\n1. Loading embeddings...")
    embeddings, subjects = load_embeddings(exp_dir)
    
    # Compute metrics
    logger.info("\n2. Computing interpretable metrics...")
    metrics_df = compute_metrics(embeddings, subjects)
    logger.info(f"   Computed metrics for {len(metrics_df)} subjects")
    logger.info(f"   Metrics: {list(metrics_df.columns)}")
    
    # Load interviews
    logger.info("\n3. Loading interview data...")
    interview_df, interview_labels = load_interviews(spectral_dir)
    logger.info(f"   Loaded {len(interview_df)} subjects, {len(interview_labels)} variables")
    
    # Match subjects
    logger.info("\n4. Matching subjects...")
    matched_subjects, matched_rows = match_subjects(
        metrics_df['subject_id'].tolist(),
        len(interview_df)
    )
    logger.info(f"   Matched {len(matched_subjects)} subjects")
    
    # Compute correlations
    logger.info("\n5. Computing correlations...")
    correlations, pvalues, metric_cols, interview_cols = compute_correlations(
        metrics_df, interview_df, matched_subjects, matched_rows
    )
    
    # Plot and save results
    logger.info("\n6. Generating visualizations...")
    results_df = plot_results(correlations, pvalues, metric_cols, interview_cols, output_dir)
    
    # Save metrics
    metrics_df.to_csv(output_dir / 'subject_metrics.csv', index=False)
    
    logger.info(f"\n✓ Results saved to: {output_dir}")
    logger.info("  - correlation_heatmap.png")
    logger.info("  - top_correlations.png")
    logger.info("  - all_correlations.csv")
    logger.info("  - subject_metrics.csv")


if __name__ == '__main__':
    main()












