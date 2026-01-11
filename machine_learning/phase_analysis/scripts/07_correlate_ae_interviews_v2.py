#!/usr/bin/env python3
"""
07_correlate_ae_interviews_v2.py

Correlate INTERPRETABLE autoencoder-derived metrics with subjective experience.

SCIENTIFIC RATIONALE:
---------------------
In pearson.py, they correlate:
- Kuramoto order parameter (mean per subject) = "coherence" = typical sync level
- Kuramoto variance (per subject) = "metastability" = variability of sync

For autoencoder, we derive ANALOGOUS metrics:
1. Reconstruction error per subject = how "typical" is this brain state?
   - Low error = brain patterns fit the learned manifold well
   - High error = unusual/atypical patterns
   
2. Latent space entropy/dispersion = how variable are the brain states?
   - High dispersion = subject traverses many different states (like high metastability)
   - Low dispersion = subject stays in similar states
   
3. Distance from global centroid = how "different" is this subject from average?

4. Cluster transition entropy = how unpredictable are state changes?

These are interpretable and analogous to what pearson.py does.

Usage:
    python scripts/07_correlate_ae_interviews_v2.py
"""

import argparse
import json
import pickle
from pathlib import Path
from itertools import product
import logging

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, entropy
from scipy.spatial.distance import cdist
from statsmodels.stats.multitest import fdrcorrection

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_autoencoder_dmt(ae_dir: Path):
    """Load autoencoder embeddings for DMT condition."""
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
    
    # Filter DMT only
    dmt_mask = conditions == 'DMT'
    embeddings = embeddings[dmt_mask]
    subjects = subjects[dmt_mask]
    
    # Clean subject IDs
    subjects = np.array([s.split('-')[0] for s in subjects])
    
    return embeddings, subjects


def compute_interpretable_metrics(embeddings: np.ndarray, subjects: np.ndarray):
    """
    Compute interpretable per-subject metrics from latent embeddings.
    
    These are ANALOGOUS to Kuramoto mean (coherence) and variance (metastability).
    
    Returns dict with:
        - latent_centroid_distance: how far is subject's mean from global mean
        - latent_dispersion: spread of subject's points in latent space (like metastability)
        - latent_magnitude: mean norm of latent vectors
        - latent_variance: total variance of latent vectors
        - state_entropy: entropy of cluster assignments (if clusters provided)
    """
    unique_subjects = sorted(set(subjects))
    n_subjects = len(unique_subjects)
    
    # Global centroid (mean of all embeddings)
    global_centroid = embeddings.mean(axis=0)
    
    metrics = {
        'subject_id': [],
        'centroid_distance': [],      # Distance from global mean (uniqueness)
        'dispersion': [],             # Within-subject spread (metastability analog)
        'mean_magnitude': [],         # Average latent norm
        'total_variance': [],         # Total variance across dims
        'trajectory_length': [],      # Total distance traveled in latent space
    }
    
    for subj in unique_subjects:
        mask = subjects == subj
        subj_emb = embeddings[mask]
        n_epochs = len(subj_emb)
        
        # Subject centroid
        subj_centroid = subj_emb.mean(axis=0)
        
        # 1. Distance from global centroid (how "different" is this subject)
        centroid_dist = np.linalg.norm(subj_centroid - global_centroid)
        
        # 2. Dispersion: mean distance from subject's own centroid (metastability)
        distances_to_centroid = np.linalg.norm(subj_emb - subj_centroid, axis=1)
        dispersion = distances_to_centroid.mean()
        
        # 3. Mean magnitude of latent vectors
        magnitudes = np.linalg.norm(subj_emb, axis=1)
        mean_mag = magnitudes.mean()
        
        # 4. Total variance (sum of variance across all dims)
        total_var = subj_emb.var(axis=0).sum()
        
        # 5. Trajectory length: sum of distances between consecutive epochs
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
    
    return pd.DataFrame(metrics)


def load_interview_data(spectral_dir: Path):
    """Load interview/questionnaire data."""
    labels = list(pd.read_csv(spectral_dir / 'target_labels.txt', header=None)[0])
    targets = pd.read_csv(spectral_dir / 'target.csv', header=None, names=labels)
    return targets, labels


def match_subjects(ae_subjects: list, n_interview_subjects: int):
    """Match autoencoder subjects with interview subjects."""
    # From pearson.py: rejected_subjects = [2, 5, 8, 16, 23, 31] (0-indexed)
    # These are 1-indexed: 3, 6, 9, 17, 24, 32
    rejected = [3, 6, 9, 17, 24, 32]
    valid_subjects = [i for i in range(1, 36) if i not in rejected]
    
    subj_to_row = {subj: idx for idx, subj in enumerate(valid_subjects)}
    
    ae_to_interview = []
    valid_ae_subjects = []
    
    for subj_id in ae_subjects:
        subj_num = int(subj_id[1:])  # S01 -> 1
        if subj_num in subj_to_row:
            row_idx = subj_to_row[subj_num]
            if row_idx < n_interview_subjects:
                ae_to_interview.append(row_idx)
                valid_ae_subjects.append(subj_id)
    
    return valid_ae_subjects, ae_to_interview


def plot_correlation_heatmap(
    correlations: np.ndarray,
    pvalues: np.ndarray,
    metric_names: list,
    interview_names: list,
    title: str,
    output_path: Path,
    fdr_alpha: float = 0.05
):
    """Plot correlation heatmap with FDR correction."""
    n_metrics = len(metric_names)
    n_interviews = len(interview_names)
    
    # FDR correction
    rejected, pvals_corrected = fdrcorrection(pvalues.flatten(), alpha=fdr_alpha)
    rejected = rejected.reshape(n_metrics, n_interviews)
    pvals_corrected = pvals_corrected.reshape(n_metrics, n_interviews)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Create color matrix
    im = ax.imshow(correlations, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    ax.set_xticks(np.arange(n_interviews))
    ax.set_xticklabels(interview_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(np.arange(n_metrics))
    ax.set_yticklabels(metric_names, fontsize=10)
    
    # Add text annotations
    for i in range(n_metrics):
        for j in range(n_interviews):
            r = correlations[i, j]
            sig = rejected[i, j]
            
            # Stars for significance
            if sig:
                if pvals_corrected[i, j] < 0.001:
                    stars = '***'
                elif pvals_corrected[i, j] < 0.01:
                    stars = '**'
                else:
                    stars = '*'
            else:
                stars = ''
            
            color = 'white' if abs(r) > 0.4 else 'black'
            weight = 'bold' if sig else 'normal'
            ax.text(j, i, f'{r:.2f}{stars}', ha='center', va='center',
                    color=color, fontsize=8, weight=weight)
    
    ax.set_title(title, fontsize=12, pad=10)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Pearson r', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    n_sig = rejected.sum()
    logger.info(f"Saved: {output_path} ({n_sig} significant, FDR α={fdr_alpha})")
    
    return rejected, pvals_corrected


def plot_scatter(x, y, xlabel, ylabel, r, p, output_path):
    """Plot scatter with regression line."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(x, y, s=100, alpha=0.7, edgecolor='black')
    
    # Regression line
    z = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, np.poly1d(z)(x_line), 'r--', linewidth=2,
            label=f'r = {r:.3f}, p = {p:.4f}')
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fdr-alpha', type=float, default=0.05)
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("AUTOENCODER-INTERVIEW CORRELATION (Interpretable Metrics)")
    logger.info("=" * 70)
    
    # Paths
    base_dir = Path(__file__).parent.parent
    ae_dir = base_dir.parent / 'autoencoder' / 'output_alpha'
    spectral_dir = Path('/media/storage_hdd/dmt_fz/spectral_sources')
    output_dir = base_dir / 'output' / 'ae_interview_correlation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("\n[Loading Data]")
    embeddings, subjects = load_autoencoder_dmt(ae_dir)
    logger.info(f"  AE embeddings: {embeddings.shape} (DMT only)")
    
    targets, interview_labels = load_interview_data(spectral_dir)
    logger.info(f"  Interview data: {targets.shape}")
    
    # Compute interpretable metrics
    logger.info("\n[Computing Interpretable Metrics per Subject]")
    metrics_df = compute_interpretable_metrics(embeddings, subjects)
    logger.info(f"  Computed {len(metrics_df.columns)-1} metrics for {len(metrics_df)} subjects")
    
    # Match subjects
    valid_subjects, interview_rows = match_subjects(
        metrics_df['subject_id'].tolist(), 
        len(targets)
    )
    logger.info(f"  Matched subjects: {len(valid_subjects)}")
    
    if len(valid_subjects) < 5:
        logger.error("Not enough matched subjects!")
        return
    
    # Filter to matched subjects
    metrics_df = metrics_df[metrics_df['subject_id'].isin(valid_subjects)].reset_index(drop=True)
    targets_matched = targets.iloc[interview_rows].reset_index(drop=True)
    
    # Metric columns (exclude subject_id)
    metric_cols = [c for c in metrics_df.columns if c != 'subject_id']
    metric_names_display = {
        'centroid_distance': 'Uniqueness\n(dist from global mean)',
        'dispersion': 'Metastability\n(within-subject spread)',
        'mean_magnitude': 'Mean Latent\nMagnitude',
        'total_variance': 'Total Latent\nVariance',
        'trajectory_length': 'Trajectory Length\n(total movement)'
    }
    
    # Compute correlations
    logger.info("\n[Computing Correlations]")
    n_metrics = len(metric_cols)
    n_interviews = len(interview_labels)
    
    correlations = np.zeros((n_metrics, n_interviews))
    pvalues = np.ones((n_metrics, n_interviews))
    
    for i, metric in enumerate(metric_cols):
        x = metrics_df[metric].values
        for j in range(n_interviews):
            y = targets_matched.iloc[:, j].values
            valid = ~(np.isnan(x) | np.isnan(y))
            if valid.sum() > 2:
                r, p = pearsonr(x[valid], y[valid])
                correlations[i, j] = r
                pvalues[i, j] = p
    
    # Plot heatmap
    metric_labels = [metric_names_display.get(m, m) for m in metric_cols]
    
    rejected, pvals_corrected = plot_correlation_heatmap(
        correlations, pvalues, 
        metric_labels, interview_labels,
        'Autoencoder Latent Metrics vs Subjective Experience (Alpha Band, DMT)',
        output_dir / 'heatmap_metrics_vs_interviews.png',
        fdr_alpha=args.fdr_alpha
    )
    
    # Generate scatter plots for significant correlations
    logger.info("\n[Generating Scatter Plots for Significant Correlations]")
    
    for i, metric in enumerate(metric_cols):
        for j in range(n_interviews):
            if not rejected[i, j]:
                continue
            
            x = metrics_df[metric].values
            y = targets_matched.iloc[:, j].values
            valid = ~(np.isnan(x) | np.isnan(y))
            
            r = correlations[i, j]
            p = pvals_corrected[i, j]
            
            fname = f'scatter_{metric}_{interview_labels[j]}.png'
            fname = fname.replace(' ', '_').replace('/', '_')
            
            plot_scatter(
                x[valid], y[valid],
                metric_names_display.get(metric, metric),
                interview_labels[j],
                r, p,
                output_dir / fname
            )
            logger.info(f"  {metric} vs {interview_labels[j]}: r={r:.3f}, p={p:.4f}")
    
    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Subjects: {len(valid_subjects)}")
    logger.info(f"  Metrics: {n_metrics}")
    logger.info(f"  Interview variables: {n_interviews}")
    logger.info(f"  Total tests: {n_metrics * n_interviews}")
    logger.info(f"  Significant (FDR α={args.fdr_alpha}): {rejected.sum()}")
    
    # Print significant correlations
    if rejected.sum() > 0:
        logger.info("\n  Significant correlations:")
        for i, metric in enumerate(metric_cols):
            for j in range(n_interviews):
                if rejected[i, j]:
                    r = correlations[i, j]
                    logger.info(f"    {metric} ↔ {interview_labels[j]}: r={r:.3f}")
    
    logger.info(f"\n  Results: {output_dir}")
    logger.info("=" * 70)
    
    # Save results
    results = {
        'n_subjects': len(valid_subjects),
        'n_metrics': n_metrics,
        'n_interviews': n_interviews,
        'n_significant': int(rejected.sum()),
        'fdr_alpha': args.fdr_alpha,
        'metrics': metric_cols,
        'correlations': correlations.tolist(),
        'pvalues': pvalues.tolist(),
        'rejected': rejected.tolist()
    }
    with open(output_dir / 'correlation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    metrics_df.to_csv(output_dir / 'subject_metrics.csv', index=False)


if __name__ == '__main__':
    main()













