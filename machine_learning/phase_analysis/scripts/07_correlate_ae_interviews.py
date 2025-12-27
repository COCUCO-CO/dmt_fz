#!/usr/bin/env python3
"""
07_correlate_ae_interviews.py

Correlate autoencoder latent features with subjective experience reports.

Similar to pearson.py but using autoencoder embeddings instead of network sync.

For each subject, we extract:
1. Mean latent vector (captures "typical" brain state during DMT)
2. Latent variance (captures "variability" of brain states)
3. Cluster proportions (how much time in each state)

Then correlate with interview scores using Pearson + FDR correction.

Usage:
    python scripts/07_correlate_ae_interviews.py
"""

import argparse
import json
import pickle
from pathlib import Path
from itertools import product
import logging

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.stats.multitest import fdrcorrection

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage import color

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
    
    # Clean subject IDs (remove condition suffix, extract number)
    subjects = np.array([s.split('-')[0] for s in subjects])
    
    return embeddings, subjects


def compute_subject_features(embeddings: np.ndarray, subjects: np.ndarray):
    """
    Compute per-subject features from latent embeddings.
    
    Returns:
        mean_features: [n_subjects, latent_dim] mean latent vector per subject
        var_features: [n_subjects, latent_dim] variance per subject
        subject_order: ordered list of subject IDs
    """
    unique_subjects = sorted(set(subjects))
    n_subjects = len(unique_subjects)
    latent_dim = embeddings.shape[1]
    
    mean_features = np.zeros((n_subjects, latent_dim))
    var_features = np.zeros((n_subjects, latent_dim))
    
    for i, subj in enumerate(unique_subjects):
        mask = subjects == subj
        subj_emb = embeddings[mask]
        mean_features[i] = subj_emb.mean(axis=0)
        var_features[i] = subj_emb.var(axis=0)
    
    return mean_features, var_features, unique_subjects


def load_interview_data(spectral_dir: Path):
    """Load interview/questionnaire data."""
    labels = list(pd.read_csv(spectral_dir / 'target_labels.txt', header=None)[0])
    targets = pd.read_csv(spectral_dir / 'target.csv', header=None, names=labels)
    return targets, labels


def match_subjects(ae_subjects: list, n_interview_subjects: int):
    """
    Match autoencoder subjects with interview subjects.
    
    The interview data has subjects in order (S01, S02, ...) but some are missing.
    We need to map autoencoder subject IDs to interview row indices.
    """
    # Extract subject numbers from AE subject IDs
    subj_numbers = []
    for s in ae_subjects:
        # S01 -> 1, S02 -> 2, etc.
        num = int(s[1:])
        subj_numbers.append(num)
    
    # The interview data typically excludes certain subjects
    # We assume row i corresponds to the i-th available subject
    # This mapping needs to match pearson.py logic
    
    # From pearson.py: rejected_subjects = [2, 5, 8, 16, 23, 31] (0-indexed: 3, 6, 9, 17, 24, 32)
    # Valid subjects are: 1,2,4,5,7,8,10,11,... (1-indexed)
    rejected = [3, 6, 9, 17, 24, 32]  # 1-indexed subject numbers to reject
    valid_subjects = [i for i in range(1, 36) if i not in rejected]
    
    # Create mapping: subject number -> interview row index
    subj_to_row = {subj: idx for idx, subj in enumerate(valid_subjects)}
    
    # Map AE subjects to interview rows
    ae_to_interview = []
    valid_ae_subjects = []
    for subj_num, subj_id in zip(subj_numbers, ae_subjects):
        if subj_num in subj_to_row:
            row_idx = subj_to_row[subj_num]
            if row_idx < n_interview_subjects:
                ae_to_interview.append(row_idx)
                valid_ae_subjects.append(subj_id)
    
    return valid_ae_subjects, ae_to_interview


def r_to_color(r):
    """Convert correlation to color (plasma colormap)."""
    return cm.plasma(abs(r))


def to_gray(x):
    """Convert to grayscale for non-significant correlations."""
    rgba = cm.twilight_shifted(abs(x))
    rgb = cm.colors.to_rgb(rgba)
    gray = color.rgb2gray(np.asarray(rgb))
    return cm.gray(gray)


def plot_correlation_heatmap(
    correlations: np.ndarray,
    pvalues: np.ndarray,
    feature_names: list,
    interview_names: list,
    title: str,
    output_path: Path,
    fdr_alpha: float = 0.05
):
    """Plot correlation heatmap with FDR correction."""
    n_features = len(feature_names)
    n_interviews = len(interview_names)
    
    # Apply FDR correction
    rejected, pvals_corrected = fdrcorrection(pvalues.flatten(), alpha=fdr_alpha)
    rejected = rejected.reshape(n_features, n_interviews)
    
    # Build color matrix
    corr_colors = np.zeros((4, n_features, n_interviews))
    for i, j in product(range(n_features), range(n_interviews)):
        if rejected[i, j]:
            corr_colors[:, i, j] = r_to_color(correlations[i, j])
        else:
            corr_colors[:, i, j] = to_gray(correlations[i, j])
    
    fig, ax = plt.subplots(figsize=(16, max(8, n_features * 0.5)))
    
    ax.imshow(np.transpose(corr_colors))
    
    ax.set_xticks(np.arange(n_features))
    ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(np.arange(n_interviews))
    ax.set_yticklabels(interview_names, fontsize=10)
    
    # Add correlation values as text
    for i in range(n_features):
        for j in range(n_interviews):
            r = correlations[i, j]
            color = 'white' if abs(r) > 0.3 else 'black'
            weight = 'bold' if rejected[i, j] else 'normal'
            ax.text(i, j, f'{r:.2f}', ha='center', va='center', 
                    color=color, fontsize=8, weight=weight)
    
    ax.set_title(title, fontsize=14)
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cm.plasma, norm=plt.Normalize(-1, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
    cbar.set_label('Pearson r')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    n_sig = rejected.sum()
    logger.info(f"Saved: {output_path} ({n_sig} significant correlations)")
    
    return rejected


def plot_significant_scatters(
    features: np.ndarray,
    targets: pd.DataFrame,
    correlations: np.ndarray,
    pvalues: np.ndarray,
    rejected: np.ndarray,
    feature_names: list,
    interview_names: list,
    output_dir: Path,
    feature_type: str
):
    """Generate scatter plots for significant correlations."""
    n_features = len(feature_names)
    n_interviews = len(interview_names)
    
    for i, j in product(range(n_features), range(n_interviews)):
        if not rejected[i, j]:
            continue
        
        x = features[:, i]
        y = targets.iloc[:, j].values
        
        # Remove NaN
        valid = ~(np.isnan(x) | np.isnan(y))
        x_valid = x[valid]
        y_valid = y[valid]
        
        if len(x_valid) < 3:
            continue
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(x_valid, y_valid, s=100, alpha=0.7, edgecolor='black')
        
        # Regression line
        z = np.polyfit(x_valid, y_valid, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x_valid.min(), x_valid.max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2,
                label=f'r = {correlations[i, j]:.3f}, p = {pvalues[i, j]:.4f}')
        
        ax.set_xlabel(f'{feature_type} - {feature_names[i]}', fontsize=12)
        ax.set_ylabel(interview_names[j], fontsize=12)
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        fname = f'scatter_{feature_type}_{feature_names[i]}_{interview_names[j]}.png'
        fname = fname.replace(' ', '_').replace('/', '_')
        plt.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Scatter: {feature_names[i]} vs {interview_names[j]} (r={correlations[i, j]:.3f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top-dims', type=int, default=10,
                        help='Number of top latent dimensions to analyze')
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("AUTOENCODER-INTERVIEW CORRELATION ANALYSIS")
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
    logger.info(f"  AE embeddings: {embeddings.shape}")
    logger.info(f"  Unique subjects: {len(set(subjects))}")
    
    targets, interview_labels = load_interview_data(spectral_dir)
    logger.info(f"  Interview data: {targets.shape}")
    
    # Compute per-subject features
    logger.info("\n[Computing Subject Features]")
    mean_features, var_features, ae_subjects = compute_subject_features(embeddings, subjects)
    logger.info(f"  Mean features: {mean_features.shape}")
    logger.info(f"  Var features: {var_features.shape}")
    
    # Match subjects
    valid_subjects, interview_rows = match_subjects(ae_subjects, len(targets))
    logger.info(f"  Matched subjects: {len(valid_subjects)}")
    
    if len(valid_subjects) < 5:
        logger.error("Not enough matched subjects for correlation analysis")
        return
    
    # Filter features and targets to matched subjects
    subj_indices = [ae_subjects.index(s) for s in valid_subjects]
    mean_features = mean_features[subj_indices]
    var_features = var_features[subj_indices]
    targets_matched = targets.iloc[interview_rows].reset_index(drop=True)
    
    logger.info(f"  Final: {len(valid_subjects)} subjects, {mean_features.shape[1]} dims, {len(interview_labels)} interviews")
    
    # Select top dimensions by variance
    dim_variance = mean_features.var(axis=0)
    top_dims = np.argsort(dim_variance)[-args.top_dims:][::-1]
    dim_names = [f'Dim_{d}' for d in top_dims]
    
    logger.info(f"\n[Analyzing Top {args.top_dims} Latent Dimensions]")
    
    # Compute correlations for MEAN features
    logger.info("\n[Mean Latent Features vs Interviews]")
    mean_subset = mean_features[:, top_dims]
    
    corr_mean = np.zeros((len(top_dims), len(interview_labels)))
    pval_mean = np.ones((len(top_dims), len(interview_labels)))
    
    for i, dim in enumerate(top_dims):
        for j in range(len(interview_labels)):
            x = mean_subset[:, i]
            y = targets_matched.iloc[:, j].values
            valid = ~(np.isnan(x) | np.isnan(y))
            if valid.sum() > 2:
                r, p = pearsonr(x[valid], y[valid])
                corr_mean[i, j] = r
                pval_mean[i, j] = p
    
    rejected_mean = plot_correlation_heatmap(
        corr_mean, pval_mean, dim_names, interview_labels,
        'Mean Latent Features vs Subjective Experience (Alpha Band)',
        output_dir / 'heatmap_mean_vs_interviews.png'
    )
    
    # Compute correlations for VARIANCE features
    logger.info("\n[Latent Variance (Metastability) vs Interviews]")
    var_subset = var_features[:, top_dims]
    
    corr_var = np.zeros((len(top_dims), len(interview_labels)))
    pval_var = np.ones((len(top_dims), len(interview_labels)))
    
    for i, dim in enumerate(top_dims):
        for j in range(len(interview_labels)):
            x = var_subset[:, i]
            y = targets_matched.iloc[:, j].values
            valid = ~(np.isnan(x) | np.isnan(y))
            if valid.sum() > 2:
                r, p = pearsonr(x[valid], y[valid])
                corr_var[i, j] = r
                pval_var[i, j] = p
    
    rejected_var = plot_correlation_heatmap(
        corr_var, pval_var, dim_names, interview_labels,
        'Latent Variance (Metastability) vs Subjective Experience (Alpha Band)',
        output_dir / 'heatmap_variance_vs_interviews.png'
    )
    
    # Generate scatter plots for significant correlations
    logger.info("\n[Generating Scatter Plots]")
    
    if rejected_mean.sum() > 0:
        plot_significant_scatters(
            mean_subset, targets_matched, corr_mean, pval_mean, rejected_mean,
            dim_names, interview_labels, output_dir, 'mean'
        )
    
    if rejected_var.sum() > 0:
        plot_significant_scatters(
            var_subset, targets_matched, corr_var, pval_var, rejected_var,
            dim_names, interview_labels, output_dir, 'variance'
        )
    
    # Also try overall latent magnitude and variance
    logger.info("\n[Global Features vs Interviews]")
    
    global_mean = np.linalg.norm(mean_features, axis=1)  # Overall latent magnitude
    global_var = var_features.mean(axis=1)  # Overall variability
    
    global_features = np.column_stack([global_mean, global_var])
    global_names = ['Latent_Magnitude', 'Latent_Variability']
    
    corr_global = np.zeros((2, len(interview_labels)))
    pval_global = np.ones((2, len(interview_labels)))
    
    for i in range(2):
        for j in range(len(interview_labels)):
            x = global_features[:, i]
            y = targets_matched.iloc[:, j].values
            valid = ~(np.isnan(x) | np.isnan(y))
            if valid.sum() > 2:
                r, p = pearsonr(x[valid], y[valid])
                corr_global[i, j] = r
                pval_global[i, j] = p
    
    rejected_global = plot_correlation_heatmap(
        corr_global, pval_global, global_names, interview_labels,
        'Global Latent Features vs Subjective Experience (Alpha Band)',
        output_dir / 'heatmap_global_vs_interviews.png'
    )
    
    if rejected_global.sum() > 0:
        plot_significant_scatters(
            global_features, targets_matched, corr_global, pval_global, rejected_global,
            global_names, interview_labels, output_dir, 'global'
        )
    
    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Subjects analyzed: {len(valid_subjects)}")
    logger.info(f"  Latent dimensions: {args.top_dims}")
    logger.info(f"  Interview variables: {len(interview_labels)}")
    logger.info(f"  Significant (FDR α=0.05):")
    logger.info(f"    - Mean features: {rejected_mean.sum()}")
    logger.info(f"    - Variance features: {rejected_var.sum()}")
    logger.info(f"    - Global features: {rejected_global.sum()}")
    logger.info(f"\n  Results saved to: {output_dir}")
    logger.info("=" * 70)
    
    # Save summary
    summary = {
        'n_subjects': len(valid_subjects),
        'n_dims': args.top_dims,
        'n_interviews': len(interview_labels),
        'significant_mean': int(rejected_mean.sum()),
        'significant_var': int(rejected_var.sum()),
        'significant_global': int(rejected_global.sum()),
        'interview_labels': interview_labels,
        'dim_names': dim_names
    }
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()









