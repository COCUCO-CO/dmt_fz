#!/usr/bin/env python3
"""
Analyze best VAE model and correlate with interview data.

This script:
1. Loads embeddings from the best hyperparameter search model
2. Computes interpretable metrics per subject
3. Correlates with interview/questionnaire data
4. Generates visualizations

Metrics computed:
- Dispersion: How variable are the brain states? (like metastability)
- Centroid distance: How different is this subject from average?
- Trajectory length: Total distance traveled in latent space
- Reconstruction quality (if available)
"""

import argparse
import json
import pickle
from pathlib import Path
import logging

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cdist, pdist
from statsmodels.stats.multitest import fdrcorrection
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def load_embeddings(model_dir: Path):
    """Load all embeddings from model directory."""
    emb_dir = model_dir / 'output' / 'embeddings'
    
    all_z = []
    all_subjects = []
    
    for split in ['train', 'val', 'test']:
        path = emb_dir / f'{split}_embeddings.pkl'
        if path.exists():
            with open(path, 'rb') as f:
                data = pickle.load(f)
            all_z.append(data['latent_embeddings'])
            all_subjects.extend(data['subjects'])
            logger.info(f"Loaded {split}: {len(data['subjects'])} samples")
    
    embeddings = np.vstack(all_z)
    subjects = np.array(all_subjects)
    
    # Clean subject IDs (remove -DMT suffix if present)
    subjects_clean = []
    for s in subjects:
        if isinstance(s, str):
            s_clean = s.replace('-DMT', '').replace('_DMT', '')
            subjects_clean.append(s_clean)
        else:
            subjects_clean.append(str(s))
    
    return embeddings, np.array(subjects_clean)


def compute_subject_metrics(embeddings: np.ndarray, subjects: np.ndarray) -> pd.DataFrame:
    """Compute interpretable metrics for each subject."""
    
    unique_subjects = np.unique(subjects)
    metrics = []
    
    # Global centroid
    global_centroid = embeddings.mean(axis=0)
    
    for subj in unique_subjects:
        mask = subjects == subj
        subj_emb = embeddings[mask]
        n_samples = len(subj_emb)
        
        if n_samples < 2:
            continue
        
        # Subject centroid
        centroid = subj_emb.mean(axis=0)
        
        # 1. Dispersion (intra-subject variance) - like metastability
        # Higher = more variable brain states
        dispersion = np.mean(np.var(subj_emb, axis=0))
        
        # 2. Distance from global centroid - how "different" is this subject
        centroid_dist = np.linalg.norm(centroid - global_centroid)
        
        # 3. Trajectory length - total path traveled in latent space
        # Sort by time (assuming samples are in order)
        trajectory_length = np.sum(np.linalg.norm(np.diff(subj_emb, axis=0), axis=1))
        trajectory_length_normalized = trajectory_length / (n_samples - 1)
        
        # 4. Exploration radius - max distance from subject's own centroid
        distances_from_centroid = np.linalg.norm(subj_emb - centroid, axis=1)
        exploration_radius = np.max(distances_from_centroid)
        mean_radius = np.mean(distances_from_centroid)
        
        # 5. Total variance (sum of variances across all dims)
        total_variance = np.sum(np.var(subj_emb, axis=0))
        
        # 6. Effective dimensionality (how many dims are used)
        variances = np.var(subj_emb, axis=0)
        if variances.sum() > 0:
            normalized_var = variances / variances.sum()
            effective_dims = np.exp(-np.sum(normalized_var * np.log(normalized_var + 1e-10)))
        else:
            effective_dims = 0
        
        metrics.append({
            'subject': subj,
            'n_samples': n_samples,
            'dispersion': dispersion,
            'centroid_distance': centroid_dist,
            'trajectory_length': trajectory_length_normalized,
            'exploration_radius': exploration_radius,
            'mean_radius': mean_radius,
            'total_variance': total_variance,
            'effective_dims': effective_dims
        })
    
    return pd.DataFrame(metrics)


def load_interview_data(interview_path: Path) -> pd.DataFrame:
    """Load interview/questionnaire data."""
    # Try different formats
    if interview_path.suffix == '.csv':
        df = pd.read_csv(interview_path)
    elif interview_path.suffix == '.xlsx':
        df = pd.read_excel(interview_path)
    elif interview_path.suffix == '.pkl':
        df = pd.read_pickle(interview_path)
    else:
        raise ValueError(f"Unknown format: {interview_path.suffix}")
    
    return df


def correlate_metrics_with_interviews(
    metrics_df: pd.DataFrame,
    interview_df: pd.DataFrame,
    output_dir: Path
):
    """Correlate latent space metrics with interview data."""
    
    # Find common subjects
    # Interview subjects might be formatted differently
    interview_subjects = set(interview_df['subject'].astype(str).values)
    metrics_subjects = set(metrics_df['subject'].astype(str).values)
    
    logger.info(f"Metrics subjects: {sorted(metrics_subjects)}")
    logger.info(f"Interview subjects: {sorted(interview_subjects)}")
    
    # Try different matching strategies
    common = metrics_subjects & interview_subjects
    
    if len(common) == 0:
        # Try with S prefix
        metrics_with_s = {f"S{s.replace('S', '')}" for s in metrics_subjects}
        common = metrics_with_s & interview_subjects
        if len(common) > 0:
            # Update metrics_df subjects
            metrics_df['subject'] = metrics_df['subject'].apply(lambda x: f"S{str(x).replace('S', '')}")
    
    logger.info(f"Common subjects: {len(common)}")
    
    if len(common) < 5:
        logger.warning("Too few common subjects for meaningful correlation!")
        return None
    
    # Merge dataframes
    merged = pd.merge(
        metrics_df,
        interview_df,
        on='subject',
        how='inner'
    )
    
    logger.info(f"Merged data: {len(merged)} subjects")
    
    # Get metric and interview columns
    metric_cols = ['dispersion', 'centroid_distance', 'trajectory_length', 
                   'exploration_radius', 'mean_radius', 'total_variance', 'effective_dims']
    
    # Interview columns (numeric only)
    interview_cols = [col for col in interview_df.columns 
                     if col != 'subject' and interview_df[col].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    logger.info(f"Metric columns: {metric_cols}")
    logger.info(f"Interview columns: {interview_cols[:10]}...")  # Show first 10
    
    # Compute correlations
    results = []
    
    for metric_col in metric_cols:
        for interview_col in interview_cols:
            try:
                x = merged[metric_col].values
                y = merged[interview_col].values
                
                # Remove NaN
                valid = ~(np.isnan(x) | np.isnan(y))
                if valid.sum() < 5:
                    continue
                
                x_valid = x[valid]
                y_valid = y[valid]
                
                # Pearson correlation
                r, p = pearsonr(x_valid, y_valid)
                
                # Spearman correlation
                rho, p_spearman = spearmanr(x_valid, y_valid)
                
                results.append({
                    'metric': metric_col,
                    'interview': interview_col,
                    'pearson_r': r,
                    'pearson_p': p,
                    'spearman_rho': rho,
                    'spearman_p': p_spearman,
                    'n_subjects': valid.sum()
                })
            except Exception as e:
                continue
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        logger.warning("No valid correlations computed!")
        return None
    
    # FDR correction
    _, fdr_pearson = fdrcorrection(results_df['pearson_p'].values)
    _, fdr_spearman = fdrcorrection(results_df['spearman_p'].values)
    results_df['pearson_fdr'] = fdr_pearson
    results_df['spearman_fdr'] = fdr_spearman
    
    # Sort by significance
    results_df = results_df.sort_values('pearson_p')
    
    # Save results
    results_df.to_csv(output_dir / 'correlation_results.csv', index=False)
    
    # Report significant correlations
    sig_pearson = results_df[results_df['pearson_fdr'] < 0.05]
    sig_spearman = results_df[results_df['spearman_fdr'] < 0.05]
    
    logger.info(f"\n{'='*60}")
    logger.info("SIGNIFICANT CORRELATIONS (FDR < 0.05)")
    logger.info('='*60)
    
    if len(sig_pearson) > 0:
        logger.info("\nPearson correlations:")
        for _, row in sig_pearson.head(20).iterrows():
            logger.info(f"  {row['metric']} vs {row['interview']}: r={row['pearson_r']:.3f}, p={row['pearson_p']:.4f}, FDR={row['pearson_fdr']:.4f}")
    else:
        logger.info("\nNo significant Pearson correlations after FDR correction")
    
    if len(sig_spearman) > 0:
        logger.info("\nSpearman correlations:")
        for _, row in sig_spearman.head(20).iterrows():
            logger.info(f"  {row['metric']} vs {row['interview']}: rho={row['spearman_rho']:.3f}, p={row['spearman_p']:.4f}, FDR={row['spearman_fdr']:.4f}")
    
    # Create visualization
    create_correlation_heatmap(results_df, metric_cols, interview_cols[:15], output_dir)
    
    return results_df, merged


def create_correlation_heatmap(results_df, metric_cols, interview_cols, output_dir):
    """Create correlation heatmap."""
    
    # Pivot to matrix
    pivot_r = results_df.pivot(index='metric', columns='interview', values='pearson_r')
    pivot_p = results_df.pivot(index='metric', columns='interview', values='pearson_fdr')
    
    # Filter to available columns
    available_cols = [c for c in interview_cols if c in pivot_r.columns]
    if len(available_cols) == 0:
        logger.warning("No interview columns available for heatmap")
        return
    
    pivot_r = pivot_r[available_cols]
    pivot_p = pivot_p[available_cols]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Heatmap
    sns.heatmap(
        pivot_r,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        cbar_kws={'label': 'Pearson r'}
    )
    
    # Add significance markers
    for i, metric in enumerate(pivot_r.index):
        for j, interview in enumerate(pivot_r.columns):
            if metric in pivot_p.index and interview in pivot_p.columns:
                p_val = pivot_p.loc[metric, interview]
                if p_val < 0.05:
                    ax.text(j + 0.5, i + 0.8, '*', ha='center', va='center', 
                           fontsize=14, color='black', fontweight='bold')
    
    ax.set_title('Latent Space Metrics vs Interview Responses\n(* = FDR < 0.05)', fontsize=14)
    ax.set_xlabel('Interview Questions', fontsize=12)
    ax.set_ylabel('Latent Space Metrics', fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved heatmap to {output_dir / 'correlation_heatmap.png'}")


def visualize_latent_space(embeddings, subjects, output_dir):
    """Visualize latent space with TSNE/PCA."""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Get unique subjects and colors
    unique_subjects = np.unique(subjects)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_subjects)))
    subject_to_color = {s: c for s, c in zip(unique_subjects, colors)}
    
    # PCA
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(embeddings)
    
    for subj in unique_subjects:
        mask = subjects == subj
        axes[0].scatter(z_pca[mask, 0], z_pca[mask, 1], 
                       c=[subject_to_color[subj]], label=subj, alpha=0.5, s=10)
    
    axes[0].set_title(f'PCA (var explained: {pca.explained_variance_ratio_.sum():.2%})')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)//5))
    z_tsne = tsne.fit_transform(embeddings)
    
    for subj in unique_subjects:
        mask = subjects == subj
        axes[1].scatter(z_tsne[mask, 0], z_tsne[mask, 1],
                       c=[subject_to_color[subj]], label=subj, alpha=0.5, s=10)
    
    axes[1].set_title('t-SNE')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    
    # Legend
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5), ncol=1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'latent_space_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved latent space viz to {output_dir / 'latent_space_visualization.png'}")


def main():
    parser = argparse.ArgumentParser(description='Analyze best VAE model')
    parser.add_argument('--model-dir', type=str, required=True,
                       help='Path to best model directory')
    parser.add_argument('--interview-data', type=str, 
                       default='/media/storage_hdd/dmt_fz/data/interviews/interview_scores.csv',
                       help='Path to interview data')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory')
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = model_dir / 'analysis'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("Best VAE Model Analysis")
    logger.info("="*60)
    logger.info(f"Model: {model_dir}")
    logger.info(f"Output: {output_dir}")
    
    # Load embeddings
    logger.info("\nLoading embeddings...")
    embeddings, subjects = load_embeddings(model_dir)
    logger.info(f"Total samples: {len(embeddings)}")
    logger.info(f"Unique subjects: {len(np.unique(subjects))}")
    logger.info(f"Latent dim: {embeddings.shape[1]}")
    
    # Compute metrics
    logger.info("\nComputing subject metrics...")
    metrics_df = compute_subject_metrics(embeddings, subjects)
    metrics_df.to_csv(output_dir / 'subject_metrics.csv', index=False)
    logger.info(f"Computed metrics for {len(metrics_df)} subjects")
    
    # Print metrics summary
    logger.info("\nMetrics summary:")
    for col in ['dispersion', 'centroid_distance', 'trajectory_length', 'total_variance']:
        logger.info(f"  {col}: mean={metrics_df[col].mean():.4f}, std={metrics_df[col].std():.4f}")
    
    # Visualize latent space
    logger.info("\nVisualizing latent space...")
    visualize_latent_space(embeddings, subjects, output_dir)
    
    # Load interview data and correlate
    interview_path = Path(args.interview_data)
    if interview_path.exists():
        logger.info(f"\nLoading interview data from {interview_path}")
        interview_df = load_interview_data(interview_path)
        logger.info(f"Interview data: {len(interview_df)} subjects, {len(interview_df.columns)} columns")
        
        results = correlate_metrics_with_interviews(metrics_df, interview_df, output_dir)
        
        if results is not None:
            results_df, merged = results
            logger.info(f"\nTotal correlations tested: {len(results_df)}")
    else:
        logger.warning(f"Interview data not found at {interview_path}")
        logger.info("Skipping correlation analysis")
        
        # Try to find interview data
        possible_paths = [
            Path('/media/storage_hdd/dmt_fz/data/interviews'),
            Path('/media/storage_hdd/dmt_fz/interviews'),
            Path('/media/storage_hdd/dmt_fz/questionnaires'),
        ]
        
        for p in possible_paths:
            if p.exists():
                logger.info(f"Found possible interview data at: {p}")
                logger.info(f"  Contents: {list(p.iterdir())[:5]}")
    
    logger.info("\n" + "="*60)
    logger.info("Analysis complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*60)


if __name__ == '__main__':
    main()

