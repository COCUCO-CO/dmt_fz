#!/usr/bin/env python3
"""
validate_clustering.py - Statistical validation of attention state clusters

This script performs rigorous statistical validation to ensure clusters are not random:

1. Permutation Test: Compare observed silhouette vs null distribution
2. Gap Statistic: Compare clustering quality against uniform random data
3. Cross-Subject Validation: Check if clusters generalize across subjects
4. Bootstrap Stability: Measure cluster consistency across resamples
5. Condition Discrimination: Test if clusters distinguish DMT vs EC

Usage:
    python validate_clustering.py --input-dir path/to/attention_states
    python validate_clustering.py --input-dir attention_states --n-permutations 1000
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging

import numpy as np
from tqdm import tqdm
from scipy import stats

# Use non-GUI backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def load_attention_data(input_dir: Path, layer: str = 'layer0'):
    """Load all attention matrices with subject/condition info."""
    all_matrices = []
    all_metadata = []
    
    exclude_dirs = {'clustering', 'plots', 'analysis', 'visualizations'}
    subject_dirs = sorted([d for d in input_dir.iterdir() 
                          if d.is_dir() and d.name not in exclude_dirs])
    
    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        
        for cond_dir in subject_dir.iterdir():
            if not cond_dir.is_dir():
                continue
            
            condition = cond_dir.name
            
            # New format
            layer_dir = cond_dir / layer
            if layer_dir.exists():
                att_file = layer_dir / 'tensors' / 'attention_mean.npy'
            else:
                att_file = cond_dir / 'tensors' / f'attention_{layer}_mean.npy'
            
            if not att_file.exists():
                continue
            
            matrices = np.load(att_file)
            
            for i in range(matrices.shape[0]):
                all_matrices.append(matrices[i])
                all_metadata.append({
                    'subject_id': subject_id,
                    'condition': condition,
                    'epoch_idx': i
                })
    
    return np.array(all_matrices), all_metadata


def vectorize_matrices(matrices: np.ndarray, use_upper_triangle: bool = True) -> np.ndarray:
    """Convert NxN matrices to feature vectors."""
    n_samples = matrices.shape[0]
    n_nodes = matrices.shape[1]
    
    if use_upper_triangle:
        indices = np.triu_indices(n_nodes, k=1)
        return np.array([m[indices] for m in matrices])
    else:
        return matrices.reshape(n_samples, -1)


def compute_gap_statistic(X: np.ndarray, k_range: range, n_references: int = 10, 
                          n_components: int = 10) -> Dict:
    """
    Compute Gap Statistic to find optimal k and validate clustering.
    
    Gap(k) = E[log(W_k*)] - log(W_k)
    
    Where W_k* is from reference (uniform random) data.
    """
    # PCA first
    pca = PCA(n_components=min(n_components, X.shape[1], X.shape[0] - 1))
    X_pca = pca.fit_transform(X)
    
    # Get data bounds for generating uniform reference
    mins = X_pca.min(axis=0)
    maxs = X_pca.max(axis=0)
    
    results = {
        'k_values': list(k_range),
        'log_W': [],
        'log_W_ref_mean': [],
        'log_W_ref_std': [],
        'gap': [],
        'gap_se': []
    }
    
    for k in tqdm(k_range, desc="Computing Gap Statistic"):
        # Fit on real data
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
        labels = kmeans.fit_predict(X_pca)
        
        # Compute W_k (within-cluster sum of squares)
        W_k = 0
        for cluster_id in range(k):
            cluster_points = X_pca[labels == cluster_id]
            if len(cluster_points) > 0:
                center = cluster_points.mean(axis=0)
                W_k += ((cluster_points - center) ** 2).sum()
        
        results['log_W'].append(np.log(W_k + 1e-10))
        
        # Generate reference datasets and compute W_k*
        log_W_refs = []
        for _ in range(n_references):
            # Uniform random in bounding box
            X_ref = np.random.uniform(mins, maxs, size=X_pca.shape)
            
            kmeans_ref = KMeans(n_clusters=k, n_init='auto', random_state=None)
            labels_ref = kmeans_ref.fit_predict(X_ref)
            
            W_k_ref = 0
            for cluster_id in range(k):
                cluster_points = X_ref[labels_ref == cluster_id]
                if len(cluster_points) > 0:
                    center = cluster_points.mean(axis=0)
                    W_k_ref += ((cluster_points - center) ** 2).sum()
            
            log_W_refs.append(np.log(W_k_ref + 1e-10))
        
        results['log_W_ref_mean'].append(np.mean(log_W_refs))
        results['log_W_ref_std'].append(np.std(log_W_refs))
        results['gap'].append(np.mean(log_W_refs) - np.log(W_k + 1e-10))
        results['gap_se'].append(np.std(log_W_refs) * np.sqrt(1 + 1/n_references))
    
    # Find optimal k using gap criterion
    # k* = smallest k such that Gap(k) >= Gap(k+1) - se(k+1)
    optimal_k = results['k_values'][0]
    for i in range(len(results['k_values']) - 1):
        if results['gap'][i] >= results['gap'][i+1] - results['gap_se'][i+1]:
            optimal_k = results['k_values'][i]
            break
    
    results['optimal_k'] = optimal_k
    
    return results


def permutation_test(X: np.ndarray, n_clusters: int, n_permutations: int = 1000,
                     n_components: int = 10) -> Dict:
    """
    Permutation test for clustering significance.
    
    H0: The observed clustering structure is due to chance
    H1: The clustering structure is statistically significant
    
    We permute the data rows and re-cluster, comparing silhouette scores.
    """
    # PCA
    pca = PCA(n_components=min(n_components, X.shape[1], X.shape[0] - 1))
    X_pca = pca.fit_transform(X)
    
    # Observed clustering
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
    labels_observed = kmeans.fit_predict(X_pca)
    silhouette_observed = silhouette_score(X_pca, labels_observed)
    
    # Permutation distribution
    silhouette_null = []
    
    for _ in tqdm(range(n_permutations), desc="Permutation test"):
        # Permute each feature independently (breaks structure)
        X_perm = X_pca.copy()
        for col in range(X_perm.shape[1]):
            np.random.shuffle(X_perm[:, col])
        
        kmeans_perm = KMeans(n_clusters=n_clusters, n_init='auto', random_state=None)
        labels_perm = kmeans_perm.fit_predict(X_perm)
        
        try:
            sil = silhouette_score(X_perm, labels_perm)
            silhouette_null.append(sil)
        except:
            pass
    
    silhouette_null = np.array(silhouette_null)
    
    # P-value: proportion of null >= observed
    p_value = (silhouette_null >= silhouette_observed).mean()
    
    # Effect size (Cohen's d)
    effect_size = (silhouette_observed - silhouette_null.mean()) / (silhouette_null.std() + 1e-10)
    
    return {
        'silhouette_observed': silhouette_observed,
        'silhouette_null_mean': silhouette_null.mean(),
        'silhouette_null_std': silhouette_null.std(),
        'silhouette_null': silhouette_null,
        'p_value': p_value,
        'effect_size': effect_size,
        'n_permutations': n_permutations
    }


def cross_subject_validation(X: np.ndarray, metadata: List[Dict], n_clusters: int,
                             n_components: int = 10) -> Dict:
    """
    Leave-one-subject-out cross-validation for clustering.
    
    Tests if clusters generalize across subjects.
    """
    # Extract subject labels
    subjects = np.array([m['subject_id'] for m in metadata])
    unique_subjects = np.unique(subjects)
    
    if len(unique_subjects) < 2:
        logger.warning("Need at least 2 subjects for cross-validation")
        return {'error': 'Insufficient subjects'}
    
    # Vectorize and standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=min(n_components, X.shape[1], X.shape[0] - 1))
    X_pca = pca.fit_transform(X_scaled)
    
    results = {
        'subjects': [],
        'silhouette_train': [],
        'silhouette_test': [],
        'ari_scores': [],  # Adjusted Rand Index
        'nmi_scores': []   # Normalized Mutual Information
    }
    
    logo = LeaveOneGroupOut()
    
    for train_idx, test_idx in tqdm(logo.split(X_pca, groups=subjects), 
                                     total=len(unique_subjects),
                                     desc="Cross-subject validation"):
        X_train, X_test = X_pca[train_idx], X_pca[test_idx]
        test_subject = subjects[test_idx][0]
        
        # Fit on training subjects
        kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
        labels_train = kmeans.fit_predict(X_train)
        
        # Predict on test subject
        labels_test = kmeans.predict(X_test)
        
        # Metrics
        sil_train = silhouette_score(X_train, labels_train) if len(set(labels_train)) > 1 else 0
        sil_test = silhouette_score(X_test, labels_test) if len(set(labels_test)) > 1 else 0
        
        results['subjects'].append(test_subject)
        results['silhouette_train'].append(sil_train)
        results['silhouette_test'].append(sil_test)
    
    # Summary statistics
    results['mean_silhouette_train'] = np.mean(results['silhouette_train'])
    results['mean_silhouette_test'] = np.mean(results['silhouette_test'])
    results['std_silhouette_test'] = np.std(results['silhouette_test'])
    
    # Generalization ratio (test/train)
    results['generalization_ratio'] = results['mean_silhouette_test'] / (results['mean_silhouette_train'] + 1e-10)
    
    return results


def bootstrap_stability(X: np.ndarray, n_clusters: int, n_bootstrap: int = 100,
                        n_components: int = 10, sample_fraction: float = 0.8) -> Dict:
    """
    Bootstrap stability analysis for cluster assignments.
    
    Measures how stable cluster assignments are across bootstrap samples.
    """
    n_samples = X.shape[0]
    sample_size = int(n_samples * sample_fraction)
    
    # Standardize and reduce
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=min(n_components, X.shape[1], X.shape[0] - 1))
    X_pca = pca.fit_transform(X_scaled)
    
    # Reference clustering on full data
    kmeans_ref = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
    labels_ref = kmeans_ref.fit_predict(X_pca)
    
    # Bootstrap
    ari_scores = []
    nmi_scores = []
    silhouette_scores = []
    
    for _ in tqdm(range(n_bootstrap), desc="Bootstrap stability"):
        # Sample with replacement
        idx = np.random.choice(n_samples, size=sample_size, replace=True)
        X_boot = X_pca[idx]
        
        # Cluster
        kmeans_boot = KMeans(n_clusters=n_clusters, n_init='auto', random_state=None)
        labels_boot = kmeans_boot.fit_predict(X_boot)
        
        # Assign labels to all samples using bootstrap model
        labels_full = kmeans_boot.predict(X_pca)
        
        # Compare to reference
        ari = adjusted_rand_score(labels_ref, labels_full)
        nmi = normalized_mutual_info_score(labels_ref, labels_full)
        
        ari_scores.append(ari)
        nmi_scores.append(nmi)
        
        if len(set(labels_boot)) > 1:
            silhouette_scores.append(silhouette_score(X_boot, labels_boot))
    
    return {
        'ari_mean': np.mean(ari_scores),
        'ari_std': np.std(ari_scores),
        'ari_scores': ari_scores,
        'nmi_mean': np.mean(nmi_scores),
        'nmi_std': np.std(nmi_scores),
        'nmi_scores': nmi_scores,
        'silhouette_mean': np.mean(silhouette_scores),
        'silhouette_std': np.std(silhouette_scores),
        'n_bootstrap': n_bootstrap
    }


def condition_discrimination_test(X: np.ndarray, metadata: List[Dict], n_clusters: int,
                                   n_components: int = 10, n_permutations: int = 1000) -> Dict:
    """
    Test if cluster assignments discriminate between conditions (DMT vs EC).
    
    Uses chi-square test with permutation-based p-value.
    """
    conditions = np.array([m['condition'] for m in metadata])
    unique_conditions = np.unique(conditions)
    
    if len(unique_conditions) < 2:
        return {'error': 'Need at least 2 conditions'}
    
    # Cluster
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=min(n_components, X.shape[1], X.shape[0] - 1))
    X_pca = pca.fit_transform(X_scaled)
    
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
    labels = kmeans.fit_predict(X_pca)
    
    # Observed contingency table
    contingency = np.zeros((len(unique_conditions), n_clusters))
    for i, cond in enumerate(unique_conditions):
        mask = conditions == cond
        for k in range(n_clusters):
            contingency[i, k] = (labels[mask] == k).sum()
    
    # Chi-square test
    chi2_observed, p_chi2, dof, expected = stats.chi2_contingency(contingency)
    
    # Permutation test for chi-square
    chi2_null = []
    for _ in tqdm(range(n_permutations), desc="Condition discrimination test"):
        # Permute condition labels
        conditions_perm = np.random.permutation(conditions)
        
        contingency_perm = np.zeros((len(unique_conditions), n_clusters))
        for i, cond in enumerate(unique_conditions):
            mask = conditions_perm == cond
            for k in range(n_clusters):
                contingency_perm[i, k] = (labels[mask] == k).sum()
        
        chi2_perm, _, _, _ = stats.chi2_contingency(contingency_perm)
        chi2_null.append(chi2_perm)
    
    chi2_null = np.array(chi2_null)
    p_value_perm = (chi2_null >= chi2_observed).mean()
    
    # Effect size (Cramér's V)
    n = contingency.sum()
    min_dim = min(contingency.shape) - 1
    cramers_v = np.sqrt(chi2_observed / (n * min_dim)) if min_dim > 0 else 0
    
    return {
        'chi2_observed': chi2_observed,
        'chi2_null_mean': chi2_null.mean(),
        'chi2_null_std': chi2_null.std(),
        'p_value_chi2': p_chi2,
        'p_value_permutation': p_value_perm,
        'cramers_v': cramers_v,
        'contingency_table': contingency,
        'conditions': list(unique_conditions),
        'dof': dof
    }


def plot_validation_results(results: Dict, output_dir: Path):
    """Generate plots for validation results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Permutation test
    ax = axes[0, 0]
    if 'permutation' in results and 'silhouette_null' in results['permutation']:
        perm = results['permutation']
        ax.hist(perm['silhouette_null'], bins=50, alpha=0.7, color='gray', 
                label='Null distribution')
        ax.axvline(perm['silhouette_observed'], color='red', linewidth=2,
                   label=f'Observed: {perm["silhouette_observed"]:.3f}')
        ax.axvline(np.percentile(perm['silhouette_null'], 95), color='orange', 
                   linestyle='--', label='95th percentile')
        ax.set_xlabel('Silhouette Score')
        ax.set_ylabel('Count')
        ax.set_title(f'Permutation Test (p={perm["p_value"]:.4f})')
        ax.legend()
    
    # 2. Gap Statistic
    ax = axes[0, 1]
    if 'gap_statistic' in results:
        gap = results['gap_statistic']
        k_vals = gap['k_values']
        ax.errorbar(k_vals, gap['gap'], yerr=gap['gap_se'], 
                   marker='o', capsize=5, label='Gap(k)')
        ax.axvline(gap['optimal_k'], color='red', linestyle='--', 
                   label=f'Optimal k={gap["optimal_k"]}')
        ax.set_xlabel('Number of clusters (k)')
        ax.set_ylabel('Gap Statistic')
        ax.set_title('Gap Statistic Analysis')
        ax.legend()
    
    # 3. Cross-subject validation
    ax = axes[1, 0]
    if 'cross_subject' in results and 'subjects' in results['cross_subject']:
        cv = results['cross_subject']
        x = range(len(cv['subjects']))
        ax.bar(x, cv['silhouette_test'], alpha=0.7, label='Test (left-out subject)')
        ax.axhline(cv['mean_silhouette_test'], color='red', linestyle='--',
                   label=f'Mean test: {cv["mean_silhouette_test"]:.3f}')
        ax.axhline(cv['mean_silhouette_train'], color='green', linestyle='--',
                   label=f'Mean train: {cv["mean_silhouette_train"]:.3f}')
        ax.set_xticks(x)
        ax.set_xticklabels(cv['subjects'], rotation=45, ha='right')
        ax.set_xlabel('Left-out Subject')
        ax.set_ylabel('Silhouette Score')
        ax.set_title(f'Cross-Subject Validation (Gen. ratio: {cv["generalization_ratio"]:.2f})')
        ax.legend()
    
    # 4. Bootstrap stability
    ax = axes[1, 1]
    if 'bootstrap' in results:
        boot = results['bootstrap']
        metrics = ['ARI', 'NMI', 'Silhouette']
        means = [boot['ari_mean'], boot['nmi_mean'], boot['silhouette_mean']]
        stds = [boot['ari_std'], boot['nmi_std'], boot['silhouette_std']]
        
        bars = ax.bar(metrics, means, yerr=stds, capsize=5, alpha=0.7,
                     color=['#3498db', '#2ecc71', '#e74c3c'])
        ax.set_ylabel('Score')
        ax.set_title(f'Bootstrap Stability (n={boot["n_bootstrap"]})')
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                   f'{mean:.3f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'validation_summary.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # Condition discrimination plot
    if 'condition_discrimination' in results:
        cond = results['condition_discrimination']
        if 'contingency_table' in cond:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            contingency_norm = cond['contingency_table'] / cond['contingency_table'].sum(axis=1, keepdims=True)
            
            sns.heatmap(contingency_norm, annot=True, fmt='.2%', cmap='YlOrRd',
                       xticklabels=[f'S{i}' for i in range(contingency_norm.shape[1])],
                       yticklabels=cond['conditions'], ax=ax)
            ax.set_xlabel('Cluster')
            ax.set_ylabel('Condition')
            ax.set_title(f'Condition × Cluster Distribution\n'
                        f'χ²={cond["chi2_observed"]:.1f}, p={cond["p_value_permutation"]:.4f}, '
                        f"Cramér's V={cond['cramers_v']:.3f}")
            
            plt.tight_layout()
            plt.savefig(output_dir / 'condition_discrimination.png', dpi=200, bbox_inches='tight')
            plt.close()
    
    logger.info(f"Validation plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Statistical validation of attention state clusters'
    )
    
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Directory with extracted attention states')
    parser.add_argument('--layer', type=str, default='layer0',
                       help='Which layer to use (default: layer0)')
    parser.add_argument('--n-clusters', type=int, default=None,
                       help='Number of clusters (default: from clustering results)')
    parser.add_argument('--n-permutations', type=int, default=1000,
                       help='Number of permutations for significance tests')
    parser.add_argument('--n-bootstrap', type=int, default=100,
                       help='Number of bootstrap samples')
    parser.add_argument('--n-components', type=int, default=10,
                       help='PCA components')
    parser.add_argument('--skip-gap', action='store_true',
                       help='Skip Gap Statistic (slow)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = input_dir / 'clustering' / 'validation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading attention data...")
    matrices, metadata = load_attention_data(input_dir, layer=args.layer)
    
    if len(matrices) == 0:
        logger.error("No attention data found!")
        sys.exit(1)
    
    # Vectorize
    X = vectorize_matrices(matrices)
    logger.info(f"Loaded {len(X)} samples, {X.shape[1]} features")
    
    # Get n_clusters from existing results or optimize
    clustering_results_path = input_dir / 'clustering' / 'clustering_results.pkl'
    if args.n_clusters:
        n_clusters = args.n_clusters
    elif clustering_results_path.exists():
        with open(clustering_results_path, 'rb') as f:
            existing = pickle.load(f)
        n_clusters = existing['n_clusters']
        logger.info(f"Using n_clusters={n_clusters} from existing results")
    else:
        n_clusters = 5  # Default
    
    results = {
        'n_samples': len(X),
        'n_features': X.shape[1],
        'n_clusters': n_clusters
    }
    
    # 1. Permutation Test
    logger.info("\n" + "="*60)
    logger.info("1. PERMUTATION TEST")
    logger.info("="*60)
    results['permutation'] = permutation_test(
        X, n_clusters, n_permutations=args.n_permutations,
        n_components=args.n_components
    )
    perm = results['permutation']
    logger.info(f"   Observed Silhouette: {perm['silhouette_observed']:.4f}")
    logger.info(f"   Null mean ± std: {perm['silhouette_null_mean']:.4f} ± {perm['silhouette_null_std']:.4f}")
    logger.info(f"   P-value: {perm['p_value']:.4f}")
    logger.info(f"   Effect size (Cohen's d): {perm['effect_size']:.2f}")
    
    # 2. Gap Statistic
    if not args.skip_gap:
        logger.info("\n" + "="*60)
        logger.info("2. GAP STATISTIC")
        logger.info("="*60)
        k_range = range(2, min(15, len(X) // 10))
        results['gap_statistic'] = compute_gap_statistic(
            X, k_range, n_references=10, n_components=args.n_components
        )
        gap = results['gap_statistic']
        logger.info(f"   Optimal k (Gap criterion): {gap['optimal_k']}")
    
    # 3. Cross-Subject Validation
    logger.info("\n" + "="*60)
    logger.info("3. CROSS-SUBJECT VALIDATION")
    logger.info("="*60)
    results['cross_subject'] = cross_subject_validation(
        X, metadata, n_clusters, n_components=args.n_components
    )
    cv = results['cross_subject']
    if 'error' not in cv:
        logger.info(f"   Mean train silhouette: {cv['mean_silhouette_train']:.4f}")
        logger.info(f"   Mean test silhouette: {cv['mean_silhouette_test']:.4f} ± {cv['std_silhouette_test']:.4f}")
        logger.info(f"   Generalization ratio: {cv['generalization_ratio']:.2f}")
    
    # 4. Bootstrap Stability
    logger.info("\n" + "="*60)
    logger.info("4. BOOTSTRAP STABILITY")
    logger.info("="*60)
    results['bootstrap'] = bootstrap_stability(
        X, n_clusters, n_bootstrap=args.n_bootstrap,
        n_components=args.n_components
    )
    boot = results['bootstrap']
    logger.info(f"   ARI: {boot['ari_mean']:.4f} ± {boot['ari_std']:.4f}")
    logger.info(f"   NMI: {boot['nmi_mean']:.4f} ± {boot['nmi_std']:.4f}")
    logger.info(f"   Silhouette: {boot['silhouette_mean']:.4f} ± {boot['silhouette_std']:.4f}")
    
    # 5. Condition Discrimination
    logger.info("\n" + "="*60)
    logger.info("5. CONDITION DISCRIMINATION TEST")
    logger.info("="*60)
    results['condition_discrimination'] = condition_discrimination_test(
        X, metadata, n_clusters, n_components=args.n_components,
        n_permutations=args.n_permutations
    )
    disc = results['condition_discrimination']
    if 'error' not in disc:
        logger.info(f"   Chi-square: {disc['chi2_observed']:.2f}")
        logger.info(f"   P-value (permutation): {disc['p_value_permutation']:.4f}")
        logger.info(f"   Cramér's V: {disc['cramers_v']:.4f}")
    
    # Generate plots
    logger.info("\n" + "="*60)
    logger.info("GENERATING PLOTS")
    logger.info("="*60)
    plot_validation_results(results, output_dir)
    
    # Save results
    with open(output_dir / 'validation_results.pkl', 'wb') as f:
        # Remove large arrays for storage
        results_save = {k: v for k, v in results.items()}
        if 'permutation' in results_save:
            results_save['permutation'] = {k: v for k, v in results_save['permutation'].items() 
                                          if k != 'silhouette_null'}
        pickle.dump(results_save, f)
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"\n  Samples: {results['n_samples']}")
    print(f"  Clusters: {results['n_clusters']}")
    
    print(f"\n  PERMUTATION TEST:")
    print(f"    Silhouette: {perm['silhouette_observed']:.4f} (null: {perm['silhouette_null_mean']:.4f})")
    print(f"    P-value: {perm['p_value']:.4f} {'✓ SIGNIFICANT' if perm['p_value'] < 0.05 else '✗ Not significant'}")
    print(f"    Effect size: {perm['effect_size']:.2f} {'(large)' if abs(perm['effect_size']) > 0.8 else '(medium)' if abs(perm['effect_size']) > 0.5 else '(small)'}")
    
    if 'cross_subject' in results and 'error' not in results['cross_subject']:
        print(f"\n  CROSS-SUBJECT VALIDATION:")
        print(f"    Generalization: {cv['generalization_ratio']:.2f} {'✓ Good' if cv['generalization_ratio'] > 0.7 else '✗ Poor'}")
    
    print(f"\n  BOOTSTRAP STABILITY:")
    print(f"    ARI: {boot['ari_mean']:.3f} {'✓ Stable' if boot['ari_mean'] > 0.7 else '~ Moderate' if boot['ari_mean'] > 0.5 else '✗ Unstable'}")
    
    if 'condition_discrimination' in results and 'error' not in results['condition_discrimination']:
        print(f"\n  CONDITION DISCRIMINATION:")
        print(f"    P-value: {disc['p_value_permutation']:.4f} {'✓ Clusters differ by condition' if disc['p_value_permutation'] < 0.05 else '✗ No significant difference'}")
        print(f"    Cramér's V: {disc['cramers_v']:.3f}")
    
    print(f"\n  Results saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()













