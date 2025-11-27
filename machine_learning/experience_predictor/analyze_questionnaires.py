#!/usr/bin/env python3
"""
Statistical Analysis of Subjective Experience Questionnaires

Analyzes the 23 subjective experience variables from DMT study:
- ASC (Altered States of Consciousness): 11 subscales
- NDE (Near-Death Experience): 4 subscales  
- MEQ (Mystical Experience Questionnaire): 5 subscales
- Post (Post-experience): 3 variables

Generates comprehensive statistical reports and visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
import warnings
warnings.filterwarnings('ignore')

# Configure plotting
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Target variable names (from actual data)
TARGET_NAMES = [
    "ASC Unity", "ASC Spiritual", "ASC Blissful",
    "ASC Insightfulness", "ASC Disembodiment", "ASC Impaired",
    "ASC Anxiety", "ASC Complex imagery", "ASC Elementary imagery",
    "ASC Audiovisual", "ASC Changed",
    "NDE Cognition", "NDE Affect", "NDE Paranormal", "NDE Transcendental",
    "MEQ Mystical", "MEQ Positive", "MEQ Transcendental",
    "MEQ Inefability", "MEQ Awe",
    "Post Social", "Post Fusion", "Post Setting"
]

CATEGORIES = {
    'ASC': list(range(11)),
    'NDE': list(range(11, 15)),
    'MEQ': list(range(15, 20)),
    'Post': list(range(20, 23))
}

CATEGORY_COLORS = {
    'ASC': '#3498db',
    'NDE': '#e74c3c', 
    'MEQ': '#2ecc71',
    'Post': '#9b59b6'
}


def load_targets(targets_path: str) -> pd.DataFrame:
    """Load target variables from CSV."""
    df = pd.read_csv(targets_path, index_col=0)
    
    # Check columns
    if 'subject' in df.columns or 'Subject' in df.columns:
        subject_col = 'subject' if 'subject' in df.columns else 'Subject'
        df = df.set_index(subject_col)
    
    # Drop any unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Create subject index
    df.index = [f'S{i+1:02d}' for i in range(len(df))]
    df.index.name = 'Subject'
    
    return df


def descriptive_statistics(df: pd.DataFrame, output_dir: Path):
    """Generate comprehensive descriptive statistics."""
    print("\n" + "="*70)
    print("DESCRIPTIVE STATISTICS")
    print("="*70)
    
    stats_df = pd.DataFrame({
        'Mean': df.mean(),
        'Std': df.std(),
        'Min': df.min(),
        '25%': df.quantile(0.25),
        'Median': df.median(),
        '75%': df.quantile(0.75),
        'Max': df.max(),
        'Skewness': df.skew(),
        'Kurtosis': df.kurtosis(),
        'N_valid': df.count()
    })
    
    print("\nPer-variable statistics:")
    print(stats_df.round(3).to_string())
    
    # Save to CSV
    stats_df.to_csv(output_dir / 'descriptive_statistics.csv')
    print(f"\nSaved to {output_dir / 'descriptive_statistics.csv'}")
    
    # Summary by category
    print("\n" + "-"*50)
    print("SUMMARY BY CATEGORY:")
    print("-"*50)
    
    for cat, indices in CATEGORIES.items():
        cat_cols = [TARGET_NAMES[i] for i in indices]
        cat_data = df[cat_cols]
        print(f"\n{cat}:")
        print(f"  Overall Mean: {cat_data.values.mean():.3f}")
        print(f"  Overall Std:  {cat_data.values.std():.3f}")
        print(f"  Range:        [{cat_data.values.min():.3f}, {cat_data.values.max():.3f}]")
    
    return stats_df


def normality_tests(df: pd.DataFrame, output_dir: Path):
    """Test normality of each variable."""
    print("\n" + "="*70)
    print("NORMALITY TESTS (Shapiro-Wilk)")
    print("="*70)
    
    results = []
    for col in df.columns:
        stat, p = stats.shapiro(df[col].dropna())
        results.append({
            'Variable': col,
            'W-statistic': stat,
            'p-value': p,
            'Normal (α=0.05)': 'Yes' if p > 0.05 else 'No'
        })
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    n_normal = (results_df['Normal (α=0.05)'] == 'Yes').sum()
    print(f"\n{n_normal}/{len(results_df)} variables are normally distributed (α=0.05)")
    
    results_df.to_csv(output_dir / 'normality_tests.csv', index=False)
    
    return results_df


def correlation_analysis(df: pd.DataFrame, output_dir: Path):
    """Analyze correlations between variables."""
    print("\n" + "="*70)
    print("CORRELATION ANALYSIS")
    print("="*70)
    
    # Compute correlation matrix
    corr = df.corr(method='pearson')
    
    # Find strongest correlations
    print("\nTop 20 strongest correlations:")
    corr_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            corr_pairs.append({
                'Var1': corr.columns[i],
                'Var2': corr.columns[j],
                'r': corr.iloc[i, j]
            })
    
    corr_pairs_df = pd.DataFrame(corr_pairs)
    corr_pairs_df['|r|'] = corr_pairs_df['r'].abs()
    corr_pairs_df = corr_pairs_df.sort_values('|r|', ascending=False)
    
    print(corr_pairs_df.head(20).to_string(index=False))
    
    # Plot correlation heatmap
    fig, ax = plt.subplots(figsize=(16, 14))
    
    # Create category color bar
    category_colors = []
    for i, col in enumerate(df.columns):
        for cat, indices in CATEGORIES.items():
            if i in indices:
                category_colors.append(CATEGORY_COLORS[cat])
                break
    
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, 
                annot=False, square=True, linewidths=0.5,
                cbar_kws={'shrink': 0.8, 'label': 'Pearson r'},
                ax=ax)
    
    # Add category labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    ax.set_title('Correlation Matrix of Subjective Experience Variables', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Within-category vs between-category correlations
    print("\n" + "-"*50)
    print("WITHIN vs BETWEEN CATEGORY CORRELATIONS:")
    print("-"*50)
    
    within_corrs = []
    between_corrs = []
    
    for i, col1 in enumerate(df.columns):
        for j, col2 in enumerate(df.columns):
            if i >= j:
                continue
            
            cat1 = [cat for cat, indices in CATEGORIES.items() if i in indices][0]
            cat2 = [cat for cat, indices in CATEGORIES.items() if j in indices][0]
            
            r = corr.iloc[i, j]
            if cat1 == cat2:
                within_corrs.append(r)
            else:
                between_corrs.append(r)
    
    print(f"\nWithin-category correlations:  mean r = {np.mean(within_corrs):.3f} ± {np.std(within_corrs):.3f}")
    print(f"Between-category correlations: mean r = {np.mean(between_corrs):.3f} ± {np.std(between_corrs):.3f}")
    
    # Statistical test
    t, p = stats.ttest_ind(within_corrs, between_corrs)
    print(f"\nDifference test: t = {t:.3f}, p = {p:.4f}")
    
    corr.to_csv(output_dir / 'correlation_matrix.csv')
    
    return corr


def distribution_plots(df: pd.DataFrame, output_dir: Path):
    """Plot distributions for all variables."""
    print("\n" + "="*70)
    print("GENERATING DISTRIBUTION PLOTS")
    print("="*70)
    
    # Histograms for all variables
    n_cols = 4
    n_rows = int(np.ceil(len(df.columns) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
    axes = axes.flatten()
    
    for i, col in enumerate(df.columns):
        ax = axes[i]
        
        # Get category color
        for cat, indices in CATEGORIES.items():
            if i in indices:
                color = CATEGORY_COLORS[cat]
                break
        
        ax.hist(df[col].dropna(), bins=15, color=color, alpha=0.7, edgecolor='white')
        ax.axvline(df[col].mean(), color='black', linestyle='--', linewidth=1.5, label='Mean')
        ax.axvline(df[col].median(), color='red', linestyle=':', linewidth=1.5, label='Median')
        
        ax.set_title(col.replace(' ', '\n'), fontsize=9, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Count')
        
        # Add skewness annotation
        skew = df[col].skew()
        ax.annotate(f'skew={skew:.2f}', xy=(0.95, 0.95), xycoords='axes fraction',
                   ha='right', va='top', fontsize=8)
    
    # Hide unused axes
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Distribution of Subjective Experience Variables', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'distributions_all.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Box plots by category
    fig, axes = plt.subplots(1, 4, figsize=(16, 6))
    
    for idx, (cat, indices) in enumerate(CATEGORIES.items()):
        ax = axes[idx]
        cat_cols = [TARGET_NAMES[i] for i in indices]
        cat_data = df[cat_cols]
        
        # Normalize for comparison
        scaler = StandardScaler()
        cat_normalized = pd.DataFrame(
            scaler.fit_transform(cat_data),
            columns=[c.replace(f'{cat} ', '').replace('ASC ', '') for c in cat_cols]
        )
        
        cat_normalized.boxplot(ax=ax, vert=True, patch_artist=True)
        for patch in ax.patches:
            patch.set_facecolor(CATEGORY_COLORS[cat])
            patch.set_alpha(0.7)
        
        ax.set_title(f'{cat} Subscales', fontsize=12, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Standardized Score')
    
    plt.suptitle('Distribution of Subscales by Category (Standardized)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'boxplots_by_category.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved distribution plots to {output_dir}")


def pca_analysis(df: pd.DataFrame, output_dir: Path):
    """Perform PCA to understand data structure."""
    print("\n" + "="*70)
    print("PRINCIPAL COMPONENT ANALYSIS")
    print("="*70)
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.dropna())
    
    # Fit PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Explained variance
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    
    print("\nExplained Variance by Component:")
    for i in range(min(10, len(pca.explained_variance_ratio_))):
        print(f"  PC{i+1}: {pca.explained_variance_ratio_[i]*100:.1f}% (cumulative: {cumsum[i]*100:.1f}%)")
    
    n_90 = np.argmax(cumsum >= 0.90) + 1
    n_95 = np.argmax(cumsum >= 0.95) + 1
    print(f"\nComponents for 90% variance: {n_90}")
    print(f"Components for 95% variance: {n_95}")
    
    # Plot explained variance
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    ax1.bar(range(1, len(pca.explained_variance_ratio_)+1), 
            pca.explained_variance_ratio_*100, alpha=0.7, color='steelblue')
    ax1.plot(range(1, len(cumsum)+1), cumsum*100, 'ro-', markersize=4)
    ax1.axhline(90, color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(95, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance (%)')
    ax1.set_title('PCA Explained Variance')
    ax1.legend(['Cumulative', 'Individual'])
    
    # Plot first two components
    ax2 = axes[1]
    colors = []
    for i in range(len(df)):
        colors.append('steelblue')
    
    scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c='steelblue', alpha=0.7, s=50)
    
    # Add subject labels
    for i, (x, y) in enumerate(zip(X_pca[:, 0], X_pca[:, 1])):
        ax2.annotate(f'S{i+1}', (x, y), fontsize=7, alpha=0.7)
    
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax2.set_title('Subjects in PCA Space')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pca_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Component loadings
    print("\n" + "-"*50)
    print("TOP LOADINGS FOR FIRST 3 COMPONENTS:")
    print("-"*50)
    
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(len(pca.components_))],
        index=df.columns
    )
    
    for pc in ['PC1', 'PC2', 'PC3']:
        print(f"\n{pc} (top 5 positive, top 5 negative):")
        sorted_loadings = loadings[pc].sort_values()
        print("  Positive:", list(sorted_loadings.tail(5).index))
        print("  Negative:", list(sorted_loadings.head(5).index))
    
    loadings.to_csv(output_dir / 'pca_loadings.csv')
    
    return pca, X_pca


def cluster_analysis(df: pd.DataFrame, output_dir: Path):
    """Cluster subjects based on their responses."""
    print("\n" + "="*70)
    print("CLUSTER ANALYSIS OF SUBJECTS")
    print("="*70)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.dropna())
    
    # Hierarchical clustering
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Dendrogram
    ax1 = axes[0]
    linkage_matrix = linkage(X_scaled, method='ward')
    dendrogram(linkage_matrix, ax=ax1, labels=[f'S{i+1}' for i in range(len(df))],
               leaf_rotation=90, leaf_font_size=8)
    ax1.set_title('Hierarchical Clustering Dendrogram', fontweight='bold')
    ax1.set_xlabel('Subject')
    ax1.set_ylabel('Distance (Ward)')
    
    # K-means elbow plot
    ax2 = axes[1]
    inertias = []
    K_range = range(2, min(11, len(df)))
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    ax2.plot(K_range, inertias, 'bo-')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Inertia')
    ax2.set_title('K-Means Elbow Plot', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cluster_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Cluster with k=3 and describe
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    print(f"\nK-means clustering with k={n_clusters}:")
    for c in range(n_clusters):
        n_subjects = (clusters == c).sum()
        cluster_data = df.iloc[clusters == c]
        print(f"\n  Cluster {c+1} ({n_subjects} subjects):")
        
        # Find distinguishing features
        cluster_means = cluster_data.mean()
        overall_means = df.mean()
        diff = cluster_means - overall_means
        
        top_high = diff.nlargest(3)
        top_low = diff.nsmallest(3)
        
        print(f"    Higher than average: {list(top_high.index)}")
        print(f"    Lower than average:  {list(top_low.index)}")
    
    # Save cluster assignments
    cluster_df = pd.DataFrame({
        'Subject': [f'S{i+1}' for i in range(len(df))],
        'Cluster': clusters + 1
    })
    cluster_df.to_csv(output_dir / 'cluster_assignments.csv', index=False)
    
    return clusters


def subject_profiles(df: pd.DataFrame, output_dir: Path):
    """Analyze individual subject profiles."""
    print("\n" + "="*70)
    print("SUBJECT PROFILE ANALYSIS")
    print("="*70)
    
    # Standardize
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df),
        columns=df.columns,
        index=df.index
    )
    
    # Find extreme subjects
    print("\nExtreme subjects (highest/lowest on each variable):")
    extremes = []
    
    for col in df.columns:
        max_subj = df[col].idxmax()
        min_subj = df[col].idxmin()
        extremes.append({
            'Variable': col,
            'Highest': max_subj,
            'Highest_Value': df[col].max(),
            'Lowest': min_subj,
            'Lowest_Value': df[col].min()
        })
    
    extremes_df = pd.DataFrame(extremes)
    print(extremes_df.to_string(index=False))
    
    # Overall intensity score (mean across all variables)
    df['Total_Intensity'] = df.mean(axis=1)
    
    print("\n" + "-"*50)
    print("TOTAL EXPERIENCE INTENSITY (mean across all variables):")
    print("-"*50)
    intensity_sorted = df['Total_Intensity'].sort_values(ascending=False)
    
    print("\nTop 5 most intense experiences:")
    for subj, val in intensity_sorted.head().items():
        print(f"  {subj}: {val:.2f}")
    
    print("\nTop 5 least intense experiences:")
    for subj, val in intensity_sorted.tail().items():
        print(f"  {subj}: {val:.2f}")
    
    # Radar plot for top 3 and bottom 3 subjects
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), subplot_kw=dict(projection='polar'))
    
    # Category means for radar
    category_means = {}
    for cat, indices in CATEGORIES.items():
        cat_cols = [TARGET_NAMES[i] for i in indices]
        category_means[cat] = df_scaled[cat_cols].mean(axis=1)
    
    categories = list(CATEGORIES.keys())
    n_cats = len(categories)
    angles = [n / float(n_cats) * 2 * np.pi for n in range(n_cats)]
    angles += angles[:1]  # Close the loop
    
    top_3 = intensity_sorted.head(3).index
    bottom_3 = intensity_sorted.tail(3).index
    
    for idx, subj in enumerate(top_3):
        ax = axes[0, idx]
        values = [category_means[cat][subj] for cat in categories]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, color='#e74c3c')
        ax.fill(angles, values, alpha=0.25, color='#e74c3c')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_title(f'{subj}\n(High Intensity)', fontweight='bold')
    
    for idx, subj in enumerate(bottom_3):
        ax = axes[1, idx]
        values = [category_means[cat][subj] for cat in categories]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, color='#3498db')
        ax.fill(angles, values, alpha=0.25, color='#3498db')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_title(f'{subj}\n(Low Intensity)', fontweight='bold')
    
    plt.suptitle('Subject Profiles: Category Means (Standardized)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'subject_profiles.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Drop temporary column
    df.drop('Total_Intensity', axis=1, inplace=True)
    
    return extremes_df


def category_analysis(df: pd.DataFrame, output_dir: Path):
    """Detailed analysis of each category."""
    print("\n" + "="*70)
    print("CATEGORY-LEVEL ANALYSIS")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, (cat, indices) in enumerate(CATEGORIES.items()):
        ax = axes[idx]
        cat_cols = [TARGET_NAMES[i] for i in indices]
        cat_data = df[cat_cols]
        
        # Internal consistency (Cronbach's alpha approximation)
        k = len(cat_cols)
        item_vars = cat_data.var()
        total_var = cat_data.sum(axis=1).var()
        alpha = (k / (k - 1)) * (1 - item_vars.sum() / total_var) if k > 1 else np.nan
        
        print(f"\n{cat}:")
        print(f"  Number of subscales: {k}")
        print(f"  Cronbach's alpha (reliability): {alpha:.3f}")
        print(f"  Mean inter-item correlation: {cat_data.corr().values[np.triu_indices(k, 1)].mean():.3f}")
        
        # Plot heatmap for this category
        corr = cat_data.corr()
        short_labels = [c.replace(f'{cat} ', '').replace('ASC ', '') for c in cat_cols]
        
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                   xticklabels=short_labels, yticklabels=short_labels,
                   ax=ax, square=True, cbar_kws={'shrink': 0.8})
        ax.set_title(f'{cat} Internal Correlations\n(α = {alpha:.2f})', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'category_internal_correlations.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Cross-category correlations
    print("\n" + "-"*50)
    print("CROSS-CATEGORY CORRELATIONS:")
    print("-"*50)
    
    category_scores = {}
    for cat, indices in CATEGORIES.items():
        cat_cols = [TARGET_NAMES[i] for i in indices]
        category_scores[cat] = df[cat_cols].mean(axis=1)
    
    cat_df = pd.DataFrame(category_scores)
    cross_corr = cat_df.corr()
    print("\n", cross_corr.round(3).to_string())
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cross_corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
               square=True, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title('Cross-Category Correlations', fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'cross_category_correlations.png', dpi=150, bbox_inches='tight')
    plt.close()


def challenging_experience_analysis(df: pd.DataFrame, output_dir: Path):
    """Analyze the 'challenging' experience variables specifically."""
    print("\n" + "="*70)
    print("CHALLENGING EXPERIENCE ANALYSIS")
    print("="*70)
    
    # Post variables are the challenging ones
    post_cols = [TARGET_NAMES[i] for i in CATEGORIES['Post']]
    post_data = df[post_cols]
    
    print("\nPost (Challenging) Experience Summary:")
    print(post_data.describe().round(2).to_string())
    
    # Correlation with positive experiences
    positive_cols = [TARGET_NAMES[i] for i in CATEGORIES['ASC'][:5]]  # First 5 ASC are positive
    mystical_cols = [TARGET_NAMES[i] for i in CATEGORIES['MEQ']]
    
    print("\n" + "-"*50)
    print("CORRELATIONS: Challenging vs Positive Experiences")
    print("-"*50)
    
    challenging_mean = post_data.mean(axis=1)
    positive_mean = df[positive_cols].mean(axis=1)
    mystical_mean = df[mystical_cols].mean(axis=1)
    
    r_pos, p_pos = stats.pearsonr(challenging_mean, positive_mean)
    r_myst, p_myst = stats.pearsonr(challenging_mean, mystical_mean)
    
    print(f"\nChallenging vs Positive ASC: r = {r_pos:.3f}, p = {p_pos:.4f}")
    print(f"Challenging vs Mystical MEQ: r = {r_myst:.3f}, p = {p_myst:.4f}")
    
    # Subjects with high challenging but also high positive
    challenging_high = challenging_mean > challenging_mean.median()
    positive_high = positive_mean > positive_mean.median()
    
    both_high = (challenging_high & positive_high).sum()
    print(f"\nSubjects with BOTH high challenging AND high positive: {both_high}/{len(df)}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#e74c3c' if ch else '#3498db' for ch in challenging_high]
    ax.scatter(positive_mean, challenging_mean, c=colors, alpha=0.7, s=100)
    
    for i, (x, y) in enumerate(zip(positive_mean, challenging_mean)):
        ax.annotate(f'S{i+1}', (x, y), fontsize=8, alpha=0.7)
    
    ax.set_xlabel('Positive Experience (ASC mean)', fontsize=12)
    ax.set_ylabel('Challenging Experience (Post mean)', fontsize=12)
    ax.set_title('Positive vs Challenging Experiences', fontsize=14, fontweight='bold')
    
    # Add regression line
    z = np.polyfit(positive_mean, challenging_mean, 1)
    p = np.poly1d(z)
    x_line = np.linspace(positive_mean.min(), positive_mean.max(), 100)
    ax.plot(x_line, p(x_line), 'k--', alpha=0.5, label=f'r = {r_pos:.2f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'challenging_vs_positive.png', dpi=150, bbox_inches='tight')
    plt.close()


def generate_summary_report(df: pd.DataFrame, output_dir: Path):
    """Generate a text summary report."""
    report = []
    report.append("="*70)
    report.append("SUBJECTIVE EXPERIENCE QUESTIONNAIRE ANALYSIS - SUMMARY REPORT")
    report.append("="*70)
    report.append(f"\nDataset: {len(df)} subjects, {len(df.columns)} variables")
    report.append(f"Categories: ASC (11), NDE (4), MEQ (5), Post (3)")
    
    report.append("\n" + "-"*50)
    report.append("KEY FINDINGS:")
    report.append("-"*50)
    
    # Most variable
    most_variable = df.std().idxmax()
    least_variable = df.std().idxmin()
    report.append(f"\n• Most variable response: {most_variable} (SD = {df[most_variable].std():.2f})")
    report.append(f"• Least variable response: {least_variable} (SD = {df[least_variable].std():.2f})")
    
    # Highest/lowest means
    highest_mean = df.mean().idxmax()
    lowest_mean = df.mean().idxmin()
    report.append(f"\n• Highest average: {highest_mean} (mean = {df[highest_mean].mean():.2f})")
    report.append(f"• Lowest average: {lowest_mean} (mean = {df[lowest_mean].mean():.2f})")
    
    # Strongest correlations
    corr = df.corr()
    np.fill_diagonal(corr.values, 0)
    max_corr_idx = np.unravel_index(np.abs(corr.values).argmax(), corr.shape)
    max_corr_vars = (corr.columns[max_corr_idx[0]], corr.columns[max_corr_idx[1]])
    max_corr_val = corr.iloc[max_corr_idx]
    report.append(f"\n• Strongest correlation: {max_corr_vars[0]} ↔ {max_corr_vars[1]} (r = {max_corr_val:.2f})")
    
    report.append("\n" + "-"*50)
    report.append("IMPLICATIONS FOR MACHINE LEARNING:")
    report.append("-"*50)
    
    # Variability
    cv = (df.std() / df.mean() * 100).mean()
    report.append(f"\n• Mean coefficient of variation: {cv:.1f}%")
    
    # Highly correlated pairs (potential redundancy)
    high_corr = (np.abs(corr) > 0.8).sum().sum() // 2
    report.append(f"• Highly correlated pairs (|r| > 0.8): {high_corr}")
    
    # Normality
    n_normal = sum(1 for col in df.columns if stats.shapiro(df[col])[1] > 0.05)
    report.append(f"• Normally distributed variables: {n_normal}/{len(df.columns)}")
    
    report.append("\n" + "="*70)
    
    # Save report
    with open(output_dir / 'summary_report.txt', 'w') as f:
        f.write('\n'.join(report))
    
    print('\n'.join(report))


def main():
    """Run complete analysis."""
    # Setup paths
    script_dir = Path(__file__).parent
    targets_path = script_dir.parent.parent / 'EEGNet' / 'targets.csv'
    
    output_dir = script_dir / 'questionnaire_analysis'
    output_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("SUBJECTIVE EXPERIENCE QUESTIONNAIRE ANALYSIS")
    print("="*70)
    print(f"\nLoading data from: {targets_path}")
    print(f"Output directory:  {output_dir}")
    
    # Load data
    df = load_targets(str(targets_path))
    print(f"\nLoaded {len(df)} subjects, {len(df.columns)} variables")
    
    # Run all analyses
    stats_df = descriptive_statistics(df, output_dir)
    normality_tests(df, output_dir)
    correlation_analysis(df, output_dir)
    distribution_plots(df, output_dir)
    pca_analysis(df, output_dir)
    cluster_analysis(df, output_dir)
    subject_profiles(df, output_dir)
    category_analysis(df, output_dir)
    challenging_experience_analysis(df, output_dir)
    generate_summary_report(df, output_dir)
    
    print("\n" + "="*70)
    print(f"ANALYSIS COMPLETE!")
    print(f"All outputs saved to: {output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()

