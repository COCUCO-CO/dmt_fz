"""
Clustering Visualizations
Creates plots for brain state clustering analysis results
"""
import pickle
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path

# ============================================================================
# CONSTANTS
# ============================================================================

BANDS = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
BAND_COLORS = {
    'Delta': '#6366f1',  # Indigo
    'Theta': '#22c55e',  # Green
    'Alpha': '#eab308',  # Yellow
    'Beta': '#f97316',   # Orange
    'Gamma': '#ef4444'   # Red
}
CONDITION_COLORS = {
    'DMT': '#ef4444',
    'EC': '#3b82f6', 
    'EO': '#22c55e'
}
CLUSTER_COLORS = [
    '#3b82f6', '#ef4444', '#22c55e', '#f97316', '#a855f7',
    '#06b6d4', '#ec4899', '#84cc16', '#f43f5e', '#8b5cf6'
]

# ============================================================================
# HELPER FOR NO DATA
# ============================================================================

def _create_no_data_figure(message="No data available", height=350):
    """Create a figure showing 'no data' message"""
    fig = go.Figure()
    fig.add_annotation(
        x=0.5, y=0.5,
        xref='paper', yref='paper',
        text=message,
        showarrow=False,
        font=dict(size=16, color='#888'),
    )
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        height=height,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    return fig

# ============================================================================
# DATA LOADING
# ============================================================================

def find_results(run_dir):
    """Find clustering result files by band"""
    results = {}
    run_path = Path(run_dir)
    
    # Look for clustering_results directory
    cdir = run_path / 'clustering_results'
    if not cdir.exists():
        # Try to find in other locations
        for p in run_path.rglob('clustering_result*.pkl'):
            for band in BANDS:
                if band.lower() in str(p).lower():
                    results.setdefault(band, []).append(p)
                    break
        return results
    
    # Search in clustering_results
    for pkl in cdir.rglob('*.pkl'):
        name = pkl.stem.lower()
        for band in BANDS:
            if band.lower() in name or band.lower() in str(pkl.parent).lower():
                results.setdefault(band, []).append(pkl)
                break
    
    return results


def load_result(filepath):
    """Load a clustering result file"""
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[clustering_viz] Error loading {filepath}: {e}")
        return None


def get_score(data, score_name='silhouette'):
    """Extract score from data, handling various formats"""
    if data is None:
        return None
    
    # Direct attribute
    if score_name in data:
        return data[score_name]
    
    # Nested in 'scores' dict
    if 'scores' in data and isinstance(data['scores'], dict):
        if score_name in data['scores']:
            return data['scores'][score_name]
    
    # Try best_score for silhouette
    if score_name == 'silhouette' and 'best_score' in data:
        return data['best_score']
    
    return None


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_band_comparison_figure(run_dir):
    """
    Compare clustering scores (silhouette, calinski, davies) across bands
    """
    results = find_results(run_dir)
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=['Silhouette Score', 'Calinski-Harabasz', 'Davies-Bouldin'],
        horizontal_spacing=0.08
    )
    
    bands_found = []
    silhouette_scores = []
    calinski_scores = []
    davies_scores = []
    
    for band in BANDS:
        if band in results and results[band]:
            data = load_result(results[band][0])
            if data:
                bands_found.append(band)
                silhouette_scores.append(get_score(data, 'silhouette') or 0)
                calinski_scores.append(get_score(data, 'calinski') or get_score(data, 'calinski_harabasz') or 0)
                davies_scores.append(get_score(data, 'davies') or get_score(data, 'davies_bouldin') or 0)
    
    # Return no data if no results found
    if not bands_found:
        return _create_no_data_figure("No clustering results found\nRun clustering.py first", 400)
    
    # Silhouette (higher is better)
    fig.add_trace(go.Bar(
        x=bands_found, y=silhouette_scores,
        marker_color=[BAND_COLORS.get(b, '#888') for b in bands_found],
        text=[f'{s:.3f}' for s in silhouette_scores],
        textposition='outside',
        showlegend=False
    ), row=1, col=1)
    
    # Calinski-Harabasz (higher is better)
    fig.add_trace(go.Bar(
        x=bands_found, y=calinski_scores,
        marker_color=[BAND_COLORS.get(b, '#888') for b in bands_found],
        text=[f'{s:.0f}' for s in calinski_scores],
        textposition='outside',
        showlegend=False
    ), row=1, col=2)
    
    # Davies-Bouldin (lower is better)
    fig.add_trace(go.Bar(
        x=bands_found, y=davies_scores,
        marker_color=[BAND_COLORS.get(b, '#888') for b in bands_found],
        text=[f'{s:.3f}' for s in davies_scores],
        textposition='outside',
        showlegend=False
    ), row=1, col=3)
    
    fig.update_layout(
        title=dict(text='Clustering Quality Metrics by Band', font=dict(size=14)),
        template='plotly_dark',
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        height=350,
        margin=dict(l=50, r=30, t=60, b=50)
    )
    
    # Set y-axis ranges
    fig.update_yaxes(range=[0, max(silhouette_scores) * 1.2 + 0.1], row=1, col=1)
    fig.update_yaxes(range=[0, max(calinski_scores) * 1.2 + 10], row=1, col=2)
    fig.update_yaxes(range=[0, max(davies_scores) * 1.2 + 0.1], row=1, col=3)
    
    return fig


def create_cluster_distribution_figure(run_dir, band='Alpha'):
    """
    Show distribution of samples across clusters for a specific band
    """
    results = find_results(run_dir)
    
    labels = None
    n_clusters = 0
    
    if band in results and results[band]:
        data = load_result(results[band][0])
        if data and 'labels' in data:
            labels = np.array(data['labels'])
            n_clusters = data.get('n_clusters', data.get('best_k', len(np.unique(labels))))
    
    # Return no data if not found
    if labels is None:
        return _create_no_data_figure(f"No cluster labels for {band}\nRun clustering.py first", 350)
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    fig = go.Figure()
    
    # Sort by count
    sorted_idx = np.argsort(counts)[::-1]
    sorted_labels = unique_labels[sorted_idx]
    sorted_counts = counts[sorted_idx]
    
    fig.add_trace(go.Bar(
        x=[f'Cluster {i}' for i in sorted_labels],
        y=sorted_counts,
        marker_color=[CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in sorted_labels],
        text=[f'{c} ({c/len(labels)*100:.1f}%)' for c in sorted_counts],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=dict(text=f'Cluster Distribution - {band} ({n_clusters} clusters, {len(labels)} samples)', font=dict(size=14)),
        xaxis=dict(title='Cluster'),
        yaxis=dict(title='Count', gridcolor='rgba(255,255,255,0.1)'),
        template='plotly_dark',
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        height=350,
        margin=dict(l=60, r=30, t=50, b=50)
    )
    
    return fig


def create_condition_breakdown_figure(run_dir, band='Alpha'):
    """
    Show how samples from each condition are distributed across clusters
    """
    results = find_results(run_dir)
    
    labels = None
    conditions = None
    
    if band in results and results[band]:
        data = load_result(results[band][0])
        if data:
            labels = np.array(data.get('labels', []))
            conditions = np.array(data.get('conditions', data.get('condition_labels', [])))
    
    # Return no data
    if labels is None or len(labels) == 0 or conditions is None or len(conditions) == 0:
        return _create_no_data_figure(f"No cluster-condition data for {band}\nRun clustering.py first", 350)
    
    unique_clusters = np.unique(labels)
    unique_conditions = np.unique(conditions)
    
    fig = go.Figure()
    
    for cond in unique_conditions:
        counts = []
        for cluster in unique_clusters:
            count = np.sum((labels == cluster) & (conditions == cond))
            counts.append(count)
        
        fig.add_trace(go.Bar(
            name=cond,
            x=[f'C{i}' for i in unique_clusters],
            y=counts,
            marker_color=CONDITION_COLORS.get(cond, '#888')
        ))
    
    fig.update_layout(
        title=dict(text=f'Condition Breakdown by Cluster - {band}', font=dict(size=14)),
        xaxis=dict(title='Cluster'),
        yaxis=dict(title='Count', gridcolor='rgba(255,255,255,0.1)'),
        barmode='group',
        template='plotly_dark',
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        height=350,
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=60, r=30, t=50, b=50)
    )
    
    return fig


def create_score_heatmap_figure(run_dir):
    """
    Heatmap of all clustering scores across bands
    """
    results = find_results(run_dir)
    
    data_matrix = []
    bands_found = []
    
    for band in BANDS:
        if band in results and results[band]:
            r = load_result(results[band][0])
            if r:
                s = get_score(r, 'silhouette') or 0
                c = get_score(r, 'calinski') or get_score(r, 'calinski_harabasz') or 0
                d = get_score(r, 'davies') or get_score(r, 'davies_bouldin') or 0
                
                # Normalize scores: silhouette [-1,1]->0-1, calinski 0-1000->0-1, davies invert
                s_norm = (s + 1) / 2  # Map from [-1,1] to [0,1]
                c_norm = min(c / 500, 1.0)  # Rough normalization
                d_norm = 1 - min(d / 2, 1.0)  # Invert and clip
                
                data_matrix.append([s_norm, c_norm, d_norm])
                bands_found.append(band)
    
    # Return no data
    if not data_matrix:
        return _create_no_data_figure("No silhouette heatmap data\nRun clustering.py first", 400)
    
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        z=data_matrix,
        x=['Silhouette (↑)', 'Calinski (↑)', 'Davies (↓)'],
        y=bands_found,
        colorscale='Viridis',
        zmin=0, zmax=1,
        text=[[f'{v:.2f}' for v in row] for row in data_matrix],
        texttemplate='%{text}',
        textfont=dict(size=12),
        hovertemplate='%{y} - %{x}<br>Score: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text='Clustering Scores Heatmap (normalized)', font=dict(size=14)),
        template='plotly_dark',
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        height=300,
        margin=dict(l=80, r=30, t=50, b=50)
    )
    
    return fig


def create_pca_scatter_figure(run_dir, band='Alpha'):
    """
    PCA scatter plot of clustered data
    """
    results = find_results(run_dir)
    
    pca_data = None
    labels = None
    conditions = None
    
    if band in results and results[band]:
        data = load_result(results[band][0])
        if data:
            pca_data = data.get('pca_data', data.get('X_pca', data.get('reduced_data')))
            labels = np.array(data.get('labels', []))
            conditions = data.get('conditions', data.get('condition_labels'))
            
            if pca_data is not None:
                pca_data = np.array(pca_data)
    
    # Return no data
    if pca_data is None or labels is None or len(labels) == 0:
        return _create_no_data_figure(f"No PCA data for {band}\nRun clustering.py first", 400)
    
    if pca_data.shape[1] < 2:
        # Need at least 2 dimensions
        pca_data = np.column_stack([pca_data, np.zeros(len(pca_data))])
    
    fig = go.Figure()
    
    unique_labels = np.unique(labels)
    
    for cluster in unique_labels:
        mask = labels == cluster
        fig.add_trace(go.Scatter(
            x=pca_data[mask, 0],
            y=pca_data[mask, 1],
            mode='markers',
            name=f'Cluster {cluster}',
            marker=dict(
                size=6,
                color=CLUSTER_COLORS[cluster % len(CLUSTER_COLORS)],
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            hovertemplate=f'Cluster {cluster}<br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>'
        ))
    
    # Add cluster centers
    for cluster in unique_labels:
        mask = labels == cluster
        center = pca_data[mask].mean(axis=0)
        fig.add_trace(go.Scatter(
            x=[center[0]], y=[center[1]],
            mode='markers',
            marker=dict(size=15, color=CLUSTER_COLORS[cluster % len(CLUSTER_COLORS)], 
                       symbol='x', line=dict(width=3, color='white')),
            showlegend=False,
            hovertemplate=f'Center {cluster}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text=f'PCA Cluster Visualization - {band}', font=dict(size=14)),
        xaxis=dict(title='PC1', gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title='PC2', gridcolor='rgba(255,255,255,0.1)'),
        template='plotly_dark',
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        height=400,
        legend=dict(x=1.02, y=0.98),
        margin=dict(l=60, r=100, t=50, b=50)
    )
    
    return fig


def create_cluster_centroids_figure(run_dir, band='Alpha'):
    """
    Visualize cluster centroids as a heatmap or radar chart
    """
    results = find_results(run_dir)
    
    centroids = None
    
    if band in results and results[band]:
        data = load_result(results[band][0])
        if data:
            centroids = data.get('centroids', data.get('cluster_centers', data.get('centers')))
            if centroids is not None:
                centroids = np.array(centroids)
    
    # Return no data
    if centroids is None:
        return _create_no_data_figure(f"No centroids data for {band}\nRun clustering.py first", 400)
    
    n_clusters, n_features = centroids.shape
    
    # Limit features for visualization
    if n_features > 20:
        centroids = centroids[:, :20]
        n_features = 20
    
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        z=centroids,
        x=[f'F{i}' for i in range(n_features)],
        y=[f'Cluster {i}' for i in range(n_clusters)],
        colorscale='RdBu_r',
        zmid=0,
        colorbar=dict(title='Value'),
        hovertemplate='Cluster %{y}<br>Feature %{x}<br>Value: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=f'Cluster Centroids - {band}', font=dict(size=14)),
        xaxis=dict(title='Features'),
        yaxis=dict(title='Cluster'),
        template='plotly_dark',
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        height=300,
        margin=dict(l=80, r=30, t=50, b=50)
    )
    
    return fig


def create_elbow_plot(run_dir, band='Alpha'):
    """
    Elbow plot for cluster selection
    """
    results = find_results(run_dir)
    
    k_values = None
    scores = None
    best_k = None
    
    if band in results and results[band]:
        data = load_result(results[band][0])
        if data:
            k_values = data.get('k_range', data.get('k_values'))
            scores = data.get('inertias', data.get('silhouette_scores', data.get('scores_by_k')))
            best_k = data.get('best_k', data.get('n_clusters'))
            
            if isinstance(scores, dict):
                k_values = list(scores.keys())
                scores = list(scores.values())
    
    # Return no data
    if k_values is None or scores is None:
        return _create_no_data_figure(f"No elbow data for {band}\nRun clustering.py first", 350)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=k_values, y=scores,
        mode='lines+markers',
        line=dict(color=BAND_COLORS.get(band, '#888'), width=2),
        marker=dict(size=10),
        name='Score'
    ))
    
    if best_k and best_k in k_values:
        idx = k_values.index(best_k) if isinstance(k_values, list) else np.where(np.array(k_values) == best_k)[0][0]
        fig.add_trace(go.Scatter(
            x=[best_k], y=[scores[idx]],
            mode='markers',
            marker=dict(size=20, color='white', symbol='star'),
            name=f'Selected k={best_k}'
        ))
    
    fig.update_layout(
        title=dict(text=f'Cluster Selection - {band}', font=dict(size=14)),
        xaxis=dict(title='Number of Clusters (k)', gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title='Score', gridcolor='rgba(255,255,255,0.1)'),
        template='plotly_dark',
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        height=350,
        legend=dict(x=0.7, y=0.98),
        margin=dict(l=60, r=30, t=50, b=50)
    )
    
    return fig


def create_summary_dashboard(run_dir):
    """
    Create a comprehensive summary dashboard
    """
    results = find_results(run_dir)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Silhouette by Band', 'Cluster Counts', 'Score Heatmap', 'Best k'],
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'heatmap'}, {'type': 'scatter'}]]
    )
    
    bands_found = []
    silhouette_scores = []
    cluster_counts = []
    best_ks = []
    
    for band in BANDS:
        if band in results and results[band]:
            data = load_result(results[band][0])
            if data:
                bands_found.append(band)
                silhouette_scores.append(get_score(data, 'silhouette') or 0)
                
                labels = data.get('labels')
                if labels is not None:
                    cluster_counts.append(len(np.unique(labels)))
                else:
                    cluster_counts.append(data.get('n_clusters', data.get('best_k', 0)))
                
                best_ks.append(data.get('best_k', data.get('n_clusters', 0)))
    
    # Return no data
    if not bands_found:
        return _create_no_data_figure("No clustering summary data\nRun clustering.py first", 450)
    
    # Silhouette scores
    fig.add_trace(go.Bar(
        x=bands_found, y=silhouette_scores,
        marker_color=[BAND_COLORS.get(b, '#888') for b in bands_found],
        showlegend=False
    ), row=1, col=1)
    
    # Cluster counts
    fig.add_trace(go.Bar(
        x=bands_found, y=cluster_counts,
        marker_color=[BAND_COLORS.get(b, '#888') for b in bands_found],
        showlegend=False
    ), row=1, col=2)
    
    # Score heatmap
    heatmap_data = [[s, c/10] for s, c in zip(silhouette_scores, cluster_counts)]
    fig.add_trace(go.Heatmap(
        z=heatmap_data,
        x=['Silhouette', 'Clusters/10'],
        y=bands_found,
        colorscale='Viridis',
        showscale=False
    ), row=2, col=1)
    
    # Best k scatter
    fig.add_trace(go.Scatter(
        x=bands_found, y=best_ks,
        mode='markers+text',
        marker=dict(size=20, color=[BAND_COLORS.get(b, '#888') for b in bands_found]),
        text=[str(k) for k in best_ks],
        textposition='middle center',
        textfont=dict(color='white', size=10),
        showlegend=False
    ), row=2, col=2)
    
    fig.update_layout(
        title=dict(text='Clustering Summary Dashboard', font=dict(size=14)),
        template='plotly_dark',
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        height=500,
        showlegend=False,
        margin=dict(l=60, r=30, t=60, b=50)
    )
    
    return fig
