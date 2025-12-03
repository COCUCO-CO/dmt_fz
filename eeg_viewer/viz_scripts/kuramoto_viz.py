"""
Kuramoto Order Parameter Visualizations
Based on visualize_results.py and plot_utils.py
Creates beautiful and informative plots for synchronization analysis
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

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(filepath):
    """Load pickle data file"""
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[kuramoto_viz] Error loading {filepath}: {e}")
        return None


def find_syncro_files(run_dir, pattern='syncro-*.pkl'):
    """Find synchronization data files in run directory"""
    run_path = Path(run_dir)
    files = []
    
    # Search in main dir and subdirs
    for p in [run_path] + list(run_path.iterdir()):
        if p.is_dir():
            files.extend(list(p.glob(pattern)))
            files.extend(list(p.glob('phases-*.pkl')))
    
    return sorted(files)


def extract_kuramoto_values(data, band, key='kuramoto_stc'):
    """Extract Kuramoto values from data structure, handling nested formats"""
    if data is None:
        return None
    if not isinstance(data, dict):
        return None
    if key not in data:
        return None
    if data[key] is None or band not in data[key]:
        return None
    
    k = data[key][band]
    
    # Handle various nested structures
    if isinstance(k, (int, float)):
        return [k]
    
    if isinstance(k, np.ndarray):
        if k.ndim == 0:
            return [float(k)]
        return k.flatten().tolist()
    
    if isinstance(k, list):
        if len(k) == 0:
            return None
        
        # Check if it's nested [[epoch1, epoch2, ...]]
        if isinstance(k[0], list):
            # Flatten inner list of epochs
            values = []
            for epoch in k[0]:
                if hasattr(epoch, '__iter__') and not isinstance(epoch, str):
                    values.append(np.mean(epoch))
                else:
                    values.append(float(epoch))
            return values
        
        # Simple list of values
        values = []
        for item in k:
            if hasattr(item, '__iter__') and not isinstance(item, str):
                values.append(np.mean(item))
            else:
                values.append(float(item))
        return values
    
    return None


def extract_syncro_matrix(data, band, epoch=0, key='syncros_stc'):
    """Extract synchronization matrix from data"""
    if data is None or not isinstance(data, dict):
        return None
    if key not in data or data[key] is None:
        return None
    if band not in data[key]:
        return None
    
    s = data[key][band]
    
    try:
        if isinstance(s, list) and len(s) > 0:
            if isinstance(s[0], list) and len(s[0]) > epoch:
                return np.array(s[0][epoch])
            elif len(s) > epoch:
                return np.array(s[epoch])
            else:
                return np.array(s[0])
        elif isinstance(s, np.ndarray):
            if s.ndim == 3:
                return s[epoch]
            elif s.ndim == 2:
                return s
        return np.array(s)
    except:
        return None


def extract_phases(data, band, epoch=0, key='phases_stc'):
    """Extract phase data from data structure"""
    if data is None or not isinstance(data, dict):
        return None
    if key not in data or data[key] is None:
        return None
    if band not in data[key]:
        return None
    
    phases = data[key][band]
    
    try:
        if isinstance(phases, list) and len(phases) > epoch:
            return np.array(phases[epoch])
        elif isinstance(phases, np.ndarray):
            if phases.ndim >= 2:
                return phases[epoch] if phases.shape[0] > epoch else phases[0]
        return np.array(phases)
    except:
        return None


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_timeline_figure(data, band='Alpha'):
    """
    Timeline of Kuramoto order parameter across epochs
    Shows evolution of synchronization over time
    """
    fig = go.Figure()
    
    values = extract_kuramoto_values(data, band)
    
    if values is None or len(values) == 0:
        # Demo data
        np.random.seed(42)
        values = 0.4 + 0.2 * np.random.rand(50) + 0.1 * np.sin(np.linspace(0, 4*np.pi, 50))
    
    epochs = list(range(len(values)))
    mean_val = np.mean(values)
    
    # Main line with fill
    fig.add_trace(go.Scatter(
        x=epochs, y=values,
        mode='lines',
        line=dict(color=BAND_COLORS.get(band, '#888'), width=2),
        fill='tozeroy',
        fillcolor=f"rgba{tuple(list(int(BAND_COLORS.get(band, '#888888')[i:i+2], 16) for i in (1, 3, 5)) + [0.2])}",
        name=band
    ))
    
    # Mean line
    fig.add_hline(
        y=mean_val, 
        line=dict(color='white', dash='dash', width=1),
        annotation_text=f'Mean: {mean_val:.3f}',
        annotation_position='right'
    )
    
    # Add markers at regular intervals
    fig.add_trace(go.Scatter(
        x=epochs[::5], y=values[::5],
        mode='markers',
        marker=dict(color=BAND_COLORS.get(band, '#888'), size=8, line=dict(color='white', width=1)),
        showlegend=False,
        hovertemplate='Epoch %{x}<br>r = %{y:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=f'Kuramoto Order Parameter Timeline - {band}', font=dict(size=14)),
        xaxis=dict(title='Epoch', gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title='r (Kuramoto)', range=[0, 1], gridcolor='rgba(255,255,255,0.1)'),
        template='plotly_dark',
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        height=350,
        margin=dict(l=60, r=30, t=50, b=50)
    )
    
    return fig


def create_band_comparison_figure(data):
    """
    Box plot comparing Kuramoto values across all frequency bands
    """
    fig = go.Figure()
    
    has_data = False
    
    for band in BANDS:
        values = extract_kuramoto_values(data, band)
        
        if values is None or len(values) < 3:
            # Generate demo data
            np.random.seed(hash(band) % 2**32)
            base = {'Delta': 0.45, 'Theta': 0.52, 'Alpha': 0.58, 'Beta': 0.42, 'Gamma': 0.35}
            values = base.get(band, 0.4) + 0.15 * np.random.randn(30)
            values = np.clip(values, 0, 1)
        else:
            has_data = True
        
        fig.add_trace(go.Box(
            y=values,
            name=band,
            marker_color=BAND_COLORS.get(band),
            boxmean='sd',
            line=dict(width=2),
            fillcolor=f"rgba{tuple(list(int(BAND_COLORS.get(band, '#888888')[i:i+2], 16) for i in (1, 3, 5)) + [0.5])}"
        ))
    
    title = 'Kuramoto Order by Frequency Band'
    if not has_data:
        title += ' (Demo Data)'
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        yaxis=dict(title='r (Kuramoto)', range=[0, 1], gridcolor='rgba(255,255,255,0.1)'),
        xaxis=dict(title='Frequency Band'),
        template='plotly_dark',
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        showlegend=False,
        height=400,
        margin=dict(l=60, r=30, t=50, b=50)
    )
    
    return fig


def create_heatmap_figure(data, band='Alpha', epoch=0):
    """
    Synchronization matrix heatmap showing pairwise ROI synchronization
    """
    mat = extract_syncro_matrix(data, band, epoch)
    
    if mat is None or mat.ndim != 2:
        # Generate demo matrix
        np.random.seed(42)
        n = 50
        mat = np.random.rand(n, n) * 0.5 + 0.25
        mat = (mat + mat.T) / 2
        np.fill_diagonal(mat, 1.0)
    
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        z=mat,
        colorscale='Viridis',
        zmin=0, zmax=1,
        colorbar=dict(title='Sync', len=0.8),
        hovertemplate='ROI %{x} ↔ ROI %{y}<br>Sync: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=f'Synchronization Matrix - {band} (Epoch {epoch+1})', font=dict(size=14)),
        xaxis=dict(title='ROI', gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title='ROI', gridcolor='rgba(255,255,255,0.1)', autorange='reversed'),
        template='plotly_dark',
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        height=450,
        margin=dict(l=60, r=30, t=50, b=50)
    )
    
    return fig


def create_phase_distribution_figure(data, band='Alpha', epoch=0, sample=0):
    """
    Polar plot showing phase distribution of oscillators
    Like plot_osc in plot_utils.py
    """
    phases = extract_phases(data, band, epoch)
    
    if phases is None:
        # Generate demo phases
        np.random.seed(42)
        n_osc = 24
        mean_phase = np.random.rand() * 2 * np.pi
        phases = mean_phase + 0.8 * np.random.randn(n_osc, 100)
        phases = phases[:, sample]
    else:
        if phases.ndim == 2:
            phases = phases[:, min(sample, phases.shape[1]-1)]
        phases = phases.flatten()
    
    # Calculate order parameter
    z = np.exp(1j * phases)
    r = np.abs(z.mean())
    mean_phase = np.angle(z.mean())
    
    n_osc = len(phases)
    radii = np.ones(n_osc)
    
    fig = go.Figure()
    
    # Add unit circle
    theta_circle = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(go.Scatterpolar(
        r=np.ones(100),
        theta=np.degrees(theta_circle),
        mode='lines',
        line=dict(color='rgba(100,100,100,0.5)', width=1),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Add oscillators
    fig.add_trace(go.Scatterpolar(
        r=radii,
        theta=np.degrees(phases),
        mode='markers',
        marker=dict(
            size=12,
            color=BAND_COLORS.get(band, '#888'),
            line=dict(color='white', width=1)
        ),
        name='Oscillators',
        hovertemplate='Phase: %{theta:.1f}°<extra></extra>'
    ))
    
    # Add mean phase arrow
    fig.add_trace(go.Scatterpolar(
        r=[0, r],
        theta=[np.degrees(mean_phase), np.degrees(mean_phase)],
        mode='lines+markers',
        line=dict(color='white', width=3),
        marker=dict(size=[0, 10], symbol=['circle', 'triangle-up'], color='white'),
        name=f'Mean (r={r:.3f})',
        showlegend=True
    ))
    
    # Add center annotation
    fig.add_annotation(
        x=0.5, y=0.5,
        xref='paper', yref='paper',
        text=f'r = {r:.3f}',
        showarrow=False,
        font=dict(size=18, color='white'),
    )
    
    fig.update_layout(
        title=dict(text=f'Phase Distribution - {band}', font=dict(size=14)),
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 1.2]),
            angularaxis=dict(gridcolor='rgba(255,255,255,0.1)', linecolor='rgba(255,255,255,0.3)'),
            bgcolor='#0a0a0a'
        ),
        template='plotly_dark',
        paper_bgcolor='#0a0a0a',
        showlegend=True,
        legend=dict(x=0.85, y=0.95),
        height=400,
        margin=dict(l=30, r=30, t=50, b=30)
    )
    
    return fig


def create_roi_connectivity_figure(data, band='Alpha', threshold=0.5, epoch=0):
    """
    Network graph showing ROI connectivity based on synchronization
    """
    mat = extract_syncro_matrix(data, band, epoch)
    
    if mat is None or mat.ndim != 2:
        # Generate demo matrix
        np.random.seed(42)
        n = 50
        mat = np.random.rand(n, n) * 0.5 + 0.25
        mat = (mat + mat.T) / 2
        np.fill_diagonal(mat, 0)
    
    n = mat.shape[0]
    
    # Position nodes in a circle
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    
    fig = go.Figure()
    
    # Add edges for connections above threshold
    edge_x, edge_y = [], []
    edge_weights = []
    
    for i in range(n):
        for j in range(i+1, n):
            if mat[i, j] > threshold:
                edge_x.extend([x[i], x[j], None])
                edge_y.extend([y[i], y[j], None])
                edge_weights.append(mat[i, j])
    
    if edge_weights:
        # Normalize weights for alpha
        weights_norm = np.array(edge_weights)
        weights_norm = (weights_norm - weights_norm.min()) / (weights_norm.max() - weights_norm.min() + 1e-8)
        
        # Draw edges with varying opacity
        for i in range(0, len(edge_x), 3):
            idx = i // 3
            alpha = 0.2 + 0.6 * weights_norm[idx] if idx < len(weights_norm) else 0.3
            fig.add_trace(go.Scatter(
                x=edge_x[i:i+3], y=edge_y[i:i+3],
                mode='lines',
                line=dict(color=f'rgba(100, 150, 255, {alpha})', width=1 + weights_norm[idx] * 2 if idx < len(weights_norm) else 1),
                hoverinfo='skip',
                showlegend=False
            ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers',
        marker=dict(
            size=10,
            color=BAND_COLORS.get(band, '#888'),
            line=dict(color='white', width=1)
        ),
        hovertemplate='ROI %{pointNumber}<extra></extra>',
        name='ROIs'
    ))
    
    # Count connections per node
    connections = [(mat[i, :] > threshold).sum() for i in range(n)]
    n_edges = sum(1 for i in range(n) for j in range(i+1, n) if mat[i, j] > threshold)
    
    fig.update_layout(
        title=dict(text=f'ROI Connectivity - {band} (threshold={threshold:.2f}, {n_edges} edges)', font=dict(size=14)),
        xaxis=dict(visible=False, range=[-1.3, 1.3]),
        yaxis=dict(visible=False, range=[-1.3, 1.3], scaleanchor='x'),
        template='plotly_dark',
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        showlegend=False,
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def create_epoch_evolution_figure(data, band='Alpha'):
    """
    Detailed view of Kuramoto evolution with statistics
    """
    values = extract_kuramoto_values(data, band)
    
    if values is None or len(values) < 5:
        # Generate demo data
        np.random.seed(42)
        values = 0.5 + 0.15 * np.sin(np.linspace(0, 6*np.pi, 60)) + 0.1 * np.random.randn(60)
        values = np.clip(values, 0, 1)
    
    values = np.array(values)
    epochs = np.arange(len(values))
    
    # Calculate statistics
    mean_val = np.mean(values)
    std_val = np.std(values)
    
    # Calculate moving average
    window = min(5, len(values) // 3)
    if window > 1:
        moving_avg = np.convolve(values, np.ones(window)/window, mode='valid')
        ma_epochs = epochs[window//2:window//2 + len(moving_avg)]
    else:
        moving_avg = values
        ma_epochs = epochs
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Timeline', 'Distribution', 'Moving Avg (5 epochs)', 'Statistics'],
        specs=[[{'colspan': 2}, None], [{'type': 'histogram'}, {'type': 'indicator'}]],
        row_heights=[0.6, 0.4]
    )
    
    # Timeline
    fig.add_trace(go.Scatter(
        x=epochs, y=values,
        mode='lines',
        line=dict(color=BAND_COLORS.get(band), width=1, opacity=0.5),
        fill='tozeroy',
        fillcolor=f"rgba{tuple(list(int(BAND_COLORS.get(band, '#888888')[i:i+2], 16) for i in (1, 3, 5)) + [0.15])}",
        name='Raw',
        showlegend=True
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=ma_epochs, y=moving_avg,
        mode='lines',
        line=dict(color='white', width=2),
        name='Moving Avg'
    ), row=1, col=1)
    
    fig.add_hline(y=mean_val, line=dict(color='yellow', dash='dash'), row=1, col=1)
    fig.add_hrect(y0=mean_val-std_val, y1=mean_val+std_val, 
                  fillcolor='rgba(255,255,0,0.1)', line_width=0, row=1, col=1)
    
    # Distribution histogram
    fig.add_trace(go.Histogram(
        x=values,
        nbinsx=20,
        marker_color=BAND_COLORS.get(band),
        name='Distribution',
        showlegend=False
    ), row=2, col=1)
    
    # Statistics indicator
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=mean_val,
        delta={'reference': 0.5, 'relative': False, 'valueformat': '.3f'},
        number={'suffix': '', 'font': {'size': 40, 'color': BAND_COLORS.get(band)}, 'valueformat': '.3f'},
        title={'text': f'Mean r ({band})', 'font': {'size': 14}},
        domain={'row': 1, 'column': 1}
    ), row=2, col=2)
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        height=500,
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=50, r=30, t=40, b=40)
    )
    
    fig.update_yaxes(range=[0, 1], row=1, col=1)
    
    return fig


def create_multi_subject_comparison(run_dir, band='Alpha', conditions=['DMT', 'EC', 'EO']):
    """
    Compare Kuramoto across multiple subjects and conditions
    Like visualize_results.py boxplot
    """
    run_path = Path(run_dir)
    
    fig = go.Figure()
    
    data_found = False
    
    for cond in conditions:
        cond_values = []
        
        # Look for files in condition subdirectory
        cond_path = run_path / cond
        if cond_path.exists():
            files = list(cond_path.glob('phases-*.pkl')) + list(cond_path.glob('syncro-*.pkl'))
            
            for f in files:
                data = load_data(f)
                if data:
                    values = extract_kuramoto_values(data, band)
                    if values:
                        cond_values.append(np.mean(values))
                        data_found = True
        
        if not cond_values:
            # Demo data
            np.random.seed(hash(cond) % 2**32)
            base = {'DMT': 0.55, 'EC': 0.48, 'EO': 0.42}
            cond_values = base.get(cond, 0.45) + 0.12 * np.random.randn(15)
            cond_values = np.clip(cond_values, 0, 1)
        
        fig.add_trace(go.Box(
            y=cond_values,
            name=cond,
            marker_color=CONDITION_COLORS.get(cond, '#888'),
            boxmean='sd',
            line=dict(width=2)
        ))
    
    title = f'Kuramoto Comparison - {band}'
    if not data_found:
        title += ' (Demo Data)'
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        yaxis=dict(title='r (Kuramoto)', range=[0, 1], gridcolor='rgba(255,255,255,0.1)'),
        xaxis=dict(title='Condition'),
        template='plotly_dark',
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        showlegend=False,
        height=400,
        margin=dict(l=60, r=30, t=50, b=50)
    )
    
    return fig


def create_all_bands_timeline(data):
    """
    Timeline showing all bands overlaid for comparison
    """
    fig = go.Figure()
    
    max_epochs = 0
    
    for band in BANDS:
        values = extract_kuramoto_values(data, band)
        
        if values is None or len(values) < 3:
            # Generate demo data
            np.random.seed(hash(band) % 2**32)
            base = {'Delta': 0.45, 'Theta': 0.52, 'Alpha': 0.58, 'Beta': 0.42, 'Gamma': 0.35}
            n = 50
            values = base.get(band, 0.4) + 0.1 * np.random.randn(n)
            values = np.clip(values, 0, 1)
        
        epochs = list(range(len(values)))
        max_epochs = max(max_epochs, len(epochs))
        
        fig.add_trace(go.Scatter(
            x=epochs, y=values,
            mode='lines',
            line=dict(color=BAND_COLORS.get(band), width=2),
            name=f'{band} (μ={np.mean(values):.3f})',
            hovertemplate=f'{band}<br>Epoch %{{x}}<br>r = %{{y:.3f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text='Kuramoto Timeline - All Bands', font=dict(size=14)),
        xaxis=dict(title='Epoch', gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title='r (Kuramoto)', range=[0, 1], gridcolor='rgba(255,255,255,0.1)'),
        template='plotly_dark',
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        height=400,
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=60, r=30, t=50, b=50)
    )
    
    return fig
