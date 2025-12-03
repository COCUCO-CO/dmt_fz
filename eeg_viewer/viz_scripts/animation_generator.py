"""
Animation Generator - Simple working version
"""
import pickle
import numpy as np
import plotly.graph_objects as go

BANDS = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
BAND_COLORS = {'Delta': '#6366f1', 'Theta': '#22c55e', 'Alpha': '#eab308', 'Beta': '#f97316', 'Gamma': '#ef4444'}

def create_kuramoto_animation(data, band='Alpha'):
    """Animated Kuramoto timeline."""
    fig = go.Figure()
    
    values = []
    if 'kuramoto_stc' in data and band in data['kuramoto_stc']:
        k = data['kuramoto_stc'][band]
        if isinstance(k, list) and len(k) > 0:
            if isinstance(k[0], list):
                values = [np.mean(e) for e in k[0]]
            else:
                values = [np.mean(e) if hasattr(e, '__iter__') else e for e in k]
    
    if not values:
        values = list(0.3 + 0.3 * np.random.rand(20))
    
    n = len(values)
    
    # Create frames
    frames = [go.Frame(data=[go.Scatter(x=list(range(i+1)), y=values[:i+1],
                        mode='lines+markers', line=dict(color=BAND_COLORS.get(band)))], 
                       name=str(i)) for i in range(n)]
    
    fig.add_trace(go.Scatter(x=[0], y=[values[0]], mode='lines+markers',
                              line=dict(color=BAND_COLORS.get(band))))
    fig.frames = frames
    
    fig.update_layout(
        title=f'Kuramoto Animation - {band}',
        xaxis=dict(range=[-0.5, n+0.5], title='Epoch'),
        yaxis=dict(range=[0, 1], title='R'),
        template='plotly_dark', paper_bgcolor='#0a0a0a', plot_bgcolor='#0a0a0a',
        updatemenus=[dict(type='buttons', showactive=False, y=1.15, x=0.5, xanchor='center',
            buttons=[
                dict(label='Play', method='animate', args=[None, dict(frame=dict(duration=200, redraw=True))]),
                dict(label='Pause', method='animate', args=[[None], dict(frame=dict(duration=0), mode='immediate')])
            ]
        )],
        sliders=[dict(active=0, yanchor='top', currentvalue=dict(prefix='Epoch: '),
                      steps=[dict(args=[[str(i)], dict(frame=dict(duration=0), mode='immediate')],
                                  label=str(i), method='animate') for i in range(n)])]
    )
    return fig

def create_matrix_animation(data, band='Alpha'):
    """Animated sync matrix."""
    import plotly.express as px
    
    matrices = []
    if 'syncros_stc' in data and band in data['syncros_stc']:
        s = data['syncros_stc'][band]
        try:
            if isinstance(s, list) and len(s) > 0:
                if isinstance(s[0], list):
                    matrices = [np.array(m) for m in s[0][:20]]
                else:
                    matrices = [np.array(m) for m in s[:20] if hasattr(m, '__iter__')]
        except:
            pass
    
    if not matrices:
        matrices = [np.random.rand(20, 20) for _ in range(10)]
    
    n = len(matrices)
    
    fig = go.Figure(data=[go.Heatmap(z=matrices[0], colorscale='Viridis', zmin=0, zmax=1)])
    
    frames = [go.Frame(data=[go.Heatmap(z=m, colorscale='Viridis', zmin=0, zmax=1)], 
                       name=str(i)) for i, m in enumerate(matrices)]
    fig.frames = frames
    
    fig.update_layout(
        title=f'Matrix Animation - {band}',
        template='plotly_dark', paper_bgcolor='#0a0a0a', plot_bgcolor='#0a0a0a',
        updatemenus=[dict(type='buttons', showactive=False, y=1.15, x=0.5, xanchor='center',
            buttons=[
                dict(label='Play', method='animate', args=[None, dict(frame=dict(duration=300, redraw=True))]),
                dict(label='Pause', method='animate', args=[[None], dict(frame=dict(duration=0), mode='immediate')])
            ]
        )],
        sliders=[dict(active=0, yanchor='top', currentvalue=dict(prefix='Epoch: '),
                      steps=[dict(args=[[str(i)], dict(frame=dict(duration=0), mode='immediate')],
                                  label=str(i), method='animate') for i in range(n)])]
    )
    return fig

def create_all_bands_animation(data):
    """All bands animated comparison."""
    fig = go.Figure()
    
    band_data = {}
    max_n = 0
    
    for band in BANDS:
        if 'kuramoto_stc' in data and band in data['kuramoto_stc']:
            k = data['kuramoto_stc'][band]
            if isinstance(k, list) and len(k) > 0:
                if isinstance(k[0], list):
                    values = [np.mean(e) for e in k[0]]
                else:
                    values = [np.mean(e) if hasattr(e, '__iter__') else e for e in k]
                band_data[band] = values
                max_n = max(max_n, len(values))
    
    if not band_data:
        for band in BANDS:
            band_data[band] = list(0.3 + 0.3 * np.random.rand(20))
        max_n = 20
    
    # Initial traces
    for band in BANDS:
        if band in band_data:
            fig.add_trace(go.Scatter(x=[0], y=[band_data[band][0]], mode='lines+markers',
                                     name=band, line=dict(color=BAND_COLORS.get(band))))
    
    # Frames
    frames = []
    for i in range(1, max_n+1):
        frame_data = []
        for band in BANDS:
            if band in band_data:
                vals = band_data[band][:i]
                frame_data.append(go.Scatter(x=list(range(len(vals))), y=vals, mode='lines+markers',
                                             name=band, line=dict(color=BAND_COLORS.get(band))))
        frames.append(go.Frame(data=frame_data, name=str(i)))
    
    fig.frames = frames
    
    fig.update_layout(
        title='All Bands Animation',
        xaxis=dict(range=[-0.5, max_n+0.5], title='Epoch'),
        yaxis=dict(range=[0, 1], title='R'),
        template='plotly_dark', paper_bgcolor='#0a0a0a', plot_bgcolor='#0a0a0a',
        legend=dict(orientation='h', y=1.1),
        updatemenus=[dict(type='buttons', showactive=False, y=1.2, x=0.5, xanchor='center',
            buttons=[
                dict(label='Play', method='animate', args=[None, dict(frame=dict(duration=200, redraw=True))]),
                dict(label='Pause', method='animate', args=[[None], dict(frame=dict(duration=0), mode='immediate')])
            ]
        )]
    )
    return fig
