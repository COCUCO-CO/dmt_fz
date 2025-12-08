"""
Unified update functions for EEG visualization.

Eliminates code duplication between EEG1 and EEG2 by using parameters
instead of separate functions.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Optional, List, Dict, Any
import numpy as np
import plotly.graph_objects as go

from .signal import process_data, compute_fft, compute_hilbert

if TYPE_CHECKING:
    from nicegui.elements.plotly import Plotly


# Color schemes for primary (EEG1) and secondary (EEG2)
PRIMARY_COLORS = [
    '#00ff88', '#00d4ff', '#ffcc00', '#ff6b9d', '#a78bfa',
    '#00ffcc', '#ff9f43', '#74b9ff', '#55efc4', '#fd79a8',
]
SECONDARY_COLORS = [
    '#f472b6', '#fb7185', '#fda4af', '#fecdd3', '#ffe4e6',
]
PRIMARY_FFT_FILLS = [
    'rgba(0,255,136,0.15)', 'rgba(0,212,255,0.15)', 
    'rgba(255,204,0,0.15)', 'rgba(255,107,157,0.15)', 
    'rgba(167,139,250,0.15)',
]
SECONDARY_FFT_FILLS = [
    'rgba(244,114,182,0.15)', 'rgba(251,113,133,0.15)',
    'rgba(253,164,175,0.15)', 'rgba(254,205,211,0.15)',
    'rgba(255,228,230,0.15)',
]


def update_eeg_plot(
    plot: 'Plotly',
    data: np.ndarray,
    times: np.ndarray,
    channel_names: List[str],
    scale_factor: float = 1.0,
    use_secondary: bool = False,
    amplitudes_dict: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Update an EEG time-series plot with new data.
    
    Args:
        plot: NiceGUI Plotly element
        data: EEG data (channels x samples) in µV
        times: Time array in seconds
        channel_names: List of channel names
        scale_factor: Y-axis scale factor
        use_secondary: Use secondary (pink) color scheme
        amplitudes_dict: Dict to store channel amplitudes (modified in place)
        
    Returns:
        Dict mapping channel names to RMS amplitudes
    """
    n = len(channel_names)
    if n == 0 or data.size == 0:
        return {}
    
    # Normalize and compute amplitudes
    norm = np.zeros_like(data)
    amplitudes = {}
    for i in range(n):
        std = np.std(data[i])
        amplitudes[channel_names[i]] = np.sqrt(np.mean(data[i]**2))
        norm[i] = data[i] / (std * 3) if std > 0 else data[i]
    
    # Update amplitudes dict if provided
    if amplitudes_dict is not None:
        amplitudes_dict.update(amplitudes)
    
    spacing = 2.0 * scale_factor
    colors = SECONDARY_COLORS if use_secondary else PRIMARY_COLORS
    
    y_min = -spacing
    y_max = n * spacing
    
    with plot:
        plot.figure.data = []
        for i in range(n):
            off = (n - 1 - i) * spacing
            plot.figure.add_trace(go.Scatter(
                x=times,
                y=norm[i] + off,
                name=channel_names[i],
                line=dict(color=colors[i % len(colors)], width=1),
                hovertemplate=f'{channel_names[i]}: %{{customdata:.1f}} µV<extra></extra>',
                customdata=data[i]
            ))
        plot.figure.update_layout(
            yaxis=dict(
                tickmode='array',
                tickvals=[(n - 1 - i) * spacing for i in range(n)],
                ticktext=channel_names,
                range=[y_min, y_max],
                fixedrange=True
            )
        )
        plot.update()
    
    return amplitudes


def update_fft_plot(
    plot: 'Plotly',
    data: np.ndarray,
    sfreq: float,
    channel_names: List[str],
    max_freq: float = 60.0,
    max_channels: int = 5,
    use_secondary: bool = False
) -> None:
    """
    Update an FFT power spectrum plot.
    
    Args:
        plot: NiceGUI Plotly element
        data: EEG data (channels x samples) in µV
        sfreq: Sampling frequency
        channel_names: List of channel names
        max_freq: Maximum frequency to display
        max_channels: Maximum number of channels to show
        use_secondary: Use secondary color scheme
    """
    if data.size == 0:
        return
    
    # Limit channels
    data = data[:max_channels]
    channel_names = channel_names[:max_channels]
    
    freqs, fft_vals = compute_fft(data, sfreq, window=None)
    mask = freqs <= max_freq
    freqs, fft_vals = freqs[mask], fft_vals[:, mask]
    
    colors = SECONDARY_COLORS[:max_channels] if use_secondary else PRIMARY_COLORS[:max_channels]
    fills = SECONDARY_FFT_FILLS if use_secondary else PRIMARY_FFT_FILLS
    
    with plot:
        plot.figure.data = []
        for i, ch in enumerate(channel_names):
            plot.figure.add_trace(go.Scatter(
                x=freqs,
                y=fft_vals[i],
                name=ch,
                line=dict(color=colors[i % len(colors)], width=1.5),
                fill='tozeroy',
                fillcolor=fills[i % len(fills)]
            ))
        plot.update()


def update_hilbert_plot(
    plot: 'Plotly',
    data: np.ndarray,
    times: np.ndarray,
    use_secondary: bool = False
) -> None:
    """
    Update a Hilbert transform plot (envelope + phase).
    
    Args:
        plot: NiceGUI Plotly element with 2-row subplot
        data: Single channel EEG data (1D array) in µV
        times: Time array in seconds
        use_secondary: Use secondary color scheme
    """
    if data.size == 0:
        return
    
    amp, phase = compute_hilbert(data)
    
    # Color scheme
    if use_secondary:
        signal_color = '#f472b6'
        envelope_color = '#fb7185'
        phase_color = '#fda4af'
    else:
        signal_color = '#00d4ff'  # THEME_SECONDARY
        envelope_color = '#00ff88'  # THEME_PRIMARY
        phase_color = '#ffcc00'  # THEME_WARN
    
    with plot:
        plot.figure.data = []
        # Row 1: Signal + Envelope
        plot.figure.add_trace(go.Scatter(
            x=times, y=data, name='Signal',
            line=dict(color=signal_color, width=1)
        ), row=1, col=1)
        plot.figure.add_trace(go.Scatter(
            x=times, y=amp, name='Envelope',
            line=dict(color=envelope_color, width=2)
        ), row=1, col=1)
        plot.figure.add_trace(go.Scatter(
            x=times, y=-amp, showlegend=False,
            line=dict(color=envelope_color, width=2)
        ), row=1, col=1)
        # Row 2: Phase
        plot.figure.add_trace(go.Scatter(
            x=times, y=phase, name='Phase',
            line=dict(color=phase_color, width=1)
        ), row=2, col=1)
        plot.update()


def update_brain_plot(
    plot: 'Plotly',
    electrode_positions: Dict[str, tuple],
    selected_channels: List[str],
    amplitudes: Dict[str, float],
    use_secondary: bool = False
) -> None:
    """
    Update a brain topography plot.
    
    Args:
        plot: NiceGUI Plotly element
        electrode_positions: Dict mapping channel names to (x, y) positions
        selected_channels: List of selected channel names
        amplitudes: Dict mapping channel names to amplitude values
        use_secondary: Use secondary color scheme
    """
    # Compute amplitude range
    if amplitudes and selected_channels:
        vals = [amplitudes.get(ch, 0) for ch in selected_channels if ch in amplitudes]
        min_a, max_a = (min(vals), max(vals)) if vals else (0, 1)
        rng = max_a - min_a if max_a > min_a else 1
    else:
        min_a, rng = 0, 1
    
    # Separate selected and unselected electrodes
    sel_x, sel_y, sel_c, sel_t, sel_l = [], [], [], [], []
    uns_x, uns_y, uns_l = [], [], []
    
    for ch, (x, y) in electrode_positions.items():
        if ch in selected_channels:
            sel_x.append(x)
            sel_y.append(y)
            sel_l.append(ch)
            a = amplitudes.get(ch, 0)
            sel_c.append((a - min_a) / rng if rng > 0 else 0.5)
            sel_t.append(f'{ch}<br>{a:.1f} µV')
        else:
            uns_x.append(x)
            uns_y.append(y)
            uns_l.append(ch)
    
    # Color scheme
    if use_secondary:
        primary_color = '#f472b6'
        colorscale = [[0, '#831843'], [0.25, '#be185d'], [0.5, '#f472b6'], [0.75, '#fda4af'], [1, '#ffe4e6']]
    else:
        primary_color = '#00ff88'
        colorscale = [[0, '#0d47a1'], [0.25, '#00bcd4'], [0.5, '#00ff88'], [0.75, '#ffcc00'], [1, '#ff5722']]
    
    with plot:
        # Keep first 4 traces (head outline, nose, ears)
        plot.figure.data = plot.figure.data[:4]
        
        # Unselected electrodes
        if uns_x:
            plot.figure.add_trace(go.Scatter(
                x=uns_x, y=uns_y, mode='markers+text',
                marker=dict(
                    size=12,
                    color='rgba(30,30,30,0.6)',
                    line=dict(width=1, color='rgba(60,60,60,0.5)')
                ),
                text=uns_l,
                textposition='top center',
                textfont=dict(size=7, color='rgba(100,100,100,0.6)', family='JetBrains Mono'),
                hoverinfo='text',
                hovertext=uns_l,
                showlegend=False
            ))
        
        # Selected electrodes
        if sel_x:
            plot.figure.add_trace(go.Scatter(
                x=sel_x, y=sel_y, mode='markers+text',
                marker=dict(
                    size=18,
                    color=sel_c,
                    colorscale=colorscale,
                    cmin=0, cmax=1,
                    line=dict(width=2, color=primary_color),
                    showscale=True,
                    colorbar=dict(
                        title=dict(text='µV', font=dict(size=9, color='#666666')),
                        len=0.5, thickness=8, x=1.02,
                        tickfont=dict(size=8, color='#666666')
                    )
                ),
                text=sel_l,
                textposition='top center',
                textfont=dict(size=8, color=primary_color, family='JetBrains Mono'),
                hoverinfo='text',
                hovertext=sel_t,
                showlegend=False
            ))
        
        plot.update()

