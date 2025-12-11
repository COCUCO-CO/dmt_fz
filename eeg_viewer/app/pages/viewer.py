"""Main EEG Viewer page."""
from pathlib import Path
import asyncio
import numpy as np
from nicegui import ui
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
    EEG_RAW_DIR, EEG_CLEAN_DIR,
    THEME_BG, THEME_CARD, THEME_BORDER, THEME_PRIMARY, THEME_SECONDARY,
    THEME_WARN, THEME_TEXT, THEME_TEXT_DIM, SIGNAL_COLORS, FREQ_BANDS
)
from eeg_loader import load_eeg_file, get_channel_data, scan_eeg_directory
from app.state import S
from app.visualization.styles.css import STYLE
from app.visualization.components.running_indicator import render_running_indicator
from app.visualization.components.global_header import render_global_header
from app.core.signal import (
    apply_notch, apply_bandpass, 
    compute_fft, compute_hilbert,
    process_data as _process_data_core
)


# Color palettes for EEG traces
_PRIMARY_COLORS = SIGNAL_COLORS
_SECONDARY_COLORS = ['#f472b6', '#fb7185', '#fda4af', '#fecdd3', '#ffe4e6']

# FFT fill colors
_PRIMARY_FFT_FILLS = ['rgba(0,255,136,0.15)', 'rgba(0,212,255,0.15)', 'rgba(255,204,0,0.15)', 'rgba(255,107,157,0.15)', 'rgba(167,139,250,0.15)']
_SECONDARY_FFT_FILLS = ['rgba(244,114,182,0.15)', 'rgba(251,113,133,0.15)', 'rgba(253,164,175,0.15)', 'rgba(254,205,211,0.15)', 'rgba(255,228,230,0.15)']

# Plot constants - Terminal style (from original working code)
PLOT_BG = 'rgba(8,8,8,1)'
PLOT_GRID = 'rgba(0,255,136,0.08)'
PLOT_GRID_MINOR = 'rgba(0,255,136,0.03)'


# ==============================================================================
# FIGURE CREATION FUNCTIONS - Original working implementations
# ==============================================================================

def make_eeg_fig(use_eeg2=False):
    """Create EEG figure with proper styling."""
    fig = go.Figure()
    title_color = '#f472b6' if use_eeg2 else THEME_PRIMARY
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=PLOT_BG,
        margin=dict(l=70, r=10, t=10, b=50),
        height=250,
        font=dict(family='JetBrains Mono, monospace', size=10, color=THEME_TEXT),
        xaxis=dict(
            title=dict(text='TIME [s]', font=dict(size=9, color=title_color)),
            gridcolor=PLOT_GRID,
            zerolinecolor=PLOT_GRID,
            tickfont=dict(size=9, color=THEME_TEXT_DIM),
            fixedrange=False
        ),
        yaxis=dict(
            gridcolor=PLOT_GRID_MINOR,
            tickfont=dict(size=9, color=title_color),
            fixedrange=True
        ),
        showlegend=False,
        hovermode='x unified',
        hoverlabel=dict(bgcolor=THEME_CARD, font=dict(family='JetBrains Mono', size=10))
    )
    return fig


def make_fft_fig(use_eeg2=False):
    """Create FFT figure with frequency band annotations."""
    fig = go.Figure()
    title_color = '#f472b6' if use_eeg2 else THEME_SECONDARY
    band_colors = ['rgba(0,212,255,0.08)', 'rgba(0,255,136,0.08)', 'rgba(255,204,0,0.08)', 'rgba(255,107,157,0.08)', 'rgba(167,139,250,0.08)']
    for i, (band, (lo, hi)) in enumerate(FREQ_BANDS.items()):
        fig.add_vrect(x0=lo, x1=hi, fillcolor=band_colors[i % len(band_colors)], line_width=0)
        fig.add_annotation(x=(lo+hi)/2, y=1.02, yref='paper', text=band, showarrow=False,
                          font=dict(size=11, color=THEME_TEXT_DIM, family='JetBrains Mono'))
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=PLOT_BG,
        margin=dict(l=60, r=10, t=30, b=50),
        height=180,
        font=dict(family='JetBrains Mono, monospace', size=10, color=THEME_TEXT),
        xaxis=dict(
            title=dict(text='FREQ [Hz]', font=dict(size=9, color=title_color)),
            gridcolor=PLOT_GRID,
            range=[0, 60],
            fixedrange=True,
            tickfont=dict(size=9, color=THEME_TEXT_DIM)
        ),
        yaxis=dict(
            title=dict(text='PWR [µV]', font=dict(size=9, color=title_color)),
            gridcolor=PLOT_GRID,
            fixedrange=True,
            tickfont=dict(size=9, color=THEME_TEXT_DIM)
        ),
        showlegend=True,
        legend=dict(orientation='h', y=1.15, font=dict(size=8, color=THEME_TEXT_DIM)),
        hovermode='x unified',
        hoverlabel=dict(bgcolor=THEME_CARD, font=dict(family='JetBrains Mono', size=10))
    )
    return fig


def make_fft_diff_fig():
    """Create FFT difference figure (EEG1 - EEG2)."""
    fig = go.Figure()
    # Use a distinct color for difference plot
    diff_color = '#a78bfa'  # Purple for difference
    band_colors = ['rgba(167,139,250,0.06)', 'rgba(167,139,250,0.08)', 'rgba(167,139,250,0.06)', 'rgba(167,139,250,0.08)', 'rgba(167,139,250,0.06)']
    for i, (band, (lo, hi)) in enumerate(FREQ_BANDS.items()):
        fig.add_vrect(x0=lo, x1=hi, fillcolor=band_colors[i % len(band_colors)], line_width=0)
        fig.add_annotation(x=(lo+hi)/2, y=1.02, yref='paper', text=band, showarrow=False,
                          font=dict(size=11, color=THEME_TEXT_DIM, family='JetBrains Mono'))
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=PLOT_BG,
        margin=dict(l=60, r=10, t=30, b=50),
        height=180,
        font=dict(family='JetBrains Mono, monospace', size=10, color=THEME_TEXT),
        xaxis=dict(
            title=dict(text='FREQ [Hz]', font=dict(size=9, color=diff_color)),
            gridcolor=PLOT_GRID,
            range=[0, 60],
            fixedrange=True,
            tickfont=dict(size=9, color=THEME_TEXT_DIM)
        ),
        yaxis=dict(
            title=dict(text='ΔPWR [µV]', font=dict(size=9, color=diff_color)),
            gridcolor=PLOT_GRID,
            fixedrange=True,
            tickfont=dict(size=9, color=THEME_TEXT_DIM),
            zeroline=True,
            zerolinecolor='rgba(255,255,255,0.3)',
            zerolinewidth=1
        ),
        showlegend=True,
        legend=dict(orientation='h', y=1.15, font=dict(size=8, color=THEME_TEXT_DIM)),
        hovermode='x unified',
        hoverlabel=dict(bgcolor=THEME_CARD, font=dict(family='JetBrains Mono', size=10))
    )
    return fig


def make_hilbert_fig(use_eeg2=False):
    """Create Hilbert figure with envelope and phase subplots."""
    title_color = '#f472b6' if use_eeg2 else THEME_WARN
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=('<b>ENVELOPE</b>', '<b>PHASE</b>'),
                        vertical_spacing=0.22)
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=PLOT_BG,
        margin=dict(l=60, r=10, t=35, b=50),
        height=180,
        font=dict(family='JetBrains Mono, monospace', size=10, color=THEME_TEXT),
        showlegend=True,
        legend=dict(orientation='h', y=1.15, font=dict(size=8, color=THEME_TEXT_DIM)),
        hovermode='x unified',
        hoverlabel=dict(bgcolor=THEME_CARD, font=dict(family='JetBrains Mono', size=10))
    )
    fig.update_annotations(font=dict(size=9, color=title_color, family='JetBrains Mono'))
    fig.update_xaxes(gridcolor=PLOT_GRID, tickfont=dict(size=9, color=THEME_TEXT_DIM), fixedrange=True)
    fig.update_yaxes(gridcolor=PLOT_GRID, tickfont=dict(size=9, color=THEME_TEXT_DIM), fixedrange=True)
    return fig


def make_hilbert_diff_fig():
    """Create Hilbert difference figure (EEG1 - EEG2) with envelope and phase subplots."""
    diff_color = '#a78bfa'  # Purple for difference
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=('<b>ΔENVELOPE</b>', '<b>ΔPHASE</b>'),
                        vertical_spacing=0.22)
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=PLOT_BG,
        margin=dict(l=60, r=10, t=35, b=50),
        height=180,
        font=dict(family='JetBrains Mono, monospace', size=10, color=THEME_TEXT),
        showlegend=True,
        legend=dict(orientation='h', y=1.15, font=dict(size=8, color=THEME_TEXT_DIM)),
        hovermode='x unified',
        hoverlabel=dict(bgcolor=THEME_CARD, font=dict(family='JetBrains Mono', size=10))
    )
    fig.update_annotations(font=dict(size=9, color=diff_color, family='JetBrains Mono'))
    fig.update_xaxes(gridcolor=PLOT_GRID, tickfont=dict(size=9, color=THEME_TEXT_DIM), fixedrange=True)
    fig.update_yaxes(gridcolor=PLOT_GRID, tickfont=dict(size=9, color=THEME_TEXT_DIM), fixedrange=True,
                    zeroline=True, zerolinecolor='rgba(255,255,255,0.3)', zerolinewidth=1)
    return fig


def make_brain_fig(use_eeg2=False):
    """Create brain topography figure with head outline."""
    fig = go.Figure()
    theta = np.linspace(0, 2*np.pi, 100)
    head_color = 'rgba(244,114,182,0.5)' if use_eeg2 else 'rgba(0,255,136,0.5)'
    # Head outline
    fig.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta), mode='lines',
                            line=dict(color=head_color, width=2), showlegend=False, hoverinfo='skip'))
    # Nose
    fig.add_trace(go.Scatter(x=[-0.08, 0, 0.08], y=[0.98, 1.12, 0.98], mode='lines',
                            line=dict(color=head_color, width=2), showlegend=False, hoverinfo='skip'))
    # Ears
    fig.add_trace(go.Scatter(x=[-1.02, -1.08, -1.02], y=[0.15, 0, -0.15], mode='lines',
                            line=dict(color=head_color, width=1.5), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=[1.02, 1.08, 1.02], y=[0.15, 0, -0.15], mode='lines',
                            line=dict(color=head_color, width=1.5), showlegend=False, hoverinfo='skip'))
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#0a0a0a',
        margin=dict(l=5, r=5, t=5, b=5),
        height=200,
        font=dict(family='JetBrains Mono, monospace', color=THEME_TEXT),
        xaxis=dict(range=[-1.25, 1.25], showgrid=False, zeroline=False, showticklabels=False, scaleanchor='y', fixedrange=True),
        yaxis=dict(range=[-0.9, 1.2], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
        showlegend=False,
        hovermode='closest',
        hoverlabel=dict(bgcolor=THEME_CARD, font=dict(family='JetBrains Mono', size=10, color='#f472b6' if use_eeg2 else THEME_PRIMARY))
    )
    return fig


def process_data(data, sfreq):
    """Apply enabled filters based on global state S."""
    return _process_data_core(
        data, sfreq,
        notch_enabled=S.notch_enabled,
        notch_freq=S.notch_freq,
        bandpass_enabled=S.bandpass_enabled,
        bandpass_low=S.bandpass_low,
        bandpass_high=S.bandpass_high
    )


# Electrode positions (10-20 system)
ELECTRODE_POSITIONS = {
    'Fp1': (-0.3, 0.9), 'Fp2': (0.3, 0.9), 'Fpz': (0.0, 0.95),
    'F7': (-0.7, 0.6), 'F3': (-0.35, 0.6), 'Fz': (0.0, 0.6), 'F4': (0.35, 0.6), 'F8': (0.7, 0.6),
    'FC1': (-0.2, 0.4), 'FC2': (0.2, 0.4), 'FC5': (-0.55, 0.4), 'FC6': (0.55, 0.4),
    'T7': (-0.85, 0.2), 'C3': (-0.4, 0.2), 'Cz': (0.0, 0.2), 'C4': (0.4, 0.2), 'T8': (0.85, 0.2),
    'T3': (-0.85, 0.2), 'T4': (0.85, 0.2),
    'CP1': (-0.2, 0.0), 'CP2': (0.2, 0.0), 'CPz': (0.0, 0.0), 'CP5': (-0.55, 0.0), 'CP6': (0.55, 0.0),
    'P7': (-0.7, -0.3), 'P3': (-0.35, -0.3), 'Pz': (0.0, -0.3), 'P4': (0.35, -0.3), 'P8': (0.7, -0.3),
    'T5': (-0.7, -0.3), 'T6': (0.7, -0.3),
    'O1': (-0.3, -0.7), 'O2': (0.3, -0.7), 'Oz': (0.0, -0.7),
    'M1': (-0.95, 0.0), 'M2': (0.95, 0.0), 'A1': (-0.95, 0.0), 'A2': (0.95, 0.0),
}

def _update_eeg_generic(eeg_plot, eeg_data, channels, use_secondary=False, update_time_label=False, amplitudes_dict=None):
    """Generic EEG update function for both EEG1 and EEG2."""
    if not eeg_plot or not eeg_data or not channels:
        return
    
    # Filter channels that exist in this EEG data
    valid_channels = [ch for ch in channels if ch in eeg_data.channel_types]
    if not valid_channels:
        return
    
    try:
        data, times, chs = get_channel_data(eeg_data, valid_channels, S.view_start, S.view_duration)
        data = process_data(data, eeg_data.sfreq) * 1e6  # Convert to µV
        n = len(chs)
        
        # Adjust spacing based on number of channels (like cleaner)
        spacing_factor = 0.35 if n <= 16 else (0.25 if n <= 24 else 0.2)
        spacing_factor *= S.scale_factor
        
        # Color scheme - white for primary (clean look), pink for secondary
        if use_secondary:
            signal_color = 'rgba(244, 114, 182, 0.7)'  # Pink
        else:
            signal_color = 'rgba(255, 255, 255, 0.6)'  # White (like cleaner)
        
        with eeg_plot:
            eeg_plot.figure.data = []
            y_ticks = []
            y_labels = []
            
            for i in range(n):
                offset = (n - 1 - i)
                y = data[i].copy()
                
                # Handle NaN/Inf values
                if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Better normalization (like cleaner) - robust to outliers
                std_val = np.std(y)
                if std_val < 1e-10:
                    # Channel is flat, show as flat line at offset
                    y_norm = np.zeros_like(y) + offset
                else:
                    y_norm = (y - np.mean(y)) / std_val * spacing_factor + offset
                
                # Compute amplitude for brain plot
                amp = np.sqrt(np.mean(y**2))
                if amplitudes_dict is not None:
                    amplitudes_dict[chs[i]] = amp
                
                y_ticks.append(offset)
                y_labels.append(chs[i])
                
                eeg_plot.figure.add_trace(go.Scatter(
                    x=times, y=y_norm, name=chs[i],
                    line=dict(color=signal_color, width=1),
                    hovertemplate=f'{chs[i]}: %{{customdata:.1f}} µV<extra></extra>',
                    customdata=y
                ))
            
            eeg_plot.figure.update_layout(
                yaxis=dict(
                    tickmode='array',
                    tickvals=y_ticks,
                    ticktext=y_labels,
                    range=[-0.5, n - 0.5],
                    fixedrange=True
                )
            )
            eeg_plot.update()
        
        if update_time_label and S.time_label and eeg_data:
            m, s = int(S.view_start // 60), S.view_start % 60
            tm, ts = int(eeg_data.duration_sec // 60), eeg_data.duration_sec % 60
            S.time_label.set_text(f'{m}:{s:04.1f} / {tm}:{ts:04.1f}')
    except Exception as e:
        print(f"EEG{'2' if use_secondary else ''} error: {e}")


def update_eeg():
    """Update primary EEG plot."""
    _update_eeg_generic(S.eeg_plot, S.eeg_data, S.selected_channels, 
                        use_secondary=False, update_time_label=True, 
                        amplitudes_dict=S.current_amplitudes)

def calculate_fixed_ranges(eeg_data, is_secondary=False):
    """Calculate fixed axis ranges for FFT and Hilbert plots based on full EEG."""
    if not eeg_data:
        return
    
    try:
        # Get EEG channels
        eeg_chs = [ch for ch, t in eeg_data.channel_types.items() if t == 'eeg'][:5]
        if not eeg_chs:
            return
        
        fft_values = []
        hilbert_values = []
        
        # Sample 10 segments across the recording for better estimation
        duration = eeg_data.duration_sec
        num_samples = min(10, int(duration / 5))  # Sample every ~5 seconds, max 10
        
        for i in range(num_samples):
            start = (duration / (num_samples + 1)) * (i + 1)
            try:
                seg_duration = min(5.0, duration - start)
                if seg_duration <= 0:
                    continue
                    
                data, times, _ = get_channel_data(eeg_data, eeg_chs, start, seg_duration)
                # Apply same processing as _update_fft_generic
                data = process_data(data, eeg_data.sfreq) * 1e6
                
                # FFT values
                freqs, fft_v = compute_fft(data, eeg_data.sfreq)
                mask = freqs <= 60
                fft_v = fft_v[:, mask]
                fft_values.append(np.max(fft_v))
                
                # Hilbert envelope max (use first channel)
                amp, _ = compute_hilbert(data[0])
                hilbert_values.extend([np.max(np.abs(data[0])), np.max(amp)])
            except Exception as e:
                pass
        
        # Use 95th percentile for robust estimation (handles outliers)
        if fft_values:
            max_fft = np.percentile(fft_values, 95) * 1.3
            # Ensure minimum sensible value
            max_fft = max(max_fft, 10)
        else:
            max_fft = 50
            
        if hilbert_values:
            max_hilbert = np.percentile(hilbert_values, 95) * 1.3
            max_hilbert = max(max_hilbert, 10)
        else:
            max_hilbert = 100
        
        # Store calculated ranges
        if is_secondary:
            S.fft_y_max2 = max_fft
            S.hilbert_amp_max2 = max_hilbert
        else:
            S.fft_y_max = max_fft
            S.hilbert_amp_max = max_hilbert
            
        print(f"Calculated ranges for EEG{'2' if is_secondary else '1'}: FFT={max_fft:.1f}, Hilbert={max_hilbert:.1f}")
    except Exception as e:
        print(f"Error calculating ranges: {e}")


def _update_fft_generic(fft_plot, eeg_data, channels, use_secondary=False):
    """Generic FFT update function for both EEG1 and EEG2."""
    if not fft_plot or not eeg_data or not channels:
        return
    
    valid_channels = [ch for ch in channels[:5] if ch in eeg_data.channel_types]
    if not valid_channels:
        return
    
    try:
        data, times, chs = get_channel_data(eeg_data, valid_channels, S.view_start, S.view_duration)
        data = process_data(data, eeg_data.sfreq) * 1e6
        freqs, fft_v = compute_fft(data, eeg_data.sfreq)
        mask = freqs <= 60
        freqs, fft_v = freqs[mask], fft_v[:, mask]
        
        colors = _SECONDARY_COLORS[:5] if use_secondary else _PRIMARY_COLORS[:5]
        fills = _SECONDARY_FFT_FILLS if use_secondary else _PRIMARY_FFT_FILLS
        
        # Use stored Y max if available, otherwise calculate from current data
        y_max_stored = S.fft_y_max2 if use_secondary else S.fft_y_max
        
        if y_max_stored is None or y_max_stored <= 0:
            # Calculate and store for this EEG
            if fft_v.size > 0:
                y_max = np.max(fft_v) * 1.3
                y_max = max(y_max, 1)
            else:
                y_max = 50
            # Store for future updates
            if use_secondary:
                S.fft_y_max2 = y_max
            else:
                S.fft_y_max = y_max
            print(f"[FFT] Calculated Y range: {y_max:.1f}")
        else:
            y_max = y_max_stored
        
        with fft_plot:
            fft_plot.figure.data = []
            for i, ch in enumerate(chs[:5]):
                fft_plot.figure.add_trace(go.Scatter(
                    x=freqs, y=fft_v[i], name=ch,
                    line=dict(color=colors[i % len(colors)], width=1.5),
                    fill='tozeroy', fillcolor=fills[i % len(fills)]
                ))
            # Fixed Y axis range (calculated once per EEG or after filter change)
            fft_plot.figure.update_layout(yaxis=dict(range=[0, y_max], fixedrange=True, autorange=False))
            fft_plot.update()
    except Exception as e:
        print(f"FFT{'2' if use_secondary else ''} error: {e}")


def update_fft():
    """Update primary FFT plot."""
    _update_fft_generic(S.fft_plot, S.eeg_data, S.selected_channels, use_secondary=False)

def _update_hilbert_generic(hilbert_plot, eeg_data, selected_channels, hilbert_channel, use_secondary=False):
    """Generic Hilbert update function for both EEG1 and EEG2."""
    if not hilbert_plot or not eeg_data or not selected_channels:
        return
    
    # Find valid channel
    ch = hilbert_channel if hilbert_channel in eeg_data.channel_types else None
    if not ch:
        for c in selected_channels:
            if c in eeg_data.channel_types:
                ch = c
                break
    if not ch:
        return
    
    try:
        data, times, _ = get_channel_data(eeg_data, [ch], S.view_start, S.view_duration)
        data = process_data(data, eeg_data.sfreq)[0] * 1e6
        amp, phase = compute_hilbert(data)
        
        # Color scheme
        if use_secondary:
            signal_color, envelope_color, phase_color = '#fb7185', '#f472b6', '#fda4af'
        else:
            signal_color, envelope_color, phase_color = THEME_SECONDARY, THEME_PRIMARY, THEME_WARN
        
        # Use pre-calculated fixed range, or compute if not available
        amp_max = S.hilbert_amp_max2 if use_secondary else S.hilbert_amp_max
        if amp_max is None or amp_max <= 0:
            amp_max = np.max(np.abs(data)) * 1.2 if data.size > 0 else 100
            # Store for next time
            if use_secondary:
                S.hilbert_amp_max2 = amp_max
            else:
                S.hilbert_amp_max = amp_max
        
        with hilbert_plot:
            hilbert_plot.figure.data = []
            hilbert_plot.figure.add_trace(go.Scatter(x=times, y=data, name='Signal', line=dict(color=signal_color, width=1)), row=1, col=1)
            hilbert_plot.figure.add_trace(go.Scatter(x=times, y=amp, name='Envelope', line=dict(color=envelope_color, width=2)), row=1, col=1)
            hilbert_plot.figure.add_trace(go.Scatter(x=times, y=-amp, showlegend=False, line=dict(color=envelope_color, width=2)), row=1, col=1)
            hilbert_plot.figure.add_trace(go.Scatter(x=times, y=phase, name='Phase', line=dict(color=phase_color, width=1)), row=2, col=1)
            # Apply fixed y-axis ranges (calculated once per EEG)
            hilbert_plot.figure.update_yaxes(range=[-amp_max, amp_max], row=1, col=1)
            hilbert_plot.figure.update_yaxes(range=[-np.pi * 1.1, np.pi * 1.1], row=2, col=1)  # Phase is always -π to π
            hilbert_plot.update()
    except Exception as e:
        print(f"Hilbert{'2' if use_secondary else ''} error: {e}")


def update_hilbert():
    """Update primary Hilbert plot."""
    ch = S.hilbert_channel if S.hilbert_channel in S.selected_channels else (S.selected_channels[0] if S.selected_channels else None)
    _update_hilbert_generic(S.hilbert_plot, S.eeg_data, S.selected_channels, ch, use_secondary=False)

def _update_brain_generic(brain_plot, eeg_data, selected_channels, amplitudes, use_secondary=False):
    """Generic brain topography update function for both EEG1 and EEG2."""
    if not brain_plot or not eeg_data:
        return
    
    try:
        # Get amplitudes - either from provided dict or compute fresh
        if amplitudes:
            amps = amplitudes
        else:
            # Compute amplitudes for provided channels
            valid_channels = [ch for ch in selected_channels if ch in eeg_data.channel_types]
            if valid_channels:
                data, times, chs = get_channel_data(eeg_data, valid_channels, S.view_start, S.view_duration)
                data = process_data(data, eeg_data.sfreq) * 1e6
                amps = {ch: np.sqrt(np.mean(data[i]**2)) for i, ch in enumerate(chs)}
            else:
                amps = {}
        
        if amps and selected_channels:
            vals = [amps.get(ch, 0) for ch in selected_channels if ch in amps]
            min_a, max_a = (min(vals), max(vals)) if vals else (0, 1)
            rng = max_a - min_a if max_a > min_a else 1
        else:
            min_a, rng = 0, 1
        
        sel_x, sel_y, sel_c, sel_t, sel_l = [], [], [], [], []
        uns_x, uns_y, uns_l = [], [], []
        
        valid_in_eeg = set(eeg_data.channel_types.keys()) if hasattr(eeg_data, 'channel_types') else set(selected_channels)
        
        for ch, (x, y) in ELECTRODE_POSITIONS.items():
            if ch in selected_channels and ch in valid_in_eeg:
                sel_x.append(x); sel_y.append(y); sel_l.append(ch)
                a = amps.get(ch, 0)
                sel_c.append((a - min_a) / rng if rng > 0 else 0.5)
                sel_t.append(f'{ch}<br>{a:.1f} µV')
            elif not use_secondary:  # Only show unselected for primary
                uns_x.append(x); uns_y.append(y); uns_l.append(ch)
        
        # Color scheme
        if use_secondary:
            primary_color = '#f472b6'
            colorscale = [[0, '#831843'], [0.25, '#be185d'], [0.5, '#f472b6'], [0.75, '#fda4af'], [1, '#ffe4e6']]
        else:
            primary_color = THEME_PRIMARY
            colorscale = [[0, '#0d47a1'], [0.25, '#00bcd4'], [0.5, '#00ff88'], [0.75, '#ffcc00'], [1, '#ff5722']]
        
        with brain_plot:
            brain_plot.figure.data = brain_plot.figure.data[:4]  # Keep head outline
            if uns_x:
                brain_plot.figure.add_trace(go.Scatter(x=uns_x, y=uns_y, mode='markers+text',
                    marker=dict(size=12, color='rgba(30,30,30,0.6)', line=dict(width=1, color='rgba(60,60,60,0.5)')),
                    text=uns_l, textposition='top center', textfont=dict(size=7, color='rgba(100,100,100,0.6)', family='JetBrains Mono'),
                    hoverinfo='text', hovertext=uns_l, showlegend=False))
            if sel_x:
                brain_plot.figure.add_trace(go.Scatter(x=sel_x, y=sel_y, mode='markers+text',
                    marker=dict(size=18, color=sel_c, colorscale=colorscale, cmin=0, cmax=1,
                               line=dict(width=2, color=primary_color), showscale=True,
                               colorbar=dict(title=dict(text='µV', font=dict(size=9, color=THEME_TEXT_DIM)),
                                           len=0.5, thickness=8, x=1.02, tickfont=dict(size=8, color=THEME_TEXT_DIM))),
                    text=sel_l, textposition='top center', textfont=dict(size=8, color=primary_color, family='JetBrains Mono'),
                    hoverinfo='text', hovertext=sel_t, showlegend=False))
            brain_plot.update()
    except Exception as e:
        print(f"Brain{'2' if use_secondary else ''} error: {e}")


def update_brain():
    """Update primary brain topography plot."""
    _update_brain_generic(S.brain_plot, S.eeg_data, S.selected_channels, S.current_amplitudes, use_secondary=False)

def clamp_view_to_both_eegs():
    """Clamp view_start to be valid for both EEGs in compare mode."""
    if S.eeg_data:
        max_start1 = max(0, S.eeg_data.duration_sec - S.view_duration)
        S.view_start = min(S.view_start, max_start1)
    
    if S.compare_mode and S.eeg_data2:
        max_start2 = max(0, S.eeg_data2.duration_sec - S.view_duration)
        S.view_start = min(S.view_start, max_start2)
    
    S.view_start = max(0, S.view_start)

def update_all():
    # Ensure view position is valid for both EEGs
    clamp_view_to_both_eegs()
    
    update_eeg()
    update_fft()
    update_hilbert()
    update_brain()
    # Update EEG 2 plots if in compare mode
    if S.compare_mode and S.eeg_data2:
        update_eeg2()
        update_fft2()
        update_hilbert2()
        update_brain2()
        update_fft_diff()  # Update FFT difference plot
        update_hilbert_diff()  # Update Hilbert difference plot

def update_eeg2():
    """Update EEG 2 plot with same time window and channels as EEG 1."""
    _update_eeg_generic(S.eeg_plot2, S.eeg_data2, S.selected_channels, use_secondary=True)

def update_fft2():
    """Update FFT 2 plot."""
    _update_fft_generic(S.fft_plot2, S.eeg_data2, S.selected_channels, use_secondary=True)

def update_fft_diff():
    """Update FFT difference plot (EEG1 - EEG2)."""
    if not S.fft_diff_plot or not S.compare_mode or not S.eeg_data or not S.eeg_data2:
        return
    
    if not S.selected_channels:
        return
    
    # Get channels that exist in both EEGs
    valid_channels = [ch for ch in S.selected_channels[:5] 
                     if ch in S.eeg_data.channel_types and ch in S.eeg_data2.channel_types]
    if not valid_channels:
        return
    
    try:
        # Get FFT for EEG1
        data1, times1, chs1 = get_channel_data(S.eeg_data, valid_channels, S.view_start, S.view_duration)
        data1 = process_data(data1, S.eeg_data.sfreq) * 1e6
        freqs1, fft_v1 = compute_fft(data1, S.eeg_data.sfreq)
        mask1 = freqs1 <= 60
        freqs1, fft_v1 = freqs1[mask1], fft_v1[:, mask1]
        
        # Get FFT for EEG2
        data2, times2, chs2 = get_channel_data(S.eeg_data2, valid_channels, S.view_start, S.view_duration)
        data2 = process_data(data2, S.eeg_data2.sfreq) * 1e6
        freqs2, fft_v2 = compute_fft(data2, S.eeg_data2.sfreq)
        mask2 = freqs2 <= 60
        freqs2, fft_v2 = freqs2[mask2], fft_v2[:, mask2]
        
        # Interpolate if sample rates differ (to align frequency bins)
        if len(freqs1) != len(freqs2):
            from scipy import interpolate
            # Use EEG1 frequencies as reference
            for i in range(len(valid_channels)):
                f_interp = interpolate.interp1d(freqs2, fft_v2[i], kind='linear', fill_value='extrapolate')
                fft_v2[i] = f_interp(freqs1)
            freqs2 = freqs1
        
        # Calculate difference: EEG1 - EEG2
        fft_diff = fft_v1 - fft_v2
        
        # Colors - gradient from cyan (EEG1 higher) to pink (EEG2 higher)
        diff_colors = ['#06b6d4', '#22d3ee', '#a78bfa', '#f472b6', '#ec4899']
        
        # Use stored Y max if available, otherwise calculate from current data
        if S.fft_diff_y_max is None or S.fft_diff_y_max <= 0:
            # Calculate and store
            max_abs = np.max(np.abs(fft_diff)) * 1.3 if fft_diff.size > 0 else 10
            max_abs = max(max_abs, 1)
            S.fft_diff_y_max = max_abs
            print(f"[FFT DIFF] Calculated Y range: ±{max_abs:.1f}")
        else:
            max_abs = S.fft_diff_y_max
        
        with S.fft_diff_plot:
            S.fft_diff_plot.figure.data = []
            
            for i, ch in enumerate(valid_channels):
                diff_data = fft_diff[i]
                
                # Create fill based on sign (positive = EEG1 higher, negative = EEG2 higher)
                S.fft_diff_plot.figure.add_trace(go.Scatter(
                    x=freqs1, y=diff_data, name=ch,
                    line=dict(color=diff_colors[i % len(diff_colors)], width=1.5),
                    fill='tozeroy',
                    fillcolor=f'rgba({114 if i % 2 == 0 else 244}, {182 if i % 2 == 0 else 114}, {244 if i % 2 == 0 else 182}, 0.1)',
                    hovertemplate=f'{ch}: %{{y:.2f}} µV<extra>EEG1-EEG2</extra>'
                ))
            
            # Symmetric Y axis around zero (fixed range)
            S.fft_diff_plot.figure.update_layout(
                yaxis=dict(range=[-max_abs, max_abs], fixedrange=True, autorange=False)
            )
            S.fft_diff_plot.update()
    except Exception as e:
        print(f"FFT diff error: {e}")

def update_hilbert2():
    """Update Hilbert 2 plot."""
    ch2 = S.hilbert_channel2 if S.hilbert_channel2 in S.selected_channels else (S.selected_channels[0] if S.selected_channels else None)
    _update_hilbert_generic(S.hilbert_plot2, S.eeg_data2, S.selected_channels, ch2, use_secondary=True)

def update_hilbert_diff():
    """Update Hilbert difference plot (EEG1 - EEG2)."""
    if not S.hilbert_diff_plot or not S.compare_mode or not S.eeg_data or not S.eeg_data2:
        return
    
    if not S.selected_channels:
        return
    
    # Use the channel selected for Hilbert 1
    ch = S.hilbert_channel if S.hilbert_channel in S.selected_channels else (S.selected_channels[0] if S.selected_channels else None)
    if not ch:
        return
    
    # Check channel exists in both EEGs
    if ch not in S.eeg_data.channel_types or ch not in S.eeg_data2.channel_types:
        return
    
    try:
        # Get Hilbert for EEG1
        data1, times1, _ = get_channel_data(S.eeg_data, [ch], S.view_start, S.view_duration)
        data1 = process_data(data1, S.eeg_data.sfreq)[0] * 1e6
        amp1, phase1 = compute_hilbert(data1)
        
        # Get Hilbert for EEG2
        data2, times2, _ = get_channel_data(S.eeg_data2, [ch], S.view_start, S.view_duration)
        data2 = process_data(data2, S.eeg_data2.sfreq)[0] * 1e6
        amp2, phase2 = compute_hilbert(data2)
        
        # Interpolate if lengths differ (different sample rates)
        if len(times1) != len(times2):
            from scipy import interpolate
            f_amp = interpolate.interp1d(times2, amp2, kind='linear', fill_value='extrapolate')
            f_phase = interpolate.interp1d(times2, phase2, kind='linear', fill_value='extrapolate')
            amp2 = f_amp(times1)
            phase2 = f_phase(times1)
            times2 = times1
        
        # Calculate differences
        amp_diff = amp1 - amp2
        phase_diff = phase1 - phase2
        # Wrap phase difference to [-π, π]
        phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))
        
        # Colors
        env_color = '#a78bfa'  # Purple
        phase_color = '#c4b5fd'  # Light purple
        
        # Use stored Y max if available, otherwise calculate
        if S.hilbert_diff_y_max is None or S.hilbert_diff_y_max <= 0:
            amp_max = np.max(np.abs(amp_diff)) * 1.3 if amp_diff.size > 0 else 10
            amp_max = max(amp_max, 1)
            S.hilbert_diff_y_max = amp_max
            print(f"[HILBERT DIFF] Calculated Y range: ±{amp_max:.1f}")
        else:
            amp_max = S.hilbert_diff_y_max
        
        with S.hilbert_diff_plot:
            S.hilbert_diff_plot.figure.data = []
            
            # Envelope difference
            S.hilbert_diff_plot.figure.add_trace(go.Scatter(
                x=times1, y=amp_diff, name=f'{ch} ΔEnv',
                line=dict(color=env_color, width=1.5),
                fill='tozeroy', fillcolor='rgba(167,139,250,0.15)',
                hovertemplate='%{y:.2f} µV<extra>EEG1-EEG2</extra>'
            ), row=1, col=1)
            
            # Phase difference
            S.hilbert_diff_plot.figure.add_trace(go.Scatter(
                x=times1, y=phase_diff, name=f'{ch} ΔPhase',
                line=dict(color=phase_color, width=1),
                hovertemplate='%{y:.3f} rad<extra>EEG1-EEG2</extra>'
            ), row=2, col=1)
            
            # Fixed Y axis ranges
            S.hilbert_diff_plot.figure.update_yaxes(range=[-amp_max, amp_max], row=1, col=1)
            S.hilbert_diff_plot.figure.update_yaxes(range=[-np.pi * 1.1, np.pi * 1.1], row=2, col=1)
            S.hilbert_diff_plot.update()
    except Exception as e:
        print(f"Hilbert diff error: {e}")

def update_brain2():
    """Update brain topography 2 plot."""
    _update_brain_generic(S.brain_plot2, S.eeg_data2, S.selected_channels, None, use_secondary=True)

# Channel management
def refresh_channels():
    if not S.channel_container:
        return
    S.channel_container.clear()
    with S.channel_container:
        if not S.eeg_data:
            ui.label('-- load file first --').style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.8rem;')
            return
        
        eeg_chs = [ch for ch, t in S.eeg_data.channel_types.items() if t == 'eeg']
        with ui.row().classes('gap-1 flex-wrap'):
            for ch in eeg_chs:
                is_sel = ch in S.selected_channels
                btn = ui.button(ch).props('dense size=sm')
                btn.classes('ch-btn' + (' ch-sel' if is_sel else ''))
                btn.on('click', lambda e, c=ch: toggle_ch(c))

def toggle_ch(ch):
    if ch in S.selected_channels:
        S.selected_channels.remove(ch)
        S.current_amplitudes.pop(ch, None)
    else:
        S.selected_channels.append(ch)
    # Reset FFT Y ranges to recalculate for new channel selection
    S.fft_y_max = None
    S.fft_y_max2 = None
    S.fft_diff_y_max = None
    S.hilbert_diff_y_max = None
    refresh_channels()
    refresh_hilbert_select()
    refresh_hilbert_select2()
    update_all()

def select_all_ch():
    if S.eeg_data:
        S.selected_channels = [ch for ch, t in S.eeg_data.channel_types.items() if t == 'eeg']
        # Reset FFT/Hilbert Y ranges to recalculate for new channel selection
        S.fft_y_max = None
        S.fft_y_max2 = None
        S.fft_diff_y_max = None
        S.hilbert_diff_y_max = None
        refresh_channels()
        refresh_hilbert_select()
        refresh_hilbert_select2()
        update_all()

def select_10_ch():
    if S.eeg_data:
        S.selected_channels = [ch for ch, t in S.eeg_data.channel_types.items() if t == 'eeg'][:10]
        # Reset FFT/Hilbert Y ranges to recalculate for new channel selection
        S.fft_y_max = None
        S.fft_y_max2 = None
        S.fft_diff_y_max = None
        S.hilbert_diff_y_max = None
        refresh_channels()
        refresh_hilbert_select()
        refresh_hilbert_select2()
        update_all()

def clear_ch():
    S.selected_channels = []
    S.current_amplitudes.clear()
    # Reset FFT/Hilbert Y ranges
    S.fft_y_max = None
    S.fft_y_max2 = None
    S.fft_diff_y_max = None
    S.hilbert_diff_y_max = None
    refresh_channels()
    update_all()

def refresh_hilbert_select():
    if not S.hilbert_select_container:
        return
    S.hilbert_select_container.clear()
    with S.hilbert_select_container:
        if S.selected_channels:
            ui.label('ch:').style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.7rem;')
            def on_sel(e):
                S.hilbert_channel = e.value
                update_hilbert()
                update_hilbert_diff()  # Also update diff when channel changes
            val = S.hilbert_channel if S.hilbert_channel in S.selected_channels else S.selected_channels[0]
            ui.select(options=S.selected_channels, value=val, on_change=on_sel).props('dense dark').classes('w-24')

def refresh_hilbert_select2():
    """Refresh Hilbert 2 channel selector."""
    if not S.hilbert_select_container2:
        return
    S.hilbert_select_container2.clear()
    with S.hilbert_select_container2:
        if S.selected_channels:
            ui.label('ch:').style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.7rem;')
            def on_sel2(e):
                S.hilbert_channel2 = e.value
                update_hilbert2()
            val = S.hilbert_channel2 if S.hilbert_channel2 in S.selected_channels else S.selected_channels[0]
            ui.select(options=S.selected_channels, value=val, on_change=on_sel2).props('dense dark').classes('w-24')

def refresh_info():
    if not S.info_container:
        return
    S.info_container.clear()
    with S.info_container:
        # EEG 1 info
        ui.label('EEG 1:').style(f'color:{THEME_PRIMARY}; font-size: 0.7rem; font-weight: bold;')
        if S.eeg_data:
            ui.label(S.eeg_data.filename).style(f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.75rem;')
            ui.label(f'{S.eeg_data.sfreq:.0f}Hz | {S.eeg_data.n_channels}ch | {S.eeg_data.duration_sec:.1f}s').style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.65rem;')
        else:
            ui.label('-- not loaded --').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
        
        ui.separator().classes('my-1')
        
        # EEG 2 info
        ui.label('EEG 2:').style(f'color:#f472b6; font-size: 0.7rem; font-weight: bold;')
        if S.eeg_data2:
            ui.label(S.eeg_data2.filename).style(f'color:#f472b6; font-family: JetBrains Mono; font-size: 0.75rem;')
            ui.label(f'{S.eeg_data2.sfreq:.0f}Hz | {S.eeg_data2.n_channels}ch | {S.eeg_data2.duration_sec:.1f}s').style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.65rem;')
        else:
            ui.label('-- not loaded --').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
        
        if S.compare_mode:
            ui.label('🔄 COMPARE MODE').style(f'color:{THEME_WARN}; font-size: 0.65rem; margin-top: 4px;')

# Navigation
def nav_start():
    S.view_start = 0
    S.is_playing = False
    update_all()

def nav_back():
    S.view_start = max(0, S.view_start - S.view_duration)
    update_all()

def nav_fwd():
    if S.eeg_data:
        S.view_start = min(S.eeg_data.duration_sec - S.view_duration, S.view_start + S.view_duration)
    update_all()

def nav_end():
    if S.eeg_data:
        S.view_start = S.eeg_data.duration_sec - S.view_duration
    S.is_playing = False
    update_all()

def set_win(d):
    S.view_duration = d
    update_all()

async def toggle_play():
    S.is_playing = not S.is_playing
    S.playback_reverse = False
    if S.is_playing:
        await run_playback()

async def toggle_play_reverse():
    S.is_playing = not S.is_playing
    S.playback_reverse = True
    if S.is_playing:
        await run_playback()

async def toggle_play_fast():
    S.is_playing = not S.is_playing
    S.playback_reverse = False
    old_speed = S.playback_speed
    S.playback_speed = 4.0
    if S.is_playing:
        await run_playback()
    S.playback_speed = old_speed

async def run_playback():
    while S.is_playing and S.eeg_data:
        step = S.view_duration * 0.08 * S.playback_speed
        if S.playback_reverse:
            S.view_start = max(0, S.view_start - step)
            if S.view_start <= 0:
                S.is_playing = False
                break
        else:
            S.view_start = min(S.eeg_data.duration_sec - S.view_duration, S.view_start + step)
            if S.view_start >= S.eeg_data.duration_sec - S.view_duration:
                S.is_playing = False
                break
        update_all()
        await asyncio.sleep(0.1)

# Epochs
def gen_epochs():
    if not S.eeg_data or not S.selected_channels:
        ui.notify('Load data first', type='warning')
        return 0
    n = S.eeg_data.n_samples // int(S.epoch_duration * S.eeg_data.sfreq)
    S.epochs = []
    for i in range(n):
        st = (i * int(S.epoch_duration * S.eeg_data.sfreq)) / S.eeg_data.sfreq
        data, times, chs = get_channel_data(S.eeg_data, S.selected_channels, st, S.epoch_duration)
        S.epochs.append({'data': process_data(data, S.eeg_data.sfreq), 'times': times, 'channels': chs, 'start': st})
    ui.notify(f'{n} epochs generated', type='positive')
    return n

def save_epochs():
    if not S.epochs:
        ui.notify('Generate epochs first', type='warning')
        return
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    nm = S.eeg_data.filename.split('.')[0]
    out = Path(S.eeg_data.filepath).parent / 'epochs_output'
    out.mkdir(exist_ok=True)
    np.save(out / f'{nm}_epochs_{ts}.npy', np.array([e['data'] for e in S.epochs]))
    fft_d = [compute_fft(e['data']*1e6, S.eeg_data.sfreq)[1] for e in S.epochs]
    np.save(out / f'{nm}_fft_{ts}.npy', np.array(fft_d))
    pd.DataFrame([{'file': S.eeg_data.filename, 'sfreq': S.eeg_data.sfreq, 'n': len(S.epochs), 'dur': S.epoch_duration}]).to_csv(out / f'{nm}_meta_{ts}.csv', index=False)
    ui.notify(f'Saved to {out}', type='positive')

# =============================================================================
# PIPELINE PAGE
# =============================================================================

# PipelineState (PS) imported from app.state

PIPELINE_DIR = Path(__file__).parent.parent.parent.parent / "pipeline"
PIPELINE_OUTPUTS = Path(__file__).parent.parent.parent / "pipeline_outputs"
RESULTS_BASE = Path(__file__).parent.parent / "fwd-inv-stc"
DEFAULT_INPUT_DIR = Path(__file__).parent.parent.parent.parent / "EEG_CLEAN"

# Update main page header to include navigation
@ui.page('/')
def main_with_nav():
    ui.add_head_html(f'<style>{STYLE}</style>')
    
    # Global header with system monitor (CPU/RAM/GPU)
    render_global_header('viewer')
    
    # Rest of the main page content (call original main function logic)
    main_content()


def main_content():
    """Main page content - separated for reuse"""
    with ui.row().classes('w-full p-4 gap-4').style('min-height: calc(100vh - 50px);'):
        
        # LEFT SIDEBAR - FULL WIDTH TO CONTENT
        with ui.column().classes('gap-4 shrink-0').style('width: 380px;'):
            
            # FILE BROWSER
            with ui.card().classes('dark-card p-4 w-full'):
                ui.label('// FILE_BROWSER').classes('terminal-header')
                
                # EEG slot selector
                with ui.row().classes('items-center gap-2 mb-2'):
                    ui.label('Load to:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                    eeg_slot_select = ui.toggle(['EEG 1', 'EEG 2'], value='EEG 1').props('dense')
                    
                    def toggle_compare():
                        S.compare_mode = not S.compare_mode
                        update_all()
                        ui.notify(f"Compare mode: {'ON' if S.compare_mode else 'OFF'}", type='info')
                    
                    compare_btn = ui.button('Compare', on_click=toggle_compare, icon='compare').props('dense flat size=sm')
                    
                    def clear_eeg2():
                        S.eeg_data2 = None
                        S.compare_mode = False
                        # Reset axis ranges so they recalculate
                        S.fft_y_max2 = None
                        S.fft_diff_y_max = None
                        S.hilbert_amp_max2 = None
                        S.hilbert_diff_y_max = None
                        update_all()
                        refresh_info()
                        ui.notify('EEG 2 cleared', type='info')
                    
                    ui.button('Clear 2', on_click=clear_eeg2, icon='close').props('dense flat size=sm')
                
                async def load_file(fp, slot=None):
                    try:
                        target_slot = slot or eeg_slot_select.value
                        loaded = await asyncio.get_event_loop().run_in_executor(None, load_eeg_file, Path(fp))
                        
                        if target_slot == 'EEG 2':
                            S.eeg_data2 = loaded
                            ui.notify(f'EEG 2: {loaded.filename}', type='positive')
                            S.compare_mode = True
                            # Calculate fixed axis ranges for EEG 2
                            S.fft_diff_y_max = None  # Reset diff ranges for new EEG2
                            S.hilbert_diff_y_max = None
                            calculate_fixed_ranges(loaded, is_secondary=True)
                            refresh_hilbert_select2()  # Refresh channel selector for Hilbert 2
                        else:
                            S.eeg_data = loaded
                            eeg_chs = [ch for ch, t in loaded.channel_types.items() if t == 'eeg']
                            S.selected_channels = eeg_chs[:10]
                            S.hilbert_channel = eeg_chs[0] if eeg_chs else ""
                            S.view_start = 0
                            S.epochs = []
                            S.current_amplitudes = {}
                            # Reset axis ranges so they recalculate for new EEG
                            S.fft_y_max = None
                            S.fft_diff_y_max = None
                            S.hilbert_amp_max = None
                            S.hilbert_diff_y_max = None
                            ui.notify(f'EEG 1: {loaded.filename}', type='positive')
                            refresh_channels()
                            refresh_hilbert_select()
                            # Calculate fixed axis ranges for EEG 1
                            calculate_fixed_ranges(loaded, is_secondary=False)
                        
                        refresh_info()
                        update_all()
                    except Exception as e:
                        ui.notify(f'Error: {e}', type='negative')
                
                # Path input for custom directories
                with ui.row().classes('w-full gap-2 mb-3 items-center'):
                    path_input = ui.input(
                        placeholder='Enter path or click Browse...'
                    ).props('dense outlined').classes('flex-1').style('font-size: 0.8rem;')
                    
                    custom_paths = []
                    
                    async def browse_folder():
                        """Open folder picker dialog."""
                        with ui.dialog() as dialog, ui.card().classes('p-4'):
                            ui.label('Select Folder').classes('text-lg font-bold mb-3')
                            
                            folder_input = ui.input(
                                value=str(Path.home()),
                                label='Folder Path'
                            ).props('outlined').classes('w-full mb-3')
                            
                            ui.label('Quick Access:').classes('text-xs opacity-50 mb-2')
                            with ui.row().classes('gap-2 flex-wrap mb-3'):
                                common_paths = [
                                    ('Home', str(Path.home())),
                                    ('EEG RAW', str(EEG_RAW_DIR)),
                                    ('EEG CLEAN', str(EEG_CLEAN_DIR)),
                                ]
                                for name, path in common_paths:
                                    if Path(path).exists():
                                        ui.button(name, on_click=lambda p=path: folder_input.set_value(p)).props('dense size=sm')
                            
                            with ui.row().classes('gap-2 justify-end'):
                                ui.button('Cancel', on_click=dialog.close).props('flat')
                                
                                def select_folder():
                                    path_input.set_value(folder_input.value)
                                    dialog.close()
                                
                                ui.button('Select', on_click=select_folder).props('color=primary')
                        
                        dialog.open()
                    
                    ui.button(icon='folder_open', on_click=browse_folder).props(
                        'flat dense'
                    ).tooltip('Browse folder')
                    
                    def add_path():
                        path = path_input.value
                        if path and path not in custom_paths:
                            if Path(path).exists() and Path(path).is_dir():
                                custom_paths.append(path)
                                refresh_file_list()
                                ui.notify(f'Added: {path}', type='positive')
                            else:
                                ui.notify('Invalid path or not a directory', type='warning')
                    
                    ui.button(icon='add', on_click=add_path).props(
                        'flat dense color=green'
                    ).tooltip('Add path to list')
                    
                    ui.button(icon='refresh', on_click=lambda: refresh_file_list()).props(
                        'flat dense'
                    ).tooltip('Refresh file list')
                
                files_container = ui.scroll_area().classes('w-full').style('height: 250px;')
                
                def refresh_file_list():
                    files_container.clear()
                    with files_container:
                        # Default directories
                        for lbl, files in [('raw/', scan_eeg_directory(EEG_RAW_DIR)), ('clean/', scan_eeg_directory(EEG_CLEAN_DIR))]:
                            if files:
                                ui.label(f'├─ {lbl}').style(f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mt-3 mb-2')
                                conds = {}
                                for f in files:
                                    conds.setdefault(f['condition'], []).append(f)
                                for cond, cfs in conds.items():
                                    with ui.expansion(f'{cond} ({len(cfs)} files)').classes('w-full'):
                                        for f in cfs:
                                            with ui.row().classes('file-item items-center w-full gap-3'):
                                                ui.icon('description', size='sm').classes('opacity-60')
                                                with ui.column().classes('flex-1'):
                                                    ui.label(f['name']).classes('text-sm font-medium')
                                                    ui.label(f"{f['size_mb']:.1f} MB").classes('text-xs opacity-50')
                                                ui.button(icon='play_arrow', on_click=lambda e, p=f['path']: load_file(p)).props('flat dense size=sm color=red')
                        
                        # Custom paths
                        for path in custom_paths:
                            files = scan_eeg_directory(path)
                            if files:
                                ui.label(f'├─ 📁 {Path(path).name}').style(f'color:{THEME_SECONDARY}; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mt-3 mb-2')
                                conds = {}
                                for f in files:
                                    conds.setdefault(f['condition'], []).append(f)
                                for cond, cfs in conds.items():
                                    with ui.expansion(f'{cond} ({len(cfs)} files)').classes('w-full'):
                                        for f in cfs:
                                            with ui.row().classes('file-item items-center w-full gap-3'):
                                                ui.icon('description', size='sm').classes('opacity-60')
                                                with ui.column().classes('flex-1'):
                                                    ui.label(f['name']).classes('text-sm font-medium')
                                                    ui.label(f"{f['size_mb']:.1f} MB").classes('text-xs opacity-50')
                                                ui.button(icon='play_arrow', on_click=lambda e, p=f['path']: load_file(p)).props('flat dense size=sm color=red')
                        
                        # Path from input
                        if path_input.value and path_input.value not in custom_paths:
                            input_path = path_input.value
                            if Path(input_path).exists():
                                files = scan_eeg_directory(input_path)
                                if files:
                                    ui.label(f'├─ 📂 {Path(input_path).name}').style(f'color:{THEME_WARN}; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mt-3 mb-2')
                                    conds = {}
                                    for f in files:
                                        conds.setdefault(f['condition'], []).append(f)
                                    for cond, cfs in conds.items():
                                        with ui.expansion(f'{cond} ({len(cfs)} files)').classes('w-full'):
                                            for f in cfs:
                                                with ui.row().classes('file-item items-center w-full gap-3'):
                                                    ui.icon('description', size='sm').classes('opacity-60')
                                                    with ui.column().classes('flex-1'):
                                                        ui.label(f['name']).classes('text-sm font-medium')
                                                        ui.label(f"{f['size_mb']:.1f} MB").classes('text-xs opacity-50')
                                                    ui.button(icon='play_arrow', on_click=lambda e, p=f['path']: load_file(p)).props('flat dense size=sm color=red')
                
                refresh_file_list()
            
            # FILE INFO
            with ui.card().classes('dark-card p-4 w-full'):
                ui.label('// FILE_INFO').classes('terminal-header')
                S.info_container = ui.column().classes('w-full gap-1')
                refresh_info()
            
            # CHANNEL SELECTION
            with ui.card().classes('dark-card p-4 w-full'):
                ui.label('// CHANNELS').classes('terminal-header')
                with ui.row().classes('gap-2 mb-3'):
                    ui.button('ALL', on_click=select_all_ch).props('dense')
                    ui.button('10', on_click=select_10_ch).props('dense')
                    ui.button('CLEAR', on_click=clear_ch).props('dense outline')
                S.channel_container = ui.column().classes('w-full')
                refresh_channels()
        
        # MAIN CONTENT - min-w-0 allows flex children to shrink properly
        with ui.column().classes('flex-1 gap-3 min-w-0 overflow-hidden'):
            
            # FILTERS
            with ui.card().classes('dark-card p-3'):
                with ui.row().classes('items-center gap-4 flex-wrap'):
                    ui.label('FILTERS:').style(f'color:{THEME_SECONDARY}; font-family: JetBrains Mono; font-size: 0.75rem; letter-spacing: 1px;')
                    notch_sw = ui.switch('Notch', value=S.notch_enabled).props('dense')
                    notch_hz = ui.number(value=S.notch_freq, min=45, max=65).props('dense').classes('w-16')
                    ui.label('Hz').classes('text-xs opacity-50')
                    ui.separator().props('vertical')
                    bp_sw = ui.switch('Bandpass', value=S.bandpass_enabled).props('dense')
                    bp_lo = ui.number(value=S.bandpass_low, min=0.1, max=100).props('dense').classes('w-16')
                    bp_hi = ui.number(value=S.bandpass_high, min=1, max=200).props('dense').classes('w-16')
                    ui.label('Hz').classes('text-xs opacity-50')
                    def apply_filt():
                        S.notch_enabled, S.notch_freq = notch_sw.value, notch_hz.value or 50
                        S.bandpass_enabled, S.bandpass_low, S.bandpass_high = bp_sw.value, bp_lo.value or 1, bp_hi.value or 45
                        # Reset FFT/Hilbert Y ranges to recalculate with new filter settings
                        S.fft_y_max = None
                        S.fft_y_max2 = None
                        S.fft_diff_y_max = None
                        S.hilbert_amp_max = None
                        S.hilbert_amp_max2 = None
                        S.hilbert_diff_y_max = None
                        update_all()
                        ui.notify('Filters applied', type='positive')
                    ui.button('Apply', on_click=apply_filt, icon='check').props('dense')
            
            # TOP ROW: EEG (with comparison support) - side by side, equal width
            with ui.row().classes('gap-3 w-full flex-nowrap'):
                # EEG 1
                with ui.card().classes('dark-card p-3 flex-1 min-w-0'):
                    with ui.row().classes('items-center justify-between mb-1'):
                        ui.label('▌EEG 1').style(f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.8rem; letter-spacing: 1px;')
                        with ui.row().classes('gap-1'):
                            ui.button('-', on_click=lambda: (setattr(S, 'scale_factor', max(0.2, S.scale_factor*0.7)), update_eeg())).props('dense flat size=xs')
                            ui.button('1x', on_click=lambda: (setattr(S, 'scale_factor', 1.0), update_eeg())).props('dense flat size=xs')
                            ui.button('+', on_click=lambda: (setattr(S, 'scale_factor', min(5, S.scale_factor*1.4)), update_eeg())).props('dense flat size=xs')
                    S.eeg_plot = ui.plotly(make_eeg_fig()).classes('w-full')
                
                # EEG 2 (comparison) - same width as EEG 1
                with ui.card().classes('dark-card p-3 flex-1 min-w-0').bind_visibility_from(S, 'compare_mode'):
                    ui.label('▌EEG 2').style(f'color:#f472b6; font-family: JetBrains Mono; font-size: 0.8rem; letter-spacing: 1px;').classes('mb-1')
                    S.eeg_plot2 = ui.plotly(make_eeg_fig(use_eeg2=True)).classes('w-full')
            
            # Navigation (shared) - Layout like cleaner
            with ui.card().classes('dark-card p-2'):
                with ui.row().classes('items-center justify-center gap-1'):
                    # Go to start
                    ui.button(icon='first_page', on_click=nav_start).props('flat dense round size=sm').tooltip('Go to start')
                    
                    # Play reverse
                    ui.button(icon='fast_rewind', on_click=toggle_play_reverse).props('flat dense round size=sm').tooltip('Play reverse')
                    
                    # Previous segment
                    ui.button(icon='chevron_left', on_click=nav_back).props('flat dense round size=sm').tooltip('Previous segment')
                    
                    # Play/Pause
                    ui.button(icon='play_arrow', on_click=toggle_play).props('flat dense round size=sm color=red').tooltip('Play/Pause')
                    
                    # Next segment
                    ui.button(icon='chevron_right', on_click=nav_fwd).props('flat dense round size=sm').tooltip('Next segment')
                    
                    # Fast forward
                    ui.button(icon='fast_forward', on_click=toggle_play_fast).props('flat dense round size=sm').tooltip('Fast forward (4x)')
                    
                    # Go to end
                    ui.button(icon='last_page', on_click=nav_end).props('flat dense round size=sm').tooltip('Go to end')
                    
                    ui.separator().props('vertical').classes('mx-2')
                    
                    # Window duration buttons
                    ui.button('2s', on_click=lambda: set_win(2)).props('dense outline size=xs')
                    ui.button('5s', on_click=lambda: set_win(5)).props('dense outline size=xs')
                    ui.button('10s', on_click=lambda: set_win(10)).props('dense outline size=xs')
                    ui.button('20s', on_click=lambda: set_win(20)).props('dense outline size=xs')
                    
                    ui.separator().props('vertical').classes('mx-2')
                    
                    S.time_label = ui.label('0:00.0 / 0:00.0').classes('text-sm font-mono opacity-70')
            
            # TOPOGRAPHY ROW - side by side, equal width
            with ui.row().classes('gap-3 w-full flex-nowrap'):
                with ui.card().classes('dark-card p-3 flex-1 min-w-0'):
                    ui.label('▌TOPO 1').style(f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-1')
                    S.brain_plot = ui.plotly(make_brain_fig()).classes('w-full')
                
                with ui.card().classes('dark-card p-3 flex-1 min-w-0').bind_visibility_from(S, 'compare_mode'):
                    ui.label('▌TOPO 2').style(f'color:#f472b6; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-1')
                    S.brain_plot2 = ui.plotly(make_brain_fig(use_eeg2=True)).classes('w-full')
            
            # FFT ROW - side by side, equal width
            with ui.row().classes('gap-3 w-full flex-nowrap'):
                with ui.card().classes('dark-card p-3 flex-1 min-w-0'):
                    ui.label('▌FFT 1').style(f'color:{THEME_SECONDARY}; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-1')
                    S.fft_plot = ui.plotly(make_fft_fig()).classes('w-full')
                
                with ui.card().classes('dark-card p-3 flex-1 min-w-0').bind_visibility_from(S, 'compare_mode'):
                    ui.label('▌FFT 2').style(f'color:#f472b6; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-1')
                    S.fft_plot2 = ui.plotly(make_fft_fig(use_eeg2=True)).classes('w-full')
            
            # FFT DIFFERENCE ROW - only visible in compare mode, full width
            with ui.card().classes('dark-card p-3 w-full').bind_visibility_from(S, 'compare_mode'):
                with ui.row().classes('items-center gap-2 mb-1'):
                    ui.label('▌FFT DIFF').style('color:#a78bfa; font-family: JetBrains Mono; font-size: 0.8rem;')
                    ui.label('(EEG1 − EEG2)').style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.7rem;')
                    ui.label('↑ positive = EEG1 higher').style('color:#06b6d4; font-size: 0.65rem; margin-left: auto;')
                    ui.label('↓ negative = EEG2 higher').style('color:#f472b6; font-size: 0.65rem;')
                S.fft_diff_plot = ui.plotly(make_fft_diff_fig()).classes('w-full')
            
            # HILBERT ROW - side by side, equal width
            with ui.row().classes('gap-3 w-full flex-nowrap'):
                with ui.card().classes('dark-card p-3 flex-1 min-w-0'):
                    with ui.row().classes('items-center gap-2 mb-1'):
                        ui.label('▌HILBERT 1').style(f'color:{THEME_WARN}; font-family: JetBrains Mono; font-size: 0.8rem;')
                        S.hilbert_select_container = ui.row().classes('items-center gap-1')
                        refresh_hilbert_select()
                    S.hilbert_plot = ui.plotly(make_hilbert_fig()).classes('w-full')
                
                with ui.card().classes('dark-card p-3 flex-1 min-w-0').bind_visibility_from(S, 'compare_mode'):
                    with ui.row().classes('items-center gap-2 mb-1'):
                        ui.label('▌HILBERT 2').style(f'color:#f472b6; font-family: JetBrains Mono; font-size: 0.8rem;')
                        S.hilbert_select_container2 = ui.row().classes('items-center gap-1')
                        refresh_hilbert_select2()
                    S.hilbert_plot2 = ui.plotly(make_hilbert_fig(use_eeg2=True)).classes('w-full')
            
            # HILBERT DIFFERENCE ROW - only visible in compare mode, full width
            with ui.card().classes('dark-card p-3 w-full').bind_visibility_from(S, 'compare_mode'):
                with ui.row().classes('items-center gap-2 mb-1'):
                    ui.label('▌HILBERT DIFF').style('color:#a78bfa; font-family: JetBrains Mono; font-size: 0.8rem;')
                    ui.label('(EEG1 − EEG2)').style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.7rem;')
                    ui.label('↑ positive = EEG1 higher').style('color:#06b6d4; font-size: 0.65rem; margin-left: auto;')
                    ui.label('↓ negative = EEG2 higher').style('color:#f472b6; font-size: 0.65rem;')
                S.hilbert_diff_plot = ui.plotly(make_hilbert_diff_fig()).classes('w-full')
            
            # EPOCHS
            with ui.card().classes('dark-card p-3'):
                with ui.row().classes('items-center gap-4'):
                    ui.label('DATASET:').style(f'color:{THEME_SECONDARY}; font-family: JetBrains Mono; font-size: 0.75rem; letter-spacing: 1px;')
                    ui.label('Epoch:').classes('text-xs opacity-50')
                    ep_dur = ui.number(value=S.epoch_duration, min=0.5, max=30, step=0.5).props('dense').classes('w-20')
                    ep_dur.on('update:model-value', lambda e: setattr(S, 'epoch_duration', e.args or 2))
                    ui.label('sec').classes('text-xs opacity-50')
                    ep_lbl = ui.label('Epochs: 0').classes('text-sm')
                    ui.button('Generate', on_click=lambda: ep_lbl.set_text(f'Epochs: {gen_epochs()}'), icon='auto_awesome').props('dense')
                    ui.button('Save Dataset', on_click=save_epochs, icon='save').props('dense color=green')

