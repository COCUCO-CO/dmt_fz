#!/usr/bin/env python3
"""
EEG Viewer - Powerful EEG Processing Interface
"""
import asyncio
import os
import sys
import re
from pathlib import Path
from nicegui import ui
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import pandas as pd
from datetime import datetime

from config import (
    EEG_RAW_DIR, EEG_CLEAN_DIR, FREQ_BANDS,
    THEME_BG, THEME_CARD, THEME_BORDER, THEME_PRIMARY, THEME_SECONDARY,
    THEME_WARN, THEME_ERROR, THEME_TEXT, THEME_TEXT_DIM, SIGNAL_COLORS
)
from eeg_loader import load_eeg_file, get_channel_data, scan_eeg_directory, EEGData

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

# Global state
class State:
    def __init__(self):
        self.eeg_data = None
        self.eeg_data2 = None  # Second EEG for comparison
        self.compare_mode = False  # Whether comparison mode is active
        self.selected_channels = []
        self.view_start = 0.0
        self.view_duration = 5.0
        self.is_playing = False
        self.scale_factor = 1.0
        self.notch_freq = 50.0
        self.notch_enabled = False
        self.bandpass_low = 1.0
        self.bandpass_high = 45.0
        self.bandpass_enabled = False
        self.hilbert_channel = ""
        self.epoch_duration = 2.0
        self.epochs = []
        self.current_amplitudes = {}
        # UI references
        self.eeg_plot = None
        self.eeg_plot2 = None  # Second EEG plot
        self.fft_plot = None
        self.fft_plot2 = None  # Second FFT plot
        self.hilbert_plot = None
        self.hilbert_plot2 = None  # Second Hilbert plot
        self.brain_plot = None
        self.brain_plot2 = None  # Second brain plot
        self.time_label = None
        self.info_container = None
        self.info_container2 = None  # Second info container
        self.channel_container = None
        self.hilbert_select_container = None

S = State()

# Signal processing
def apply_notch(data, sfreq, freq=50.0):
    b, a = signal.iirnotch(freq, 30, sfreq)
    return signal.filtfilt(b, a, data, axis=-1)

def apply_bandpass(data, sfreq, low, high):
    nyq = sfreq / 2
    b, a = signal.butter(4, [max(low/nyq, 0.001), min(high/nyq, 0.99)], btype='band')
    return signal.filtfilt(b, a, data, axis=-1)

def compute_fft(data, sfreq):
    n = data.shape[-1]
    freqs = fftfreq(n, 1/sfreq)[:n//2]
    fft_vals = np.abs(fft(data, axis=-1))[..., :n//2] / n * 2
    return freqs, fft_vals

def compute_hilbert(data):
    analytic = signal.hilbert(data, axis=-1)
    return np.abs(analytic), np.angle(analytic)

def process_data(data, sfreq):
    out = data.copy()
    if S.notch_enabled:
        out = apply_notch(out, sfreq, S.notch_freq)
    if S.bandpass_enabled:
        out = apply_bandpass(out, sfreq, S.bandpass_low, S.bandpass_high)
    return out

# Styles - Konsole/Terminal aesthetic
STYLE = f"""
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {{
    --bg: {THEME_BG};
    --card: {THEME_CARD};
    --border: {THEME_BORDER};
    --primary: {THEME_PRIMARY};
    --secondary: {THEME_SECONDARY};
    --text: {THEME_TEXT};
    --text-dim: {THEME_TEXT_DIM};
}}

* {{ scrollbar-width: thin; scrollbar-color: var(--primary) var(--bg); }}
*::-webkit-scrollbar {{ width: 6px; height: 6px; }}
*::-webkit-scrollbar-track {{ background: var(--bg); }}
*::-webkit-scrollbar-thumb {{ background: var(--primary); border-radius: 3px; }}

body {{
    background: var(--bg) !important;
    font-family: 'JetBrains Mono', 'IBM Plex Mono', 'SF Mono', monospace !important;
    color: var(--text) !important;
}}

.nicegui-content {{ background: transparent !important; }}

.dark-card {{
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4) !important;
}}

.accent-text {{ color: var(--primary) !important; text-shadow: 0 0 10px rgba(0, 255, 136, 0.3); }}
.secondary-text {{ color: var(--secondary) !important; }}

.file-item {{
    transition: all 0.15s ease;
    cursor: pointer;
    padding: 6px 10px;
    border-radius: 2px;
    margin: 2px 0;
    border-left: 2px solid transparent;
}}
.file-item:hover {{
    background: rgba(0, 255, 136, 0.1) !important;
    border-left-color: var(--primary);
}}

.ch-btn {{
    font-size: 0.75rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    padding: 3px 8px !important;
    min-width: 42px !important;
    margin: 2px !important;
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--text-dim) !important;
    border-radius: 2px !important;
    transition: all 0.15s ease !important;
}}
.ch-btn:hover {{
    border-color: var(--primary) !important;
    color: var(--primary) !important;
}}
.ch-sel {{
    background: rgba(0, 255, 136, 0.15) !important;
    border-color: var(--primary) !important;
    color: var(--primary) !important;
    box-shadow: 0 0 8px rgba(0, 255, 136, 0.2) !important;
}}

.q-btn {{
    font-family: 'JetBrains Mono', monospace !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
}}

.q-field__control {{ background: rgba(0, 0, 0, 0.3) !important; border-radius: 2px !important; }}
.q-field--dark .q-field__control {{ border: 1px solid var(--border) !important; }}

.terminal-header {{
    font-size: 0.7rem;
    color: var(--text-dim);
    letter-spacing: 0.5px;
    text-transform: uppercase;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 8px;
}}

.terminal-label {{
    font-size: 0.85rem;
    color: var(--text);
}}

.glow-text {{ text-shadow: 0 0 15px rgba(0, 255, 136, 0.5); }}

/* Status indicators */
@keyframes pulse-orange {{
    0%, 100% {{ opacity: 1; box-shadow: 0 0 8px #f59e0b; }}
    50% {{ opacity: 0.4; box-shadow: 0 0 2px #f59e0b; }}
}}

.status-idle {{
    width: 12px; height: 12px; border-radius: 50%;
    background: #6b7280; 
    display: inline-block;
}}

.status-training {{
    width: 12px; height: 12px; border-radius: 50%;
    background: #f59e0b;
    animation: pulse-orange 1s ease-in-out infinite;
    display: inline-block;
}}

.status-completed {{
    width: 12px; height: 12px; border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 8px #10b981;
    display: inline-block;
}}

.status-error {{
    width: 12px; height: 12px; border-radius: 50%;
    background: #ef4444;
    box-shadow: 0 0 8px #ef4444;
    display: inline-block;
}}
"""

# Plot creation with FIXED axes - Terminal style
PLOT_BG = 'rgba(8,8,8,1)'
PLOT_GRID = 'rgba(0,255,136,0.08)'
PLOT_GRID_MINOR = 'rgba(0,255,136,0.03)'

def make_eeg_fig(use_eeg2=False):
    """Create EEG figure. use_eeg2=True uses S.eeg_data2 instead of S.eeg_data."""
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
    """Create FFT figure. use_eeg2=True uses S.eeg_data2 instead of S.eeg_data."""
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

def make_hilbert_fig(use_eeg2=False):
    """Create Hilbert figure. use_eeg2=True uses S.eeg_data2 instead of S.eeg_data."""
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

def make_brain_fig(use_eeg2=False):
    """Create brain topography figure. use_eeg2=True uses S.eeg_data2 instead of S.eeg_data."""
    fig = go.Figure()
    theta = np.linspace(0, 2*np.pi, 100)
    # Head outline - use different color for EEG 2
    head_color = 'rgba(244,114,182,0.5)' if use_eeg2 else 'rgba(0,255,136,0.5)'
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

# Update functions
def update_eeg():
    if not S.eeg_plot or not S.eeg_data or not S.selected_channels:
        return
    try:
        data, times, chs = get_channel_data(S.eeg_data, S.selected_channels, S.view_start, S.view_duration)
        data = process_data(data, S.eeg_data.sfreq) * 1e6
        n = len(chs)
        
        # Normalize and store amplitudes
        norm = np.zeros_like(data)
        for i in range(n):
            std = np.std(data[i])
            S.current_amplitudes[chs[i]] = np.sqrt(np.mean(data[i]**2))
            norm[i] = data[i] / (std * 3) if std > 0 else data[i]
        
        spacing = 2.0 * S.scale_factor
        colors = SIGNAL_COLORS
        
        # Fixed Y-axis range based on number of channels
        y_min = -spacing
        y_max = (n) * spacing
        
        with S.eeg_plot:
            S.eeg_plot.figure.data = []
            for i in range(n):
                off = (n - 1 - i) * spacing
                S.eeg_plot.figure.add_trace(go.Scatter(
                    x=times, y=norm[i] + off, name=chs[i], line=dict(color=colors[i % len(colors)], width=1),
                    hovertemplate=f'{chs[i]}: %{{customdata:.1f}} µV<extra></extra>', customdata=data[i]
                ))
            S.eeg_plot.figure.update_layout(
                yaxis=dict(
                    tickmode='array', 
                    tickvals=[(n-1-i)*spacing for i in range(n)], 
                    ticktext=chs,
                    range=[y_min, y_max],  # Fixed range
                    fixedrange=True
                )
            )
            S.eeg_plot.update()
        
        if S.time_label and S.eeg_data:
            m, s = int(S.view_start // 60), S.view_start % 60
            tm, ts = int(S.eeg_data.duration_sec // 60), S.eeg_data.duration_sec % 60
            S.time_label.set_text(f'{m}:{s:04.1f} / {tm}:{ts:04.1f}')
    except Exception as e:
        print(f"EEG error: {e}")

def update_fft():
    if not S.fft_plot or not S.eeg_data or not S.selected_channels:
        return
    try:
        data, times, chs = get_channel_data(S.eeg_data, S.selected_channels[:5], S.view_start, S.view_duration)
        data = process_data(data, S.eeg_data.sfreq) * 1e6
        freqs, fft_v = compute_fft(data, S.eeg_data.sfreq)
        mask = freqs <= 60
        freqs, fft_v = freqs[mask], fft_v[:, mask]
        
        colors = SIGNAL_COLORS[:5]
        fills = ['rgba(0,255,136,0.15)', 'rgba(0,212,255,0.15)', 'rgba(255,204,0,0.15)', 'rgba(255,107,157,0.15)', 'rgba(167,139,250,0.15)']
        
        with S.fft_plot:
            S.fft_plot.figure.data = []
            for i, ch in enumerate(chs[:5]):
                S.fft_plot.figure.add_trace(go.Scatter(x=freqs, y=fft_v[i], name=ch, line=dict(color=colors[i], width=1.5), fill='tozeroy', fillcolor=fills[i]))
            S.fft_plot.update()
    except Exception as e:
        print(f"FFT error: {e}")

def update_hilbert():
    if not S.hilbert_plot or not S.eeg_data or not S.selected_channels:
        return
    ch = S.hilbert_channel if S.hilbert_channel in S.selected_channels else (S.selected_channels[0] if S.selected_channels else None)
    if not ch:
        return
    try:
        data, times, _ = get_channel_data(S.eeg_data, [ch], S.view_start, S.view_duration)
        data = process_data(data, S.eeg_data.sfreq)[0] * 1e6
        amp, phase = compute_hilbert(data)
        
        with S.hilbert_plot:
            S.hilbert_plot.figure.data = []
            S.hilbert_plot.figure.add_trace(go.Scatter(x=times, y=data, name='Signal', line=dict(color=THEME_SECONDARY, width=1)), row=1, col=1)
            S.hilbert_plot.figure.add_trace(go.Scatter(x=times, y=amp, name='Envelope', line=dict(color=THEME_PRIMARY, width=2)), row=1, col=1)
            S.hilbert_plot.figure.add_trace(go.Scatter(x=times, y=-amp, showlegend=False, line=dict(color=THEME_PRIMARY, width=2)), row=1, col=1)
            S.hilbert_plot.figure.add_trace(go.Scatter(x=times, y=phase, name='Phase', line=dict(color=THEME_WARN, width=1)), row=2, col=1)
            S.hilbert_plot.update()
    except Exception as e:
        print(f"Hilbert error: {e}")

def update_brain():
    if not S.brain_plot or not S.eeg_data:
        return
    try:
        amps = S.current_amplitudes
        if amps and S.selected_channels:
            vals = [amps.get(ch, 0) for ch in S.selected_channels if ch in amps]
            min_a, max_a = (min(vals), max(vals)) if vals else (0, 1)
            rng = max_a - min_a if max_a > min_a else 1
        else:
            min_a, rng = 0, 1
        
        sel_x, sel_y, sel_c, sel_t, sel_l = [], [], [], [], []
        uns_x, uns_y, uns_l = [], [], []
        
        for ch, (x, y) in ELECTRODE_POSITIONS.items():
            if ch in S.selected_channels:
                sel_x.append(x); sel_y.append(y); sel_l.append(ch)
                a = amps.get(ch, 0)
                sel_c.append((a - min_a) / rng if rng > 0 else 0.5)
                sel_t.append(f'{ch}<br>{a:.1f} µV')
            else:
                uns_x.append(x); uns_y.append(y); uns_l.append(ch)
        
        with S.brain_plot:
            S.brain_plot.figure.data = S.brain_plot.figure.data[:4]
            if uns_x:
                S.brain_plot.figure.add_trace(go.Scatter(x=uns_x, y=uns_y, mode='markers+text',
                    marker=dict(size=12, color='rgba(30,30,30,0.6)', line=dict(width=1, color='rgba(60,60,60,0.5)')),
                    text=uns_l, textposition='top center', textfont=dict(size=7, color='rgba(100,100,100,0.6)', family='JetBrains Mono'),
                    hoverinfo='text', hovertext=uns_l, showlegend=False))
            if sel_x:
                # Custom colorscale: dark blue -> cyan -> green -> yellow
                terminal_scale = [[0, '#0d47a1'], [0.25, '#00bcd4'], [0.5, '#00ff88'], [0.75, '#ffcc00'], [1, '#ff5722']]
                S.brain_plot.figure.add_trace(go.Scatter(x=sel_x, y=sel_y, mode='markers+text',
                    marker=dict(size=18, color=sel_c, colorscale=terminal_scale, cmin=0, cmax=1,
                               line=dict(width=2, color=THEME_PRIMARY), showscale=True,
                               colorbar=dict(title=dict(text='µV', font=dict(size=9, color=THEME_TEXT_DIM)),
                                           len=0.5, thickness=8, x=1.02, tickfont=dict(size=8, color=THEME_TEXT_DIM))),
                    text=sel_l, textposition='top center', textfont=dict(size=8, color=THEME_PRIMARY, family='JetBrains Mono'),
                    hoverinfo='text', hovertext=sel_t, showlegend=False))
            S.brain_plot.update()
    except Exception as e:
        print(f"Brain error: {e}")

def update_all():
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

def update_eeg2():
    """Update EEG 2 plot with same time window and channels as EEG 1."""
    if not S.eeg_plot2 or not S.eeg_data2 or not S.selected_channels:
        return
    try:
        # Use same channels if they exist in EEG 2
        eeg2_chs = [ch for ch in S.selected_channels if ch in S.eeg_data2.channel_types]
        if not eeg2_chs:
            return
        data, times, chs = get_channel_data(S.eeg_data2, eeg2_chs, S.view_start, S.view_duration)
        data = process_data(data, S.eeg_data2.sfreq) * 1e6
        n = len(chs)
        
        norm = np.zeros_like(data)
        for i in range(n):
            std = np.std(data[i])
            norm[i] = data[i] / (std * 3) if std > 0 else data[i]
        
        spacing = 2.0 * S.scale_factor
        colors = ['#f472b6', '#fb7185', '#fda4af', '#fecdd3', '#ffe4e6']  # Pink tones for EEG 2
        
        y_min = -spacing
        y_max = (n) * spacing
        
        with S.eeg_plot2:
            S.eeg_plot2.figure.data = []
            for i in range(n):
                off = (n - 1 - i) * spacing
                S.eeg_plot2.figure.add_trace(go.Scatter(
                    x=times, y=norm[i] + off, name=chs[i], line=dict(color=colors[i % len(colors)], width=1),
                    hovertemplate=f'{chs[i]}: %{{customdata:.1f}} µV<extra></extra>', customdata=data[i]
                ))
            S.eeg_plot2.figure.update_layout(
                yaxis=dict(tickmode='array', tickvals=[(n-1-i)*spacing for i in range(n)], ticktext=chs, range=[y_min, y_max], fixedrange=True)
            )
            S.eeg_plot2.update()
    except Exception as e:
        print(f"EEG2 error: {e}")

def update_fft2():
    """Update FFT 2 plot."""
    if not S.fft_plot2 or not S.eeg_data2 or not S.selected_channels:
        return
    try:
        eeg2_chs = [ch for ch in S.selected_channels[:5] if ch in S.eeg_data2.channel_types]
        if not eeg2_chs:
            return
        data, times, chs = get_channel_data(S.eeg_data2, eeg2_chs, S.view_start, S.view_duration)
        data = process_data(data, S.eeg_data2.sfreq) * 1e6
        freqs, fft_v = compute_fft(data, S.eeg_data2.sfreq)
        mask = freqs <= 60
        freqs, fft_v = freqs[mask], fft_v[:, mask]
        
        colors = ['#f472b6', '#fb7185', '#fda4af', '#fecdd3', '#ffe4e6']
        fills = ['rgba(244,114,182,0.15)', 'rgba(251,113,133,0.15)', 'rgba(253,164,175,0.15)', 'rgba(254,205,211,0.15)', 'rgba(255,228,230,0.15)']
        
        with S.fft_plot2:
            S.fft_plot2.figure.data = []
            for i, ch in enumerate(chs[:5]):
                S.fft_plot2.figure.add_trace(go.Scatter(x=freqs, y=fft_v[i], name=ch, line=dict(color=colors[i], width=1.5), fill='tozeroy', fillcolor=fills[i]))
            S.fft_plot2.update()
    except Exception as e:
        print(f"FFT2 error: {e}")

def update_hilbert2():
    """Update Hilbert 2 plot."""
    if not S.hilbert_plot2 or not S.eeg_data2 or not S.selected_channels:
        return
    ch = S.hilbert_channel if S.hilbert_channel in S.eeg_data2.channel_types else None
    if not ch:
        # Try first available channel
        for c in S.selected_channels:
            if c in S.eeg_data2.channel_types:
                ch = c
                break
    if not ch:
        return
    try:
        data, times, _ = get_channel_data(S.eeg_data2, [ch], S.view_start, S.view_duration)
        data = process_data(data, S.eeg_data2.sfreq)[0] * 1e6
        amp, phase = compute_hilbert(data)
        
        with S.hilbert_plot2:
            S.hilbert_plot2.figure.data = []
            S.hilbert_plot2.figure.add_trace(go.Scatter(x=times, y=data, name='Signal', line=dict(color='#fb7185', width=1)), row=1, col=1)
            S.hilbert_plot2.figure.add_trace(go.Scatter(x=times, y=amp, name='Envelope', line=dict(color='#f472b6', width=2)), row=1, col=1)
            S.hilbert_plot2.figure.add_trace(go.Scatter(x=times, y=-amp, showlegend=False, line=dict(color='#f472b6', width=2)), row=1, col=1)
            S.hilbert_plot2.figure.add_trace(go.Scatter(x=times, y=phase, name='Phase', line=dict(color='#fda4af', width=1)), row=2, col=1)
            S.hilbert_plot2.update()
    except Exception as e:
        print(f"Hilbert2 error: {e}")

def update_brain2():
    """Update brain topography 2 plot."""
    if not S.brain_plot2 or not S.eeg_data2:
        return
    try:
        # Calculate amplitudes for EEG 2
        eeg2_chs = [ch for ch in S.selected_channels if ch in S.eeg_data2.channel_types]
        if not eeg2_chs:
            return
        data, times, chs = get_channel_data(S.eeg_data2, eeg2_chs, S.view_start, S.view_duration)
        data = process_data(data, S.eeg_data2.sfreq) * 1e6
        
        amps = {ch: np.sqrt(np.mean(data[i]**2)) for i, ch in enumerate(chs)}
        vals = list(amps.values())
        min_a, max_a = (min(vals), max(vals)) if vals else (0, 1)
        rng = max_a - min_a if max_a > min_a else 1
        
        sel_x, sel_y, sel_c, sel_t, sel_l = [], [], [], [], []
        
        for ch, (x, y) in ELECTRODE_POSITIONS.items():
            if ch in eeg2_chs:
                sel_x.append(x); sel_y.append(y); sel_l.append(ch)
                a = amps.get(ch, 0)
                sel_c.append((a - min_a) / rng if rng > 0 else 0.5)
                sel_t.append(f'{ch}<br>{a:.1f} µV')
        
        with S.brain_plot2:
            S.brain_plot2.figure.data = S.brain_plot2.figure.data[:4]
            if sel_x:
                pink_scale = [[0, '#831843'], [0.25, '#be185d'], [0.5, '#f472b6'], [0.75, '#fda4af'], [1, '#ffe4e6']]
                S.brain_plot2.figure.add_trace(go.Scatter(x=sel_x, y=sel_y, mode='markers+text',
                    marker=dict(size=18, color=sel_c, colorscale=pink_scale, cmin=0, cmax=1,
                               line=dict(width=2, color='#f472b6'), showscale=True,
                               colorbar=dict(title=dict(text='µV', font=dict(size=9, color=THEME_TEXT_DIM)),
                                           len=0.5, thickness=8, x=1.02, tickfont=dict(size=8, color=THEME_TEXT_DIM))),
                    text=sel_l, textposition='top center', textfont=dict(size=8, color='#f472b6', family='JetBrains Mono'),
                    hoverinfo='text', hovertext=sel_t, showlegend=False))
            S.brain_plot2.update()
    except Exception as e:
        print(f"Brain2 error: {e}")

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
    refresh_channels()
    refresh_hilbert_select()
    update_all()

def select_all_ch():
    if S.eeg_data:
        S.selected_channels = [ch for ch, t in S.eeg_data.channel_types.items() if t == 'eeg']
        refresh_channels()
        refresh_hilbert_select()
        update_all()

def select_10_ch():
    if S.eeg_data:
        S.selected_channels = [ch for ch, t in S.eeg_data.channel_types.items() if t == 'eeg'][:10]
        refresh_channels()
        refresh_hilbert_select()
        update_all()

def clear_ch():
    S.selected_channels = []
    S.current_amplitudes.clear()
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
            val = S.hilbert_channel if S.hilbert_channel in S.selected_channels else S.selected_channels[0]
            ui.select(options=S.selected_channels, value=val, on_change=on_sel).props('dense dark').classes('w-24')

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
    update_all()

def set_win(d):
    S.view_duration = d
    update_all()

async def toggle_play():
    S.is_playing = not S.is_playing
    while S.is_playing and S.eeg_data:
        S.view_start += S.view_duration * 0.08
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

# Pipeline state
class PipelineState:
    def __init__(self):
        self.running = False
        self.current_step = ""
        self.start_time = None
        self.current_process = None
        self.log_container = None
        self.status_label = None
        self.progress = 0
        self.refresh_files = None  # Function to refresh file browser
        # Pipeline parameters
        self.max_subjects = 0  # 0 = all
        self.conditions = ["DMT", "EC", "EO"]
        self.jobs = 0  # 0 = auto
        self.workers = 7
        self.bands = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
        self.min_k = 2
        self.max_k = 15
        self.min_comps = 2
        self.max_comps = 10

PS = PipelineState()

PIPELINE_DIR = Path(__file__).parent.parent / "dashboard" / "pipeline_backend"
PIPELINE_OUTPUTS = Path(__file__).parent.parent / "pipeline_outputs"
RESULTS_BASE = Path(__file__).parent.parent / "fwd-inv-stc"
DEFAULT_INPUT_DIR = Path(__file__).parent.parent / "EEG_CLEAN"

def get_run_dirs():
    """List existing pipeline runs"""
    if not PIPELINE_OUTPUTS.exists():
        return []
    return sorted([d.name for d in PIPELINE_OUTPUTS.iterdir() if d.is_dir() and d.name.startswith('run_')], reverse=True)

def create_new_run():
    """Create a new run directory with timestamp"""
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = PIPELINE_OUTPUTS / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for cond in ["DMT", "EC", "EO"]:
        (run_dir / cond).mkdir(exist_ok=True)
    return run_dir

def pipeline_log(msg):
    """Add message to pipeline log"""
    if PS.log_container:
        with PS.log_container:
            ui.label(msg).style(f'color:{THEME_TEXT}; font-family: JetBrains Mono; font-size: 0.75rem;')

async def run_pipeline_step(script_name, args_list, step_name, output_dir=None, input_dir=None):
    """Run a pipeline script with arguments"""
    import subprocess
    import os as _os
    
    if PS.running:
        ui.notify('Pipeline already running', type='warning')
        return
    
    PS.running = True
    PS.current_step = step_name
    PS.start_time = datetime.now()
    
    script_path = PIPELINE_DIR / script_name
    if not script_path.exists():
        ui.notify(f'Script not found: {script_path}', type='negative')
        PS.running = False
        return
    
    # Use -u for unbuffered output so we see logs in real-time
    cmd = ["python", "-u", str(script_path)] + args_list
    pipeline_log(f"[{step_name}] Starting: {' '.join(cmd)}")
    
    # Set environment variables for input/output directories
    env = _os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'  # Force unbuffered output
    if input_dir:
        env['PIPELINE_INPUT_DIR'] = str(input_dir)
        pipeline_log(f"[{step_name}] Input dir: {input_dir}")
    if output_dir:
        env['PIPELINE_OUTPUT_DIR'] = str(output_dir)
        pipeline_log(f"[{step_name}] Output dir: {output_dir}")
    
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(PIPELINE_DIR.parent.parent),
            env=env
        )
        
        PS.current_process = process
        
        # Read both stdout and stderr
        async def read_stream(stream, prefix=""):
            while True:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode().strip()
                if text:
                    pipeline_log(f"{prefix}{text}")
        
        # Read both streams concurrently
        await asyncio.gather(
            read_stream(process.stdout),
            read_stream(process.stderr, "[ERR] ")
        )
        
        await process.wait()
        
        if process.returncode == 0:
            pipeline_log(f"[{step_name}] Completed successfully")
            ui.notify(f'{step_name} completed!', type='positive')
        else:
            pipeline_log(f"[{step_name}] Failed with code {process.returncode}")
            ui.notify(f'{step_name} failed', type='negative')
            
    except Exception as e:
        import traceback
        pipeline_log(f"[{step_name}] ERROR: {str(e)}")
        pipeline_log(traceback.format_exc())
        ui.notify(f'Error: {e}', type='negative')
    finally:
        PS.running = False
        PS.current_step = ""
        PS.start_time = None
        PS.current_process = None


@ui.page('/pipeline')
def pipeline_page():
    ui.add_head_html(f'<style>{STYLE}</style>')
    
    # Header with navigation
    with ui.header().classes('items-center px-4 py-1').style(f'background: {THEME_BG}; border-bottom: 1px solid {THEME_BORDER};'):
        ui.label('▶').style(f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.75rem;')
        ui.label('EEG_PIPELINE').classes('text-base font-medium ml-2').style(f'color: {THEME_PRIMARY}; font-family: JetBrains Mono;')
        ui.label('v1.0').classes('text-xs ml-2').style(f'color: {THEME_TEXT_DIM}; font-family: JetBrains Mono;')
        with ui.row().classes('ml-auto gap-2'):
            ui.button('VIEWER', on_click=lambda: ui.navigate.to('/')).props('flat dense').style(f'color:{THEME_TEXT_DIM};')
            ui.button('PIPELINE', on_click=lambda: ui.navigate.to('/pipeline')).props('flat dense').style(f'color:{THEME_PRIMARY};')
            ui.button('MODEL', on_click=lambda: ui.navigate.to('/model')).props('flat dense').style(f'color:{THEME_TEXT_DIM};')
            ui.button('ANALYSIS', on_click=lambda: ui.navigate.to('/analysis')).props('flat dense').style(f'color:{THEME_TEXT_DIM};')
    
    with ui.row().classes('w-full p-4 gap-4').style('height: calc(100vh - 50px); align-items: stretch;'):
        
        # LEFT: Pipeline Controls
        with ui.column().classes('gap-4').style('width: 450px;'):
            
            # INPUT/OUTPUT CONFIGURATION
            with ui.card().classes('dark-card p-4 w-full').style(f'border: 1px solid {THEME_PRIMARY};'):
                ui.label('// INPUT_OUTPUT_DIRS').classes('terminal-header')
                
                # INPUT DIRECTORY
                with ui.row().classes('items-center gap-2 mt-2 w-full'):
                    ui.label('INPUT:').style(f'color:{THEME_SECONDARY}; font-family: JetBrains Mono; font-size: 0.75rem; min-width: 60px;')
                    input_dir_field = ui.input(value=str(DEFAULT_INPUT_DIR)).props('dense').classes('flex-1')
                    
                    def scan_input_dir():
                        p = Path(input_dir_field.value)
                        if p.exists():
                            conds = [d.name for d in p.iterdir() if d.is_dir() and d.name in ['DMT', 'EC', 'EO']]
                            files = sum(len(list((p / c).glob('*.set'))) for c in conds if (p / c).exists())
                            ui.notify(f'Found: {conds}, {files} files', type='info')
                            pipeline_log(f"[INPUT] Scanned {p}: {conds}, {files} .set files")
                        else:
                            ui.notify('Directory not found', type='warning')
                    
                    ui.button(icon='search', on_click=scan_input_dir).props('flat dense size=sm')
                
                ui.label('Directorio con subcarpetas DMT/, EC/, EO/ y archivos .set').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem; margin-left: 68px;')
                
                # Mutable container for input dir
                current_input_dir = [DEFAULT_INPUT_DIR]
                def get_input_dir():
                    return Path(input_dir_field.value) if input_dir_field.value else DEFAULT_INPUT_DIR
                
                ui.separator().classes('my-2')
                
                # OUTPUT DIRECTORY
                with ui.row().classes('items-center gap-2 w-full'):
                    ui.label('OUTPUT:').style(f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.75rem; min-width: 60px;')
                    
                    run_label = ui.label('(crear NEW RUN)').style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.8rem;')
                    current_run_dir = [None]
                    
                    def refresh_run_label():
                        if current_run_dir[0]:
                            run_label.text = str(current_run_dir[0].name)
                            run_label.style(f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.8rem;')
                        else:
                            run_label.text = '(crear NEW RUN)'
                            run_label.style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.8rem;')
                    
                    async def new_run():
                        run_dir = await asyncio.get_event_loop().run_in_executor(None, create_new_run)
                        current_run_dir[0] = run_dir
                        refresh_run_label()
                        if PS.refresh_files: PS.refresh_files()
                        ui.notify(f'Nuevo run: {run_dir.name}', type='positive')
                        pipeline_log(f"[RUN] Created: {run_dir}")
                    
                    ui.button('NEW RUN', on_click=new_run, icon='add').props('dense').style(f'background:{THEME_PRIMARY}; color:black;')
                    
                    existing_runs = get_run_dirs()
                    if existing_runs:
                        run_select = ui.select(existing_runs, label='continuar:').props('dense').classes('w-36')
                        def use_existing():
                            if run_select.value:
                                current_run_dir[0] = PIPELINE_OUTPUTS / run_select.value
                                refresh_run_label()
                                if PS.refresh_files: PS.refresh_files()
                                pipeline_log(f"[RUN] Using existing: {current_run_dir[0]}")
                        run_select.on('update:model-value', lambda e: use_existing())
                
                ui.label(f'Output: {PIPELINE_OUTPUTS}/run_* (no pisa fwd-inv-stc/)').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem; margin-left: 68px;')
            
            # GLOBAL PARAMETERS - Clean grid layout
            with ui.card().classes('dark-card p-4 w-full'):
                ui.label('// GLOBAL_PARAMS').classes('terminal-header')
                
                # Use CSS grid for aligned parameters
                with ui.element('div').classes('w-full').style('display: grid; grid-template-columns: 100px 1fr; gap: 8px 12px; align-items: center;'):
                    # Conditions row
                    ui.label('Conditions').style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.75rem;')
                    with ui.row().classes('gap-3 items-center'):
                        cond_dmt = ui.checkbox('DMT', value=True).props('dense')
                        cond_ec = ui.checkbox('EC', value=True).props('dense')
                        cond_eo = ui.checkbox('EO', value=True).props('dense')
                    
                    # Max subjects row
                    ui.label('Max subjects').style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.75rem;')
                    with ui.row().classes('gap-2 items-center'):
                        max_subj = ui.number(value=0, min=0, max=100).props('dense').classes('w-20')
                        ui.label('0 = all').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                    
                    # Max epochs row
                    ui.label('Max epochs').style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.75rem;')
                    with ui.row().classes('gap-2 items-center'):
                        max_epochs = ui.number(value=0, min=0, max=500).props('dense').classes('w-20')
                        ui.label('0 = all').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                    
                    # Workers row
                    ui.label('Workers').style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.75rem;')
                    with ui.row().classes('gap-2 items-center'):
                        workers_num = ui.number(value=7, min=1, max=32).props('dense').classes('w-20')
                    
                    # Jobs row
                    ui.label('Jobs').style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.75rem;')
                    with ui.row().classes('gap-2 items-center'):
                        jobs_num = ui.number(value=0, min=0, max=64).props('dense').classes('w-20')
                        ui.label('0 = auto').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                
                def get_conditions():
                    conds = []
                    if cond_dmt.value: conds.append('DMT')
                    if cond_ec.value: conds.append('EC')
                    if cond_eo.value: conds.append('EO')
                    return conds
            
            # STEP 1: FWD.PY
            with ui.card().classes('dark-card p-4 w-full'):
                ui.label('// STEP_1: SOURCE_LOCALIZATION').classes('terminal-header')
                ui.label('fwd.py - Forward/Inverse Solution + Metrics').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                ui.label('→ run_*/phases-{subj}.pkl').style(f'color:{THEME_PRIMARY}; font-size: 0.65rem;')
                
                with ui.row().classes('gap-2 mt-3'):
                    async def run_fwd():
                        if not current_run_dir[0]:
                            ui.notify('Primero creá un NEW RUN', type='warning')
                            return
                        args = [
                            '--max-subjects', str(int(max_subj.value or 0)),
                            '--conditions'] + get_conditions() + [
                            '--jobs', str(int(jobs_num.value or 0)),
                            '--workers', str(int(workers_num.value or 7)),
                            '--max-epochs', str(int(max_epochs.value or 0))
                        ]
                        await run_pipeline_step('fwd.py', args, 'Source Localization', current_run_dir[0], get_input_dir())
                    
                    ui.button('RUN fwd.py', on_click=run_fwd, icon='play_arrow').props('dense').style(f'background:{THEME_PRIMARY}; color:black;')
                    ui.label('~3-4h (o menos con max_epochs)').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
            
            # STEP 2: MULTI2POOL2.PY
            with ui.card().classes('dark-card p-4 w-full'):
                ui.label('// STEP_2: NETWORK_FILTERING').classes('terminal-header')
                ui.label('multi2pool2.py - Filter by brain networks (DMN, FPN, etc)').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                ui.label('→ run_*/order_all-{subj}.pkl').style(f'color:{THEME_PRIMARY}; font-size: 0.65rem;')
                
                with ui.row().classes('gap-2 mt-3'):
                    async def run_multi():
                        if not current_run_dir[0]:
                            ui.notify('Primero creá un NEW RUN', type='warning')
                            return
                        await run_pipeline_step('multi2pool2.py', [], 'Network Filtering', current_run_dir[0])
                    
                    ui.button('RUN multi2pool2.py', on_click=run_multi, icon='play_arrow').props('dense').style(f'background:{THEME_SECONDARY}; color:black;')
                    ui.label('~2-5 min').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
            
            # STEP 3: GENERATE_ORDER.PY
            with ui.card().classes('dark-card p-4 w-full'):
                ui.label('// STEP_3: KURAMOTO_ORDER').classes('terminal-header')
                ui.label('generate_order.py - Calculate Kuramoto order parameter').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                ui.label('→ run_*/order-{subj}.pkl').style(f'color:{THEME_PRIMARY}; font-size: 0.65rem;')
                
                with ui.row().classes('gap-2 mt-3'):
                    async def run_order():
                        if not current_run_dir[0]:
                            ui.notify('Primero creá un NEW RUN', type='warning')
                            return
                        args = ['--workers', str(int(workers_num.value or 7)), '--conditions'] + get_conditions()
                        await run_pipeline_step('generate_order.py', args, 'Kuramoto Order', current_run_dir[0])
                    
                    ui.button('RUN generate_order.py', on_click=run_order, icon='play_arrow').props('dense').style(f'background:{THEME_WARN}; color:black;')
                    ui.label('~1 min').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
            
            # STEP 4: CLUSTERING.PY
            with ui.card().classes('dark-card p-4 w-full'):
                ui.label('// STEP_4: CLUSTERING').classes('terminal-header')
                ui.label('clustering.py - Brain state identification').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                ui.label('→ run_*/clustering_results/').style(f'color:{THEME_PRIMARY}; font-size: 0.65rem;')
                
                with ui.row().classes('gap-4 items-center mt-2'):
                    ui.label('Bands:').style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.7rem;')
                    band_delta = ui.checkbox('δ', value=True).props('dense')
                    band_theta = ui.checkbox('θ', value=True).props('dense')
                    band_alpha = ui.checkbox('α', value=True).props('dense')
                    band_beta = ui.checkbox('β', value=True).props('dense')
                    band_gamma = ui.checkbox('γ', value=True).props('dense')
                
                with ui.row().classes('gap-3 items-center mt-2'):
                    ui.label('Clusters:').style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.7rem;')
                    min_k = ui.number(value=2, min=2, max=20).props('dense').classes('w-16')
                    ui.label('-').style(f'color:{THEME_TEXT_DIM};')
                    max_k = ui.number(value=15, min=2, max=30).props('dense').classes('w-16')
                    
                    ui.label('PCA:').style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.7rem;')
                    min_pca = ui.number(value=2, min=2, max=20).props('dense').classes('w-16')
                    ui.label('-').style(f'color:{THEME_TEXT_DIM};')
                    max_pca = ui.number(value=10, min=2, max=30).props('dense').classes('w-16')
                
                search_mode = ui.toggle(['Quick', 'Full'], value='Quick').props('dense')
                
                with ui.row().classes('gap-2 mt-3'):
                    async def run_clustering():
                        if not current_run_dir[0]:
                            ui.notify('Primero creá un NEW RUN', type='warning')
                            return
                        bands = []
                        if band_delta.value: bands.append('Delta')
                        if band_theta.value: bands.append('Theta')
                        if band_alpha.value: bands.append('Alpha')
                        if band_beta.value: bands.append('Beta')
                        if band_gamma.value: bands.append('Gamma')
                        
                        mode_arg = '--quick-search' if search_mode.value == 'Quick' else '--full-search'
                        cluster_out = str(current_run_dir[0] / "clustering_results")
                        args = [
                            mode_arg,
                            '--conditions'] + get_conditions() + [
                            '--bands'] + bands + [
                            '--min-k', str(int(min_k.value)),
                            '--max-k', str(int(max_k.value)),
                            '--min-comps', str(int(min_pca.value)),
                            '--max-comps', str(int(max_pca.value)),
                            '--workers', str(int(workers_num.value or 4)),
                            '--output-dir', cluster_out
                        ]
                        await run_pipeline_step('clustering.py', args, 'Clustering', current_run_dir[0])
                    
                    ui.button('RUN clustering.py', on_click=run_clustering, icon='play_arrow').props('dense').style(f'background:#ff6b9d; color:black;')
                    ui.label('Quick: ~30min, Full: ~4h').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
            
            # STEP 5: BUILD_ORDER_DATA.PY
            with ui.card().classes('dark-card p-4 w-full'):
                ui.label('// STEP_5: AGGREGATE_DATA').classes('terminal-header')
                ui.label('build_order_data.py - Aggregate Kuramoto metrics').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                ui.label('-> r_kuramoto_nets_*.pkl').style(f'color:{THEME_PRIMARY}; font-size: 0.65rem;')
                
                with ui.row().classes('gap-2 mt-3'):
                    async def run_build_order():
                        if not current_run_dir[0]:
                            ui.notify('Primero creá un NEW RUN', type='warning')
                            return
                        args = ['--build-all', '--workers', str(int(workers_num.value or 7))]
                        await run_pipeline_step('build_order_data.py', args, 'Aggregate Data', current_run_dir[0])
                    
                    ui.button('RUN build_order_data.py', on_click=run_build_order, icon='play_arrow').props('dense').style(f'background:#60a5fa; color:black;')
                    ui.label('~2-5 min').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
            
            # STEP 6: PEARSON.PY
            with ui.card().classes('dark-card p-4 w-full'):
                ui.label('// STEP_6: CORRELATIONS').classes('terminal-header')
                ui.label('pearson.py - Correlate with questionnaires').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                ui.label('-> pearson_results/').style(f'color:{THEME_PRIMARY}; font-size: 0.65rem;')
                
                with ui.row().classes('gap-2 mt-3'):
                    async def run_pearson():
                        if not current_run_dir[0]:
                            ui.notify('Primero creá un NEW RUN', type='warning')
                            return
                        await run_pipeline_step('pearson.py', [], 'Correlations', current_run_dir[0])
                    
                    ui.button('RUN pearson.py', on_click=run_pearson, icon='play_arrow').props('dense').style(f'background:#a78bfa; color:black;')
                    ui.label('~3-5 min').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
        
        # RIGHT: Tabbed Panel (Console, Files, System, Visualize)
        with ui.column().classes('flex-1').style('min-height: 0; display: flex; flex-direction: column;'):
            with ui.card().classes('dark-card p-2 w-full flex-1').style('display: flex; flex-direction: column; min-height: 0;'):
                with ui.tabs().classes('w-full').style(f'background: {THEME_BG};') as tabs:
                    tab_console = ui.tab('CONSOLE', icon='terminal').style(f'color:{THEME_PRIMARY};')
                    tab_files = ui.tab('FILES', icon='folder').style(f'color:{THEME_SECONDARY};')
                    tab_system = ui.tab('SYSTEM', icon='memory').style(f'color:{THEME_WARN};')
                    tab_viz = ui.tab('VISUALIZE', icon='analytics').style(f'color:#a78bfa;')
                
                with ui.tab_panels(tabs, value=tab_console).classes('w-full').style('flex: 1; min-height: 0; overflow: hidden;'):
                    # CONSOLE TAB
                    with ui.tab_panel(tab_console).classes('p-2').style('height: 100%; display: flex; flex-direction: column;'):
                        with ui.row().classes('items-center gap-3 mb-2'):
                            ui.label('// OUTPUT_LOG').classes('terminal-header')
                            
                            # Status indicator
                            # Reset pipeline state on page load
                            PS.running = False
                            PS.current_step = ""
                            PS.start_time = None
                            PS.current_process = None
                            
                            status_label = ui.label('Idle').style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.7rem; margin-left: auto;')
                            PS.status_label = status_label
                            
                            def update_status():
                                if PS.running and PS.start_time:
                                    elapsed = (datetime.now() - PS.start_time).seconds
                                    mins, secs = divmod(elapsed, 60)
                                    status_label.text = f'Running: {PS.current_step} ({mins}m {secs}s)'
                                    status_label.style(f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.7rem;')
                                else:
                                    status_label.text = 'Idle'
                                    status_label.style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.7rem;')
                            
                            ui.timer(1.0, update_status)
                            
                            async def stop_pipeline():
                                if PS.current_process:
                                    try:
                                        PS.current_process.terminate()
                                        pipeline_log(f"[{PS.current_step}] STOPPED by user")
                                        ui.notify('Pipeline stopped', type='warning')
                                    except:
                                        pass
                            
                            ui.button('STOP', on_click=stop_pipeline, icon='stop').props('flat dense size=sm color=negative')
                            
                            def clear_log():
                                if PS.log_container:
                                    PS.log_container.clear()
                            ui.button('CLEAR', on_click=clear_log, icon='delete').props('flat dense size=sm')
                        
                        with ui.scroll_area().classes('w-full flex-1').style('background: #050505; border-radius: 4px; min-height: 200px;'):
                            PS.log_container = ui.column().classes('w-full p-3 gap-0')
                            with PS.log_container:
                                ui.label('Pipeline ready. Select a step and click RUN.').style(f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.75rem;')
                                ui.label(f'Pipeline directory: {PIPELINE_DIR}').style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.7rem;')
                    
                    # FILES TAB
                    with ui.tab_panel(tab_files).classes('p-2').style('height: 100%; display: flex; flex-direction: column;'):
                        file_browser_container = ui.column().classes('w-full flex-1').style('min-height: 0; overflow: hidden;')
                        
                        def refresh_files():
                            file_browser_container.clear()
                            if not current_run_dir[0] or not current_run_dir[0].exists():
                                with file_browser_container:
                                    ui.label('No run selected').style(f'color:{THEME_TEXT_DIM}; font-size: 0.75rem;')
                                return
                            
                            run_path = current_run_dir[0]
                            with file_browser_container:
                                def count_files(pattern):
                                    return len(list(run_path.rglob(pattern)))
                                
                                stats = {
                                    'syncro': count_files('syncro-*.pkl') + count_files('phases-*.pkl'),
                                    'order_all': count_files('order_all-*.pkl'),
                                    'order': count_files('order-*.pkl'),
                                    'clustering': count_files('clustering_results/**/*.pkl') + count_files('clustering_results/**/*.csv'),
                                    'pearson': count_files('pearson_results/**/*'),
                                }
                                
                                ui.label(f'{run_path.name}').style(f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.8rem;')
                                
                                with ui.row().classes('gap-4 mt-2 flex-wrap'):
                                    for name, count in stats.items():
                                        color = THEME_PRIMARY if count > 0 else THEME_TEXT_DIM
                                        ui.label(f'{name}: {count}').style(f'color:{color}; font-family: JetBrains Mono; font-size: 0.7rem;')
                                
                                ui.separator().classes('my-2')
                                
                                with ui.scroll_area().classes('w-full flex-1').style('min-height: 150px;'):
                                    for item in sorted(run_path.iterdir()):
                                        if item.is_dir():
                                            file_count = len(list(item.rglob('*')))
                                            ui.label(f'[dir] {item.name}/ ({file_count} files)').style(f'color:{THEME_SECONDARY}; font-family: JetBrains Mono; font-size: 0.7rem;')
                                        else:
                                            size_kb = item.stat().st_size / 1024
                                            size_str = f'{size_kb:.1f}KB' if size_kb < 1024 else f'{size_kb/1024:.1f}MB'
                                            ui.label(f'[file] {item.name} ({size_str})').style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.7rem;')
                        
                        with ui.row().classes('mt-2'):
                            ui.button('REFRESH', on_click=refresh_files, icon='refresh').props('flat dense size=sm')
                        
                        PS.refresh_files = refresh_files
                        refresh_files()
                    
                    # SYSTEM TAB
                    with ui.tab_panel(tab_system).classes('p-2'):
                        ui.label('// SYSTEM_MONITOR').classes('terminal-header mb-2')
                        system_container = ui.column().classes('w-full gap-4')
                        
                        def update_system_stats():
                            import psutil
                            system_container.clear()
                            with system_container:
                                cpu_percent = psutil.cpu_percent(interval=0.1)
                                cpu_count = psutil.cpu_count()
                                cpu_color = THEME_PRIMARY if cpu_percent < 50 else (THEME_WARN if cpu_percent < 80 else '#ff4444')
                                
                                with ui.row().classes('gap-4 items-center'):
                                    with ui.column().classes('gap-0'):
                                        ui.label(f'CPU {cpu_percent:.0f}%').style(f'color:{cpu_color}; font-family: JetBrains Mono; font-size: 1rem; font-weight: bold;')
                                        ui.label(f'{cpu_count} cores').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                                    
                                    per_cpu = psutil.cpu_percent(percpu=True)
                                    with ui.row().classes('gap-1 flex-wrap'):
                                        for i, pct in enumerate(per_cpu[:16]):
                                            color = THEME_PRIMARY if pct < 50 else (THEME_WARN if pct < 80 else '#ff4444')
                                            ui.label(f'{pct:.0f}').style(f'color:{color}; font-family: JetBrains Mono; font-size: 0.65rem; min-width: 22px; text-align: center;')
                                
                                mem = psutil.virtual_memory()
                                mem_used_gb = mem.used / (1024**3)
                                mem_total_gb = mem.total / (1024**3)
                                mem_color = THEME_PRIMARY if mem.percent < 60 else (THEME_WARN if mem.percent < 85 else '#ff4444')
                                
                                with ui.row().classes('gap-4 items-center'):
                                    ui.label(f'RAM {mem.percent:.0f}%').style(f'color:{mem_color}; font-family: JetBrains Mono; font-size: 1rem; font-weight: bold;')
                                    ui.label(f'{mem_used_gb:.1f} / {mem_total_gb:.0f} GB').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                                
                                try:
                                    import GPUtil
                                    gpus = GPUtil.getGPUs()
                                    if gpus:
                                        gpu = gpus[0]
                                        gpu_color = THEME_PRIMARY if gpu.load*100 < 50 else (THEME_WARN if gpu.load*100 < 80 else '#ff4444')
                                        with ui.row().classes('gap-4 items-center'):
                                            ui.label(f'GPU {gpu.load*100:.0f}%').style(f'color:{gpu_color}; font-family: JetBrains Mono; font-size: 1rem; font-weight: bold;')
                                            ui.label(f'{gpu.memoryUsed:.0f} / {gpu.memoryTotal:.0f} MB').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                                except:
                                    ui.label('GPU: N/A').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                        
                        ui.timer(2.0, update_system_stats)
                        update_system_stats()
                    
                    # VISUALIZE TAB - Unified Visualization Dashboard
                    with ui.tab_panel(tab_viz).classes('p-0').style('height: 100%; overflow: hidden;'):
                        # State for visualization - use nonlocal dict to persist across tab switches
                        if not hasattr(PS, 'viz_state'):
                            PS.viz_state = {'data': None, 'file': None, 'loaded': False}
                        viz_state = PS.viz_state
                        
                        with ui.column().classes('w-full h-full').style('display: flex; flex-direction: column; overflow: hidden;'):
                            # FIXED HEADER - Data Selection with custom path support (outside scroll area)
                            with ui.card().classes('dark-card p-3 w-full').style('flex-shrink: 0;'):
                                # Custom data path
                                with ui.row().classes('items-center gap-2 w-full mb-2'):
                                    ui.label('Data Path:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                                    custom_data_path = ui.input(placeholder='/path/to/data or leave empty for run dir').props('dense dark').classes('flex-1')
                                    
                                    def browse_path():
                                        """Load data from custom path"""
                                        path = custom_data_path.value.strip() if custom_data_path.value else None
                                        if path:
                                            from pathlib import Path
                                            p = Path(path)
                                            if p.exists():
                                                viz_state['custom_path'] = p
                                                load_subjects_from_path(p)
                                            else:
                                                ui.notify(f'Path not found: {path}', type='warning')
                                        elif current_run_dir[0]:
                                            viz_state['custom_path'] = None
                                            load_subjects_from_path(current_run_dir[0])
                                    
                                    ui.button('Load', on_click=browse_path, icon='folder_open').props('dense flat')
                                
                                with ui.row().classes('items-center gap-4 flex-wrap'):
                                    ui.label('▌VISUALIZATION').style(f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.9rem; letter-spacing: 1px;')
                                    
                                    viz_band = ui.select(['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'], value='Alpha', label='Band').props('dense dark').classes('w-24')
                                    viz_subject = ui.select([], label='Subject').props('dense dark').classes('w-32')
                                    viz_epoch = ui.number(value=0, min=0, max=100, label='Epoch').props('dense').classes('w-20')
                                    
                                    def load_subjects_from_path(path):
                                        """Load subjects from given path"""
                                        from pathlib import Path
                                        p = Path(path)
                                        viz_subject.options = []
                                        all_files = list(p.rglob('syncro-*.pkl')) + list(p.rglob('phases-*.pkl'))
                                        subjects = sorted(list(set([f.stem.split('-')[1] if '-' in f.stem else f.stem for f in all_files])))[:30]
                                        viz_subject.options = subjects
                                        if subjects:
                                            viz_subject.value = subjects[0]
                                        ui.notify(f'Found {len(subjects)} subjects in {p.name}', type='info')
                                    
                                    def load_subjects():
                                        path = viz_state.get('custom_path') or current_run_dir[0]
                                        if path:
                                            load_subjects_from_path(path)
                                    
                                    ui.button('Load Subjects', on_click=load_subjects, icon='refresh').props('dense flat')
                                    
                                    viz_status = ui.label('No data loaded').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem; margin-left: auto;')
                                    
                                    # Function to refresh all plots when parameters change
                                    def refresh_all_plots():
                                        """Refresh all visualizations with current parameters"""
                                        # Load data for current subject
                                        data_path = viz_state.get('custom_path') or current_run_dir[0]
                                        if data_path and viz_subject.value:
                                            import pickle
                                            from pathlib import Path
                                            try:
                                                files = list(Path(data_path).rglob(f'*{viz_subject.value}*.pkl'))
                                                if files:
                                                    with open(files[0], 'rb') as f:
                                                        viz_state['data'] = pickle.load(f)
                                                    viz_state['file'] = files[0]
                                                    viz_status.text = f'Loaded: {files[0].name}'
                                                    viz_status.style(f'color:{THEME_PRIMARY}; font-size: 0.7rem;')
                                            except Exception as e:
                                                viz_status.text = f'Error: {e}'
                                        
                                        # Update all plots
                                        try:
                                            # Update brain plot using stored reference
                                            if 'update_brain_plot' in viz_state:
                                                viz_state['update_brain_plot']()
                                            update_network_plot()
                                            update_kuramoto_timeline()
                                            update_band_comparison()
                                            update_phase_plot()
                                            update_sync_matrix()
                                            update_connectivity()
                                            # Use stored references for Hilbert functions (defined later)
                                            if 'update_hilbert_2d' in viz_state:
                                                viz_state['update_hilbert_2d']()
                                            if 'update_hilbert_3d' in viz_state:
                                                viz_state['update_hilbert_3d']()
                                        except:
                                            pass  # Some functions might not be defined yet
                                    
                                    # Connect selectors to auto-refresh
                                    viz_band.on('update:model-value', lambda e: refresh_all_plots())
                                    viz_subject.on('update:model-value', lambda e: refresh_all_plots())
                                    viz_epoch.on('update:model-value', lambda e: refresh_all_plots())
                            
                            # VISUALIZATION CONTENT - scroll area fills remaining space
                            with ui.scroll_area().classes('w-full flex-1').style('min-height: 0;'):
                                with ui.column().classes('w-full p-3 gap-3'):
                                    # MAIN VISUALIZATION AREA - 3D Brain + Stats
                                    with ui.row().classes('w-full gap-3'):
                                        # LEFT: 3D Brain Visualization
                                        with ui.card().classes('dark-card p-3').style('flex: 2; min-width: 400px;'):
                                            ui.label('▌3D BRAIN NETWORK').style(f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-2')
                                            
                                            brain_plot_container = ui.column().classes('w-full')
                                            
                                            # Track current plot type for auto-refresh
                                            viz_state['current_brain_plot'] = 'network'
                                            
                                            def update_brain_plot(plot_type=None):
                                                if plot_type is None:
                                                    plot_type = viz_state.get('current_brain_plot', 'network')
                                                else:
                                                    viz_state['current_brain_plot'] = plot_type
                                                
                                                brain_plot_container.clear()
                                                try:
                                                    from viz_scripts import brain_3d
                                                    import pickle
                                                    
                                                    # Try to load data from custom path or run dir
                                                    data = viz_state.get('data')
                                                    data_path = viz_state.get('custom_path') or current_run_dir[0]
                                                    if data_path and viz_subject.value:
                                                        files = list(data_path.rglob(f'*{viz_subject.value}*.pkl'))
                                                        if files:
                                                            with open(files[0], 'rb') as f:
                                                                data = pickle.load(f)
                                                            viz_state['data'] = data
                                                            viz_state['file'] = files[0]
                                                            viz_status.text = f'Loaded: {files[0].name}'
                                                            viz_status.style(f'color:{THEME_PRIMARY}; font-size: 0.7rem;')
                                                    
                                                    with brain_plot_container:
                                                        if plot_type == 'network':
                                                            fig = brain_3d.create_brain_network_figure()
                                                        elif plot_type == 'colored':
                                                            fig = brain_3d.create_colored_brain_figure()
                                                        elif plot_type == 'sync':
                                                            fig = brain_3d.create_sync_brain_figure(data, viz_band.value, int(viz_epoch.value or 0))
                                                        elif plot_type == 'all_bands':
                                                            fig = brain_3d.create_all_bands_brain_figure(data)
                                                        else:
                                                            fig = brain_3d.create_brain_network_figure()
                                                        
                                                        ui.plotly(fig).classes('w-full').style('height: 450px;')
                                                    
                                                    # Update button states
                                                    update_brain_buttons(plot_type)
                                                except Exception as e:
                                                    with brain_plot_container:
                                                        ui.label(f'Error: {e}').style(f'color:{THEME_ERROR}; font-size: 0.75rem;')
                                                        import traceback
                                                        ui.label(traceback.format_exc()[:500]).style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem; white-space: pre-wrap;')
                                            
                                            # Store reference for refresh_all_plots
                                            viz_state['update_brain_plot'] = update_brain_plot
                                        
                                        # Brain plot type buttons with active state
                                        brain_buttons = {}
                                        with ui.row().classes('gap-2 mb-2'):
                                            brain_buttons['network'] = ui.button('Networks', on_click=lambda: update_brain_plot('network')).props('dense')
                                            brain_buttons['colored'] = ui.button('Parcellation', on_click=lambda: update_brain_plot('colored')).props('dense')
                                            brain_buttons['sync'] = ui.button('Sync Map', on_click=lambda: update_brain_plot('sync')).props('dense')
                                            brain_buttons['all_bands'] = ui.button('All Bands', on_click=lambda: update_brain_plot('all_bands')).props('dense')
                                        
                                        def update_brain_buttons(active_type):
                                            for btn_type, btn in brain_buttons.items():
                                                if btn_type == active_type:
                                                    btn.style(f'background:{THEME_PRIMARY}; color:black;')
                                                else:
                                                    btn.style(f'background:transparent; color:{THEME_TEXT_DIM}; border: 1px solid {THEME_BORDER};')
                                        
                                        # Initial button state
                                        update_brain_buttons('network')
                                        
                                        # Load initial plot
                                        update_brain_plot('network')
                                    
                                    # RIGHT: Network Stats
                                    with ui.card().classes('dark-card p-3').style('flex: 1; min-width: 300px;'):
                                        ui.label('▌NETWORK SYNC').style(f'color:{THEME_SECONDARY}; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-2')
                                        
                                        network_plot_container = ui.column().classes('w-full')
                                        
                                        def update_network_plot():
                                            network_plot_container.clear()
                                            data = viz_state.get('data')
                                            if not data:
                                                with network_plot_container:
                                                    ui.label('Load data first').style(f'color:{THEME_TEXT_DIM}; font-size: 0.75rem;')
                                                return
                                            try:
                                                from viz_scripts import brain_3d
                                                with network_plot_container:
                                                    fig = brain_3d.create_network_comparison_figure(data, viz_band.value)
                                                    ui.plotly(fig).classes('w-full').style('height: 350px;')
                                            except Exception as e:
                                                with network_plot_container:
                                                    ui.label(f'Error: {e}').style(f'color:{THEME_TEXT_DIM};')
                                        
                                        # Don't auto-load without data
                                        if viz_state.get('data'):
                                            update_network_plot()
                                
                                # SECOND ROW - Kuramoto Analysis
                                with ui.row().classes('w-full gap-3'):
                                    # Timeline
                                    with ui.card().classes('dark-card p-3 flex-1'):
                                        ui.label('▌KURAMOTO TIMELINE').style(f'color:{THEME_WARN}; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-2')
                                        
                                        kura_timeline_container = ui.column().classes('w-full')
                                        
                                        def update_kuramoto_timeline():
                                            kura_timeline_container.clear()
                                            data = viz_state.get('data')
                                            if not data:
                                                with kura_timeline_container:
                                                    ui.label('Load data first').style(f'color:{THEME_TEXT_DIM}; font-size: 0.75rem;')
                                                return
                                            try:
                                                from viz_scripts import kuramoto_viz
                                                with kura_timeline_container:
                                                    fig = kuramoto_viz.create_timeline_figure(data, viz_band.value)
                                                    ui.plotly(fig).classes('w-full').style('height: 280px;')
                                            except Exception as e:
                                                with kura_timeline_container:
                                                    ui.label(f'Error: {e}').style(f'color:{THEME_TEXT_DIM};')
                                        
                                        def show_all_bands():
                                            kura_timeline_container.clear()
                                            try:
                                                from viz_scripts import kuramoto_viz
                                                data = viz_state.get('data')
                                                with kura_timeline_container:
                                                    fig = kuramoto_viz.create_all_bands_timeline(data)
                                                    ui.plotly(fig).classes('w-full').style('height: 280px;')
                                            except Exception as e:
                                                with kura_timeline_container:
                                                    ui.label(f'Error: {e}').style(f'color:{THEME_TEXT_DIM};')
                                        
                                        ui.button('All Bands', on_click=show_all_bands).props('dense flat size=sm').classes('mb-1')
                                        if viz_state.get('data'):
                                            update_kuramoto_timeline()
                                    
                                    # Band Comparison
                                    with ui.card().classes('dark-card p-3 flex-1'):
                                        ui.label('▌BAND COMPARISON').style(f'color:{THEME_SECONDARY}; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-2')
                                        
                                        band_comp_container = ui.column().classes('w-full')
                                        
                                        def update_band_comparison():
                                            band_comp_container.clear()
                                            data = viz_state.get('data')
                                            if not data:
                                                with band_comp_container:
                                                    ui.label('Load data first').style(f'color:{THEME_TEXT_DIM}; font-size: 0.75rem;')
                                                return
                                            try:
                                                from viz_scripts import kuramoto_viz
                                                with band_comp_container:
                                                    fig = kuramoto_viz.create_band_comparison_figure(data)
                                                    ui.plotly(fig).classes('w-full').style('height: 280px;')
                                            except Exception as e:
                                                with band_comp_container:
                                                    ui.label(f'Error: {e}').style(f'color:{THEME_TEXT_DIM};')
                                        
                                        if viz_state.get('data'):
                                            update_band_comparison()
                                
                                # THIRD ROW - More Analysis
                                with ui.row().classes('w-full gap-3'):
                                    # Phase Distribution
                                    with ui.card().classes('dark-card p-3 flex-1'):
                                        ui.label('▌PHASE DISTRIBUTION').style(f'color:#a78bfa; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-2')
                                        
                                        phase_container = ui.column().classes('w-full')
                                        
                                        def update_phase_plot():
                                            phase_container.clear()
                                            data = viz_state.get('data')
                                            if not data:
                                                with phase_container:
                                                    ui.label('Load data first').style(f'color:{THEME_TEXT_DIM}; font-size: 0.75rem;')
                                                return
                                            try:
                                                from viz_scripts import kuramoto_viz
                                                with phase_container:
                                                    fig = kuramoto_viz.create_phase_distribution_figure(data, viz_band.value, int(viz_epoch.value or 0))
                                                    ui.plotly(fig).classes('w-full').style('height: 280px;')
                                            except Exception as e:
                                                with phase_container:
                                                    ui.label(f'Error: {e}').style(f'color:{THEME_TEXT_DIM};')
                                        
                                        if viz_state.get('data'):
                                            update_phase_plot()
                                    
                                    # Sync Matrix
                                    with ui.card().classes('dark-card p-3 flex-1'):
                                        ui.label('▌SYNC MATRIX').style(f'color:#60a5fa; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-2')
                                        
                                        sync_matrix_container = ui.column().classes('w-full')
                                        
                                        def update_sync_matrix():
                                            sync_matrix_container.clear()
                                            data = viz_state.get('data')
                                            if not data:
                                                with sync_matrix_container:
                                                    ui.label('Load data first').style(f'color:{THEME_TEXT_DIM}; font-size: 0.75rem;')
                                                return
                                            try:
                                                from viz_scripts import kuramoto_viz
                                                with sync_matrix_container:
                                                    fig = kuramoto_viz.create_heatmap_figure(data, viz_band.value, int(viz_epoch.value or 0))
                                                    ui.plotly(fig).classes('w-full').style('height: 280px;')
                                            except Exception as e:
                                                with sync_matrix_container:
                                                    ui.label(f'Error: {e}').style(f'color:{THEME_TEXT_DIM};')
                                        
                                        if viz_state.get('data'):
                                            update_sync_matrix()
                                    
                                    # Connectivity Graph
                                    with ui.card().classes('dark-card p-3 flex-1'):
                                        ui.label('▌ROI CONNECTIVITY').style(f'color:#22c55e; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-2')
                                        
                                        connectivity_container = ui.column().classes('w-full')
                                        conn_threshold = ui.slider(min=0.3, max=0.9, step=0.1, value=0.5).props('label-always').classes('w-full')
                                        
                                        def update_connectivity():
                                            connectivity_container.clear()
                                            data = viz_state.get('data')
                                            if not data:
                                                with connectivity_container:
                                                    ui.label('Load data first').style(f'color:{THEME_TEXT_DIM}; font-size: 0.75rem;')
                                                return
                                            try:
                                                from viz_scripts import kuramoto_viz
                                                with connectivity_container:
                                                    fig = kuramoto_viz.create_roi_connectivity_figure(data, viz_band.value, conn_threshold.value)
                                                    ui.plotly(fig).classes('w-full').style('height: 250px;')
                                            except Exception as e:
                                                with connectivity_container:
                                                    ui.label(f'Error: {e}').style(f'color:{THEME_TEXT_DIM};')
                                        
                                        conn_threshold.on('update:model-value', lambda e: update_connectivity())
                                        if viz_state.get('data'):
                                            update_connectivity()
                                
                                # FOURTH ROW - Hilbert Transform Visualizations
                                with ui.row().classes('w-full gap-3'):
                                    # Hilbert 2D
                                    with ui.card().classes('dark-card p-3 flex-1'):
                                        ui.label('▌HILBERT 2D').style(f'color:#f472b6; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-2')
                                        
                                        hilbert_2d_container = ui.column().classes('w-full')
                                        
                                        def update_hilbert_2d():
                                            hilbert_2d_container.clear()
                                            data = viz_state.get('data')
                                            if not data:
                                                with hilbert_2d_container:
                                                    ui.label('Load data first').style(f'color:{THEME_TEXT_DIM}; font-size: 0.75rem;')
                                                return
                                            try:
                                                from viz_scripts import kuramoto_viz
                                                with hilbert_2d_container:
                                                    fig = kuramoto_viz.create_hilbert_2d_figure(data, viz_band.value, int(viz_epoch.value or 0))
                                                    ui.plotly(fig).classes('w-full').style('height: 500px;')
                                            except Exception as e:
                                                with hilbert_2d_container:
                                                    ui.label(f'Error: {e}').style(f'color:{THEME_TEXT_DIM};')
                                        
                                        # Store reference for later updates
                                        viz_state['update_hilbert_2d'] = update_hilbert_2d
                                        if viz_state.get('data'):
                                            update_hilbert_2d()
                                    
                                    # Hilbert 3D
                                    with ui.card().classes('dark-card p-3 flex-1'):
                                        ui.label('▌HILBERT 3D PHASE SPACE').style(f'color:#c084fc; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-2')
                                        
                                        hilbert_3d_container = ui.column().classes('w-full')
                                        
                                        def update_hilbert_3d():
                                            hilbert_3d_container.clear()
                                            data = viz_state.get('data')
                                            if not data:
                                                with hilbert_3d_container:
                                                    ui.label('Load data first').style(f'color:{THEME_TEXT_DIM}; font-size: 0.75rem;')
                                                return
                                            try:
                                                from viz_scripts import kuramoto_viz
                                                subj = viz_subject.value or 'S01'
                                                # Extract condition from subject (S01-DMT -> DMT)
                                                cond = 'DMT'
                                                if '-' in str(subj):
                                                    cond = str(subj).split('-')[-1]
                                                
                                                with hilbert_3d_container:
                                                    fig = kuramoto_viz.create_hilbert_3d_figure(
                                                        data, 
                                                        viz_band.value, 
                                                        int(viz_epoch.value or 0),
                                                        roi_idx=0,
                                                        subject=subj,
                                                        condition=cond
                                                    )
                                                    ui.plotly(fig).classes('w-full').style('height: 500px;')
                                            except Exception as e:
                                                with hilbert_3d_container:
                                                    ui.label(f'Error: {e}').style(f'color:{THEME_TEXT_DIM};')
                                        
                                        # Store reference for later updates
                                        viz_state['update_hilbert_3d'] = update_hilbert_3d
                                        if viz_state.get('data'):
                                            update_hilbert_3d()
                                
                                # FIFTH ROW - Clustering (if available)
                                with ui.expansion('CLUSTERING ANALYSIS', icon='analytics').classes('w-full').style(f'background:{THEME_CARD};'):
                                    with ui.row().classes('w-full gap-3 p-2'):
                                        # Clustering Scores
                                        with ui.card().classes('dark-card p-3 flex-1'):
                                            ui.label('▌CLUSTER SCORES').style(f'color:#ec4899; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-2')
                                            
                                            cluster_scores_container = ui.column().classes('w-full')
                                            
                                            def update_cluster_scores():
                                                cluster_scores_container.clear()
                                                try:
                                                    from viz_scripts import clustering_viz
                                                    if current_run_dir[0]:
                                                        with cluster_scores_container:
                                                            fig = clustering_viz.create_band_comparison_figure(current_run_dir[0])
                                                            ui.plotly(fig).classes('w-full').style('height: 280px;')
                                                    else:
                                                        with cluster_scores_container:
                                                            ui.label('Select a run first').style(f'color:{THEME_TEXT_DIM};')
                                                except Exception as e:
                                                    with cluster_scores_container:
                                                        ui.label(f'Error: {e}').style(f'color:{THEME_TEXT_DIM};')
                                            
                                            ui.button('Load', on_click=update_cluster_scores).props('dense flat size=sm').classes('mb-2')
                                        
                                        # PCA Scatter
                                        with ui.card().classes('dark-card p-3 flex-1'):
                                            ui.label('▌PCA CLUSTERS').style(f'color:#f97316; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-2')
                                            
                                            pca_container = ui.column().classes('w-full')
                                            
                                            def update_pca_scatter():
                                                pca_container.clear()
                                                try:
                                                    from viz_scripts import clustering_viz
                                                    if current_run_dir[0]:
                                                        with pca_container:
                                                            fig = clustering_viz.create_pca_scatter_figure(current_run_dir[0], viz_band.value)
                                                            ui.plotly(fig).classes('w-full').style('height: 280px;')
                                                    else:
                                                        with pca_container:
                                                            ui.label('Select a run first').style(f'color:{THEME_TEXT_DIM};')
                                                except Exception as e:
                                                    with pca_container:
                                                        ui.label(f'Error: {e}').style(f'color:{THEME_TEXT_DIM};')
                                            
                                            ui.button('Load', on_click=update_pca_scatter).props('dense flat size=sm').classes('mb-2')
                                
                                # FIFTH ROW - Pearson Results Gallery
                                with ui.expansion('PEARSON CORRELATIONS', icon='insights').classes('w-full').style(f'background:{THEME_CARD};'):
                                    with ui.column().classes('w-full p-2'):
                                        with ui.row().classes('gap-2 items-center mb-2'):
                                            prs_metric = ui.select(['All', 'Coherence', 'Metastability'], value='All', label='Metric').props('dense').classes('w-28')
                                            prs_cond = ui.select(['All', 'DMT', 'EC', 'EO'], value='All', label='Condition').props('dense').classes('w-20')
                                            prs_band = ui.select(['All', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'], value='All', label='Band').props('dense').classes('w-24')
                                        
                                        pearson_gallery = ui.column().classes('w-full')
                                        pearson_image_dialog = ui.dialog().classes('w-full max-w-4xl')
                                        
                                        def refresh_pearson_gallery():
                                            pearson_gallery.clear()
                                            if not current_run_dir[0]:
                                                with pearson_gallery:
                                                    ui.label('No run selected').style(f'color:{THEME_TEXT_DIM};')
                                                return
                                            
                                            pearson_dir = current_run_dir[0] / 'pearson_results'
                                            if not pearson_dir.exists():
                                                with pearson_gallery:
                                                    ui.label('No Pearson results. Run pearson.py first.').style(f'color:{THEME_TEXT_DIM};')
                                                return
                                            
                                            all_files = list(pearson_dir.glob('*.png')) + list(pearson_dir.glob('*.svg'))
                                            filtered = [f for f in all_files if 
                                                (prs_metric.value == 'All' or prs_metric.value.lower() in f.name.lower()) and
                                                (prs_cond.value == 'All' or prs_cond.value.lower() in f.name.lower()) and
                                                (prs_band.value == 'All' or prs_band.value.lower() in f.name.lower())]
                                            
                                            with pearson_gallery:
                                                ui.label(f'{len(filtered)} images').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                                                with ui.element('div').classes('grid gap-3 mt-2').style('grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));'):
                                                    for img_file in sorted(filtered)[:30]:
                                                        def show_full_image(p=img_file):
                                                            pearson_image_dialog.clear()
                                                            with pearson_image_dialog:
                                                                with ui.column().classes('w-full items-center'):
                                                                    ui.label(p.name).style(f'color:{THEME_PRIMARY}; font-size: 0.9rem; margin-bottom: 10px;')
                                                                    if p.suffix == '.png':
                                                                        ui.image(str(p)).classes('w-full').style('max-height: 70vh;')
                                                                    else:
                                                                        ui.html(f'<object data="{p}" type="image/svg+xml" style="width:100%; max-height: 70vh;"></object>', sanitize=False)
                                                                    ui.button('Close', on_click=pearson_image_dialog.close).props('flat').classes('mt-3')
                                                            pearson_image_dialog.open()
                                                        
                                                        with ui.card().classes('cursor-pointer p-2').style(f'background:{THEME_CARD};').on('click', show_full_image):
                                                            ui.label(img_file.stem[:30] + ('...' if len(img_file.stem) > 30 else '')).style(f'color:{THEME_TEXT}; font-size: 0.65rem;')
                                        
                                        for sel in [prs_metric, prs_cond, prs_band]:
                                            sel.on('update:model-value', lambda e: refresh_pearson_gallery())
                                        
                                        ui.button('Load Images', on_click=refresh_pearson_gallery, icon='refresh').props('dense flat').classes('mt-2')
                                
                                # SEVENTH ROW - Animation/Frame Generator
                                with ui.expansion('ANIMATION GENERATOR', icon='movie').classes('w-full').style(f'background:{THEME_CARD};'):
                                    with ui.column().classes('w-full p-3 gap-3'):
                                        ui.label('Generate Kuramoto visualization frames and animations').style(f'color:{THEME_TEXT_DIM}; font-size: 0.75rem;')
                                        
                                        with ui.row().classes('gap-4 items-center flex-wrap'):
                                            anim_mode = ui.select(['stc', 'eeg', 'all', 'advanced'], value='stc', label='Mode').props('dense').classes('w-28')
                                            anim_quality = ui.select(['high', 'medium', 'low'], value='medium', label='Quality').props('dense').classes('w-24')
                                            anim_format = ui.select(['png', 'jpg'], value='png', label='Format').props('dense').classes('w-20')
                                            anim_start_epoch = ui.number(value=0, min=0, max=100, label='Start').props('dense').classes('w-20')
                                            anim_end_epoch = ui.number(value=10, min=1, max=200, label='End').props('dense').classes('w-20')
                                            anim_fps = ui.number(value=5, min=1, max=30, label='FPS').props('dense').classes('w-16')
                                        
                                        with ui.row().classes('gap-4 items-center'):
                                            anim_output_dir = ui.input(value='', placeholder='/path/to/output or auto').props('dense').classes('flex-1')
                                            ui.label('Output dir (leave empty for auto)').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                                        
                                        anim_log = ui.column().classes('w-full').style('max-height: 150px; overflow-y: auto; background: #050505; border-radius: 4px; padding: 8px;')
                                        
                                        async def generate_frames():
                                            """Generate visualization frames"""
                                            anim_log.clear()
                                            with anim_log:
                                                ui.label('Starting frame generation...').style(f'color:{THEME_PRIMARY}; font-size: 0.7rem;')
                                            
                                            data_path = viz_state.get('custom_path') or current_run_dir[0]
                                            if not data_path or not viz_subject.value:
                                                with anim_log:
                                                    ui.label('Error: No data loaded. Load a subject first.').style(f'color:{THEME_ERROR}; font-size: 0.7rem;')
                                                return
                                            
                                            # Build command
                                            import subprocess
                                            from pathlib import Path
                                            
                                            # Use plot.py directly (generate_frames.py is deprecated)
                                            script_path = Path('/media/storage_hdd/dmt_fz/viz_scripts/plot.py')
                                            
                                            # Use conda environment python
                                            import os
                                            conda_prefix = os.environ.get('CONDA_PREFIX', os.path.expanduser('~/anaconda3/envs/dmt_fz'))
                                            python_path = Path(conda_prefix) / 'bin' / 'python'
                                            if not python_path.exists():
                                                python_path = 'python'  # Fallback
                                            
                                            # Extract condition from subject if present (e.g., S01-DMT -> DMT)
                                            subj = viz_subject.value or 'S01'
                                            cond = 'DMT'
                                            if '-' in str(subj):
                                                parts = str(subj).split('-')
                                                subj = parts[0]
                                                cond = parts[1] if len(parts) > 1 else 'DMT'
                                            
                                            # Map mode names
                                            mode_map = {'stc': 'stc', 'eeg': 'eeg', 'all': 'all', 'advanced': 'advanced'}
                                            mode = mode_map.get(anim_mode.value, 'stc')
                                            
                                            # Build epochs range
                                            start_ep = int(anim_start_epoch.value or 0)
                                            end_ep = int(anim_end_epoch.value or 10)
                                            
                                            cmd = [
                                                str(python_path), str(script_path),
                                                '--subject', subj,
                                                '--condition', cond,
                                                '--band', viz_band.value,
                                                '--mode', mode,
                                                '--epochs', f'{start_ep}:{end_ep}',
                                            ]
                                            
                                            # plot.py saves frames to: visualizations/plot/{mode}/{subj}_{cond}_{band}
                                            band = viz_band.value or 'Alpha'
                                            expected_output = Path('/media/storage_hdd/dmt_fz/visualizations/plot') / mode / f'{subj}_{cond}_{band}'
                                            
                                            with anim_log:
                                                ui.label(f'Command: {" ".join(cmd)}').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                                                ui.label(f'Frames will be saved to: {expected_output}').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                                            
                                            try:
                                                import os
                                                env = os.environ.copy()
                                                env['PIPELINE_OUTPUT_DIR'] = str(data_path)
                                                # Add required paths for viz_scripts modules
                                                base_path = Path('/media/storage_hdd/dmt_fz')
                                                pythonpath = [
                                                    str(base_path / 'viz_scripts'),
                                                    str(base_path / 'pipeline'),
                                                    str(base_path),
                                                ]
                                                existing_pythonpath = env.get('PYTHONPATH', '')
                                                env['PYTHONPATH'] = ':'.join(pythonpath) + (':' + existing_pythonpath if existing_pythonpath else '')
                                                
                                                process = await asyncio.create_subprocess_exec(
                                                    *cmd,
                                                    stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.PIPE,
                                                    env=env
                                                )
                                                
                                                async def read_output(stream):
                                                    while True:
                                                        line = await stream.readline()
                                                        if not line:
                                                            break
                                                        text = line.decode().strip()
                                                        if text:
                                                            with anim_log:
                                                                ui.label(text).style(f'color:{THEME_TEXT}; font-size: 0.65rem;')
                                                
                                                await asyncio.gather(
                                                    read_output(process.stdout),
                                                    read_output(process.stderr)
                                                )
                                                
                                                await process.wait()
                                                
                                                with anim_log:
                                                    if process.returncode == 0:
                                                        ui.label('✓ Frames generated successfully!').style(f'color:{THEME_PRIMARY}; font-size: 0.75rem;')
                                                        # Show preview of first frame
                                                        frames = sorted(expected_output.glob(f'*.{anim_format.value}'))
                                                        if frames:
                                                            ui.label(f'Generated {len(frames)} frames').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                                                            # Show first frame as preview
                                                            with ui.card().classes('mt-2 p-2').style('background: #1a1a1a;'):
                                                                ui.label('Preview (first frame):').style(f'color:{THEME_TEXT_DIM}; font-size: 0.6rem;')
                                                                ui.image(str(frames[0])).classes('w-full').style('max-height: 300px; object-fit: contain;')
                                                    else:
                                                        ui.label(f'Process exited with code {process.returncode}').style(f'color:{THEME_WARN}; font-size: 0.7rem;')
                                            except Exception as e:
                                                with anim_log:
                                                    ui.label(f'Error: {e}').style(f'color:{THEME_ERROR}; font-size: 0.7rem;')
                                        
                                        async def generate_video():
                                            """Generate video from frames"""
                                            anim_log.clear()
                                            with anim_log:
                                                ui.label('🎬 Generating video from frames...').style(f'color:{THEME_PRIMARY}; font-size: 0.7rem;')
                                            
                                            from pathlib import Path
                                            
                                            # Build the correct frames path based on plot.py output structure
                                            # plot.py saves to: visualizations/plot/{mode}/{subject}_{condition}_{band}
                                            subj = viz_subject.value or 'S01'
                                            cond = 'DMT'
                                            if '-' in str(subj):
                                                parts = str(subj).split('-')
                                                subj = parts[0]
                                                cond = parts[1] if len(parts) > 1 else 'DMT'
                                            
                                            mode = anim_mode.value or 'stc'
                                            band = viz_band.value or 'Alpha'
                                            
                                            # Check custom path first, then default visualizations path
                                            if anim_output_dir.value and anim_output_dir.value.strip():
                                                frames_dir = Path(anim_output_dir.value.strip())
                                            else:
                                                frames_dir = Path('/media/storage_hdd/dmt_fz/visualizations/plot') / mode / f'{subj}_{cond}_{band}'
                                            
                                            with anim_log:
                                                ui.label(f'Looking for frames in: {frames_dir}').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                                            
                                            if not frames_dir.exists():
                                                with anim_log:
                                                    ui.label(f'Frames directory not found: {frames_dir}').style(f'color:{THEME_ERROR}; font-size: 0.7rem;')
                                                return
                                            
                                            output_video = frames_dir.parent / f'{viz_subject.value}_{viz_band.value}_animation.mp4'
                                            
                                            # Use ffmpeg to create video
                                            cmd = [
                                                'ffmpeg', '-y',
                                                '-framerate', str(int(anim_fps.value or 5)),
                                                '-pattern_type', 'glob',
                                                '-i', str(frames_dir / f'*.{anim_format.value}'),
                                                '-c:v', 'libx264',
                                                '-pix_fmt', 'yuv420p',
                                                str(output_video)
                                            ]
                                            
                                            with anim_log:
                                                ui.label(f'Running: ffmpeg -> {output_video.name}').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                                            
                                            try:
                                                process = await asyncio.create_subprocess_exec(
                                                    *cmd,
                                                    stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.PIPE
                                                )
                                                _, stderr = await process.communicate()
                                                
                                                with anim_log:
                                                    if process.returncode == 0:
                                                        ui.label(f'✓ Video saved: {output_video}').style(f'color:{THEME_PRIMARY}; font-size: 0.75rem;')
                                                        # Show video player
                                                        if output_video.exists():
                                                            with ui.card().classes('mt-2 p-2 w-full').style('background: #1a1a1a;'):
                                                                ui.label('Generated video:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.6rem;')
                                                                ui.video(str(output_video)).classes('w-full').style('max-height: 400px;')
                                                    else:
                                                        ui.label(f'ffmpeg error: {stderr.decode()[:200]}').style(f'color:{THEME_ERROR}; font-size: 0.65rem;')
                                            except Exception as e:
                                                with anim_log:
                                                    ui.label(f'Error: {e}').style(f'color:{THEME_ERROR}; font-size: 0.7rem;')
                                        
                                        with ui.row().classes('gap-2'):
                                            ui.button('Generate Frames', on_click=generate_frames, icon='photo_library').props('dense').style(f'background:{THEME_PRIMARY}; color:black;')
                                            ui.button('Create Video', on_click=generate_video, icon='movie').props('dense').style(f'background:{THEME_SECONDARY}; color:black;')
                                            
                                            def clear_log():
                                                anim_log.clear()
                                            ui.button('Clear Log', on_click=clear_log, icon='delete').props('dense flat')


# =============================================================================
# MODEL TRAINING PAGE
# =============================================================================

class ModelState:
    """State for model training page."""
    def __init__(self):
        self.dataset_path = ""
        self.dataset_info = {}
        self.dataset_type = None  # 'graph', 'image', etc.
        self.model_type = "vae"
        self.training = False
        self.current_process = None
        self.history = {'train_loss': [], 'val_loss': [], 'recon_loss': [], 'kl_loss': [], 'epoch': []}
        self.log_container = None
        self.loss_plot = None
        self.config = {}
        # Persistence for tab switching
        self.status = 'idle'  # 'idle', 'training', 'completed', 'error'
        self.log_history = []  # Store log messages
        self.status_indicator = None
        self.update_plots = None
        self.tabs = None
        self.tab_console = None

MS = ModelState()


class AnalysisState:
    """State for model analysis page."""
    def __init__(self):
        # Model
        self.model = None
        self.model_path = ""
        self.model_config = {}
        self.model_params = {}
        self.device = 'cpu'
        self.model_type = 'graph'  # 'graph' or 'image'
        
        # Dataset
        self.dataset = None
        self.dataset_path = ""
        self.current_split = 'test'  # train, val, test
        self.current_idx = 0
        self.total_samples = 0
        self.dataset_type = 'graph'  # 'graph' or 'image'
        self.class_names = []  # For image datasets with classes
        
        # Playback
        self.playing = False
        self.speed = 1.0  # samples per second
        self.play_timer = None
        
        # Activations (stored during forward pass)
        self.activations = {}  # layer_name -> tensor
        self.attention_weights = {}  # layer_name -> attention matrix
        self.latent_codes = []  # history of z vectors for PCA
        self.latent_labels = []  # labels for each z
        
        # Current sample info
        self.current_sample = None
        self.current_recon = None
        self.current_z = None
        self.current_label = None  # For image classification
        
        # Fixed axis ranges (computed from dataset)
        self.axis_ranges = {
            'node_features': {'min': -5, 'max': 5},
            'latent': {'x_min': -5, 'x_max': 5, 'y_min': -5, 'y_max': 5},
            'attention': {'min': 0, 'max': 1},
            'activations': {'min': -500, 'max': 500},
            'diff': {'min': 0, 'max': 1},
            'kuramoto': {'min': 0, 'max': 1}
        }
        self.ranges_computed = False
        
        # Kuramoto tracking (for graph models)
        self.kuramoto_history = []  # (idx, kuramoto_mean) for each processed sample
        self.kuramoto_avg = 0.5  # Average across test set
        self.kuramoto_std = 0.1  # Std (metastability proxy)
        
        # UI references
        self.log_container = None
        self.arch_diagram = None
        self.activation_plots = {}
        self.attention_plots = {}
        self.latent_plot = None
        self.recon_plot = None
        self.progress_slider = None
        self.sample_info_container = None

AS = AnalysisState()

# Autoencoder paths
AUTOENCODER_DIR = Path(__file__).parent.parent / "machine_learning" / "autoencoder"
AUTOENCODER_CACHE_DIR = Path(__file__).parent / "cache" / "autoencoder"


def detect_dataset_type(path: Path) -> dict:
    """
    Detect dataset type and structure from a given path.
    
    Returns dict with:
        - type: 'graph', 'image', 'unknown'
        - structure: 'single_file', 'split_files', 'hierarchical'
        - conditions: list of detected conditions (e.g., ['DMT', 'EC', 'EO'])
        - file_count: number of data files
        - sample_file: path to a sample file
        - data_sources: what data is available (phases, syncro, etc.)
        - has_stc: whether STC (source localized) data is available
        - has_eeg: whether EEG data is available
        - num_nodes: number of nodes (channels/parcels)
    """
    info = {
        'type': 'unknown',
        'structure': 'unknown',
        'conditions': [],
        'file_count': 0,
        'sample_file': None,
        'bands': [],
        'data_sources': [],
        'has_stc': False,
        'has_eeg': False,
        'num_nodes_eeg': 0,
        'num_nodes_stc': 0,
        'num_epochs_sample': 0,
        'error': None
    }
    
    if not path.exists():
        info['error'] = f"Path does not exist: {path}"
        return info
    
    # Check for phases-*.pkl files (graph data from pipeline)
    phases_files = list(path.rglob("phases-*.pkl"))
    syncro_files = list(path.rglob("syncro-*.pkl"))
    order_files = list(path.rglob("order-*.pkl"))
    
    if phases_files:
        info['type'] = 'graph'
        info['file_count'] = len(phases_files)
        info['sample_file'] = str(phases_files[0])
        info['data_sources'].append('phases (syncro + phases + amplitudes + kuramoto)')
        
        # Detect conditions from folder structure
        conditions = set()
        for f in phases_files:
            parent = f.parent.name
            if parent in ['DMT', 'EC', 'EO']:
                conditions.add(parent)
        info['conditions'] = sorted(list(conditions)) if conditions else ['Unknown']
        
        # Check structure
        subdirs = [d for d in path.iterdir() if d.is_dir()]
        info['structure'] = 'hierarchical' if subdirs else 'flat'
        
        # Analyze sample file for detailed info
        try:
            import pickle
            with open(info['sample_file'], 'rb') as f:
                data = pickle.load(f)
            
            # Check what data is available
            if 'phases_stc' in data:
                info['has_stc'] = True
                info['bands'] = list(data['phases_stc'].keys())
                # Get number of parcels from first band, first epoch
                first_band = info['bands'][0]
                if data['phases_stc'][first_band]:
                    info['num_nodes_stc'] = data['phases_stc'][first_band][0].shape[0]
                    info['num_epochs_sample'] = len(data['phases_stc'][first_band])
            
            if 'phases_eeg' in data:
                info['has_eeg'] = True
                if not info['bands']:
                    info['bands'] = list(data['phases_eeg'].keys())
                first_band = info['bands'][0]
                if data['phases_eeg'][first_band]:
                    info['num_nodes_eeg'] = data['phases_eeg'][first_band][0].shape[0]
                    if info['num_epochs_sample'] == 0:
                        info['num_epochs_sample'] = len(data['phases_eeg'][first_band])
            
            # Check what else is in the file
            available_keys = list(data.keys())
            if 'syncros_stc' in data or 'syncros_eeg' in data:
                if 'syncro' not in str(info['data_sources']):
                    pass  # Already included in phases
            if 'kuramoto_stc' in data or 'kuramoto_eeg' in data:
                pass  # Already included in phases
                
        except Exception as e:
            info['error'] = f"Could not analyze sample file: {e}"
        
        return info
    
    # Fallback: check for standalone syncro files
    if syncro_files:
        info['type'] = 'graph'
        info['file_count'] = len(syncro_files)
        info['sample_file'] = str(syncro_files[0])
        info['data_sources'].append('syncro (only synchronization matrices)')
        
        conditions = set()
        for f in syncro_files:
            parent = f.parent.name
            if parent in ['DMT', 'EC', 'EO']:
                conditions.add(parent)
        info['conditions'] = sorted(list(conditions)) if conditions else ['Unknown']
        
        return info
    
    # Check for order files
    if order_files:
        info['type'] = 'order'
        info['file_count'] = len(order_files)
        info['sample_file'] = str(order_files[0])
        info['data_sources'].append('order (Kuramoto order parameter)')
        info['error'] = "Order files contain pre-computed Kuramoto values, not suitable for VAE training. Use phases-*.pkl files instead."
        return info
    
    # Check for image files
    image_files = list(path.rglob("*.png")) + list(path.rglob("*.jpg")) + list(path.rglob("*.jpeg"))
    if image_files:
        info['type'] = 'image'
        info['file_count'] = len(image_files)
        info['structure'] = 'flat' if not any(d.is_dir() for d in path.iterdir()) else 'hierarchical'
        info['sample_file'] = str(image_files[0])
        return info
    
    # Check for numpy arrays
    npy_files = list(path.rglob("*.npy")) + list(path.rglob("*.npz"))
    if npy_files:
        info['type'] = 'array'
        info['file_count'] = len(npy_files)
        info['sample_file'] = str(npy_files[0])
        return info
    
    info['error'] = "No recognized data files found (phases-*.pkl, images, or numpy arrays)"
    return info


def model_log(msg: str, msg_type: str = 'info'):
    """Add message to model training log."""
    # Store in history for persistence
    MS.log_history.append((msg, msg_type))
    # Keep only last 500 messages
    if len(MS.log_history) > 500:
        MS.log_history = MS.log_history[-500:]
    
    if MS.log_container:
        colors = {
            'info': THEME_TEXT,
            'success': THEME_PRIMARY,
            'warning': THEME_WARN,
            'error': THEME_ERROR
        }
        with MS.log_container:
            ui.label(msg).style(f'color:{colors.get(msg_type, THEME_TEXT)}; font-family: JetBrains Mono; font-size: 0.75rem;')


def update_status_indicator(status: str):
    """Update the training status indicator."""
    MS.status = status
    if MS.status_indicator:
        MS.status_indicator.classes(replace=f'status-{status}')


def create_default_config(dataset_path: str, dataset_info: dict) -> dict:
    """Create default VAE configuration based on detected dataset."""
    config = {
        'paths': {
            'phases_dir': dataset_path,
            'output_dir': str(AUTOENCODER_CACHE_DIR / 'output'),
            'dataset_cache': str(AUTOENCODER_CACHE_DIR / 'dataset_cache'),
            'checkpoints': str(AUTOENCODER_CACHE_DIR / 'checkpoints'),
            'tensorboard': str(AUTOENCODER_CACHE_DIR / 'runs'),
        },
        'data': {
            'conditions': dataset_info.get('conditions', ['DMT', 'EC', 'EO']),
            'bands': dataset_info.get('bands', ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']),
            'use_stc': dataset_info.get('has_stc', False),
            'graph': {
                'fully_connected': True,
                'edge_threshold': 0.3,
                'edge_threshold_percentile': None,
                'max_edges_per_node': None,
                'self_loops': False,
                'directed': False,
            },
            'node_features': {
                'use_phase_stats': True,
                'use_amplitude_stats': True,
                'use_temporal_complexity': True,
                'use_network_label': False,
            },
            'graph_features': {
                'use_kuramoto': True,
                'use_global_sync': True,
                'use_topology': True,
            },
            'split': {
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'test_ratio': 0.15,
                'stratify': True,
                'random_state': 42,
                'group_by_subject': True,
            }
        },
        'model': {
            'name': 'BrainStateVAE',
            'encoder': {
                'conv_type': 'gatv2',
                'hidden_dim': 64,
                'num_gat_layers': 3,
                'num_attention_heads': 4,
                'cheby_k': 3,
                'dropout': 0.2,
                'attention_dropout': 0.1,
                'use_edge_attr': True,
                'concat_heads': True,
                'negative_slope': 0.2,
                'use_skip_connections': True,
            },
            'latent': {
                'dim': 64,
            },
            'decoder': {
                'hidden_dims': [256, 128],
                'reconstruct_edges': True,
                'activation': 'leaky_relu',
                'dropout': 0.1,
            },
            'pooling': {
                'method': 'mean',
            }
        },
        'loss': {
            'reconstruction': {
                'node_weight': 0.3,
                'edge_weight': 1.0,
                'type': 'mse',
            },
            'kl': {
                'weight': 0.01,
                'annealing': {
                    'enabled': True,
                    'start': 0.0,
                    'end': 0.05,
                    'epochs': 50,
                    'type': 'linear',
                }
            },
            'free_bits': 0.1,
        },
        'training': {
            'num_epochs': 100,
            'batch_size': 256,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'optimizer': 'adamw',
            'scheduler': {
                'type': 'cosine',
                'patience': 10,
                'factor': 0.5,
                'min_lr': 1e-6,
            },
            'early_stopping': {
                'patience': 20,
                'min_delta': 0.0001,
                'monitor': 'val_loss',
            },
            'gradient_clipping': {
                'enabled': True,
                'max_norm': 0.5,
            }
        },
        'logging': {
            'level': 'INFO',
            'tensorboard': True,
            'save_frequency': 10,
            'log_frequency': 1,
            'tensorboard_extras': {
                'latent_space': True,
                'reconstructions': True,
                'attention_weights': False,
                'kl_per_dim': True,
                'gradient_norms': False,
            }
        },
        'visualization': {
            'plot_training_curves': True,
            'plot_latent_space': True,
            'plot_reconstructions': True,
            'num_examples_to_visualize': 10,
            'latent_method': 'tsne',
        },
        'seed': 42,
        'deterministic': True,
        'device': 'cuda',  # Will fallback to CPU if not available
        'num_workers': 4,  # Use workers with GPU
        'dataset_workers': 4,
        'pin_memory': True,  # Enabled for GPU
    }
    return config


@ui.page('/model')
def model_page():
    """Model training page."""
    ui.add_head_html(f'<style>{STYLE}</style>')
    
    # Header with navigation
    with ui.header().classes('items-center px-4 py-1').style(f'background: {THEME_BG}; border-bottom: 1px solid {THEME_BORDER};'):
        ui.label('▶').style(f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.75rem; letter-spacing: 2px;')
        ui.label('EEG_VIEWER').classes('text-base font-medium ml-2').style(f'color: {THEME_PRIMARY}; font-family: JetBrains Mono; letter-spacing: 1px;')
        ui.label('// MODEL').classes('text-xs ml-2').style(f'color: #f472b6; font-family: JetBrains Mono;')
        
        # Status indicator
        with ui.row().classes('items-center gap-2 ml-4'):
            MS.status_indicator = ui.html('<div></div>', sanitize=False).classes(f'status-{MS.status}')
            status_labels = {'idle': 'IDLE', 'training': 'TRAINING...', 'completed': 'COMPLETED', 'error': 'ERROR'}
            status_colors = {'idle': THEME_TEXT_DIM, 'training': '#f59e0b', 'completed': '#10b981', 'error': '#ef4444'}
            ui.label(status_labels.get(MS.status, 'IDLE')).style(f'color:{status_colors.get(MS.status, THEME_TEXT_DIM)}; font-size: 0.7rem; font-family: JetBrains Mono;').bind_text_from(MS, 'status', lambda s: {'idle': 'IDLE', 'training': 'TRAINING...', 'completed': 'COMPLETED', 'error': 'ERROR'}.get(s, 'IDLE'))
        
        with ui.row().classes('ml-auto gap-2'):
            ui.button('VIEWER', on_click=lambda: ui.navigate.to('/')).props('flat dense').style(f'color:{THEME_TEXT_DIM};')
            ui.button('PIPELINE', on_click=lambda: ui.navigate.to('/pipeline')).props('flat dense').style(f'color:{THEME_TEXT_DIM};')
            ui.button('MODEL', on_click=lambda: ui.navigate.to('/model')).props('flat dense').style(f'color:#f472b6;')
            ui.button('ANALYSIS', on_click=lambda: ui.navigate.to('/analysis')).props('flat dense').style(f'color:{THEME_TEXT_DIM};')
    
    with ui.row().classes('w-full p-4 gap-4').style('height: calc(100vh - 50px); align-items: stretch;'):
        
        # LEFT PANEL: Configuration
        with ui.column().classes('gap-4').style('width: 400px; overflow-y: auto;'):
            
            # DATASET CONFIGURATION
            with ui.card().classes('dark-card p-4 w-full').style(f'border: 1px solid #f472b6;'):
                ui.label('// DATASET').classes('terminal-header')
                
                with ui.row().classes('items-center gap-2 mt-2 w-full'):
                    ui.label('Path:').style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.75rem; min-width: 50px;')
                    dataset_path_input = ui.input(
                        value='/media/storage_hdd/dmt_fz/fwd-inv-stc'
                    ).props('dense dark').classes('flex-1')
                
                dataset_info_container = ui.column().classes('w-full mt-2 gap-1')
                
                def scan_dataset():
                    """Scan and detect dataset."""
                    path = Path(dataset_path_input.value.strip())
                    MS.dataset_path = str(path)
                    
                    dataset_info_container.clear()
                    with dataset_info_container:
                        ui.label('Scanning...').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                    
                    info = detect_dataset_type(path)
                    MS.dataset_info = info
                    MS.dataset_type = info['type']
                    
                    dataset_info_container.clear()
                    with dataset_info_container:
                        if info.get('error'):
                            ui.label(f"❌ {info['error']}").style(f'color:{THEME_ERROR}; font-size: 0.7rem;')
                        else:
                            ui.label(f"✓ Type: {info['type'].upper()}").style(f'color:{THEME_PRIMARY}; font-size: 0.75rem;')
                            ui.label(f"  Files: {info['file_count']}").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                            
                            if info['conditions']:
                                ui.label(f"  Conditions: {', '.join(info['conditions'])}").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                            if info['bands']:
                                ui.label(f"  Bands: {', '.join(info['bands'])}").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                            
                            # Show data sources
                            if info['data_sources']:
                                ui.label(f"  Data: {info['data_sources'][0]}").style(f'color:{THEME_SECONDARY}; font-size: 0.7rem;')
                            
                            # Show EEG/STC availability
                            if info['has_eeg'] or info['has_stc']:
                                sources = []
                                if info['has_eeg']:
                                    sources.append(f"EEG ({info['num_nodes_eeg']} ch)")
                                if info['has_stc']:
                                    sources.append(f"STC ({info['num_nodes_stc']} parcels)")
                                ui.label(f"  Sources: {' | '.join(sources)}").style(f'color:{THEME_PRIMARY}; font-size: 0.7rem;')
                            
                            if info['num_epochs_sample'] > 0:
                                ui.label(f"  Epochs/subject: ~{info['num_epochs_sample']}").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                            
                            # Update source selector options
                            if info['has_eeg'] and info['has_stc']:
                                data_source_select.options = ['EEG (channels)', 'STC (parcels)']
                                data_source_select.value = 'STC (parcels)' if info['has_stc'] else 'EEG (channels)'
                            elif info['has_eeg']:
                                data_source_select.options = ['EEG (channels)']
                                data_source_select.value = 'EEG (channels)'
                            elif info['has_stc']:
                                data_source_select.options = ['STC (parcels)']
                                data_source_select.value = 'STC (parcels)'
                            
                            # Create default config
                            MS.config = create_default_config(str(path), info)
                            model_log(f"Dataset detected: {info['type']} ({info['file_count']} files)", 'success')
                
                ui.button('Scan Dataset', on_click=scan_dataset, icon='search').props('dense').classes('mt-2').style(f'background:#f472b6; color:black;')
                
                # Data source selector (EEG vs STC)
                ui.separator().classes('my-2')
                with ui.row().classes('items-center gap-2'):
                    ui.label('Source:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.75rem; min-width: 50px;')
                    data_source_select = ui.select(
                        ['EEG (channels)', 'STC (parcels)'],
                        value='EEG (channels)'
                    ).props('dense dark').classes('flex-1')
                
                ui.label('EEG: 24 electrodes | STC: ~200 brain parcels').style(f'color:{THEME_TEXT_DIM}; font-size: 0.6rem;')
                
                # Band selection
                ui.separator().classes('my-2')
                ui.label('Bands to use:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.75rem;')
                with ui.row().classes('gap-2 flex-wrap'):
                    band_checks = {}
                    for band in ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']:
                        band_checks[band] = ui.checkbox(band, value=(band == 'Alpha')).props('dense')
                
                ui.label('Tip: Start with 1-2 bands for faster training').style(f'color:{THEME_TEXT_DIM}; font-size: 0.6rem;')
                
                # Subsample option
                with ui.row().classes('items-center gap-2 mt-2'):
                    ui.label('Subsample:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.75rem; min-width: 70px;')
                    subsample_slider = ui.slider(min=0.1, max=1.0, step=0.1, value=0.3).props('label-always').classes('flex-1')
                
                ui.label('Use 0.1-0.3 for quick tests, 1.0 for full training').style(f'color:{THEME_TEXT_DIM}; font-size: 0.6rem;')
            
            # MODEL CONFIGURATION
            with ui.card().classes('dark-card p-4 w-full'):
                ui.label('// MODEL CONFIG').classes('terminal-header')
                
                with ui.column().classes('gap-1 mt-2'):
                    # Model type selector
                    with ui.row().classes('items-center gap-2'):
                        ui.label('Type:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem; min-width: 70px;')
                        model_type_select = ui.select(
                            ['VAE (Graph)', 'VAE (Image)', 'AE (Graph)'],
                            value='VAE (Graph)'
                        ).props('dense dark').classes('flex-1')
                    
                    ui.separator().classes('my-1')
                    
                    # Architecture params
                    ui.label('Architecture').style(f'color:{THEME_SECONDARY}; font-size: 0.65rem;')
                    
                    with ui.row().classes('items-center gap-2'):
                        ui.label('Latent:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem; min-width: 70px;')
                        latent_dim = ui.number(value=64, min=8, max=512, step=8).props('dense').classes('w-16')
                        ui.label('Hidden:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                        hidden_dim = ui.number(value=64, min=16, max=256, step=16).props('dense').classes('w-16')
                    
                    with ui.row().classes('items-center gap-2'):
                        ui.label('GAT layers:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem; min-width: 70px;')
                        gat_layers = ui.number(value=3, min=1, max=6).props('dense').classes('w-16')
                        ui.label('Heads:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                        attention_heads = ui.number(value=4, min=1, max=8).props('dense').classes('w-16')
                    
                    with ui.row().classes('items-center gap-2'):
                        ui.label('Dropout:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem; min-width: 70px;')
                        dropout = ui.number(value=0.2, min=0.0, max=0.5, step=0.05).props('dense').classes('w-16')
                        ui.label('Attn drop:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                        attn_dropout = ui.number(value=0.1, min=0.0, max=0.3, step=0.05).props('dense').classes('w-16')
                    
                    ui.separator().classes('my-1')
                    
                    # Training params
                    ui.label('Training').style(f'color:{THEME_SECONDARY}; font-size: 0.65rem;')
                    
                    with ui.row().classes('items-center gap-2'):
                        ui.label('Epochs:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem; min-width: 70px;')
                        num_epochs = ui.number(value=100, min=10, max=500, step=10).props('dense').classes('w-16')
                        ui.label('Batch:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                        batch_size = ui.number(value=256, min=16, max=1024, step=16).props('dense').classes('w-16')
                    
                    with ui.row().classes('items-center gap-2'):
                        ui.label('Learn rate:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem; min-width: 70px;')
                        learning_rate = ui.select(
                            ['1e-2', '5e-3', '1e-3', '5e-4', '1e-4'],
                            value='1e-3'
                        ).props('dense dark').classes('w-20')
                        ui.label('Decay:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                        weight_decay = ui.select(
                            ['0', '1e-5', '1e-4', '1e-3'],
                            value='1e-5'
                        ).props('dense dark').classes('w-20')
                    
                    with ui.row().classes('items-center gap-2'):
                        ui.label('Optimizer:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem; min-width: 70px;')
                        optimizer_select = ui.select(
                            ['adamw', 'adam', 'sgd'],
                            value='adamw'
                        ).props('dense dark').classes('w-20')
                        ui.label('Scheduler:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                        scheduler_select = ui.select(
                            ['cosine', 'reduce_on_plateau', 'step', 'none'],
                            value='cosine'
                        ).props('dense dark').classes('w-24')
                    
                    ui.separator().classes('my-1')
                    
                    # Early stopping & regularization
                    ui.label('Regularization').style(f'color:{THEME_SECONDARY}; font-size: 0.65rem;')
                    
                    with ui.row().classes('items-center gap-2'):
                        ui.label('Patience:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem; min-width: 70px;')
                        patience = ui.number(value=25, min=5, max=100, step=5).props('dense').classes('w-16')
                        ui.label('Grad clip:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                        grad_clip = ui.number(value=0.5, min=0.0, max=5.0, step=0.1).props('dense').classes('w-16')
                    
                    ui.separator().classes('my-1')
                    
                    # Loss params
                    ui.label('Loss').style(f'color:{THEME_SECONDARY}; font-size: 0.65rem;')
                    
                    with ui.row().classes('items-center gap-2'):
                        ui.label('KL weight:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem; min-width: 70px;')
                        kl_weight = ui.number(value=0.01, min=0.0, max=1.0, step=0.01).props('dense').classes('w-16')
                        ui.label('β anneal:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                        beta_annealing = ui.switch(value=True).props('dense')
                    
                    with ui.row().classes('items-center gap-2'):
                        ui.label('Node wt:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem; min-width: 70px;')
                        node_weight = ui.number(value=0.3, min=0.0, max=1.0, step=0.1).props('dense').classes('w-16')
                        ui.label('Edge wt:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                        edge_weight = ui.number(value=1.0, min=0.0, max=2.0, step=0.1).props('dense').classes('w-16')
            
            # TRAINING CONTROLS
            with ui.card().classes('dark-card p-4 w-full'):
                ui.label('// TRAINING').classes('terminal-header')
                
                training_status = ui.label('Ready').style(f'color:{THEME_TEXT_DIM}; font-size: 0.75rem;').classes('mt-2')
                
                with ui.row().classes('gap-2 mt-3'):
                    async def start_training():
                        if MS.training:
                            ui.notify('Training already in progress', type='warning')
                            return
                        
                        if not MS.dataset_path or not MS.dataset_info.get('type'):
                            ui.notify('Please scan a dataset first', type='warning')
                            return
                        
                        if MS.dataset_info.get('type') == 'order':
                            ui.notify('Order files not suitable for VAE. Use phases-*.pkl', type='error')
                            return
                        
                        # Update config with UI values - Architecture
                        MS.config['model']['latent']['dim'] = int(latent_dim.value)
                        MS.config['model']['encoder']['hidden_dim'] = int(hidden_dim.value)
                        MS.config['model']['encoder']['num_gat_layers'] = int(gat_layers.value)
                        MS.config['model']['encoder']['num_attention_heads'] = int(attention_heads.value)
                        MS.config['model']['encoder']['dropout'] = float(dropout.value)
                        MS.config['model']['encoder']['attention_dropout'] = float(attn_dropout.value)
                        
                        # Training params
                        MS.config['training']['num_epochs'] = int(num_epochs.value)
                        MS.config['training']['batch_size'] = int(batch_size.value)
                        MS.config['training']['learning_rate'] = float(learning_rate.value)
                        MS.config['training']['weight_decay'] = float(weight_decay.value) if weight_decay.value != '0' else 0.0
                        MS.config['training']['optimizer'] = optimizer_select.value
                        MS.config['training']['scheduler']['type'] = scheduler_select.value if scheduler_select.value != 'none' else None
                        MS.config['training']['early_stopping']['patience'] = int(patience.value)
                        MS.config['training']['gradient_clipping']['max_norm'] = float(grad_clip.value)
                        MS.config['training']['gradient_clipping']['enabled'] = grad_clip.value > 0
                        
                        # Loss params
                        MS.config['loss']['kl']['weight'] = float(kl_weight.value)
                        MS.config['loss']['kl']['annealing']['enabled'] = beta_annealing.value
                        MS.config['loss']['reconstruction']['node_weight'] = float(node_weight.value)
                        MS.config['loss']['reconstruction']['edge_weight'] = float(edge_weight.value)
                        
                        # Set data source (EEG vs STC)
                        use_stc = 'STC' in data_source_select.value
                        MS.config['data']['use_stc'] = use_stc
                        model_log(f"Using {'STC (parcels)' if use_stc else 'EEG (channels)'} data", 'info')
                        
                        # Set selected bands
                        selected_bands = [band for band, cb in band_checks.items() if cb.value]
                        if not selected_bands:
                            ui.notify('Select at least one band', type='warning')
                            return
                        MS.config['data']['bands'] = selected_bands
                        model_log(f"Bands: {', '.join(selected_bands)}", 'info')
                        
                        # Estimate dataset size
                        n_files = MS.dataset_info.get('file_count', 0)
                        n_epochs = MS.dataset_info.get('num_epochs_sample', 50)
                        n_bands = len(selected_bands)
                        estimated_graphs = n_files * n_epochs * n_bands
                        subsample = subsample_slider.value
                        final_estimate = int(estimated_graphs * subsample)
                        model_log(f"Estimated graphs: ~{final_estimate} (subsample={subsample:.0%})", 'info')
                        
                        # Check GPU availability
                        try:
                            import torch
                            if torch.cuda.is_available():
                                gpu_name = torch.cuda.get_device_name(0)
                                model_log(f"GPU: {gpu_name}", 'success')
                            else:
                                model_log("⚠ No GPU available - training on CPU (slower)", 'warning')
                        except:
                            model_log("⚠ Could not detect GPU", 'warning')
                        
                        # Create cache directories
                        AUTOENCODER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                        (AUTOENCODER_CACHE_DIR / 'output').mkdir(exist_ok=True)
                        (AUTOENCODER_CACHE_DIR / 'checkpoints').mkdir(exist_ok=True)
                        (AUTOENCODER_CACHE_DIR / 'runs').mkdir(exist_ok=True)
                        (AUTOENCODER_CACHE_DIR / 'dataset_cache').mkdir(exist_ok=True)
                        
                        # Save config to temp file
                        import yaml
                        config_path = AUTOENCODER_CACHE_DIR / 'train_config.yaml'
                        with open(config_path, 'w') as f:
                            yaml.dump(MS.config, f, default_flow_style=False)
                        
                        MS.training = True
                        MS.history = {'train_loss': [], 'val_loss': [], 'recon_loss': [], 'kl_loss': [], 'epoch': []}
                        MS.log_history = []  # Clear log history
                        
                        # Clear previous logs and reset plot
                        if MS.log_container:
                            MS.log_container.clear()
                        update_loss_plot()  # Reset the plot with empty data
                        
                        # Update status indicator
                        update_status_indicator('training')
                        
                        training_status.text = 'Training...'
                        training_status.style(f'color:{THEME_PRIMARY}; font-size: 0.75rem;')
                        model_log(f"Starting training with config: {config_path}", 'info')
                        
                        # Run training in subprocess
                        import subprocess
                        cmd = [
                            sys.executable,
                            '-u',  # Unbuffered output - critical for real-time logs
                            str(AUTOENCODER_DIR / 'train.py'),
                            '--config', str(config_path),
                            '--subsample', str(subsample_slider.value)
                        ]
                        
                        env = os.environ.copy()
                        env['PYTHONPATH'] = str(AUTOENCODER_DIR.parent)
                        env['PYTHONUNBUFFERED'] = '1'  # Force unbuffered output
                        
                        try:
                            process = await asyncio.create_subprocess_exec(
                                *cmd,
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.STDOUT,
                                env=env,
                                cwd=str(AUTOENCODER_DIR)
                            )
                            MS.current_process = process
                            
                            # Read output line by line
                            epoch_pattern = re.compile(r'Epoch (\d+).*Loss: ([\d.]+).*Recon: ([\d.]+).*KL: ([\d.]+)')
                            val_pattern = re.compile(r'Epoch \d+.*\[val\].*Loss: ([\d.]+)')
                            
                            while True:
                                line = await process.stdout.readline()
                                if not line:
                                    break
                                line = line.decode('utf-8', errors='replace').strip()
                                if line:
                                    model_log(line, 'info')
                                    
                                    # Parse training metrics
                                    epoch_match = epoch_pattern.search(line)
                                    if epoch_match and '[train]' in line:
                                        epoch = int(epoch_match.group(1))
                                        loss = float(epoch_match.group(2))
                                        recon = float(epoch_match.group(3))
                                        kl = float(epoch_match.group(4))
                                        
                                        MS.history['epoch'].append(epoch)
                                        MS.history['train_loss'].append(loss)
                                        MS.history['recon_loss'].append(recon)
                                        MS.history['kl_loss'].append(kl)
                                        
                                        # Update loss plot
                                        update_loss_plot()
                                    
                                    val_match = val_pattern.search(line)
                                    if val_match:
                                        val_loss = float(val_match.group(1))
                                        MS.history['val_loss'].append(val_loss)
                                        update_loss_plot()
                            
                            await process.wait()
                            
                            if process.returncode == 0:
                                model_log("Training completed successfully!", 'success')
                                training_status.text = 'Completed'
                                training_status.style(f'color:{THEME_PRIMARY}; font-size: 0.75rem;')
                                update_status_indicator('completed')
                            else:
                                model_log(f"Training failed with code {process.returncode}", 'error')
                                training_status.text = 'Failed'
                                training_status.style(f'color:{THEME_ERROR}; font-size: 0.75rem;')
                                update_status_indicator('error')
                        
                        except Exception as e:
                            model_log(f"Error: {e}", 'error')
                            training_status.text = 'Error'
                            training_status.style(f'color:{THEME_ERROR}; font-size: 0.75rem;')
                            update_status_indicator('error')
                        
                        finally:
                            MS.training = False
                            MS.current_process = None
                    
                    async def stop_training():
                        if MS.current_process:
                            MS.current_process.terminate()
                            model_log("Training stopped by user", 'warning')
                            training_status.text = 'Stopped'
                            training_status.style(f'color:{THEME_WARN}; font-size: 0.75rem;')
                            MS.training = False
                            update_status_indicator('idle')
                    
                    ui.button('Train', on_click=start_training, icon='play_arrow').props('dense').style(f'background:{THEME_PRIMARY}; color:black;')
                    ui.button('Stop', on_click=stop_training, icon='stop').props('dense color=negative')
        
        # RIGHT PANEL: Visualization & Logs
        with ui.column().classes('flex-1').style('min-height: 0; display: flex; flex-direction: column;'):
            
            with ui.card().classes('dark-card p-2 w-full flex-1').style('display: flex; flex-direction: column; min-height: 0;'):
                with ui.tabs().classes('w-full').style(f'background: {THEME_BG};') as model_tabs:
                    tab_metrics = ui.tab('METRICS', icon='show_chart').style(f'color:#f472b6;')
                    tab_recon = ui.tab('RECON', icon='compare').style(f'color:{THEME_SECONDARY};')
                    tab_console = ui.tab('CONSOLE', icon='terminal').style(f'color:{THEME_PRIMARY};')
                
                with ui.tab_panels(model_tabs, value=tab_console).classes('w-full').style('flex: 1; min-height: 0; overflow: hidden;'):
                    
                    # METRICS TAB
                    with ui.tab_panel(tab_metrics).classes('p-2').style('height: 100%; display: flex; flex-direction: column;'):
                        ui.label('▌TRAINING METRICS').style(f'color:#f472b6; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-2')
                        
                        loss_plot_container = ui.column().classes('w-full flex-1')
                        
                        def make_loss_figure():
                            """Create separate plots for different metrics."""
                            from plotly.subplots import make_subplots
                            
                            epochs = MS.history.get('epoch', [])
                            train_loss = MS.history.get('train_loss', [])
                            val_loss = MS.history.get('val_loss', [])
                            recon_loss = MS.history.get('recon_loss', [])
                            kl_loss = MS.history.get('kl_loss', [])
                            
                            # Create 1x3 subplot grid
                            fig = make_subplots(
                                rows=1, cols=3,
                                subplot_titles=('Total Loss', 'Recon Loss', 'KL Loss'),
                                horizontal_spacing=0.08
                            )
                            
                            if epochs:
                                # Plot 1: Train + Val Loss
                                fig.add_trace(go.Scatter(
                                    x=epochs, y=train_loss,
                                    mode='lines', name='Train',
                                    line=dict(color=THEME_PRIMARY, width=2)
                                ), row=1, col=1)
                                
                                if val_loss:
                                    fig.add_trace(go.Scatter(
                                        x=epochs[:len(val_loss)], y=val_loss,
                                        mode='lines', name='Val',
                                        line=dict(color='#f472b6', width=2)
                                    ), row=1, col=1)
                                
                                # Plot 2: Recon Loss
                                fig.add_trace(go.Scatter(
                                    x=epochs, y=recon_loss,
                                    mode='lines', name='Recon',
                                    line=dict(color=THEME_SECONDARY, width=2),
                                    showlegend=False
                                ), row=1, col=2)
                                
                                # Plot 3: KL Loss
                                fig.add_trace(go.Scatter(
                                    x=epochs, y=kl_loss,
                                    mode='lines', name='KL',
                                    line=dict(color=THEME_WARN, width=2),
                                    showlegend=False
                                ), row=1, col=3)
                            
                            fig.update_layout(
                                template='plotly_dark',
                                paper_bgcolor='rgba(8,8,8,1)',
                                plot_bgcolor='rgba(8,8,8,1)',
                                margin=dict(l=40, r=20, t=40, b=40),
                                height=280,
                                legend=dict(
                                    orientation='h',
                                    yanchor='bottom',
                                    y=1.08,
                                    xanchor='left',
                                    x=0
                                ),
                                font=dict(family='JetBrains Mono', size=10, color=THEME_TEXT)
                            )
                            
                            # Update axes
                            fig.update_xaxes(gridcolor='rgba(0,255,136,0.1)', showgrid=True)
                            fig.update_yaxes(gridcolor='rgba(0,255,136,0.1)', showgrid=True)
                            
                            return fig
                        
                        def update_loss_plot():
                            """Update the loss plot with current history."""
                            loss_plot_container.clear()
                            with loss_plot_container:
                                fig = make_loss_figure()
                                MS.loss_plot = ui.plotly(fig).classes('w-full').style('height: 280px;')
                        
                        # Initial empty plot
                        update_loss_plot()
                        
                        # Stats summary
                        with ui.row().classes('w-full gap-4 mt-4'):
                            with ui.card().classes('dark-card p-3 flex-1'):
                                ui.label('Best Val Loss').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                                best_val_label = ui.label('--').style(f'color:#f472b6; font-size: 1.2rem; font-weight: bold;')
                            
                            with ui.card().classes('dark-card p-3 flex-1'):
                                ui.label('Current Epoch').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                                current_epoch_label = ui.label('0').style(f'color:{THEME_PRIMARY}; font-size: 1.2rem; font-weight: bold;')
                            
                            with ui.card().classes('dark-card p-3 flex-1'):
                                ui.label('Train Loss').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                                train_loss_label = ui.label('--').style(f'color:{THEME_SECONDARY}; font-size: 1.2rem; font-weight: bold;')
                        
                        def update_stats():
                            """Update stats labels."""
                            if MS.history['epoch']:
                                current_epoch_label.text = str(MS.history['epoch'][-1])
                            if MS.history['train_loss']:
                                train_loss_label.text = f"{MS.history['train_loss'][-1]:.4f}"
                            if MS.history['val_loss']:
                                best_val_label.text = f"{min(MS.history['val_loss']):.4f}"
                        
                        ui.timer(2.0, update_stats)
                    
                    # RECONSTRUCTION TAB - Show original vs reconstructed with epoch slider
                    with ui.tab_panel(tab_recon).classes('p-2').style('height: 100%; display: flex; flex-direction: column;'):
                        ui.label('▌RECONSTRUCTION QUALITY').style(f'color:{THEME_SECONDARY}; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-2')
                        
                        recon_container = ui.column().classes('w-full flex-1')
                        
                        # Epoch selector
                        with ui.row().classes('items-center gap-3 mb-3'):
                            ui.label('Epoch:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.75rem;')
                            epoch_slider = ui.slider(min=1, max=100, step=10, value=10).props('label-always').classes('flex-1')
                            refresh_btn = ui.button('Refresh', icon='refresh').props('flat dense size=sm')
                        
                        def load_reconstruction(epoch_val):
                            """Load and display reconstruction for given epoch."""
                            recon_dir = AUTOENCODER_CACHE_DIR / 'output' / 'reconstructions'
                            recon_file = recon_dir / f'recon_epoch_{int(epoch_val):03d}.npz'
                            
                            recon_container.clear()
                            with recon_container:
                                if not recon_file.exists():
                                    ui.label(f'No reconstruction for epoch {int(epoch_val)}').style(f'color:{THEME_TEXT_DIM};')
                                    ui.label('Reconstructions are saved every 10 epochs during training').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                                    
                                    # List available epochs
                                    if recon_dir.exists():
                                        available = sorted([f.stem.split('_')[-1] for f in recon_dir.glob('*.npz')])
                                        if available:
                                            ui.label(f'Available: {", ".join(available)}').style(f'color:{THEME_PRIMARY}; font-size: 0.7rem;')
                                    return
                                
                                try:
                                    data = np.load(recon_file)
                                    original = data['original']
                                    reconstructed = data['reconstructed']
                                    
                                    # Create side-by-side heatmaps
                                    from plotly.subplots import make_subplots
                                    
                                    # Use first 100 samples, reshape to 10x10 grid
                                    n_features = min(original.shape[1], 10) if len(original.shape) > 1 else 10
                                    n_samples = min(100, original.shape[0])
                                    
                                    # Take mean across features for visualization
                                    if len(original.shape) > 1:
                                        orig_grid = original[:n_samples, :n_features]
                                        recon_grid = reconstructed[:n_samples, :n_features]
                                    else:
                                        orig_grid = original[:n_samples].reshape(-1, 1)
                                        recon_grid = reconstructed[:n_samples].reshape(-1, 1)
                                    
                                    diff_grid = np.abs(orig_grid - recon_grid)
                                    
                                    fig = make_subplots(
                                        rows=1, cols=3,
                                        subplot_titles=(f'Original (Epoch {int(epoch_val)})', 'Reconstructed', 'Difference'),
                                        horizontal_spacing=0.05
                                    )
                                    
                                    # Original heatmap
                                    fig.add_trace(go.Heatmap(
                                        z=orig_grid, colorscale='Viridis', showscale=False,
                                        name='Original'
                                    ), row=1, col=1)
                                    
                                    # Reconstructed heatmap  
                                    fig.add_trace(go.Heatmap(
                                        z=recon_grid, colorscale='Viridis', showscale=False,
                                        name='Reconstructed'
                                    ), row=1, col=2)
                                    
                                    # Difference heatmap
                                    fig.add_trace(go.Heatmap(
                                        z=diff_grid, colorscale='Reds', showscale=True,
                                        colorbar=dict(title='|Δ|', x=1.02, len=0.9),
                                        name='Difference'
                                    ), row=1, col=3)
                                    
                                    fig.update_layout(
                                        template='plotly_dark',
                                        paper_bgcolor='rgba(8,8,8,1)',
                                        plot_bgcolor='rgba(8,8,8,1)',
                                        height=500,
                                        margin=dict(l=40, r=60, t=50, b=40),
                                        font=dict(family='JetBrains Mono', size=10, color=THEME_TEXT)
                                    )
                                    
                                    ui.plotly(fig).classes('w-full').style('height: 500px;')
                                    
                                    # Stats
                                    mse = np.mean(diff_grid ** 2)
                                    mae = np.mean(diff_grid)
                                    with ui.row().classes('gap-4 mt-2'):
                                        ui.label(f'MSE: {mse:.6f}').style(f'color:{THEME_PRIMARY}; font-size: 0.8rem;')
                                        ui.label(f'MAE: {mae:.6f}').style(f'color:{THEME_SECONDARY}; font-size: 0.8rem;')
                                        ui.label(f'Samples: {n_samples} × {n_features} features').style(f'color:{THEME_TEXT_DIM}; font-size: 0.75rem;')
                                    
                                except Exception as e:
                                    ui.label(f'Error loading: {e}').style(f'color:{THEME_ERROR};')
                        
                        # Bind events
                        epoch_slider.on('update:model-value', lambda e: load_reconstruction(e.args))
                        refresh_btn.on('click', lambda: load_reconstruction(epoch_slider.value))
                        
                        # Initial load
                        load_reconstruction(10)
                    
                    # CONSOLE TAB
                    with ui.tab_panel(tab_console).classes('p-2').style('height: 100%; display: flex; flex-direction: column;'):
                        with ui.row().classes('items-center gap-3 mb-2'):
                            ui.label('// TRAINING_LOG').classes('terminal-header')
                            
                            def clear_log():
                                if MS.log_container:
                                    MS.log_container.clear()
                                MS.log_history = []
                            ui.button('CLEAR', on_click=clear_log, icon='delete').props('flat dense size=sm').classes('ml-auto')
                        
                        with ui.scroll_area().classes('w-full flex-1').style('background: #050505; border-radius: 4px; min-height: 200px;'):
                            MS.log_container = ui.column().classes('w-full p-3 gap-0')
                            with MS.log_container:
                                # Restore previous logs if any
                                if MS.log_history:
                                    colors = {'info': THEME_TEXT, 'success': THEME_PRIMARY, 'warning': THEME_WARN, 'error': THEME_ERROR}
                                    for msg, msg_type in MS.log_history[-100:]:  # Show last 100
                                        ui.label(msg).style(f'color:{colors.get(msg_type, THEME_TEXT)}; font-family: JetBrains Mono; font-size: 0.75rem;')
                                else:
                                    ui.label('Ready. Select a dataset and click Train.').style(f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.75rem;')


# ============================================================================
# ANALYSIS PAGE - Model Analysis & Visualization
# ============================================================================

def analysis_log(msg: str, msg_type: str = 'info'):
    """Add message to analysis log."""
    if AS.log_container:
        colors = {'info': THEME_TEXT, 'success': THEME_PRIMARY, 'warning': THEME_WARN, 'error': THEME_ERROR}
        with AS.log_container:
            ui.label(msg).style(f'color:{colors.get(msg_type, THEME_TEXT)}; font-family: JetBrains Mono; font-size: 0.7rem;')


def load_trained_model(model_path: Path):
    """Load a trained VAE model from checkpoint. Supports both graph and image models."""
    import torch
    import pickle
    import sys
    
    try:
        # For external models, add the project root to sys.path
        # This handles models that were saved with project-specific module references
        # Search upward from model path to find project root (containing 'src' folder)
        current = model_path.parent
        for _ in range(5):  # Search up to 5 levels
            if (current / 'src').exists():
                if str(current) not in sys.path:
                    sys.path.insert(0, str(current))
                    analysis_log(f"Added {current} to Python path", 'info')
                break
            parent = current.parent
            if parent == current:  # Reached root
                break
            current = parent
        
        # Try to load as pickle first (for image models with full_model)
        is_pickle = False
        checkpoint = None
        
        if model_path.suffix == '.pkl':
            try:
                with open(model_path, 'rb') as f:
                    checkpoint = pickle.load(f)
                is_pickle = True
            except Exception:
                pass
        
        if checkpoint is None:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Detect model type: image model has 'full_model' key, graph model has 'model_state_dict'
        if 'full_model' in checkpoint:
            # Image-based VAE (convolutional)
            return load_image_model(checkpoint, model_path)
        else:
            # Graph-based VAE
            return load_graph_model(checkpoint, model_path)
            
    except Exception as e:
        raise Exception(f"Failed to load model: {e}")


def load_image_model(checkpoint, model_path: Path):
    """Load an image-based convolutional VAE model."""
    import torch
    import sys
    
    # The model is stored directly in the checkpoint
    model = checkpoint['full_model']
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    # Extract model info
    model_config = checkpoint.get('model_config', {})
    model_params = {
        'latent_dim': checkpoint.get('latent_dim', model_config.get('latent_dim', 256)),
        'input_size': checkpoint.get('input_size', model_config.get('input_size', 512)),
        'num_classes': checkpoint.get('num_classes', model_config.get('num_classes', 17)),
        'model_type': checkpoint.get('model_type', 'image_vae'),
    }
    
    AS.model = model
    AS.model_config = model_config
    AS.model_path = str(model_path)
    AS.device = device
    AS.model_params = model_params
    AS.model_type = 'image'
    
    # Register hooks for activation extraction (image model)
    register_image_activation_hooks(model)
    
    return {
        'epoch': checkpoint.get('epoch', '?'),
        'val_loss': checkpoint.get('best_val_loss', '?'),
        'config': model_config,
        'model_params': model_params,
        'model_type': 'image'
    }


def load_graph_model(checkpoint, model_path: Path):
    """Load a graph-based VAE model."""
    import torch
    import pickle
    import sys
    
    config = checkpoint.get('config', {})
    
    # Get model parameters from checkpoint
    model_params = checkpoint.get('model_params', {})
    
    # If no model_params in checkpoint, try to infer from dataset or config
    if not model_params:
        # Try to load a sample from dataset cache to get dimensions
        dataset_cache = AUTOENCODER_CACHE_DIR / 'dataset_cache'
        try:
            # Try both .pt and .pkl files
            for pattern in ['*.pt', '*.pkl']:
                for cache_file in dataset_cache.glob(pattern):
                    try:
                        if cache_file.suffix == '.pkl':
                            with open(cache_file, 'rb') as f:
                                data = pickle.load(f)
                        else:
                            data = torch.load(cache_file, weights_only=False)
                        
                        # Handle dict with train/val/test splits
                        if isinstance(data, dict) and 'train' in data:
                            data = data['train']
                        
                        sample = data[0] if isinstance(data, list) and len(data) > 0 else data
                        if hasattr(sample, 'x'):
                            model_params = {
                                'num_node_features': sample.x.shape[1],
                                'num_edge_features': sample.edge_attr.shape[1] if hasattr(sample, 'edge_attr') and sample.edge_attr is not None else 1,
                                'num_graph_features': sample.graph_attr.shape[0] if hasattr(sample, 'graph_attr') and sample.graph_attr is not None else 3,
                                'num_nodes': sample.num_nodes
                            }
                            break
                    except Exception:
                        continue
                if model_params:
                    break
        except Exception:
            pass
    
    # Use values from model_params or defaults
    num_node_features = model_params.get('num_node_features', 68)
    num_edge_features = model_params.get('num_edge_features', 1)
    num_graph_features = model_params.get('num_graph_features', 3)
    num_nodes = model_params.get('num_nodes', 68)
    
    # Import model creation function
    if str(AUTOENCODER_DIR) not in sys.path:
        sys.path.insert(0, str(AUTOENCODER_DIR))
    from models import create_vae_from_config
    
    model = create_vae_from_config(
        config,
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        num_graph_features=num_graph_features,
        num_nodes=num_nodes
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    AS.model = model
    AS.model_config = config
    AS.model_path = str(model_path)
    AS.device = device
    AS.model_params = model_params
    AS.model_type = 'graph'
    
    # Register hooks for activation extraction
    register_activation_hooks(model)
    
    return {
        'epoch': checkpoint.get('epoch', '?'),
        'val_loss': checkpoint.get('val_loss', '?'),
        'config': config,
        'model_params': model_params,
        'model_type': 'graph'
    }


def register_image_activation_hooks(model):
    """Register forward hooks to capture activations for image models."""
    import torch.nn as nn
    AS.activations = {}
    AS.attention_weights = {}
    AS.layer_info = {'encoder': [], 'decoder': []}  # Store layer info for UI
    
    def get_activation(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                AS.activations[name] = output[0].detach().cpu()
            else:
                AS.activations[name] = output.detach().cpu()
        return hook
    
    # Register hooks on stem
    if hasattr(model, 'stem'):
        model.stem.register_forward_hook(get_activation('stem'))
    
    # Register hooks on encoder layers
    if hasattr(model, 'encoder_layers'):
        for i, layer in enumerate(model.encoder_layers):
            layer.register_forward_hook(get_activation(f'encoder.layer_{i}'))
            # Get output channels info
            out_ch = None
            for m in layer.modules():
                if hasattr(m, 'out_channels'):
                    out_ch = m.out_channels
                    break
            AS.layer_info['encoder'].append({'name': f'Layer {i}', 'channels': out_ch or 'N/A'})
    elif hasattr(model, 'encoder'):
        # Fallback for other architectures
        for name, child in model.encoder.named_children():
            if 'stage' in name or 'layer' in name:
                child.register_forward_hook(get_activation(f'encoder.{name}'))
                AS.layer_info['encoder'].append({'name': name, 'channels': 'N/A'})
    
    # Register hooks on decoder layers  
    if hasattr(model, 'decoder_layers'):
        for i, layer in enumerate(model.decoder_layers):
            layer.register_forward_hook(get_activation(f'decoder.layer_{i}'))
            out_ch = None
            for m in layer.modules():
                if hasattr(m, 'out_channels'):
                    out_ch = m.out_channels
                    break
            AS.layer_info['decoder'].append({'name': f'Layer {i}', 'channels': out_ch or 'N/A'})
    
    # Register hook on latent
    if hasattr(model, 'fc_mu'):
        model.fc_mu.register_forward_hook(get_activation('latent_mu'))
    if hasattr(model, 'fc_logvar'):
        model.fc_logvar.register_forward_hook(get_activation('latent_logvar'))
    
    # Register hook on output
    if hasattr(model, 'output'):
        model.output.register_forward_hook(get_activation('output'))


def register_activation_hooks(model):
    """Register forward hooks to capture activations."""
    AS.activations = {}
    AS.attention_weights = {}
    
    def get_activation(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                AS.activations[name] = output[0].detach().cpu()
            else:
                AS.activations[name] = output.detach().cpu()
        return hook
    
    def get_attention(name):
        def hook(module, input, output):
            # GAT layers store attention in return_attention_weights
            if hasattr(module, 'return_attention_weights') and module.return_attention_weights:
                if isinstance(output, tuple) and len(output) > 1:
                    AS.attention_weights[name] = output[1].detach().cpu()
        return hook
    
    # Register hooks on encoder layers
    if hasattr(model, 'encoder'):
        conv_layers = getattr(model.encoder, 'conv_layers', None)
        if conv_layers is not None:
            for i, layer in enumerate(conv_layers):
                layer.register_forward_hook(get_activation(f'encoder.conv_{i}'))
    
    # Register hooks on decoder layers
    if hasattr(model, 'decoder'):
        conv_layers = getattr(model.decoder, 'conv_layers', None)
        if conv_layers is not None:
            for i, layer in enumerate(conv_layers):
                layer.register_forward_hook(get_activation(f'decoder.conv_{i}'))
        elif hasattr(model.decoder, 'layers'):
            for i, layer in enumerate(model.decoder.layers):
                layer.register_forward_hook(get_activation(f'decoder.conv_{i}'))
    
    # Register hook on latent
    if hasattr(model, 'fc_mu'):
        model.fc_mu.register_forward_hook(get_activation('latent_mu'))
    if hasattr(model, 'fc_logvar'):
        model.fc_logvar.register_forward_hook(get_activation('latent_logvar'))


def process_sample(sample, store_latent=True):
    """Process a single sample through the model and extract activations."""
    import torch
    
    if AS.model is None:
        analysis_log("No model loaded", 'warning')
        return None
    
    try:
        AS.model.eval()
        with torch.no_grad():
            if AS.model_type == 'image':
                return process_image_sample(sample, store_latent)
            else:
                return process_graph_sample(sample, store_latent)
    except Exception as e:
        analysis_log(f"Error in process_sample: {e}", 'error')
        import traceback
        analysis_log(traceback.format_exc(), 'error')
        return None


def process_image_sample(sample, store_latent=True):
    """Process an image sample through the model."""
    import torch
    
    # sample is a tuple (image_tensor, label) or just image_tensor
    if isinstance(sample, tuple):
        image, label = sample
        AS.current_label = label.item() if hasattr(label, 'item') else label
    else:
        image = sample
        AS.current_label = None
    
    # Ensure batch dimension
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    image = image.to(AS.device)
    
    # Forward pass
    output = AS.model(image)
    
    AS.current_sample = image
    AS.current_recon = output
    AS.current_z = output.get('mu', None) if isinstance(output, dict) else None
    
    if store_latent and AS.current_z is not None:
        z_np = AS.current_z.cpu().numpy().flatten()
        AS.latent_codes.append(z_np)
        label = AS.current_label if AS.current_label is not None else 0
        AS.latent_labels.append(label)
        # Keep only last 500 for PCA
        if len(AS.latent_codes) > 500:
            AS.latent_codes = AS.latent_codes[-500:]
            AS.latent_labels = AS.latent_labels[-500:]
    
    return output


def process_graph_sample(sample, store_latent=True):
    """Process a graph sample through the model."""
    import torch
    from torch_geometric.data import Batch
    
    # Create a batch from single sample (required by PyG)
    if not isinstance(sample, Batch):
        batch = Batch.from_data_list([sample])
    else:
        batch = sample
    
    batch = batch.to(AS.device)
    
    # Forward pass with activation storage ENABLED to capture GAT attention
    output = AS.model(batch, store_activations=True)
    
    AS.current_sample = batch
    AS.current_recon = output
    AS.current_z = output.get('mu', None) if isinstance(output, dict) else None
    
    # Extract REAL attention weights from encoder
    if hasattr(AS.model, 'encoder') and hasattr(AS.model.encoder, 'get_all_attention_weights'):
        AS.attention_weights = AS.model.encoder.get_all_attention_weights()
    
    # Extract layer activations
    if hasattr(AS.model, 'encoder') and hasattr(AS.model.encoder, 'get_layer_activations'):
        layer_acts = AS.model.encoder.get_layer_activations()
        for key, val in layer_acts.items():
            AS.activations[f'encoder.{key}'] = val
    
    if store_latent and AS.current_z is not None:
        z_np = AS.current_z.cpu().numpy().flatten()
        AS.latent_codes.append(z_np)
        # Get label from sample
        label = batch.y[0].item() if hasattr(batch, 'y') and batch.y is not None else 0
        AS.latent_labels.append(label)
        # Keep only last 500 for PCA
        if len(AS.latent_codes) > 500:
            AS.latent_codes = AS.latent_codes[-500:]
            AS.latent_labels = AS.latent_labels[-500:]
    
    return output


def compute_latent_pca():
    """Compute PCA on accumulated latent codes."""
    if len(AS.latent_codes) < 10:
        return None, None
    
    from sklearn.decomposition import PCA
    import numpy as np
    
    X = np.array(AS.latent_codes)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    return X_pca, np.array(AS.latent_labels)


def load_hdf5_image_dataset(h5_path, split, dataset_info_container, progress_slider):
    """Load an image dataset from HDF5 file using lazy loading."""
    import h5py
    import torch
    import numpy as np
    
    try:
        # First, just read metadata without loading all data
        with h5py.File(h5_path, 'r') as f:
            total_samples = len(f['labels'])
            resolution = f.attrs.get('resolution', 512)
            labels_all = f['labels'][:]
            
            # Get group names if available
            if 'group_names' in f:
                AS.class_names = [n.decode() if isinstance(n, bytes) else n for n in f['group_names'][:]]
            else:
                AS.class_names = [str(i) for i in range(len(np.unique(labels_all)))]
        
        # Check for splits file
        indices = None
        splits_path = h5_path.parent / 'splits.npz'
        if splits_path.exists():
            splits = np.load(splits_path)
            if split in splits:
                indices = splits[split]
                analysis_log(f"Using '{split}' split: {len(indices)} samples", 'info')
        
        # Create lazy-loading wrapper for HDF5
        class HDF5DatasetWrapper:
            """Lazy-loading wrapper for HDF5 image dataset."""
            def __init__(self, h5_path, indices=None):
                self.h5_path = str(h5_path)
                self.indices = indices
                self._file = None
                self._labels = labels_all[indices] if indices is not None else labels_all
                
            def _open(self):
                if self._file is None:
                    self._file = h5py.File(self.h5_path, 'r')
                return self._file
            
            def __len__(self):
                return len(self.indices) if self.indices is not None else len(self._labels)
            
            def __getitem__(self, idx):
                f = self._open()
                real_idx = self.indices[idx] if self.indices is not None else idx
                # Load single image on demand
                pattern = f['patterns'][real_idx]  # [H, W, 3]
                label = self._labels[idx]
                # Convert to tensor [3, H, W]
                tensor = torch.from_numpy(pattern.astype(np.float32)).permute(2, 0, 1)
                return tensor, int(label)
            
            def __del__(self):
                if self._file is not None:
                    try:
                        self._file.close()
                    except:
                        pass
        
        AS.dataset = HDF5DatasetWrapper(h5_path, indices)
        AS.dataset_type = 'image'
        AS.total_samples = len(AS.dataset)
        AS.current_idx = 0
        AS.dataset_path = str(h5_path)
        
        # Clear latent history
        AS.latent_codes = []
        AS.latent_labels = []
        
        dataset_info_container.clear()
        with dataset_info_container:
            ui.label(f"✓ Image dataset loaded").style(f'color:{THEME_PRIMARY}; font-size: 0.75rem;')
            ui.label(f"  Split: {split}").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
            ui.label(f"  Samples: {AS.total_samples}").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
            ui.label(f"  Size: {resolution}×{resolution} RGB").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
            ui.label(f"  Classes: {len(AS.class_names)}").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
        
        analysis_log(f"HDF5 dataset loaded: {AS.total_samples} images (lazy loading)", 'success')
        
        if progress_slider:
            progress_slider.set_value(0)
            progress_slider._props['max'] = max(1, AS.total_samples - 1)
        
        return True
    except Exception as e:
        analysis_log(f"Error loading HDF5: {e}", 'error')
        import traceback
        analysis_log(traceback.format_exc(), 'error')
        return False


def load_image_folder_dataset(images_dir, split, base_path, dataset_info_container, progress_slider):
    """Load an image dataset from folder structure (class_name/image.png)."""
    import torch
    import numpy as np
    from PIL import Image
    
    try:
        # Find all class directories
        class_dirs = sorted([d for d in images_dir.iterdir() if d.is_dir()])
        AS.class_names = [d.name for d in class_dirs]
        
        # Collect all images
        samples = []
        for class_idx, class_dir in enumerate(class_dirs):
            for img_path in sorted(class_dir.glob('*.png')) + sorted(class_dir.glob('*.jpg')):
                samples.append((str(img_path), class_idx))
        
        # Check for splits file
        splits_path = base_path / 'splits.npz'
        if splits_path.exists():
            splits = np.load(splits_path)
            if split in splits:
                indices = splits[split]
                samples = [samples[i] for i in indices if i < len(samples)]
                analysis_log(f"Using '{split}' split: {len(samples)} samples", 'info')
        
        # Create lazy-loading dataset
        class ImageDatasetWrapper:
            def __init__(self, samples, class_names):
                self.samples = samples
                self.class_names = class_names
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                img_path, label = self.samples[idx]
                img = Image.open(img_path).convert('RGB')
                img_np = np.array(img, dtype=np.float32) / 255.0
                # [H, W, 3] -> [3, H, W]
                tensor = torch.from_numpy(img_np).permute(2, 0, 1)
                return tensor, label
        
        AS.dataset = ImageDatasetWrapper(samples, AS.class_names)
        AS.dataset_type = 'image'
        AS.total_samples = len(AS.dataset)
        AS.current_idx = 0
        AS.dataset_path = str(images_dir)
        
        # Clear latent history
        AS.latent_codes = []
        AS.latent_labels = []
        
        # Get sample info
        sample_img, _ = AS.dataset[0]
        resolution = sample_img.shape[-1]
        
        dataset_info_container.clear()
        with dataset_info_container:
            ui.label(f"✓ Image folder loaded").style(f'color:{THEME_PRIMARY}; font-size: 0.75rem;')
            ui.label(f"  Split: {split}").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
            ui.label(f"  Samples: {AS.total_samples}").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
            ui.label(f"  Size: {resolution}×{resolution} RGB").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
            ui.label(f"  Classes: {len(AS.class_names)}").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
        
        analysis_log(f"Image folder loaded: {AS.total_samples} images, {len(AS.class_names)} classes", 'success')
        
        if progress_slider:
            progress_slider.set_value(0)
            progress_slider._props['max'] = max(1, AS.total_samples - 1)
        
        return True
    except Exception as e:
        analysis_log(f"Error loading image folder: {e}", 'error')
        import traceback
        analysis_log(traceback.format_exc(), 'error')
        return False


def compute_dataset_ranges():
    """
    Analyze dataset and model to compute fixed axis ranges.
    Should be called after loading both model and dataset.
    """
    import torch
    from torch_geometric.data import Batch
    import numpy as np
    
    if not AS.dataset or not AS.model:
        return
    
    analysis_log("Computing axis ranges from dataset sample...", 'info')
    
    # Compute Kuramoto statistics from entire dataset
    all_kuramoto = []
    for sample in AS.dataset:
        if hasattr(sample, 'graph_attr') and sample.graph_attr is not None and len(sample.graph_attr) > 0:
            k_val = sample.graph_attr[0].item() if hasattr(sample.graph_attr[0], 'item') else float(sample.graph_attr[0])
            all_kuramoto.append(k_val)
    
    if all_kuramoto:
        AS.kuramoto_avg = float(np.mean(all_kuramoto))
        AS.kuramoto_std = float(np.std(all_kuramoto))
        analysis_log(f"Kuramoto: avg={AS.kuramoto_avg:.3f}, std={AS.kuramoto_std:.3f} (metastability)", 'info')
    
    # Clear kuramoto history for fresh start
    AS.kuramoto_history = []
    
    # Sample a subset of the dataset for analysis
    n_samples = min(100, len(AS.dataset))
    indices = np.linspace(0, len(AS.dataset)-1, n_samples, dtype=int)
    
    all_features = []
    all_latents = []
    all_activations = {f'encoder.conv_{i}': [] for i in range(3)}
    
    AS.model.eval()
    with torch.no_grad():
        for idx in indices:
            try:
                sample = AS.dataset[idx]
                batch = Batch.from_data_list([sample]).to(AS.device)
                
                # Get node features
                all_features.append(batch.x.cpu().numpy())
                
                # Forward pass with activations
                output = AS.model(batch, store_activations=True)
                
                # Get latent
                if isinstance(output, dict) and 'mu' in output:
                    all_latents.append(output['mu'].cpu().numpy())
                
                # Get activations
                for name, act in AS.activations.items():
                    if name in all_activations:
                        all_activations[name].append(act.numpy().flatten())
            except Exception:
                continue
    
    # Compute ranges for node features
    if all_features:
        all_feat = np.concatenate(all_features, axis=0)
        feat_min, feat_max = np.percentile(all_feat, [2, 98])  # Use percentiles to ignore outliers
        margin = (feat_max - feat_min) * 0.1
        AS.axis_ranges['node_features'] = {
            'min': float(feat_min - margin),
            'max': float(feat_max + margin)
        }
        analysis_log(f"Node features range: [{feat_min:.2f}, {feat_max:.2f}]", 'info')
    
    # Compute ranges for latent space (will be updated as PCA accumulates)
    if all_latents:
        all_lat = np.concatenate(all_latents, axis=0)
        from sklearn.decomposition import PCA
        if all_lat.shape[0] >= 10:
            pca = PCA(n_components=2)
            lat_pca = pca.fit_transform(all_lat)
            x_min, x_max = np.percentile(lat_pca[:, 0], [2, 98])
            y_min, y_max = np.percentile(lat_pca[:, 1], [2, 98])
            margin_x = (x_max - x_min) * 0.15
            margin_y = (y_max - y_min) * 0.15
            AS.axis_ranges['latent'] = {
                'x_min': float(x_min - margin_x),
                'x_max': float(x_max + margin_x),
                'y_min': float(y_min - margin_y),
                'y_max': float(y_max + margin_y)
            }
            analysis_log(f"Latent PCA range: x[{x_min:.1f}, {x_max:.1f}], y[{y_min:.1f}, {y_max:.1f}]", 'info')
    
    # Compute ranges for activations
    for name, acts in all_activations.items():
        if acts:
            all_act = np.concatenate(acts)
            act_min, act_max = np.percentile(all_act, [2, 98])
            margin = (act_max - act_min) * 0.1
            if 'activations_per_layer' not in AS.axis_ranges:
                AS.axis_ranges['activations_per_layer'] = {}
            AS.axis_ranges['activations_per_layer'][name] = {
                'min': float(act_min - margin),
                'max': float(act_max + margin)
            }
    
    # Global activation range
    if all_activations:
        all_acts = []
        for acts in all_activations.values():
            if acts:
                all_acts.extend(acts)
        if all_acts:
            all_act = np.concatenate(all_acts)
            act_min, act_max = np.percentile(all_act, [2, 98])
            margin = (act_max - act_min) * 0.1
            AS.axis_ranges['activations'] = {
                'min': float(act_min - margin),
                'max': float(act_max + margin)
            }
            analysis_log(f"Activations range: [{act_min:.1f}, {act_max:.1f}]", 'info')
    
    AS.ranges_computed = True
    analysis_log("✓ Axis ranges computed", 'success')


@ui.page('/analysis')
def analysis_page():
    """Model Analysis Page - Visualize trained model internals."""
    ui.add_head_html(f'<style>{STYLE}</style>')
    
    # Header
    with ui.header().classes('items-center px-4 py-1').style(f'background: {THEME_BG}; border-bottom: 1px solid {THEME_BORDER};'):
        ui.label('▶').style(f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.75rem;')
        ui.label('EEG_VIEWER').classes('text-base font-medium ml-2').style(f'color: {THEME_PRIMARY}; font-family: JetBrains Mono;')
        ui.label('// ANALYSIS').classes('text-sm ml-3').style(f'color:{THEME_WARN}; font-family: JetBrains Mono;')
        with ui.row().classes('ml-auto gap-2'):
            ui.button('VIEWER', on_click=lambda: ui.navigate.to('/')).props('flat dense').style(f'color:{THEME_TEXT_DIM};')
            ui.button('PIPELINE', on_click=lambda: ui.navigate.to('/pipeline')).props('flat dense').style(f'color:{THEME_TEXT_DIM};')
            ui.button('MODEL', on_click=lambda: ui.navigate.to('/model')).props('flat dense').style(f'color:{THEME_TEXT_DIM};')
            ui.button('ANALYSIS', on_click=lambda: ui.navigate.to('/analysis')).props('flat dense').style(f'color:{THEME_WARN};')
    
    with ui.row().classes('w-full p-4 gap-4').style('height: calc(100vh - 50px); overflow: hidden;'):
        
        # LEFT PANEL: Controls
        with ui.column().classes('gap-3').style('width: 280px; flex-shrink: 0; overflow-y: auto; max-height: 100%;'):
            
            # MODEL LOADER
            with ui.card().classes('dark-card p-3 w-full').style(f'border: 1px solid {THEME_WARN};'):
                ui.label('// LOAD MODEL').classes('terminal-header')
                
                model_path_input = ui.input(
                    value=str(AUTOENCODER_CACHE_DIR / 'checkpoints' / 'best_model.pt'),
                    placeholder='Path to model checkpoint'
                ).props('dense dark').classes('w-full mt-2')
                
                model_info_container = ui.column().classes('w-full mt-2 gap-1')
                
                async def load_model():
                    try:
                        path = Path(model_path_input.value.strip())
                        if not path.exists():
                            ui.notify(f'Model not found: {path}', type='negative')
                            return
                        
                        # Show loading indicator
                        model_info_container.clear()
                        with model_info_container:
                            ui.label(f"⏳ Loading model...").style(f'color:{THEME_WARN}; font-size: 0.75rem;')
                            ui.label(f"  This may take a moment").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                        
                        # Load model in background thread to avoid blocking UI
                        loop = asyncio.get_event_loop()
                        info = await loop.run_in_executor(None, load_trained_model, path)
                        
                        model_info_container.clear()
                        with model_info_container:
                            ui.label(f"✓ Model loaded").style(f'color:{THEME_PRIMARY}; font-size: 0.75rem;')
                            ui.label(f"  Type: {info.get('model_type', 'graph')}").style(f'color:{THEME_SECONDARY}; font-size: 0.7rem;')
                            ui.label(f"  Epoch: {info['epoch']}").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                            ui.label(f"  Val Loss: {info['val_loss']:.4f}" if isinstance(info['val_loss'], float) else f"  Val Loss: {info['val_loss']}").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                            ui.label(f"  Device: {AS.device}").style(f'color:{THEME_SECONDARY}; font-size: 0.7rem;')
                        analysis_log(f"Model loaded: {path.name}", 'success')
                        
                        # Compute axis ranges if dataset is loaded
                        if AS.dataset is not None:
                            compute_dataset_ranges()
                    except Exception as e:
                        model_info_container.clear()
                        with model_info_container:
                            ui.label(f"✗ Error loading model").style(f'color:{THEME_ERROR}; font-size: 0.75rem;')
                        ui.notify(f'Error: {e}', type='negative')
                        analysis_log(f"Error loading model: {e}", 'error')
                        import traceback
                        analysis_log(traceback.format_exc(), 'error')
                
                ui.button('Load Model', on_click=load_model, icon='upload').props('dense').classes('mt-2').style(f'background:{THEME_WARN}; color:black;')
            
            # DATASET LOADER
            with ui.card().classes('dark-card p-3 w-full'):
                ui.label('// DATASET').classes('terminal-header')
                
                with ui.row().classes('items-center gap-2 mt-2'):
                    ui.label('Split:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                    split_select = ui.select(['train', 'val', 'test'], value='test').props('dense dark').classes('flex-1')
                
                dataset_path_input = ui.input(
                    value=str(AUTOENCODER_CACHE_DIR / 'dataset_cache'),
                    placeholder='Path to dataset cache'
                ).props('dense dark').classes('w-full mt-2')
                
                dataset_info_container = ui.column().classes('w-full mt-2 gap-1')
                
                def load_dataset():
                    import torch
                    import pickle
                    import h5py
                    import numpy as np
                    try:
                        cache_path = Path(dataset_path_input.value.strip())
                        split = split_select.value
                        
                        # Detect dataset type
                        dataset_loaded = False
                        
                        # Check for HDF5 image dataset
                        h5_files = list(cache_path.glob('*.h5')) if cache_path.is_dir() else []
                        if cache_path.suffix == '.h5' or h5_files:
                            h5_path = cache_path if cache_path.suffix == '.h5' else h5_files[0]
                            dataset_loaded = load_hdf5_image_dataset(h5_path, split, dataset_info_container, progress_slider)
                        
                        # Check for image folder dataset
                        if not dataset_loaded and cache_path.is_dir():
                            images_dir = cache_path / 'images' if (cache_path / 'images').exists() else cache_path
                            subdirs = [d for d in images_dir.iterdir() if d.is_dir()]
                            has_images = any(list(d.glob('*.png'))[:1] or list(d.glob('*.jpg'))[:1] for d in subdirs[:3])
                            if has_images:
                                dataset_loaded = load_image_folder_dataset(images_dir, split, cache_path, dataset_info_container, progress_slider)
                        
                        # Fallback to graph dataset loading
                        if not dataset_loaded:
                            cache_file = None
                            
                            # Check if path is a file directly
                            if cache_path.is_file():
                                cache_file = cache_path
                            else:
                                # It's a directory, search for dataset files
                                possible_files = [
                                    cache_path / f'{split}_dataset.pt',
                                    cache_path / 'processed_dataset.pt',
                                    cache_path / f'{split}_dataset.pkl',
                                    cache_path / 'processed_dataset.pkl',
                                ]
                                for f in possible_files:
                                    if f.exists():
                                        cache_file = f
                                        break
                                
                                # If still not found, search for any dataset file
                                if not cache_file:
                                    for pattern in ['dataset*.pkl', 'dataset*.pt', '*.pkl', '*.pt']:
                                        files = list(cache_path.glob(pattern))
                                        if files:
                                            cache_file = files[0]
                                            break
                            
                            if cache_file and cache_file.exists():
                                # Load based on file extension
                                if cache_file.suffix == '.pkl':
                                    with open(cache_file, 'rb') as f:
                                        data = pickle.load(f)
                                else:
                                    data = torch.load(cache_file, weights_only=False)
                                
                                # Handle dict with train/val/test splits
                                if isinstance(data, dict) and split in data:
                                    graphs = data[split]
                                    analysis_log(f"Using '{split}' split from dataset", 'info')
                                elif isinstance(data, dict) and 'train' in data:
                                    # Default to train if requested split not found
                                    available = list(data.keys())
                                    graphs = data.get(split, data['train'])
                                    analysis_log(f"Available splits: {available}, using '{split}'", 'info')
                                elif isinstance(data, list):
                                    graphs = data
                                elif hasattr(data, '__len__'):
                                    graphs = list(data)
                                else:
                                    graphs = [data]
                                
                                AS.dataset = graphs
                                AS.dataset_type = 'graph'
                                AS.total_samples = len(AS.dataset)
                                AS.current_idx = 0
                                AS.dataset_path = str(cache_file)
                                
                                # Clear latent history for fresh PCA
                                AS.latent_codes = []
                                AS.latent_labels = []
                                
                                dataset_info_container.clear()
                                with dataset_info_container:
                                    ui.label(f"✓ Graph dataset loaded").style(f'color:{THEME_PRIMARY}; font-size: 0.75rem;')
                                    ui.label(f"  Split: {split}").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                                    ui.label(f"  Samples: {AS.total_samples}").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                                    if AS.dataset and hasattr(AS.dataset[0], 'x'):
                                        sample = AS.dataset[0]
                                        ui.label(f"  Nodes: {sample.x.shape[0]}, Features: {sample.x.shape[1]}").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                                
                                analysis_log(f"Dataset loaded: {AS.total_samples} samples from {cache_file.name}", 'success')
                                if progress_slider:
                                    progress_slider.set_value(0)
                                    progress_slider._props['max'] = max(1, AS.total_samples - 1)
                                
                                # Compute axis ranges if model is loaded
                                if AS.model is not None:
                                    compute_dataset_ranges()
                                dataset_loaded = True
                        
                        if not dataset_loaded:
                            ui.notify(f'Dataset not found at {cache_path}', type='warning')
                            analysis_log(f"Dataset not found: {cache_path}", 'warning')
                    except Exception as e:
                        import traceback
                        ui.notify(f'Error: {e}', type='negative')
                        analysis_log(f"Error loading dataset: {e}", 'error')
                        analysis_log(traceback.format_exc(), 'error')
                
                ui.button('Load Dataset', on_click=load_dataset, icon='dataset').props('dense').classes('mt-2')
            
            # PLAYBACK CONTROLS
            with ui.card().classes('dark-card p-3 w-full'):
                ui.label('// PLAYBACK').classes('terminal-header')
                
                with ui.row().classes('items-center gap-2 mt-2'):
                    ui.label('Speed:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                    speed_select = ui.select(
                        ['0.5x', '1x', '2x', '5x', '10x'],
                        value='1x'
                    ).props('dense dark').classes('w-20')
                
                progress_slider = ui.slider(min=0, max=100, value=0).props('label-always').classes('w-full mt-2')
                AS.progress_slider = progress_slider
                
                sample_label = ui.label('Sample: 0 / 0').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;').classes('mt-1')
                
                with ui.row().classes('gap-2 mt-2 justify-center'):
                    def prev_sample():
                        if AS.dataset and AS.current_idx > 0:
                            AS.current_idx -= 1
                            progress_slider.set_value(AS.current_idx)
                            process_current_sample()
                    
                    def next_sample():
                        if AS.dataset and AS.current_idx < AS.total_samples - 1:
                            AS.current_idx += 1
                            progress_slider.set_value(AS.current_idx)
                            process_current_sample()
                    
                    async def toggle_play():
                        AS.playing = not AS.playing
                        if AS.playing:
                            play_btn.props('icon=pause color=negative')
                            analysis_log("Playback started", 'info')
                            # Start playback loop
                            while AS.playing and AS.dataset and AS.current_idx < AS.total_samples - 1:
                                AS.current_idx += 1
                                progress_slider.set_value(AS.current_idx)
                                process_current_sample()
                                # Speed control
                                speed_map = {'0.5x': 2.0, '1x': 1.0, '2x': 0.5, '5x': 0.2, '10x': 0.1}
                                delay = speed_map.get(speed_select.value, 1.0)
                                await asyncio.sleep(delay)
                            AS.playing = False
                            play_btn.props('icon=play_arrow color=primary')
                            analysis_log("Playback stopped", 'info')
                        else:
                            play_btn.props('icon=play_arrow color=primary')
                    
                    def stop_play():
                        AS.playing = False
                        AS.current_idx = 0
                        progress_slider.set_value(0)
                        play_btn.props('icon=play_arrow color=primary')
                        process_current_sample()
                    
                    ui.button(icon='skip_previous', on_click=prev_sample).props('round dense size=sm')
                    play_btn = ui.button(icon='play_arrow', on_click=toggle_play).props('round dense size=sm color=primary')
                    ui.button(icon='skip_next', on_click=next_sample).props('round dense size=sm')
                    ui.button(icon='stop', on_click=stop_play).props('round dense size=sm color=negative')
                
                def on_slider_change(e):
                    if AS.dataset:
                        AS.current_idx = int(e.args)
                        process_current_sample()
                
                progress_slider.on('update:model-value', on_slider_change)
            
            # SAMPLE INFO
            with ui.card().classes('dark-card p-3 w-full'):
                ui.label('// CURRENT SAMPLE').classes('terminal-header')
                AS.sample_info_container = ui.column().classes('w-full mt-2 gap-1')
                with AS.sample_info_container:
                    ui.label('No sample loaded').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
            
            # LOG
            with ui.card().classes('dark-card p-3 w-full'):
                ui.label('// LOG').classes('terminal-header')
                with ui.scroll_area().classes('w-full').style('height: 100px; background: #050505; border-radius: 4px;'):
                    AS.log_container = ui.column().classes('w-full p-2 gap-0')
                    with AS.log_container:
                        ui.label('Ready. Load a model and dataset.').style(f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.7rem;')
        
        # RIGHT PANEL: Visualizations
        with ui.column().classes('flex-1 gap-3').style('min-height: 0; overflow-y: auto;'):
            
            # ROW 1: KURAMOTO ORDER PARAMETER (progress bar style)
            with ui.card().classes('dark-card p-3 w-full'):
                ui.label('▌KURAMOTO ORDER PARAMETER').style(f'color:{THEME_WARN}; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-2')
                
                def make_kuramoto_fig():
                    """Create Kuramoto order parameter visualization."""
                    fig = go.Figure()
                    
                    n_total = AS.total_samples if AS.total_samples > 0 else 100
                    
                    # Metastability band (std around mean)
                    if AS.kuramoto_avg > 0:
                        fig.add_trace(go.Scatter(
                            x=list(range(n_total)) + list(range(n_total-1, -1, -1)),
                            y=[AS.kuramoto_avg + AS.kuramoto_std] * n_total + [AS.kuramoto_avg - AS.kuramoto_std] * n_total,
                            fill='toself',
                            fillcolor='rgba(100, 150, 200, 0.2)',
                            line=dict(width=0),
                            name='Metastability',
                            showlegend=True
                        ))
                    
                    # Average line (dashed)
                    fig.add_trace(go.Scatter(
                        x=[0, n_total-1],
                        y=[AS.kuramoto_avg, AS.kuramoto_avg],
                        mode='lines',
                        line=dict(color='rgba(150, 200, 255, 0.7)', width=2, dash='dash'),
                        name=f'Avg: {AS.kuramoto_avg:.3f}'
                    ))
                    
                    # Scatter points for processed samples
                    if AS.kuramoto_history:
                        indices, values = zip(*AS.kuramoto_history)
                        # Color by label
                        colors = [['#00ff88', '#f472b6', '#00d4ff'][AS.latent_labels[i] % 3] if i < len(AS.latent_labels) else THEME_PRIMARY for i in range(len(indices))]
                        fig.add_trace(go.Scatter(
                            x=indices,
                            y=values,
                            mode='markers',
                            marker=dict(size=5, color=colors, opacity=0.8),
                            name='Samples'
                        ))
                        
                        # Current point highlighted
                        if len(indices) > 0:
                            fig.add_trace(go.Scatter(
                                x=[indices[-1]],
                                y=[values[-1]],
                                mode='markers',
                                marker=dict(size=12, color=THEME_WARN, symbol='star', line=dict(width=2, color='white')),
                                name='Current'
                            ))
                    
                    fig.update_layout(
                        template='plotly_dark',
                        paper_bgcolor='rgba(8,8,8,1)',
                        plot_bgcolor='rgba(8,8,8,1)',
                        height=120,
                        margin=dict(l=40, r=10, t=5, b=30),
                        xaxis=dict(title='Sample Index', range=[0, n_total], gridcolor='rgba(255,204,0,0.1)'),
                        yaxis=dict(title='r', range=[0, 1], gridcolor='rgba(255,204,0,0.1)'),
                        legend=dict(orientation='h', y=1.15, font=dict(size=8)),
                        font=dict(family='JetBrains Mono', size=9, color=THEME_TEXT)
                    )
                    return fig
                
                kuramoto_plot = ui.plotly(make_kuramoto_fig()).classes('w-full').style('height: 120px;')
                AS.activation_plots['kuramoto'] = kuramoto_plot
            
            # ROW 2: ENCODER + DECODER ACTIVATIONS
            with ui.row().classes('gap-3 w-full'):
                
                # ENCODER ACTIVATIONS
                with ui.card().classes('dark-card p-3 flex-1'):
                    with ui.row().classes('items-center gap-2 mb-2'):
                        ui.label('▌ENCODER').style(f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.75rem;')
                        encoder_layer_select = ui.select([], value=None).props('dense dark').classes('w-24').style('font-size: 0.7rem;')
                        encoder_channel_select = ui.select([], value=None).props('dense dark').classes('w-20').style('font-size: 0.7rem;')
                    
                    def update_encoder_selectors():
                        """Update encoder layer/channel selectors based on available activations."""
                        layers = [k for k in AS.activations.keys() if k.startswith('encoder.') or k == 'stem']
                        if layers:
                            encoder_layer_select.options = layers
                            if encoder_layer_select.value not in layers:
                                encoder_layer_select.value = layers[0]
                            # Update channel selector
                            if encoder_layer_select.value and encoder_layer_select.value in AS.activations:
                                act = AS.activations[encoder_layer_select.value]
                                if len(act.shape) >= 2:
                                    n_channels = act.shape[1] if len(act.shape) == 4 else act.shape[0]
                                    ch_options = [f'Ch {i}' for i in range(min(n_channels, 64))]
                                    encoder_channel_select.options = ch_options
                                    if not encoder_channel_select.value or encoder_channel_select.value not in ch_options:
                                        encoder_channel_select.value = ch_options[0] if ch_options else None
                    
                    def make_encoder_fig():
                        """Create encoder activation visualization."""
                        fig = go.Figure()
                        
                        layer_name = encoder_layer_select.value
                        channel_str = encoder_channel_select.value
                        
                        if layer_name and layer_name in AS.activations and channel_str:
                            act = AS.activations[layer_name]
                            channel_idx = int(channel_str.split(' ')[1]) if channel_str else 0
                            
                            # Handle different activation shapes
                            if len(act.shape) == 4:  # [B, C, H, W]
                                act_2d = act[0, channel_idx].numpy()
                            elif len(act.shape) == 3:  # [C, H, W]
                                act_2d = act[channel_idx].numpy()
                            elif len(act.shape) == 2:  # [H, W] or [B, features]
                                act_2d = act[0].numpy() if act.shape[0] == 1 else act.numpy()
                            else:
                                act_2d = act.numpy().flatten().reshape(-1, 1)
                            
                            fig.add_trace(go.Heatmap(
                                z=act_2d,
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(len=0.8, thickness=10)
                            ))
                        else:
                            fig.add_annotation(text="No activations", x=0.5, y=0.5, showarrow=False,
                                             font=dict(color=THEME_TEXT_DIM))
                        
                        fig.update_layout(
                            template='plotly_dark',
                            paper_bgcolor='rgba(8,8,8,1)',
                            plot_bgcolor='rgba(8,8,8,1)',
                            height=180,
                            margin=dict(l=10, r=40, t=5, b=10),
                            xaxis=dict(showticklabels=False, showgrid=False),
                            yaxis=dict(showticklabels=False, showgrid=False, scaleanchor='x'),
                            font=dict(family='JetBrains Mono', size=8, color=THEME_TEXT)
                        )
                        return fig
                    
                    encoder_plot = ui.plotly(make_encoder_fig()).classes('w-full').style('height: 180px;')
                    AS.activation_plots['encoder'] = encoder_plot
                    
                    def on_encoder_layer_change(e):
                        update_encoder_selectors()
                        encoder_plot.figure = make_encoder_fig()
                        encoder_plot.update()
                    
                    def on_encoder_channel_change(e):
                        encoder_plot.figure = make_encoder_fig()
                        encoder_plot.update()
                    
                    encoder_layer_select.on('update:model-value', on_encoder_layer_change)
                    encoder_channel_select.on('update:model-value', on_encoder_channel_change)
                
                # DECODER ACTIVATIONS
                with ui.card().classes('dark-card p-3 flex-1'):
                    with ui.row().classes('items-center gap-2 mb-2'):
                        ui.label('▌DECODER').style(f'color:#f472b6; font-family: JetBrains Mono; font-size: 0.75rem;')
                        decoder_layer_select = ui.select([], value=None).props('dense dark').classes('w-24').style('font-size: 0.7rem;')
                        decoder_channel_select = ui.select([], value=None).props('dense dark').classes('w-20').style('font-size: 0.7rem;')
                    
                    def update_decoder_selectors():
                        """Update decoder layer/channel selectors based on available activations."""
                        layers = [k for k in AS.activations.keys() if k.startswith('decoder.') or k == 'output']
                        if layers:
                            decoder_layer_select.options = layers
                            if decoder_layer_select.value not in layers:
                                decoder_layer_select.value = layers[0]
                            # Update channel selector
                            if decoder_layer_select.value and decoder_layer_select.value in AS.activations:
                                act = AS.activations[decoder_layer_select.value]
                                if len(act.shape) >= 2:
                                    n_channels = act.shape[1] if len(act.shape) == 4 else act.shape[0]
                                    ch_options = [f'Ch {i}' for i in range(min(n_channels, 64))]
                                    decoder_channel_select.options = ch_options
                                    if not decoder_channel_select.value or decoder_channel_select.value not in ch_options:
                                        decoder_channel_select.value = ch_options[0] if ch_options else None
                    
                    def make_decoder_fig():
                        """Create decoder activation visualization."""
                        fig = go.Figure()
                        
                        layer_name = decoder_layer_select.value
                        channel_str = decoder_channel_select.value
                        
                        if layer_name and layer_name in AS.activations and channel_str:
                            act = AS.activations[layer_name]
                            channel_idx = int(channel_str.split(' ')[1]) if channel_str else 0
                            
                            # Handle different activation shapes
                            if len(act.shape) == 4:  # [B, C, H, W]
                                act_2d = act[0, channel_idx].numpy()
                            elif len(act.shape) == 3:  # [C, H, W]
                                act_2d = act[channel_idx].numpy()
                            elif len(act.shape) == 2:
                                act_2d = act[0].numpy() if act.shape[0] == 1 else act.numpy()
                            else:
                                act_2d = act.numpy().flatten().reshape(-1, 1)
                            
                            fig.add_trace(go.Heatmap(
                                z=act_2d,
                                colorscale='Magma',
                                showscale=True,
                                colorbar=dict(len=0.8, thickness=10)
                            ))
                        else:
                            fig.add_annotation(text="No activations", x=0.5, y=0.5, showarrow=False,
                                             font=dict(color=THEME_TEXT_DIM))
                        
                        fig.update_layout(
                            template='plotly_dark',
                            paper_bgcolor='rgba(8,8,8,1)',
                            plot_bgcolor='rgba(8,8,8,1)',
                            height=180,
                            margin=dict(l=10, r=40, t=5, b=10),
                            xaxis=dict(showticklabels=False, showgrid=False),
                            yaxis=dict(showticklabels=False, showgrid=False, scaleanchor='x'),
                            font=dict(family='JetBrains Mono', size=8, color=THEME_TEXT)
                        )
                        return fig
                    
                    decoder_plot = ui.plotly(make_decoder_fig()).classes('w-full').style('height: 180px;')
                    AS.activation_plots['decoder'] = decoder_plot
                    
                    def on_decoder_layer_change(e):
                        update_decoder_selectors()
                        decoder_plot.figure = make_decoder_fig()
                        decoder_plot.update()
                    
                    def on_decoder_channel_change(e):
                        decoder_plot.figure = make_decoder_fig()
                        decoder_plot.update()
                    
                    decoder_layer_select.on('update:model-value', on_decoder_layer_change)
                    decoder_channel_select.on('update:model-value', on_decoder_channel_change)
            
            # ROW 3: LATENT SPACE (full width, larger)
            with ui.card().classes('dark-card p-3 w-full'):
                with ui.row().classes('items-center gap-4 mb-2'):
                    ui.label('▌LATENT SPACE (PCA)').style(f'color:{THEME_SECONDARY}; font-family: JetBrains Mono; font-size: 0.8rem;')
                    ui.label('').bind_text_from(AS, 'latent_codes', lambda x: f'{len(x)} samples').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                
                def make_latent_fig():
                    """Create latent space PCA visualization."""
                    fig = go.Figure()
                    
                    X_pca, labels = compute_latent_pca()
                    if X_pca is not None and len(X_pca) > 0:
                        # Use more colors for more classes
                        n_classes = len(np.unique(labels))
                        if n_classes <= 10:
                            colors = ['#00ff88', '#f472b6', '#00d4ff', '#ffcc00', '#a78bfa', 
                                     '#00ffcc', '#ff9f43', '#74b9ff', '#55efc4', '#fd79a8']
                        else:
                            # Use colorscale for many classes
                            import plotly.express as px
                            colors = px.colors.qualitative.Alphabet[:n_classes]
                        
                        for label in np.unique(labels):
                            mask = labels == label
                            label_name = AS.class_names[int(label)] if int(label) < len(AS.class_names) else f'C{int(label)}'
                            fig.add_trace(go.Scatter(
                                x=X_pca[mask, 0], y=X_pca[mask, 1],
                                mode='markers',
                                marker=dict(size=6, color=colors[int(label) % len(colors)], opacity=0.7),
                                name=label_name
                            ))
                        
                        # Current point - bright yellow circle instead of star
                        if len(X_pca) > 0:
                            fig.add_trace(go.Scatter(
                                x=[X_pca[-1, 0]], y=[X_pca[-1, 1]],
                                mode='markers',
                                marker=dict(size=14, color='#ffff00', opacity=1, 
                                           line=dict(width=2, color='#000000')),
                                name='Current',
                                showlegend=False
                            ))
                    else:
                        fig.add_annotation(text="Process samples to visualize latent space", 
                                         x=0.5, y=0.5, showarrow=False,
                                         font=dict(color=THEME_TEXT_DIM, size=12))
                    
                    lat_range = AS.axis_ranges['latent']
                    fig.update_layout(
                        template='plotly_dark',
                        paper_bgcolor='rgba(8,8,8,1)',
                        plot_bgcolor='rgba(8,8,8,1)',
                        height=300,
                        margin=dict(l=40, r=20, t=10, b=40),
                        xaxis=dict(title='PC1', gridcolor='rgba(0,255,136,0.1)', 
                                  range=[lat_range['x_min'], lat_range['x_max']]),
                        yaxis=dict(title='PC2', gridcolor='rgba(0,255,136,0.1)',
                                  range=[lat_range['y_min'], lat_range['y_max']]),
                        legend=dict(orientation='h', y=-0.15, x=0.5, xanchor='center', 
                                   font=dict(size=9), bgcolor='rgba(0,0,0,0.5)'),
                        font=dict(family='JetBrains Mono', size=10, color=THEME_TEXT)
                    )
                    return fig
                
                latent_plot = ui.plotly(make_latent_fig()).classes('w-full').style('height: 300px;')
                AS.latent_plot = latent_plot
            
            # ROW 3: ATTENTION WEIGHTS (per layer)
            with ui.card().classes('dark-card p-3 w-full'):
                ui.label('▌ATTENTION WEIGHTS (GAT layers)').style(f'color:{THEME_WARN}; font-family: JetBrains Mono; font-size: 0.75rem;').classes('mb-1')
                
                def make_attention_fig():
                    """Create attention heatmap visualization using REAL GAT attention weights."""
                    from plotly.subplots import make_subplots
                    
                    n_nodes = AS.model_params.get('num_nodes', 24) if AS.model_params else 24
                    
                    # Check how many layers have attention
                    n_layers = len(AS.attention_weights) if AS.attention_weights else 0
                    
                    if n_layers == 0:
                        # Fallback: no attention data yet
                        fig = go.Figure()
                        fig.add_annotation(text="No attention data", x=0.5, y=0.5, showarrow=False)
                        fig.update_layout(
                            template='plotly_dark',
                            paper_bgcolor='rgba(8,8,8,1)',
                            plot_bgcolor='rgba(8,8,8,1)',
                            height=150,
                            margin=dict(l=10, r=10, t=10, b=10)
                        )
                        return fig
                    
                    # Create subplot for each layer
                    fig = make_subplots(rows=1, cols=n_layers, 
                                       subplot_titles=[f'Layer {i+1}' for i in range(n_layers)])
                    
                    for i, (layer_name, att_data) in enumerate(AS.attention_weights.items()):
                        if 'attention' in att_data and 'edge_index' in att_data:
                            edge_index = att_data['edge_index'].numpy()
                            alpha = att_data['attention'].numpy()
                            
                            # Build attention matrix from sparse edge data
                            # alpha shape: (num_edges, num_heads) - average across heads
                            if len(alpha.shape) > 1:
                                alpha_avg = alpha.mean(axis=1)
                            else:
                                alpha_avg = alpha
                            
                            # Create dense attention matrix
                            actual_nodes = min(n_nodes, int(edge_index.max()) + 1) if edge_index.size > 0 else n_nodes
                            attn_matrix = np.zeros((actual_nodes, actual_nodes))
                            
                            for e_idx in range(edge_index.shape[1]):
                                src, tgt = edge_index[0, e_idx], edge_index[1, e_idx]
                                if src < actual_nodes and tgt < actual_nodes:
                                    attn_matrix[src, tgt] = alpha_avg[e_idx] if e_idx < len(alpha_avg) else 0
                            
                            fig.add_trace(go.Heatmap(
                                z=attn_matrix,
                                colorscale='Viridis',
                                showscale=(i == n_layers - 1),  # Only show colorbar on last
                                zmin=0,
                                zmax=1,
                                colorbar=dict(title='α', len=0.8) if i == n_layers - 1 else None
                            ), row=1, col=i+1)
                    
                    fig.update_layout(
                        template='plotly_dark',
                        paper_bgcolor='rgba(8,8,8,1)',
                        plot_bgcolor='rgba(8,8,8,1)',
                        height=150,
                        margin=dict(l=30, r=50, t=25, b=25),
                        font=dict(family='JetBrains Mono', size=8, color=THEME_TEXT)
                    )
                    
                    # Update axes for all subplots
                    for i in range(n_layers):
                        fig.update_xaxes(title_text='Target' if i == 0 else '', row=1, col=i+1)
                        fig.update_yaxes(title_text='Source' if i == 0 else '', row=1, col=i+1)
                    
                    return fig
                
                attention_plot = ui.plotly(make_attention_fig()).classes('w-full').style('height: 180px;')
                AS.attention_plots['main'] = attention_plot
            
            # BOTTOM ROW: Reconstruction
            with ui.row().classes('gap-3 w-full'):
                
                # ORIGINAL
                with ui.card().classes('dark-card p-3 flex-1'):
                    ui.label('▌ORIGINAL').style(f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-2')
                    
                    def make_original_fig():
                        fig = go.Figure()
                        
                        if AS.model_type == 'image' and AS.current_sample is not None:
                            # Image model: show RGB image
                            img = AS.current_sample.cpu().numpy()
                            if img.ndim == 4:
                                img = img[0]  # Remove batch dim
                            # [C, H, W] -> [H, W, C]
                            if img.shape[0] == 3:
                                img = np.transpose(img, (1, 2, 0))
                            # Clip to [0, 1] for display
                            img = np.clip(img, 0, 1)
                            # Resize for faster display if too large
                            display_size = min(256, img.shape[0])
                            if img.shape[0] > display_size:
                                from scipy.ndimage import zoom
                                scale = display_size / img.shape[0]
                                img = zoom(img, (scale, scale, 1), order=1)
                            fig.add_trace(go.Image(z=(img * 255).astype(np.uint8)))
                            fig.update_layout(height=200)
                        elif AS.current_sample is not None and hasattr(AS.current_sample, 'x'):
                            # Graph model: show heatmap
                            feat_range = AS.axis_ranges['node_features']
                            x = AS.current_sample.x.cpu().numpy()
                            fig.add_trace(go.Heatmap(
                                z=x[:24, :].T, 
                                colorscale='Viridis', 
                                showscale=False,
                                zmin=feat_range['min'],
                                zmax=feat_range['max']
                            ))
                        
                        fig.update_layout(
                            template='plotly_dark',
                            paper_bgcolor='rgba(8,8,8,1)',
                            plot_bgcolor='rgba(8,8,8,1)',
                            height=200 if AS.model_type == 'image' else 150,
                            margin=dict(l=10, r=10, t=10, b=10) if AS.model_type == 'image' else dict(l=30, r=10, t=10, b=30),
                            xaxis=dict(showticklabels=False, showgrid=False) if AS.model_type == 'image' else dict(title='Nodes'),
                            yaxis=dict(showticklabels=False, showgrid=False) if AS.model_type == 'image' else dict(title='Features'),
                            font=dict(family='JetBrains Mono', size=9, color=THEME_TEXT)
                        )
                        return fig
                    
                    original_plot = ui.plotly(make_original_fig()).classes('w-full').style('height: 200px;')
                    AS.activation_plots['original'] = original_plot
                
                # RECONSTRUCTED
                with ui.card().classes('dark-card p-3 flex-1'):
                    ui.label('▌RECONSTRUCTED').style(f'color:#f472b6; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-2')
                    
                    def make_recon_fig():
                        fig = go.Figure()
                        
                        if AS.model_type == 'image' and AS.current_recon is not None:
                            # Image model: show RGB reconstruction
                            recon_key = 'reconstruction' if 'reconstruction' in AS.current_recon else 'x_recon'
                            if recon_key in AS.current_recon:
                                img = AS.current_recon[recon_key].cpu().numpy()
                                if img.ndim == 4:
                                    img = img[0]  # Remove batch dim
                                # [C, H, W] -> [H, W, C]
                                if img.shape[0] == 3:
                                    img = np.transpose(img, (1, 2, 0))
                                img = np.clip(img, 0, 1)
                                # Resize for display
                                display_size = min(256, img.shape[0])
                                if img.shape[0] > display_size:
                                    from scipy.ndimage import zoom
                                    scale = display_size / img.shape[0]
                                    img = zoom(img, (scale, scale, 1), order=1)
                                fig.add_trace(go.Image(z=(img * 255).astype(np.uint8)))
                                fig.update_layout(height=200)
                        elif AS.current_recon is not None and 'x_recon' in AS.current_recon:
                            # Graph model: show heatmap
                            feat_range = AS.axis_ranges['node_features']
                            x_recon = AS.current_recon['x_recon'].cpu().numpy()
                            fig.add_trace(go.Heatmap(
                                z=x_recon[:24, :].T, 
                                colorscale='Viridis', 
                                showscale=False,
                                zmin=feat_range['min'],
                                zmax=feat_range['max']
                            ))
                        
                        fig.update_layout(
                            template='plotly_dark',
                            paper_bgcolor='rgba(8,8,8,1)',
                            plot_bgcolor='rgba(8,8,8,1)',
                            height=200 if AS.model_type == 'image' else 150,
                            margin=dict(l=10, r=10, t=10, b=10) if AS.model_type == 'image' else dict(l=30, r=10, t=10, b=30),
                            xaxis=dict(showticklabels=False, showgrid=False) if AS.model_type == 'image' else dict(title='Nodes'),
                            yaxis=dict(showticklabels=False, showgrid=False) if AS.model_type == 'image' else dict(title='Features'),
                            font=dict(family='JetBrains Mono', size=9, color=THEME_TEXT)
                        )
                        return fig
                    
                    recon_plot = ui.plotly(make_recon_fig()).classes('w-full').style('height: 200px;')
                    AS.recon_plot = recon_plot
                
                # DIFFERENCE
                with ui.card().classes('dark-card p-3 flex-1'):
                    ui.label('▌DIFFERENCE').style(f'color:{THEME_ERROR}; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-2')
                    
                    def make_diff_fig():
                        fig = go.Figure()
                        
                        if AS.model_type == 'image' and AS.current_sample is not None and AS.current_recon is not None:
                            # Image model: show difference image
                            recon_key = 'reconstruction' if 'reconstruction' in AS.current_recon else 'x_recon'
                            if recon_key in AS.current_recon:
                                orig = AS.current_sample.cpu().numpy()
                                recon = AS.current_recon[recon_key].cpu().numpy()
                                if orig.ndim == 4:
                                    orig = orig[0]
                                if recon.ndim == 4:
                                    recon = recon[0]
                                # Compute absolute difference
                                diff = np.abs(orig - recon)
                                # [C, H, W] -> [H, W, C]
                                if diff.shape[0] == 3:
                                    diff = np.transpose(diff, (1, 2, 0))
                                # Amplify for visibility and convert to grayscale-ish
                                diff_gray = np.mean(diff, axis=2)
                                diff_display = np.stack([diff_gray, diff_gray * 0.3, diff_gray * 0.3], axis=2)
                                diff_display = np.clip(diff_display * 3, 0, 1)  # Amplify
                                # Resize
                                display_size = min(256, diff_display.shape[0])
                                if diff_display.shape[0] > display_size:
                                    from scipy.ndimage import zoom
                                    scale = display_size / diff_display.shape[0]
                                    diff_display = zoom(diff_display, (scale, scale, 1), order=1)
                                fig.add_trace(go.Image(z=(diff_display * 255).astype(np.uint8)))
                                fig.update_layout(height=200)
                        elif AS.current_sample is not None and AS.current_recon is not None:
                            # Graph model
                            if hasattr(AS.current_sample, 'x') and 'x_recon' in AS.current_recon:
                                feat_range = AS.axis_ranges['node_features']
                                diff_max = (feat_range['max'] - feat_range['min']) * 0.5
                                x = AS.current_sample.x.cpu().numpy()
                                x_recon = AS.current_recon['x_recon'].cpu().numpy()
                                diff = np.abs(x - x_recon)
                                fig.add_trace(go.Heatmap(
                                    z=diff[:24, :].T, 
                                    colorscale='Reds', 
                                    showscale=True,
                                    zmin=0,
                                    zmax=diff_max,
                                    colorbar=dict(title='|Δ|', len=0.8)
                                ))
                        
                        fig.update_layout(
                            template='plotly_dark',
                            paper_bgcolor='rgba(8,8,8,1)',
                            plot_bgcolor='rgba(8,8,8,1)',
                            height=200 if AS.model_type == 'image' else 150,
                            margin=dict(l=10, r=10, t=10, b=10) if AS.model_type == 'image' else dict(l=30, r=50, t=10, b=30),
                            xaxis=dict(showticklabels=False, showgrid=False) if AS.model_type == 'image' else dict(title='Nodes'),
                            yaxis=dict(showticklabels=False, showgrid=False) if AS.model_type == 'image' else dict(title='Features'),
                            font=dict(family='JetBrains Mono', size=9, color=THEME_TEXT)
                        )
                        return fig
                    
                    diff_plot = ui.plotly(make_diff_fig()).classes('w-full').style('height: 200px;')
                    AS.activation_plots['diff'] = diff_plot
        
        # Process sample and update all plots
        def process_current_sample():
            if not AS.dataset:
                analysis_log("No dataset loaded", 'warning')
                return
            if not AS.model:
                analysis_log("No model loaded", 'warning')
                return
            
            try:
                sample = AS.dataset[AS.current_idx]
                result = process_sample(sample)
                
                if result is None:
                    analysis_log(f"Failed to process sample {AS.current_idx}", 'error')
                    return
                
                # Update sample info based on model type
                if AS.sample_info_container:
                    AS.sample_info_container.clear()
                    with AS.sample_info_container:
                        ui.label(f'Index: {AS.current_idx}').style(f'color:{THEME_PRIMARY}; font-size: 0.7rem;')
                        
                        if AS.model_type == 'image':
                            # Image model info
                            if AS.current_label is not None:
                                lbl = AS.current_label
                                class_name = AS.class_names[lbl] if lbl < len(AS.class_names) else str(lbl)
                                ui.label(f'Class: {class_name} ({lbl})').style(f'color:{THEME_SECONDARY}; font-size: 0.7rem;')
                            if AS.current_sample is not None:
                                shape = AS.current_sample.shape
                                if len(shape) == 4:
                                    ui.label(f'Size: {shape[2]}×{shape[3]} RGB').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                                elif len(shape) == 3:
                                    ui.label(f'Size: {shape[1]}×{shape[2]} RGB').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                            # Show classification prediction if available
                            if result is not None and 'class_logits' in result:
                                import torch
                                pred = torch.argmax(result['class_logits'], dim=-1).item()
                                pred_name = AS.class_names[pred] if pred < len(AS.class_names) else str(pred)
                                ui.label(f'Predicted: {pred_name}').style(f'color:{THEME_WARN}; font-size: 0.7rem;')
                        else:
                            # Graph model info
                            if hasattr(sample, 'y') and sample.y is not None:
                                lbl = sample.y.item() if hasattr(sample.y, "item") else sample.y
                                ui.label(f'Label: {lbl}').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                            if hasattr(sample, 'x'):
                                ui.label(f'Nodes: {sample.x.shape[0]}, Feat: {sample.x.shape[1]}').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                            if hasattr(sample, 'graph_attr') and sample.graph_attr is not None:
                                k_mean = sample.graph_attr[0].item() if hasattr(sample.graph_attr[0], 'item') else sample.graph_attr[0]
                                ui.label(f'Kuramoto: {k_mean:.3f}').style(f'color:{THEME_WARN}; font-size: 0.7rem;')
                                # Track Kuramoto
                                AS.kuramoto_history.append((AS.current_idx, k_mean))
                
                # Update sample label
                sample_label.set_text(f'Sample: {AS.current_idx + 1} / {AS.total_samples}')
                
                # Update all plots
                try:
                    kuramoto_plot.figure = make_kuramoto_fig()
                    kuramoto_plot.update()
                except Exception as e:
                    analysis_log(f"Kuramoto plot error: {e}", 'warning')
                
                try:
                    # Update encoder selectors and plot
                    update_encoder_selectors()
                    encoder_plot.figure = make_encoder_fig()
                    encoder_plot.update()
                except Exception as e:
                    analysis_log(f"Encoder plot error: {e}", 'warning')
                
                try:
                    # Update decoder selectors and plot
                    update_decoder_selectors()
                    decoder_plot.figure = make_decoder_fig()
                    decoder_plot.update()
                except Exception as e:
                    analysis_log(f"Decoder plot error: {e}", 'warning')
                
                try:
                    latent_plot.figure = make_latent_fig()
                    latent_plot.update()
                except Exception as e:
                    analysis_log(f"Latent plot error: {e}", 'warning')
                
                try:
                    attention_plot.figure = make_attention_fig()
                    attention_plot.update()
                except Exception as e:
                    analysis_log(f"Attention plot error: {e}", 'warning')
                
                try:
                    original_plot.figure = make_original_fig()
                    original_plot.update()
                except Exception as e:
                    analysis_log(f"Original plot error: {e}", 'warning')
                
                try:
                    recon_plot.figure = make_recon_fig()
                    recon_plot.update()
                except Exception as e:
                    analysis_log(f"Recon plot error: {e}", 'warning')
                
                try:
                    diff_plot.figure = make_diff_fig()
                    diff_plot.update()
                except Exception as e:
                    analysis_log(f"Diff plot error: {e}", 'warning')
                
            except Exception as e:
                import traceback
                analysis_log(f"Error processing sample: {e}", 'error')
                analysis_log(traceback.format_exc(), 'error')


# Update main page header to include navigation
@ui.page('/')
def main_with_nav():
    ui.add_head_html(f'<style>{STYLE}</style>')
    
    with ui.header().classes('items-center px-4 py-1').style(f'background: {THEME_BG}; border-bottom: 1px solid {THEME_BORDER};'):
        ui.label('▶').style(f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.75rem; letter-spacing: 2px;')
        ui.label('EEG_VIEWER').classes('text-base font-medium ml-2').style(f'color: {THEME_PRIMARY}; font-family: JetBrains Mono; letter-spacing: 1px;')
        ui.label('v1.0').classes('text-xs ml-2').style(f'color: {THEME_TEXT_DIM}; font-family: JetBrains Mono;')
        with ui.row().classes('ml-auto gap-2'):
            ui.button('VIEWER', on_click=lambda: ui.navigate.to('/')).props('flat dense').style(f'color:{THEME_PRIMARY};')
            ui.button('PIPELINE', on_click=lambda: ui.navigate.to('/pipeline')).props('flat dense').style(f'color:{THEME_TEXT_DIM};')
            ui.button('MODEL', on_click=lambda: ui.navigate.to('/model')).props('flat dense').style(f'color:{THEME_TEXT_DIM};')
            ui.button('ANALYSIS', on_click=lambda: ui.navigate.to('/analysis')).props('flat dense').style(f'color:{THEME_TEXT_DIM};')
    
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
                        else:
                            S.eeg_data = loaded
                            eeg_chs = [ch for ch, t in loaded.channel_types.items() if t == 'eeg']
                            S.selected_channels = eeg_chs[:10]
                            S.hilbert_channel = eeg_chs[0] if eeg_chs else ""
                            S.view_start = 0
                            S.epochs = []
                            S.current_amplitudes = {}
                            ui.notify(f'EEG 1: {loaded.filename}', type='positive')
                            refresh_channels()
                            refresh_hilbert_select()
                        
                        refresh_info()
                        update_all()
                    except Exception as e:
                        ui.notify(f'Error: {e}', type='negative')
                
                with ui.scroll_area().classes('w-full').style('height: 250px;'):
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
        
        # MAIN CONTENT
        with ui.column().classes('flex-1 gap-3'):
            
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
                        update_all()
                    ui.button('Apply', on_click=apply_filt, icon='check').props('dense')
            
            # TOP ROW: EEG + BRAIN (with comparison support)
            with ui.row().classes('gap-3 w-full'):
                # EEG 1
                with ui.card().classes('dark-card p-3 flex-1'):
                    with ui.row().classes('items-center justify-between mb-1'):
                        ui.label('▌EEG 1').style(f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.8rem; letter-spacing: 1px;')
                        with ui.row().classes('gap-1'):
                            ui.button('-', on_click=lambda: (setattr(S, 'scale_factor', max(0.2, S.scale_factor*0.7)), update_eeg())).props('dense flat size=xs')
                            ui.button('1x', on_click=lambda: (setattr(S, 'scale_factor', 1.0), update_eeg())).props('dense flat size=xs')
                            ui.button('+', on_click=lambda: (setattr(S, 'scale_factor', min(5, S.scale_factor*1.4)), update_eeg())).props('dense flat size=xs')
                    S.eeg_plot = ui.plotly(make_eeg_fig()).classes('w-full')
                
                # EEG 2 (comparison)
                with ui.card().classes('dark-card p-3 flex-1').bind_visibility_from(S, 'compare_mode'):
                    ui.label('▌EEG 2').style(f'color:#f472b6; font-family: JetBrains Mono; font-size: 0.8rem; letter-spacing: 1px;').classes('mb-1')
                    S.eeg_plot2 = ui.plotly(make_eeg_fig(use_eeg2=True)).classes('w-full')
            
            # Navigation (shared)
            with ui.card().classes('dark-card p-2'):
                with ui.row().classes('items-center justify-center gap-2'):
                    ui.button(icon='skip_previous', on_click=nav_start).props('round dense size=sm')
                    ui.button(icon='fast_rewind', on_click=nav_back).props('round dense size=sm')
                    ui.button(icon='play_arrow', on_click=toggle_play).props('round dense size=sm color=red')
                    ui.button(icon='fast_forward', on_click=nav_fwd).props('round dense size=sm')
                    ui.button(icon='skip_next', on_click=nav_end).props('round dense size=sm')
                    ui.separator().props('vertical').classes('mx-2')
                    ui.button('2s', on_click=lambda: set_win(2)).props('dense size=xs')
                    ui.button('5s', on_click=lambda: set_win(5)).props('dense size=xs')
                    ui.button('10s', on_click=lambda: set_win(10)).props('dense size=xs')
                    ui.button('20s', on_click=lambda: set_win(20)).props('dense size=xs')
                    ui.separator().props('vertical').classes('mx-2')
                    S.time_label = ui.label('0:00.0 / 0:00.0').classes('text-sm font-mono opacity-70')
            
            # TOPOGRAPHY ROW
            with ui.row().classes('gap-3 w-full'):
                with ui.card().classes('dark-card p-3 flex-1'):
                    ui.label('▌TOPO 1').style(f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-1')
                    S.brain_plot = ui.plotly(make_brain_fig()).classes('w-full')
                
                with ui.card().classes('dark-card p-3 flex-1').bind_visibility_from(S, 'compare_mode'):
                    ui.label('▌TOPO 2').style(f'color:#f472b6; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-1')
                    S.brain_plot2 = ui.plotly(make_brain_fig(use_eeg2=True)).classes('w-full')
            
            # FFT ROW
            with ui.row().classes('gap-3 w-full'):
                with ui.card().classes('dark-card p-3 flex-1'):
                    ui.label('▌FFT 1').style(f'color:{THEME_SECONDARY}; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-1')
                    S.fft_plot = ui.plotly(make_fft_fig()).classes('w-full')
                
                with ui.card().classes('dark-card p-3 flex-1').bind_visibility_from(S, 'compare_mode'):
                    ui.label('▌FFT 2').style(f'color:#f472b6; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-1')
                    S.fft_plot2 = ui.plotly(make_fft_fig(use_eeg2=True)).classes('w-full')
            
            # HILBERT ROW
            with ui.row().classes('gap-3 w-full'):
                with ui.card().classes('dark-card p-3 flex-1'):
                    with ui.row().classes('items-center gap-2 mb-1'):
                        ui.label('▌HILBERT 1').style(f'color:{THEME_WARN}; font-family: JetBrains Mono; font-size: 0.8rem;')
                        S.hilbert_select_container = ui.row().classes('items-center gap-1')
                        refresh_hilbert_select()
                    S.hilbert_plot = ui.plotly(make_hilbert_fig()).classes('w-full')
                
                with ui.card().classes('dark-card p-3 flex-1').bind_visibility_from(S, 'compare_mode'):
                    ui.label('▌HILBERT 2').style(f'color:#f472b6; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-1')
                    S.hilbert_plot2 = ui.plotly(make_hilbert_fig(use_eeg2=True)).classes('w-full')
            
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


if __name__ in {"__main__", "__mp_main__"}:
    print("EEG VIEWER - http://localhost:8080")
    print("  - Viewer:   http://localhost:8080/")
    print("  - Pipeline: http://localhost:8080/pipeline")
    print("  - Model:    http://localhost:8080/model")
    ui.run(title='EEG Viewer', port=8080, reload=False, show=False, dark=True)
