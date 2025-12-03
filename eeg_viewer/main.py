#!/usr/bin/env python3
"""
EEG Viewer - Powerful EEG Processing Interface
"""
import asyncio
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
        self.fft_plot = None
        self.hilbert_plot = None
        self.brain_plot = None
        self.time_label = None
        self.info_container = None
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
"""

# Plot creation with FIXED axes - Terminal style
PLOT_BG = 'rgba(8,8,8,1)'
PLOT_GRID = 'rgba(0,255,136,0.08)'
PLOT_GRID_MINOR = 'rgba(0,255,136,0.03)'

def make_eeg_fig():
    fig = go.Figure()
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=PLOT_BG,
        margin=dict(l=70, r=10, t=10, b=50),
        height=280,
        font=dict(family='JetBrains Mono, monospace', size=10, color=THEME_TEXT),
        xaxis=dict(
            title=dict(text='TIME [s]', font=dict(size=9, color=THEME_PRIMARY)),
            gridcolor=PLOT_GRID,
            zerolinecolor=PLOT_GRID,
            tickfont=dict(size=9, color=THEME_TEXT_DIM),
            fixedrange=False
        ),
        yaxis=dict(
            gridcolor=PLOT_GRID_MINOR,
            tickfont=dict(size=9, color=THEME_PRIMARY),
            fixedrange=True
        ),
        showlegend=False,
        hovermode='x unified',
        hoverlabel=dict(bgcolor=THEME_CARD, font=dict(family='JetBrains Mono', size=10))
    )
    return fig

def make_fft_fig():
    fig = go.Figure()
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
        height=220,
        font=dict(family='JetBrains Mono, monospace', size=10, color=THEME_TEXT),
        xaxis=dict(
            title=dict(text='FREQ [Hz]', font=dict(size=9, color=THEME_SECONDARY)),
            gridcolor=PLOT_GRID,
            range=[0, 60],
            fixedrange=True,
            tickfont=dict(size=9, color=THEME_TEXT_DIM)
        ),
        yaxis=dict(
            title=dict(text='PWR [µV]', font=dict(size=9, color=THEME_SECONDARY)),
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

def make_hilbert_fig():
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=('<b>ENVELOPE</b>', '<b>PHASE</b>'),
                        vertical_spacing=0.22)
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=PLOT_BG,
        margin=dict(l=60, r=10, t=35, b=50),
        height=220,
        font=dict(family='JetBrains Mono, monospace', size=10, color=THEME_TEXT),
        showlegend=True,
        legend=dict(orientation='h', y=1.15, font=dict(size=8, color=THEME_TEXT_DIM)),
        hovermode='x unified',
        hoverlabel=dict(bgcolor=THEME_CARD, font=dict(family='JetBrains Mono', size=10))
    )
    fig.update_annotations(font=dict(size=9, color=THEME_WARN, family='JetBrains Mono'))
    fig.update_xaxes(gridcolor=PLOT_GRID, tickfont=dict(size=9, color=THEME_TEXT_DIM), fixedrange=True)
    fig.update_yaxes(gridcolor=PLOT_GRID, tickfont=dict(size=9, color=THEME_TEXT_DIM), fixedrange=True)
    return fig

def make_brain_fig():
    fig = go.Figure()
    theta = np.linspace(0, 2*np.pi, 100)
    # Head outline - terminal green with glow effect
    head_color = 'rgba(0,255,136,0.5)'
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
        height=280,
        font=dict(family='JetBrains Mono, monospace', color=THEME_TEXT),
        xaxis=dict(range=[-1.25, 1.25], showgrid=False, zeroline=False, showticklabels=False, scaleanchor='y', fixedrange=True),
        yaxis=dict(range=[-0.9, 1.2], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
        showlegend=False,
        hovermode='closest',
        hoverlabel=dict(bgcolor=THEME_CARD, font=dict(family='JetBrains Mono', size=10, color=THEME_PRIMARY))
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
        if S.eeg_data:
            ui.label(S.eeg_data.filename).style(f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.85rem;')
            with ui.row().classes('gap-1 items-center'):
                ui.label('sfreq:').style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.75rem;')
                ui.label(f'{S.eeg_data.sfreq:.0f} Hz').style(f'color:{THEME_SECONDARY}; font-family: JetBrains Mono; font-size: 0.75rem;')
            with ui.row().classes('gap-1 items-center'):
                ui.label('channels:').style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.75rem;')
                ui.label(str(S.eeg_data.n_channels)).style(f'color:{THEME_SECONDARY}; font-family: JetBrains Mono; font-size: 0.75rem;')
            with ui.row().classes('gap-1 items-center'):
                ui.label('duration:').style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.75rem;')
                ui.label(f'{S.eeg_data.duration_sec:.1f}s').style(f'color:{THEME_SECONDARY}; font-family: JetBrains Mono; font-size: 0.75rem;')
        else:
            ui.label('-- no file loaded --').style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.8rem;')

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
                        # State for visualization
                        viz_state = {'data': None, 'file': None}
                        
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
                                            try:
                                                from viz_scripts import brain_3d
                                                data = viz_state.get('data')
                                                
                                                with network_plot_container:
                                                    fig = brain_3d.create_network_comparison_figure(data, viz_band.value)
                                                    ui.plotly(fig).classes('w-full').style('height: 350px;')
                                            except Exception as e:
                                                with network_plot_container:
                                                    ui.label(f'Error: {e}').style(f'color:{THEME_TEXT_DIM};')
                                        
                                        update_network_plot()
                                
                                # SECOND ROW - Kuramoto Analysis
                                with ui.row().classes('w-full gap-3'):
                                    # Timeline
                                    with ui.card().classes('dark-card p-3 flex-1'):
                                        ui.label('▌KURAMOTO TIMELINE').style(f'color:{THEME_WARN}; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-2')
                                        
                                        kura_timeline_container = ui.column().classes('w-full')
                                        
                                        def update_kuramoto_timeline():
                                            kura_timeline_container.clear()
                                            try:
                                                from viz_scripts import kuramoto_viz
                                                data = viz_state.get('data')
                                                
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
                                        update_kuramoto_timeline()
                                    
                                    # Band Comparison
                                    with ui.card().classes('dark-card p-3 flex-1'):
                                        ui.label('▌BAND COMPARISON').style(f'color:{THEME_SECONDARY}; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-2')
                                        
                                        band_comp_container = ui.column().classes('w-full')
                                        
                                        def update_band_comparison():
                                            band_comp_container.clear()
                                            try:
                                                from viz_scripts import kuramoto_viz
                                                data = viz_state.get('data')
                                                
                                                with band_comp_container:
                                                    fig = kuramoto_viz.create_band_comparison_figure(data)
                                                    ui.plotly(fig).classes('w-full').style('height: 280px;')
                                            except Exception as e:
                                                with band_comp_container:
                                                    ui.label(f'Error: {e}').style(f'color:{THEME_TEXT_DIM};')
                                        
                                        update_band_comparison()
                                
                                # THIRD ROW - More Analysis
                                with ui.row().classes('w-full gap-3'):
                                    # Phase Distribution
                                    with ui.card().classes('dark-card p-3 flex-1'):
                                        ui.label('▌PHASE DISTRIBUTION').style(f'color:#a78bfa; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-2')
                                        
                                        phase_container = ui.column().classes('w-full')
                                        
                                        def update_phase_plot():
                                            phase_container.clear()
                                            try:
                                                from viz_scripts import kuramoto_viz
                                                data = viz_state.get('data')
                                                
                                                with phase_container:
                                                    fig = kuramoto_viz.create_phase_distribution_figure(data, viz_band.value, int(viz_epoch.value or 0))
                                                    ui.plotly(fig).classes('w-full').style('height: 280px;')
                                            except Exception as e:
                                                with phase_container:
                                                    ui.label(f'Error: {e}').style(f'color:{THEME_TEXT_DIM};')
                                        
                                        update_phase_plot()
                                    
                                    # Sync Matrix
                                    with ui.card().classes('dark-card p-3 flex-1'):
                                        ui.label('▌SYNC MATRIX').style(f'color:#60a5fa; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-2')
                                        
                                        sync_matrix_container = ui.column().classes('w-full')
                                        
                                        def update_sync_matrix():
                                            sync_matrix_container.clear()
                                            try:
                                                from viz_scripts import kuramoto_viz
                                                data = viz_state.get('data')
                                                
                                                with sync_matrix_container:
                                                    fig = kuramoto_viz.create_heatmap_figure(data, viz_band.value, int(viz_epoch.value or 0))
                                                    ui.plotly(fig).classes('w-full').style('height: 280px;')
                                            except Exception as e:
                                                with sync_matrix_container:
                                                    ui.label(f'Error: {e}').style(f'color:{THEME_TEXT_DIM};')
                                        
                                        update_sync_matrix()
                                    
                                    # Connectivity Graph
                                    with ui.card().classes('dark-card p-3 flex-1'):
                                        ui.label('▌ROI CONNECTIVITY').style(f'color:#22c55e; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-2')
                                        
                                        connectivity_container = ui.column().classes('w-full')
                                        conn_threshold = ui.slider(min=0.3, max=0.9, step=0.1, value=0.5).props('label-always').classes('w-full')
                                        
                                        def update_connectivity():
                                            connectivity_container.clear()
                                            try:
                                                from viz_scripts import kuramoto_viz
                                                data = viz_state.get('data')
                                                
                                                with connectivity_container:
                                                    fig = kuramoto_viz.create_roi_connectivity_figure(data, viz_band.value, conn_threshold.value)
                                                    ui.plotly(fig).classes('w-full').style('height: 250px;')
                                            except Exception as e:
                                                with connectivity_container:
                                                    ui.label(f'Error: {e}').style(f'color:{THEME_TEXT_DIM};')
                                        
                                        conn_threshold.on('update:model-value', lambda e: update_connectivity())
                                        update_connectivity()
                                
                                # FOURTH ROW - Hilbert Transform Visualizations
                                with ui.row().classes('w-full gap-3'):
                                    # Hilbert 2D
                                    with ui.card().classes('dark-card p-3 flex-1'):
                                        ui.label('▌HILBERT 2D').style(f'color:#f472b6; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-2')
                                        
                                        hilbert_2d_container = ui.column().classes('w-full')
                                        
                                        def update_hilbert_2d():
                                            hilbert_2d_container.clear()
                                            try:
                                                from viz_scripts import kuramoto_viz
                                                data = viz_state.get('data')
                                                
                                                with hilbert_2d_container:
                                                    fig = kuramoto_viz.create_hilbert_2d_figure(data, viz_band.value, int(viz_epoch.value or 0))
                                                    ui.plotly(fig).classes('w-full').style('height: 500px;')
                                            except Exception as e:
                                                with hilbert_2d_container:
                                                    ui.label(f'Error: {e}').style(f'color:{THEME_TEXT_DIM};')
                                        
                                        # Store reference for later updates
                                        viz_state['update_hilbert_2d'] = update_hilbert_2d
                                        update_hilbert_2d()
                                    
                                    # Hilbert 3D
                                    with ui.card().classes('dark-card p-3 flex-1'):
                                        ui.label('▌HILBERT 3D PHASE SPACE').style(f'color:#c084fc; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-2')
                                        
                                        hilbert_3d_container = ui.column().classes('w-full')
                                        
                                        def update_hilbert_3d():
                                            hilbert_3d_container.clear()
                                            try:
                                                from viz_scripts import kuramoto_viz
                                                data = viz_state.get('data')
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
                
                async def load_file(fp):
                    try:
                        S.eeg_data = await asyncio.get_event_loop().run_in_executor(None, load_eeg_file, Path(fp))
                        eeg_chs = [ch for ch, t in S.eeg_data.channel_types.items() if t == 'eeg']
                        S.selected_channels = eeg_chs[:10]
                        S.hilbert_channel = eeg_chs[0] if eeg_chs else ""
                        S.view_start = 0
                        S.epochs = []
                        S.current_amplitudes = {}
                        ui.notify(f'Loaded: {S.eeg_data.filename}', type='positive')
                        refresh_info()
                        refresh_channels()
                        refresh_hilbert_select()
                        update_all()
                    except Exception as e:
                        ui.notify(f'Error: {e}', type='negative')
                
                with ui.scroll_area().classes('w-full').style('height: 300px;'):
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
            
            # TOP ROW: EEG + BRAIN
            with ui.row().classes('gap-4 w-full'):
                with ui.card().classes('dark-card p-3').style('flex: 2;'):
                    with ui.row().classes('items-center justify-between mb-1'):
                        ui.label('▌EEG_SIGNAL').style(f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.8rem; letter-spacing: 1px;')
                        with ui.row().classes('gap-1'):
                            ui.button('-', on_click=lambda: (setattr(S, 'scale_factor', max(0.2, S.scale_factor*0.7)), update_eeg())).props('dense flat size=xs')
                            ui.button('1x', on_click=lambda: (setattr(S, 'scale_factor', 1.0), update_eeg())).props('dense flat size=xs')
                            ui.button('+', on_click=lambda: (setattr(S, 'scale_factor', min(5, S.scale_factor*1.4)), update_eeg())).props('dense flat size=xs')
                    
                    S.eeg_plot = ui.plotly(make_eeg_fig()).classes('w-full')
                    
                    with ui.row().classes('items-center justify-center gap-2 mt-2'):
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
                
                with ui.card().classes('dark-card p-3').style('flex: 1; min-width: 320px;'):
                    ui.label('▌TOPOGRAPHY').style(f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.8rem; letter-spacing: 1px;').classes('mb-1')
                    S.brain_plot = ui.plotly(make_brain_fig()).classes('w-full')
            
            # BOTTOM ROW: FFT + HILBERT
            with ui.row().classes('gap-4 w-full'):
                with ui.card().classes('dark-card p-3 flex-1'):
                    ui.label('▌FFT_SPECTRUM').style(f'color:{THEME_SECONDARY}; font-family: JetBrains Mono; font-size: 0.8rem; letter-spacing: 1px;').classes('mb-1')
                    S.fft_plot = ui.plotly(make_fft_fig()).classes('w-full')
                
                with ui.card().classes('dark-card p-3 flex-1'):
                    with ui.row().classes('items-center gap-2 mb-1'):
                        ui.label('▌HILBERT').style(f'color:{THEME_WARN}; font-family: JetBrains Mono; font-size: 0.8rem; letter-spacing: 1px;')
                        S.hilbert_select_container = ui.row().classes('items-center gap-1')
                        refresh_hilbert_select()
                    S.hilbert_plot = ui.plotly(make_hilbert_fig()).classes('w-full')
            
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
    print("  - Viewer:  http://localhost:8080/")
    print("  - Pipeline: http://localhost:8080/pipeline")
    ui.run(title='EEG Viewer', port=8080, reload=False, show=False, dark=True)
