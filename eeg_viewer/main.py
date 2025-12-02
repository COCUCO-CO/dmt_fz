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
    THEME_WARN, THEME_TEXT, THEME_TEXT_DIM, SIGNAL_COLORS
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

# Main page
@ui.page('/')
def main():
    ui.add_head_html(f'<style>{STYLE}</style>')
    
    with ui.header().classes('items-center px-4 py-1').style(f'background: {THEME_BG}; border-bottom: 1px solid {THEME_BORDER};'):
        ui.label('▶').style(f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.75rem; letter-spacing: 2px;')
        ui.label('EEG_VIEWER').classes('text-base font-medium ml-2').style(f'color: {THEME_PRIMARY}; font-family: JetBrains Mono; letter-spacing: 1px;')
        ui.label('v1.0').classes('text-xs ml-2').style(f'color: {THEME_TEXT_DIM}; font-family: JetBrains Mono;')
    
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
                                with ui.expansion(f'📁 {cond} ({len(cfs)} files)').classes('w-full'):
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
    ui.run(title='EEG Viewer', port=8080, reload=False, show=False, dark=True)
