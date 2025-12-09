"""Main EEG Viewer page."""
from pathlib import Path
import asyncio
import numpy as np
from nicegui import ui
import plotly.graph_objects as go

from config import (
    EEG_RAW_DIR, EEG_CLEAN_DIR,
    THEME_BG, THEME_CARD, THEME_BORDER, THEME_PRIMARY, THEME_SECONDARY,
    THEME_WARN, THEME_TEXT, THEME_TEXT_DIM, SIGNAL_COLORS
)
from eeg_loader import load_eeg_file, get_channel_data, scan_eeg_directory
from app.state import S
from app.visualization.components.debug_console import render_debug_toggle, render_debug_console
from app.visualization.styles.css import STYLE
from app.visualization import make_eeg_fig, make_fft_fig, make_hilbert_fig, make_brain_fig
from app.core.signal import (
    apply_notch, apply_bandpass, 
    compute_fft, compute_hilbert,
    process_data as _process_data_core
)


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
        data = process_data(data, eeg_data.sfreq) * 1e6
        n = len(chs)
        
        # Normalize and compute amplitudes
        norm = np.zeros_like(data)
        for i in range(n):
            std = np.std(data[i])
            amp = np.sqrt(np.mean(data[i]**2))
            if amplitudes_dict is not None:
                amplitudes_dict[chs[i]] = amp
            norm[i] = data[i] / (std * 3) if std > 0 else data[i]
        
        spacing = 2.0 * S.scale_factor
        colors = _SECONDARY_COLORS if use_secondary else _PRIMARY_COLORS
        
        y_min = -spacing
        y_max = n * spacing
        
        with eeg_plot:
            eeg_plot.figure.data = []
            for i in range(n):
                off = (n - 1 - i) * spacing
                eeg_plot.figure.add_trace(go.Scatter(
                    x=times, y=norm[i] + off, name=chs[i],
                    line=dict(color=colors[i % len(colors)], width=1),
                    hovertemplate=f'{chs[i]}: %{{customdata:.1f}} µV<extra></extra>',
                    customdata=data[i]
                ))
            eeg_plot.figure.update_layout(
                yaxis=dict(
                    tickmode='array',
                    tickvals=[(n-1-i)*spacing for i in range(n)],
                    ticktext=chs,
                    range=[y_min, y_max],
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
        
        with fft_plot:
            fft_plot.figure.data = []
            for i, ch in enumerate(chs[:5]):
                fft_plot.figure.add_trace(go.Scatter(
                    x=freqs, y=fft_v[i], name=ch,
                    line=dict(color=colors[i % len(colors)], width=1.5),
                    fill='tozeroy', fillcolor=fills[i % len(fills)]
                ))
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
        
        with hilbert_plot:
            hilbert_plot.figure.data = []
            hilbert_plot.figure.add_trace(go.Scatter(x=times, y=data, name='Signal', line=dict(color=signal_color, width=1)), row=1, col=1)
            hilbert_plot.figure.add_trace(go.Scatter(x=times, y=amp, name='Envelope', line=dict(color=envelope_color, width=2)), row=1, col=1)
            hilbert_plot.figure.add_trace(go.Scatter(x=times, y=-amp, showlegend=False, line=dict(color=envelope_color, width=2)), row=1, col=1)
            hilbert_plot.figure.add_trace(go.Scatter(x=times, y=phase, name='Phase', line=dict(color=phase_color, width=1)), row=2, col=1)
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
    _update_eeg_generic(S.eeg_plot2, S.eeg_data2, S.selected_channels, use_secondary=True)

def update_fft2():
    """Update FFT 2 plot."""
    _update_fft_generic(S.fft_plot2, S.eeg_data2, S.selected_channels, use_secondary=True)

def update_hilbert2():
    """Update Hilbert 2 plot."""
    _update_hilbert_generic(S.hilbert_plot2, S.eeg_data2, S.selected_channels, S.hilbert_channel, use_secondary=True)

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

# PipelineState (PS) imported from app.state

PIPELINE_DIR = Path(__file__).parent.parent / "dashboard" / "pipeline_backend"
PIPELINE_OUTPUTS = Path(__file__).parent.parent / "pipeline_outputs"
RESULTS_BASE = Path(__file__).parent.parent / "fwd-inv-stc"
DEFAULT_INPUT_DIR = Path(__file__).parent.parent / "EEG_CLEAN"

# Update main page header to include navigation
@ui.page('/')
def main_with_nav():
    ui.add_head_html(f'<style>{STYLE}</style>')
    
    with ui.header().classes('items-center px-4 py-1').style(f'background: {THEME_BG}; border-bottom: 1px solid {THEME_BORDER};'):
        ui.label('▶').style(f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.75rem; letter-spacing: 2px;')
        ui.label('EEG_VIEWER').classes('text-base font-medium ml-2').style(f'color: {THEME_PRIMARY}; font-family: JetBrains Mono; letter-spacing: 1px;')
        ui.label('v1.0').classes('text-xs ml-2').style(f'color: {THEME_TEXT_DIM}; font-family: JetBrains Mono;')
        render_debug_toggle()
        with ui.row().classes('ml-auto gap-2'):
            ui.button('VIEWER', on_click=lambda: ui.navigate.to('/')).props('flat dense').style(f'color:{THEME_PRIMARY};')
            ui.button('CLEANER', on_click=lambda: ui.navigate.to('/cleaner')).props('flat dense').style(f'color:{THEME_TEXT_DIM};')
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
    
    render_debug_console()

