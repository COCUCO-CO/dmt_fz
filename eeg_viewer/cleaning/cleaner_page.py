"""
EEG Cleaner Page

Main UI page for the EEG cleaning pipeline.
Redesigned for better UX with clear visual hierarchy and workflow.

Features:
- Step-by-step pipeline with per-step reset
- Live preview (changes apply immediately)
- Tooltips with explanations (toggle-able)
- Full reset to original data
"""

import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any, Set
from nicegui import ui
import numpy as np
import mne

# Import cleaning modules
from .state import CleaningState, CleaningStep
from .filters import (
    FilterPreset, FilterParams, get_filter_presets,
    apply_filter_preset, apply_bandpass_filter, apply_notch_filter
)
from .bad_channels import (
    detect_bad_channels, interpolate_channels,
    compute_channel_quality_metrics, BadChannelResult
)
from .rereferencing import (
    ReferenceType, apply_reference, get_available_references,
    find_mastoid_channels, get_current_reference
)
from .ica import (
    compute_ica, detect_eog_components, detect_ecg_components,
    apply_ica_exclusion, ICAResult, get_component_timeseries,
    get_component_properties, detect_muscle_components
)
from .epochs import (
    create_epochs, detect_bad_epochs, apply_epoch_rejection,
    EpochRejectionCriteria, EpochResult, get_epoch_data,
    export_rejection_info, add_manual_rejection, get_rejection_stats
)
from .export import (
    ExportFormat, export_cleaned_eeg, export_epochs,
    export_preprocessing_log, create_export_bundle
)

# Import from parent
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    THEME_BG, THEME_CARD, THEME_BORDER, THEME_PRIMARY, THEME_SECONDARY,
    THEME_WARN, THEME_ERROR, THEME_TEXT, THEME_TEXT_DIM, SIGNAL_COLORS,
    EEG_RAW_DIR, EEG_CLEAN_DIR, SUPPORTED_FORMATS
)
from eeg_loader import load_eeg_file, scan_eeg_directory

import plotly.graph_objects as go


# ============ Tooltips / Help Text ============

HELP_TEXTS = {
    # Filter
    'highpass': 'High-pass filter cutoff (Hz). Removes slow drifts. Common: 0.1-1 Hz',
    'lowpass': 'Low-pass filter cutoff (Hz). Removes high-frequency noise. Common: 30-45 Hz',
    'notch': 'Notch filter frequency (Hz). Removes power line noise. Use 50 Hz (EU) or 60 Hz (US)',
    
    # Bad channels
    'std_threshold': 'Z-score threshold for noisy channels. Higher = less sensitive. Typical: 3-5',
    'flat_threshold': 'Variance threshold for flat/dead channels. Typical: 1e-12 (V²)',
    'corr_threshold': 'Minimum correlation with neighbors. Lower = less sensitive. Typical: 0.1-0.3',
    'check_correlation': 'Enable correlation check (can cause false positives in some datasets)',
    
    # Epochs
    'epoch_duration': 'Length of each epoch in seconds. Common: 1-4 seconds',
    'epoch_overlap': 'Overlap between consecutive epochs in seconds. 0 = no overlap',
    
    # Rejection
    'peak_to_peak': 'Maximum allowed amplitude range (µV). Epochs exceeding this are rejected. Typical: 100-200 µV',
    'flat_epoch': 'Minimum required activity (µV). Epochs below this are considered flat. Typical: 0.5-1 µV',
    
    # ICA
    'n_components': 'Number of ICA components to compute. Typically 15-25 or n_channels - n_bads',
    'ica_method': 'ICA algorithm. FastICA is fastest, Infomax and Picard are more robust',
}


# ============ Page State ============

class CleanerPageState:
    """State for the cleaner page."""
    def __init__(self):
        self.cleaning = CleaningState()
        
        # UI references
        self.main_plot = None
        self.step_container = None
        self.control_container = None
        self.info_container = None
        self.history_container = None
        
        # View settings
        self.view_start = 0.0
        self.view_duration = 5.0
        self.current_epoch_index = 0  # For VIEW step navigation
        
        # Playback settings
        self.is_playing = False
        self.playback_speed = 1.0  # 1x, 2x, 0.5x etc
        self._playback_timer = None
        self._play_button = None  # Reference to play button for icon updates
        
        # Filter state
        self.selected_filter_preset: Optional[FilterPreset] = None
        
        # Bad channel detection state (persist across re-renders)
        self.bad_ch_std_threshold: float = 3.5
        self.bad_ch_flat_threshold: float = 1e-12
        self.bad_ch_check_correlation: bool = False
        self.bad_ch_corr_threshold: float = 0.1
        
        # Epoch creation state (persist across re-renders)
        self.epoch_duration: float = 2.0
        self.epoch_overlap: float = 0.0
        
        # Epoch rejection state (persist across re-renders)
        self.reject_ptp_threshold: float = 150.0
        self.reject_flat_threshold: float = 0.5
        
        # ICA parameters state (persist across re-renders)
        self.ica_n_components: int = 20
        self.ica_method: str = 'fastica'
        
        # ICA results
        self.ica_result: Optional[ICAResult] = None
        self._ica_computing = False
        self._ica_applied = False  # Track if ICA was confirmed/applied
        
        # Epoch results  
        self.epoch_result: Optional[EpochResult] = None
        
        # UI Settings
        self.show_tooltips = True  # Toggle for help tooltips
        self.show_bad_channels_highlight = True  # Show bad channels in red
        self.show_rejected_epochs_overlay = True  # Show rejected epoch regions (default on)
        
        # Reverse state tracking - which steps have reversed data
        self.reversed_steps: Set[CleaningStep] = set()
        
        # Per-step EEG snapshots - stores raw data state for each completed step
        self._step_raw_snapshots: Dict[CleaningStep, Any] = {}
        
        # State snapshots for per-step reset
        self._step_snapshots: Dict[CleaningStep, Any] = {}
        
        # Debounce timer for live updates
        self._update_timer = None
    
    def save_step_raw(self, step: CleaningStep):
        """Save the current raw EEG state for a step."""
        if self.cleaning.raw is not None:
            self._step_raw_snapshots[step] = self.cleaning.raw.copy()
    
    def get_step_raw(self, step: CleaningStep):
        """Get the raw EEG state for a specific step."""
        return self._step_raw_snapshots.get(step)


# Global page state
PS = CleanerPageState()


# ============ Styles ============

CLEANER_STYLES = f'''
<style>
    .cleaner-card {{
        background: {THEME_CARD} !important;
        border: 1px solid {THEME_BORDER} !important;
        border-radius: 6px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
    }}
    
    .step-item {{
        padding: 10px 12px;
        border-radius: 4px;
        cursor: pointer;
        transition: all 0.2s ease;
        border-left: 3px solid transparent;
        margin-bottom: 4px;
    }}
    .step-item:hover {{
        background: rgba(0, 255, 136, 0.05);
    }}
    .step-active {{
        background: rgba(0, 255, 136, 0.12) !important;
        border-left-color: {THEME_PRIMARY} !important;
    }}
    .step-completed {{
        border-left-color: {THEME_PRIMARY} !important;
    }}
    .step-pending {{
        opacity: 0.5;
    }}
    
    .control-section {{
        background: rgba(0,0,0,0.2);
        border-radius: 6px;
        padding: 16px;
        margin-bottom: 12px;
    }}
    
    .section-title {{
        font-size: 0.7rem;
        color: {THEME_TEXT_DIM};
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 1px solid {THEME_BORDER};
    }}
    
    .channel-btn {{
        font-size: 0.7rem !important;
        font-family: 'JetBrains Mono', monospace !important;
        padding: 4px 8px !important;
        min-width: 48px !important;
        margin: 2px !important;
        border-radius: 3px !important;
        transition: all 0.15s ease !important;
    }}
    .channel-btn-normal {{
        background: transparent !important;
        border: 1px solid {THEME_BORDER} !important;
        color: {THEME_TEXT_DIM} !important;
    }}
    .channel-btn-normal:hover {{
        border-color: {THEME_PRIMARY} !important;
        color: {THEME_PRIMARY} !important;
    }}
    .channel-btn-bad {{
        background: rgba(255, 85, 85, 0.2) !important;
        border: 1px solid {THEME_ERROR} !important;
        color: {THEME_ERROR} !important;
    }}
    
    .ica-card {{
        background: {THEME_CARD};
        border: 1px solid {THEME_BORDER};
        border-radius: 4px;
        padding: 8px;
        min-width: 80px;
        cursor: pointer;
        transition: all 0.15s ease;
        text-align: center;
    }}
    .ica-card:hover {{
        border-color: {THEME_SECONDARY};
    }}
    .ica-card-excluded {{
        background: rgba(255, 85, 85, 0.15) !important;
        border-color: {THEME_ERROR} !important;
    }}
    
    .stat-card {{
        background: rgba(0,0,0,0.3);
        border-radius: 6px;
        padding: 12px 16px;
        text-align: center;
        min-width: 80px;
    }}
    .stat-value {{
        font-size: 1.5rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }}
    .stat-label {{
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }}
    
    .action-bar {{
        display: flex;
        gap: 8px;
        padding-top: 16px;
        margin-top: 16px;
        border-top: 1px solid {THEME_BORDER};
    }}
    
    .file-row {{
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 12px;
        border-radius: 4px;
        cursor: pointer;
        transition: all 0.15s ease;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }}
    .file-row:hover {{
        background: rgba(0, 255, 136, 0.08);
    }}
    
    .nav-btn {{
        min-width: 36px !important;
        height: 36px !important;
    }}
    
    .help-toggle {{
        opacity: 0.6;
        transition: opacity 0.2s;
    }}
    .help-toggle:hover {{
        opacity: 1;
    }}
    
    .reset-btn {{
        color: {THEME_ERROR} !important;
        border-color: {THEME_ERROR} !important;
    }}
    .reset-btn:hover {{
        background: rgba(255, 85, 85, 0.1) !important;
    }}
</style>
'''


# ============ Helper Functions ============

def safe_notify(message: str, type: str = 'info'):
    """Safely show notification, catching context errors."""
    try:
        safe_notify(message, type=type)
    except RuntimeError:
        # Context was deleted (page navigated away)
        print(f"[Notify] {type}: {message}")


def tooltip(text: str) -> str:
    """Return tooltip text if tooltips are enabled."""
    return text if PS.show_tooltips else ''


def create_input_with_help(label: str, help_key: str, **kwargs):
    """Create an input field with optional tooltip."""
    inp = ui.number(label, **kwargs).props('dense outlined')
    if PS.show_tooltips and help_key in HELP_TEXTS:
        inp.tooltip(HELP_TEXTS[help_key])
    return inp


def save_step_snapshot(step: CleaningStep):
    """Save current state snapshot for a step."""
    if PS.cleaning.raw is not None:
        PS._step_snapshots[step] = PS.cleaning.raw.copy()


def restore_step_snapshot(step: CleaningStep) -> bool:
    """Restore state from step snapshot. Returns True if successful."""
    if step in PS._step_snapshots:
        PS.cleaning._raw = PS._step_snapshots[step].copy()
        # Remove this and later snapshots
        steps = list(CleaningStep)
        step_idx = steps.index(step)
        for s in steps[step_idx:]:
            PS._step_snapshots.pop(s, None)
            PS.cleaning.completed_steps.discard(s)
        return True
    return False


def debounced_update(delay: float = 0.5):
    """Decorator for debounced updates."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            if PS._update_timer:
                PS._update_timer.cancel()
            PS._update_timer = asyncio.get_event_loop().call_later(
                delay,
                lambda: asyncio.create_task(func(*args, **kwargs))
            )
        return wrapper
    return decorator


# ============ Main Page ============

def cleaner_page():
    """Main cleaner page."""
    ui.add_head_html(CLEANER_STYLES)
    
    # Header
    with ui.header().classes('items-center px-4 py-2').style(
        f'background: {THEME_BG}; border-bottom: 1px solid {THEME_BORDER};'
    ):
        ui.icon('cleaning_services', size='sm').style(f'color: {THEME_PRIMARY};')
        ui.label('EEG CLEANER').classes('text-lg font-bold ml-2').style(
            f'color: {THEME_PRIMARY}; font-family: JetBrains Mono;'
        )
        ui.label('v1.1').classes('text-xs ml-2 opacity-50')
        
        ui.element('div').classes('flex-1')
        
        # Help toggle
        help_switch = ui.switch('Help', value=PS.show_tooltips).props('dense').classes('help-toggle')
        help_switch.on('update:model-value', lambda e: toggle_help(e.args))
        help_switch.tooltip('Show/hide parameter explanations')
        
        ui.separator().props('vertical').classes('mx-3')
        
        # Header actions
        with ui.row().classes('gap-2 items-center'):
            ui.button(icon='undo', on_click=do_undo).props('flat dense round').style(
                f'color: {THEME_WARN};'
            ).tooltip('Undo last operation')
            
            ui.button(icon='restart_alt', on_click=do_full_reset).props('flat dense round').style(
                f'color: {THEME_ERROR};'
            ).tooltip('Reset to original EEG')
            
            ui.separator().props('vertical').classes('mx-2')
            
            ui.button('VIEWER', icon='visibility', on_click=lambda: ui.navigate.to('/')).props('flat dense')
            ui.button('PIPELINE', icon='account_tree', on_click=lambda: ui.navigate.to('/pipeline')).props('flat dense')
    
    # Main layout
    with ui.row().classes('w-full gap-0').style(
        f'height: calc(100vh - 56px); background: {THEME_BG};'
    ):
        # Left sidebar
        with ui.column().classes('p-4').style(
            f'width: 200px; background: rgba(0,0,0,0.2); border-right: 1px solid {THEME_BORDER};'
        ):
            ui.label('PIPELINE').classes('section-title')
            PS.step_container = ui.column().classes('gap-1 w-full')
            render_step_list()
            
            ui.element('div').classes('flex-1')
            
            ui.label('HISTORY').classes('section-title mt-4')
            PS.history_container = ui.scroll_area().classes('w-full').style('max-height: 200px;')
            render_history()
        
        # Center
        with ui.column().classes('flex-1 p-4 gap-4').style('min-width: 0; overflow-y: auto;'):
            # Signal Preview
            with ui.card().classes('cleaner-card p-4 w-full'):
                with ui.row().classes('items-center justify-between mb-3'):
                    with ui.row().classes('items-center gap-2'):
                        ui.icon('show_chart', size='xs').style(f'color: {THEME_SECONDARY};')
                        ui.label('SIGNAL PREVIEW').classes('text-xs font-medium').style(
                            f'color: {THEME_SECONDARY}; letter-spacing: 1px;'
                        )
                    
                    # Playback controls
                    with ui.row().classes('gap-1 items-center'):
                        # Go to start
                        ui.button(icon='first_page', on_click=nav_to_start).props(
                            'flat dense round size=sm'
                        ).tooltip('Go to start')
                        
                        # Play reverse button
                        ui.button(icon='fast_rewind', on_click=toggle_playback_reverse).props(
                            'flat dense round size=sm'
                        ).tooltip('Play reverse')
                        
                        # Navigation back
                        ui.button(icon='chevron_left', on_click=nav_back).props(
                            'flat dense round size=sm'
                        ).classes('nav-btn').tooltip('Previous segment')
                        
                        # Play/Pause button
                        play_icon = 'pause' if PS.is_playing else 'play_arrow'
                        PS._play_button = ui.button(
                            icon=play_icon,
                            on_click=toggle_playback
                        ).props('flat dense round size=sm').tooltip('Play/Pause')
                        
                        # Navigation forward
                        ui.button(icon='chevron_right', on_click=nav_forward).props(
                            'flat dense round size=sm'
                        ).classes('nav-btn').tooltip('Next segment')
                        
                        # Play forward fast
                        ui.button(icon='fast_forward', on_click=toggle_playback_fast).props(
                            'flat dense round size=sm'
                        ).tooltip('Play fast forward')
                        
                        # Go to end
                        ui.button(icon='last_page', on_click=nav_to_end).props(
                            'flat dense round size=sm'
                        ).tooltip('Go to end')
                        
                        ui.separator().props('vertical').classes('mx-1')
                        
                        # Speed selector
                        speed_select = ui.select(
                            {0.5: '0.5x', 1.0: '1x', 2.0: '2x', 4.0: '4x'},
                            value=PS.playback_speed,
                            label=None
                        ).props('dense borderless').classes('w-16').style('font-size: 0.7rem;')
                        speed_select.on('update:model-value', lambda e: set_playback_speed(e.args))
                        
                        ui.separator().props('vertical').classes('mx-1')
                        
                        # Reverse button
                        is_reversed = PS.cleaning.current_step in PS.reversed_steps
                        reverse_btn = ui.button(
                            icon='swap_horiz',
                            on_click=toggle_reverse_eeg
                        ).props(f'flat dense round size=sm {"color=warning" if is_reversed else ""}')
                        reverse_btn.tooltip('Reverse EEG temporally (flip time axis)')
                        
                        # Show reverse indicator if any step is reversed
                        if PS.reversed_steps:
                            steps_str = ', '.join([s.short_name for s in PS.reversed_steps])
                            ui.chip(f'⟲ {steps_str}', color='warning').props('dense').classes('ml-2').style('font-size: 0.65rem;')
                
                PS.main_plot = ui.plotly({}).classes('w-full').style('height: 420px;')
                update_main_plot()
            
            # Step Controls
            with ui.card().classes('cleaner-card p-4 w-full flex-1').style('min-height: 280px;'):
                PS.control_container = ui.column().classes('w-full')
                render_step_controls()
        
        # Right sidebar
        with ui.column().classes('p-4').style(
            f'width: 260px; background: rgba(0,0,0,0.2); border-left: 1px solid {THEME_BORDER};'
        ):
            ui.label('FILE INFO').classes('section-title')
            PS.info_container = ui.column().classes('gap-3 w-full')
            render_info()
            
            ui.separator().classes('my-3')
            ui.label('DISPLAY OPTIONS').classes('section-title')
            
            # Switches for visualization options - use on_change for reliable updates
            def on_bad_ch_switch_change(e):
                PS.show_bad_channels_highlight = e.value
                update_main_plot()
            
            bad_ch_switch = ui.switch('Highlight bad channels', value=PS.show_bad_channels_highlight, on_change=on_bad_ch_switch_change).props('dense')
            bad_ch_switch.tooltip('Show bad channels in red on the plot')
            
            def on_rej_epoch_switch_change(e):
                PS.show_rejected_epochs_overlay = e.value
                update_main_plot()
            
            rej_epoch_switch = ui.switch('Show rejected epochs', value=PS.show_rejected_epochs_overlay, on_change=on_rej_epoch_switch_change).props('dense')
            rej_epoch_switch.tooltip('Highlight rejected epoch regions on the plot')


# ============ Render Functions ============

def render_step_list():
    """Render the pipeline step list."""
    if PS.step_container is None:
        return
    
    PS.step_container.clear()
    
    with PS.step_container:
        for step in CleaningStep:
            is_current = step == PS.cleaning.current_step
            is_completed = step in PS.cleaning.completed_steps
            
            if is_current:
                css_class = 'step-item step-active'
                icon = 'radio_button_checked'
                color = THEME_PRIMARY
            elif is_completed:
                css_class = 'step-item step-completed'
                icon = 'check_circle'
                color = THEME_PRIMARY
            else:
                css_class = 'step-item step-pending'
                icon = 'radio_button_unchecked'
                color = THEME_TEXT_DIM
            
            with ui.row().classes(css_class).on('click', lambda s=step: go_to_step(s)):
                ui.icon(icon, size='xs').style(f'color: {color};')
                ui.label(step.short_name).style(
                    f'color: {color}; font-family: JetBrains Mono; font-size: 0.75rem; margin-left: 8px;'
                )


def render_history():
    """Render operation history."""
    if PS.history_container is None:
        return
    
    PS.history_container.clear()
    
    with PS.history_container:
        if not PS.cleaning.operations:
            ui.label('No operations yet').style(
                f'color: {THEME_TEXT_DIM}; font-size: 0.7rem; font-style: italic;'
            )
        else:
            for op in reversed(PS.cleaning.operations[-10:]):
                with ui.row().classes('items-center gap-2 py-1'):
                    ui.icon('history', size='xs').style(f'color: {THEME_TEXT_DIM};')
                    ui.label(op.description[:22] + ('...' if len(op.description) > 22 else '')).style(
                        f'color: {THEME_TEXT}; font-size: 0.65rem; font-family: JetBrains Mono;'
                    )


def render_info():
    """Render file info panel."""
    if PS.info_container is None:
        return
    
    PS.info_container.clear()
    
    with PS.info_container:
        if not PS.cleaning.is_loaded:
            with ui.column().classes('items-center justify-center py-8 opacity-50'):
                ui.icon('upload_file', size='xl')
                ui.label('No file loaded').classes('text-sm mt-2')
        else:
            info_item('insert_drive_file', 'File', 
                     PS.cleaning.filename[:18] + '...' if len(PS.cleaning.filename) > 18 else PS.cleaning.filename)
            info_item('speed', 'Sample Rate', f'{PS.cleaning.sfreq:.0f} Hz')
            info_item('sensors', 'Channels', str(PS.cleaning.n_channels))
            info_item('timer', 'Duration', f'{PS.cleaning.duration:.1f} s')
            
            if PS.cleaning.bad_channels or PS.cleaning.reference_type or PS.cleaning.ica_excluded:
                ui.separator().classes('my-3')
                ui.label('PROCESSING').classes('section-title')
                
                if PS.cleaning.bad_channels:
                    info_item('warning', 'Bad Ch', ', '.join(PS.cleaning.bad_channels[:3]) + 
                             (f' +{len(PS.cleaning.bad_channels)-3}' if len(PS.cleaning.bad_channels) > 3 else ''),
                             color=THEME_WARN)
                if PS.cleaning.interpolated_channels:
                    info_item('auto_fix_high', 'Interpolated', str(len(PS.cleaning.interpolated_channels)))
                if PS.cleaning.reference_type:
                    info_item('compare_arrows', 'Reference', PS.cleaning.reference_type[:15])
                if PS.cleaning.ica_excluded:
                    info_item('blur_on', 'ICA Excluded', str(len(PS.cleaning.ica_excluded)))
                if PS.epoch_result and PS.epoch_result.n_total > 0:
                    info_item('view_module', 'Epochs', 
                             f'{PS.epoch_result.n_good}/{PS.epoch_result.n_total}')
            
            # Show reversed steps indicator
            if PS.reversed_steps:
                ui.separator().classes('my-3')
                ui.label('REVERSED').classes('section-title')
                steps_str = ', '.join(sorted([s.short_name for s in PS.reversed_steps]))
                with ui.row().classes('items-center gap-2'):
                    ui.icon('swap_horiz', size='xs').style(f'color: {THEME_WARN};')
                    ui.label(steps_str).style(f'color: {THEME_WARN}; font-size: 0.7rem; font-family: JetBrains Mono;')


def info_item(icon: str, label: str, value: str, color: str = None):
    """Create an info item row."""
    with ui.row().classes('items-center w-full gap-2'):
        ui.icon(icon, size='xs').style(f'color: {color or THEME_TEXT_DIM};')
        ui.label(f'{label}:').style(f'color: {THEME_TEXT_DIM}; font-size: 0.7rem; min-width: 70px;')
        ui.label(value).style(f'color: {color or THEME_TEXT}; font-size: 0.7rem; font-family: JetBrains Mono;')


def format_indices_with_ranges(indices: List[int]) -> str:
    """Format a list of indices, grouping consecutive numbers into ranges.
    
    Example: [0, 1, 2, 5, 10, 11, 12] -> '0-2, 5, 10-12'
    """
    if not indices:
        return ''
    
    indices = sorted(set(indices))
    ranges = []
    start = indices[0]
    end = indices[0]
    
    for i in range(1, len(indices)):
        if indices[i] == end + 1:
            end = indices[i]
        else:
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f'{start}-{end}')
            start = indices[i]
            end = indices[i]
    
    # Add last range
    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f'{start}-{end}')
    
    return ', '.join(ranges)


def render_step_controls():
    """Render controls for current step."""
    if PS.control_container is None:
        return
    
    PS.control_container.clear()
    step = PS.cleaning.current_step
    
    with PS.control_container:
        # Step header with reset button
        step_num = list(CleaningStep).index(step) + 1
        with ui.row().classes('items-center justify-between w-full mb-4'):
            with ui.row().classes('items-center gap-3'):
                ui.label(f'{step_num}').style(
                    f'background: {THEME_PRIMARY}; color: {THEME_BG}; '
                    f'width: 28px; height: 28px; border-radius: 50%; '
                    f'display: flex; align-items: center; justify-content: center; '
                    f'font-weight: 600; font-size: 0.85rem;'
                )
                ui.label(step.display_name.split('. ')[1] if '. ' in step.display_name else step.display_name).style(
                    f'color: {THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 1rem; font-weight: 500;'
                )
            
            # Reset this step button
            if step != CleaningStep.LOAD and step in PS.cleaning.completed_steps:
                ui.button('Reset Step', icon='replay', on_click=lambda s=step: reset_step(s)).props(
                    'flat dense size=sm'
                ).classes('reset-btn').tooltip(f'Reset {step.short_name} and redo')
        
        # Render step-specific controls
        if step == CleaningStep.LOAD:
            render_load_controls()
        elif step == CleaningStep.FILTER:
            render_filter_controls()
        elif step == CleaningStep.BAD_CHANNELS:
            render_bad_channels_controls()
        elif step == CleaningStep.REREFERENCE:
            render_rereference_controls()
        elif step == CleaningStep.ICA:
            render_ica_controls()
        elif step == CleaningStep.EPOCHS:
            render_epochs_controls()
        elif step == CleaningStep.REJECT:
            render_reject_controls()
        elif step == CleaningStep.VISUALIZE:
            render_visualize_controls()
        elif step == CleaningStep.EXPORT:
            render_export_controls()


# ============ Step Control Renderers ============

def render_load_controls():
    """Render load step controls."""
    ui.label('Select an EEG file to begin the cleaning process.').style(
        f'color: {THEME_TEXT_DIM}; font-size: 0.8rem; margin-bottom: 16px;'
    )
    
    files = scan_eeg_directory(EEG_RAW_DIR)
    
    if files:
        with ui.scroll_area().classes('w-full').style(
            f'max-height: 220px; border: 1px solid {THEME_BORDER}; border-radius: 4px;'
        ):
            for f in files:
                with ui.row().classes('file-row').on('click', lambda path=f['path']: load_file(path)):
                    ui.icon('description', size='sm').style(f'color: {THEME_SECONDARY};')
                    with ui.column().classes('gap-0 flex-1'):
                        ui.label(f['name']).style(f'color: {THEME_TEXT}; font-size: 0.8rem; font-weight: 500;')
                        ui.label(f"{f['condition']} • {f['size_mb']:.1f} MB").style(
                            f'color: {THEME_TEXT_DIM}; font-size: 0.65rem;'
                        )
    else:
        ui.label(f'No EEG files found in {EEG_RAW_DIR}').style(f'color: {THEME_TEXT_DIM};')
    
    with ui.expansion('Or enter file path manually', icon='edit').classes('w-full mt-4'):
        with ui.row().classes('items-end gap-2 w-full'):
            file_input = ui.input('File path').props('dense outlined').classes('flex-1')
            ui.button('Load', icon='upload', on_click=lambda: load_file(file_input.value)).props('dense')
    
    # Action bar - only show Next Step if file is loaded
    if PS.cleaning.is_loaded:
        with ui.element('div').classes('action-bar'):
            ui.element('div').classes('flex-1')
            ui.button('Next Step →', icon='arrow_forward', on_click=lambda: complete_step(CleaningStep.LOAD)).props('dense')


def render_filter_controls():
    """Render filter step controls with LIVE updates."""
    if not PS.cleaning.is_loaded:
        show_not_loaded_message()
        return
    
    # Save snapshot before filtering
    if CleaningStep.FILTER not in PS._step_snapshots:
        save_step_snapshot(CleaningStep.FILTER)
    
    ui.label('Filters apply automatically as you adjust values.').style(
        f'color: {THEME_TEXT_DIM}; font-size: 0.8rem; margin-bottom: 16px;'
    )
    
    # Presets
    with ui.element('div').classes('control-section'):
        ui.label('Quick Presets').style(f'color: {THEME_TEXT}; font-size: 0.8rem; margin-bottom: 12px;')
        with ui.row().classes('gap-2 flex-wrap'):
            for preset in [FilterPreset.STANDARD, FilterPreset.ALPHA_THETA, FilterPreset.ERP, FilterPreset.GAMMA]:
                is_selected = PS.selected_filter_preset == preset
                btn = ui.button(
                    preset.display_name.split('(')[0].strip(),
                    on_click=lambda p=preset: apply_preset(p)
                ).props(f'{"" if is_selected else "outline"} dense')
                if is_selected:
                    btn.style(f'background: {THEME_PRIMARY} !important; color: {THEME_BG} !important;')
                btn.tooltip(preset.display_name)
    
    # Custom filter with LIVE updates
    with ui.element('div').classes('control-section'):
        ui.label('Custom Filter (live)').style(f'color: {THEME_TEXT}; font-size: 0.8rem; margin-bottom: 12px;')
        
        with ui.row().classes('gap-4 items-end flex-wrap'):
            hp_input = ui.number('Highpass', value=0.1, min=0, max=50, step=0.1, suffix='Hz').props('dense outlined').classes('w-32')
            if PS.show_tooltips:
                hp_input.tooltip(HELP_TEXTS['highpass'])
            
            lp_input = ui.number('Lowpass', value=45, min=1, max=200, step=1, suffix='Hz').props('dense outlined').classes('w-32')
            if PS.show_tooltips:
                lp_input.tooltip(HELP_TEXTS['lowpass'])
            
            notch_input = ui.number('Notch', value=50, min=45, max=65, step=5, suffix='Hz').props('dense outlined').classes('w-32')
            if PS.show_tooltips:
                notch_input.tooltip(HELP_TEXTS['notch'])
        
        # Live update on change
        def on_filter_change():
            apply_custom_filter_live(hp_input.value, lp_input.value, notch_input.value)
        
        hp_input.on('update:model-value', lambda: on_filter_change())
        lp_input.on('update:model-value', lambda: on_filter_change())
        notch_input.on('update:model-value', lambda: on_filter_change())
    
    # Action bar
    with ui.element('div').classes('action-bar'):
        ui.button('Reset Filter', icon='replay', on_click=lambda: reset_step(CleaningStep.FILTER)).props(
            'flat dense'
        ).classes('reset-btn')
        ui.element('div').classes('flex-1')
        ui.button('← Previous', icon='arrow_back', on_click=go_to_previous_step).props('flat dense')
        ui.button('Next Step →', icon='arrow_forward', on_click=lambda: complete_step(CleaningStep.FILTER)).props('dense')


def render_bad_channels_controls():
    """Render bad channels step controls with all parameters."""
    if not PS.cleaning.is_loaded:
        show_not_loaded_message()
        return
    
    if CleaningStep.BAD_CHANNELS not in PS._step_snapshots:
        save_step_snapshot(CleaningStep.BAD_CHANNELS)
    
    ui.label('Identify and handle bad channels.').style(
        f'color: {THEME_TEXT_DIM}; font-size: 0.8rem; margin-bottom: 16px;'
    )
    
    # Detection parameters
    with ui.element('div').classes('control-section'):
        ui.label('Detection Parameters').style(f'color: {THEME_TEXT}; font-size: 0.8rem; margin-bottom: 12px;')
        
        with ui.row().classes('gap-4 items-end flex-wrap'):
            def on_std_thresh_change(e):
                PS.bad_ch_std_threshold = float(e.value) if e.value else 3.5
            
            def on_flat_thresh_change(e):
                PS.bad_ch_flat_threshold = float(e.value) if e.value else 1e-12
            
            std_thresh = ui.number('Noise threshold', value=PS.bad_ch_std_threshold, min=1, max=10, step=0.5, suffix='σ',
                                  on_change=on_std_thresh_change).props('dense outlined').classes('w-36')
            if PS.show_tooltips:
                std_thresh.tooltip(HELP_TEXTS['std_threshold'])
            
            flat_thresh = ui.number('Flat threshold', value=PS.bad_ch_flat_threshold, min=1e-15, max=1e-8, step=1e-13, format='%.1e',
                                   on_change=on_flat_thresh_change).props('dense outlined').classes('w-36')
            if PS.show_tooltips:
                flat_thresh.tooltip(HELP_TEXTS['flat_threshold'])
        
        with ui.row().classes('gap-4 items-center mt-3'):
            def on_check_corr_change(e):
                PS.bad_ch_check_correlation = e.value
            
            def on_corr_thresh_change(e):
                PS.bad_ch_corr_threshold = float(e.value) if e.value else 0.1
            
            check_corr = ui.checkbox('Check correlation', value=PS.bad_ch_check_correlation, on_change=on_check_corr_change).props('dense')
            if PS.show_tooltips:
                check_corr.tooltip(HELP_TEXTS['check_correlation'])
            
            corr_thresh = ui.number('Corr threshold', value=PS.bad_ch_corr_threshold, min=0, max=1, step=0.05,
                                   on_change=on_corr_thresh_change).props('dense outlined').classes('w-28')
            corr_thresh.bind_visibility_from(check_corr, 'value')
            if PS.show_tooltips:
                corr_thresh.tooltip(HELP_TEXTS['corr_threshold'])
        
        ui.button('Detect Bad Channels', icon='search', on_click=lambda: auto_detect_bad(
            std_thresh.value or 3.5, 
            flat_thresh.value or 1e-12,
            corr_thresh.value or 0.1,
            check_corr.value
        )).props('dense').classes('mt-3')
    
    # Results
    if PS.cleaning.bad_channels:
        with ui.row().classes('items-center gap-2 my-3'):
            ui.icon('warning', size='sm').style(f'color: {THEME_WARN};')
            ui.label(f'{len(PS.cleaning.bad_channels)} bad channel(s): {", ".join(PS.cleaning.bad_channels)}').style(
                f'color: {THEME_WARN}; font-size: 0.8rem;'
            )
    
    # Channel grid
    with ui.element('div').classes('control-section'):
        ui.label('Click channels to mark/unmark as bad:').style(
            f'color: {THEME_TEXT_DIM}; font-size: 0.75rem; margin-bottom: 12px;'
        )
        
        with ui.row().classes('gap-1 flex-wrap'):
            for ch in PS.cleaning.ch_names:
                is_bad = ch in PS.cleaning.bad_channels
                css = 'channel-btn channel-btn-bad' if is_bad else 'channel-btn channel-btn-normal'
                ui.button(ch, on_click=lambda c=ch: toggle_bad_channel(c)).props('flat dense').classes(css)
    
    # Action bar
    with ui.element('div').classes('action-bar'):
        if PS.cleaning.bad_channels:
            ui.button('Interpolate Bad', icon='auto_fix_high', on_click=do_interpolate).props('dense outlined')
        ui.button('Reset Step', icon='replay', on_click=lambda: reset_step(CleaningStep.BAD_CHANNELS)).props(
            'flat dense'
        ).classes('reset-btn')
        ui.element('div').classes('flex-1')
        ui.button('← Previous', icon='arrow_back', on_click=go_to_previous_step).props('flat dense')
        ui.button('Next Step →', icon='arrow_forward', on_click=lambda: complete_step(CleaningStep.BAD_CHANNELS)).props('dense')


def render_rereference_controls():
    """Render re-reference step controls with explanations."""
    if not PS.cleaning.is_loaded:
        show_not_loaded_message()
        return
    
    if CleaningStep.REREFERENCE not in PS._step_snapshots:
        save_step_snapshot(CleaningStep.REREFERENCE)
    
    current_ref = get_current_reference(PS.cleaning.raw) if PS.cleaning.raw else "Unknown"
    
    # Explanation of re-referencing
    ui.label('Re-referencing changes the electrical reference point of your EEG data.').style(
        f'color: {THEME_TEXT_DIM}; font-size: 0.8rem; margin-bottom: 8px;'
    )
    ui.label(f'Current reference: {current_ref}').style(
        f'color: {THEME_SECONDARY}; font-size: 0.85rem; font-weight: 500; margin-bottom: 16px;'
    )
    
    # Reference type explanations
    REF_EXPLANATIONS = {
        ReferenceType.AVERAGE: 'Uses the average of all electrodes. Most common choice, reduces noise.',
        ReferenceType.LINKED_MASTOIDS: 'Uses average of mastoid electrodes (M1, M2, TP9, TP10). Good for auditory research.',
        ReferenceType.SINGLE_ELECTRODE: 'Uses a single electrode. Select which one below.',
        ReferenceType.REST: 'Reference Electrode Standardization Technique. Requires forward model.',
    }
    
    with ui.element('div').classes('control-section'):
        available = get_available_references(PS.cleaning.ch_names)
        
        ref_type = ui.radio(
            {rt.value: rt.display_name for rt in available},
            value=ReferenceType.AVERAGE.value
        ).props('dense')
        
        # Show explanation for selected reference
        ref_explanation = ui.label(REF_EXPLANATIONS.get(ReferenceType.AVERAGE, '')).style(
            f'color: {THEME_TEXT_DIM}; font-size: 0.7rem; font-style: italic; margin-top: 8px; margin-bottom: 12px;'
        )
        
        # Single electrode selector (shown only when SINGLE_ELECTRODE selected)
        single_ch = None
        with ui.row().classes('mt-3 items-end gap-3').bind_visibility_from(
            ref_type, 'value', lambda v: v == ReferenceType.SINGLE_ELECTRODE.value
        ):
            single_ch = ui.select(
                PS.cleaning.ch_names,
                label='Reference electrode',
                value=PS.cleaning.ch_names[0] if PS.cleaning.ch_names else None
            ).props('dense outlined').classes('w-48')
            single_ch.tooltip('Select the electrode to use as reference')
        
        # Update explanation on selection change
        def on_ref_change(e):
            try:
                val = e.args if isinstance(e.args, str) else str(e.args)
                new_ref = ReferenceType(val)
                ref_explanation.text = REF_EXPLANATIONS.get(new_ref, '')
            except (ValueError, KeyError):
                pass  # Ignore invalid values
        
        ref_type.on('update:model-value', on_ref_change)
        
        # Apply button always visible
        def do_apply_ref():
            ref = ReferenceType(ref_type.value)
            channels = [single_ch.value] if ref == ReferenceType.SINGLE_ELECTRODE and single_ch and single_ch.value else None
            apply_ref(ref, channels)
        
        ui.button('Apply Reference', icon='check', on_click=do_apply_ref).props('dense').classes('mt-3').tooltip(
            'Apply the selected reference to the EEG data'
        )
    
    # Action bar
    with ui.element('div').classes('action-bar'):
        ui.button('Reset Step', icon='replay', on_click=lambda: reset_step(CleaningStep.REREFERENCE)).props(
            'flat dense'
        ).classes('reset-btn')
        ui.element('div').classes('flex-1')
        ui.button('← Previous', icon='arrow_back', on_click=go_to_previous_step).props('flat dense')
        ui.button('Next Step →', icon='arrow_forward', on_click=lambda: complete_step(CleaningStep.REREFERENCE)).props('dense')


def render_ica_controls():
    """Render ICA step controls with component statistics."""
    if not PS.cleaning.is_loaded:
        show_not_loaded_message()
        return
    
    if CleaningStep.ICA not in PS._step_snapshots:
        save_step_snapshot(CleaningStep.ICA)
    
    if PS.ica_result is None or PS.ica_result.ica is None:
        ui.label('ICA separates neural signals from artifacts like eye blinks and heartbeat.').style(
            f'color: {THEME_TEXT_DIM}; font-size: 0.8rem; margin-bottom: 16px;'
        )
        
        # Calculate max components
        n_eeg = len(mne.pick_types(PS.cleaning.raw.info, eeg=True, exclude='bads')) if PS.cleaning.raw else 20
        max_comp = max(5, n_eeg - 1)
        
        # Ensure stored value is within valid range
        if PS.ica_n_components > max_comp:
            PS.ica_n_components = max_comp
        
        with ui.element('div').classes('control-section'):
            ui.label(f'Available EEG channels: {n_eeg} (max components: {max_comp})').style(
                f'color: {THEME_TEXT_DIM}; font-size: 0.7rem; margin-bottom: 8px;'
            )
            
            with ui.row().classes('gap-4 items-end'):
                def on_n_comp_change(e):
                    PS.ica_n_components = int(e.value) if e.value else 20
                
                def on_method_change(e):
                    PS.ica_method = e.value if e.value in ['fastica', 'infomax'] else 'fastica'
                
                # Ensure method is valid (in case picard was previously selected)
                if PS.ica_method not in ['fastica', 'infomax']:
                    PS.ica_method = 'fastica'
                
                n_comp = ui.number('Components', value=PS.ica_n_components, min=5, max=max_comp,
                                  on_change=on_n_comp_change).props('dense outlined').classes('w-32')
                if PS.show_tooltips:
                    n_comp.tooltip(HELP_TEXTS['n_components'])
                
                method = ui.select(['fastica', 'infomax'], value=PS.ica_method, label='Method',
                                  on_change=on_method_change).props('dense outlined').classes('w-32')
                if PS.show_tooltips:
                    method.tooltip(HELP_TEXTS['ica_method'])
            
            with ui.row().classes('mt-4 items-center gap-4'):
                ui.button('Compute ICA', icon='blur_on', 
                         on_click=lambda: compute_ica_async(PS.ica_n_components, PS.ica_method)).props('dense')
                
                if PS._ica_computing:
                    ui.spinner(size='sm')
                    ui.label('Computing...').style(f'color: {THEME_TEXT_DIM}; font-size: 0.8rem;')
    else:
        # ICA computed - show results with statistics
        with ui.row().classes('items-center gap-2 mb-3'):
            ui.icon('check_circle', size='sm').style(f'color: {THEME_PRIMARY};')
            ui.label(f'{PS.ica_result.n_components} components computed').style(f'color: {THEME_TEXT}; font-size: 0.85rem;')
        
        # Detection buttons
        with ui.row().classes('gap-2 mb-3'):
            ui.button('Detect EOG', icon='visibility', on_click=detect_eog).props('dense outlined').tooltip(
                'Auto-detect eye movement components (blinks, saccades)'
            )
            ui.button('Detect ECG', icon='favorite', on_click=detect_ecg).props('dense outlined').tooltip(
                'Auto-detect heartbeat components'
            )
            ui.button('Detect Muscle', icon='fitness_center', on_click=detect_muscle).props('dense outlined').tooltip(
                'Auto-detect muscle artifact components (high-frequency)'
            )
        
        # Info box explaining how to interpret
        with ui.element('div').style(
            f'background: rgba(0, 255, 136, 0.05); border: 1px solid {THEME_BORDER}; '
            f'border-radius: 4px; padding: 8px; margin-bottom: 12px;'
        ):
            ui.label('💡 How to select components to remove:').style(
                f'color: {THEME_SECONDARY}; font-size: 0.75rem; font-weight: 500;'
            )
            ui.label('• EOG (eye): High frontal activity, correlates with blinks').style(
                f'color: {THEME_TEXT_DIM}; font-size: 0.7rem;'
            )
            ui.label('• ECG (heart): Periodic ~1Hz pattern, temporal distribution').style(
                f'color: {THEME_TEXT_DIM}; font-size: 0.7rem;'
            )
            ui.label('• Muscle: High-frequency noise (>30Hz), edge electrodes').style(
                f'color: {THEME_TEXT_DIM}; font-size: 0.7rem;'
            )
        
        # Component grid with stats
        ui.label('Click to select/deselect components for removal:').style(
            f'color: {THEME_TEXT_DIM}; font-size: 0.75rem; margin-bottom: 8px;'
        )
        
        with ui.scroll_area().classes('w-full').style(
            f'max-height: 180px; border: 1px solid {THEME_BORDER}; border-radius: 4px; padding: 8px;'
        ):
            with ui.row().classes('gap-2 flex-wrap'):
                for i in range(PS.ica_result.n_components):
                    is_excluded = i in PS.ica_result.excluded
                    label = PS.ica_result.get_label(i)
                    css = 'ica-card ica-card-excluded' if is_excluded else 'ica-card'
                    
                    # Get scores if available
                    eog_score = PS.ica_result.eog_scores[i] if len(PS.ica_result.eog_scores) > i else None
                    ecg_score = PS.ica_result.ecg_scores[i] if len(PS.ica_result.ecg_scores) > i else None
                    
                    with ui.element('div').classes(css).on('click', lambda idx=i: toggle_ica_component(idx)):
                        ui.label(f'IC{i:02d}').style(
                            f'color: {THEME_TEXT}; font-size: 0.75rem; font-family: JetBrains Mono; font-weight: 500;'
                        )
                        
                        if label:
                            color = THEME_ERROR if is_excluded else THEME_WARN
                            ui.label(f'[{label}]').style(f'color: {color}; font-size: 0.6rem; font-weight: 600;')
                        
                        # Show scores as small indicators
                        with ui.row().classes('gap-1 mt-1'):
                            if eog_score is not None and abs(eog_score) > 0.5:
                                ui.label(f'👁{abs(eog_score):.1f}').style(
                                    f'color: {THEME_WARN}; font-size: 0.55rem;'
                                )
                            if ecg_score is not None and abs(ecg_score) > 0.5:
                                ui.label(f'❤{abs(ecg_score):.1f}').style(
                                    f'color: {THEME_ERROR}; font-size: 0.55rem;'
                                )
                        
                        status_icon = 'close' if is_excluded else 'radio_button_unchecked'
                        ui.icon(status_icon, size='xs').style(
                            f'color: {THEME_ERROR if is_excluded else THEME_TEXT_DIM}; margin-top: 2px;'
                        )
        
        # Summary and status
        if PS._ica_applied:
            with ui.row().classes('items-center gap-2 mt-3'):
                ui.icon('check_circle', size='sm').style(f'color: {THEME_PRIMARY};')
                ui.label(f'ICA applied! {len(PS.cleaning.ica_excluded)} components removed.').style(
                    f'color: {THEME_PRIMARY}; font-size: 0.8rem; font-weight: 500;'
                )
        elif PS.ica_result.excluded:
            ui.label(f'✓ {len(PS.ica_result.excluded)} component(s) selected: IC{", IC".join(map(str, sorted(PS.ica_result.excluded)))}').style(
                f'color: {THEME_WARN}; font-size: 0.75rem; margin-top: 8px;'
            )
        else:
            ui.label('No components selected. Click on components to mark for removal.').style(
                f'color: {THEME_TEXT_DIM}; font-size: 0.75rem; margin-top: 8px; font-style: italic;'
            )
    
    with ui.element('div').classes('action-bar'):
        if PS.ica_result and PS.ica_result.excluded and not PS._ica_applied:
            ui.button('Confirm & Apply ICA', icon='check_circle', on_click=apply_ica_with_feedback).props(
                'dense color=positive'
            ).tooltip('Remove selected components from the EEG and update the plot')
        ui.button('Reset Step', icon='replay', on_click=lambda: reset_step(CleaningStep.ICA)).props('flat dense').classes('reset-btn')
        ui.element('div').classes('flex-1')
        ui.button('← Previous', icon='arrow_back', on_click=go_to_previous_step).props('flat dense')
        ui.button('Skip →', on_click=lambda: complete_step(CleaningStep.ICA)).props('flat dense')
        ui.button('Next Step →', icon='arrow_forward', on_click=lambda: complete_step(CleaningStep.ICA)).props('dense')


def render_epochs_controls():
    """Render epochs step controls."""
    if not PS.cleaning.is_loaded:
        show_not_loaded_message()
        return
    
    ui.label('Segment the continuous EEG into fixed-length epochs.').style(
        f'color: {THEME_TEXT_DIM}; font-size: 0.8rem; margin-bottom: 16px;'
    )
    
    with ui.element('div').classes('control-section'):
        with ui.row().classes('gap-4 items-end'):
            def on_duration_change(e):
                PS.epoch_duration = float(e.value) if e.value else 2.0
            
            def on_overlap_change(e):
                PS.epoch_overlap = float(e.value) if e.value else 0.0
            
            dur = ui.number('Duration', value=PS.epoch_duration, min=0.5, max=10, step=0.5, suffix='s',
                           on_change=on_duration_change).props('dense outlined').classes('w-32')
            if PS.show_tooltips:
                dur.tooltip(HELP_TEXTS['epoch_duration'])
            
            overlap = ui.number('Overlap', value=PS.epoch_overlap, min=0, max=5, step=0.5, suffix='s',
                               on_change=on_overlap_change).props('dense outlined').classes('w-32')
            if PS.show_tooltips:
                overlap.tooltip(HELP_TEXTS['epoch_overlap'])
            
            ui.button('Create Epochs', icon='view_module', 
                     on_click=lambda: create_epochs_async(PS.epoch_duration, PS.epoch_overlap)).props('dense')
    
    if PS.epoch_result is not None:
        with ui.row().classes('gap-4 mt-4'):
            stat_card('Epochs', str(PS.epoch_result.n_total), THEME_PRIMARY)
            stat_card('Duration', f'{PS.cleaning.epoch_duration}s', THEME_SECONDARY)
    
    with ui.element('div').classes('action-bar'):
        ui.element('div').classes('flex-1')
        ui.button('← Previous', icon='arrow_back', on_click=go_to_previous_step).props('flat dense')
        ui.button('Skip →', on_click=lambda: complete_step(CleaningStep.EPOCHS)).props('flat dense')
        if PS.epoch_result:
            ui.button('Next Step →', icon='arrow_forward', on_click=lambda: complete_step(CleaningStep.EPOCHS)).props('dense')


def render_reject_controls():
    """Render rejection step controls with per-channel stats and manual rejection."""
    if PS.epoch_result is None:
        ui.label('Create epochs first in the previous step.').style(f'color: {THEME_WARN}; font-size: 0.85rem;')
        ui.button('Go to Epochs', on_click=lambda: go_to_step(CleaningStep.EPOCHS)).props('flat dense').classes('mt-2')
        return
    
    ui.label('Detect and reject bad epochs. You can use automatic detection or manually specify indices.').style(
        f'color: {THEME_TEXT_DIM}; font-size: 0.8rem; margin-bottom: 16px;'
    )
    
    # Automatic detection section
    with ui.element('div').classes('control-section'):
        ui.label('Automatic Detection').style(f'color: {THEME_SECONDARY}; font-size: 0.8rem; margin-bottom: 8px;')
        with ui.row().classes('gap-4 items-end'):
            def on_ptp_change(e):
                PS.reject_ptp_threshold = float(e.value) if e.value else 150.0
            
            def on_flat_change(e):
                PS.reject_flat_threshold = float(e.value) if e.value else 0.5
            
            ptp = ui.number('Peak-to-peak', value=PS.reject_ptp_threshold, min=50, max=500, suffix='µV',
                           on_change=on_ptp_change).props('dense outlined').classes('w-36')
            if PS.show_tooltips:
                ptp.tooltip(HELP_TEXTS['peak_to_peak'])
            
            flat = ui.number('Flat threshold', value=PS.reject_flat_threshold, min=0.1, max=5, suffix='µV',
                            on_change=on_flat_change).props('dense outlined').classes('w-32')
            if PS.show_tooltips:
                flat.tooltip(HELP_TEXTS['flat_epoch'])
            
            ui.button('Detect Bad Epochs', icon='search', 
                     on_click=lambda: detect_bad_epochs_async(PS.reject_ptp_threshold, PS.reject_flat_threshold)).props('dense')
    
    # Summary stats
    if PS.epoch_result.n_rejected > 0 or PS.epoch_result.n_good > 0:
        with ui.row().classes('gap-4 mt-4'):
            stat_card('Total', str(PS.epoch_result.n_total), THEME_TEXT)
            stat_card('Good', str(PS.epoch_result.n_good), THEME_PRIMARY)
            stat_card('Rejected', str(PS.epoch_result.n_rejected), THEME_ERROR)
            stat_card('Rate', f'{PS.epoch_result.rejection_rate:.1f}%', THEME_TEXT_DIM)
    
    # Show rejected epoch indices explicitly
    if PS.epoch_result.rejected_indices:
        with ui.element('div').classes('control-section mt-3'):
            ui.label('Rejected Epoch Indices:').style(f'color: {THEME_ERROR}; font-size: 0.8rem; margin-bottom: 8px;')
            
            # Format indices, grouping consecutive ranges
            indices = sorted(PS.epoch_result.rejected_indices)
            indices_display = format_indices_with_ranges(indices)
            
            ui.label(indices_display).style(
                f'color: {THEME_TEXT}; font-size: 0.75rem; font-family: JetBrains Mono; '
                f'background: rgba(255,85,85,0.1); padding: 8px 12px; border-radius: 4px; '
                f'border-left: 3px solid {THEME_ERROR};'
            ).classes('w-full')
    
    # Per-channel stats if available
    if PS.epoch_result.channel_rejection_stats:
        with ui.expansion('Per-Channel Rejection Stats', icon='bar_chart').classes('w-full mt-3'):
            with ui.scroll_area().classes('w-full').style('max-height: 150px;'):
                for ch_name, stats in PS.epoch_result.channel_rejection_stats.items():
                    rate = stats.get('rejection_rate_percent', 0)
                    n_bad = stats.get('n_bad_epochs', 0)
                    color = THEME_ERROR if rate > 30 else (THEME_WARN if rate > 10 else THEME_PRIMARY)
                    with ui.row().classes('items-center gap-2 py-1'):
                        ui.label(f'{ch_name}:').style(f'color: {THEME_TEXT}; font-size: 0.7rem; font-family: JetBrains Mono; width: 50px;')
                        ui.linear_progress(value=rate/100, show_value=False).style('width: 100px;').props(f'color={"red" if rate > 30 else "orange" if rate > 10 else "green"}')
                        ui.label(f'{rate:.1f}% ({n_bad} epochs)').style(f'color: {color}; font-size: 0.65rem;')
    
    # Manual rejection section
    with ui.element('div').classes('control-section mt-3'):
        ui.label('Manual Rejection').style(f'color: {THEME_SECONDARY}; font-size: 0.8rem; margin-bottom: 8px;')
        ui.label('Enter indices or ranges to reject (e.g. 1-5, 10, 15-20):').style(
            f'color: {THEME_TEXT_DIM}; font-size: 0.7rem; margin-bottom: 8px;'
        )
        
        manual_input = ui.input('Indices to reject', placeholder='e.g. 1-5, 10, 15-20, 25').props('dense outlined').classes('w-full')
        
        def parse_indices_with_ranges(indices_str: str) -> list:
            """Parse indices supporting both individual values and ranges (e.g. '1-5, 10, 15-20')."""
            indices = []
            parts = [p.strip() for p in indices_str.split(',') if p.strip()]
            for part in parts:
                if '-' in part and not part.startswith('-'):
                    # Handle range like "1-5"
                    range_parts = part.split('-')
                    if len(range_parts) == 2:
                        try:
                            start = int(range_parts[0].strip())
                            end = int(range_parts[1].strip())
                            indices.extend(range(start, end + 1))  # inclusive
                        except ValueError:
                            continue
                else:
                    # Single index
                    try:
                        indices.append(int(part))
                    except ValueError:
                        continue
            return sorted(set(indices))  # Remove duplicates and sort
        
        def add_manual_rejections():
            try:
                indices_str = manual_input.value or ''
                indices = parse_indices_with_ranges(indices_str)
                added_count = 0
                if indices:
                    for idx in indices:
                        if idx not in PS.epoch_result.rejected_indices and 0 <= idx < PS.epoch_result.n_total:
                            PS.epoch_result.rejected_indices.append(idx)
                            PS.epoch_result.rejection_reasons[idx] = PS.epoch_result.rejection_reasons.get(idx, []) + ['manual']
                            added_count += 1
                    PS.epoch_result.rejected_indices = sorted(PS.epoch_result.rejected_indices)
                    PS.epoch_result.n_rejected = len(PS.epoch_result.rejected_indices)
                    PS.epoch_result.n_good = PS.epoch_result.n_total - PS.epoch_result.n_rejected
                    render_step_controls()
                    update_main_plot()  # Update plot to show rejected epochs
                    safe_notify(f'Added {added_count} manual rejection(s)', type='info')
            except Exception as e:
                safe_notify(f'Error parsing indices: {e}', type='negative')
        
        def clear_all_rejections():
            """Clear all rejected epochs (both automatic and manual)."""
            PS.epoch_result.rejected_indices = []
            PS.epoch_result.rejection_reasons = {}
            PS.epoch_result.n_rejected = 0
            PS.epoch_result.n_good = PS.epoch_result.n_total
            PS.epoch_result.channel_rejection_stats = {}
            render_step_controls()
            update_main_plot()
            safe_notify('Cleared all rejections', type='info')
        
        with ui.row().classes('gap-2 mt-2'):
            ui.button('Add Rejections', icon='add', on_click=add_manual_rejections).props('dense outlined')
            ui.button('Clear All', icon='clear', on_click=clear_all_rejections).props('dense outlined').style(
                f'color: {THEME_WARN};'
            ).tooltip('Remove all rejected epochs (automatic and manual)')
    
    # Export section
    with ui.row().classes('gap-2 mt-3'):
        def export_rejection_json():
            from pathlib import Path
            import json
            output_path = Path(EEG_CLEAN_DIR) / f'{PS.cleaning.filename.rsplit(".", 1)[0]}_rejection_info.json'
            export_rejection_info(PS.epoch_result, str(output_path))
            safe_notify(f'Exported rejection info to {output_path}', type='positive')
        
        ui.button('Export Rejection Info (JSON)', icon='save', on_click=export_rejection_json).props('dense outlined').tooltip(
            'Export per-channel rejection stats and indices to JSON file'
        )
    
    with ui.element('div').classes('action-bar'):
        if PS.epoch_result.n_rejected > 0:
            ui.button('Apply Rejection', icon='delete_sweep', on_click=apply_rejection).props('dense').tooltip(
                'Remove rejected epochs from the dataset'
            )
        ui.button('Reset Step', icon='replay', on_click=lambda: reset_reject_step()).props(
            'flat dense'
        ).classes('reset-btn')
        ui.element('div').classes('flex-1')
        ui.button('← Previous', icon='arrow_back', on_click=go_to_previous_step).props('flat dense')
        ui.button('Skip →', on_click=lambda: complete_step(CleaningStep.REJECT)).props('flat dense').tooltip(
            'Skip to Visualize step without rejecting epochs'
        )
        ui.button('Next Step →', icon='arrow_forward', on_click=lambda: complete_step(CleaningStep.REJECT)).props('dense')


def reset_reject_step():
    """Reset the reject step - clears all rejections."""
    if PS.epoch_result:
        PS.epoch_result.rejected_indices = []
        PS.epoch_result.rejection_reasons = {}
        PS.epoch_result.n_rejected = 0
        PS.epoch_result.n_good = PS.epoch_result.n_total
        PS.epoch_result.channel_rejection_stats = {}
    refresh_all()
    safe_notify('Reset REJECT step', type='info')


def render_visualize_controls():
    """Render visualization step controls - preview final cleaned EEG and epochs."""
    if not PS.cleaning.is_loaded:
        show_not_loaded_message()
        return
    
    ui.label('Preview your cleaned EEG data before exporting.').style(
        f'color: {THEME_TEXT_DIM}; font-size: 0.8rem; margin-bottom: 16px;'
    )
    
    # Summary of preprocessing applied
    with ui.element('div').classes('control-section'):
        ui.label('Preprocessing Summary').style(
            f'color: {THEME_SECONDARY}; font-size: 0.85rem; font-weight: 500; margin-bottom: 12px;'
        )
        
        summary = PS.cleaning.get_preprocessing_summary()
        
        with ui.grid(columns=2).classes('gap-2'):
            # Original info
            summary_item('Original', f'{summary.get("original_channels", "?")} ch @ {summary.get("original_sfreq", "?")} Hz')
            summary_item('Duration', f'{summary.get("duration_sec", 0):.1f} seconds')
            
            # Filter
            fp = summary.get('filter_params', {})
            if fp:
                filter_str = f'{fp.get("highpass", "-")}-{fp.get("lowpass", "-")} Hz'
                if fp.get('notch'):
                    filter_str += f', notch {fp.get("notch")} Hz'
                summary_item('Filter', filter_str)
            else:
                summary_item('Filter', 'None applied')
            
            # Bad channels
            bad_ch = summary.get('bad_channels', [])
            interp_ch = summary.get('interpolated_channels', [])
            if interp_ch:
                summary_item('Interpolated', f'{len(interp_ch)} channels')
            elif bad_ch:
                summary_item('Bad Channels', f'{len(bad_ch)} marked')
            else:
                summary_item('Bad Channels', 'None')
            
            # Reference
            ref = summary.get('reference_type', '')
            summary_item('Reference', ref if ref else 'Original')
            
            # ICA
            ica_exc = summary.get('ica_components_excluded', [])
            if ica_exc:
                summary_item('ICA Removed', f'{len(ica_exc)} components')
            else:
                summary_item('ICA', 'Not applied')
    
    # Epochs preview if available
    if PS.epoch_result is not None and PS.epoch_result.epochs is not None:
        with ui.element('div').classes('control-section mt-4'):
            ui.label('Epochs Preview').style(
                f'color: {THEME_SECONDARY}; font-size: 0.85rem; font-weight: 500; margin-bottom: 12px;'
            )
            
            with ui.row().classes('gap-4'):
                stat_card('Total', str(PS.epoch_result.n_total), THEME_TEXT)
                stat_card('Good', str(PS.epoch_result.n_good), THEME_PRIMARY)
                stat_card('Rejected', str(PS.epoch_result.n_rejected), THEME_ERROR)
            
            # Show rejected indices
            if PS.epoch_result.rejected_indices:
                ui.label(f'Rejected: {format_indices_with_ranges(PS.epoch_result.rejected_indices)}').style(
                    f'color: {THEME_ERROR}; font-size: 0.7rem; font-family: JetBrains Mono; margin-top: 8px;'
                )
            
            # Current epoch indicator
            epoch_dur = PS.cleaning.epoch_duration or 2.0
            current_epoch = int(PS.view_start / epoch_dur) if epoch_dur > 0 else 0
            max_epoch = PS.epoch_result.n_total - 1
            
            with ui.row().classes('items-center gap-3 mt-4'):
                ui.label(f'Current Epoch:').style(f'color: {THEME_TEXT_DIM}; font-size: 0.8rem;')
                ui.label(f'{current_epoch}').style(
                    f'color: {THEME_PRIMARY}; font-size: 1.2rem; font-weight: 600; font-family: JetBrains Mono;'
                )
                ui.label(f'/ {max_epoch}').style(f'color: {THEME_TEXT_DIM}; font-size: 0.9rem;')
                
                is_rejected = current_epoch in PS.epoch_result.rejected_indices
                if is_rejected:
                    ui.chip('REJECTED', color='red').props('dense')
                else:
                    ui.chip('GOOD', color='green').props('dense')
            
            # Epoch navigation
            if PS.epoch_result.n_total > 0:
                with ui.row().classes('gap-2 mt-3 items-center'):
                    ui.button(icon='first_page', on_click=lambda: view_epoch(0)).props('dense flat round').tooltip('First epoch')
                    ui.button(icon='chevron_left', on_click=lambda: view_epoch(max(0, current_epoch - 1))).props('dense flat round').tooltip('Previous epoch')
                    
                    # Direct epoch input
                    epoch_input = ui.number(value=current_epoch, min=0, max=max_epoch).props('dense outlined').classes('w-20')
                    epoch_input.on('update:model-value', lambda e: view_epoch(int(e.args) if e.args is not None else 0))
                    
                    ui.button(icon='chevron_right', on_click=lambda: view_epoch(min(max_epoch, current_epoch + 1))).props('dense flat round').tooltip('Next epoch')
                    ui.button(icon='last_page', on_click=lambda: view_epoch(max_epoch)).props('dense flat round').tooltip('Last epoch')
    else:
        with ui.element('div').style(
            f'background: rgba(255, 255, 255, 0.03); border: 1px dashed {THEME_BORDER}; '
            f'border-radius: 4px; padding: 16px; margin-top: 12px; text-align: center;'
        ):
            ui.icon('view_module', size='md').style(f'color: {THEME_TEXT_DIM}; opacity: 0.5;')
            ui.label('No epochs created').style(f'color: {THEME_TEXT_DIM}; font-size: 0.8rem;')
            ui.label('Go back to the Epochs step to segment data').style(
                f'color: {THEME_TEXT_DIM}; font-size: 0.7rem; font-style: italic;'
            )
    
    # Signal quality metrics
    with ui.element('div').classes('control-section mt-4'):
        ui.label('Signal Preview').style(
            f'color: {THEME_SECONDARY}; font-size: 0.85rem; font-weight: 500; margin-bottom: 8px;'
        )
        ui.label('The plot above shows your cleaned continuous EEG data.').style(
            f'color: {THEME_TEXT_DIM}; font-size: 0.75rem;'
        )
    
    # Action bar
    with ui.element('div').classes('action-bar'):
        ui.element('div').classes('flex-1')
        ui.button('← Previous', icon='arrow_back', on_click=go_to_previous_step).props('flat dense')
        ui.button('Next Step →', icon='arrow_forward', on_click=lambda: complete_step(CleaningStep.VISUALIZE)).props('dense')


def summary_item(label: str, value: str):
    """Create a summary item."""
    with ui.row().classes('items-center gap-2'):
        ui.label(f'{label}:').style(f'color: {THEME_TEXT_DIM}; font-size: 0.75rem; min-width: 80px;')
        ui.label(value).style(f'color: {THEME_TEXT}; font-size: 0.75rem; font-family: JetBrains Mono;')


def view_epoch(idx: int):
    """Navigate to view a specific epoch."""
    if PS.epoch_result is None:
        return
    
    # Clamp to valid range
    max_epoch = PS.epoch_result.n_total - 1
    idx = max(0, min(idx, max_epoch))
    
    # Calculate time position for this epoch
    epoch_dur = PS.cleaning.epoch_duration or 2.0
    start_time = idx * epoch_dur
    
    # Update view
    PS.view_start = start_time
    PS.current_epoch_index = idx
    update_main_plot()
    
    # Update step controls to reflect new epoch
    if PS.cleaning.current_step == CleaningStep.VISUALIZE:
        render_step_controls()


def render_export_controls():
    """Render export step controls."""
    if not PS.cleaning.is_loaded:
        show_not_loaded_message()
        return
    
    ui.label('Export your cleaned EEG data.').style(
        f'color: {THEME_TEXT_DIM}; font-size: 0.8rem; margin-bottom: 16px;'
    )
    
    with ui.element('div').classes('control-section'):
        fmt = ui.select(
            {f.value: f.display_name for f in [ExportFormat.FIF, ExportFormat.SET, ExportFormat.EDF]},
            value=ExportFormat.FIF.value,
            label='Export Format'
        ).props('dense outlined').classes('w-64')
        
        out_dir = ui.input('Output Directory', value=str(EEG_CLEAN_DIR)).props('dense outlined').classes('w-full mt-3')
    
    with ui.row().classes('gap-3 mt-4'):
        ui.button('Export All', icon='save', on_click=lambda: export_all(out_dir.value, ExportFormat(fmt.value))).props('dense')
        if PS.epoch_result and PS.epoch_result.epochs:
            ui.button('Export Epochs Only', icon='view_module', on_click=lambda: export_epochs_only(out_dir.value)).props('dense outlined')
    
    with ui.element('div').classes('action-bar'):
        ui.element('div').classes('flex-1')
        ui.button('← Previous', icon='arrow_back', on_click=go_to_previous_step).props('flat dense')


# ============ Helper UI Functions ============

def show_not_loaded_message():
    with ui.column().classes('items-center py-8'):
        ui.icon('warning', size='lg').style(f'color: {THEME_WARN}; opacity: 0.5;')
        ui.label('Load a file first').style(f'color: {THEME_WARN}; font-size: 0.9rem; margin-top: 8px;')
        ui.button('Go to Load Step', on_click=lambda: go_to_step(CleaningStep.LOAD)).props('flat dense').classes('mt-2')


def stat_card(label: str, value: str, color: str):
    with ui.element('div').classes('stat-card'):
        ui.label(value).classes('stat-value').style(f'color: {color};')
        ui.label(label).classes('stat-label').style(f'color: {THEME_TEXT_DIM};')


def toggle_help(enabled):
    """Toggle help tooltips."""
    # Handle both direct bool and event args
    if isinstance(enabled, bool):
        PS.show_tooltips = enabled
    else:
        PS.show_tooltips = bool(enabled)
    render_step_controls()


def toggle_bad_channel_highlight(enabled):
    """Toggle bad channel highlighting on plot."""
    # NiceGUI switch sends the new value directly
    if enabled is None:
        PS.show_bad_channels_highlight = not PS.show_bad_channels_highlight
    elif isinstance(enabled, bool):
        PS.show_bad_channels_highlight = enabled
    elif isinstance(enabled, (int, float)):
        PS.show_bad_channels_highlight = bool(enabled)
    else:
        PS.show_bad_channels_highlight = str(enabled).lower() in ('true', '1', 'yes')
    update_main_plot()


def toggle_rejected_epochs_overlay(enabled):
    """Toggle rejected epochs overlay on plot."""
    if enabled is None:
        PS.show_rejected_epochs_overlay = not PS.show_rejected_epochs_overlay
    elif isinstance(enabled, bool):
        PS.show_rejected_epochs_overlay = enabled
    elif isinstance(enabled, (int, float)):
        PS.show_rejected_epochs_overlay = bool(enabled)
    else:
        PS.show_rejected_epochs_overlay = str(enabled).lower() in ('true', '1', 'yes')
    update_main_plot()


# ============ Action Handlers ============

async def load_file(path: str):
    if not path:
        safe_notify('Please select or enter a file path', type='warning')
        return
    
    try:
        path = Path(path)
        if not path.exists():
            safe_notify(f'File not found: {path}', type='negative')
            return
            
        eeg_data = load_eeg_file(path)
        PS.cleaning.load_raw(eeg_data.raw, path)
        PS.ica_result = None
        PS.epoch_result = None
        PS._step_snapshots = {}
        PS._step_raw_snapshots = {}
        PS._ica_applied = False
        
        # Save snapshot for LOAD step (original raw)
        PS.save_step_raw(CleaningStep.LOAD)
        
        refresh_all()
        safe_notify(f'Loaded: {path.name}', type='positive')
    except Exception as e:
        safe_notify(f'Error loading file: {e}', type='negative')


def get_step_input(step: CleaningStep):
    """Get the input for a step (output of previous step in the chain)."""
    steps = list(CleaningStep)
    step_idx = steps.index(step)
    
    if step_idx == 0:
        # LOAD step - input is original
        return PS.cleaning._raw_original.copy() if PS.cleaning._raw_original else PS.cleaning.raw.copy()
    
    prev_step = steps[step_idx - 1]
    if prev_step in PS._step_raw_snapshots:
        return PS._step_raw_snapshots[prev_step].copy()
    elif PS.cleaning._raw_original is not None:
        return PS.cleaning._raw_original.copy()
    else:
        return PS.cleaning.raw.copy()


def apply_preset(preset: FilterPreset):
    if not PS.cleaning.is_loaded:
        return
    
    try:
        # Get input from previous step (LOAD's output)
        raw_to_filter = get_step_input(CleaningStep.FILTER)
        
        # Apply filter
        raw_filtered, params = apply_filter_preset(raw_to_filter, preset)
        PS.cleaning._raw = raw_filtered
        PS.cleaning.filter_params = params.__dict__
        PS.selected_filter_preset = preset
        
        # Save this step's output
        PS.save_step_raw(CleaningStep.FILTER)
        
        # Invalidate subsequent steps since data changed
        invalidate_subsequent_steps(CleaningStep.FILTER)
        
        # Update history
        PS.cleaning._add_operation(CleaningStep.FILTER, 'filter', params.__dict__, f'Filter: {preset.display_name}')
        
        refresh_all()
        safe_notify(f'Applied: {preset.display_name}', type='positive')
    except Exception as e:
        safe_notify(f'Filter error: {e}', type='negative')


def apply_custom_filter_live(hp: float, lp: float, notch: float):
    """Apply custom filter with live update - uses input from previous step."""
    if not PS.cleaning.is_loaded:
        return
    
    # Get input from previous step (LOAD's output)
    base_raw = get_step_input(CleaningStep.FILTER)
    
    try:
        raw_filtered = base_raw.copy()
        
        if hp is not None and lp is not None and hp > 0 and lp > 0:
            raw_filtered = apply_bandpass_filter(raw_filtered, hp, lp)
        if notch is not None and notch > 0:
            raw_filtered = apply_notch_filter(raw_filtered, notch)
        
        PS.cleaning._raw = raw_filtered
        PS.cleaning.filter_params = {'highpass': hp, 'lowpass': lp, 'notch': notch}
        PS.selected_filter_preset = None  # Clear preset selection when using custom
        
        # Save this step's output
        PS.save_step_raw(CleaningStep.FILTER)
        
        # Invalidate subsequent steps since data changed
        invalidate_subsequent_steps(CleaningStep.FILTER)
        
        update_main_plot()
    except Exception as e:
        print(f"Live filter error: {e}")


def auto_detect_bad(std_threshold: float, flat_threshold: float, corr_threshold: float, check_corr: bool):
    if not PS.cleaning.is_loaded:
        return
    
    try:
        result = detect_bad_channels(
            PS.cleaning.raw, 
            std_threshold=std_threshold,
            flat_threshold=flat_threshold,
            corr_threshold=corr_threshold,
            check_correlation=check_corr
        )
        PS.cleaning.bad_channels = result.all_bad
        PS.cleaning.bad_channels_auto = result.all_bad.copy()
        
        # Invalidate subsequent steps since bad channels changed
        invalidate_subsequent_steps(CleaningStep.BAD_CHANNELS)
        
        render_step_controls()
        update_main_plot()
        
        n = len(result.all_bad)
        if n > 0:
            safe_notify(f'Detected {n} bad channel(s): {", ".join(result.all_bad)}', type='warning')
        else:
            safe_notify('No bad channels detected ✓', type='positive')
    except Exception as e:
        safe_notify(f'Detection error: {e}', type='negative')


def toggle_bad_channel(ch: str):
    if ch in PS.cleaning.bad_channels:
        PS.cleaning.bad_channels.remove(ch)
    else:
        PS.cleaning.bad_channels.append(ch)
        if ch not in PS.cleaning.bad_channels_manual:
            PS.cleaning.bad_channels_manual.append(ch)
    
    if PS.cleaning.raw:
        PS.cleaning.raw.info['bads'] = PS.cleaning.bad_channels.copy()
    
    # Invalidate subsequent steps since bad channels changed
    invalidate_subsequent_steps(CleaningStep.BAD_CHANNELS)
    
    render_step_controls()
    update_main_plot()


def do_interpolate():
    if not PS.cleaning.bad_channels:
        safe_notify('No bad channels to interpolate', type='warning')
        return
    
    try:
        raw_interp = interpolate_channels(PS.cleaning.raw, PS.cleaning.bad_channels)
        interpolated = PS.cleaning.bad_channels.copy()
        PS.cleaning.interpolated_channels.extend(interpolated)
        PS.cleaning.update_raw(raw_interp, 'interpolate', CleaningStep.BAD_CHANNELS, 
                              {'channels': interpolated}, f'Interpolated {len(interpolated)} ch')
        PS.cleaning.bad_channels = []
        
        # Invalidate subsequent steps since data changed
        invalidate_subsequent_steps(CleaningStep.BAD_CHANNELS)
        
        refresh_all()
        safe_notify(f'Interpolated: {", ".join(interpolated)}', type='positive')
    except Exception as e:
        safe_notify(f'Interpolation error: {e}', type='negative')


def apply_ref(ref_type: ReferenceType, ref_channels: List[str] = None):
    if not PS.cleaning.is_loaded:
        return
    
    try:
        # Get input from previous step (BAD_CHANNELS's output)
        base_raw = get_step_input(CleaningStep.REREFERENCE)
        
        raw_ref, desc = apply_reference(base_raw, ref_type, ref_channels)
        PS.cleaning._raw = raw_ref
        PS.cleaning.reference_type = desc
        
        # Save this step's output
        PS.save_step_raw(CleaningStep.REREFERENCE)
        
        # Invalidate subsequent steps since data changed
        invalidate_subsequent_steps(CleaningStep.REREFERENCE)
        
        update_main_plot()
        render_info()
        safe_notify(f'Applied: {desc}', type='positive')
    except Exception as e:
        safe_notify(f'Re-reference error: {e}', type='negative')


async def compute_ica_async(n_components: int, method: str):
    if not PS.cleaning.is_loaded:
        return
    
    # Limit n_components to available EEG channels - 1
    n_eeg = len(mne.pick_types(PS.cleaning.raw.info, eeg=True, exclude='bads'))
    max_components = n_eeg - 1
    if n_components > max_components:
        n_components = max_components
        safe_notify(f'Components limited to {max_components} (channels - 1)', type='warning')
    
    PS._ica_computing = True
    render_step_controls()
    
    try:
        loop = asyncio.get_event_loop()
        PS.ica_result = await loop.run_in_executor(None, lambda: compute_ica(PS.cleaning.raw, n_components, method))
        PS.cleaning.ica = PS.ica_result.ica
        safe_notify(f'ICA computed: {PS.ica_result.n_components} components', type='positive')
    except Exception as e:
        safe_notify(f'ICA error: {e}', type='negative')
    finally:
        PS._ica_computing = False
        render_step_controls()


def detect_eog():
    if PS.ica_result is None:
        return
    try:
        indices, _ = detect_eog_components(PS.ica_result, PS.cleaning.raw)
        if indices:
            for idx in indices:
                if idx not in PS.ica_result.excluded:
                    PS.ica_result.excluded.append(idx)
            render_step_controls()
            safe_notify(f'EOG detected: IC{", IC".join(map(str, indices))}', type='info')
        else:
            safe_notify('No EOG components found', type='warning')
    except Exception as e:
        safe_notify(f'EOG detection error: {e}', type='negative')


def detect_ecg():
    if PS.ica_result is None:
        return
    try:
        indices, _ = detect_ecg_components(PS.ica_result, PS.cleaning.raw)
        if indices:
            for idx in indices:
                if idx not in PS.ica_result.excluded:
                    PS.ica_result.excluded.append(idx)
            render_step_controls()
            safe_notify(f'ECG detected: IC{", IC".join(map(str, indices))}', type='info')
        else:
            safe_notify('No ECG components found', type='warning')
    except Exception as e:
        safe_notify(f'ECG detection error: {e}', type='negative')


def detect_muscle():
    """Detect muscle artifact components."""
    if PS.ica_result is None:
        return
    try:
        indices = detect_muscle_components(PS.ica_result, PS.cleaning.raw)
        if indices:
            for idx in indices:
                if idx not in PS.ica_result.excluded:
                    PS.ica_result.excluded.append(idx)
            render_step_controls()
            safe_notify(f'Muscle detected: IC{", IC".join(map(str, indices))}', type='info')
        else:
            safe_notify('No muscle artifact components found', type='warning')
    except Exception as e:
        safe_notify(f'Muscle detection error: {e}', type='negative')


def toggle_ica_component(idx: int):
    if PS.ica_result is None:
        return
    if idx in PS.ica_result.excluded:
        PS.ica_result.excluded.remove(idx)
    else:
        PS.ica_result.excluded.append(idx)
    render_step_controls()


def apply_ica():
    if PS.ica_result is None or not PS.ica_result.excluded:
        safe_notify('Select components to exclude first', type='warning')
        return
    
    try:
        raw_clean = apply_ica_exclusion(PS.cleaning.raw, PS.ica_result)
        excluded = PS.ica_result.excluded.copy()
        PS.cleaning.update_raw(raw_clean, 'ica', CleaningStep.ICA, {'excluded': excluded}, f'ICA: removed {len(excluded)} comp')
        PS.cleaning.ica_excluded = excluded
        PS.cleaning.ica_labels = PS.ica_result.labels.copy()
        PS._ica_applied = True
        refresh_all()
        safe_notify(f'Removed {len(excluded)} component(s)', type='positive')
    except Exception as e:
        safe_notify(f'ICA apply error: {e}', type='negative')


def apply_ica_with_feedback():
    """Apply ICA with visual feedback and plot update."""
    if PS.ica_result is None or not PS.ica_result.excluded:
        safe_notify('Select components to exclude first', type='warning')
        return
    
    try:
        raw_clean = apply_ica_exclusion(PS.cleaning.raw, PS.ica_result)
        excluded = PS.ica_result.excluded.copy()
        PS.cleaning.update_raw(raw_clean, 'ica', CleaningStep.ICA, {'excluded': excluded}, f'ICA: removed {len(excluded)} comp')
        PS.cleaning.ica_excluded = excluded
        PS.cleaning.ica_labels = PS.ica_result.labels.copy()
        PS._ica_applied = True
        
        # Invalidate subsequent steps since data changed
        invalidate_subsequent_steps(CleaningStep.ICA)
        
        # Update everything
        refresh_all()
        
        # Show success notification
        safe_notify(f'✓ ICA Applied! Removed {len(excluded)} component(s). Plot updated.', type='positive')
    except Exception as e:
        safe_notify(f'ICA apply error: {e}', type='negative')


async def create_epochs_async(duration: float, overlap: float):
    if not PS.cleaning.is_loaded:
        return
    try:
        loop = asyncio.get_event_loop()
        PS.epoch_result = await loop.run_in_executor(None, lambda: create_epochs(PS.cleaning.raw, duration, overlap))
        PS.cleaning.epoch_duration = duration
        PS.cleaning.epochs_total = PS.epoch_result.n_total
        render_step_controls()
        render_info()
        safe_notify(f'Created {PS.epoch_result.n_total} epochs', type='positive')
    except Exception as e:
        safe_notify(f'Epoch error: {e}', type='negative')


async def detect_bad_epochs_async(ptp: float, flat: float):
    if PS.epoch_result is None:
        return
    try:
        criteria = EpochRejectionCriteria(peak_to_peak_uv=ptp, flat_uv=flat)
        PS.epoch_result = detect_bad_epochs(PS.epoch_result, criteria)
        PS.cleaning.rejection_criteria = criteria.to_dict()
        render_step_controls()
        update_main_plot()  # Update plot to show rejected epochs
        safe_notify(f'Found {PS.epoch_result.n_rejected} bad epoch(s) ({PS.epoch_result.rejection_rate:.1f}%)', type='info')
    except Exception as e:
        safe_notify(f'Detection error: {e}', type='negative')


def apply_rejection():
    if PS.epoch_result is None:
        return
    try:
        n_rejected = len(PS.epoch_result.rejected_indices)
        PS.epoch_result = apply_epoch_rejection(PS.epoch_result)
        PS.cleaning.epochs = PS.epoch_result.epochs
        PS.cleaning.epochs_rejected = PS.epoch_result.rejected_indices
        render_step_controls()
        render_info()
        safe_notify(f'Rejected {n_rejected} epoch(s)', type='positive')
    except Exception as e:
        safe_notify(f'Rejection error: {e}', type='negative')


def export_all(out_dir: str, fmt: ExportFormat):
    if not PS.cleaning.is_loaded:
        return
    try:
        outputs = create_export_bundle(PS.cleaning, Path(out_dir), raw_format=fmt,
                                       include_epochs=PS.epoch_result is not None and PS.epoch_result.epochs is not None)
        safe_notify(f'Exported {len(outputs)} file(s) to {out_dir}', type='positive')
        complete_step(CleaningStep.EXPORT)
    except Exception as e:
        safe_notify(f'Export error: {e}', type='negative')


def export_epochs_only(out_dir: str):
    if PS.epoch_result is None or PS.epoch_result.epochs is None:
        safe_notify('No epochs to export', type='warning')
        return
    try:
        base_name = PS.cleaning.filename.rsplit('.', 1)[0] if PS.cleaning.filename else 'epochs'
        export_epochs(PS.epoch_result.epochs, Path(out_dir), base_name)
        safe_notify(f'Epochs exported to {out_dir}', type='positive')
    except Exception as e:
        safe_notify(f'Export error: {e}', type='negative')


# ============ Navigation ============

def go_to_step(step: CleaningStep):
    PS.cleaning.go_to_step(step)
    render_step_list()
    render_step_controls()
    update_main_plot()  # Update plot to show correct state for this step


def go_to_previous_step():
    """Navigate to the previous step in the pipeline."""
    steps = list(CleaningStep)
    current_idx = steps.index(PS.cleaning.current_step)
    if current_idx > 0:
        go_to_step(steps[current_idx - 1])


def complete_step(step: CleaningStep):
    # Save snapshot before completing
    save_step_snapshot(step)
    # Also save the raw EEG state for this step
    PS.save_step_raw(step)
    PS.cleaning.complete_step(step)
    refresh_all()


def reset_step(step: CleaningStep):
    """Reset a specific step - restores the EEG to the state from the previous step."""
    steps = list(CleaningStep)
    current_idx = steps.index(step)
    
    # Find the previous step to restore from
    if current_idx > 0:
        prev_step = steps[current_idx - 1]
        # Restore from previous step's snapshot
        if prev_step in PS._step_raw_snapshots:
            PS.cleaning._raw = PS._step_raw_snapshots[prev_step].copy()
        elif prev_step == CleaningStep.LOAD and PS.cleaning._raw_original is not None:
            PS.cleaning._raw = PS.cleaning._raw_original.copy()
    else:
        # For LOAD step, restore original
        if PS.cleaning._raw_original is not None:
            PS.cleaning._raw = PS.cleaning._raw_original.copy()
    
    # Remove snapshot for this step and later steps
    for s in steps[current_idx:]:
        PS._step_snapshots.pop(s, None)
        PS._step_raw_snapshots.pop(s, None)
        PS.cleaning.completed_steps.discard(s)
    
    # Clear step-specific data
    if step == CleaningStep.FILTER:
        PS.cleaning.filter_params = {}
        PS.selected_filter_preset = None
    elif step == CleaningStep.BAD_CHANNELS:
        PS.cleaning.bad_channels = []
        PS.cleaning.bad_channels_auto = []
        PS.cleaning.bad_channels_manual = []
        PS.cleaning.interpolated_channels = []
    elif step == CleaningStep.REREFERENCE:
        PS.cleaning.reference_type = ""
    elif step == CleaningStep.ICA:
        PS.ica_result = None
        PS.cleaning.ica = None
        PS.cleaning.ica_excluded = []
        PS.cleaning.ica_labels = {}
        PS._ica_applied = False
    elif step == CleaningStep.EPOCHS:
        PS.epoch_result = None
        PS.cleaning.epochs = None
        PS.cleaning.epochs_total = 0
    elif step == CleaningStep.REJECT:
        if PS.epoch_result:
            PS.epoch_result.rejected_indices = []
            PS.epoch_result.n_rejected = 0
            PS.epoch_result.n_good = PS.epoch_result.n_total
    
    PS.cleaning.go_to_step(step)
    refresh_all()
    safe_notify(f'Reset {step.short_name}', type='info')


def nav_back():
    PS.view_start = max(0, PS.view_start - PS.view_duration)
    update_main_plot()


def nav_forward():
    if PS.cleaning.is_loaded:
        max_start = max(0, PS.cleaning.duration - PS.view_duration)
        PS.view_start = min(max_start, PS.view_start + PS.view_duration)
    update_main_plot()


def invalidate_subsequent_steps(from_step: CleaningStep):
    """Mark all steps after from_step as not completed (need to be re-run)."""
    steps = list(CleaningStep)
    from_idx = steps.index(from_step)
    
    # Clear completed status and snapshots for all subsequent steps
    for step in steps[from_idx + 1:]:
        PS.cleaning.completed_steps.discard(step)
        PS._step_snapshots.pop(step, None)
        PS._step_raw_snapshots.pop(step, None)
        PS.reversed_steps.discard(step)
    
    # Clear epoch results if we're before EPOCHS step
    if from_idx < steps.index(CleaningStep.EPOCHS):
        PS.epoch_result = None
        PS.cleaning.epochs = None
    
    # Clear ICA results if we're before ICA step
    if from_idx < steps.index(CleaningStep.ICA):
        PS.ica_result = None
        PS._ica_applied = False


def toggle_reverse_eeg():
    """Toggle temporal reversal for the CURRENT step's OUTPUT.
    
    The reverse applies to the CURRENT output of the step, not the input.
    Example: If FILTER step has already filtered the data, reverse applies
    to the filtered data, not the original.
    
    When reverse is toggled:
    - If not reversed: take current output and reverse it
    - If already reversed: take current output and reverse it back (un-reverse)
    """
    if not PS.cleaning.is_loaded:
        safe_notify('Load a file first', type='warning')
        return
    
    current_step = PS.cleaning.current_step
    
    try:
        steps = list(CleaningStep)
        current_idx = steps.index(current_step)
        
        # Get the CURRENT output of this step (what we want to reverse)
        # This is either the snapshot if it exists, or the current raw
        if current_step in PS._step_raw_snapshots:
            current_output = PS._step_raw_snapshots[current_step]
        else:
            current_output = PS.cleaning.raw
        
        # Apply reverse to current output
        data = current_output.get_data()
        current_output._data = data[:, ::-1]
        PS.cleaning._raw = current_output
        
        # Toggle reverse state tracking
        if current_step in PS.reversed_steps:
            PS.reversed_steps.remove(current_step)
            safe_notify(f'EEG un-reversed at {current_step.short_name}', type='info')
        else:
            PS.reversed_steps.add(current_step)
            safe_notify(f'EEG reversed at {current_step.short_name}', type='warning')
        
        # Save this step's new output (now reversed/un-reversed)
        PS.save_step_raw(current_step)
        
        # Invalidate all subsequent steps - they need to re-process with new input
        invalidate_subsequent_steps(current_step)
        
        # Handle epochs if we're at or after EPOCHS step
        if PS.epoch_result and PS.epoch_result.epochs and current_idx >= steps.index(CleaningStep.EPOCHS):
            n_epochs = PS.epoch_result.n_total
            
            # Reverse epoch data
            epochs_data = PS.epoch_result.epochs.get_data()
            reversed_epochs_data = epochs_data[:, :, ::-1]
            PS.epoch_result.epochs._data = reversed_epochs_data
            PS.epoch_result.epochs._data = PS.epoch_result.epochs._data[::-1]
            
            # Update rejected indices
            if PS.epoch_result.rejected_indices:
                old_indices = PS.epoch_result.rejected_indices.copy()
                new_indices = [n_epochs - 1 - idx for idx in old_indices]
                PS.epoch_result.rejected_indices = sorted(new_indices)
                
                old_reasons = PS.epoch_result.rejection_reasons.copy()
                PS.epoch_result.rejection_reasons = {}
                for old_idx, reasons in old_reasons.items():
                    new_idx = n_epochs - 1 - old_idx
                    PS.epoch_result.rejection_reasons[new_idx] = reasons
        
        refresh_all()
    except Exception as e:
        safe_notify(f'Reverse error: {e}', type='negative')


def nav_to_start():
    """Navigate to the start of the signal."""
    PS.view_start = 0
    PS.is_playing = False
    if PS._play_button:
        PS._play_button.props('icon=play_arrow')
    update_main_plot()


def nav_to_end():
    """Navigate to the end of the signal."""
    if PS.cleaning.is_loaded:
        PS.view_start = max(0, PS.cleaning.duration - PS.view_duration)
    PS.is_playing = False
    if PS._play_button:
        PS._play_button.props('icon=play_arrow')
    update_main_plot()


async def toggle_playback_reverse():
    """Toggle reverse playback."""
    PS.is_playing = not PS.is_playing
    
    if PS._play_button is not None:
        PS._play_button.props(f'icon={"pause" if PS.is_playing else "play_arrow"}')
    
    if PS.is_playing:
        await run_playback(reverse=True)


async def toggle_playback_fast():
    """Toggle fast forward playback (2x speed)."""
    PS.is_playing = not PS.is_playing
    
    if PS._play_button is not None:
        PS._play_button.props(f'icon={"pause" if PS.is_playing else "play_arrow"}')
    
    if PS.is_playing:
        old_speed = PS.playback_speed
        PS.playback_speed = 4.0
        await run_playback(reverse=False)
        PS.playback_speed = old_speed


def set_playback_speed(speed):
    """Set playback speed."""
    if isinstance(speed, (int, float)):
        PS.playback_speed = float(speed)
    else:
        PS.playback_speed = 1.0


async def toggle_playback():
    """Toggle play/pause for signal preview."""
    PS.is_playing = not PS.is_playing
    
    # Update button icon immediately
    if PS._play_button is not None:
        PS._play_button.props(f'icon={"pause" if PS.is_playing else "play_arrow"}')
    
    if PS.is_playing:
        # Start playback
        await run_playback()


async def run_playback(reverse: bool = False):
    """Run the playback loop."""
    import asyncio
    
    while PS.is_playing and PS.cleaning.is_loaded:
        # Calculate step based on speed
        step_duration = PS.view_duration * 0.1  # Move 10% of view per frame
        step = step_duration * PS.playback_speed
        
        if reverse:
            PS.view_start = max(0, PS.view_start - step)
            # Stop at start
            if PS.view_start <= 0:
                PS.is_playing = False
                if PS._play_button is not None:
                    PS._play_button.props('icon=play_arrow')
                break
        else:
            max_start = max(0, PS.cleaning.duration - PS.view_duration)
            PS.view_start = min(max_start, PS.view_start + step)
            # Stop at end
            if PS.view_start >= max_start:
                PS.is_playing = False
                if PS._play_button is not None:
                    PS._play_button.props('icon=play_arrow')
                break
        
        update_main_plot()
        
        # Wait based on speed (faster speed = shorter wait)
        await asyncio.sleep(0.1 / PS.playback_speed)


def do_undo():
    """Undo the last operation."""
    history_size = len(PS.cleaning._raw_history)
    if history_size > 0:
        # Get the operation that will be undone
        op_desc = PS.cleaning.operations[-1].description if PS.cleaning.operations else 'last change'
        
        if PS.cleaning.undo():
            refresh_all()
            safe_notify(f'Undone: {op_desc}', type='info')
        else:
            safe_notify('Nothing to undo', type='warning')
    else:
        safe_notify(f'No history to undo. Use Reset to go back to original.', type='warning')


def do_full_reset():
    """Full reset to original loaded data."""
    PS.cleaning.reset_to_original()
    PS.ica_result = None
    PS.epoch_result = None
    PS._step_snapshots = {}
    PS._step_raw_snapshots = {}
    PS._ica_applied = False
    PS.is_playing = False
    PS.selected_filter_preset = None
    PS.reversed_steps = set()  # Clear all reverse states
    refresh_all()
    safe_notify('Reset to original EEG', type='info')


# ============ UI Updates ============

def refresh_all():
    render_step_list()
    render_history()
    render_info()
    render_step_controls()
    update_main_plot()


def update_main_plot():
    if PS.main_plot is None:
        return
    
    if not PS.cleaning.is_loaded:
        fig = go.Figure()
        fig.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(8,8,8,1)',
            margin=dict(l=80, r=20, t=20, b=40),
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            annotations=[dict(text='Load a file to preview signal', x=0.5, y=0.5, xref='paper', yref='paper',
                            showarrow=False, font=dict(color=THEME_TEXT_DIM, size=14))]
        )
        PS.main_plot.update_figure(fig)
        return
    
    try:
        # Get the appropriate raw data for the current step
        current_step = PS.cleaning.current_step
        
        # Use step-specific snapshot if viewing a completed step, otherwise use current raw
        raw = None
        if current_step in PS._step_raw_snapshots:
            raw = PS._step_raw_snapshots[current_step]
        elif current_step == CleaningStep.LOAD and PS.cleaning._raw_original is not None:
            raw = PS.cleaning._raw_original
        else:
            raw = PS.cleaning.raw
        
        if raw is None:
            return
            
        sfreq = raw.info['sfreq']
        start_sample = int(PS.view_start * sfreq)
        end_sample = min(int((PS.view_start + PS.view_duration) * sfreq), raw.n_times)
        
        # Get all EEG channels (not limited to 16)
        picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        data, times = raw[picks, start_sample:end_sample]
        ch_names = [raw.ch_names[i] for i in picks]
        
        fig = go.Figure()
        n_ch = len(ch_names)
        
        # Add rejected epochs overlay if enabled and epochs exist
        if PS.show_rejected_epochs_overlay and PS.epoch_result is not None:
            epoch_dur = PS.cleaning.epoch_duration or 2.0
            for rej_idx in PS.epoch_result.rejected_indices:
                epoch_start = rej_idx * epoch_dur
                epoch_end = epoch_start + epoch_dur
                # Only show if visible in current view
                if epoch_end >= PS.view_start and epoch_start <= PS.view_start + PS.view_duration:
                    fig.add_vrect(
                        x0=max(epoch_start, PS.view_start),
                        x1=min(epoch_end, PS.view_start + PS.view_duration),
                        fillcolor='rgba(255, 85, 85, 0.1)',
                        line=dict(color='rgba(255, 85, 85, 0.3)', width=1),
                        layer='below'
                    )
        
        # Create Y-axis tick values and labels for channels
        y_ticks = []
        y_labels = []
        
        # Adjust spacing based on number of channels
        spacing_factor = 0.35 if n_ch <= 16 else (0.25 if n_ch <= 24 else 0.2)
        
        for i, ch in enumerate(ch_names):
            offset = (n_ch - 1 - i)
            y = data[i] * 1e6
            y_norm = (y - np.mean(y)) / (np.std(y) + 1e-10) * spacing_factor + offset
            
            is_bad = ch in PS.cleaning.bad_channels
            
            # Store tick position
            y_ticks.append(offset)
            y_labels.append(ch)
            
            # Color scheme based on user preference
            if PS.show_bad_channels_highlight and is_bad:
                color = THEME_ERROR
                width = 1.5
            else:
                color = 'rgba(255, 255, 255, 0.6)'
                width = 1
            
            fig.add_trace(go.Scatter(x=times, y=y_norm, mode='lines', name=ch,
                                    line=dict(color=color, width=width),
                                    hovertemplate=f'{ch}: %{{customdata:.1f}} µV<extra></extra>',
                                    customdata=y))
        
        # Build title based on current step
        step_labels = {
            CleaningStep.LOAD: 'Raw EEG',
            CleaningStep.FILTER: 'Filtered EEG',
            CleaningStep.BAD_CHANNELS: 'Bad Channel Detection',
            CleaningStep.REREFERENCE: 'Re-Referenced EEG',
            CleaningStep.ICA: 'ICA Cleaned EEG',
            CleaningStep.EPOCHS: 'Epoched Data',
            CleaningStep.REJECT: 'Artifact Detection',
            CleaningStep.VISUALIZE: 'Final Preview',
            CleaningStep.EXPORT: 'Export Preview',
        }
        
        title_text = f'Time (s)'
        if PS.cleaning.current_step == CleaningStep.VISUALIZE and PS.epoch_result is not None:
            epoch_dur = PS.cleaning.epoch_duration or 2.0
            if epoch_dur > 0:
                current_epoch = int(PS.view_start / epoch_dur)
                title_text = f'Time (s) - Epoch {current_epoch}'
                PS.current_epoch_index = current_epoch
        
        fig.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(8,8,8,1)',
            margin=dict(l=80, r=20, t=20, b=40),
            xaxis=dict(title=title_text, gridcolor='rgba(0,255,136,0.06)', tickfont=dict(size=10)),
            yaxis=dict(
                tickmode='array',
                tickvals=y_ticks,
                ticktext=y_labels,
                tickfont=dict(size=9, family='JetBrains Mono', color=THEME_TEXT_DIM),
                gridcolor='rgba(0,255,136,0.04)',
                zeroline=False
            ),
            showlegend=False, hovermode='x unified'
        )
        
        PS.main_plot.update_figure(fig)
    except Exception as e:
        print(f"Plot error: {e}")
