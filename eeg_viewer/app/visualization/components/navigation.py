"""
Navigation controls component for EEG viewer.

Provides reusable navigation buttons and time controls.
"""
from __future__ import annotations
from typing import Callable, Optional, TYPE_CHECKING
from nicegui import ui

from ..styles.theme import THEME_PRIMARY, THEME_CARD, THEME_BORDER


class NavigationControls:
    """
    Reusable navigation controls for EEG time-series.
    
    Provides:
    - Start/Back/Forward/End buttons
    - Play/Pause toggle
    - Window duration selector
    - Time display
    
    Usage:
        nav = NavigationControls(
            on_start=lambda: state.view_start = 0,
            on_back=lambda: state.view_start -= state.view_duration,
            on_forward=lambda: state.view_start += state.view_duration,
            on_end=lambda: state.view_start = max_time,
            on_play_toggle=toggle_playback,
            on_window_change=lambda val: state.view_duration = val,
        )
        nav.render()
    """
    
    def __init__(
        self,
        on_start: Callable[[], None],
        on_back: Callable[[], None],
        on_forward: Callable[[], None],
        on_end: Callable[[], None],
        on_play_toggle: Callable[[], None],
        on_window_change: Callable[[float], None],
        initial_window: float = 5.0,
        window_options: list = None
    ):
        self.on_start = on_start
        self.on_back = on_back
        self.on_forward = on_forward
        self.on_end = on_end
        self.on_play_toggle = on_play_toggle
        self.on_window_change = on_window_change
        self.initial_window = initial_window
        self.window_options = window_options or [2, 5, 10, 30, 60]
        
        self.play_button: Optional[ui.button] = None
        self.time_label: Optional[ui.label] = None
        self.is_playing = False
    
    def render(self) -> 'NavigationControls':
        """Render the navigation controls."""
        with ui.row().classes('items-center gap-2'):
            # Navigation buttons
            ui.button(icon='first_page', on_click=self.on_start).props(
                'flat dense'
            ).classes('text-white').tooltip('Start')
            
            ui.button(icon='navigate_before', on_click=self.on_back).props(
                'flat dense'
            ).classes('text-white').tooltip('Back')
            
            self.play_button = ui.button(
                icon='play_arrow',
                on_click=self._handle_play_toggle
            ).props('flat dense').classes('text-white').tooltip('Play/Pause')
            
            ui.button(icon='navigate_next', on_click=self.on_forward).props(
                'flat dense'
            ).classes('text-white').tooltip('Forward')
            
            ui.button(icon='last_page', on_click=self.on_end).props(
                'flat dense'
            ).classes('text-white').tooltip('End')
            
            ui.separator().props('vertical')
            
            # Window duration selector
            ui.label('Window:').classes('text-xs text-gray-400')
            ui.select(
                options=self.window_options,
                value=self.initial_window,
                on_change=lambda e: self.on_window_change(e.value)
            ).props('dense borderless').classes('w-16')
            ui.label('s').classes('text-xs text-gray-400')
            
            ui.separator().props('vertical')
            
            # Time display
            self.time_label = ui.label('0:00.0 / 0:00.0').classes(
                'font-mono text-sm'
            ).style(f'color: {THEME_PRIMARY}')
        
        return self
    
    def _handle_play_toggle(self):
        """Handle play/pause toggle."""
        self.is_playing = not self.is_playing
        if self.play_button:
            self.play_button.props(
                f"icon={'pause' if self.is_playing else 'play_arrow'}"
            )
        self.on_play_toggle()
    
    def set_playing(self, playing: bool):
        """Update play state externally."""
        self.is_playing = playing
        if self.play_button:
            self.play_button.props(
                f"icon={'pause' if playing else 'play_arrow'}"
            )
    
    def update_time_display(self, current: float, total: float):
        """Update the time display label."""
        if self.time_label:
            cm, cs = int(current // 60), current % 60
            tm, ts = int(total // 60), total % 60
            self.time_label.set_text(f'{cm}:{cs:04.1f} / {tm}:{ts:04.1f}')


class FilterControls:
    """
    Reusable filter controls for EEG signal processing.
    
    Provides:
    - Notch filter toggle and frequency selector
    - Bandpass filter toggle and frequency range
    
    Usage:
        filters = FilterControls(
            on_notch_toggle=lambda enabled: state.notch_enabled = enabled,
            on_notch_freq_change=lambda freq: state.notch_freq = freq,
            on_bandpass_toggle=lambda enabled: state.bandpass_enabled = enabled,
            on_bandpass_change=lambda low, high: (state.bandpass_low, state.bandpass_high) = (low, high),
        )
        filters.render()
    """
    
    def __init__(
        self,
        on_notch_toggle: Callable[[bool], None],
        on_notch_freq_change: Callable[[float], None],
        on_bandpass_toggle: Callable[[bool], None],
        on_bandpass_change: Callable[[float, float], None],
        initial_notch_enabled: bool = False,
        initial_notch_freq: float = 50.0,
        initial_bandpass_enabled: bool = False,
        initial_bandpass_low: float = 1.0,
        initial_bandpass_high: float = 40.0
    ):
        self.on_notch_toggle = on_notch_toggle
        self.on_notch_freq_change = on_notch_freq_change
        self.on_bandpass_toggle = on_bandpass_toggle
        self.on_bandpass_change = on_bandpass_change
        
        self.notch_enabled = initial_notch_enabled
        self.notch_freq = initial_notch_freq
        self.bandpass_enabled = initial_bandpass_enabled
        self.bandpass_low = initial_bandpass_low
        self.bandpass_high = initial_bandpass_high
    
    def render(self) -> 'FilterControls':
        """Render the filter controls."""
        with ui.column().classes('gap-2'):
            # Notch filter
            with ui.row().classes('items-center gap-2'):
                ui.checkbox(
                    'Notch',
                    value=self.notch_enabled,
                    on_change=lambda e: self._handle_notch_toggle(e.value)
                ).classes('text-sm')
                
                ui.select(
                    options=[50, 60],
                    value=int(self.notch_freq),
                    on_change=lambda e: self._handle_notch_freq(e.value)
                ).props('dense borderless').classes('w-16')
                ui.label('Hz').classes('text-xs text-gray-400')
            
            # Bandpass filter
            with ui.row().classes('items-center gap-2'):
                ui.checkbox(
                    'Bandpass',
                    value=self.bandpass_enabled,
                    on_change=lambda e: self._handle_bandpass_toggle(e.value)
                ).classes('text-sm')
                
                ui.number(
                    value=self.bandpass_low,
                    min=0.1, max=100, step=0.5,
                    on_change=lambda e: self._handle_bandpass_low(e.value)
                ).props('dense borderless').classes('w-16')
                ui.label('-').classes('text-gray-400')
                ui.number(
                    value=self.bandpass_high,
                    min=1, max=200, step=1,
                    on_change=lambda e: self._handle_bandpass_high(e.value)
                ).props('dense borderless').classes('w-16')
                ui.label('Hz').classes('text-xs text-gray-400')
        
        return self
    
    def _handle_notch_toggle(self, enabled: bool):
        self.notch_enabled = enabled
        self.on_notch_toggle(enabled)
    
    def _handle_notch_freq(self, freq: float):
        self.notch_freq = freq
        self.on_notch_freq_change(freq)
    
    def _handle_bandpass_toggle(self, enabled: bool):
        self.bandpass_enabled = enabled
        self.on_bandpass_toggle(enabled)
    
    def _handle_bandpass_low(self, value: float):
        self.bandpass_low = value
        self.on_bandpass_change(self.bandpass_low, self.bandpass_high)
    
    def _handle_bandpass_high(self, value: float):
        self.bandpass_high = value
        self.on_bandpass_change(self.bandpass_low, self.bandpass_high)


class ScaleControls:
    """
    Reusable scale controls for EEG amplitude scaling.
    
    Usage:
        scale = ScaleControls(
            on_scale_change=lambda factor: state.scale_factor = factor,
            initial_scale=1.0
        )
        scale.render()
    """
    
    def __init__(
        self,
        on_scale_change: Callable[[float], None],
        initial_scale: float = 1.0
    ):
        self.on_scale_change = on_scale_change
        self.scale = initial_scale
    
    def render(self) -> 'ScaleControls':
        """Render the scale controls."""
        with ui.row().classes('items-center gap-2'):
            ui.label('Scale:').classes('text-xs text-gray-400')
            
            ui.button(icon='remove', on_click=self._decrease).props(
                'flat dense size=sm'
            ).classes('text-white')
            
            ui.label(f'{self.scale:.1f}x').classes(
                'font-mono text-sm w-12 text-center'
            ).bind_text_from(self, 'scale', lambda s: f'{s:.1f}x')
            
            ui.button(icon='add', on_click=self._increase).props(
                'flat dense size=sm'
            ).classes('text-white')
        
        return self
    
    def _decrease(self):
        self.scale = max(0.25, self.scale * 0.8)
        self.on_scale_change(self.scale)
    
    def _increase(self):
        self.scale = min(4.0, self.scale * 1.25)
        self.on_scale_change(self.scale)

