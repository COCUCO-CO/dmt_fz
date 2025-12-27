"""
Signal Preview Component - Reusable EEG signal viewer.

A complete, exportable visualization component based on the cleaning module's
signal preview. Can be used in any page that needs to display EEG data.

Features:
- All EEG channels with Y-axis labels
- Playback controls (play, pause, reverse, fast forward)
- Speed selector (0.5x, 1x, 2x, 4x)
- Navigation (start, back, forward, end)
- Reverse signal toggle
- Bad channel highlighting
- Rejected epochs overlay
"""
import asyncio
import numpy as np
from typing import List, Optional, Callable, Any
from dataclasses import dataclass, field
from nicegui import ui
import plotly.graph_objects as go
import mne

# Import theme from config
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config import (
    THEME_SECONDARY,
    THEME_ERROR, THEME_TEXT_DIM
)


@dataclass
class SignalPreviewState:
    """State for signal preview component."""
    # View settings
    view_start: float = 0.0
    view_duration: float = 5.0
    
    # Playback
    is_playing: bool = False
    playback_speed: float = 1.0
    play_reverse: bool = False
    
    # Data
    raw: Any = None  # mne.io.Raw
    bad_channels: List[str] = field(default_factory=list)
    
    # Display options
    show_bad_channels_highlight: bool = True
    
    # Epochs (optional)
    rejected_epochs: List[int] = field(default_factory=list)
    epoch_duration: float = 2.0
    show_rejected_overlay: bool = False
    
    # Reverse tracking
    is_reversed: bool = False


class SignalPreviewComponent:
    """
    Reusable signal preview component.
    
    Usage:
        preview = SignalPreviewComponent(height=420)
        preview.build()  # Call inside a NiceGUI context
        preview.set_data(raw_eeg)
        preview.update()
    """
    
    def __init__(
        self,
        height: int = 420,
        show_reverse_button: bool = True,
        show_speed_selector: bool = True,
        on_reverse: Optional[Callable] = None,
        label: str = "SIGNAL PREVIEW"
    ):
        self.height = height
        self.show_reverse_button = show_reverse_button
        self.show_speed_selector = show_speed_selector
        self.on_reverse = on_reverse
        self.label = label
        
        # State
        self.state = SignalPreviewState()
        
        # UI references
        self.plot_widget = None
        self.play_button = None
        self.time_label = None
        self.speed_select = None
        
        # Playback task
        self._playback_task = None
    
    def build(self) -> ui.card:
        """Build the component UI. Must be called inside a NiceGUI context."""
        with ui.card().classes('dark-card p-4 w-full') as card:
            # Header row with controls
            with ui.row().classes('items-center justify-between mb-3'):
                # Label
                with ui.row().classes('items-center gap-2'):
                    ui.icon('show_chart', size='xs').style(f'color: {THEME_SECONDARY};')
                    ui.label(self.label).classes('text-xs font-medium').style(
                        f'color: {THEME_SECONDARY}; letter-spacing: 1px;'
                    )
                
                # Playback controls
                with ui.row().classes('gap-1 items-center'):
                    # Go to start
                    ui.button(icon='first_page', on_click=self.nav_to_start).props(
                        'flat dense round size=sm'
                    ).tooltip('Go to start')
                    
                    # Play reverse
                    ui.button(icon='fast_rewind', on_click=self._play_reverse).props(
                        'flat dense round size=sm'
                    ).tooltip('Play reverse')
                    
                    # Back
                    ui.button(icon='chevron_left', on_click=self.nav_back).props(
                        'flat dense round size=sm'
                    ).tooltip('Previous segment')
                    
                    # Play/Pause
                    self.play_button = ui.button(
                        icon='play_arrow',
                        on_click=self.toggle_playback
                    ).props('flat dense round size=sm').tooltip('Play/Pause')
                    
                    # Forward
                    ui.button(icon='chevron_right', on_click=self.nav_forward).props(
                        'flat dense round size=sm'
                    ).tooltip('Next segment')
                    
                    # Fast forward
                    ui.button(icon='fast_forward', on_click=self._play_fast).props(
                        'flat dense round size=sm'
                    ).tooltip('Play fast forward')
                    
                    # Go to end
                    ui.button(icon='last_page', on_click=self.nav_to_end).props(
                        'flat dense round size=sm'
                    ).tooltip('Go to end')
                    
                    ui.separator().props('vertical').classes('mx-1')
                    
                    # Speed selector
                    if self.show_speed_selector:
                        self.speed_select = ui.select(
                            {0.5: '0.5x', 1.0: '1x', 2.0: '2x', 4.0: '4x'},
                            value=self.state.playback_speed,
                            label=None
                        ).props('dense borderless').classes('w-16').style('font-size: 0.7rem;')
                        self.speed_select.on('update:model-value', self._on_speed_change)
                        
                        ui.separator().props('vertical').classes('mx-1')
                    
                    # Reverse button
                    if self.show_reverse_button:
                        self.reverse_btn = ui.button(
                            icon='swap_horiz',
                            on_click=self._toggle_reverse
                        ).props('flat dense round size=sm')
                        self.reverse_btn.tooltip('Reverse EEG temporally')
                    
                    # Time display
                    self.time_label = ui.label('0:00.0 / 0:00.0').style(
                        f'color: {THEME_TEXT_DIM}; font-size: 0.75rem; font-family: JetBrains Mono;'
                    ).classes('ml-2')
            
            # Plot
            self.plot_widget = ui.plotly({}).classes('w-full').style(f'height: {self.height}px;')
            self._create_empty_figure()
        
        return card
    
    def set_data(self, raw: mne.io.Raw, bad_channels: List[str] = None):
        """Set EEG data to display."""
        self.state.raw = raw
        self.state.bad_channels = bad_channels or []
        self.state.view_start = 0.0
        self.update()
    
    def set_rejected_epochs(self, indices: List[int], epoch_duration: float = 2.0):
        """Set rejected epoch indices for overlay."""
        self.state.rejected_epochs = indices
        self.state.epoch_duration = epoch_duration
        self.state.show_rejected_overlay = True
        self.update()
    
    def update(self):
        """Update the plot with current state."""
        if self.plot_widget is None:
            return
        
        if self.state.raw is None:
            self._create_empty_figure()
            return
        
        try:
            raw = self.state.raw
            sfreq = raw.info['sfreq']
            actual_duration = raw.n_times / sfreq
            
            # Clamp view_start
            if self.state.view_start > actual_duration - self.state.view_duration:
                self.state.view_start = max(0, actual_duration - self.state.view_duration)
            if self.state.view_start < 0:
                self.state.view_start = 0
            
            start_sample = int(self.state.view_start * sfreq)
            end_sample = min(int((self.state.view_start + self.state.view_duration) * sfreq), raw.n_times)
            
            if start_sample >= raw.n_times:
                start_sample = max(0, raw.n_times - int(self.state.view_duration * sfreq))
                self.state.view_start = start_sample / sfreq
            
            # Get all EEG channels
            picks = mne.pick_types(raw.info, eeg=True, exclude=[])
            data, times = raw[picks, start_sample:end_sample]
            ch_names = [raw.ch_names[i] for i in picks]
            
            fig = go.Figure()
            n_ch = len(ch_names)
            
            # Add rejected epochs overlay
            if self.state.show_rejected_overlay and self.state.rejected_epochs:
                epoch_dur = self.state.epoch_duration
                for rej_idx in self.state.rejected_epochs:
                    epoch_start = rej_idx * epoch_dur
                    epoch_end = epoch_start + epoch_dur
                    
                    if (epoch_end >= self.state.view_start and 
                        epoch_start <= self.state.view_start + self.state.view_duration and
                        epoch_start < actual_duration and epoch_start >= 0):
                        fig.add_vrect(
                            x0=max(epoch_start, self.state.view_start),
                            x1=min(epoch_end, self.state.view_start + self.state.view_duration, actual_duration),
                            fillcolor='rgba(255, 85, 85, 0.15)',
                            line=dict(color='rgba(255, 85, 85, 0.5)', width=1),
                            layer='below'
                        )
            
            # Y-axis setup
            y_ticks = []
            y_labels = []
            spacing_factor = 0.35 if n_ch <= 16 else (0.25 if n_ch <= 24 else 0.2)
            
            for i, ch in enumerate(ch_names):
                offset = (n_ch - 1 - i)
                y = data[i] * 1e6
                y_norm = (y - np.mean(y)) / (np.std(y) + 1e-10) * spacing_factor + offset
                
                is_bad = ch in self.state.bad_channels
                y_ticks.append(offset)
                y_labels.append(ch)
                
                # Color based on bad channel status
                if self.state.show_bad_channels_highlight and is_bad:
                    color = THEME_ERROR
                    width = 1.5
                else:
                    color = 'rgba(255, 255, 255, 0.6)'
                    width = 1
                
                fig.add_trace(go.Scatter(
                    x=times, y=y_norm, mode='lines', name=ch,
                    line=dict(color=color, width=width),
                    hovertemplate=f'{ch}: %{{customdata:.1f}} µV<extra></extra>',
                    customdata=y
                ))
            
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(8,8,8,1)',
                margin=dict(l=80, r=20, t=20, b=40),
                xaxis=dict(
                    title='Time (s)',
                    gridcolor='rgba(0,255,136,0.06)',
                    tickfont=dict(size=10)
                ),
                yaxis=dict(
                    tickmode='array',
                    tickvals=y_ticks,
                    ticktext=y_labels,
                    tickfont=dict(size=9, family='JetBrains Mono', color=THEME_TEXT_DIM),
                    gridcolor='rgba(0,255,136,0.04)',
                    zeroline=False
                ),
                showlegend=False,
                hovermode='x unified'
            )
            
            self.plot_widget.update_figure(fig)
            self._update_time_label()
            
        except Exception as e:
            print(f"SignalPreview error: {e}")
    
    def _create_empty_figure(self):
        """Create empty placeholder figure."""
        fig = go.Figure()
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(8,8,8,1)',
            margin=dict(l=80, r=20, t=20, b=40),
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            annotations=[dict(
                text='Load a file to preview signal',
                x=0.5, y=0.5, xref='paper', yref='paper',
                showarrow=False, font=dict(color=THEME_TEXT_DIM, size=14)
            )]
        )
        if self.plot_widget:
            self.plot_widget.update_figure(fig)
    
    def _update_time_label(self):
        """Update time display."""
        if self.time_label is None or self.state.raw is None:
            return
        
        current = self.state.view_start
        total = self.state.raw.n_times / self.state.raw.info['sfreq']
        
        curr_min = int(current // 60)
        curr_sec = current % 60
        total_min = int(total // 60)
        total_sec = total % 60
        
        self.time_label.text = f'{curr_min}:{curr_sec:04.1f} / {total_min}:{total_sec:04.1f}'
    
    # Navigation methods
    def nav_to_start(self):
        """Navigate to start."""
        self.state.view_start = 0
        self.state.is_playing = False
        if self.play_button:
            self.play_button.props('icon=play_arrow')
        self.update()
    
    def nav_to_end(self):
        """Navigate to end."""
        if self.state.raw is not None:
            actual_duration = self.state.raw.n_times / self.state.raw.info['sfreq']
            self.state.view_start = max(0, actual_duration - self.state.view_duration)
        self.state.is_playing = False
        if self.play_button:
            self.play_button.props('icon=play_arrow')
        self.update()
    
    def nav_back(self):
        """Navigate backward."""
        self.state.view_start = max(0, self.state.view_start - self.state.view_duration)
        self.update()
    
    def nav_forward(self):
        """Navigate forward."""
        if self.state.raw is not None:
            actual_duration = self.state.raw.n_times / self.state.raw.info['sfreq']
            max_start = max(0, actual_duration - self.state.view_duration)
            self.state.view_start = min(max_start, self.state.view_start + self.state.view_duration)
        self.update()
    
    # Playback methods
    async def toggle_playback(self):
        """Toggle play/pause."""
        self.state.is_playing = not self.state.is_playing
        self.state.play_reverse = False
        
        if self.play_button:
            self.play_button.props(f'icon={"pause" if self.state.is_playing else "play_arrow"}')
        
        if self.state.is_playing:
            await self._run_playback()
    
    async def _play_reverse(self):
        """Play in reverse."""
        self.state.is_playing = not self.state.is_playing
        self.state.play_reverse = True
        
        if self.play_button:
            self.play_button.props(f'icon={"pause" if self.state.is_playing else "play_arrow"}')
        
        if self.state.is_playing:
            await self._run_playback(reverse=True)
    
    async def _play_fast(self):
        """Play fast forward."""
        self.state.is_playing = not self.state.is_playing
        self.state.play_reverse = False
        
        if self.play_button:
            self.play_button.props(f'icon={"pause" if self.state.is_playing else "play_arrow"}')
        
        if self.state.is_playing:
            old_speed = self.state.playback_speed
            self.state.playback_speed = 4.0
            await self._run_playback()
            self.state.playback_speed = old_speed
    
    async def _run_playback(self, reverse: bool = False):
        """Run playback loop."""
        if self.state.raw is None:
            return
        
        actual_duration = self.state.raw.n_times / self.state.raw.info['sfreq']
        
        while self.state.is_playing:
            step_duration = self.state.view_duration * 0.1
            step = step_duration * self.state.playback_speed
            
            if reverse:
                self.state.view_start = max(0, self.state.view_start - step)
                if self.state.view_start <= 0:
                    self.state.is_playing = False
            else:
                max_start = max(0, actual_duration - self.state.view_duration)
                self.state.view_start = min(max_start, self.state.view_start + step)
                if self.state.view_start >= max_start:
                    self.state.is_playing = False
            
            self.update()
            
            if self.play_button and not self.state.is_playing:
                self.play_button.props('icon=play_arrow')
            
            await asyncio.sleep(0.05)
    
    def _on_speed_change(self, e):
        """Handle speed change."""
        if e.args is not None:
            self.state.playback_speed = float(e.args)
    
    def _toggle_reverse(self):
        """Toggle signal reverse."""
        if self.state.raw is None:
            ui.notify('Load a file first', type='warning')
            return
        
        try:
            # Reverse the data
            data = self.state.raw.get_data()
            self.state.raw._data = data[:, ::-1]
            self.state.is_reversed = not self.state.is_reversed
            
            # Update button appearance
            if hasattr(self, 'reverse_btn'):
                if self.state.is_reversed:
                    self.reverse_btn.props('color=warning')
                else:
                    self.reverse_btn.props('')
            
            ui.notify(
                f'EEG {"reversed" if self.state.is_reversed else "un-reversed"}',
                type='warning' if self.state.is_reversed else 'info'
            )
            
            # Call external handler if provided
            if self.on_reverse:
                self.on_reverse(self.state.is_reversed)
            
            self.update()
        except Exception as e:
            ui.notify(f'Reverse error: {e}', type='negative')
    
    def set_view_duration(self, duration: float):
        """Set view window duration."""
        self.state.view_duration = duration
        self.update()
    
    def set_bad_channels(self, channels: List[str]):
        """Set bad channels for highlighting."""
        self.state.bad_channels = channels
        self.update()
    
    def set_highlight_bad(self, enabled: bool):
        """Enable/disable bad channel highlighting."""
        self.state.show_bad_channels_highlight = enabled
        self.update()






