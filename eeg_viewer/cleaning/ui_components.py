"""
UI Components for EEG Cleaning Pipeline

Reusable NiceGUI components for the cleaning interface.
"""

from typing import Callable, Optional, List, Dict
from nicegui import ui
import plotly.graph_objects as go
import numpy as np

from .state import CleaningState, CleaningStep


# Import theme from config
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    THEME_CARD, THEME_BORDER, THEME_PRIMARY, THEME_SECONDARY,
    THEME_WARN, THEME_ERROR, THEME_TEXT, THEME_TEXT_DIM, SIGNAL_COLORS
)


class StepIndicator:
    """Pipeline step indicator component."""
    
    def __init__(self, state: CleaningState, on_step_click: Optional[Callable] = None):
        self.state = state
        self.on_step_click = on_step_click
        self.container = None
        self._build()
    
    def _build(self):
        with ui.column().classes('gap-1 w-full') as self.container:
            self.refresh()
    
    def refresh(self):
        if self.container is None:
            return
        
        self.container.clear()
        
        with self.container:
            ui.label('// PIPELINE').classes('text-xs mb-2').style(
                f'color: {THEME_TEXT_DIM}; letter-spacing: 1px;'
            )
            
            for step in CleaningStep:
                is_current = step == self.state.current_step
                is_completed = step in self.state.completed_steps
                
                # Determine styling
                if is_current:
                    bg = 'rgba(0, 255, 136, 0.15)'
                    border = THEME_PRIMARY
                    text_color = THEME_PRIMARY
                    icon = '●'
                elif is_completed:
                    bg = 'transparent'
                    border = THEME_PRIMARY
                    text_color = THEME_PRIMARY
                    icon = '✓'
                else:
                    bg = 'transparent'
                    border = THEME_BORDER
                    text_color = THEME_TEXT_DIM
                    icon = '○'
                
                with ui.row().classes('items-center gap-2 p-2 cursor-pointer rounded').style(
                    f'background: {bg}; border-left: 2px solid {border};'
                ).on('click', lambda s=step: self._on_click(s)):
                    ui.label(icon).style(f'color: {text_color}; font-size: 0.75rem;')
                    ui.label(step.short_name).style(
                        f'color: {text_color}; font-family: JetBrains Mono; font-size: 0.75rem;'
                    )
    
    def _on_click(self, step: CleaningStep):
        if self.on_step_click:
            self.on_step_click(step)


class HistoryPanel:
    """Panel showing operation history with undo."""
    
    def __init__(self, state: CleaningState, on_undo: Optional[Callable] = None):
        self.state = state
        self.on_undo = on_undo
        self.container = None
        self._build()
    
    def _build(self):
        with ui.column().classes('gap-1 w-full') as self.container:
            self.refresh()
    
    def refresh(self):
        if self.container is None:
            return
        
        self.container.clear()
        
        with self.container:
            with ui.row().classes('items-center justify-between w-full mb-2'):
                ui.label('// HISTORY').classes('text-xs').style(
                    f'color: {THEME_TEXT_DIM}; letter-spacing: 1px;'
                )
                if self.state.can_undo:
                    ui.button('UNDO', on_click=self._do_undo).props('dense flat size=xs').style(
                        f'color: {THEME_WARN};'
                    )
            
            # Show last 5 operations
            for op in reversed(self.state.operations[-5:]):
                with ui.row().classes('items-center gap-2 py-1'):
                    ui.label('↩').style(f'color: {THEME_TEXT_DIM}; font-size: 0.7rem;')
                    ui.label(op.description[:30]).style(
                        f'color: {THEME_TEXT}; font-size: 0.7rem; font-family: JetBrains Mono;'
                    )
    
    def _do_undo(self):
        if self.state.undo() and self.on_undo:
            self.on_undo()


class ChannelBadgeGrid:
    """Grid of channel badges for selection/marking."""
    
    def __init__(self, 
                 channels: List[str],
                 selected: List[str] = None,
                 bad_channels: List[str] = None,
                 on_toggle: Optional[Callable] = None,
                 multi_select: bool = True):
        self.channels = channels
        self.selected = set(selected or [])
        self.bad_channels = set(bad_channels or [])
        self.on_toggle = on_toggle
        self.multi_select = multi_select
        self.container = None
        self._build()
    
    def _build(self):
        with ui.row().classes('gap-1 flex-wrap') as self.container:
            self._render_badges()
    
    def _render_badges(self):
        for ch in self.channels:
            is_selected = ch in self.selected
            is_bad = ch in self.bad_channels
            
            if is_bad:
                bg = 'rgba(255, 85, 85, 0.2)'
                border = THEME_ERROR
                color = THEME_ERROR
            elif is_selected:
                bg = 'rgba(0, 255, 136, 0.15)'
                border = THEME_PRIMARY
                color = THEME_PRIMARY
            else:
                bg = 'transparent'
                border = THEME_BORDER
                color = THEME_TEXT_DIM
            
            ui.button(ch, on_click=lambda c=ch: self._toggle(c)).props('dense flat').style(
                f'background: {bg} !important; '
                f'border: 1px solid {border} !important; '
                f'color: {color} !important; '
                f'font-size: 0.7rem !important; '
                f'min-width: 45px !important; '
                f'padding: 2px 6px !important;'
            )
    
    def _toggle(self, channel: str):
        if channel in self.selected:
            self.selected.remove(channel)
        else:
            if not self.multi_select:
                self.selected.clear()
            self.selected.add(channel)
        
        if self.on_toggle:
            self.on_toggle(channel, channel in self.selected)
        
        self.refresh()
    
    def refresh(self):
        if self.container:
            self.container.clear()
            with self.container:
                self._render_badges()
    
    def set_bad_channels(self, bad: List[str]):
        self.bad_channels = set(bad)
        self.refresh()
    
    def get_selected(self) -> List[str]:
        return list(self.selected)


class EEGPlotComponent:
    """EEG signal plot component."""
    
    def __init__(self, height: int = 400):
        self.height = height
        self.figure = None
        self._build()
    
    def _build(self):
        self.figure = ui.plotly({}).classes('w-full').style(f'height: {self.height}px;')
    
    def update(self, 
               data: np.ndarray,
               times: np.ndarray,
               ch_names: List[str],
               bad_channels: List[str] = None,
               title: str = "EEG Signal"):
        """Update plot with new data."""
        bad_channels = bad_channels or []
        
        fig = go.Figure()
        
        n_channels = len(ch_names)
        spacing = 1.0
        
        for i, ch in enumerate(ch_names):
            offset = (n_channels - 1 - i) * spacing
            y_data = data[i] * 1e6  # Convert to µV
            y_data = (y_data - np.mean(y_data)) / (np.std(y_data) + 1e-10) * 0.3 + offset
            
            color = THEME_ERROR if ch in bad_channels else SIGNAL_COLORS[i % len(SIGNAL_COLORS)]
            
            fig.add_trace(go.Scatter(
                x=times,
                y=y_data,
                mode='lines',
                name=ch,
                line=dict(color=color, width=1),
                hovertemplate=f'{ch}: %{{y:.2f}} µV<extra></extra>'
            ))
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(8,8,8,1)',
            margin=dict(l=60, r=20, t=40, b=40),
            title=dict(text=title, font=dict(color=THEME_PRIMARY, size=12)),
            xaxis=dict(
                title='Time (s)',
                gridcolor='rgba(0,255,136,0.08)',
                zerolinecolor='rgba(0,255,136,0.08)',
            ),
            yaxis=dict(
                showticklabels=False,
                gridcolor='rgba(0,255,136,0.08)',
            ),
            showlegend=False,
            hovermode='x unified',
        )
        
        self.figure.update_figure(fig)


class ICAComponentGrid:
    """Grid for displaying ICA components."""
    
    def __init__(self,
                 n_components: int,
                 excluded: List[int] = None,
                 labels: Dict[int, str] = None,
                 on_toggle: Optional[Callable] = None,
                 on_select: Optional[Callable] = None):
        self.n_components = n_components
        self.excluded = set(excluded or [])
        self.labels = labels or {}
        self.on_toggle = on_toggle
        self.on_select = on_select
        self.selected = None
        self.container = None
        self._build()
    
    def _build(self):
        with ui.row().classes('gap-2 flex-wrap') as self.container:
            self._render_components()
    
    def _render_components(self):
        for i in range(self.n_components):
            is_excluded = i in self.excluded
            is_selected = i == self.selected
            label = self.labels.get(i, '')
            
            if is_excluded:
                bg = 'rgba(255, 85, 85, 0.2)'
                border = THEME_ERROR
            elif is_selected:
                bg = 'rgba(0, 212, 255, 0.2)'
                border = THEME_SECONDARY
            else:
                bg = THEME_CARD
                border = THEME_BORDER
            
            with ui.card().classes('p-2 cursor-pointer').style(
                f'background: {bg}; border: 1px solid {border}; min-width: 80px;'
            ).on('click', lambda idx=i: self._select(idx)):
                ui.label(f'ICA{i:03d}').style(
                    f'color: {THEME_TEXT}; font-size: 0.75rem; font-family: JetBrains Mono;'
                )
                if label:
                    ui.label(f'[{label}]').style(
                        f'color: {THEME_WARN}; font-size: 0.65rem;'
                    )
                
                with ui.row().classes('items-center gap-1 mt-1'):
                    checked = i in self.excluded
                    cb = ui.checkbox('Exclude', value=checked).props('dense size=xs').on(
                        'update:model-value',
                        lambda e, idx=i: self._toggle_exclude(idx, e.args)
                    )
    
    def _select(self, idx: int):
        self.selected = idx
        if self.on_select:
            self.on_select(idx)
        self.refresh()
    
    def _toggle_exclude(self, idx: int, excluded: bool):
        if excluded:
            self.excluded.add(idx)
        else:
            self.excluded.discard(idx)
        
        if self.on_toggle:
            self.on_toggle(idx, excluded)
    
    def refresh(self):
        if self.container:
            self.container.clear()
            with self.container:
                self._render_components()
    
    def get_excluded(self) -> List[int]:
        return sorted(list(self.excluded))


class ProgressCard:
    """Card showing progress of a long operation."""
    
    def __init__(self, title: str = "Processing"):
        self.title = title
        self.progress = 0
        self.message = ""
        self.card = None
        self.progress_bar = None
        self.message_label = None
    
    def show(self):
        with ui.dialog() as dialog, ui.card().classes('p-6').style(
            f'background: {THEME_CARD}; border: 1px solid {THEME_PRIMARY};'
        ):
            self.card = dialog
            ui.label(self.title).style(
                f'color: {THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 1rem;'
            )
            self.progress_bar = ui.linear_progress(value=0).classes('w-64 mt-4')
            self.message_label = ui.label('').style(
                f'color: {THEME_TEXT_DIM}; font-size: 0.8rem; margin-top: 8px;'
            )
        
        dialog.open()
        return self
    
    def update(self, progress: float, message: str = ""):
        self.progress = progress
        self.message = message
        if self.progress_bar:
            self.progress_bar.value = progress
        if self.message_label:
            self.message_label.text = message
    
    def close(self):
        if self.card:
            self.card.close()


def create_info_row(label: str, value: str, accent: bool = False) -> None:
    """Create a label-value info row."""
    with ui.row().classes('items-center gap-2'):
        ui.label(f'{label}:').style(
            f'color: {THEME_TEXT_DIM}; font-size: 0.75rem; min-width: 80px;'
        )
        color = THEME_PRIMARY if accent else THEME_TEXT
        ui.label(str(value)).style(
            f'color: {color}; font-size: 0.75rem; font-family: JetBrains Mono;'
        )


def create_section_header(title: str) -> None:
    """Create a section header."""
    ui.label(f'// {title}').classes('text-xs mb-2').style(
        f'color: {THEME_TEXT_DIM}; letter-spacing: 1px; '
        f'border-bottom: 1px solid {THEME_BORDER}; padding-bottom: 4px;'
    )









