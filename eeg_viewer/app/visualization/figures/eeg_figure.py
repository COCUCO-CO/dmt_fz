"""
EEG time-series figure.

Creates the main EEG waveform visualization figure.
"""
from __future__ import annotations
import plotly.graph_objects as go

from .base import BaseFigure, FigureFactory
from ..styles.theme import (
    THEME_PRIMARY,
    PLOT_GRID_MINOR,
    EEG_HEIGHT,
)


class EEGFigure(BaseFigure):
    """
    EEG time-series visualization figure.
    
    Shows multiple channel waveforms stacked vertically with:
    - Time on X-axis
    - Normalized amplitude on Y-axis
    - Color-coded channels
    """
    
    def _get_default_primary(self) -> str:
        return THEME_PRIMARY
    
    def create(self) -> go.Figure:
        """Create configured EEG figure."""
        fig = go.Figure()
        
        self.apply_base_layout(
            fig,
            height=EEG_HEIGHT,
            margins=dict(l=70, r=10, t=10, b=50),
            show_legend=False,
            hover_mode='x unified'
        )
        
        # X-axis: Time
        self.configure_axis(
            fig, 'x',
            title='TIME [s]',
            fixed_range=False
        )
        
        # Y-axis: Amplitude (no title, uses channel labels)
        fig.update_yaxes(
            gridcolor=PLOT_GRID_MINOR,
            tickfont=dict(size=9, color=self.primary_color),
            fixedrange=True
        )
        
        return fig


# Convenience function matching original API
def make_eeg_fig(use_eeg2: bool = False) -> go.Figure:
    """
    Create EEG figure.
    
    Args:
        use_eeg2: Use secondary (pink) color scheme
        
    Returns:
        Configured Plotly figure
    """
    return EEGFigure(use_secondary_style=use_eeg2).create()


# Register with factory
FigureFactory.register('eeg', EEGFigure)

