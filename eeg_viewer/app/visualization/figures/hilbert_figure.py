"""
Hilbert transform visualization figure.

Creates dual-panel figure for envelope and phase display.
"""
from __future__ import annotations
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .base import BaseFigure, FigureFactory
from ..styles.theme import (
    THEME_WARN,
    THEME_TEXT_DIM,
    THEME_CARD,
    PLOT_BG,
    PLOT_GRID,
    HILBERT_HEIGHT,
    FONT_FAMILY,
    FONT_SIZE_NORMAL,
    THEME_TEXT,
)


class HilbertFigure(BaseFigure):
    """
    Hilbert transform visualization figure.
    
    Two-panel subplot showing:
    - Top: Amplitude envelope
    - Bottom: Instantaneous phase
    """
    
    def _get_default_primary(self) -> str:
        return THEME_WARN
    
    def create(self) -> go.Figure:
        """Create configured Hilbert figure with subplots."""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=('<b>ENVELOPE</b>', '<b>PHASE</b>'),
            vertical_spacing=0.22
        )
        
        # Apply base layout
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor=PLOT_BG,
            margin=dict(l=60, r=10, t=35, b=50),
            height=HILBERT_HEIGHT,
            font=dict(
                family=FONT_FAMILY,
                size=FONT_SIZE_NORMAL,
                color=THEME_TEXT
            ),
            showlegend=True,
            legend=dict(
                orientation='h',
                y=1.15,
                font=dict(size=8, color=THEME_TEXT_DIM)
            ),
            hovermode='x unified',
            hoverlabel=dict(
                bgcolor=THEME_CARD,
                font=dict(family=FONT_FAMILY, size=FONT_SIZE_NORMAL)
            )
        )
        
        # Style subplot titles
        fig.update_annotations(
            font=dict(size=9, color=self.primary_color, family=FONT_FAMILY)
        )
        
        # Configure axes
        fig.update_xaxes(
            gridcolor=PLOT_GRID,
            tickfont=dict(size=9, color=THEME_TEXT_DIM),
            fixedrange=True
        )
        fig.update_yaxes(
            gridcolor=PLOT_GRID,
            tickfont=dict(size=9, color=THEME_TEXT_DIM),
            fixedrange=True
        )
        
        return fig


# Convenience function matching original API
def make_hilbert_fig(use_eeg2: bool = False) -> go.Figure:
    """
    Create Hilbert figure.
    
    Args:
        use_eeg2: Use secondary (pink) color scheme
        
    Returns:
        Configured Plotly figure with subplots
    """
    return HilbertFigure(use_secondary_style=use_eeg2).create()


# Register with factory
FigureFactory.register('hilbert', HilbertFigure)

