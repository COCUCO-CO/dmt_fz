"""
Brain topography figure.

Creates the 2D head map with electrode positions.
"""
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go

from .base import BaseFigure, FigureFactory
from ..styles.theme import (
    THEME_PRIMARY,
    THEME_CARD,
    BRAIN_HEIGHT,
    FONT_FAMILY,
    FONT_SIZE_NORMAL,
    THEME_TEXT,
    SECONDARY_COLOR,
)


class BrainFigure(BaseFigure):
    """
    Brain topography visualization figure.
    
    Shows 2D head outline with:
    - Circular head outline
    - Nose indicator
    - Ear markers
    - Electrode positions (added separately)
    """
    
    def _get_default_primary(self) -> str:
        return THEME_PRIMARY
    
    @property
    def head_color(self) -> str:
        """Color for head outline based on style."""
        if self.use_secondary:
            return 'rgba(244,114,182,0.5)'  # Pink
        return 'rgba(0,255,136,0.5)'  # Green
    
    def create(self) -> go.Figure:
        """Create configured brain figure with head outline."""
        fig = go.Figure()
        
        # Head outline
        theta = np.linspace(0, 2 * np.pi, 100)
        fig.add_trace(go.Scatter(
            x=np.cos(theta),
            y=np.sin(theta),
            mode='lines',
            line=dict(color=self.head_color, width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Nose
        fig.add_trace(go.Scatter(
            x=[-0.08, 0, 0.08],
            y=[0.98, 1.12, 0.98],
            mode='lines',
            line=dict(color=self.head_color, width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Left ear
        fig.add_trace(go.Scatter(
            x=[-1.02, -1.08, -1.02],
            y=[0.15, 0, -0.15],
            mode='lines',
            line=dict(color=self.head_color, width=1.5),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Right ear
        fig.add_trace(go.Scatter(
            x=[1.02, 1.08, 1.02],
            y=[0.15, 0, -0.15],
            mode='lines',
            line=dict(color=self.head_color, width=1.5),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Apply layout
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#0a0a0a',
            margin=dict(l=5, r=5, t=5, b=5),
            height=BRAIN_HEIGHT,
            font=dict(family=FONT_FAMILY, color=THEME_TEXT),
            xaxis=dict(
                range=[-1.25, 1.25],
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                scaleanchor='y',
                fixedrange=True
            ),
            yaxis=dict(
                range=[-0.9, 1.2],
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                fixedrange=True
            ),
            showlegend=False,
            hovermode='closest',
            hoverlabel=dict(
                bgcolor=THEME_CARD,
                font=dict(
                    family=FONT_FAMILY,
                    size=FONT_SIZE_NORMAL,
                    color=SECONDARY_COLOR if self.use_secondary else THEME_PRIMARY
                )
            )
        )
        
        return fig


# Convenience function matching original API
def make_brain_fig(use_eeg2: bool = False) -> go.Figure:
    """
    Create brain topography figure.
    
    Args:
        use_eeg2: Use secondary (pink) color scheme
        
    Returns:
        Configured Plotly figure with head outline
    """
    return BrainFigure(use_secondary_style=use_eeg2).create()


# Register with factory
FigureFactory.register('brain', BrainFigure)

