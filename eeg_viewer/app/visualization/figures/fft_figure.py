"""
FFT power spectrum figure.

Creates the frequency domain visualization with band annotations.
"""
from __future__ import annotations
import plotly.graph_objects as go

from .base import BaseFigure, FigureFactory
from ..styles.theme import (
    THEME_SECONDARY,
    THEME_TEXT_DIM,
    THEME_CARD,
    FFT_HEIGHT,
    FREQ_BANDS,
    BAND_COLORS,
    FONT_FAMILY,
)


class FFTFigure(BaseFigure):
    """
    FFT power spectrum visualization figure.
    
    Shows frequency content with:
    - Frequency bands highlighted (δ, θ, α, β, γ)
    - Power on Y-axis
    - Optional log scale
    """
    
    def __init__(self, use_secondary_style: bool = False, max_freq: float = 60.0):
        super().__init__(use_secondary_style)
        self.max_freq = max_freq
    
    def _get_default_primary(self) -> str:
        return THEME_SECONDARY
    
    def create(self) -> go.Figure:
        """Create configured FFT figure with band annotations."""
        fig = go.Figure()
        
        # Add frequency band rectangles and labels
        for i, (band, (lo, hi)) in enumerate(FREQ_BANDS.items()):
            # Band rectangle
            fig.add_vrect(
                x0=lo, x1=hi,
                fillcolor=BAND_COLORS[i % len(BAND_COLORS)],
                line_width=0
            )
            # Band label
            fig.add_annotation(
                x=(lo + hi) / 2,
                y=1.02,
                yref='paper',
                text=band,
                showarrow=False,
                font=dict(size=11, color=THEME_TEXT_DIM, family=FONT_FAMILY)
            )
        
        self.apply_base_layout(
            fig,
            height=FFT_HEIGHT,
            margins=dict(l=60, r=10, t=30, b=50),
            show_legend=True,
            hover_mode='x unified'
        )
        
        # Configure legend
        fig.update_layout(
            legend=dict(
                orientation='h',
                y=1.15,
                font=dict(size=8, color=THEME_TEXT_DIM)
            )
        )
        
        # X-axis: Frequency
        self.configure_axis(
            fig, 'x',
            title='FREQ [Hz]',
            fixed_range=True,
            axis_range=[0, self.max_freq]
        )
        
        # Y-axis: Power - fixed range (calculated once per EEG, reset on filter change)
        self.configure_axis(
            fig, 'y',
            title='PWR [µV]',
            fixed_range=True
        )
        
        return fig


# Convenience function matching original API
def make_fft_fig(use_eeg2: bool = False, max_freq: float = 60.0) -> go.Figure:
    """
    Create FFT figure.
    
    Args:
        use_eeg2: Use secondary (pink) color scheme
        max_freq: Maximum frequency to display
        
    Returns:
        Configured Plotly figure
    """
    return FFTFigure(use_secondary_style=use_eeg2, max_freq=max_freq).create()


# Register with factory
FigureFactory.register('fft', FFTFigure)

