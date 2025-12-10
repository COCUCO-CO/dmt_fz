"""
Plotly figure factories for EEG visualization.

Provides consistent, themed figures for all visualizations.
"""
from .base import BaseFigure, FigureFactory
from .eeg_figure import EEGFigure, make_eeg_fig
from .fft_figure import FFTFigure, make_fft_fig
from .hilbert_figure import HilbertFigure, make_hilbert_fig
from .brain_figure import BrainFigure, make_brain_fig

__all__ = [
    # Base classes
    'BaseFigure',
    'FigureFactory',
    # Figure classes
    'EEGFigure',
    'FFTFigure',
    'HilbertFigure',
    'BrainFigure',
    # Convenience functions (matching original API)
    'make_eeg_fig',
    'make_fft_fig',
    'make_hilbert_fig',
    'make_brain_fig',
]

