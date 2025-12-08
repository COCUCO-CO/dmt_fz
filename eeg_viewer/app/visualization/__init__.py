"""
Visualization module for EEG Viewer.

Provides Plotly figures, UI components, and styling.
"""
from . import figures
from . import styles

# Convenience re-exports
from .figures import (
    make_eeg_fig,
    make_fft_fig,
    make_hilbert_fig,
    make_brain_fig,
    FigureFactory,
)

__all__ = [
    'figures',
    'styles',
    'make_eeg_fig',
    'make_fft_fig',
    'make_hilbert_fig',
    'make_brain_fig',
    'FigureFactory',
]

