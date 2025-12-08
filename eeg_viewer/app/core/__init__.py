"""
Core business logic module for EEG Viewer.

Contains signal processing, plot updaters, and other core functionality.
"""
from . import signal
from .updaters import (
    update_eeg_plot,
    update_fft_plot,
    update_hilbert_plot,
    update_brain_plot,
    PRIMARY_COLORS,
    SECONDARY_COLORS,
)

__all__ = [
    'signal',
    'update_eeg_plot',
    'update_fft_plot',
    'update_hilbert_plot',
    'update_brain_plot',
    'PRIMARY_COLORS',
    'SECONDARY_COLORS',
]

