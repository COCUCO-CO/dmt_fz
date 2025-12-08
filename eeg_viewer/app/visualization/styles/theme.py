"""
Theme constants for EEG Viewer visualization.

Re-exports theme variables from config.py for consistent styling.
"""
from __future__ import annotations

# Import from existing config to avoid duplication
import sys
from pathlib import Path

# Add parent directory to path for imports
_eeg_viewer_dir = Path(__file__).parent.parent.parent.parent
if str(_eeg_viewer_dir) not in sys.path:
    sys.path.insert(0, str(_eeg_viewer_dir))

from config import (
    THEME_BG,
    THEME_CARD,
    THEME_BORDER,
    THEME_PRIMARY,
    THEME_SECONDARY,
    THEME_WARN,
    THEME_ERROR,
    THEME_TEXT,
    THEME_TEXT_DIM,
    THEME_ACCENT,
    SIGNAL_COLORS,
    FREQ_BANDS,
)

# Plot-specific colors
PLOT_BG = '#0a0a0a'
PLOT_GRID = 'rgba(50,50,50,0.5)'
PLOT_GRID_MINOR = 'rgba(40,40,40,0.3)'

# Secondary (EEG2) colors
SECONDARY_COLOR = '#f472b6'  # Pink for second EEG

# Font settings
FONT_FAMILY = 'JetBrains Mono, monospace'
FONT_SIZE_SMALL = 9
FONT_SIZE_NORMAL = 10
FONT_SIZE_LABEL = 11

# Figure dimensions
EEG_HEIGHT = 250
FFT_HEIGHT = 180
HILBERT_HEIGHT = 180
BRAIN_HEIGHT = 200

# Band visualization colors (translucent for vrect)
BAND_COLORS = [
    'rgba(0,212,255,0.08)',   # Cyan (Delta)
    'rgba(0,255,136,0.08)',   # Green (Theta)
    'rgba(255,204,0,0.08)',   # Yellow (Alpha)
    'rgba(255,107,157,0.08)', # Pink (Beta)
    'rgba(167,139,250,0.08)', # Purple (Gamma)
]

__all__ = [
    # Theme colors
    'THEME_BG',
    'THEME_CARD',
    'THEME_BORDER',
    'THEME_PRIMARY',
    'THEME_SECONDARY',
    'THEME_WARN',
    'THEME_ERROR',
    'THEME_TEXT',
    'THEME_TEXT_DIM',
    'THEME_ACCENT',
    'SIGNAL_COLORS',
    'FREQ_BANDS',
    # Plot colors
    'PLOT_BG',
    'PLOT_GRID',
    'PLOT_GRID_MINOR',
    'SECONDARY_COLOR',
    # Fonts
    'FONT_FAMILY',
    'FONT_SIZE_SMALL',
    'FONT_SIZE_NORMAL',
    'FONT_SIZE_LABEL',
    # Dimensions
    'EEG_HEIGHT',
    'FFT_HEIGHT',
    'HILBERT_HEIGHT',
    'BRAIN_HEIGHT',
    # Band colors
    'BAND_COLORS',
]

