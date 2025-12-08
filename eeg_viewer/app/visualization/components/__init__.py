"""
Reusable UI components for EEG Viewer.

Provides modular, reusable components following NiceGUI patterns.
"""
from .navigation import (
    NavigationControls,
    FilterControls,
    ScaleControls,
)
from .signal_preview import SignalPreviewComponent, SignalPreviewState
from .file_browser import FileBrowserComponent

__all__ = [
    'NavigationControls',
    'FilterControls',
    'ScaleControls',
    'SignalPreviewComponent',
    'SignalPreviewState',
    'FileBrowserComponent',
]

