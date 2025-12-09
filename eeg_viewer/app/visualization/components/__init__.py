"""
Reusable UI components for EEG Viewer.

Provides modular, reusable components following NiceGUI patterns.
"""
from .navigation import (
    NavigationControls,
    FilterControls,
    ScaleControls,
)
from .debug_console import render_debug_toggle, render_debug_console, debug_log

__all__ = [
    'NavigationControls',
    'FilterControls',
    'ScaleControls',
    'render_debug_toggle',
    'render_debug_console',
    'debug_log',
]

