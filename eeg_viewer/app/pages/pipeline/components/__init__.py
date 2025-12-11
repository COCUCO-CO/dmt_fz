"""
Pipeline UI components module.

Reusable components for the pipeline page:
- Header with navigation
- IO configuration panel
- Global parameters panel
"""

from .header import render_header
from .io_config import IOConfigPanel
from .global_params import GlobalParamsPanel


__all__ = [
    'render_header',
    'IOConfigPanel',
    'GlobalParamsPanel',
]

