"""
Pipeline UI components module.

Reusable components for the pipeline page:
- Header with navigation
- IO configuration panel
- Global parameters panel
- Phase container for step grouping
"""

from .header import render_header
from .io_config import IOConfigPanel
from .global_params import GlobalParamsPanel
from .phase_container import PhaseContainer, render_phases_with_steps


__all__ = [
    'render_header',
    'IOConfigPanel',
    'GlobalParamsPanel',
    'PhaseContainer',
    'render_phases_with_steps',
]

