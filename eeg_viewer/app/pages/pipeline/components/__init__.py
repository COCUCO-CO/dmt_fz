"""
Pipeline UI components module.

Reusable components for the pipeline page:
- Header with navigation
- IO configuration panel
- Global parameters panel
- Phase container for step grouping
- File browser with auto-refresh
- Visualization panel with contextual views
- Pearson gallery
- Animation generator
"""

from .header import render_header
from .io_config import IOConfigPanel
from .global_params import GlobalParamsPanel
from .phase_container import PhaseContainer, render_phases_with_steps
from .file_browser import FileBrowserPanel, get_run_file_stats, format_file_size
from .visualization_panel import VisualizationPanel, detect_step_from_file
from .pearson_gallery import PearsonGallery
from .animation_generator import AnimationGenerator


__all__ = [
    'render_header',
    'IOConfigPanel',
    'GlobalParamsPanel',
    'PhaseContainer',
    'render_phases_with_steps',
    'FileBrowserPanel',
    'get_run_file_stats',
    'format_file_size',
    'VisualizationPanel',
    'detect_step_from_file',
    'PearsonGallery',
    'AnimationGenerator',
]

