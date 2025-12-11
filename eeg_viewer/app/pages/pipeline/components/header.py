"""
Pipeline page header component.

Uses the global header for consistency across all pages.
"""
from app.visualization.components.global_header import render_global_header


def render_header() -> None:
    """
    Render the pipeline page header.
    
    Uses the global header component which includes:
    - App title and version
    - Global running indicator
    - System monitor (CPU/RAM/GPU)
    - Navigation buttons to other pages
    """
    render_global_header('pipeline')

