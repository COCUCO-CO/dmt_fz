"""
Pipeline page header component.

Contains title, version, running indicator, system monitor, and navigation buttons.
"""
from nicegui import ui

from config import THEME_BG, THEME_BORDER, THEME_PRIMARY, THEME_TEXT_DIM
from app.visualization.components.running_indicator import render_running_indicator
from app.visualization.components.system_monitor import render_system_monitor_compact


def render_header() -> None:
    """
    Render the pipeline page header.
    
    Contains:
    - App title and version
    - Global running indicator
    - System monitor (CPU/RAM/GPU)
    - Navigation buttons to other pages
    """
    with ui.header().classes('items-center px-4 py-1').style(
        f'background: {THEME_BG}; border-bottom: 1px solid {THEME_BORDER};'
    ):
        # Title
        ui.label('▶').style(
            f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.75rem;'
        )
        ui.label('EEG_PIPELINE').classes('text-base font-medium ml-2').style(
            f'color: {THEME_PRIMARY}; font-family: JetBrains Mono;'
        )
        ui.label('v1.0').classes('text-xs ml-2').style(
            f'color: {THEME_TEXT_DIM}; font-family: JetBrains Mono;'
        )
        
        # Global running indicator
        render_running_indicator()
        
        # System monitor (CPU/RAM/GPU) - global visibility
        with ui.element('div').classes('ml-4'):
            render_system_monitor_compact()
        
        # Navigation buttons
        with ui.row().classes('ml-auto gap-2'):
            _nav_button('VIEWER', '/', False)
            _nav_button('CLEANER', '/cleaner', False)
            _nav_button('PIPELINE', '/pipeline', True)  # Current page
            _nav_button('MODEL', '/model', False)
            _nav_button('ANALYSIS', '/analysis', False)


def _nav_button(text: str, url: str, active: bool) -> None:
    """Create a navigation button."""
    color = THEME_PRIMARY if active else THEME_TEXT_DIM
    ui.button(text, on_click=lambda: ui.navigate.to(url)).props('flat dense').style(
        f'color:{color};'
    )

