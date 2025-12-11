"""
Global header component for all pages.

Contains:
- App title and page indicator
- System monitor (CPU/RAM/GPU)  
- Running indicator
- Navigation buttons
"""
from nicegui import ui

from config import THEME_BG, THEME_BORDER, THEME_PRIMARY, THEME_TEXT_DIM, THEME_WARN
from app.visualization.components.running_indicator import render_running_indicator
from app.visualization.components.system_monitor import render_system_monitor_compact


# Page configurations
PAGE_CONFIG = {
    'viewer': {'label': 'EEG_VIEWER', 'url': '/', 'color': THEME_PRIMARY},
    'cleaner': {'label': 'EEG_CLEANER', 'url': '/cleaner', 'color': THEME_PRIMARY},
    'pipeline': {'label': 'EEG_PIPELINE', 'url': '/pipeline', 'color': THEME_PRIMARY},
    'model': {'label': 'EEG_MODEL', 'url': '/model', 'color': '#f472b6'},
    'analysis': {'label': 'EEG_ANALYSIS', 'url': '/analysis', 'color': THEME_WARN},
}


def render_global_header(current_page: str) -> None:
    """
    Render the global header with system monitor.
    
    Args:
        current_page: Current page key ('viewer', 'cleaner', 'pipeline', 'model', 'analysis')
    """
    page_info = PAGE_CONFIG.get(current_page, PAGE_CONFIG['viewer'])
    
    with ui.header().classes('items-center px-4 py-1').style(
        f'background: {THEME_BG}; border-bottom: 1px solid {THEME_BORDER};'
    ):
        # Title
        ui.label('▶').style(
            f'color:{page_info["color"]}; font-family: JetBrains Mono; font-size: 0.75rem;'
        )
        ui.label(page_info['label']).classes('text-base font-medium ml-2').style(
            f'color: {page_info["color"]}; font-family: JetBrains Mono;'
        )
        ui.label('v1.0').classes('text-xs ml-2').style(
            f'color: {THEME_TEXT_DIM}; font-family: JetBrains Mono;'
        )
        
        # Global running indicator
        render_running_indicator()
        
        # System monitor (CPU/RAM/GPU) - GLOBAL across all pages
        with ui.element('div').classes('ml-4'):
            render_system_monitor_compact()
        
        # Navigation buttons
        with ui.row().classes('ml-auto gap-2'):
            _nav_button('VIEWER', '/', current_page == 'viewer')
            _nav_button('CLEANER', '/cleaner', current_page == 'cleaner')
            _nav_button('PIPELINE', '/pipeline', current_page == 'pipeline')
            _nav_button('MODEL', '/model', current_page == 'model')
            _nav_button('ANALYSIS', '/analysis', current_page == 'analysis')


def _nav_button(text: str, url: str, active: bool) -> None:
    """Create a navigation button."""
    if active:
        # Get color from page config
        for key, config in PAGE_CONFIG.items():
            if config['url'] == url:
                color = config['color']
                break
        else:
            color = THEME_PRIMARY
    else:
        color = THEME_TEXT_DIM
    
    ui.button(text, on_click=lambda: ui.navigate.to(url)).props('flat dense').style(
        f'color:{color};'
    )


