"""
Contextual Visualization Panel for Pipeline.

Shows visualizations relevant to the currently selected/running pipeline step.
Automatically switches based on file detection or manual selection.
"""
from pathlib import Path
from typing import Callable, Optional
from nicegui import ui

from config import THEME_PRIMARY, THEME_SECONDARY, THEME_TEXT_DIM, THEME_BORDER, THEME_BG
from app.state import PS
from ..visualizers import get_visualizer, VISUALIZER_REGISTRY


# File pattern to step mapping for auto-detection
FILE_STEP_PATTERNS = {
    'phases-': 1,
    'subject_phases_': 2,
    'network_': 3,
    'syncro-': 4,
    'order-': 5,
    'order_all': 6,
    'pearson_results': 7,
    'clustering_results': 8,
}


def detect_step_from_file(filename: str) -> Optional[int]:
    """
    Detect which step generated a file based on its name.
    
    Args:
        filename: File name or path string
        
    Returns:
        Step number (1-8) or None if unknown
    """
    filename_lower = filename.lower()
    
    for pattern, step in FILE_STEP_PATTERNS.items():
        if pattern.lower() in filename_lower:
            return step
    
    return None


class VisualizationPanel:
    """
    Contextual visualization panel with step selector.
    
    Features:
    - Step selector buttons [1] [2] [3] ... [8]
    - Auto-switch when running a step or detecting new files
    - Renders appropriate visualizer for selected step
    """
    
    def __init__(self, get_run_dir: Callable[[], Optional[Path]]):
        """
        Initialize visualization panel.
        
        Args:
            get_run_dir: Function that returns current run directory
        """
        self._get_run_dir = get_run_dir
        # Restore active step from persistent state
        self._active_step = PS.viz_active_step if PS.viz_active_step else 1
        self._container = None
        self._step_buttons = {}
        self._current_visualizer = None
        self._panel_id = f"viz-panel-{id(self)}"  # Unique ID for JS
    
    @property
    def active_step(self) -> int:
        """Get currently active step."""
        return self._active_step
    
    def set_active_step(self, step: int) -> None:
        """
        Set active step and refresh visualization.
        
        Args:
            step: Step number (1-8)
        """
        if 1 <= step <= 8:
            self._active_step = step
            PS.viz_active_step = step  # Persist for page navigation
            self._update_button_styles()
            self._render_visualizer()
    
    def refresh(self) -> None:
        """Refresh current visualization."""
        self._render_visualizer()
    
    def on_file_detected(self, filename: str) -> None:
        """
        Handle file detection for auto-switching.
        
        Args:
            filename: Name of detected file
        """
        detected_step = detect_step_from_file(filename)
        if detected_step:
            self.set_active_step(detected_step)
    
    def render(self, height: str = "300px") -> None:
        """
        Render the visualization panel.
        
        Args:
            height: Initial CSS height for the panel (resizable)
        """
        # Add CSS for resizable panels
        ui.add_head_html('''
            <style>
            .resizable-panel {
                resize: vertical;
                overflow: auto;
                min-height: 150px;
                max-height: 80vh;
            }
            .resizable-panel::-webkit-resizer {
                background: linear-gradient(135deg, transparent 50%, #00d4aa 50%);
                border-radius: 2px;
            }
            </style>
        ''')
        
        with ui.card().classes('w-full p-2 resizable-panel').style(
            f'background: {THEME_BG}; border: 1px solid {THEME_BORDER}; height: {height};'
        ):
            # Header with step selector
            self._render_header()
            
            # Visualization container (fills remaining space)
            with ui.scroll_area().classes('w-full').style('flex: 1; min-height: 0;'):
                self._container = ui.column().classes('w-full p-2')
            
            # Initial render
            self._render_visualizer()
    
    def _render_header(self) -> None:
        """Render header with step selector."""
        with ui.row().classes('items-center w-full gap-2 mb-2'):
            ui.icon('analytics', size='sm').style(f'color: {THEME_SECONDARY};')
            ui.label('VISUALIZATION').style(
                f'color: {THEME_SECONDARY}; font-family: JetBrains Mono; '
                f'font-size: 0.8rem; font-weight: bold;'
            )
            
            # Step selector buttons - show consecutive 1-6, map to actual steps [1,3,4,5,7,8]
            # Steps 2 and 6 are data processing only (no visualization)
            DISPLAY_TO_STEP = {1: 1, 2: 3, 3: 4, 4: 5, 5: 7, 6: 8}
            with ui.row().classes('gap-1 ml-2'):
                for display_num, actual_step in DISPLAY_TO_STEP.items():
                    btn = ui.button(
                        str(display_num),
                        on_click=lambda s=actual_step: self.set_active_step(s)
                    ).props('dense flat size=sm').classes('min-w-[28px]')
                    self._step_buttons[actual_step] = btn
            
            # Auto indicator
            ui.label('auto').style(
                f'color: {THEME_TEXT_DIM}; font-size: 0.6rem; margin-left: auto;'
            ).tooltip('Cambia automáticamente al detectar archivos nuevos')
            
            # Refresh button
            ui.button(icon='refresh', on_click=self.refresh).props(
                'flat dense size=sm'
            ).style(f'color: {THEME_TEXT_DIM};')
        
        self._update_button_styles()
    
    def _update_button_styles(self) -> None:
        """Update step button styles based on active step."""
        for step, btn in self._step_buttons.items():
            if step == self._active_step:
                btn.style(
                    f'background: {THEME_PRIMARY}; color: black; font-weight: bold;'
                )
            else:
                btn.style(
                    f'background: transparent; color: {THEME_TEXT_DIM}; '
                    f'border: 1px solid {THEME_BORDER};'
                )
    
    def _render_visualizer(self) -> None:
        """Render the visualizer for current step with loading indicator."""
        if not self._container:
            return
        
        self._container.clear()
        
        # Show loading spinner first
        with self._container:
            with ui.row().classes('w-full justify-center items-center p-8'):
                ui.spinner('dots', size='lg', color='primary')
                ui.label('Cargando visualización...').style(
                    f'color: {THEME_TEXT_DIM}; font-size: 0.8rem; margin-left: 8px;'
                )
        
        # Use ui.timer to defer rendering (allows spinner to show)
        ui.timer(0.05, lambda: self._do_render_visualizer(), once=True)
    
    def _do_render_visualizer(self) -> None:
        """Actually render the visualizer content."""
        if not self._container:
            return
        
        self._container.clear()
        
        try:
            # Get visualizer for current step
            self._current_visualizer = get_visualizer(self._active_step, self._get_run_dir)
            
            if self._current_visualizer:
                self._current_visualizer.render(self._container)
            else:
                with self._container:
                    ui.label(f'No visualizer for step {self._active_step}').style(
                        f'color: {THEME_TEXT_DIM};'
                    )
        except Exception as e:
            # Handle errors gracefully without breaking the UI
            with self._container:
                ui.label(f'⚠️ Error in Step {self._active_step} visualizer').style(
                    f'color: #f59e0b; font-family: JetBrains Mono; font-size: 0.8rem;'
                )
                ui.label(str(e)).style(
                    f'color: {THEME_TEXT_DIM}; font-size: 0.7rem; max-width: 400px;'
                )
                ui.button('Reintentar', on_click=self.refresh).props('flat dense').style(
                    f'color: {THEME_PRIMARY}; margin-top: 8px;'
                )
            print(f"[VisualizationPanel] Error in step {self._active_step}: {e}")

