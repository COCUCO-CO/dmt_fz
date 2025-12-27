"""
Console component for pipeline output logging.

Displays real-time log output with status indicator and controls.
Can be used standalone or within a tab panel.
"""
from datetime import datetime
from nicegui import ui

from config import THEME_PRIMARY, THEME_TEXT_DIM, THEME_TEXT

from app.state import PS
from ..config import PIPELINE_DIR, LOG_SECTION_MARKERS
from ..runner import stop_current_process


class ConsoleTab:
    """
    Console component.
    
    Responsibilities:
    - Display log output from pipeline execution
    - Show status indicator with elapsed time
    - Provide STOP and CLEAR controls
    - Restore logs from history on tab switch
    
    Can be rendered standalone or inside a tab panel.
    """
    
    def __init__(self):
        """Initialize console component."""
        self._status_label = None
    
    def render(self, as_tab: bool = False) -> None:
        """
        Render the console content.
        
        Args:
            as_tab: If True, wraps in ui.tab_panel. If False, renders standalone.
        """
        if as_tab:
            with ui.tab_panel('console').classes('p-2').style(
                'height: 100%; display: flex; flex-direction: column; overflow: hidden;'
            ):
                self._render_header()
                self._render_log_area()
        else:
            # Standalone mode - render directly
            with ui.column().classes('w-full h-full p-2').style(
                'display: flex; flex-direction: column; overflow: hidden;'
            ):
                self._render_header()
                self._render_log_area()
    
    def _render_header(self) -> None:
        """Render header with status and controls."""
        with ui.row().classes('items-center gap-3 mb-2 shrink-0'):
            ui.label('// OUTPUT_LOG').classes('terminal-header')
            
            # Only reset stale state if NO task is currently running
            # This preserves state when user navigates away and back during execution
            if not PS.running:
                PS.current_step = ""
                PS.start_time = None
                PS.current_process = None
            
            # Status indicator
            self._status_label = ui.label('Idle').style(
                f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; '
                f'font-size: 0.7rem; margin-left: auto;'
            )
            PS.status_label = self._status_label
            
            # Update status timer
            ui.timer(1.0, self._update_status)
            
            # Control buttons
            ui.button('STOP', on_click=self._stop_clicked, icon='stop').props(
                'flat dense size=sm color=negative'
            )
            ui.button('CLEAR', on_click=self._clear_clicked, icon='delete').props(
                'flat dense size=sm'
            )
    
    def _render_log_area(self) -> None:
        """Render scrollable log area."""
        PS.log_scroll = ui.scroll_area().classes('w-full').style(
            'background: #050505; border-radius: 4px; flex: 1; min-height: 0;'
        )
        
        with PS.log_scroll:
            PS.log_container = ui.column().classes('w-full p-3 gap-0')
            with PS.log_container:
                # Restore logs from history if available
                if PS.log_history:
                    for msg in PS.log_history:
                        if any(x in msg for x in LOG_SECTION_MARKERS):
                            ui.label('').style('height: 12px;')
                        ui.label(msg).style(
                            f'color:{THEME_TEXT}; font-family: JetBrains Mono; '
                            f'font-size: 0.75rem;'
                        )
                else:
                    ui.label('Pipeline ready. Select a step and click RUN.').style(
                        f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; '
                        f'font-size: 0.75rem;'
                    )
                    ui.label(f'Pipeline directory: {PIPELINE_DIR}').style(
                        f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; '
                        f'font-size: 0.7rem;'
                    )
    
    def _update_status(self) -> None:
        """Update status label with current state."""
        if PS.running and PS.start_time:
            elapsed = (datetime.now() - PS.start_time).seconds
            mins, secs = divmod(elapsed, 60)
            self._status_label.text = f'Running: {PS.current_step} ({mins}m {secs}s)'
            self._status_label.style(
                f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.7rem;'
            )
        else:
            self._status_label.text = 'Idle'
            self._status_label.style(
                f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.7rem;'
            )
    
    async def _stop_clicked(self) -> None:
        """Handle STOP button click."""
        await stop_current_process()
    
    def _clear_clicked(self) -> None:
        """Handle CLEAR button click."""
        if PS.log_container:
            PS.log_container.clear()
        PS.log_history.clear()

