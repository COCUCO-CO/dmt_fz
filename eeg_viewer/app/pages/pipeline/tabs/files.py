"""
Files tab for pipeline output browsing.

Displays file statistics and directory contents for the current run.
"""
from pathlib import Path
from typing import Callable, Optional
from nicegui import ui

from config import THEME_PRIMARY, THEME_SECONDARY, THEME_TEXT_DIM

from app.state import PS
from ..utils import count_output_files


class FilesTab:
    """
    Files tab component.
    
    Responsibilities:
    - Show file statistics for current run
    - Display directory contents
    - Provide refresh functionality
    """
    
    def __init__(self, get_run_dir: Callable[[], Optional[Path]]):
        """
        Initialize files tab.
        
        Args:
            get_run_dir: Function that returns current run directory
        """
        self._get_run_dir = get_run_dir
        self._container = None
    
    def render(self) -> None:
        """Render the files tab content."""
        with ui.tab_panel('files').classes('p-2').style(
            'height: 100%; display: flex; flex-direction: column;'
        ):
            self._container = ui.column().classes('w-full flex-1').style(
                'min-height: 0; overflow: hidden;'
            )
            
            with ui.row().classes('mt-2'):
                ui.button('REFRESH', on_click=self.refresh, icon='refresh').props(
                    'flat dense size=sm'
                )
            
            # Store refresh function in PS for external access
            PS.refresh_files = self.refresh
            
            # Initial render
            self.refresh()
    
    def refresh(self) -> None:
        """Refresh the files view."""
        self._container.clear()
        run_dir = self._get_run_dir()
        
        if not run_dir or not run_dir.exists():
            with self._container:
                ui.label('No run selected').style(
                    f'color:{THEME_TEXT_DIM}; font-size: 0.75rem;'
                )
            return
        
        with self._container:
            self._render_stats(run_dir)
            self._render_contents(run_dir)
    
    def _render_stats(self, run_dir: Path) -> None:
        """Render file statistics."""
        stats = count_output_files(run_dir)
        
        ui.label(f'{run_dir.name}').style(
            f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.8rem;'
        )
        
        with ui.row().classes('gap-4 mt-2 flex-wrap'):
            for name, count in stats.items():
                color = THEME_PRIMARY if count > 0 else THEME_TEXT_DIM
                ui.label(f'{name}: {count}').style(
                    f'color:{color}; font-family: JetBrains Mono; font-size: 0.7rem;'
                )
        
        ui.separator().classes('my-2')
    
    def _render_contents(self, run_dir: Path) -> None:
        """Render directory contents."""
        with ui.scroll_area().classes('w-full flex-1').style('min-height: 150px;'):
            for item in sorted(run_dir.iterdir()):
                if item.is_dir():
                    file_count = len(list(item.rglob('*')))
                    ui.label(f'[dir] {item.name}/ ({file_count} files)').style(
                        f'color:{THEME_SECONDARY}; font-family: JetBrains Mono; '
                        f'font-size: 0.7rem;'
                    )
                else:
                    size_kb = item.stat().st_size / 1024
                    size_str = f'{size_kb:.1f}KB' if size_kb < 1024 else f'{size_kb/1024:.1f}MB'
                    ui.label(f'[file] {item.name} ({size_str})').style(
                        f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; '
                        f'font-size: 0.7rem;'
                    )








