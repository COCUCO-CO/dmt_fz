"""
Live file browser component for pipeline.

Displays files in the current run directory with auto-refresh
to show new files as they are generated.
"""
from pathlib import Path
from typing import Callable, Optional
from nicegui import ui

from config import THEME_PRIMARY, THEME_SECONDARY, THEME_TEXT_DIM, THEME_BG, THEME_BORDER



def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f}MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f}GB"


def get_run_file_stats(run_dir: Optional[Path]) -> dict:
    """
    Get file statistics for a run directory.
    
    Args:
        run_dir: Path to run directory
        
    Returns:
        Dictionary with file counts by category
    """
    if not run_dir or not run_dir.exists():
        return {}
    
    def count(pattern):
        return len(list(run_dir.rglob(pattern)))
    
    return {
        'phases': count('phases-*.pkl'),
        'syncro': count('syncro-*.pkl'),
        'order': count('order-*.pkl') + count('order_all*.pkl'),
        'clustering': count('clustering_results/**/*.pkl'),
        'pearson': count('pearson_results/**/*.pkl'),
        'total_pkl': count('*.pkl'),
        'total_all': sum(1 for _ in run_dir.rglob('*') if _.is_file()),
    }


class FileBrowserPanel:
    """
    Live file browser panel with auto-refresh.
    
    Shows files in the current run directory and updates
    automatically when new files appear.
    """
    
    def __init__(self, get_run_dir: Callable[[], Optional[Path]]):
        """
        Initialize file browser.
        
        Args:
            get_run_dir: Function that returns current run directory
        """
        self._get_run_dir = get_run_dir
        self._file_container = None
        self._stats_container = None
        self._last_file_count = 0
        self._expanded_dirs = set()
    
    def render(self, height: str = "200px") -> None:
        """
        Render the file browser panel.
        
        Args:
            height: CSS height for the panel
        """
        with ui.card().classes('w-full p-2').style(
            f'background: {THEME_BG}; border: 1px solid {THEME_BORDER};'
        ):
            # Header with stats
            with ui.row().classes('items-center w-full gap-2 mb-2'):
                ui.icon('folder_open', size='sm').style(f'color: {THEME_PRIMARY};')
                ui.label('FILES').style(
                    f'color: {THEME_PRIMARY}; font-family: JetBrains Mono; '
                    f'font-size: 0.8rem; font-weight: bold;'
                )
                
                # Stats badges
                self._stats_container = ui.row().classes('gap-2 ml-auto items-center')
                
                # Auto-refresh indicator
                ui.icon('autorenew', size='xs').style(
                    f'color: {THEME_TEXT_DIM}; animation: spin 2s linear infinite;'
                )
            
            # Add CSS for spin animation
            ui.add_head_html('''
                <style>
                @keyframes spin {
                    from { transform: rotate(0deg); }
                    to { transform: rotate(360deg); }
                }
                </style>
            ''')
            
            # File list with scroll
            with ui.scroll_area().style(f'height: {height}; width: 100%;'):
                self._file_container = ui.column().classes('w-full gap-0 p-1')
            
            # Initial render
            self._refresh()
            
            # Auto-refresh timer (every 3 seconds)
            ui.timer(3.0, self._refresh)
    
    def _refresh(self) -> None:
        """Refresh file list and stats."""
        try:
            run_dir = self._get_run_dir()
            
            # Update stats
            self._update_stats(run_dir)
            
            # Update file list
            self._update_files(run_dir)
        except RuntimeError:
            # Client disconnected, ignore
            pass
    
    def _update_stats(self, run_dir: Optional[Path]) -> None:
        """Update stats badges."""
        if not self._stats_container:
            return
        
        self._stats_container.clear()
        
        if not run_dir or not run_dir.exists():
            with self._stats_container:
                ui.label('No run selected').style(
                    f'color: {THEME_TEXT_DIM}; font-size: 0.65rem;'
                )
            return
        
        stats = get_run_file_stats(run_dir)
        
        with self._stats_container:
            # Show key counts as badges
            for key, count in [('phases', stats.get('phases', 0)), 
                               ('syncro', stats.get('syncro', 0)),
                               ('order', stats.get('order', 0))]:
                if count > 0:
                    ui.label(f'{key}: {count}').style(
                        f'background: {THEME_PRIMARY}20; color: {THEME_PRIMARY}; '
                        f'padding: 1px 6px; border-radius: 8px; font-size: 0.6rem; '
                        f'font-family: JetBrains Mono;'
                    )
            
            # Total
            total = stats.get('total_all', 0)
            ui.label(f'total: {total}').style(
                f'color: {THEME_TEXT_DIM}; font-size: 0.6rem; font-family: JetBrains Mono;'
            )
    
    def _update_files(self, run_dir: Optional[Path]) -> None:
        """Update file list."""
        if not self._file_container:
            return
        
        self._file_container.clear()
        
        if not run_dir or not run_dir.exists():
            with self._file_container:
                ui.label('Seleccioná o creá un RUN para ver archivos').style(
                    f'color: {THEME_TEXT_DIM}; font-size: 0.7rem; padding: 8px;'
                )
            return
        
        with self._file_container:
            # Run name header
            ui.label(run_dir.name).style(
                f'color: {THEME_PRIMARY}; font-family: JetBrains Mono; '
                f'font-size: 0.75rem; font-weight: bold; margin-bottom: 4px;'
            )
            
            # List contents
            self._render_directory(run_dir, depth=0)
    
    def _render_directory(self, dir_path: Path, depth: int) -> None:
        """Render directory contents recursively."""
        try:
            items = sorted(dir_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        except PermissionError:
            return
        
        for item in items:
            indent = depth * 16
            
            if item.is_dir():
                # Directory
                file_count = sum(1 for _ in item.rglob('*') if _.is_file())
                dir_key = str(item)
                is_expanded = dir_key in self._expanded_dirs
                
                with ui.row().classes('items-center gap-1 w-full hover:bg-gray-800 cursor-pointer').style(
                    f'padding: 2px 4px; padding-left: {indent + 4}px; border-radius: 2px;'
                ) as row:
                    icon = 'expand_more' if is_expanded else 'chevron_right'
                    expand_icon = ui.icon(icon, size='xs').style(f'color: {THEME_TEXT_DIM};')
                    ui.icon('folder', size='xs').style(f'color: {THEME_SECONDARY};')
                    ui.label(item.name).style(
                        f'color: {THEME_SECONDARY}; font-family: JetBrains Mono; font-size: 0.7rem;'
                    )
                    ui.label(f'({file_count})').style(
                        f'color: {THEME_TEXT_DIM}; font-size: 0.6rem; margin-left: auto;'
                    )
                
                def toggle_dir(d=dir_key):
                    if d in self._expanded_dirs:
                        self._expanded_dirs.discard(d)
                    else:
                        self._expanded_dirs.add(d)
                    self._refresh()
                
                row.on('click', toggle_dir)
                
                # Show children if expanded
                if is_expanded:
                    self._render_directory(item, depth + 1)
            else:
                # File
                size = format_file_size(item.stat().st_size)
                
                # Color based on file type
                if item.suffix == '.pkl':
                    color = THEME_PRIMARY
                elif item.suffix in ['.png', '.jpg', '.gif']:
                    color = '#a78bfa'  # Purple for images
                elif item.suffix == '.csv':
                    color = THEME_WARN
                else:
                    color = THEME_TEXT_DIM
                
                with ui.row().classes('items-center gap-1 w-full').style(
                    f'padding: 1px 4px; padding-left: {indent + 20}px;'
                ):
                    ui.icon('description', size='xs').style(f'color: {color};')
                    ui.label(item.name).style(
                        f'color: {color}; font-family: JetBrains Mono; font-size: 0.65rem;'
                    )
                    ui.label(size).style(
                        f'color: {THEME_TEXT_DIM}; font-size: 0.6rem; margin-left: auto;'
                    )

