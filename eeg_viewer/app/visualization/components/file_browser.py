"""
Reusable File Browser Component.

Allows browsing any directory for EEG files with:
- Path input field
- Browse button (folder picker)
- Scan button to refresh
- File listing with load buttons
"""
from pathlib import Path
from typing import Callable, Optional, List, Dict
from nicegui import ui
import os

# Import theme
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config import THEME_PRIMARY, THEME_SECONDARY, THEME_TEXT_DIM
from eeg_loader import scan_eeg_directory


class FileBrowserComponent:
    """
    Reusable file browser for EEG files.
    
    Usage:
        browser = FileBrowserComponent(
            on_file_select=lambda path: load_my_file(path),
            initial_paths=["/path/to/raw", "/path/to/clean"]
        )
        browser.build()
    """
    
    def __init__(
        self,
        on_file_select: Callable[[str], None],
        initial_paths: List[str] = None,
        height: str = "250px",
        label: str = "// FILE_BROWSER"
    ):
        self.on_file_select = on_file_select
        self.initial_paths = initial_paths or []
        self.height = height
        self.label = label
        
        # UI references
        self.path_input = None
        self.files_container = None
        self.custom_paths: List[str] = []
    
    def build(self) -> ui.card:
        """Build the file browser UI."""
        with ui.card().classes('dark-card p-4 w-full') as card:
            ui.label(self.label).classes('terminal-header')
            
            # Path input row
            with ui.row().classes('w-full gap-2 mb-3 items-center'):
                self.path_input = ui.input(
                    placeholder='Enter path or click Browse...'
                ).props('dense outlined').classes('flex-1').style('font-size: 0.8rem;')
                
                ui.button(icon='folder_open', on_click=self._browse_folder).props(
                    'flat dense'
                ).tooltip('Browse folder')
                
                ui.button(icon='add', on_click=self._add_path).props(
                    'flat dense color=green'
                ).tooltip('Add path to list')
                
                ui.button(icon='refresh', on_click=self._refresh_files).props(
                    'flat dense'
                ).tooltip('Refresh file list')
            
            # Files container
            self.files_container = ui.scroll_area().classes('w-full').style(f'height: {self.height};')
            
            # Initial scan
            self._refresh_files()
        
        return card
    
    async def _browse_folder(self):
        """Open folder picker dialog."""
        # NiceGUI doesn't have native folder picker, use a workaround
        # Show a dialog with common paths and text input
        with ui.dialog() as dialog, ui.card().classes('p-4'):
            ui.label('Select Folder').classes('text-lg font-bold mb-3')
            
            folder_input = ui.input(
                value=str(Path.home()),
                label='Folder Path'
            ).props('outlined').classes('w-full mb-3')
            
            # Quick access buttons for common locations
            ui.label('Quick Access:').classes('text-xs opacity-50 mb-2')
            with ui.row().classes('gap-2 flex-wrap mb-3'):
                common_paths = [
                    ('Home', str(Path.home())),
                    ('Desktop', str(Path.home() / 'Desktop')),
                    ('Downloads', str(Path.home() / 'Downloads')),
                    ('Documents', str(Path.home() / 'Documents')),
                ]
                for name, path in common_paths:
                    if Path(path).exists():
                        ui.button(name, on_click=lambda p=path: folder_input.set_value(p)).props('dense size=sm')
            
            with ui.row().classes('gap-2 justify-end'):
                ui.button('Cancel', on_click=dialog.close).props('flat')
                
                def select_folder():
                    self.path_input.set_value(folder_input.value)
                    dialog.close()
                
                ui.button('Select', on_click=select_folder).props('color=primary')
        
        dialog.open()
    
    def _add_path(self):
        """Add current path to the list."""
        path = self.path_input.value
        if path and path not in self.custom_paths:
            if Path(path).exists() and Path(path).is_dir():
                self.custom_paths.append(path)
                self._refresh_files()
                ui.notify(f'Added: {path}', type='positive')
            else:
                ui.notify('Invalid path or not a directory', type='warning')
    
    def _refresh_files(self):
        """Refresh the file listing."""
        if self.files_container is None:
            return
        
        self.files_container.clear()
        
        with self.files_container:
            # Scan initial paths
            for path in self.initial_paths:
                self._render_path_section(path, Path(path).name)
            
            # Scan custom added paths
            for path in self.custom_paths:
                self._render_path_section(path, f"📁 {Path(path).name}")
            
            # If path input has a value, scan it too
            if self.path_input and self.path_input.value:
                input_path = self.path_input.value
                if input_path not in self.initial_paths and input_path not in self.custom_paths:
                    if Path(input_path).exists():
                        self._render_path_section(input_path, f"📂 {Path(input_path).name}")
    
    def _render_path_section(self, path: str, label: str):
        """Render a section for a path."""
        try:
            files = scan_eeg_directory(path)
            if files:
                ui.label(f'├─ {label}').style(
                    f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.8rem;'
                ).classes('mt-3 mb-2')
                
                # Group by condition/folder
                conds = {}
                for f in files:
                    conds.setdefault(f['condition'], []).append(f)
                
                for cond, cfs in conds.items():
                    with ui.expansion(f'{cond} ({len(cfs)} files)').classes('w-full'):
                        for f in cfs:
                            with ui.row().classes('file-item items-center w-full gap-3'):
                                ui.icon('description', size='sm').classes('opacity-60')
                                with ui.column().classes('flex-1'):
                                    ui.label(f['name']).classes('text-sm font-medium')
                                    ui.label(f"{f['size_mb']:.1f} MB").classes('text-xs opacity-50')
                                ui.button(
                                    icon='play_arrow',
                                    on_click=lambda e, p=f['path']: self.on_file_select(p)
                                ).props('flat dense size=sm color=red')
        except Exception as e:
            ui.label(f'Error scanning {label}: {e}').style(f'color: {THEME_TEXT_DIM};').classes('text-xs')
    
    def add_initial_path(self, path: str):
        """Add a path to initial paths."""
        if path not in self.initial_paths:
            self.initial_paths.append(path)
    
    def set_paths(self, paths: List[str]):
        """Set the paths to scan."""
        self.initial_paths = paths
        self._refresh_files()



