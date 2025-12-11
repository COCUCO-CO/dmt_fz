"""
Input/Output configuration panel component.

Handles input directory selection and run directory management.
"""
import asyncio
from pathlib import Path
from typing import Callable, Optional
from nicegui import ui

from config import THEME_PRIMARY, THEME_SECONDARY, THEME_TEXT_DIM

from app.state import PS
from ..config import DEFAULT_INPUT_DIR, PIPELINE_OUTPUTS
from ..utils import get_run_dirs, create_new_run, scan_input_directory, pipeline_log


class IOConfigPanel:
    """
    Input/Output configuration panel.
    
    Responsibilities:
    - Input directory selection and scanning
    - Run directory creation and selection
    - Persisting selections to PS state
    """
    
    def __init__(self):
        """Initialize IO config panel."""
        # Use mutable list for closure pattern
        self._current_run_dir = [PS.selected_run]
        
        # UI elements
        self._input_dir_field = None
        self._run_label = None
        self._run_select = None
    
    @property
    def input_dir(self) -> Path:
        """Get current input directory from PS."""
        if PS.input_dir:
            return Path(PS.input_dir)
        return DEFAULT_INPUT_DIR
    
    @property
    def run_dir(self) -> Optional[Path]:
        """Get current run directory."""
        return self._current_run_dir[0]
    
    def render(self) -> None:
        """Render the IO configuration panel."""
        with ui.card().classes('dark-card p-4 w-full').style(f'border: 1px solid {THEME_PRIMARY};'):
            ui.label('// INPUT_OUTPUT_DIRS').classes('terminal-header')
            
            self._render_input_section()
            ui.separator().classes('my-2')
            self._render_output_section()
    
    def _render_input_section(self) -> None:
        """Render input directory section with value from PS."""
        with ui.row().classes('items-center gap-2 mt-2 w-full'):
            ui.label('INPUT:').style(
                f'color:{THEME_SECONDARY}; font-family: JetBrains Mono; '
                f'font-size: 0.75rem; min-width: 60px;'
            )
            # Initialize from PS state if available, otherwise use default
            initial_value = str(PS.input_dir) if PS.input_dir else str(DEFAULT_INPUT_DIR)
            self._input_dir_field = ui.input(value=initial_value).props('dense').classes('flex-1')
            
            # Update PS when input changes - use args for the new value
            def update_input_dir(e):
                PS.input_dir = e.args if e.args else None
            self._input_dir_field.on('update:model-value', update_input_dir)
            
            ui.button(icon='search', on_click=self._scan_input_dir).props('flat dense size=sm')
        
        ui.label('Directorio con DMT/, EC/, EO/ o archivos .set con condición en nombre').style(
            f'color:{THEME_TEXT_DIM}; font-size: 0.65rem; margin-left: 68px;'
        )
    
    def _render_output_section(self) -> None:
        """Render output directory section."""
        with ui.row().classes('items-center gap-2 w-full'):
            ui.label('OUTPUT:').style(
                f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; '
                f'font-size: 0.75rem; min-width: 60px;'
            )
            
            self._run_label = ui.label('(crear NEW RUN)').style(
                f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.8rem;'
            )
            
            # Restore label if we have a persisted run
            if PS.selected_run:
                self._refresh_run_label()
            
            ui.button('NEW RUN', on_click=self._new_run, icon='add').props('dense').style(
                f'background:{THEME_PRIMARY}; color:black;'
            )
            
            # Existing runs dropdown
            existing_runs = get_run_dirs()
            if existing_runs:
                initial_run = PS.selected_run.name if PS.selected_run and PS.selected_run.name in existing_runs else None
                self._run_select = ui.select(
                    existing_runs, value=initial_run, label='continuar:'
                ).props('dense').classes('w-36')
                self._run_select.on('update:model-value', lambda e: self._use_existing())
        
        ui.label(f'Output: {PIPELINE_OUTPUTS}/run_* (no pisa fwd-inv-stc/)').style(
            f'color:{THEME_TEXT_DIM}; font-size: 0.65rem; margin-left: 68px;'
        )
    
    def _scan_input_dir(self) -> None:
        """Scan input directory, show results, and extract metadata."""
        p = Path(self._input_dir_field.value)
        result = scan_input_directory(p)
        
        if result['structure'] == 'subdirs':
            ui.notify(
                f'Found: {result["conditions"]}, {result["total"]} .set files',
                type='info'
            )
            pipeline_log(f"[INPUT] Scanned {p}: {result['conditions']}, {result['total']} .set files")
        elif result['structure'] == 'flat':
            counts = result['counts']
            ui.notify(
                f'Flat: DMT={counts.get("DMT", 0)}, EC={counts.get("EC", 0)}, EO={counts.get("EO", 0)} .set files',
                type='info'
            )
            pipeline_log(f"[INPUT] Scanned {p}: Flat - DMT={counts.get('DMT', 0)}, EC={counts.get('EC', 0)}, EO={counts.get('EO', 0)} .set files")
        else:
            ui.notify('Directory not found', type='warning')
            return
        
        # Log metadata if extracted
        metadata = result.get('metadata')
        if metadata:
            pipeline_log(f"[INPUT] Dataset metadata:")
            pipeline_log(f"        Frecuencia: {metadata.get('sfreq', '?')} Hz")
            pipeline_log(f"        Canales: {metadata.get('n_channels', '?')}")
            pipeline_log(f"        Duración época: {metadata.get('epoch_duration', '?')} seg")
            pipeline_log(f"        Muestras/época: {metadata.get('n_times', '?')}")
            ui.notify(
                f"Dataset: {metadata.get('sfreq', '?')}Hz, {metadata.get('n_channels', '?')}ch, {metadata.get('epoch_duration', '?')}s/época",
                type='positive'
            )
    
    async def _new_run(self) -> None:
        """Create a new run directory."""
        run_dir = await asyncio.get_event_loop().run_in_executor(None, create_new_run)
        self._current_run_dir[0] = run_dir
        PS.selected_run = run_dir
        self._refresh_run_label()
        
        if PS.refresh_files:
            PS.refresh_files()
        
        ui.notify(f'Nuevo run: {run_dir.name}', type='positive')
        pipeline_log(f"[RUN] Created: {run_dir}")
    
    def _use_existing(self) -> None:
        """Use an existing run directory."""
        if self._run_select and self._run_select.value:
            self._current_run_dir[0] = PIPELINE_OUTPUTS / self._run_select.value
            PS.selected_run = self._current_run_dir[0]
            self._refresh_run_label()
            
            if PS.refresh_files:
                PS.refresh_files()
            
            pipeline_log(f"[RUN] Using existing: {self._current_run_dir[0]}")
    
    def _refresh_run_label(self) -> None:
        """Update the run label with current selection."""
        if self._current_run_dir[0]:
            self._run_label.text = str(self._current_run_dir[0].name)
            self._run_label.style(f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.8rem;')
        else:
            self._run_label.text = '(crear NEW RUN)'
            self._run_label.style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.8rem;')

