"""
Global parameters panel component.

Controls for conditions, subjects, epochs, workers, and jobs.
Parameters persist in PS (PipelineState) across page navigations.
"""
from typing import List
from nicegui import ui

from config import THEME_TEXT_DIM

from app.state import PS
from ..config import (
    DEFAULT_MAX_SUBJECTS,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_WORKERS,
    DEFAULT_JOBS,
)


class GlobalParamsPanel:
    """
    Global parameters panel.
    
    Responsibilities:
    - Condition checkboxes (DMT, EC, EO)
    - Subject and epoch limits
    - Worker and job counts
    - Persist all values in PS for tab navigation
    """
    
    def __init__(self):
        """Initialize global params panel."""
        # Condition checkboxes
        self._cond_dmt = None
        self._cond_ec = None
        self._cond_eo = None
        
        # Numeric inputs
        self._max_subj = None
        self._max_epochs = None
        self._workers = None
        self._jobs = None
    
    @property
    def conditions(self) -> List[str]:
        """Get selected conditions from PS."""
        return PS.conditions
    
    @property
    def max_subjects(self) -> int:
        """Get max subjects from PS."""
        return PS.max_subjects
    
    @property
    def max_epochs(self) -> int:
        """Get max epochs from PS."""
        return PS.max_epochs
    
    @property
    def workers(self) -> int:
        """Get worker count from PS."""
        return PS.workers
    
    @property
    def jobs(self) -> int:
        """Get job count from PS."""
        return PS.jobs
    
    def render(self) -> None:
        """Render the global parameters panel."""
        with ui.card().classes('dark-card p-4 w-full'):
            ui.label('// GLOBAL_PARAMS').classes('terminal-header')
            
            # Use CSS grid for aligned parameters
            with ui.element('div').classes('w-full').style(
                'display: grid; grid-template-columns: 100px 1fr; gap: 8px 12px; align-items: center;'
            ):
                self._render_conditions()
                self._render_max_subjects()
                self._render_max_epochs()
                self._render_workers()
                self._render_jobs()
    
    def _render_conditions(self) -> None:
        """Render conditions row with values from PS."""
        ui.label('Conditions').style(
            f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.75rem;'
        )
        with ui.row().classes('gap-3 items-center'):
            # Initialize from PS state
            self._cond_dmt = ui.checkbox('DMT', value='DMT' in PS.conditions).props('dense')
            self._cond_ec = ui.checkbox('EC', value='EC' in PS.conditions).props('dense')
            self._cond_eo = ui.checkbox('EO', value='EO' in PS.conditions).props('dense')
            
            # Update PS on change
            def update_conditions():
                conds = []
                if self._cond_dmt.value:
                    conds.append('DMT')
                if self._cond_ec.value:
                    conds.append('EC')
                if self._cond_eo.value:
                    conds.append('EO')
                PS.conditions = conds
            
            self._cond_dmt.on('update:model-value', lambda: update_conditions())
            self._cond_ec.on('update:model-value', lambda: update_conditions())
            self._cond_eo.on('update:model-value', lambda: update_conditions())
    
    def _render_max_subjects(self) -> None:
        """Render max subjects row with value from PS."""
        ui.label('Max subjects').style(
            f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.75rem;'
        )
        with ui.row().classes('gap-2 items-center'):
            # Initialize from PS state
            self._max_subj = ui.number(value=PS.max_subjects, min=0, max=100).props('dense').classes('w-20')
            ui.label('0 = all').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
            
            # Update PS on change - use args[0] for the new value
            self._max_subj.on('update:model-value', lambda e: setattr(PS, 'max_subjects', int(e.args if e.args else 0)))
    
    def _render_max_epochs(self) -> None:
        """Render max epochs row with value from PS."""
        ui.label('Max epochs').style(
            f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.75rem;'
        )
        with ui.row().classes('gap-2 items-center'):
            # Initialize from PS state
            self._max_epochs = ui.number(value=PS.max_epochs, min=0, max=500).props('dense').classes('w-20')
            ui.label('0 = all').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
            
            # Update PS on change - use args for the new value
            self._max_epochs.on('update:model-value', lambda e: setattr(PS, 'max_epochs', int(e.args if e.args else 0)))
    
    def _render_workers(self) -> None:
        """Render workers row with value from PS."""
        ui.label('Workers').style(
            f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.75rem;'
        )
        with ui.row().classes('gap-2 items-center'):
            # Initialize from PS state
            self._workers = ui.number(value=PS.workers, min=1, max=32).props('dense').classes('w-20')
            
            # Update PS on change - use args for the new value
            self._workers.on('update:model-value', lambda e: setattr(PS, 'workers', int(e.args if e.args else DEFAULT_WORKERS)))
    
    def _render_jobs(self) -> None:
        """Render jobs row with value from PS."""
        ui.label('Jobs').style(
            f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.75rem;'
        )
        with ui.row().classes('gap-2 items-center'):
            # Initialize from PS state
            self._jobs = ui.number(value=PS.jobs, min=0, max=64).props('dense').classes('w-20')
            ui.label('0 = auto').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
            
            # Update PS on change - use args for the new value
            self._jobs.on('update:model-value', lambda e: setattr(PS, 'jobs', int(e.args if e.args else 0)))

