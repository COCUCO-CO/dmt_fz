"""
Global parameters panel component.

Controls for conditions, subjects, epochs, workers, and jobs.
"""
from typing import List
from nicegui import ui

from config import THEME_TEXT_DIM

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
        """Get selected conditions."""
        conds = []
        if self._cond_dmt and self._cond_dmt.value:
            conds.append('DMT')
        if self._cond_ec and self._cond_ec.value:
            conds.append('EC')
        if self._cond_eo and self._cond_eo.value:
            conds.append('EO')
        return conds
    
    @property
    def max_subjects(self) -> int:
        """Get max subjects (0 = all)."""
        return int(self._max_subj.value or 0) if self._max_subj else DEFAULT_MAX_SUBJECTS
    
    @property
    def max_epochs(self) -> int:
        """Get max epochs (0 = all)."""
        return int(self._max_epochs.value or 0) if self._max_epochs else DEFAULT_MAX_EPOCHS
    
    @property
    def workers(self) -> int:
        """Get worker count."""
        return int(self._workers.value or DEFAULT_WORKERS) if self._workers else DEFAULT_WORKERS
    
    @property
    def jobs(self) -> int:
        """Get job count (0 = auto)."""
        return int(self._jobs.value or 0) if self._jobs else DEFAULT_JOBS
    
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
        """Render conditions row."""
        ui.label('Conditions').style(
            f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.75rem;'
        )
        with ui.row().classes('gap-3 items-center'):
            self._cond_dmt = ui.checkbox('DMT', value=True).props('dense')
            self._cond_ec = ui.checkbox('EC', value=True).props('dense')
            self._cond_eo = ui.checkbox('EO', value=True).props('dense')
    
    def _render_max_subjects(self) -> None:
        """Render max subjects row."""
        ui.label('Max subjects').style(
            f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.75rem;'
        )
        with ui.row().classes('gap-2 items-center'):
            self._max_subj = ui.number(value=DEFAULT_MAX_SUBJECTS, min=0, max=100).props('dense').classes('w-20')
            ui.label('0 = all').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
    
    def _render_max_epochs(self) -> None:
        """Render max epochs row."""
        ui.label('Max epochs').style(
            f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.75rem;'
        )
        with ui.row().classes('gap-2 items-center'):
            self._max_epochs = ui.number(value=DEFAULT_MAX_EPOCHS, min=0, max=500).props('dense').classes('w-20')
            ui.label('0 = all').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
    
    def _render_workers(self) -> None:
        """Render workers row."""
        ui.label('Workers').style(
            f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.75rem;'
        )
        with ui.row().classes('gap-2 items-center'):
            self._workers = ui.number(value=DEFAULT_WORKERS, min=1, max=32).props('dense').classes('w-20')
    
    def _render_jobs(self) -> None:
        """Render jobs row."""
        ui.label('Jobs').style(
            f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.75rem;'
        )
        with ui.row().classes('gap-2 items-center'):
            self._jobs = ui.number(value=DEFAULT_JOBS, min=0, max=64).props('dense').classes('w-20')
            ui.label('0 = auto').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')

