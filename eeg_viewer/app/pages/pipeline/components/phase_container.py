"""
Phase container component for grouping pipeline steps.

Provides a collapsible container that groups related steps
with a header showing phase name, description, and icon.
"""
from typing import List, Callable
from nicegui import ui

from config import THEME_BG, THEME_BORDER, THEME_TEXT_DIM

from ..constants.step_info import Phase, PhaseInfo, PHASES, get_steps_for_phase
from ..steps.base import BasePipelineStep


class PhaseContainer:
    """
    Collapsible container for a pipeline phase.
    
    Groups related steps together with a descriptive header.
    """
    
    def __init__(self, phase: Phase):
        """
        Initialize phase container.
        
        Args:
            phase: The phase to display
        """
        self.phase = phase
        self.phase_info = PHASES[phase]
        self._expanded = not self.phase_info.collapsed_default
        self._content_container = None
    
    def render(self, steps: List[BasePipelineStep], on_run: Callable) -> None:
        """
        Render the phase container with its steps.
        
        Args:
            steps: List of pipeline steps belonging to this phase
            on_run: Callback for running a step
        """
        info = self.phase_info
        
        with ui.card().classes('w-full mb-2').style(
            f'background: {THEME_BG}; border: 1px solid {info.color}40; border-radius: 8px;'
        ):
            # Header (clickable to expand/collapse)
            with ui.row().classes('items-center w-full p-3 cursor-pointer').style(
                f'border-bottom: 1px solid {THEME_BORDER};'
            ) as header:
                # Phase icon
                ui.icon(info.icon, size='sm').style(f'color: {info.color};')
                
                # Phase title and subtitle
                with ui.column().classes('gap-0 flex-1 ml-2'):
                    ui.label(info.title).style(
                        f'color: {info.color}; font-family: JetBrains Mono; '
                        f'font-size: 0.85rem; font-weight: 600; letter-spacing: 1px;'
                    )
                    ui.label(info.subtitle).style(
                        f'color: {THEME_TEXT_DIM}; font-size: 0.7rem;'
                    )
                
                # Step count badge
                ui.label(f'{len(steps)} steps').style(
                    f'color: {THEME_TEXT_DIM}; font-size: 0.65rem; '
                    f'background: {THEME_BORDER}; padding: 2px 8px; border-radius: 10px;'
                )
                
                # Expand/collapse icon
                expand_icon = ui.icon('expand_more' if self._expanded else 'chevron_right', size='sm').style(
                    f'color: {THEME_TEXT_DIM}; transition: transform 0.2s;'
                )
            
            # Content container (steps)
            self._content_container = ui.column().classes('w-full gap-3 p-3')
            
            if self._expanded:
                self._content_container.style('display: block;')
            else:
                self._content_container.style('display: none;')
            
            with self._content_container:
                for step in steps:
                    step.render(on_run)
            
            # Toggle expansion on header click
            def toggle_expand():
                self._expanded = not self._expanded
                if self._expanded:
                    self._content_container.style('display: block;')
                    expand_icon.props(remove='name=chevron_right', add='name=expand_more')
                else:
                    self._content_container.style('display: none;')
                    expand_icon.props(remove='name=expand_more', add='name=chevron_right')
            
            header.on('click', toggle_expand)


def render_phases_with_steps(
    all_steps: List[BasePipelineStep],
    on_run: Callable
) -> None:
    """
    Render all phases with their steps.
    
    Args:
        all_steps: All pipeline steps
        on_run: Callback for running a step
    """
    # Group steps by phase using their script_name to look up phase
    from ..constants.step_info import STEP_INFO
    
    phase_steps = {phase: [] for phase in Phase}
    
    for step in all_steps:
        # Find the step info to get its phase
        for step_num, info in STEP_INFO.items():
            if info.script_name == step.script_name:
                phase_steps[info.phase].append(step)
                break
    
    # Render each phase
    for phase in Phase:
        steps = phase_steps.get(phase, [])
        if steps:
            container = PhaseContainer(phase)
            container.render(steps, on_run)

