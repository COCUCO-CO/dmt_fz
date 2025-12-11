"""
Base class for pipeline steps.

Implements the Template Method pattern:
- Defines the skeleton of step execution
- Subclasses implement specific argument building
- UI rendering can be overridden for custom controls

Interface Segregation: Only defines what's necessary for step execution.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from nicegui import ui

from config import THEME_TEXT_DIM, THEME_PRIMARY


@dataclass
class StepContext:
    """
    Context data passed to steps for argument building and execution.
    
    Dependency Inversion: Steps depend on this abstraction, not concrete UI elements.
    """
    # Directories
    run_dir: Optional[Path]
    input_dir: Path
    
    # Global parameters
    conditions: list[str]
    max_subjects: int
    max_epochs: int
    workers: int
    jobs: int
    
    # Clustering-specific (optional)
    bands: list[str] = None
    min_k: int = 2
    max_k: int = 15
    min_comps: int = 2
    max_comps: int = 10
    search_mode: str = 'Quick'


class BasePipelineStep(ABC):
    """
    Abstract base class for pipeline steps.
    
    Each step must define:
    - script_name: The Python script to execute
    - display_name: Human-readable name
    - title: Card header text
    - description: Brief description
    - output_pattern: Expected output pattern
    - color: Button color
    - estimated_time: Human-readable time estimate
    
    Subclasses implement:
    - build_args(): Construct command-line arguments
    - render_extra_controls() (optional): Additional UI controls
    """
    
    # Class attributes to be defined by subclasses
    script_name: str = ""
    display_name: str = ""
    title: str = ""
    description: str = ""
    output_pattern: str = ""
    color: str = "#00ff88"
    estimated_time: str = ""
    
    def __init__(self, context_getter: Callable[[], StepContext]):
        """
        Initialize step with a context getter function.
        
        Args:
            context_getter: Function that returns current StepContext
        """
        self._get_context = context_getter
    
    @property
    def context(self) -> StepContext:
        """Get current context (always fresh)."""
        return self._get_context()
    
    @abstractmethod
    def build_args(self) -> list[str]:
        """
        Build command-line arguments for this step.
        
        Returns:
            List of command-line arguments
        """
        pass
    
    def validate(self) -> tuple[bool, str]:
        """
        Validate that the step can be executed.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.context.run_dir:
            return False, "Primero creá un NEW RUN"
        return True, ""
    
    def render_extra_controls(self, container) -> None:
        """
        Render additional controls specific to this step.
        
        Override in subclasses that need extra UI elements.
        
        Args:
            container: NiceGUI container to render into
        """
        pass
    
    def render(self, on_run: Callable) -> None:
        """
        Render the step card UI.
        
        Template Method: Defines the structure, subclasses customize content.
        
        Args:
            on_run: Async callback to execute when RUN is clicked
        """
        with ui.card().classes('dark-card p-4 w-full'):
            ui.label(self.title).classes('terminal-header')
            ui.label(self.description).style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
            ui.label(self.output_pattern).style(f'color:{THEME_PRIMARY}; font-size: 0.65rem;')
            
            # Allow subclasses to add extra controls
            extra_container = ui.column().classes('w-full')
            self.render_extra_controls(extra_container)
            
            with ui.row().classes('gap-2 mt-3'):
                async def run_step():
                    is_valid, error_msg = self.validate()
                    if not is_valid:
                        ui.notify(error_msg, type='warning')
                        return
                    await on_run(
                        self.script_name,
                        self.build_args(),
                        self.display_name
                    )
                
                ui.button(
                    f'RUN {self.script_name}',
                    on_click=run_step,
                    icon='play_arrow'
                ).props('dense').style(f'background:{self.color}; color:black;')
                
                ui.label(self.estimated_time).style(
                    f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;'
                )

