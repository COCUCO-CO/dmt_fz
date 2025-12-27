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

from app.state import PS
from ..constants.step_info import get_step_by_script
from ..utils import format_dataset_metadata


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
    - display_name: Human-readable name (auto-loaded from step_info)
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
        # Load enhanced metadata from step_info if available
        self._step_info = get_step_by_script(self.script_name)
    
    @property
    def context(self) -> StepContext:
        """Get current context (always fresh)."""
        return self._get_context()
    
    @property
    def info(self):
        """Get step info metadata."""
        return self._step_info
    
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
        Render the step card UI with enhanced metadata.
        
        Template Method: Defines the structure, subclasses customize content.
        Uses step_info metadata for user-friendly display.
        
        Args:
            on_run: Async callback to execute when RUN is clicked
        """
        info = self._step_info
        
        with ui.card().classes('dark-card p-4 w-full'):
            # Header with step number and name
            with ui.row().classes('items-center gap-2 w-full'):
                if info:
                    # Step number badge
                    ui.label(f'{info.step_number}').style(
                        f'background: {self.color}; color: black; '
                        f'padding: 2px 8px; border-radius: 4px; '
                        f'font-family: JetBrains Mono; font-size: 0.75rem; font-weight: bold;'
                    )
                    # Display name
                    ui.label(info.display_name).style(
                        f'color: {self.color}; font-family: JetBrains Mono; '
                        f'font-size: 0.9rem; font-weight: 600;'
                    )
                    # Help button with dialog
                    with ui.element('div').classes('ml-auto'):
                        def show_help_dialog(step_info=info):
                            with ui.dialog() as dialog, ui.card().classes('w-full max-w-4xl'):
                                # Header
                                with ui.row().classes('items-center w-full mb-2'):
                                    ui.label(f'Step {step_info.step_number}: {step_info.display_name}').style(
                                        f'color: {self.color}; font-size: 1.1rem; font-weight: bold;'
                                    )
                                    ui.space()
                                    ui.button(icon='close', on_click=dialog.close).props('flat round dense')
                                
                                ui.separator()
                                
                                # Description
                                ui.label('📋 Descripción').style('font-weight: bold; margin-top: 8px;')
                                ui.label(step_info.detailed_description).style(
                                    'font-size: 0.85rem; white-space: pre-wrap; color: #ccc;'
                                )
                                
                                # Dataset metadata section (if available and step 1)
                                if step_info.step_number == 1 and PS.dataset_metadata:
                                    ui.separator().classes('my-3')
                                    ui.label('📊 DATOS DEL DATASET (detectado)').style(
                                        'font-weight: bold; color: #60a5fa;'
                                    )
                                    ui.label(format_dataset_metadata(PS.dataset_metadata)).style(
                                        'font-family: JetBrains Mono; font-size: 0.75rem; '
                                        'white-space: pre-wrap; background: #0a1929; '
                                        'padding: 10px; border-radius: 4px; color: #60a5fa; '
                                        'border: 1px solid #1e3a5f;'
                                    )
                                elif step_info.step_number == 1:
                                    ui.separator().classes('my-3')
                                    ui.label('📊 DATOS DEL DATASET').style(
                                        'font-weight: bold; color: #60a5fa;'
                                    )
                                    ui.label('⚠️ Escaneá la carpeta INPUT para ver los datos reales del dataset').style(
                                        'font-size: 0.75rem; color: #888; font-style: italic;'
                                    )
                                
                                # Two column layout for input/output
                                with ui.row().classes('w-full gap-4 mt-4'):
                                    # Input structure
                                    with ui.column().classes('flex-1'):
                                        ui.label('📥 ESTRUCTURA ENTRADA').style(
                                            'font-weight: bold; color: #f59e0b;'
                                        )
                                        ui.label(step_info.input_structure).style(
                                            'font-family: JetBrains Mono; font-size: 0.7rem; '
                                            'white-space: pre-wrap; background: #1a1a1a; '
                                            'padding: 8px; border-radius: 4px; color: #aaa;'
                                        )
                                    
                                    # Output structure
                                    with ui.column().classes('flex-1'):
                                        ui.label('📤 ESTRUCTURA SALIDA').style(
                                            'font-weight: bold; color: #00d4aa;'
                                        )
                                        ui.label(step_info.output_structure).style(
                                            'font-family: JetBrains Mono; font-size: 0.7rem; '
                                            'white-space: pre-wrap; background: #1a1a1a; '
                                            'padding: 8px; border-radius: 4px; color: #aaa;'
                                        )
                                
                                # Footer with time info
                                with ui.row().classes('mt-4 gap-4'):
                                    ui.label(f'⏱️ Tiempo: {step_info.time_estimate}').style(
                                        'font-size: 0.8rem; color: #888;'
                                    )
                                    if step_info.requires_steps:
                                        req = ', '.join(str(s) for s in step_info.requires_steps)
                                        ui.label(f'⚠️ Requiere: Step {req}').style(
                                            'font-size: 0.8rem; color: #f59e0b;'
                                        )
                            
                            dialog.open()
                        
                        ui.button(icon='help_outline', on_click=show_help_dialog).props(
                            'flat dense round size=sm'
                        ).style('color: #666;')
                else:
                    ui.label(self.title).classes('terminal-header')
            
            # Short description
            if info:
                ui.label(info.short_description).style(
                    f'color: {THEME_TEXT_DIM}; font-size: 0.75rem; margin-top: 4px;'
                )
            else:
                ui.label(self.description).style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
            
            # Output info with icon
            with ui.row().classes('items-center gap-1 mt-1'):
                ui.icon('folder_open', size='xs').style(f'color: {THEME_PRIMARY};')
                if info:
                    ui.label(info.output_example).style(
                        f'color: {THEME_PRIMARY}; font-size: 0.65rem; font-family: JetBrains Mono;'
                    )
                else:
                    ui.label(self.output_pattern).style(f'color:{THEME_PRIMARY}; font-size: 0.65rem;')
            
            # Allow subclasses to add extra controls
            extra_container = ui.column().classes('w-full')
            self.render_extra_controls(extra_container)
            
            # Action row
            with ui.row().classes('gap-3 mt-3 items-center'):
                async def run_step():
                    is_valid, error_msg = self.validate()
                    if not is_valid:
                        ui.notify(error_msg, type='warning')
                        return
                    await on_run(
                        self.script_name,
                        self.build_args(),
                        info.display_name if info else self.display_name
                    )
                
                # Enhanced button with icon from metadata
                button_icon = info.button_icon if info else 'play_arrow'
                button_text = info.button_text if info else f'RUN {self.script_name}'
                
                ui.button(
                    button_text,
                    on_click=run_step,
                    icon=button_icon
                ).props('dense').style(f'background:{self.color}; color:black;')
                
                # Time estimate with icon
                with ui.row().classes('items-center gap-1'):
                    ui.icon('schedule', size='xs').style(f'color: {THEME_TEXT_DIM};')
                    time_text = info.time_estimate if info else self.estimated_time
                    ui.label(time_text).style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')

