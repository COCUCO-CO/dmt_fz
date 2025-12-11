"""
Pipeline page main orchestrator.

Composes all components, tabs, and steps into the complete pipeline page.
This follows the Composition pattern - the page is built from smaller, focused components.
"""
from pathlib import Path
from nicegui import ui

from config import THEME_BG, THEME_PRIMARY, THEME_SECONDARY, THEME_WARN, THEME_BORDER
from app.visualization.styles.css import STYLE

from .config import LEFT_PANEL_WIDTH
from .runner import run_pipeline_step
from .components import render_header, IOConfigPanel, GlobalParamsPanel, render_phases_with_steps, FileBrowserPanel
from .steps import STEPS, StepContext
from .tabs import ConsoleTab


def pipeline_page_route():
    """Route decorator target for /pipeline."""
    ui.add_head_html(f'<style>{STYLE}</style>')
    _render_pipeline_page()


@ui.page('/pipeline')
def pipeline_page():
    """Main pipeline page entry point."""
    ui.add_head_html(f'<style>{STYLE}</style>')
    _render_pipeline_page()


def _render_pipeline_page() -> None:
    """
    Render the complete pipeline page with improved UX.
    
    Layout:
    ┌─────────────────────────────────────────────────────────────┐
    │ HEADER - Title, Running Indicator, CPU/RAM/GPU, Navigation  │
    ├────────────────┬────────────────────────────────────────────┤
    │ LEFT PANEL     │ RIGHT PANEL                                │
    │ (450px fixed)  │ (flexible)                                 │
    │                │                                            │
    │ - IO Config    │ ┌──────────────────────────────────────┐  │
    │ - Global Params│ │ FILE BROWSER (auto-refresh)          │  │
    │                │ ├──────────────────────────────────────┤  │
    │ PHASES:        │ │ CONSOLE (output log)                 │  │
    │ ┌────────────┐ │ │                                      │  │
    │ │ FASE 1     │ │ │                                      │  │
    │ │ Steps 1-2  │ │ │                                      │  │
    │ ├────────────┤ │ │                                      │  │
    │ │ FASE 2     │ │ └──────────────────────────────────────┘  │
    │ │ Steps 3-6  │ │                                            │
    │ ├────────────┤ │                                            │
    │ │ FASE 3     │ │                                            │
    │ │ Steps 7-8  │ │                                            │
    │ └────────────┘ │                                            │
    └────────────────┴────────────────────────────────────────────┘
    """
    # Header (includes system monitor now)
    render_header()
    
    # Main layout
    with ui.row().classes('w-full p-4 gap-4').style(
        'height: calc(100vh - 50px); align-items: stretch; overflow: hidden;'
    ):
        # Create shared components
        io_config = IOConfigPanel()
        global_params = GlobalParamsPanel()
        
        # Context getter for steps - captures closures
        def get_step_context() -> StepContext:
            return StepContext(
                run_dir=io_config.run_dir,
                input_dir=io_config.input_dir,
                conditions=global_params.conditions,
                max_subjects=global_params.max_subjects,
                max_epochs=global_params.max_epochs,
                workers=global_params.workers,
                jobs=global_params.jobs,
            )
        
        # Run handler for steps
        async def on_step_run(script_name: str, args: list, step_name: str):
            await run_pipeline_step(
                script_name,
                args,
                step_name,
                output_dir=io_config.run_dir,
                input_dir=io_config.input_dir
            )
        
        # Left panel - Pipeline controls
        _render_left_panel(io_config, global_params, get_step_context, on_step_run)
        
        # Right panel - File Browser + Console (no tabs)
        _render_right_panel(lambda: io_config.run_dir)


def _render_left_panel(
    io_config: IOConfigPanel,
    global_params: GlobalParamsPanel,
    get_context,
    on_run
) -> None:
    """Render the left panel with controls and steps grouped by phases."""
    with ui.scroll_area().style(f'width: {LEFT_PANEL_WIDTH}; height: 100%;'):
        with ui.column().classes('gap-4 pr-2'):
            # IO Configuration
            io_config.render()
            
            # Global Parameters
            global_params.render()
            
            # Pipeline Steps grouped by phases
            # Instantiate all steps
            all_steps = [step_cls(get_context) for step_cls in STEPS]
            
            # Render using phase containers
            render_phases_with_steps(all_steps, on_run)


def _render_right_panel(get_run_dir) -> None:
    """
    Render the right panel with File Browser and Console.
    
    No tabs - both are always visible:
    - File Browser on top (with auto-refresh)
    - Console below (shows execution output)
    """
    with ui.column().classes('flex-1 gap-2').style(
        'height: 100%; min-height: 0; display: flex; flex-direction: column; overflow: hidden;'
    ):
        # File Browser (top, fixed height)
        file_browser = FileBrowserPanel(get_run_dir)
        file_browser.render(height="180px")
        
        # Console (bottom, flexible height)
        with ui.card().classes('dark-card p-2 w-full flex-1').style(
            f'display: flex; flex-direction: column; min-height: 0; '
            f'border: 1px solid {THEME_BORDER};'
        ):
            console_tab = ConsoleTab()
            console_tab.render()

