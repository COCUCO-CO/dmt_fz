"""
Pipeline page main orchestrator.

Composes all components, tabs, and steps into the complete pipeline page.
This follows the Composition pattern - the page is built from smaller, focused components.
"""
from pathlib import Path
from nicegui import ui

from config import THEME_BG, THEME_PRIMARY, THEME_SECONDARY, THEME_WARN
from app.visualization.styles.css import STYLE

from .config import LEFT_PANEL_WIDTH
from .runner import run_pipeline_step
from .components import render_header, IOConfigPanel, GlobalParamsPanel
from .steps import STEPS, StepContext
from .tabs import ConsoleTab, FilesTab, SystemTab, VisualizeTab


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
    Render the complete pipeline page.
    
    Layout:
    ┌─────────────────────────────────────────────────────────────┐
    │ HEADER - Title, Running Indicator, Navigation               │
    ├────────────────┬────────────────────────────────────────────┤
    │ LEFT PANEL     │ RIGHT PANEL                                │
    │ (450px fixed)  │ (flexible)                                 │
    │                │                                            │
    │ - IO Config    │ ┌──────────────────────────────────────┐  │
    │ - Global Params│ │ TABS: Console | Files | System | Viz │  │
    │ - Steps 1-8    │ ├──────────────────────────────────────┤  │
    │   (scrollable) │ │ Tab Content (scrollable)             │  │
    │                │ │                                      │  │
    │                │ │                                      │  │
    │                │ └──────────────────────────────────────┘  │
    └────────────────┴────────────────────────────────────────────┘
    """
    # Header
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
        
        # Right panel - Tabs
        _render_right_panel(lambda: io_config.run_dir)


def _render_left_panel(
    io_config: IOConfigPanel,
    global_params: GlobalParamsPanel,
    get_context,
    on_run
) -> None:
    """Render the left panel with controls and steps."""
    with ui.scroll_area().style(f'width: {LEFT_PANEL_WIDTH}; height: 100%;'):
        with ui.column().classes('gap-4 pr-2'):
            # IO Configuration
            io_config.render()
            
            # Global Parameters
            global_params.render()
            
            # Pipeline Steps
            for step_cls in STEPS:
                step = step_cls(get_context)
                step.render(on_run)


def _render_right_panel(get_run_dir) -> None:
    """Render the right panel with tabs."""
    with ui.column().classes('flex-1').style(
        'height: 100%; min-height: 0; display: flex; flex-direction: column; overflow: hidden;'
    ):
        with ui.card().classes('dark-card p-2 w-full flex-1').style(
            'display: flex; flex-direction: column; min-height: 0;'
        ):
            # Tab headers
            with ui.tabs().classes('w-full').style(f'background: {THEME_BG};') as tabs:
                tab_console = ui.tab('console', label='CONSOLE', icon='terminal').style(
                    f'color:{THEME_PRIMARY};'
                )
                tab_files = ui.tab('files', label='FILES', icon='folder').style(
                    f'color:{THEME_SECONDARY};'
                )
                tab_system = ui.tab('system', label='SYSTEM', icon='memory').style(
                    f'color:{THEME_WARN};'
                )
                tab_viz = ui.tab('visualize', label='VISUALIZE', icon='analytics').style(
                    f'color:#a78bfa;'
                )
            
            # Tab panels
            with ui.tab_panels(tabs, value=tab_console).classes('w-full').style(
                'flex: 1; min-height: 0; overflow: hidden;'
            ):
                # Console Tab
                console_tab = ConsoleTab()
                console_tab.render()
                
                # Files Tab
                files_tab = FilesTab(get_run_dir)
                files_tab.render()
                
                # System Tab
                system_tab = SystemTab()
                system_tab.render()
                
                # Visualize Tab
                viz_tab = VisualizeTab(get_run_dir)
                viz_tab.render()

