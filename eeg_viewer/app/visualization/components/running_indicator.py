"""Global running task indicator component."""
from nicegui import ui
from app.state import PS, MS
from config import THEME_WARN

# CSS for pulsing animation
INDICATOR_CSS = """
@keyframes pulse-orange {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
.running-indicator {
    animation: pulse-orange 1s ease-in-out infinite;
    background: #f59e0b;
    color: black;
    padding: 2px 10px;
    border-radius: 4px;
    font-family: JetBrains Mono, monospace;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.running-indicator-hidden {
    display: none;
}
"""


def get_running_task() -> tuple[str, str]:
    """
    Check if any background task is running.
    Returns (task_name, source) where source is 'pipeline' or 'model'.
    Returns ('', '') if nothing is running.
    """
    if PS.running and PS.running_task_name:
        return (PS.running_task_name, 'pipeline')
    if MS.training and MS.running_task_name:
        return (MS.running_task_name, 'model')
    return ('', '')


def render_running_indicator():
    """
    Render the global running indicator that updates dynamically.
    Should be called within a header row context.
    Uses a timer to periodically check for running tasks.
    """
    # Add the pulsing CSS
    ui.add_head_html(f'<style>{INDICATOR_CSS}</style>')
    
    # Create container that will be updated
    with ui.row().classes('items-center gap-2 ml-4') as container:
        indicator_label = ui.label('').classes('running-indicator-hidden')
        source_label = ui.label('').style(f'color:{THEME_WARN}; font-size: 0.65rem; font-family: JetBrains Mono;')
    
    def update_indicator():
        """Update indicator based on current running state"""
        try:
            task_name, source = get_running_task()
            
            if task_name:
                indicator_label.text = f'⚡ {task_name}'
                indicator_label.classes(remove='running-indicator-hidden', add='running-indicator')
                source_text = 'PIPELINE' if source == 'pipeline' else 'MODEL'
                source_label.text = f'({source_text})'
            else:
                indicator_label.text = ''
                indicator_label.classes(remove='running-indicator', add='running-indicator-hidden')
                source_label.text = ''
        except RuntimeError:
            # Client disconnected, timer will be cleaned up automatically
            pass
    
    # Initial update
    update_indicator()
    
    # Timer to check for updates every second (active=True ensures it runs)
    ui.timer(1.0, update_indicator, active=True)
    
    return container

