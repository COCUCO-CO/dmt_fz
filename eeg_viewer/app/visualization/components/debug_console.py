"""
Debug Console Component - Global debug/stdout viewer for all pages.

Features:
- Captures stdout/stderr from backend
- Real-time updates without page refresh
- Resizable console height
- Persists logs across page navigation
- Global TRAINING badge that shows on all pages

Usage:
    from app.visualization.components.debug_console import render_debug_toggle, render_debug_console, debug_log, render_training_badge
    
    # In page header (before nav buttons):
    render_debug_toggle()
    render_training_badge()  # Shows pulsing badge when training is active
    
    # At end of page function:
    render_debug_console()
    
    # To log messages:
    debug_log("Processing started...", 'info', 'pipeline')
"""
import sys
from nicegui import ui
from app.state import DS, MS
from config import (
    THEME_BG, THEME_CARD, THEME_BORDER, THEME_PRIMARY, 
    THEME_SECONDARY, THEME_WARN, THEME_ERROR, THEME_TEXT, THEME_TEXT_DIM
)


class StdoutCapture:
    """Capture stdout and send to debug console."""
    
    def __init__(self, original_stdout):
        self.original = original_stdout
    
    def write(self, text):
        # Write to original stdout
        self.original.write(text)
        
        # Also capture for debug console if enabled
        if DS.enabled and text.strip():
            for line in text.strip().split('\n'):
                if line.strip():
                    DS.add_log(line.strip(), 'info', 'stdout')
    
    def flush(self):
        self.original.flush()


class StderrCapture:
    """Capture stderr and send to debug console."""
    
    def __init__(self, original_stderr):
        self.original = original_stderr
    
    def write(self, text):
        # Write to original stderr
        self.original.write(text)
        
        # Also capture for debug console if enabled
        if DS.enabled and text.strip():
            for line in text.strip().split('\n'):
                if line.strip():
                    DS.add_log(line.strip(), 'warning', 'stderr')
    
    def flush(self):
        self.original.flush()


# Store original streams
_original_stdout = None
_original_stderr = None
_capture_installed = False


def install_stdout_capture():
    """Install stdout/stderr capture. Safe to call multiple times."""
    global _original_stdout, _original_stderr, _capture_installed
    
    if _capture_installed:
        return
    
    _original_stdout = sys.stdout
    _original_stderr = sys.stderr
    sys.stdout = StdoutCapture(_original_stdout)
    sys.stderr = StderrCapture(_original_stderr)
    _capture_installed = True


def debug_log(msg: str, msg_type: str = 'info', source: str = ''):
    """
    Add message to debug history. Safe to call anytime.
    
    Args:
        msg: Message to display
        msg_type: 'info', 'success', 'warning', 'error'
        source: Origin identifier (e.g., 'pipeline', 'model', 'analysis')
    """
    DS.add_log(msg, msg_type, source)


def render_debug_toggle():
    """
    Render debug toggle button. Call inside header row.
    Returns the button element.
    """
    def toggle():
        DS.enabled = not DS.enabled
        
        # Install stdout capture when enabling
        if DS.enabled:
            install_stdout_capture()
            DS.last_rendered_count = 0  # Reset to show all messages
        
        # Notify user and trigger page refresh to show/hide console
        ui.notify(
            f"Debug mode: {'ON' if DS.enabled else 'OFF'}", 
            type='positive' if DS.enabled else 'info',
            position='bottom-right'
        )
        # Refresh the page to show/hide the console
        ui.navigate.to(ui.context.client.page.path)
    
    icon = 'bug_report' if DS.enabled else 'code_off'
    color = THEME_WARN if DS.enabled else THEME_TEXT_DIM
    
    btn = ui.button(
        icon=icon,
        on_click=toggle
    ).props('flat dense size=sm').style(
        f'color: {color};'
    ).tooltip('Toggle Debug Console')
    
    return btn


def render_training_badge():
    """
    Render a pulsing TRAINING badge that shows when model is training.
    This is global - shows on all pages when MS.training is True.
    Call inside header row after render_debug_toggle().
    """
    # Container that updates based on training state
    badge_container = ui.element('div').classes('ml-2')
    
    def update_badge():
        badge_container.clear()
        if MS.training:
            with badge_container:
                with ui.element('div').classes('flex items-center gap-1').style(
                    'animation: pulse 1.5s ease-in-out infinite;'
                ):
                    ui.icon('circle', size='xs').style('color: #ff5555;')
                    ui.label('TRAINING...').style(
                        f'color: {THEME_WARN}; font-family: JetBrains Mono; '
                        f'font-size: 0.7rem; font-weight: bold; letter-spacing: 1px;'
                    )
                # Add CSS animation
                ui.add_head_html('''
                    <style>
                        @keyframes pulse {
                            0%, 100% { opacity: 1; }
                            50% { opacity: 0.5; }
                        }
                    </style>
                ''')
    
    # Initial update
    update_badge()
    
    # Timer to check training state periodically
    ui.timer(1.0, update_badge)
    
    return badge_container


# Colors for different message types
_MSG_COLORS = {
    'info': THEME_TEXT,
    'success': THEME_PRIMARY,
    'warning': THEME_WARN,
    'error': THEME_ERROR
}

_SOURCE_COLORS = {
    'pipeline': THEME_PRIMARY,
    'model': '#f472b6',
    'analysis': THEME_WARN,
    'system': THEME_SECONDARY,
    'stdout': '#00d4ff',
    'stderr': THEME_ERROR
}


def render_debug_console():
    """
    Render debug console at bottom of page.
    Call at the END of each page function.
    Uses real-time updates via timer without page refresh.
    """
    if not DS.enabled:
        return  # Don't render if disabled
    
    # Install capture if not already done
    install_stdout_capture()
    
    # Get current height from state
    current_height = DS.console_height
    
    # Add padding to page content using JavaScript (works with any layout)
    ui.run_javascript(f'''
        document.body.style.paddingBottom = '{current_height + 20}px';
        const containers = document.querySelectorAll('.q-page, .nicegui-content, main');
        containers.forEach(el => {{
            el.style.paddingBottom = '{current_height + 20}px';
        }});
    ''')
    
    # Create console panel - fixed at bottom
    with ui.element('div').style(
        f'position: fixed; bottom: 0; left: 0; right: 0; '
        f'background: {THEME_BG}; border-top: 2px solid {THEME_WARN}; '
        f'z-index: 1000; height: {current_height}px; display: flex; flex-direction: column;'
    ):
        # Header row with controls
        with ui.row().classes('items-center gap-2 px-3 py-1').style(
            f'background: {THEME_CARD}; border-bottom: 1px solid {THEME_BORDER}; flex-shrink: 0;'
        ):
            ui.label('// DEBUG_CONSOLE').style(
                f'color: {THEME_WARN}; font-family: JetBrains Mono; font-size: 0.7rem; letter-spacing: 1px;'
            )
            
            # Height controls
            def decrease_height():
                DS.console_height = max(100, DS.console_height - 50)
                ui.navigate.to(ui.context.client.page.path)
            
            def increase_height():
                DS.console_height = min(500, DS.console_height + 50)
                ui.navigate.to(ui.context.client.page.path)
            
            ui.button(icon='expand_more', on_click=decrease_height).props(
                'flat dense size=xs round'
            ).style(f'color: {THEME_TEXT_DIM};').tooltip('Decrease height')
            
            ui.label(f'{current_height}px').style(
                f'color: {THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.6rem; min-width: 45px; text-align: center;'
            )
            
            ui.button(icon='expand_less', on_click=increase_height).props(
                'flat dense size=xs round'
            ).style(f'color: {THEME_TEXT_DIM};').tooltip('Increase height')
            
            # Spacer
            ui.element('div').classes('flex-1')
            
            # Message count - will be updated by timer
            msg_count_label = ui.label(f'{len(DS.log_history)} messages').style(
                f'color: {THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.65rem;'
            )
            
            # Clear button
            def clear_logs():
                DS.log_history = []
                DS.last_rendered_count = 0
                log_container.clear()
                with log_container:
                    ui.label('Console cleared.').style(
                        f'color: {THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.7rem;'
                    )
                msg_count_label.text = '0 messages'
            
            ui.button('CLEAR', on_click=clear_logs, icon='delete').props('flat dense size=sm').style(
                f'color: {THEME_TEXT_DIM};'
            )
            
            # Close button
            def close_console():
                DS.enabled = False
                ui.run_javascript('''
                    document.body.style.paddingBottom = '';
                    const containers = document.querySelectorAll('.q-page, .nicegui-content, main');
                    containers.forEach(el => { el.style.paddingBottom = ''; });
                ''')
                ui.notify('Debug mode OFF', type='info', position='bottom-right')
                ui.navigate.to(ui.context.client.page.path)
            
            ui.button(icon='close', on_click=close_console).props('flat dense size=sm').style(
                f'color: {THEME_TEXT_DIM};'
            )
        
        # Log content area with unique ID for JavaScript targeting
        content_height = current_height - 35
        scroll_area = ui.scroll_area().style(
            f'flex: 1; background: #050505; height: {content_height}px;'
        ).props('id="debug-console-scroll"')
        
        with scroll_area:
            log_container = ui.column().classes('p-2 gap-0').props('id="debug-console-logs"')
        
        async def scroll_to_bottom():
            """Scroll to bottom using JavaScript for reliability."""
            await ui.run_javascript('''
                const container = document.getElementById('debug-console-scroll');
                if (container) {
                    const scrollable = container.querySelector('.q-scrollarea__container');
                    if (scrollable) {
                        scrollable.scrollTop = scrollable.scrollHeight;
                    }
                }
            ''')
        
        def render_single_message(entry):
            """Render a single log entry."""
            with ui.row().classes('gap-2 items-baseline').style('flex-wrap: nowrap;'):
                # Timestamp
                ui.label(entry['time']).style(
                    f'color: {THEME_TEXT_DIM}; font-family: JetBrains Mono; '
                    f'font-size: 0.6rem; min-width: 55px; flex-shrink: 0;'
                )
                # Source badge
                source = entry.get('source', '')
                if source:
                    src_color = _SOURCE_COLORS.get(source, THEME_TEXT_DIM)
                    ui.label(f"[{source.upper()}]").style(
                        f'color: {src_color}; font-family: JetBrains Mono; '
                        f'font-size: 0.6rem; min-width: 65px; flex-shrink: 0;'
                    )
                # Message
                msg_color = _MSG_COLORS.get(entry.get('type', 'info'), THEME_TEXT)
                ui.label(entry['msg']).style(
                    f'color: {msg_color}; font-family: JetBrains Mono; font-size: 0.7rem; '
                    f'word-break: break-word;'
                )
        
        def render_messages():
            """Render all current messages."""
            log_container.clear()
            with log_container:
                if not DS.log_history:
                    ui.label('Debug console ready. Stdout/stderr will appear here.').style(
                        f'color: {THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.7rem;'
                    )
                else:
                    # Render last 200 messages
                    for entry in DS.log_history[-200:]:
                        render_single_message(entry)
            DS.mark_rendered()
            # Scroll to bottom after content is rendered
            ui.timer(0.15, scroll_to_bottom, once=True)
        
        def append_new_messages():
            """Append only new messages (incremental update)."""
            if not DS.has_new_messages():
                return
            
            # Get new messages since last render
            new_start = DS.last_rendered_count
            new_messages = DS.log_history[new_start:]
            
            # Append new messages
            with log_container:
                for entry in new_messages:
                    render_single_message(entry)
            
            # Update count and scroll
            msg_count_label.text = f'{len(DS.log_history)} messages'
            DS.mark_rendered()
            
            # Scroll to bottom after DOM update
            ui.timer(0.05, scroll_to_bottom, once=True)
        
        # Initial render
        render_messages()
        msg_count_label.text = f'{len(DS.log_history)} messages'
        
        # Timer for real-time updates (check every 500ms)
        ui.timer(0.5, append_new_messages)

