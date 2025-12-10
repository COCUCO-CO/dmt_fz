"""
CSS styles for EEG Viewer application.

Konsole/Terminal aesthetic theme.
"""
from .theme import (
    THEME_BG, THEME_CARD, THEME_BORDER,
    THEME_PRIMARY, THEME_SECONDARY,
    THEME_TEXT, THEME_TEXT_DIM,
)


def get_app_style() -> str:
    """Generate the main application CSS."""
    return f"""
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {{
    --bg: {THEME_BG};
    --card: {THEME_CARD};
    --border: {THEME_BORDER};
    --primary: {THEME_PRIMARY};
    --secondary: {THEME_SECONDARY};
    --text: {THEME_TEXT};
    --text-dim: {THEME_TEXT_DIM};
}}

* {{ scrollbar-width: thin; scrollbar-color: var(--primary) var(--bg); }}
*::-webkit-scrollbar {{ width: 6px; height: 6px; }}
*::-webkit-scrollbar-track {{ background: var(--bg); }}
*::-webkit-scrollbar-thumb {{ background: var(--primary); border-radius: 3px; }}

body {{
    background: var(--bg) !important;
    font-family: 'JetBrains Mono', 'IBM Plex Mono', 'SF Mono', monospace !important;
    color: var(--text) !important;
}}

.nicegui-content {{ background: transparent !important; }}

.dark-card {{
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4) !important;
}}

.accent-text {{ color: var(--primary) !important; text-shadow: 0 0 10px rgba(0, 255, 136, 0.3); }}
.secondary-text {{ color: var(--secondary) !important; }}

.file-item {{
    transition: all 0.15s ease;
    cursor: pointer;
    padding: 6px 10px;
    border-radius: 2px;
    margin: 2px 0;
    border-left: 2px solid transparent;
}}
.file-item:hover {{
    background: rgba(0, 255, 136, 0.1) !important;
    border-left-color: var(--primary);
}}

.ch-btn {{
    font-size: 0.75rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    padding: 3px 8px !important;
    min-width: 42px !important;
    margin: 2px !important;
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--text-dim) !important;
    border-radius: 2px !important;
    transition: all 0.15s ease !important;
}}
.ch-btn:hover {{
    border-color: var(--primary) !important;
    color: var(--primary) !important;
}}
.ch-sel {{
    background: rgba(0, 255, 136, 0.15) !important;
    border-color: var(--primary) !important;
    color: var(--primary) !important;
    box-shadow: 0 0 8px rgba(0, 255, 136, 0.2) !important;
}}

.q-btn {{
    font-family: 'JetBrains Mono', monospace !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
}}

.q-field__control {{ background: rgba(0, 0, 0, 0.3) !important; border-radius: 2px !important; }}
.q-field--dark .q-field__control {{ border: 1px solid var(--border) !important; }}

.terminal-header {{
    font-size: 0.7rem;
    color: var(--text-dim);
    letter-spacing: 0.5px;
    text-transform: uppercase;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 8px;
}}

.terminal-label {{
    font-size: 0.85rem;
    color: var(--text);
}}

.glow-text {{ text-shadow: 0 0 15px rgba(0, 255, 136, 0.5); }}

/* Status indicators */
@keyframes pulse-orange {{
    0%, 100% {{ opacity: 1; box-shadow: 0 0 8px #f59e0b; }}
    50% {{ opacity: 0.4; box-shadow: 0 0 2px #f59e0b; }}
}}

.status-idle {{
    width: 12px; height: 12px; border-radius: 50%;
    background: #6b7280; 
    display: inline-block;
}}

.status-training {{
    width: 12px; height: 12px; border-radius: 50%;
    background: #f59e0b;
    animation: pulse-orange 1s ease-in-out infinite;
    display: inline-block;
}}

.status-completed {{
    width: 12px; height: 12px; border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 8px #10b981;
    display: inline-block;
}}

.status-error {{
    width: 12px; height: 12px; border-radius: 50%;
    background: #ef4444;
    box-shadow: 0 0 8px #ef4444;
    display: inline-block;
}}
"""


# Pre-generated style for direct import
STYLE = get_app_style()

