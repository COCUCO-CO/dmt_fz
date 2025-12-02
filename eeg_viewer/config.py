"""Configuration for EEG Viewer application."""
from pathlib import Path

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
EEG_RAW_DIR = BASE_DIR / "EEG"
EEG_CLEAN_DIR = BASE_DIR / "EEG_CLEAN"
RESULTS_DIR = BASE_DIR / "fwd-inv-stc"

# EEG Settings
SUPPORTED_FORMATS = [".bdf", ".set", ".edf", ".fif"]
DEFAULT_SFREQ = 512  # Default sampling frequency

# UI Theme - Konsole/Terminal Style
THEME_BG = "#0a0a0a"           # Deep black background
THEME_CARD = "#111111"         # Card background
THEME_BORDER = "#1e1e1e"       # Subtle borders
THEME_PRIMARY = "#00ff88"      # Terminal green (main accent)
THEME_SECONDARY = "#00d4ff"    # Cyan (secondary accent)
THEME_WARN = "#ffcc00"         # Amber/yellow
THEME_ERROR = "#ff5555"        # Red for errors
THEME_TEXT = "#c8c8c8"         # Light gray text
THEME_TEXT_DIM = "#666666"     # Dimmed text
THEME_ACCENT = "#00ff88"       # Keep for backwards compatibility

# Signal colors (terminal palette)
SIGNAL_COLORS = [
    '#00ff88',  # Terminal green
    '#00d4ff',  # Cyan
    '#ffcc00',  # Yellow
    '#ff6b9d',  # Pink
    '#a78bfa',  # Purple
    '#00ffcc',  # Turquoise
    '#ff9f43',  # Orange
    '#74b9ff',  # Light blue
    '#55efc4',  # Mint
    '#fd79a8',  # Rose
    '#81ecec',  # Aqua
    '#ffeaa7',  # Cream
]

# Frequency Bands
FREQ_BANDS = {
    "δ": (1, 4),     # Delta
    "θ": (4, 8),     # Theta
    "α": (8, 13),    # Alpha
    "β": (13, 30),   # Beta
    "γ": (30, 45),   # Gamma
}

