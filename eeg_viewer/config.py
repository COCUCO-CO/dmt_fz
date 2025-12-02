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

# UI Settings
THEME_PRIMARY = "#1a1a2e"
THEME_SECONDARY = "#16213e"
THEME_ACCENT = "#e94560"
THEME_TEXT = "#eaeaea"

# Frequency Bands
FREQ_BANDS = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 45),
}

