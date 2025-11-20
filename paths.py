from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

EEG_CLEAN_DIR = BASE_DIR / "EEG_CLEAN"
EEG_DIR = BASE_DIR / "EEG"
RESULTS_DIR = BASE_DIR / "fwd-inv-stc"
SPECTRAL_DIR = BASE_DIR / "spectral_sources"
EEGNET_DIR = BASE_DIR / "EEGNet"
RESULTS_PLOTS_DIR = BASE_DIR / "results_plots"
FRAMES_DIR = BASE_DIR / "frames"


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist and return the resolved Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

