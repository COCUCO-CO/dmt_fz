"""EEG file loading utilities."""
import mne
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

from config import SUPPORTED_FORMATS, FREQ_BANDS


@dataclass
class EEGData:
    """Container for loaded EEG data."""
    raw: Optional[mne.io.Raw] = None
    filename: str = ""
    filepath: Path = field(default_factory=Path)
    sfreq: float = 0.0
    n_channels: int = 0
    n_samples: int = 0
    duration_sec: float = 0.0
    channel_names: List[str] = field(default_factory=list)
    channel_types: Dict[str, str] = field(default_factory=dict)
    info: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_loaded(self) -> bool:
        return self.raw is not None


def get_file_reader(filepath: Path):
    """Get the appropriate MNE reader for the file format."""
    suffix = filepath.suffix.lower()
    readers = {
        ".bdf": mne.io.read_raw_bdf,
        ".edf": mne.io.read_raw_edf,
        ".set": mne.io.read_raw_eeglab,
        ".fif": mne.io.read_raw_fif,
    }
    return readers.get(suffix)


def load_eeg_file(filepath: Path | str, preload: bool = True) -> EEGData:
    """
    Load an EEG file and return structured data.
    
    Args:
        filepath: Path to the EEG file
        preload: Whether to preload data into memory
        
    Returns:
        EEGData object with loaded data
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if filepath.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {filepath.suffix}. Supported: {SUPPORTED_FORMATS}")
    
    reader = get_file_reader(filepath)
    if reader is None:
        raise ValueError(f"No reader available for: {filepath.suffix}")
    
    # Load raw data
    raw = reader(filepath, preload=preload, verbose=False)
    
    # Extract metadata
    # Get channel types using the correct MNE API
    channel_types = {}
    for i, ch in enumerate(raw.ch_names):
        ch_type = raw.get_channel_types(picks=[i])[0]
        channel_types[ch] = ch_type
    
    eeg_data = EEGData(
        raw=raw,
        filename=filepath.name,
        filepath=filepath,
        sfreq=raw.info["sfreq"],
        n_channels=len(raw.ch_names),
        n_samples=raw.n_times,
        duration_sec=raw.n_times / raw.info["sfreq"],
        channel_names=raw.ch_names.copy(),
        channel_types=channel_types,
        info={
            "highpass": raw.info.get("highpass", 0),
            "lowpass": raw.info.get("lowpass", 0),
            "meas_date": str(raw.info.get("meas_date", "Unknown")),
            "subject_info": raw.info.get("subject_info", {}),
        }
    )
    
    return eeg_data


def get_channel_data(eeg_data: EEGData, channels: List[str] | None = None, 
                     start_sec: float = 0, duration_sec: float | None = None) -> tuple:
    """
    Extract channel data for visualization.
    
    Args:
        eeg_data: Loaded EEG data
        channels: List of channel names (None = all EEG channels)
        start_sec: Start time in seconds
        duration_sec: Duration in seconds (None = all data)
        
    Returns:
        Tuple of (data array, times array, channel names)
    """
    if not eeg_data.is_loaded:
        raise ValueError("No EEG data loaded")
    
    raw = eeg_data.raw
    
    # Select channels
    if channels is None:
        # Get only EEG channels by default
        picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        channels = [raw.ch_names[i] for i in picks]
    
    # Calculate sample indices
    start_sample = int(start_sec * eeg_data.sfreq)
    if duration_sec is None:
        stop_sample = None
    else:
        stop_sample = int((start_sec + duration_sec) * eeg_data.sfreq)
    
    # Extract data
    data, times = raw[channels, start_sample:stop_sample]
    
    return data, times, channels


def get_channel_stats(data: np.ndarray, channel_names: List[str]) -> Dict[str, Dict[str, float]]:
    """Calculate statistics for each channel."""
    stats = {}
    for i, ch_name in enumerate(channel_names):
        ch_data = data[i]
        stats[ch_name] = {
            "mean": float(np.mean(ch_data)),
            "std": float(np.std(ch_data)),
            "min": float(np.min(ch_data)),
            "max": float(np.max(ch_data)),
            "range": float(np.ptp(ch_data)),
        }
    return stats


def scan_eeg_directory(directory: Path) -> List[Dict[str, Any]]:
    """
    Scan a directory for EEG files.
    
    Returns:
        List of dicts with file info
    """
    directory = Path(directory)
    files = []
    
    if not directory.exists():
        return files
    
    for fmt in SUPPORTED_FORMATS:
        for filepath in directory.rglob(f"*{fmt}"):
            files.append({
                "name": filepath.name,
                "path": str(filepath),
                "format": fmt,
                "size_mb": filepath.stat().st_size / (1024 * 1024),
                "condition": filepath.parent.name,  # e.g., "DMT", "EC", "EO"
            })
    
    return sorted(files, key=lambda x: (x["condition"], x["name"]))

