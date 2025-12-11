"""
EEG Synchronization utilities for analysis page.

Simple helper functions for syncing EEG visualization with dataset playback.
The user manually selects the EEG file - no auto-detection needed.
"""
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def epoch_to_time(epoch_idx: int, epoch_duration: float) -> float:
    """
    Convert epoch index to time position in seconds.
    
    Args:
        epoch_idx: Index of the epoch (0-based)
        epoch_duration: Duration of each epoch in seconds
        
    Returns:
        Start time of the epoch in seconds
    """
    return float(epoch_idx) * epoch_duration


def clamp_view_to_eeg(
    view_start: float,
    view_duration: float,
    eeg_duration: float
) -> float:
    """
    Clamp view start position to valid range within EEG.
    
    Args:
        view_start: Desired start position in seconds
        view_duration: Duration of the view window in seconds
        eeg_duration: Total duration of the EEG in seconds
        
    Returns:
        Clamped view start position
    """
    max_start = max(0, eeg_duration - view_duration)
    return max(0, min(view_start, max_start))


def get_epoch_count(eeg_duration: float, epoch_duration: float) -> int:
    """
    Calculate the number of complete epochs in an EEG recording.
    
    Args:
        eeg_duration: Total duration of EEG in seconds
        epoch_duration: Duration of each epoch in seconds
        
    Returns:
        Number of complete epochs (floor division)
    """
    if epoch_duration <= 0:
        return 0
    return int(eeg_duration // epoch_duration)


def find_eeg_files_in_dir(directory: Path, extensions: list = None) -> list:
    """
    Find all EEG files in a directory.
    
    Args:
        directory: Directory to search
        extensions: List of file extensions to look for (default: ['.set', '.fif', '.edf', '.bdf'])
        
    Returns:
        List of Path objects for found EEG files
    """
    if extensions is None:
        extensions = ['.set', '.fif', '.edf', '.bdf']
    
    if not directory.exists():
        return []
    
    files = []
    for ext in extensions:
        files.extend(directory.glob(f'*{ext}'))
    
    return sorted(files, key=lambda x: x.name)
