"""
EEG Synchronization utilities for analysis page.

Provides functions to synchronize EEG visualization with autoencoder
dataset playback, allowing visualization of EEG epochs as they are
processed through the model.
"""
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


def find_eeg_file(
    subject_id: str,
    condition: str,
    eeg_base_dir: Path,
    extensions: List[str] = ['.set', '.fif', '.edf', '.bdf']
) -> Optional[Path]:
    """
    Find the EEG file for a given subject and condition.
    
    Searches in: eeg_base_dir/{condition}/ for files matching the subject/condition.
    
    Args:
        subject_id: Subject ID (e.g., 'S01')
        condition: Condition name (e.g., 'DMT', 'EC', 'EO')
        eeg_base_dir: Base directory for EEG files
        extensions: List of valid file extensions to search for
        
    Returns:
        Path to EEG file or None if not found
    """
    if not eeg_base_dir.exists():
        logger.warning(f"EEG base directory does not exist: {eeg_base_dir}")
        return None
    
    condition_dir = eeg_base_dir / condition
    if not condition_dir.exists():
        logger.warning(f"Condition directory does not exist: {condition_dir}")
        return None
    
    # Try different filename patterns
    patterns = [
        f"{subject_id}-{condition}_ICA_pruned",  # S01-DMT_ICA_pruned
        f"{subject_id}_{condition}_ICA_pruned",  # S01_DMT_ICA_pruned
        f"{subject_id}-{condition}",              # S01-DMT
        f"{subject_id}_{condition}",              # S01_DMT
    ]
    
    for pattern in patterns:
        for ext in extensions:
            filepath = condition_dir / f"{pattern}{ext}"
            if filepath.exists():
                logger.debug(f"Found EEG file: {filepath}")
                return filepath
    
    # Fallback: search for any file containing subject_id and condition
    for ext in extensions:
        for filepath in condition_dir.glob(f"*{subject_id}*{condition}*{ext}"):
            logger.debug(f"Found EEG file via glob: {filepath}")
            return filepath
        for filepath in condition_dir.glob(f"*{subject_id}*{ext}"):
            if condition.lower() in filepath.name.lower():
                logger.debug(f"Found EEG file via partial match: {filepath}")
                return filepath
    
    logger.warning(f"No EEG file found for {subject_id}/{condition} in {condition_dir}")
    return None


def extract_eeg_metadata(sample: Any) -> Optional[Dict[str, Any]]:
    """
    Extract EEG-related metadata from a graph sample.
    
    Args:
        sample: Graph sample (torch_geometric Data object or similar)
        
    Returns:
        Dict with subject_id, condition, epoch_idx, band or None if not available
    """
    required_attrs = ['subject_id', 'condition', 'epoch_idx']
    
    # Check all required attributes exist
    for attr in required_attrs:
        if not hasattr(sample, attr):
            return None
    
    try:
        metadata = {
            'subject_id': sample.subject_id,
            'condition': sample.condition,
            'epoch_idx': int(sample.epoch_idx),
            'band': getattr(sample, 'band', 'unknown'),
        }
        return metadata
    except Exception as e:
        logger.warning(f"Error extracting metadata: {e}")
        return None


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


def should_enable_eeg_sync(dataset: Any) -> bool:
    """
    Check if EEG synchronization should be enabled for a dataset.
    
    Returns True if the dataset contains samples with valid EEG metadata.
    
    Args:
        dataset: Dataset (list-like) to check
        
    Returns:
        True if sync should be enabled, False otherwise
    """
    if dataset is None:
        return False
    
    try:
        if len(dataset) == 0:
            return False
    except TypeError:
        return False
    
    # Check first sample for EEG metadata
    try:
        first_sample = dataset[0]
        metadata = extract_eeg_metadata(first_sample)
        return metadata is not None
    except (IndexError, KeyError, TypeError):
        return False


def update_eeg_sync_state(
    state: Any,
    sample: Any,
    eeg_base_dir: Path,
    epoch_duration: float = 2.0
) -> bool:
    """
    Update the EEG sync state based on current sample.
    
    This function:
    1. Checks if sync is enabled
    2. Extracts metadata from sample
    3. Loads EEG file if needed (different subject/condition)
    4. Updates view position based on epoch_idx
    
    Args:
        state: Analysis state object with eeg_* attributes
        sample: Current graph sample
        eeg_base_dir: Base directory for EEG files
        epoch_duration: Duration of each epoch in seconds
        
    Returns:
        True if update was successful, False otherwise
    """
    # Check if sync is enabled
    if not getattr(state, 'eeg_sync_enabled', True):
        return False
    
    # Extract metadata
    metadata = extract_eeg_metadata(sample)
    if metadata is None:
        return False
    
    subject_id = metadata['subject_id']
    condition = metadata['condition']
    epoch_idx = metadata['epoch_idx']
    
    # Build expected file identifier
    expected_file_id = f"{subject_id}-{condition}"
    current_file = getattr(state, 'current_eeg_file', None)
    
    # Check if we need to load a new EEG file
    need_new_eeg = True
    if current_file and expected_file_id in str(current_file):
        need_new_eeg = False
    
    if need_new_eeg:
        # Find and load new EEG file
        eeg_path = find_eeg_file(subject_id, condition, eeg_base_dir)
        
        if eeg_path is None:
            logger.warning(f"Could not find EEG file for {subject_id}/{condition}")
            return False
        
        try:
            # Import here to avoid circular imports
            from eeg_loader import load_eeg_file
            
            eeg_data = load_eeg_file(eeg_path)
            state.eeg_data = eeg_data
            state.current_eeg_file = str(eeg_path)
            logger.info(f"Loaded EEG: {eeg_path.name}")
        except Exception as e:
            logger.error(f"Failed to load EEG file: {e}")
            return False
    
    # Update view position based on epoch
    if state.eeg_data is not None:
        view_start = epoch_to_time(epoch_idx, epoch_duration)
        view_start = clamp_view_to_eeg(
            view_start, 
            epoch_duration, 
            state.eeg_data.duration_sec
        )
        state.eeg_view_start = view_start
        return True
    
    return False


def get_sync_info_text(
    sample: Any,
    eeg_data: Any,
    epoch_duration: float
) -> str:
    """
    Get informational text about current sync state.
    
    Args:
        sample: Current graph sample
        eeg_data: Loaded EEG data (or None)
        epoch_duration: Epoch duration in seconds
        
    Returns:
        Human-readable sync status string
    """
    metadata = extract_eeg_metadata(sample)
    if metadata is None:
        return "No EEG metadata in sample"
    
    if eeg_data is None:
        return f"EEG not loaded for {metadata['subject_id']}/{metadata['condition']}"
    
    epoch_idx = metadata['epoch_idx']
    time_pos = epoch_to_time(epoch_idx, epoch_duration)
    total_epochs = get_epoch_count(eeg_data.duration_sec, epoch_duration)
    
    return f"Epoch {epoch_idx}/{total_epochs} | {time_pos:.1f}s | {metadata['band']}"

