"""
Pipeline utility functions.

Single Responsibility: Provides helper functions for file operations and logging.
"""
from pathlib import Path
from datetime import datetime
from nicegui import ui

from config import THEME_TEXT
from app.state import PS

from . import config as _config


# =============================================================================
# RUN DIRECTORY MANAGEMENT
# =============================================================================

def get_run_dirs() -> list[str]:
    """
    List existing pipeline runs sorted by timestamp (newest first).
    
    Returns:
        List of run directory names (e.g., ['run_20241203_150000', 'run_20241202_120000'])
    """
    # Use module reference for testability (allows monkeypatching)
    pipeline_outputs = _config.PIPELINE_OUTPUTS
    if not pipeline_outputs.exists():
        return []
    return sorted(
        [d.name for d in pipeline_outputs.iterdir() 
         if d.is_dir() and d.name.startswith('run_')],
        reverse=True
    )


def create_new_run() -> Path:
    """
    Create a new run directory with timestamp and condition subdirectories.
    
    Returns:
        Path to the new run directory
    """
    # Use module reference for testability (allows monkeypatching)
    pipeline_outputs = _config.PIPELINE_OUTPUTS
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = pipeline_outputs / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create condition subdirectories
    for cond in ["DMT", "EC", "EO"]:
        (run_dir / cond).mkdir(exist_ok=True)
    
    return run_dir


# =============================================================================
# LOGGING
# =============================================================================

def pipeline_log(msg: str) -> None:
    """
    Add message to pipeline log with persistence support.
    
    Logs are stored in PS.log_history for persistence across tab switches.
    If the UI container is available and connected, displays the message.
    
    Args:
        msg: Message to log
    """
    # Always store in history for persistence
    PS.log_history.append(msg)
    
    # Try to update UI if container exists and client is connected
    if PS.log_container:
        try:
            with PS.log_container:
                # Add line break before new steps/sections
                if any(marker in msg for marker in _config.LOG_SECTION_MARKERS):
                    ui.label('').style('height: 12px;')
                ui.label(msg).style(
                    f'color:{THEME_TEXT}; font-family: JetBrains Mono; font-size: 0.75rem;'
                )
            # Force UI update and scroll to bottom
            PS.log_container.update()
            if hasattr(PS, 'log_scroll') and PS.log_scroll:
                PS.log_scroll.scroll_to(percent=1.0)
        except RuntimeError:
            # Client disconnected (tab switched), log is still stored in history
            pass


# =============================================================================
# FILE SCANNING
# =============================================================================

def extract_dataset_metadata(file_path: Path) -> dict:
    """
    Extract metadata from a .set (EEGLAB) file.
    
    Reads the file to get actual values for:
    - Sample rate (sfreq)
    - Number of channels
    - Number of epochs
    - Epoch duration in seconds
    - Samples per epoch (n_times)
    
    Args:
        file_path: Path to .set file
        
    Returns:
        Dictionary with metadata, or empty dict on error
    """
    try:
        import mne
        # Suppress MNE output for cleaner scanning
        epochs = mne.io.read_epochs_eeglab(str(file_path), montage_units='dm', verbose=False)
        
        # Extract metadata
        n_epochs, n_channels, n_times = epochs.get_data().shape
        sfreq = epochs.info['sfreq']
        epoch_duration = n_times / sfreq
        
        # Get channel names
        ch_names = epochs.info['ch_names']
        
        return {
            'sfreq': sfreq,
            'n_channels': n_channels,
            'n_epochs': n_epochs,
            'epoch_duration': round(epoch_duration, 3),
            'n_times': n_times,
            'ch_names': ch_names,
            'file': file_path.name
        }
    except Exception as e:
        # Return empty dict on any error (file not found, corrupt, etc.)
        return {}


def format_dataset_metadata(metadata: dict) -> str:
    """
    Format dataset metadata for display.
    
    Args:
        metadata: Dictionary with sfreq, n_channels, etc.
        
    Returns:
        Formatted string for display
    """
    if not metadata:
        return "No hay metadata disponible. Escaneá la carpeta INPUT primero."
    
    lines = []
    
    if 'sfreq' in metadata:
        lines.append(f"• Frecuencia de muestreo: {metadata['sfreq']:.0f} Hz")
    
    if 'n_channels' in metadata:
        lines.append(f"• Canales: {metadata['n_channels']}")
    
    if 'epoch_duration' in metadata:
        lines.append(f"• Duración época: {metadata['epoch_duration']} segundos")
    
    if 'n_times' in metadata:
        lines.append(f"• Muestras por época: {metadata['n_times']}")
    
    if 'n_epochs' in metadata:
        lines.append(f"• Épocas (archivo muestra): {metadata['n_epochs']}")
    
    if 'file' in metadata:
        lines.append(f"• Archivo escaneado: {metadata['file']}")
    
    return '\n'.join(lines)


def scan_input_directory(path: Path) -> dict:
    """
    Scan input directory for EEG files and extract metadata.
    
    Supports two structures:
    1. Subdirectory structure: DMT/, EC/, EO/ subdirectories
    2. Flat structure: Files with condition in filename
    
    Also extracts metadata from the first .set file found.
    
    Args:
        path: Path to scan
        
    Returns:
        Dictionary with scan results:
        {
            'structure': 'subdirs' | 'flat',
            'conditions': list of found conditions,
            'counts': dict of {condition: file_count},
            'total': total file count,
            'metadata': extracted metadata from first file
        }
    """
    if not path.exists():
        return {'structure': None, 'conditions': [], 'counts': {}, 'total': 0, 'metadata': None}
    
    # Check for subdirectory structure
    conds = [d.name for d in path.iterdir() if d.is_dir() and d.name in ['DMT', 'EC', 'EO']]
    
    metadata = None
    first_set_file = None
    
    if conds:
        # Subdirectory structure
        counts = {}
        for cond in conds:
            set_files = list((path / cond).glob('*.set'))
            counts[cond] = len(set_files)
            # Get first .set file for metadata
            if not first_set_file and set_files:
                first_set_file = set_files[0]
        
        # Extract metadata from first file
        if first_set_file:
            metadata = extract_dataset_metadata(first_set_file)
            # Store in PS for persistence
            PS.dataset_metadata = metadata
        
        return {
            'structure': 'subdirs',
            'conditions': conds,
            'counts': counts,
            'total': sum(counts.values()),
            'metadata': metadata
        }
    else:
        # Flat structure
        all_set = list(path.glob('*.set'))
        counts = {
            'DMT': len([f for f in all_set if 'DMT' in f.name.upper()]),
            'EC': len([f for f in all_set if 'EC' in f.name.upper()]),
            'EO': len([f for f in all_set if 'EO' in f.name.upper()]),
        }
        
        # Extract metadata from first file
        if all_set:
            metadata = extract_dataset_metadata(all_set[0])
            PS.dataset_metadata = metadata
        
        return {
            'structure': 'flat',
            'conditions': [c for c, n in counts.items() if n > 0],
            'counts': counts,
            'total': sum(counts.values()),
            'metadata': metadata
        }


def count_output_files(run_dir: Path) -> dict:
    """
    Count output files in a run directory by type.
    
    Args:
        run_dir: Path to run directory
        
    Returns:
        Dictionary with file counts by category
    """
    if not run_dir or not run_dir.exists():
        return {}
    
    def count(pattern):
        return len(list(run_dir.rglob(pattern)))
    
    return {
        'phases': count('phases-*.pkl'),
        'syncro': count('syncro-*.pkl'),
        'order_all': count('order_all-*.pkl'),
        'order': count('order-*.pkl'),
        'clustering': count('clustering_results/**/*.pkl') + count('clustering_results/**/*.csv'),
        'pearson': count('pearson_results/**/*'),
    }


def find_subjects_in_directory(path: Path, limit: int = 30) -> list[str]:
    """
    Find subject identifiers from pkl files in directory.
    
    Args:
        path: Directory to search
        limit: Maximum number of subjects to return
        
    Returns:
        Sorted list of subject identifiers
    """
    all_files = list(path.rglob('syncro-*.pkl')) + list(path.rglob('phases-*.pkl'))
    subjects = set()
    
    for f in all_files:
        # Extract subject ID from filename (e.g., syncro-S01-DMT.pkl -> S01-DMT)
        if '-' in f.stem:
            parts = f.stem.split('-')
            if len(parts) >= 2:
                subjects.add(parts[1] if len(parts) == 2 else f"{parts[1]}-{parts[2]}")
        else:
            subjects.add(f.stem)
    
    return sorted(list(subjects))[:limit]

