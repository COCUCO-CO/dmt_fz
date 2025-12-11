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

def scan_input_directory(path: Path) -> dict:
    """
    Scan input directory for EEG files.
    
    Supports two structures:
    1. Subdirectory structure: DMT/, EC/, EO/ subdirectories
    2. Flat structure: Files with condition in filename
    
    Args:
        path: Path to scan
        
    Returns:
        Dictionary with scan results:
        {
            'structure': 'subdirs' | 'flat',
            'conditions': list of found conditions,
            'counts': dict of {condition: file_count},
            'total': total file count
        }
    """
    if not path.exists():
        return {'structure': None, 'conditions': [], 'counts': {}, 'total': 0}
    
    # Check for subdirectory structure
    conds = [d.name for d in path.iterdir() if d.is_dir() and d.name in ['DMT', 'EC', 'EO']]
    
    if conds:
        # Subdirectory structure
        counts = {}
        for cond in conds:
            counts[cond] = len(list((path / cond).glob('*.set')))
        return {
            'structure': 'subdirs',
            'conditions': conds,
            'counts': counts,
            'total': sum(counts.values())
        }
    else:
        # Flat structure
        all_set = list(path.glob('*.set'))
        counts = {
            'DMT': len([f for f in all_set if 'DMT' in f.name.upper()]),
            'EC': len([f for f in all_set if 'EC' in f.name.upper()]),
            'EO': len([f for f in all_set if 'EO' in f.name.upper()]),
        }
        return {
            'structure': 'flat',
            'conditions': [c for c, n in counts.items() if n > 0],
            'counts': counts,
            'total': sum(counts.values())
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

