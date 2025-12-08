"""
Export Module

Provides export functionality for cleaned EEG data.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import json
import numpy as np
import mne

from .state import CleaningState


def _convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {_convert_numpy_types(k): _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy_types(item) for item in obj]
    return obj


import re

def get_pipeline_compatible_name(filename: str) -> str:
    """
    Generate a pipeline-compatible filename from the original filename.
    
    Pipeline expects: Subject_Condition.set (e.g., S01_DMT.set)
    
    Extracts subject ID and condition from filenames like:
    - S01-DMT.bdf -> S01_DMT
    - S01_DMT.set -> S01_DMT
    - S01-DMT-raw.fif -> S01_DMT
    - sub-01_task-DMT_eeg.bdf -> sub-01_DMT
    - Subject01_condition_DMT.bdf -> Subject01_DMT
    
    If pattern not recognized, returns cleaned version of original name.
    """
    # Remove extension
    base = Path(filename).stem
    
    # Remove common suffixes added by processing
    for suffix in ['_raw', '-raw', '_cleaned', '-cleaned', '_epo', '-epo', '_ICA_pruned', '_ica']:
        base = re.sub(rf'{suffix}$', '', base, flags=re.IGNORECASE)
    
    # Try to extract subject and condition
    conditions = ['DMT', 'EC', 'EO']
    
    # Pattern 1: S01-DMT or S01_DMT format
    match = re.match(r'^(S\d+)[-_](\w+)', base, re.IGNORECASE)
    if match:
        subject = match.group(1).upper()
        rest = match.group(2).upper()
        # Check if rest contains a condition
        for cond in conditions:
            if cond in rest:
                return f"{subject}_{cond}"
    
    # Pattern 2: sub-XX_task-CONDITION format (BIDS-like)
    match = re.match(r'^(sub-\d+).*?(DMT|EC|EO)', base, re.IGNORECASE)
    if match:
        subject = match.group(1)
        condition = match.group(2).upper()
        return f"{subject}_{condition}"
    
    # Pattern 3: Any filename containing a condition
    base_upper = base.upper()
    for cond in conditions:
        if cond in base_upper:
            # Try to extract subject number
            subj_match = re.search(r'(S?\d+)', base)
            if subj_match:
                subj = subj_match.group(1)
                if not subj.upper().startswith('S'):
                    subj = f"S{subj.zfill(2)}"
                return f"{subj.upper()}_{cond}"
            else:
                # No subject number found, use cleaned base name
                # Remove timestamps and extra info
                clean = re.sub(r'_?\d{8}[-_]\d{6}', '', base)  # Remove timestamps
                clean = re.sub(r'[-_]+(cleaned|raw|epo|ica)[-_]*', '_', clean, flags=re.IGNORECASE)
                clean = re.sub(r'[-_]+', '_', clean).strip('_')
                return clean
    
    # No condition found - return cleaned original name
    clean = re.sub(r'_?\d{8}[-_]\d{6}', '', base)  # Remove timestamps
    clean = re.sub(r'[-_]+(cleaned|raw|epo|ica)[-_]*', '_', clean, flags=re.IGNORECASE)
    clean = re.sub(r'[-_]+', '_', clean).strip('_')
    return clean if clean else base


class ExportFormat(Enum):
    """Supported export formats."""
    FIF = "fif"      # MNE native format
    SET = "set"      # EEGLAB format
    EDF = "edf"      # European Data Format
    NPY = "npy"      # NumPy array (epochs only)
    CSV = "csv"      # CSV (epochs metadata)
    
    @property
    def extension(self) -> str:
        extensions = {
            ExportFormat.FIF: ".fif",
            ExportFormat.SET: ".set",
            ExportFormat.EDF: ".edf",
            ExportFormat.NPY: ".npy",
            ExportFormat.CSV: ".csv",
        }
        return extensions.get(self, "")
    
    @property
    def display_name(self) -> str:
        names = {
            ExportFormat.FIF: "MNE/FIF (recommended)",
            ExportFormat.SET: "EEGLAB (.set)",
            ExportFormat.EDF: "EDF (.edf)",
            ExportFormat.NPY: "NumPy (.npy)",
            ExportFormat.CSV: "CSV (.csv)",
        }
        return names.get(self, self.name)


def export_cleaned_eeg(raw: mne.io.Raw,
                       output_path: Path,
                       format: ExportFormat = ExportFormat.FIF,
                       overwrite: bool = False) -> Path:
    """
    Export cleaned EEG data.
    
    Args:
        raw: Cleaned MNE Raw object
        output_path: Output file path (extension will be added if needed)
        format: Export format
        overwrite: Overwrite existing file
        
    Returns:
        Path to exported file
    """
    output_path = Path(output_path)
    
    # Ensure correct extension
    if not output_path.suffix:
        output_path = output_path.with_suffix(format.extension)
    
    # Create directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == ExportFormat.FIF:
        raw.save(output_path, overwrite=overwrite, verbose=False)
    
    elif format == ExportFormat.SET:
        # Export as EEGLAB .set
        raw.export(output_path, overwrite=overwrite, verbose=False)
    
    elif format == ExportFormat.EDF:
        raw.export(output_path, overwrite=overwrite, verbose=False)
    
    else:
        raise ValueError(f"Unsupported format for raw export: {format}")
    
    return output_path


def export_epochs(epochs: mne.Epochs,
                  output_dir: Path,
                  base_name: str,
                  formats: List[ExportFormat] = None,
                  include_metadata: bool = True) -> Dict[str, Path]:
    """
    Export epochs in multiple formats.
    
    Args:
        epochs: MNE Epochs object
        output_dir: Output directory
        base_name: Base filename (without extension)
        formats: List of formats to export (default: FIF + NPY)
        include_metadata: Include CSV with metadata
        
    Returns:
        Dictionary mapping format name to output path
    """
    if formats is None:
        formats = [ExportFormat.FIF, ExportFormat.NPY]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    outputs = {}
    
    for fmt in formats:
        if fmt == ExportFormat.FIF:
            # Use MNE naming convention for epochs: _epo.fif
            output_path = output_dir / f"{base_name}_epo.fif"
            epochs.save(output_path, overwrite=True, verbose=False)
            outputs['fif'] = output_path
        
        elif fmt == ExportFormat.NPY:
            output_path = output_dir / f"{base_name}{fmt.extension}"
            # Save as numpy array
            data = epochs.get_data()
            np.save(output_path, data)
            outputs['npy'] = output_path
            
            # Also save times
            times_path = output_dir / f"{base_name}_times.npy"
            np.save(times_path, epochs.times)
            outputs['times'] = times_path
        
        elif fmt == ExportFormat.SET:
            output_path = output_dir / f"{base_name}{fmt.extension}"
            epochs.export(output_path, overwrite=True, verbose=False)
            outputs['set'] = output_path
    
    # Export metadata
    if include_metadata:
        meta_path = output_dir / f"{base_name}_metadata.csv"
        _export_epochs_metadata(epochs, meta_path)
        outputs['metadata'] = meta_path
    
    return outputs


def _export_epochs_metadata(epochs: mne.Epochs, output_path: Path):
    """Export epochs metadata to CSV."""
    import pandas as pd
    
    data = {
        'n_epochs': len(epochs),
        'n_channels': len(epochs.ch_names),
        'sfreq': epochs.info['sfreq'],
        'tmin': epochs.tmin,
        'tmax': epochs.tmax,
        'duration': epochs.tmax - epochs.tmin,
        'channels': ','.join(epochs.ch_names),
    }
    
    df = pd.DataFrame([data])
    df.to_csv(output_path, index=False)


def export_preprocessing_log(state: CleaningState,
                             output_path: Path) -> Path:
    """
    Export preprocessing log as JSON.
    
    Args:
        state: CleaningState with preprocessing history
        output_path: Output path for JSON file
        
    Returns:
        Path to exported file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    log = state.get_preprocessing_summary()
    
    # Add export info
    log['export_timestamp'] = datetime.now().isoformat()
    
    # Convert numpy types to native Python types
    log = _convert_numpy_types(log)
    
    with open(output_path, 'w') as f:
        json.dump(log, f, indent=2, default=str)
    
    return output_path


def create_export_bundle(state: CleaningState,
                         output_dir: Path,
                         base_name: Optional[str] = None,
                         include_raw: bool = True,
                         include_epochs: bool = True,
                         raw_format: ExportFormat = ExportFormat.FIF,
                         epochs_formats: List[ExportFormat] = None) -> Dict[str, Path]:
    """
    Create complete export bundle with all data and logs.
    
    Args:
        state: CleaningState with cleaned data
        output_dir: Output directory
        base_name: Base filename (default: original filename)
        include_raw: Include cleaned continuous data
        include_epochs: Include epochs if available
        raw_format: Format for raw data
        epochs_formats: Formats for epochs
        
    Returns:
        Dictionary mapping output type to path
    """
    if epochs_formats is None:
        epochs_formats = [ExportFormat.FIF, ExportFormat.NPY]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if base_name is None:
        base_name = state.filename.rsplit('.', 1)[0] if state.filename else 'cleaned_eeg'
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = f"{base_name}_cleaned_{timestamp}"
    
    outputs = {}
    
    # Export raw
    if include_raw and state.raw is not None:
        raw_path = export_cleaned_eeg(
            state.raw,
            output_dir / f"{base_name}_raw",
            format=raw_format,
            overwrite=True
        )
        outputs['raw'] = raw_path
    
    # Export epochs
    if include_epochs and state.epochs is not None:
        epoch_outputs = export_epochs(
            state.epochs,
            output_dir,
            f"{base_name}_epochs",
            formats=epochs_formats
        )
        outputs.update({f'epochs_{k}': v for k, v in epoch_outputs.items()})
    
    # Export preprocessing log
    log_path = export_preprocessing_log(
        state,
        output_dir / f"{base_name}_preprocessing_log.json"
    )
    outputs['log'] = log_path
    
    # Export channel info
    if state.raw is not None:
        ch_info_path = output_dir / f"{base_name}_channels.txt"
        with open(ch_info_path, 'w') as f:
            f.write("# Channel Information\n")
            f.write(f"# Channels: {len(state.ch_names)}\n")
            f.write(f"# Bad channels: {state.bad_channels}\n")
            f.write(f"# Interpolated: {state.interpolated_channels}\n\n")
            for i, ch in enumerate(state.ch_names):
                status = "BAD" if ch in state.bad_channels else "OK"
                interp = " (interpolated)" if ch in state.interpolated_channels else ""
                f.write(f"{i+1:3d}. {ch:8s} [{status}]{interp}\n")
        outputs['channels'] = ch_info_path
    
    return outputs

