"""
Epoch Creation and Rejection Module

Provides epoch segmentation and artifact rejection for EEG.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import mne


@dataclass
class EpochRejectionCriteria:
    """Criteria for epoch rejection."""
    peak_to_peak_uv: float = 150.0  # Maximum peak-to-peak amplitude in µV
    flat_uv: float = 0.5            # Minimum activity (flat epoch threshold) in µV
    gradient_uv_ms: float = 10.0    # Maximum gradient in µV/ms
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'peak_to_peak_uv': self.peak_to_peak_uv,
            'flat_uv': self.flat_uv,
            'gradient_uv_ms': self.gradient_uv_ms,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'EpochRejectionCriteria':
        return cls(
            peak_to_peak_uv=d.get('peak_to_peak_uv', 150.0),
            flat_uv=d.get('flat_uv', 0.5),
            gradient_uv_ms=d.get('gradient_uv_ms', 10.0),
        )
    
    @classmethod
    def default(cls) -> 'EpochRejectionCriteria':
        return cls()
    
    @classmethod
    def lenient(cls) -> 'EpochRejectionCriteria':
        """More permissive criteria."""
        return cls(peak_to_peak_uv=200.0, flat_uv=0.3, gradient_uv_ms=15.0)
    
    @classmethod
    def strict(cls) -> 'EpochRejectionCriteria':
        """Stricter criteria."""
        return cls(peak_to_peak_uv=100.0, flat_uv=1.0, gradient_uv_ms=5.0)


@dataclass
class EpochResult:
    """Results from epoch creation and rejection."""
    epochs: Optional[mne.Epochs] = None
    n_total: int = 0
    n_good: int = 0
    n_rejected: int = 0
    
    # Per-epoch info
    rejected_indices: List[int] = field(default_factory=list)
    rejection_reasons: Dict[int, List[str]] = field(default_factory=dict)
    epoch_stats: List[Dict[str, float]] = field(default_factory=list)
    
    # Per-channel rejection stats
    channel_rejection_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Manual rejections by channel
    manual_rejections: Dict[str, List[int]] = field(default_factory=dict)
    
    @property
    def rejection_rate(self) -> float:
        """Percentage of rejected epochs."""
        if self.n_total == 0:
            return 0.0
        return (self.n_rejected / self.n_total) * 100
    
    def get_rejection_summary(self) -> Dict[str, int]:
        """Get count of rejections by reason."""
        summary = {'peak_to_peak': 0, 'flat': 0, 'gradient': 0, 'manual': 0}
        for reasons in self.rejection_reasons.values():
            for reason in reasons:
                if reason in summary:
                    summary[reason] += 1
        return summary
    
    def to_json_export(self) -> Dict[str, Any]:
        """Export rejection info as JSON-serializable dict."""
        return {
            'n_total': self.n_total,
            'n_good': self.n_good,
            'n_rejected': self.n_rejected,
            'rejection_rate_percent': self.rejection_rate,
            'rejected_indices': self.rejected_indices,
            'rejection_reasons': self.rejection_reasons,
            'channel_stats': self.channel_rejection_stats,
            'manual_rejections': self.manual_rejections,
        }


def create_epochs(raw: mne.io.Raw,
                  duration: float = 2.0,
                  overlap: float = 0.0,
                  baseline: Optional[Tuple[float, float]] = None,
                  preload: bool = True) -> EpochResult:
    """
    Create epochs from continuous EEG.
    
    Args:
        raw: MNE Raw object
        duration: Epoch duration in seconds
        overlap: Overlap between epochs in seconds
        baseline: Baseline correction tuple (start, end) or None
        preload: Preload data into memory
        
    Returns:
        EpochResult with created epochs
    """
    # Create events (fixed interval)
    sfreq = raw.info['sfreq']
    n_samples = raw.n_times
    
    step = duration - overlap
    if step <= 0:
        step = duration / 2
    
    # Generate event times
    event_times = np.arange(0, n_samples / sfreq - duration, step)
    n_events = len(event_times)
    
    # Create events array [sample, 0, event_id]
    events = np.zeros((n_events, 3), dtype=int)
    events[:, 0] = (event_times * sfreq).astype(int)
    events[:, 2] = 1  # Event ID
    
    # Create epochs
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id={'epoch': 1},
        tmin=0,
        tmax=duration,
        baseline=baseline,
        preload=preload,
        verbose=False
    )
    
    result = EpochResult(
        epochs=epochs,
        n_total=len(epochs),
        n_good=len(epochs),
        n_rejected=0
    )
    
    # Calculate stats for each epoch
    if preload:
        data = epochs.get_data()  # (n_epochs, n_channels, n_samples)
        for i in range(len(epochs)):
            epoch_data = data[i] * 1e6  # Convert to µV
            result.epoch_stats.append({
                'index': i,
                'peak_to_peak': float(np.max(np.ptp(epoch_data, axis=1))),
                'mean_abs': float(np.mean(np.abs(epoch_data))),
                'std': float(np.std(epoch_data)),
            })
    
    return result


def detect_bad_epochs(epoch_result: EpochResult,
                      criteria: EpochRejectionCriteria) -> EpochResult:
    """
    Detect bad epochs based on rejection criteria.
    
    Args:
        epoch_result: EpochResult with epochs
        criteria: Rejection criteria
        
    Returns:
        Updated EpochResult with rejection info including per-channel stats
    """
    if epoch_result.epochs is None:
        return epoch_result
    
    epochs = epoch_result.epochs
    data = epochs.get_data() * 1e6  # Convert to µV
    sfreq = epochs.info['sfreq']
    ch_names = epochs.ch_names
    n_epochs = len(epochs)
    n_channels = len(ch_names)
    
    rejected_indices = []
    rejection_reasons = {}
    
    # Per-channel tracking: which epochs are bad for each channel
    channel_bad_epochs: Dict[str, List[int]] = {ch: [] for ch in ch_names}
    channel_reasons: Dict[str, Dict[int, List[str]]] = {ch: {} for ch in ch_names}
    
    for i in range(n_epochs):
        epoch_data = data[i]  # (n_channels, n_samples)
        epoch_reasons = []
        
        # Check each channel
        for ch_idx, ch_name in enumerate(ch_names):
            ch_data = epoch_data[ch_idx]
            ch_reasons = []
            
            # Peak-to-peak check per channel
            ptp = np.ptp(ch_data)
            if ptp > criteria.peak_to_peak_uv:
                ch_reasons.append('peak_to_peak')
            
            # Flat check per channel
            std_val = np.std(ch_data)
            if std_val < criteria.flat_uv:
                ch_reasons.append('flat')
            
            # Gradient check per channel
            gradient = np.max(np.abs(np.diff(ch_data))) * sfreq / 1000
            if gradient > criteria.gradient_uv_ms:
                ch_reasons.append('gradient')
            
            if ch_reasons:
                channel_bad_epochs[ch_name].append(i)
                channel_reasons[ch_name][i] = ch_reasons
        
        # Global epoch rejection (any channel bad = epoch bad)
        ptp_global = np.max(np.ptp(epoch_data, axis=1))
        if ptp_global > criteria.peak_to_peak_uv:
            epoch_reasons.append('peak_to_peak')
        
        min_std = np.min(np.std(epoch_data, axis=1))
        if min_std < criteria.flat_uv:
            epoch_reasons.append('flat')
        
        gradient_global = np.max(np.abs(np.diff(epoch_data, axis=1))) * sfreq / 1000
        if gradient_global > criteria.gradient_uv_ms:
            epoch_reasons.append('gradient')
        
        if epoch_reasons:
            rejected_indices.append(i)
            rejection_reasons[i] = epoch_reasons
        
        # Update stats
        if i < len(epoch_result.epoch_stats):
            epoch_result.epoch_stats[i]['peak_to_peak'] = float(ptp_global)
            epoch_result.epoch_stats[i]['min_std'] = float(min_std)
            epoch_result.epoch_stats[i]['max_gradient'] = float(gradient_global)
            epoch_result.epoch_stats[i]['is_rejected'] = len(epoch_reasons) > 0
    
    # Calculate per-channel rejection statistics
    channel_rejection_stats = {}
    for ch_name in ch_names:
        n_bad = len(channel_bad_epochs[ch_name])
        channel_rejection_stats[ch_name] = {
            'n_bad_epochs': n_bad,
            'bad_epoch_indices': channel_bad_epochs[ch_name],
            'rejection_rate_percent': (n_bad / n_epochs * 100) if n_epochs > 0 else 0,
            'reasons': channel_reasons[ch_name],
        }
    
    epoch_result.rejected_indices = rejected_indices
    epoch_result.rejection_reasons = rejection_reasons
    epoch_result.channel_rejection_stats = channel_rejection_stats
    epoch_result.n_rejected = len(rejected_indices)
    epoch_result.n_good = epoch_result.n_total - epoch_result.n_rejected
    
    return epoch_result


def apply_epoch_rejection(epoch_result: EpochResult,
                          additional_rejects: Optional[List[int]] = None) -> EpochResult:
    """
    Apply epoch rejection (drop bad epochs).
    
    Args:
        epoch_result: EpochResult with rejection info
        additional_rejects: Additional epoch indices to reject
        
    Returns:
        Updated EpochResult with rejected epochs dropped
    """
    if epoch_result.epochs is None:
        return epoch_result
    
    all_rejects = set(epoch_result.rejected_indices)
    if additional_rejects:
        all_rejects.update(additional_rejects)
    
    if not all_rejects:
        return epoch_result
    
    # Create mask for good epochs
    n_epochs = len(epoch_result.epochs)
    keep_mask = [i not in all_rejects for i in range(n_epochs)]
    
    # Drop bad epochs
    epochs_clean = epoch_result.epochs.copy()
    epochs_clean.drop(list(all_rejects), verbose=False)
    
    epoch_result.epochs = epochs_clean
    epoch_result.rejected_indices = sorted(list(all_rejects))
    epoch_result.n_rejected = len(all_rejects)
    epoch_result.n_good = n_epochs - len(all_rejects)
    
    return epoch_result


def get_epoch_data(epoch_result: EpochResult,
                   epoch_index: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Get data for a specific epoch.
    
    Args:
        epoch_result: EpochResult with epochs
        epoch_index: Index of epoch to get
        
    Returns:
        Tuple of (data [n_channels, n_samples], times, channel_names)
    """
    if epoch_result.epochs is None:
        return np.array([]), np.array([]), []
    
    epochs = epoch_result.epochs
    
    if epoch_index < 0 or epoch_index >= len(epochs):
        return np.array([]), np.array([]), []
    
    data = epochs.get_data()[epoch_index]
    times = epochs.times
    ch_names = epochs.ch_names
    
    return data, times, list(ch_names)


def get_rejection_stats(epoch_result: EpochResult) -> Dict[str, Any]:
    """
    Get statistics about epoch rejection.
    
    Returns:
        Dictionary with rejection statistics
    """
    return {
        'n_total': epoch_result.n_total,
        'n_good': epoch_result.n_good,
        'n_rejected': epoch_result.n_rejected,
        'rejection_rate_percent': epoch_result.rejection_rate,
        'rejection_by_reason': epoch_result.get_rejection_summary(),
        'channel_stats': epoch_result.channel_rejection_stats,
    }


def export_rejection_info(epoch_result: EpochResult, filepath: str) -> None:
    """
    Export epoch rejection information to JSON file.
    
    Args:
        epoch_result: EpochResult with rejection info
        filepath: Path to save JSON file
    """
    import json
    from pathlib import Path
    
    export_data = epoch_result.to_json_export()
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(export_data, f, indent=2)


def add_manual_rejection(epoch_result: EpochResult, 
                        channel: str, 
                        epoch_indices: List[int]) -> EpochResult:
    """
    Add manual epoch rejections for a specific channel.
    
    Args:
        epoch_result: EpochResult
        channel: Channel name
        epoch_indices: Epoch indices to reject for this channel
        
    Returns:
        Updated EpochResult
    """
    if channel not in epoch_result.manual_rejections:
        epoch_result.manual_rejections[channel] = []
    
    for idx in epoch_indices:
        if idx not in epoch_result.manual_rejections[channel]:
            epoch_result.manual_rejections[channel].append(idx)
        if idx not in epoch_result.rejected_indices:
            epoch_result.rejected_indices.append(idx)
            epoch_result.rejection_reasons[idx] = epoch_result.rejection_reasons.get(idx, []) + ['manual']
    
    epoch_result.rejected_indices = sorted(epoch_result.rejected_indices)
    epoch_result.n_rejected = len(epoch_result.rejected_indices)
    epoch_result.n_good = epoch_result.n_total - epoch_result.n_rejected
    
    return epoch_result

