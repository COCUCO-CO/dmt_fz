"""
Bad Channel Detection Module

Provides automatic and semi-automatic detection of bad EEG channels.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import mne


@dataclass
class BadChannelResult:
    """Results of bad channel detection."""
    flat: List[str] = field(default_factory=list)
    noisy: List[str] = field(default_factory=list)
    uncorrelated: List[str] = field(default_factory=list)
    
    # Detailed info for UI
    channel_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    @property
    def all_bad(self) -> List[str]:
        """All detected bad channels (unique)."""
        return list(set(self.flat + self.noisy + self.uncorrelated))
    
    def get_reason(self, ch_name: str) -> str:
        """Get reason why channel is marked as bad."""
        reasons = []
        if ch_name in self.flat:
            reasons.append("flat")
        if ch_name in self.noisy:
            reasons.append("noisy")
        if ch_name in self.uncorrelated:
            reasons.append("uncorrelated")
        return ", ".join(reasons) if reasons else "manual"


def detect_bad_channels(raw: mne.io.Raw,
                        std_threshold: float = 3.5,
                        flat_threshold: float = 1e-12,
                        corr_threshold: float = 0.1,
                        check_correlation: bool = False,
                        duration: Optional[float] = None) -> BadChannelResult:
    """
    Detect bad channels automatically.
    
    Args:
        raw: MNE Raw object
        std_threshold: Z-score threshold for noisy channels (higher = less sensitive)
        flat_threshold: Variance threshold for flat channels in V² (typical EEG variance ~1e-10)
        corr_threshold: Minimum mean correlation with other channels (lower = less sensitive)
        check_correlation: Whether to check correlation (can cause false positives)
        duration: Duration in seconds to analyze (None = all)
        
    Returns:
        BadChannelResult with detected bad channels and stats
    """
    # Get EEG data
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    ch_names = [raw.ch_names[i] for i in picks]
    
    if duration is not None:
        max_samples = int(duration * raw.info['sfreq'])
        data = raw.get_data(picks=picks)[:, :max_samples]
    else:
        data = raw.get_data(picks=picks)
    
    result = BadChannelResult()
    
    # Calculate variance per channel
    variances = np.var(data, axis=1)
    mean_var = np.mean(variances)
    std_var = np.std(variances)
    
    # Calculate mean absolute value
    mean_abs = np.mean(np.abs(data), axis=1)
    
    # Calculate correlation matrix
    if len(ch_names) > 1:
        corr_matrix = np.corrcoef(data)
        # Mean correlation with other channels (excluding self)
        np.fill_diagonal(corr_matrix, np.nan)
        mean_corr = np.nanmean(np.abs(corr_matrix), axis=1)
    else:
        mean_corr = np.array([1.0])
    
    for i, ch in enumerate(ch_names):
        var = variances[i]
        
        # Store stats
        result.channel_stats[ch] = {
            'variance': float(var),
            'variance_zscore': float((var - mean_var) / std_var) if std_var > 0 else 0,
            'mean_abs': float(mean_abs[i]),
            'mean_correlation': float(mean_corr[i]),
        }
        
        # Check flat
        if var < flat_threshold:
            result.flat.append(ch)
            continue
        
        # Check noisy (high variance)
        z_score = (var - mean_var) / std_var if std_var > 0 else 0
        if z_score > std_threshold:
            result.noisy.append(ch)
            continue
        
        # Check uncorrelated (only if enabled, as it can cause false positives)
        if check_correlation and mean_corr[i] < corr_threshold:
            result.uncorrelated.append(ch)
    
    return result


def detect_bridged_channels(raw: mne.io.Raw,
                            threshold: float = 0.99) -> List[Tuple[str, str]]:
    """
    Detect bridged (short-circuited) channel pairs.
    
    Bridged channels show nearly identical signals.
    
    Args:
        raw: MNE Raw object
        threshold: Correlation threshold for bridging
        
    Returns:
        List of tuples (ch1, ch2) of bridged pairs
    """
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    ch_names = [raw.ch_names[i] for i in picks]
    data = raw.get_data(picks=picks)
    
    if len(ch_names) < 2:
        return []
    
    corr_matrix = np.corrcoef(data)
    bridged = []
    
    for i in range(len(ch_names)):
        for j in range(i + 1, len(ch_names)):
            if abs(corr_matrix[i, j]) > threshold:
                bridged.append((ch_names[i], ch_names[j]))
    
    return bridged


def get_channel_neighbors(ch_names: List[str], 
                          montage_name: str = 'standard_1020') -> Dict[str, List[str]]:
    """
    Get neighboring channels based on electrode positions.
    
    Args:
        ch_names: List of channel names
        montage_name: Montage name (default: standard_1020)
        
    Returns:
        Dictionary mapping each channel to its neighbors
    """
    try:
        montage = mne.channels.make_standard_montage(montage_name)
        positions = montage.get_positions()['ch_pos']
        
        # Filter to only channels we have
        valid_channels = [ch for ch in ch_names if ch in positions]
        
        if len(valid_channels) < 2:
            return {ch: [] for ch in ch_names}
        
        # Calculate distances
        pos_array = np.array([positions[ch] for ch in valid_channels])
        distances = squareform(pdist(pos_array))
        
        # Find neighbors (closest N channels)
        n_neighbors = min(4, len(valid_channels) - 1)
        neighbors = {}
        
        for i, ch in enumerate(valid_channels):
            sorted_idx = np.argsort(distances[i])
            # Skip self (index 0)
            neighbor_idx = sorted_idx[1:n_neighbors + 1]
            neighbors[ch] = [valid_channels[j] for j in neighbor_idx]
        
        # Add empty lists for channels not in montage
        for ch in ch_names:
            if ch not in neighbors:
                neighbors[ch] = []
        
        return neighbors
        
    except Exception:
        # Fallback: no neighbor info
        return {ch: [] for ch in ch_names}


def interpolate_channels(raw: mne.io.Raw, 
                         bad_channels: List[str],
                         method: str = 'spline') -> mne.io.Raw:
    """
    Interpolate bad channels using neighboring channels.
    
    Args:
        raw: MNE Raw object
        bad_channels: List of channel names to interpolate
        method: Interpolation method ('spline' or 'MNE')
        
    Returns:
        Raw object with interpolated channels (copy)
    """
    if not bad_channels:
        return raw.copy()
    
    raw_interp = raw.copy()
    
    # Mark bad channels
    raw_interp.info['bads'] = list(bad_channels)
    
    # Set montage if not set
    try:
        if raw_interp.get_montage() is None:
            montage = mne.channels.make_standard_montage('standard_1020')
            # Only set for channels that exist in montage
            raw_interp.set_montage(montage, on_missing='ignore')
    except Exception:
        pass
    
    # Interpolate
    try:
        raw_interp.interpolate_bads(reset_bads=True, verbose=False)
    except Exception as e:
        # If interpolation fails, just return copy without interpolation
        print(f"Interpolation failed: {e}")
        raw_interp = raw.copy()
    
    return raw_interp


def compute_channel_quality_metrics(raw: mne.io.Raw) -> Dict[str, Dict[str, float]]:
    """
    Compute comprehensive quality metrics for each channel.
    
    Useful for displaying in UI to help user decide which channels are bad.
    
    Returns:
        Dict mapping channel name to dict of metrics
    """
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    ch_names = [raw.ch_names[i] for i in picks]
    data = raw.get_data(picks=picks)
    
    metrics = {}
    
    # Global stats for normalization
    global_var = np.var(data)
    global_mean = np.mean(np.abs(data))
    
    for i, ch in enumerate(ch_names):
        ch_data = data[i]
        
        # Variance
        var = np.var(ch_data)
        
        # Kurtosis (high = spiky artifacts)
        kurt = stats.kurtosis(ch_data)
        
        # Range (peak-to-peak)
        ptp = np.ptp(ch_data)
        
        # Zero crossings (low = drift, very high = noise)
        zero_crossings = np.sum(np.diff(np.signbit(ch_data)))
        zc_rate = zero_crossings / len(ch_data) * raw.info['sfreq']
        
        # High frequency power ratio (noise indicator)
        # Simple approximation using difference
        hf_power = np.var(np.diff(ch_data))
        hf_ratio = hf_power / var if var > 0 else 0
        
        metrics[ch] = {
            'variance': float(var),
            'variance_normalized': float(var / global_var) if global_var > 0 else 0,
            'kurtosis': float(kurt),
            'peak_to_peak_uv': float(ptp * 1e6),  # Convert to µV
            'zero_crossing_rate': float(zc_rate),
            'hf_ratio': float(hf_ratio),
        }
    
    return metrics

