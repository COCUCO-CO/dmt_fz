"""
High-level signal processing pipeline.

Combines filters and transforms into processing pipelines.
Uses the ViewerState to determine which processing steps to apply.
"""
from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, Optional

from .filters import apply_notch, apply_bandpass

if TYPE_CHECKING:
    from ...state.viewer_state import ViewerState


def process_data(
    data: np.ndarray,
    sfreq: float,
    notch_enabled: bool = False,
    notch_freq: float = 50.0,
    bandpass_enabled: bool = False,
    bandpass_low: float = 1.0,
    bandpass_high: float = 40.0
) -> np.ndarray:
    """
    Apply configured filters to EEG data.
    
    This is the main processing function that applies filters
    based on the provided parameters.
    
    Args:
        data: EEG data array (channels x samples)
        sfreq: Sampling frequency in Hz
        notch_enabled: Whether to apply notch filter
        notch_freq: Notch filter frequency (50 or 60 Hz typically)
        bandpass_enabled: Whether to apply bandpass filter
        bandpass_low: Bandpass lower cutoff
        bandpass_high: Bandpass upper cutoff
        
    Returns:
        Processed data with same shape as input
        
    Example:
        >>> processed = process_data(
        ...     eeg_data, sfreq=250,
        ...     notch_enabled=True, notch_freq=50,
        ...     bandpass_enabled=True, bandpass_low=1, bandpass_high=40
        ... )
    """
    result = data.copy()
    
    if notch_enabled:
        result = apply_notch(result, sfreq, notch_freq)
    
    if bandpass_enabled:
        result = apply_bandpass(result, sfreq, bandpass_low, bandpass_high)
    
    return result


def process_with_state(
    data: np.ndarray,
    sfreq: float,
    state: ViewerState
) -> np.ndarray:
    """
    Apply filters based on ViewerState settings.
    
    Convenience wrapper that extracts filter settings from state.
    
    Args:
        data: EEG data array (channels x samples)
        sfreq: Sampling frequency in Hz
        state: ViewerState containing filter settings
        
    Returns:
        Processed data with same shape as input
    """
    return process_data(
        data=data,
        sfreq=sfreq,
        notch_enabled=state.notch_enabled,
        notch_freq=state.notch_freq,
        bandpass_enabled=state.bandpass_enabled,
        bandpass_low=state.bandpass_low,
        bandpass_high=state.bandpass_high
    )


def get_channel_data(
    raw,  # mne.io.Raw
    channels: list[str],
    start_time: float,
    duration: float,
    process: bool = True,
    **filter_kwargs
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract and optionally process channel data from MNE Raw object.
    
    Args:
        raw: MNE Raw object
        channels: List of channel names to extract
        start_time: Start time in seconds
        duration: Duration in seconds
        process: Whether to apply filters
        **filter_kwargs: Filter parameters for process_data
        
    Returns:
        Tuple of (times, data) where:
        - times: Time array in seconds
        - data: EEG data array (channels x samples)
    """
    sfreq = raw.info['sfreq']
    start_sample = int(start_time * sfreq)
    n_samples = int(duration * sfreq)
    end_sample = min(start_sample + n_samples, raw.n_times)
    
    # Get channel indices
    picks = [raw.ch_names.index(ch) for ch in channels if ch in raw.ch_names]
    
    if not picks:
        return np.array([]), np.array([])
    
    # Extract data
    data, times = raw[picks, start_sample:end_sample]
    
    # Process if requested
    if process and filter_kwargs:
        data = process_data(data, sfreq, **filter_kwargs)
    
    return times, data


def extract_epochs(
    raw,  # mne.io.Raw
    duration: float,
    overlap: float = 0.0,
    channels: Optional[list[str]] = None,
    **filter_kwargs
) -> list[np.ndarray]:
    """
    Extract overlapping epochs from continuous EEG data.
    
    Args:
        raw: MNE Raw object
        duration: Epoch duration in seconds
        overlap: Overlap between epochs in seconds
        channels: Channels to include (None = all)
        **filter_kwargs: Filter parameters for process_data
        
    Returns:
        List of epoch arrays, each (channels x samples)
        
    Example:
        >>> epochs = extract_epochs(raw, duration=2.0, overlap=0.5)
        >>> print(f"Generated {len(epochs)} epochs")
    """
    sfreq = raw.info['sfreq']
    total_duration = raw.n_times / sfreq
    
    step = duration - overlap
    epochs = []
    
    start = 0.0
    while start + duration <= total_duration:
        times, data = get_channel_data(
            raw, 
            channels or raw.ch_names,
            start_time=start,
            duration=duration,
            process=bool(filter_kwargs),
            **filter_kwargs
        )
        if data.size > 0:
            epochs.append(data)
        start += step
    
    return epochs

