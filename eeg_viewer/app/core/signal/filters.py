"""
Signal filtering functions for EEG data.

Pure functions for applying various filters to EEG signals.
All functions are stateless and can be easily tested.
"""
from __future__ import annotations
import numpy as np
from scipy import signal


def apply_notch(
    data: np.ndarray, 
    sfreq: float, 
    freq: float = 50.0, 
    quality: float = 30.0
) -> np.ndarray:
    """
    Apply a notch filter to remove power line interference.
    
    Args:
        data: EEG data array (channels x samples)
        sfreq: Sampling frequency in Hz
        freq: Frequency to remove (default: 50 Hz for EU, use 60 for US)
        quality: Q factor of the notch filter (higher = narrower notch)
        
    Returns:
        Filtered data with same shape as input
        
    Example:
        >>> filtered = apply_notch(eeg_data, sfreq=250, freq=50)
    """
    b, a = signal.iirnotch(freq, quality, sfreq)
    return signal.filtfilt(b, a, data, axis=-1)


def apply_bandpass(
    data: np.ndarray,
    sfreq: float,
    low_freq: float,
    high_freq: float,
    order: int = 4
) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter.
    
    Args:
        data: EEG data array (channels x samples)
        sfreq: Sampling frequency in Hz
        low_freq: Lower cutoff frequency in Hz
        high_freq: Upper cutoff frequency in Hz
        order: Filter order (default: 4)
        
    Returns:
        Filtered data with same shape as input
        
    Example:
        >>> filtered = apply_bandpass(eeg_data, sfreq=250, low_freq=1, high_freq=40)
    """
    nyq = sfreq / 2
    low = low_freq / nyq
    high = high_freq / nyq
    
    # Clamp to valid range
    low = max(0.001, min(low, 0.999))
    high = max(low + 0.001, min(high, 0.999))
    
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data, axis=-1)


def apply_lowpass(
    data: np.ndarray,
    sfreq: float,
    cutoff: float,
    order: int = 4
) -> np.ndarray:
    """
    Apply a Butterworth lowpass filter.
    
    Args:
        data: EEG data array (channels x samples)
        sfreq: Sampling frequency in Hz
        cutoff: Cutoff frequency in Hz
        order: Filter order (default: 4)
        
    Returns:
        Filtered data with same shape as input
    """
    nyq = sfreq / 2
    normalized_cutoff = min(cutoff / nyq, 0.999)
    b, a = signal.butter(order, normalized_cutoff, btype='low')
    return signal.filtfilt(b, a, data, axis=-1)


def apply_highpass(
    data: np.ndarray,
    sfreq: float,
    cutoff: float,
    order: int = 4
) -> np.ndarray:
    """
    Apply a Butterworth highpass filter.
    
    Args:
        data: EEG data array (channels x samples)
        sfreq: Sampling frequency in Hz
        cutoff: Cutoff frequency in Hz
        order: Filter order (default: 4)
        
    Returns:
        Filtered data with same shape as input
    """
    nyq = sfreq / 2
    normalized_cutoff = max(cutoff / nyq, 0.001)
    b, a = signal.butter(order, normalized_cutoff, btype='high')
    return signal.filtfilt(b, a, data, axis=-1)

