"""
Signal transform functions for EEG analysis.

Pure functions for computing FFT, Hilbert transform, and other analyses.
All functions are stateless and can be easily tested.
"""
from __future__ import annotations
import numpy as np
from scipy import signal as scipy_signal
from typing import Tuple


def compute_fft(
    data: np.ndarray, 
    sfreq: float,
    window: str = 'hann'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the FFT of EEG data.
    
    Args:
        data: EEG data array (channels x samples) or (samples,)
        sfreq: Sampling frequency in Hz
        window: Window function to apply (default: 'hann')
        
    Returns:
        Tuple of (frequencies, fft_magnitudes)
        - frequencies: Array of frequency bins in Hz
        - fft_magnitudes: Magnitude spectrum (same dims as data, but last axis is half length)
        
    Example:
        >>> freqs, mags = compute_fft(eeg_data, sfreq=250)
        >>> plt.plot(freqs, mags[0])  # Plot first channel
    """
    n = data.shape[-1]
    
    # Apply window
    if window:
        win = scipy_signal.get_window(window, n)
        if data.ndim == 1:
            data = data * win
        else:
            data = data * win[np.newaxis, :]
    
    # Compute FFT
    fft_vals = np.fft.rfft(data, axis=-1)
    fft_mags = np.abs(fft_vals) * 2 / n
    
    # Frequency bins
    freqs = np.fft.rfftfreq(n, 1 / sfreq)
    
    return freqs, fft_mags


def compute_hilbert(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Hilbert transform to extract amplitude envelope and instantaneous phase.
    
    Args:
        data: EEG data array (channels x samples) or (samples,)
        
    Returns:
        Tuple of (amplitude_envelope, instantaneous_phase)
        - amplitude_envelope: Magnitude of analytic signal
        - instantaneous_phase: Phase in radians (-π to π)
        
    Example:
        >>> amp, phase = compute_hilbert(eeg_data)
        >>> plt.plot(amp[0])  # Plot envelope of first channel
    """
    analytic = scipy_signal.hilbert(data, axis=-1)
    amplitude = np.abs(analytic)
    phase = np.angle(analytic)
    return amplitude, phase


def compute_psd(
    data: np.ndarray,
    sfreq: float,
    nperseg: int = 256,
    noverlap: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Power Spectral Density using Welch's method.
    
    Args:
        data: EEG data array (channels x samples) or (samples,)
        sfreq: Sampling frequency in Hz
        nperseg: Length of each segment (default: 256)
        noverlap: Number of overlapping points (default: nperseg // 2)
        
    Returns:
        Tuple of (frequencies, psd)
        - frequencies: Array of frequency bins in Hz
        - psd: Power spectral density
    """
    if noverlap is None:
        noverlap = nperseg // 2
    
    freqs, psd = scipy_signal.welch(
        data, 
        fs=sfreq, 
        nperseg=nperseg, 
        noverlap=noverlap,
        axis=-1
    )
    return freqs, psd


def compute_band_power(
    data: np.ndarray,
    sfreq: float,
    band: Tuple[float, float]
) -> np.ndarray:
    """
    Compute power in a specific frequency band.
    
    Args:
        data: EEG data array (channels x samples)
        sfreq: Sampling frequency in Hz
        band: Tuple of (low_freq, high_freq) in Hz
        
    Returns:
        Band power for each channel
        
    Example:
        >>> alpha_power = compute_band_power(eeg_data, sfreq=250, band=(8, 13))
    """
    freqs, psd = compute_psd(data, sfreq)
    
    # Find frequency indices in band
    idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    
    # Integrate power in band
    if data.ndim == 1:
        return np.trapz(psd[idx], freqs[idx])
    else:
        return np.trapz(psd[:, idx], freqs[idx], axis=-1)


# Standard EEG frequency bands
FREQUENCY_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 100),
    'low_gamma': (30, 50),
    'high_gamma': (50, 100),
}


def compute_all_band_powers(
    data: np.ndarray,
    sfreq: float
) -> dict:
    """
    Compute power in all standard EEG frequency bands.
    
    Args:
        data: EEG data array (channels x samples)
        sfreq: Sampling frequency in Hz
        
    Returns:
        Dictionary mapping band names to power arrays
        
    Example:
        >>> powers = compute_all_band_powers(eeg_data, sfreq=250)
        >>> print(powers['alpha'])  # Alpha power per channel
    """
    return {
        name: compute_band_power(data, sfreq, band)
        for name, band in FREQUENCY_BANDS.items()
    }

