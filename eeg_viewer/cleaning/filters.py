"""
EEG Filtering Module

Provides filtering functions and presets for EEG preprocessing.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
from enum import Enum
import numpy as np
from scipy import signal
import mne


class FilterPreset(Enum):
    """Predefined filter presets for common use cases."""
    STANDARD = "standard"
    ALPHA_THETA = "alpha_theta"
    ERP = "erp"
    GAMMA = "gamma"
    CUSTOM = "custom"
    
    @property
    def display_name(self) -> str:
        names = {
            FilterPreset.STANDARD: "Standard (0.1-45 Hz)",
            FilterPreset.ALPHA_THETA: "Alpha/Theta (1-30 Hz)",
            FilterPreset.ERP: "ERP Analysis (0.1-40 Hz)",
            FilterPreset.GAMMA: "Gamma (1-100 Hz)",
            FilterPreset.CUSTOM: "Custom",
        }
        return names.get(self, self.name)


@dataclass
class FilterParams:
    """Filter parameters."""
    highpass: Optional[float] = None  # Hz
    lowpass: Optional[float] = None   # Hz
    notch: Optional[float] = None     # Hz (50 or 60)
    notch_harmonics: bool = False     # Include 100/120, 150/180 Hz etc


def get_filter_presets() -> dict:
    """Get all available filter presets with their parameters."""
    return {
        FilterPreset.STANDARD: FilterParams(
            highpass=0.1,
            lowpass=45.0,
            notch=50.0,
            notch_harmonics=False
        ),
        FilterPreset.ALPHA_THETA: FilterParams(
            highpass=1.0,
            lowpass=30.0,
            notch=50.0,
            notch_harmonics=False
        ),
        FilterPreset.ERP: FilterParams(
            highpass=0.1,
            lowpass=40.0,
            notch=50.0,
            notch_harmonics=False
        ),
        FilterPreset.GAMMA: FilterParams(
            highpass=1.0,
            lowpass=100.0,
            notch=50.0,
            notch_harmonics=True
        ),
        FilterPreset.CUSTOM: FilterParams(),
    }


def apply_notch_filter(raw: mne.io.Raw, freq: float = 50.0, 
                       include_harmonics: bool = False,
                       quality: float = 30.0) -> mne.io.Raw:
    """
    Apply notch filter to remove line noise.
    
    Args:
        raw: MNE Raw object
        freq: Notch frequency (50 or 60 Hz)
        include_harmonics: If True, also filter at 2x, 3x frequency
        quality: Q factor for the notch filter
        
    Returns:
        Filtered Raw object (copy)
    """
    raw_filtered = raw.copy()
    
    freqs = [freq]
    if include_harmonics:
        nyquist = raw.info['sfreq'] / 2
        harmonic = freq * 2
        while harmonic < nyquist - 5:
            freqs.append(harmonic)
            harmonic += freq
    
    raw_filtered.notch_filter(
        freqs=freqs,
        filter_length='auto',
        notch_widths=freq / quality,
        verbose=False
    )
    
    return raw_filtered


def apply_bandpass_filter(raw: mne.io.Raw, 
                          l_freq: Optional[float] = None,
                          h_freq: Optional[float] = None,
                          method: str = 'fir') -> mne.io.Raw:
    """
    Apply bandpass filter.
    
    Args:
        raw: MNE Raw object
        l_freq: Low cutoff frequency (highpass)
        h_freq: High cutoff frequency (lowpass)
        method: 'fir' or 'iir'
        
    Returns:
        Filtered Raw object (copy)
    """
    raw_filtered = raw.copy()
    
    raw_filtered.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method=method,
        filter_length='auto',
        verbose=False
    )
    
    return raw_filtered


def apply_filter_preset(raw: mne.io.Raw, 
                        preset: FilterPreset,
                        custom_params: Optional[FilterParams] = None,
                        notch_freq: float = 50.0) -> Tuple[mne.io.Raw, FilterParams]:
    """
    Apply a filter preset to raw EEG data.
    
    Args:
        raw: MNE Raw object
        preset: Filter preset to apply
        custom_params: Custom parameters (only used if preset is CUSTOM)
        notch_freq: Notch frequency (50 or 60 Hz, region-dependent)
        
    Returns:
        Tuple of (filtered Raw, applied FilterParams)
    """
    presets = get_filter_presets()
    
    if preset == FilterPreset.CUSTOM and custom_params is not None:
        params = custom_params
    else:
        params = presets[preset]
        # Update notch frequency based on region
        if params.notch is not None:
            params = FilterParams(
                highpass=params.highpass,
                lowpass=params.lowpass,
                notch=notch_freq,
                notch_harmonics=params.notch_harmonics
            )
    
    raw_filtered = raw.copy()
    
    # Apply bandpass
    if params.highpass is not None or params.lowpass is not None:
        raw_filtered = apply_bandpass_filter(
            raw_filtered,
            l_freq=params.highpass,
            h_freq=params.lowpass
        )
    
    # Apply notch
    if params.notch is not None:
        raw_filtered = apply_notch_filter(
            raw_filtered,
            freq=params.notch,
            include_harmonics=params.notch_harmonics
        )
    
    return raw_filtered, params


def compute_psd(raw: mne.io.Raw, 
                fmin: float = 0.5, 
                fmax: float = 50.0,
                n_fft: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectral density.
    
    Args:
        raw: MNE Raw object
        fmin: Minimum frequency
        fmax: Maximum frequency
        n_fft: FFT length (None = auto based on data length)
        
    Returns:
        Tuple of (frequencies, psd array [n_channels x n_freqs])
    """
    # Auto-adjust n_fft based on data length
    if n_fft is None:
        n_fft = min(2048, raw.n_times)
    else:
        n_fft = min(n_fft, raw.n_times)
    
    spectrum = raw.compute_psd(
        method='welch',
        fmin=fmin,
        fmax=fmax,
        n_fft=n_fft,
        verbose=False
    )
    
    return spectrum.freqs, spectrum.get_data()

