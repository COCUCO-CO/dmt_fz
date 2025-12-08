"""
Signal processing module for EEG data.

Provides pure functions for filtering, transforming, and processing EEG signals.
"""
from .filters import (
    apply_notch,
    apply_bandpass,
    apply_lowpass,
    apply_highpass,
)
from .transforms import (
    compute_fft,
    compute_hilbert,
    compute_psd,
    compute_band_power,
    compute_all_band_powers,
    FREQUENCY_BANDS,
)
from .processing import (
    process_data,
    process_with_state,
    get_channel_data,
    extract_epochs,
)

__all__ = [
    # Filters
    'apply_notch',
    'apply_bandpass',
    'apply_lowpass',
    'apply_highpass',
    # Transforms
    'compute_fft',
    'compute_hilbert',
    'compute_psd',
    'compute_band_power',
    'compute_all_band_powers',
    'FREQUENCY_BANDS',
    # Processing
    'process_data',
    'process_with_state',
    'get_channel_data',
    'extract_epochs',
]

