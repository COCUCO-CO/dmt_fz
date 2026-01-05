"""
Pytest configuration and shared fixtures for cleaning module tests.
"""

import pytest
import numpy as np
import mne
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cleaning.state import CleaningState


# Test data paths
TEST_EEG_PATH = Path("/media/storage_hdd/dmt_fz/EEG/DMT/S01-DMT.bdf")


@pytest.fixture(scope="session")
def sample_raw():
    """
    Load a short segment of real EEG data for testing.
    Uses only first 30 seconds to keep tests fast.
    """
    if not TEST_EEG_PATH.exists():
        pytest.skip(f"Test EEG file not found: {TEST_EEG_PATH}")
    
    # Load raw data
    raw = mne.io.read_raw_bdf(TEST_EEG_PATH, preload=False, verbose=False)
    
    # Crop to first 30 seconds for speed
    raw = raw.crop(tmin=0, tmax=30)
    raw.load_data()
    
    # Pick only EEG channels (exclude Status, etc.)
    raw.pick_types(eeg=True, stim=False, exclude=['Status'])
    
    return raw


@pytest.fixture(scope="session")
def sample_raw_short():
    """
    Even shorter segment (10 seconds) for very fast tests.
    """
    if not TEST_EEG_PATH.exists():
        pytest.skip(f"Test EEG file not found: {TEST_EEG_PATH}")
    
    raw = mne.io.read_raw_bdf(TEST_EEG_PATH, preload=False, verbose=False)
    raw = raw.crop(tmin=0, tmax=10)
    raw.load_data()
    raw.pick_types(eeg=True, stim=False, exclude=['Status'])
    
    return raw


@pytest.fixture
def cleaning_state():
    """Fresh CleaningState for each test."""
    return CleaningState()


@pytest.fixture
def loaded_state(sample_raw):
    """CleaningState with loaded EEG data."""
    state = CleaningState()
    state.load_raw(sample_raw, TEST_EEG_PATH)
    return state


@pytest.fixture
def synthetic_raw():
    """
    Create synthetic EEG data for tests that don't need real data.
    Much faster than loading real data.
    """
    # Create synthetic data
    sfreq = 256
    duration = 5  # seconds
    n_channels = 8
    n_samples = int(sfreq * duration)
    
    # Generate random EEG-like data
    np.random.seed(42)
    data = np.random.randn(n_channels, n_samples) * 1e-5  # Scale to realistic EEG amplitude
    
    # Add some structure (alpha-like oscillation)
    t = np.arange(n_samples) / sfreq
    for i in range(n_channels):
        data[i] += np.sin(2 * np.pi * 10 * t) * 2e-5  # 10 Hz alpha
    
    # Create channel names
    ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'O1', 'O2']
    ch_types = ['eeg'] * n_channels
    
    # Create info
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    
    # Create Raw object
    raw = mne.io.RawArray(data, info, verbose=False)
    
    # Set montage for interpolation tests
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='ignore')
    
    return raw


@pytest.fixture
def synthetic_raw_with_artifacts():
    """
    Synthetic EEG with artificial artifacts for testing detection.
    """
    sfreq = 256
    duration = 10
    n_channels = 8
    n_samples = int(sfreq * duration)
    
    np.random.seed(42)
    data = np.random.randn(n_channels, n_samples) * 1e-5
    
    t = np.arange(n_samples) / sfreq
    
    # Add alpha oscillation
    for i in range(n_channels):
        data[i] += np.sin(2 * np.pi * 10 * t) * 2e-5
    
    # Make channel 0 flat (bad channel)
    data[0] = np.zeros(n_samples)
    
    # Make channel 1 very noisy (bad channel)
    data[1] = np.random.randn(n_samples) * 1e-3
    
    # Add blink artifacts to frontal channels (2, 3)
    blink_times = [1.0, 3.0, 5.0, 7.0, 9.0]
    for blink_t in blink_times:
        blink_idx = int(blink_t * sfreq)
        blink_width = int(0.2 * sfreq)
        blink_shape = np.exp(-0.5 * ((np.arange(blink_width) - blink_width//2) / (blink_width/6))**2)
        data[2, blink_idx:blink_idx+blink_width] += blink_shape * 1e-4
        data[3, blink_idx:blink_idx+blink_width] += blink_shape * 1e-4
    
    ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'O1', 'O2']
    ch_types = ['eeg'] * n_channels
    
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info, verbose=False)
    
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='ignore')
    
    return raw









