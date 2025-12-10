"""
Tests for app/core/signal module.

Tests signal processing functions: filters, transforms, and processing.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.signal import (
    # Filters
    apply_notch,
    apply_bandpass,
    apply_lowpass,
    apply_highpass,
    # Transforms
    compute_fft,
    compute_hilbert,
    compute_psd,
    compute_band_power,
    compute_all_band_powers,
    FREQUENCY_BANDS,
    # Processing
    process_data,
)


class TestFilters:
    """Tests for signal filters."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample EEG-like data."""
        np.random.seed(42)
        sfreq = 250.0
        duration = 2.0
        n_channels = 5
        n_samples = int(sfreq * duration)
        
        # Create signal with known frequency components
        t = np.linspace(0, duration, n_samples)
        data = np.zeros((n_channels, n_samples))
        for i in range(n_channels):
            # Add 10 Hz sine + 50 Hz noise
            data[i] = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)
            data[i] += np.random.randn(n_samples) * 0.1
        
        return data, sfreq
    
    def test_apply_notch_shape_preserved(self, sample_data):
        """Test notch filter preserves data shape."""
        data, sfreq = sample_data
        filtered = apply_notch(data, sfreq, freq=50.0)
        
        assert filtered.shape == data.shape
    
    def test_apply_notch_removes_frequency(self, sample_data):
        """Test notch filter attenuates target frequency."""
        data, sfreq = sample_data
        
        # Get power at 50 Hz before and after
        freqs_before, fft_before = compute_fft(data, sfreq)
        filtered = apply_notch(data, sfreq, freq=50.0)
        freqs_after, fft_after = compute_fft(filtered, sfreq)
        
        # Find index closest to 50 Hz
        idx_50 = np.argmin(np.abs(freqs_before - 50))
        
        # Power at 50 Hz should be reduced
        power_before = np.mean(fft_before[:, idx_50])
        power_after = np.mean(fft_after[:, idx_50])
        
        assert power_after < power_before * 0.5  # At least 50% reduction
    
    def test_apply_bandpass_shape_preserved(self, sample_data):
        """Test bandpass filter preserves data shape."""
        data, sfreq = sample_data
        filtered = apply_bandpass(data, sfreq, low_freq=1.0, high_freq=40.0)
        
        assert filtered.shape == data.shape
    
    def test_apply_bandpass_attenuates_outside(self, sample_data):
        """Test bandpass attenuates frequencies outside band."""
        data, sfreq = sample_data
        
        # Filter 5-15 Hz (should keep 10 Hz, remove 50 Hz)
        filtered = apply_bandpass(data, sfreq, low_freq=5.0, high_freq=15.0)
        
        freqs, fft_vals = compute_fft(filtered, sfreq)
        
        # Find indices
        idx_10 = np.argmin(np.abs(freqs - 10))
        idx_50 = np.argmin(np.abs(freqs - 50))
        
        # 10 Hz should be stronger than 50 Hz
        power_10 = np.mean(fft_vals[:, idx_10])
        power_50 = np.mean(fft_vals[:, idx_50])
        
        assert power_10 > power_50 * 10  # Much stronger at 10 Hz
    
    def test_apply_lowpass(self, sample_data):
        """Test lowpass filter."""
        data, sfreq = sample_data
        filtered = apply_lowpass(data, sfreq, cutoff=20.0)
        
        assert filtered.shape == data.shape
        
        # High frequencies should be attenuated
        freqs, fft_vals = compute_fft(filtered, sfreq)
        idx_50 = np.argmin(np.abs(freqs - 50))
        power_50 = np.mean(fft_vals[:, idx_50])
        
        freqs_orig, fft_orig = compute_fft(data, sfreq)
        power_50_orig = np.mean(fft_orig[:, idx_50])
        
        assert power_50 < power_50_orig * 0.1
    
    def test_apply_highpass(self, sample_data):
        """Test highpass filter."""
        data, sfreq = sample_data
        
        # Add DC offset
        data_with_dc = data + 5.0
        filtered = apply_highpass(data_with_dc, sfreq, cutoff=1.0)
        
        assert filtered.shape == data_with_dc.shape
        
        # DC should be removed (mean should be near 0)
        assert np.abs(np.mean(filtered)) < 1.0


class TestTransforms:
    """Tests for signal transforms."""
    
    @pytest.fixture
    def sine_wave(self):
        """Generate pure sine wave."""
        sfreq = 250.0
        duration = 2.0
        freq = 10.0
        t = np.linspace(0, duration, int(sfreq * duration))
        data = np.sin(2 * np.pi * freq * t)
        return data, sfreq, freq
    
    @pytest.fixture
    def multichannel_data(self):
        """Generate multichannel data."""
        np.random.seed(42)
        return np.random.randn(5, 500), 250.0
    
    def test_compute_fft_output_shape(self, multichannel_data):
        """Test FFT output shapes."""
        data, sfreq = multichannel_data
        freqs, mags = compute_fft(data, sfreq)
        
        assert freqs.ndim == 1
        assert mags.shape[0] == data.shape[0]  # Same number of channels
        assert len(freqs) == mags.shape[1]
    
    def test_compute_fft_peak_detection(self, sine_wave):
        """Test FFT detects correct frequency peak."""
        data, sfreq, expected_freq = sine_wave
        freqs, mags = compute_fft(data, sfreq, window=None)
        
        # Find peak frequency
        peak_idx = np.argmax(mags)
        peak_freq = freqs[peak_idx]
        
        # Should be close to 10 Hz
        assert np.abs(peak_freq - expected_freq) < 1.0
    
    def test_compute_hilbert_output_shape(self, multichannel_data):
        """Test Hilbert transform output shapes."""
        data, _ = multichannel_data
        amp, phase = compute_hilbert(data)
        
        assert amp.shape == data.shape
        assert phase.shape == data.shape
    
    def test_compute_hilbert_amplitude_positive(self, multichannel_data):
        """Test Hilbert amplitude is always positive."""
        data, _ = multichannel_data
        amp, _ = compute_hilbert(data)
        
        assert np.all(amp >= 0)
    
    def test_compute_hilbert_phase_range(self, multichannel_data):
        """Test Hilbert phase is in valid range."""
        data, _ = multichannel_data
        _, phase = compute_hilbert(data)
        
        assert np.all(phase >= -np.pi)
        assert np.all(phase <= np.pi)
    
    def test_compute_psd_output(self, multichannel_data):
        """Test PSD computation."""
        data, sfreq = multichannel_data
        freqs, psd = compute_psd(data, sfreq)
        
        assert freqs.ndim == 1
        assert psd.shape[0] == data.shape[0]
        assert np.all(psd >= 0)  # PSD is always positive
    
    def test_compute_band_power(self, multichannel_data):
        """Test band power computation."""
        data, sfreq = multichannel_data
        
        # Compute alpha band power (8-13 Hz)
        alpha_power = compute_band_power(data, sfreq, band=(8, 13))
        
        assert alpha_power.shape == (data.shape[0],)
        assert np.all(alpha_power >= 0)
    
    def test_compute_all_band_powers(self, multichannel_data):
        """Test computing all frequency band powers."""
        data, sfreq = multichannel_data
        
        powers = compute_all_band_powers(data, sfreq)
        
        assert 'alpha' in powers
        assert 'beta' in powers
        assert 'delta' in powers
        assert 'theta' in powers
        assert 'gamma' in powers
    
    def test_frequency_bands_defined(self):
        """Test standard frequency bands are defined."""
        assert 'delta' in FREQUENCY_BANDS
        assert 'theta' in FREQUENCY_BANDS
        assert 'alpha' in FREQUENCY_BANDS
        assert 'beta' in FREQUENCY_BANDS
        assert 'gamma' in FREQUENCY_BANDS
        
        # Check alpha band is reasonable
        assert FREQUENCY_BANDS['alpha'] == (8, 13)


class TestProcessing:
    """Tests for high-level processing functions."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data."""
        np.random.seed(42)
        return np.random.randn(5, 500), 250.0
    
    def test_process_data_no_filters(self, sample_data):
        """Test process_data with no filters returns copy."""
        data, sfreq = sample_data
        
        processed = process_data(data, sfreq)
        
        assert processed.shape == data.shape
        np.testing.assert_array_almost_equal(processed, data)
    
    def test_process_data_with_notch(self, sample_data):
        """Test process_data with notch filter."""
        data, sfreq = sample_data
        
        processed = process_data(
            data, sfreq,
            notch_enabled=True,
            notch_freq=50.0
        )
        
        assert processed.shape == data.shape
        assert not np.allclose(processed, data)
    
    def test_process_data_with_bandpass(self, sample_data):
        """Test process_data with bandpass filter."""
        data, sfreq = sample_data
        
        processed = process_data(
            data, sfreq,
            bandpass_enabled=True,
            bandpass_low=1.0,
            bandpass_high=40.0
        )
        
        assert processed.shape == data.shape
        assert not np.allclose(processed, data)
    
    def test_process_data_with_both_filters(self, sample_data):
        """Test process_data with both filters."""
        data, sfreq = sample_data
        
        processed = process_data(
            data, sfreq,
            notch_enabled=True,
            notch_freq=50.0,
            bandpass_enabled=True,
            bandpass_low=1.0,
            bandpass_high=40.0
        )
        
        assert processed.shape == data.shape

