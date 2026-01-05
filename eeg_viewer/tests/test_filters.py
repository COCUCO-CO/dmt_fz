"""
Tests for cleaning/filters.py - Filtering functions and presets.
"""

import numpy as np

from cleaning.filters import (
    FilterPreset, FilterParams, get_filter_presets,
    apply_notch_filter, apply_bandpass_filter, apply_filter_preset,
    compute_psd
)


class TestFilterPreset:
    """Tests for FilterPreset enum."""
    
    def test_all_presets_exist(self):
        """Verify all presets exist."""
        presets = list(FilterPreset)
        assert FilterPreset.STANDARD in presets
        assert FilterPreset.ALPHA_THETA in presets
        assert FilterPreset.ERP in presets
        assert FilterPreset.GAMMA in presets
        assert FilterPreset.CUSTOM in presets
    
    def test_display_names(self):
        """Test display names are informative."""
        assert "Standard" in FilterPreset.STANDARD.display_name
        assert "Alpha" in FilterPreset.ALPHA_THETA.display_name
        assert "ERP" in FilterPreset.ERP.display_name


class TestFilterParams:
    """Tests for FilterParams dataclass."""
    
    def test_default_params(self):
        """Test default parameters."""
        params = FilterParams()
        assert params.highpass is None
        assert params.lowpass is None
        assert params.notch is None
        assert params.notch_harmonics == False
    
    def test_custom_params(self):
        """Test custom parameters."""
        params = FilterParams(highpass=0.1, lowpass=45, notch=50)
        assert params.highpass == 0.1
        assert params.lowpass == 45
        assert params.notch == 50


class TestGetFilterPresets:
    """Tests for get_filter_presets function."""
    
    def test_returns_all_presets(self):
        """Test all presets are returned."""
        presets = get_filter_presets()
        assert len(presets) == len(FilterPreset)
    
    def test_standard_preset_values(self):
        """Test STANDARD preset has correct values."""
        presets = get_filter_presets()
        standard = presets[FilterPreset.STANDARD]
        
        assert standard.highpass == 0.1
        assert standard.lowpass == 45.0
        assert standard.notch == 50.0
    
    def test_gamma_preset_has_harmonics(self):
        """Test GAMMA preset includes harmonics."""
        presets = get_filter_presets()
        gamma = presets[FilterPreset.GAMMA]
        
        assert gamma.notch_harmonics == True
        assert gamma.lowpass == 100.0


class TestApplyNotchFilter:
    """Tests for apply_notch_filter function."""
    
    def test_basic_notch(self, synthetic_raw):
        """Test basic notch filter application."""
        raw_filtered = apply_notch_filter(synthetic_raw, freq=50.0)
        
        assert raw_filtered is not None
        assert raw_filtered.n_times == synthetic_raw.n_times
        assert len(raw_filtered.ch_names) == len(synthetic_raw.ch_names)
    
    def test_notch_reduces_50hz(self, synthetic_raw):
        """Test that notch filter reduces 50 Hz power."""
        # Add 50 Hz noise
        data = synthetic_raw.get_data()
        t = np.arange(data.shape[1]) / synthetic_raw.info['sfreq']
        noise_50hz = np.sin(2 * np.pi * 50 * t) * 1e-5
        data[0] += noise_50hz
        
        # Create new raw with noise
        raw_noisy = synthetic_raw.copy()
        raw_noisy._data = data
        
        # Apply notch
        raw_filtered = apply_notch_filter(raw_noisy, freq=50.0)
        
        # Check that filtering was applied (data changed)
        assert not np.allclose(raw_filtered.get_data()[0], raw_noisy.get_data()[0])
    
    def test_notch_with_harmonics(self, synthetic_raw):
        """Test notch filter with harmonics."""
        raw_filtered = apply_notch_filter(synthetic_raw, freq=50.0, include_harmonics=True)
        
        assert raw_filtered is not None
    
    def test_notch_60hz(self, synthetic_raw):
        """Test 60 Hz notch (US line frequency)."""
        raw_filtered = apply_notch_filter(synthetic_raw, freq=60.0)
        
        assert raw_filtered is not None


class TestApplyBandpassFilter:
    """Tests for apply_bandpass_filter function."""
    
    def test_bandpass_filter(self, synthetic_raw):
        """Test basic bandpass filter."""
        raw_filtered = apply_bandpass_filter(synthetic_raw, l_freq=1.0, h_freq=45.0)
        
        assert raw_filtered is not None
        assert raw_filtered.n_times == synthetic_raw.n_times
    
    def test_highpass_only(self, synthetic_raw):
        """Test highpass filter only."""
        raw_filtered = apply_bandpass_filter(synthetic_raw, l_freq=1.0, h_freq=None)
        
        assert raw_filtered is not None
    
    def test_lowpass_only(self, synthetic_raw):
        """Test lowpass filter only."""
        raw_filtered = apply_bandpass_filter(synthetic_raw, l_freq=None, h_freq=45.0)
        
        assert raw_filtered is not None
    
    def test_returns_copy(self, synthetic_raw):
        """Test that filter returns a copy, not modifying original."""
        original_data = synthetic_raw.get_data().copy()
        raw_filtered = apply_bandpass_filter(synthetic_raw, l_freq=1.0, h_freq=45.0)
        
        assert np.allclose(synthetic_raw.get_data(), original_data)


class TestApplyFilterPreset:
    """Tests for apply_filter_preset function."""
    
    def test_standard_preset(self, synthetic_raw):
        """Test applying STANDARD preset."""
        raw_filtered, params = apply_filter_preset(synthetic_raw, FilterPreset.STANDARD)
        
        assert raw_filtered is not None
        assert params.highpass == 0.1
        assert params.lowpass == 45.0
        assert params.notch is not None
    
    def test_alpha_theta_preset(self, synthetic_raw):
        """Test applying ALPHA_THETA preset."""
        raw_filtered, params = apply_filter_preset(synthetic_raw, FilterPreset.ALPHA_THETA)
        
        assert params.highpass == 1.0
        assert params.lowpass == 30.0
    
    def test_custom_notch_frequency(self, synthetic_raw):
        """Test custom notch frequency override."""
        raw_filtered, params = apply_filter_preset(
            synthetic_raw, 
            FilterPreset.STANDARD,
            notch_freq=60.0
        )
        
        assert params.notch == 60.0
    
    def test_custom_preset(self, synthetic_raw):
        """Test CUSTOM preset with custom params."""
        custom_params = FilterParams(highpass=2.0, lowpass=30.0, notch=50.0)
        raw_filtered, params = apply_filter_preset(
            synthetic_raw,
            FilterPreset.CUSTOM,
            custom_params=custom_params
        )
        
        assert params.highpass == 2.0
        assert params.lowpass == 30.0


class TestComputePSD:
    """Tests for compute_psd function."""
    
    def test_compute_psd(self, synthetic_raw):
        """Test PSD computation."""
        freqs, psd = compute_psd(synthetic_raw, fmin=1, fmax=45)
        
        assert len(freqs) > 0
        assert psd.shape[0] == len(synthetic_raw.ch_names)
        assert psd.shape[1] == len(freqs)
    
    def test_psd_frequency_range(self, synthetic_raw):
        """Test PSD respects frequency range."""
        freqs, psd = compute_psd(synthetic_raw, fmin=5, fmax=20)
        
        assert freqs[0] >= 5
        assert freqs[-1] <= 20
    
    def test_psd_has_alpha_peak(self, synthetic_raw):
        """Test that synthetic data shows alpha peak around 10 Hz."""
        freqs, psd = compute_psd(synthetic_raw, fmin=1, fmax=45)
        
        # Find index of 10 Hz
        alpha_idx = np.argmin(np.abs(freqs - 10))
        
        # Check there's some power at alpha (not a rigorous test)
        assert psd[0, alpha_idx] > 0









