"""
SOLID tests using real EEG data.

These tests load actual EEG files and run pipeline processing to verify:
1. Real data can be loaded and processed
2. Output dimensions match expected for real data
3. Values are in expected ranges for real EEG
4. Processing is consistent and reproducible
"""

import pytest
import sys
import numpy as np
from pathlib import Path
import mne

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "pipeline"))


# =============================================================================
# Real EEG Data Fixtures
# =============================================================================

REAL_EEG_PATH = Path("/media/storage_hdd/dmt_fz/EEG_CLEAN/DMT/S01-DMT_ICA_pruned.set")


@pytest.fixture(scope="module")
def real_epochs():
    """Load real cleaned EEG epochs."""
    if not REAL_EEG_PATH.exists():
        pytest.skip(f"Real EEG file not found: {REAL_EEG_PATH}")
    
    epochs = mne.io.read_epochs_eeglab(str(REAL_EEG_PATH), montage_units='dm', verbose=False)
    
    # Prepare epochs like the pipeline does
    non_eeg = [ch for ch in epochs.ch_names if ch.upper() in ['STATUS', 'STI 014', 'STI014', 'TRIGGER']]
    if non_eeg:
        epochs.drop_channels(non_eeg)
    epochs.pick('eeg')
    
    return epochs


@pytest.fixture(scope="module")
def real_epochs_limited(real_epochs):
    """Return limited epochs for faster tests."""
    # Only use first 3 epochs for speed
    return real_epochs[:3]


# =============================================================================
# Test: Real Data Loading
# =============================================================================

class TestRealDataLoading:
    """Test loading of real EEG data."""
    
    def test_epochs_load_successfully(self, real_epochs):
        """Real epochs should load without error."""
        assert real_epochs is not None
    
    def test_epochs_have_expected_sfreq(self, real_epochs):
        """Sampling frequency should be reasonable (100-1000 Hz)."""
        sfreq = real_epochs.info['sfreq']
        assert 100 <= sfreq <= 1000, f"Unexpected sfreq: {sfreq}"
    
    def test_epochs_have_multiple_channels(self, real_epochs):
        """Should have multiple EEG channels."""
        n_channels = len(real_epochs.ch_names)
        assert n_channels >= 10, f"Too few channels: {n_channels}"
    
    def test_epochs_have_multiple_trials(self, real_epochs):
        """Should have multiple epochs/trials."""
        n_epochs = len(real_epochs)
        assert n_epochs >= 1, "No epochs found"
    
    def test_epochs_data_shape(self, real_epochs):
        """Epochs data should have correct shape (n_epochs, n_channels, n_samples)."""
        data = real_epochs.get_data()
        assert data.ndim == 3, f"Expected 3D data, got {data.ndim}D"
        assert data.shape[0] == len(real_epochs)  # n_epochs
        assert data.shape[1] == len(real_epochs.ch_names)  # n_channels
    
    def test_epochs_data_not_all_zeros(self, real_epochs):
        """Data should not be all zeros."""
        data = real_epochs.get_data()
        assert not np.allclose(data, 0), "Data is all zeros"
    
    def test_epochs_data_no_nan(self, real_epochs):
        """Data should not contain NaN."""
        data = real_epochs.get_data()
        assert not np.any(np.isnan(data)), "Data contains NaN"


# =============================================================================
# Test: Real Data Filtering
# =============================================================================

class TestRealDataFiltering:
    """Test bandpass filtering on real EEG data."""
    
    @pytest.fixture
    def single_epoch_data(self, real_epochs_limited):
        """Get single epoch as 2D array (channels, samples)."""
        data = real_epochs_limited.get_data()
        return data[0, :, :]  # First epoch
    
    def test_filter_alpha_on_real_data(self, single_epoch_data, real_epochs_limited):
        """Alpha filter should work on real data."""
        from fwd import filtered
        
        sfreq = real_epochs_limited.info['sfreq']
        result = filtered(single_epoch_data, "Alpha", sfreq=sfreq)
        
        assert result.shape == single_epoch_data.shape
        assert not np.any(np.isnan(result)), "Filter produced NaN"
    
    def test_all_bands_work_on_real_data(self, single_epoch_data, real_epochs_limited):
        """All frequency bands should work on real data."""
        from fwd import filtered, band_list
        
        sfreq = real_epochs_limited.info['sfreq']
        
        for band in band_list:
            result = filtered(single_epoch_data, band, sfreq=sfreq)
            assert result.shape == single_epoch_data.shape, f"{band} filter changed shape"
            assert not np.any(np.isnan(result)), f"{band} filter produced NaN"
            assert not np.any(np.isinf(result)), f"{band} filter produced Inf"
    
    def test_filter_reduces_out_of_band_power(self, single_epoch_data, real_epochs_limited):
        """Filter should reduce power outside the passband."""
        from fwd import filtered
        
        sfreq = real_epochs_limited.info['sfreq']
        
        # Use Alpha band (8-13 Hz)
        filtered_signal = filtered(single_epoch_data, "Alpha", sfreq=sfreq)
        
        # Check one channel
        signal = filtered_signal[0, :]
        
        # FFT
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/sfreq)
        power = np.abs(fft) ** 2
        
        # Calculate power in-band vs out-of-band
        in_band_mask = (np.abs(freqs) >= 8) & (np.abs(freqs) <= 13)
        out_band_mask = (np.abs(freqs) > 20)  # Well outside alpha
        
        in_band_power = power[in_band_mask].sum()
        out_band_power = power[out_band_mask].sum()
        
        # In-band should dominate (unless there's very little alpha in the signal)
        ratio = in_band_power / (out_band_power + 1e-10)
        assert ratio > 1, f"Filter didn't suppress out-of-band: ratio = {ratio}"


# =============================================================================
# Test: Real Data Hilbert Transform
# =============================================================================

class TestRealDataHilbert:
    """Test Hilbert transform on real EEG data."""
    
    @pytest.fixture
    def filtered_epoch(self, real_epochs_limited):
        """Get filtered single epoch."""
        from fwd import filtered
        
        data = real_epochs_limited.get_data()[0, :, :]
        sfreq = real_epochs_limited.info['sfreq']
        return filtered(data, "Alpha", sfreq=sfreq)
    
    def test_hilbert_on_real_filtered_data(self, filtered_epoch):
        """Hilbert transform should work on real filtered data."""
        from fwd import hilbert_transform
        
        signals = [filtered_epoch[i, :] for i in range(filtered_epoch.shape[0])]
        amplitude, phase = hilbert_transform(signals, trim=100)
        
        # Check shapes
        n_channels = filtered_epoch.shape[0]
        expected_samples = filtered_epoch.shape[1] - 200
        
        assert amplitude.shape == (n_channels, expected_samples)
        assert phase.shape == (n_channels, expected_samples)
    
    def test_hilbert_amplitude_positive(self, filtered_epoch):
        """Amplitude envelope should be non-negative."""
        from fwd import hilbert_transform
        
        signals = [filtered_epoch[i, :] for i in range(filtered_epoch.shape[0])]
        amplitude, _ = hilbert_transform(signals, trim=100)
        
        assert amplitude.min() >= -1e-10, f"Negative amplitude: {amplitude.min()}"
    
    def test_hilbert_phase_in_range(self, filtered_epoch):
        """Phase should be in [-π, π]."""
        from fwd import hilbert_transform
        
        signals = [filtered_epoch[i, :] for i in range(filtered_epoch.shape[0])]
        _, phase = hilbert_transform(signals, trim=100)
        
        assert phase.min() >= -np.pi - 1e-5, f"Phase < -π: {phase.min()}"
        assert phase.max() <= np.pi + 1e-5, f"Phase > π: {phase.max()}"
    
    def test_hilbert_no_nan_on_real_data(self, filtered_epoch):
        """Hilbert should not produce NaN on real data."""
        from fwd import hilbert_transform
        
        signals = [filtered_epoch[i, :] for i in range(filtered_epoch.shape[0])]
        amplitude, phase = hilbert_transform(signals, trim=100)
        
        assert not np.any(np.isnan(amplitude)), "Amplitude has NaN"
        assert not np.any(np.isnan(phase)), "Phase has NaN"


# =============================================================================
# Test: Real Data Synchronization
# =============================================================================

class TestRealDataSynchronization:
    """Test synchronization calculation on real EEG data."""
    
    @pytest.fixture
    def real_phase_data(self, real_epochs_limited):
        """Get phase data from real epochs."""
        from fwd import filtered, hilbert_transform
        
        data = real_epochs_limited.get_data()[0, :, :]
        sfreq = real_epochs_limited.info['sfreq']
        
        filtered_signal = filtered(data, "Alpha", sfreq=sfreq)
        signals = [filtered_signal[i, :] for i in range(filtered_signal.shape[0])]
        _, phase = hilbert_transform(signals, trim=100)
        
        return phase
    
    def test_syncro_on_real_data(self, real_phase_data):
        """Synchronization should work on real phase data."""
        from fwd import calculate_syncro
        
        syncro = calculate_syncro(real_phase_data)
        
        n_channels = real_phase_data.shape[0]
        assert syncro.shape == (n_channels, n_channels)
    
    def test_syncro_symmetric_on_real_data(self, real_phase_data):
        """Real data syncro should be symmetric."""
        from fwd import calculate_syncro
        
        syncro = calculate_syncro(real_phase_data)
        
        assert np.allclose(syncro, syncro.T), "Syncro not symmetric on real data"
    
    def test_syncro_diagonal_one_on_real_data(self, real_phase_data):
        """Diagonal should be 1 for real data."""
        from fwd import calculate_syncro
        
        syncro = calculate_syncro(real_phase_data)
        
        assert np.allclose(np.diag(syncro), 1.0, atol=1e-5), \
            f"Diagonal not 1: {np.diag(syncro)[:5]}"
    
    def test_syncro_values_reasonable_for_eeg(self, real_phase_data):
        """Syncro values should be in expected range for EEG."""
        from fwd import calculate_syncro
        
        syncro = calculate_syncro(real_phase_data)
        
        # EEG typically has moderate synchronization (not all 0 or all 1)
        off_diag = syncro[~np.eye(syncro.shape[0], dtype=bool)]
        
        assert off_diag.min() >= 0, f"Negative sync: {off_diag.min()}"
        assert off_diag.max() <= 1, f"Sync > 1: {off_diag.max()}"
        
        # Mean should be moderate (0.2-0.8 is typical)
        mean_sync = off_diag.mean()
        assert 0.1 <= mean_sync <= 0.9, f"Unusual mean sync: {mean_sync}"


# =============================================================================
# Test: Real Data Kuramoto
# =============================================================================

class TestRealDataKuramoto:
    """Test Kuramoto order parameter on real EEG data."""
    
    @pytest.fixture
    def real_phase_data(self, real_epochs_limited):
        """Get phase data from real epochs."""
        from fwd import filtered, hilbert_transform
        
        data = real_epochs_limited.get_data()[0, :, :]
        sfreq = real_epochs_limited.info['sfreq']
        
        filtered_signal = filtered(data, "Alpha", sfreq=sfreq)
        signals = [filtered_signal[i, :] for i in range(filtered_signal.shape[0])]
        _, phase = hilbert_transform(signals, trim=100)
        
        return phase
    
    def test_kuramoto_on_real_data(self, real_phase_data):
        """Kuramoto should work on real data."""
        from fwd import order_parameter
        
        r = order_parameter(real_phase_data)
        
        assert r.shape == (real_phase_data.shape[1],)
    
    def test_kuramoto_in_valid_range(self, real_phase_data):
        """Kuramoto should be in [0, 1]."""
        from fwd import order_parameter
        
        r = order_parameter(real_phase_data)
        
        assert r.min() >= 0, f"r < 0: {r.min()}"
        assert r.max() <= 1, f"r > 1: {r.max()}"
    
    def test_kuramoto_reasonable_for_eeg(self, real_phase_data):
        """Kuramoto should have reasonable values for EEG."""
        from fwd import order_parameter
        
        r = order_parameter(real_phase_data)
        
        # EEG typically has moderate global synchronization
        mean_r = r.mean()
        assert 0.05 <= mean_r <= 0.95, f"Unusual mean Kuramoto: {mean_r}"
    
    def test_kuramoto_varies_over_time(self, real_phase_data):
        """Kuramoto should vary over time (not constant)."""
        from fwd import order_parameter
        
        r = order_parameter(real_phase_data)
        
        # Should have some variation
        std_r = r.std()
        assert std_r > 0.001, f"Kuramoto too constant: std = {std_r}"


# =============================================================================
# Test: Full Mini-Pipeline on Real Data
# =============================================================================

class TestFullMiniPipelineRealData:
    """Test complete mini-pipeline on real data."""
    
    def test_full_pipeline_single_epoch(self, real_epochs_limited):
        """Run full pipeline on single epoch and verify all outputs."""
        from fwd import filtered, hilbert_transform, calculate_syncro, order_parameter
        
        data = real_epochs_limited.get_data()[0, :, :]
        sfreq = real_epochs_limited.info['sfreq']
        n_channels = data.shape[0]
        
        # Process each band
        for band in ["Alpha", "Beta"]:
            # Filter
            filtered_signal = filtered(data, band, sfreq=sfreq)
            assert filtered_signal.shape == data.shape
            
            # Hilbert
            signals = [filtered_signal[i, :] for i in range(n_channels)]
            amplitude, phase = hilbert_transform(signals, trim=100)
            
            assert amplitude.shape[0] == n_channels
            assert phase.shape[0] == n_channels
            assert not np.any(np.isnan(amplitude))
            assert not np.any(np.isnan(phase))
            
            # Syncro
            syncro = calculate_syncro(phase)
            assert syncro.shape == (n_channels, n_channels)
            assert np.allclose(syncro, syncro.T)
            assert 0 <= syncro.min() and syncro.max() <= 1
            
            # Kuramoto
            r = order_parameter(phase)
            assert r.shape == (phase.shape[1],)
            assert 0 <= r.min() and r.max() <= 1
    
    def test_different_epochs_give_different_results(self, real_epochs_limited):
        """Different epochs should produce different results."""
        from fwd import filtered, hilbert_transform, order_parameter
        
        data = real_epochs_limited.get_data()
        sfreq = real_epochs_limited.info['sfreq']
        
        if data.shape[0] < 2:
            pytest.skip("Need at least 2 epochs")
        
        results = []
        for epoch_idx in range(min(2, data.shape[0])):
            epoch_data = data[epoch_idx, :, :]
            filtered_signal = filtered(epoch_data, "Alpha", sfreq=sfreq)
            signals = [filtered_signal[i, :] for i in range(epoch_data.shape[0])]
            _, phase = hilbert_transform(signals, trim=100)
            r = order_parameter(phase)
            results.append(r.mean())
        
        # Results should be different (unless it's repeated data)
        assert not np.allclose(results[0], results[1], atol=0.01), \
            f"Different epochs gave same Kuramoto: {results}"
    
    def test_reproducibility_on_real_data(self, real_epochs_limited):
        """Same input should give same output."""
        from fwd import filtered, hilbert_transform, calculate_syncro, order_parameter
        
        data = real_epochs_limited.get_data()[0, :, :]
        sfreq = real_epochs_limited.info['sfreq']
        n_channels = data.shape[0]
        
        def run_pipeline(data):
            filtered_signal = filtered(data, "Alpha", sfreq=sfreq)
            signals = [filtered_signal[i, :] for i in range(n_channels)]
            amplitude, phase = hilbert_transform(signals, trim=100)
            syncro = calculate_syncro(phase)
            r = order_parameter(phase)
            return filtered_signal, amplitude, phase, syncro, r
        
        # Run twice
        f1, a1, p1, s1, r1 = run_pipeline(data)
        f2, a2, p2, s2, r2 = run_pipeline(data)
        
        assert np.allclose(f1, f2), "Filtering not reproducible"
        assert np.allclose(p1, p2), "Phase not reproducible"
        assert np.allclose(s1, s2), "Syncro not reproducible"
        assert np.allclose(r1, r2), "Kuramoto not reproducible"









