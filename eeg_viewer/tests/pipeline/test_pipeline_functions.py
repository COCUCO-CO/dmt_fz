"""
SOLID tests for pipeline processing functions.

These tests execute the actual pipeline functions and verify:
1. Functions produce correct output types
2. Mathematical operations are correct
3. Signal processing is accurate
4. Results are reproducible
"""

import pytest
import sys
import numpy as np
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "pipeline"))


# =============================================================================
# Import Pipeline Functions
# =============================================================================

# Import from pipeline/fwd.py
try:
    from fwd import (
        filtered,
        hilbert_transform,
        calculate_syncro,
        order_parameter,
        diff_ang,
        freq_bands,
        band_list,
    )
    PIPELINE_AVAILABLE = True
except ImportError as e:
    PIPELINE_AVAILABLE = False
    IMPORT_ERROR = str(e)


@pytest.fixture(autouse=True)
def check_pipeline_available():
    """Skip tests if pipeline functions not available."""
    if not PIPELINE_AVAILABLE:
        pytest.skip(f"Pipeline functions not importable: {IMPORT_ERROR}")


# =============================================================================
# Test: Bandpass Filtering
# =============================================================================

class TestBandpassFiltering:
    """Test the filtered() function for bandpass filtering."""
    
    @pytest.fixture
    def synthetic_signal(self):
        """Create synthetic signal with known frequency content."""
        sfreq = 500
        duration = 2  # seconds
        t = np.arange(0, duration, 1/sfreq)
        
        # Create signal with multiple frequency components
        # 2 Hz (Delta), 6 Hz (Theta), 10 Hz (Alpha), 20 Hz (Beta), 35 Hz (Gamma)
        signal = (
            np.sin(2 * np.pi * 2 * t) +     # Delta
            np.sin(2 * np.pi * 6 * t) +     # Theta
            np.sin(2 * np.pi * 10 * t) +    # Alpha
            np.sin(2 * np.pi * 20 * t) +    # Beta
            np.sin(2 * np.pi * 35 * t)      # Gamma
        )
        
        return signal.reshape(1, -1).astype(np.float64)  # Shape: (1, samples)
    
    def test_filter_returns_correct_shape(self, synthetic_signal):
        """Filtered signal should have same shape as input."""
        result = filtered(synthetic_signal, "Alpha", sfreq=500)
        assert result.shape == synthetic_signal.shape
    
    def test_alpha_filter_extracts_alpha(self, synthetic_signal):
        """Alpha filter should primarily contain 8-13 Hz."""
        result = filtered(synthetic_signal, "Alpha", sfreq=500)
        
        # FFT to check frequency content
        fft = np.fft.fft(result[0, :])
        freqs = np.fft.fftfreq(result.shape[1], 1/500)
        power = np.abs(fft)
        
        # Find peak frequency in positive frequencies
        pos_mask = freqs > 0
        peak_idx = np.argmax(power[pos_mask])
        peak_freq = freqs[pos_mask][peak_idx]
        
        # Peak should be around 10 Hz (our alpha component)
        assert 8 <= peak_freq <= 13, f"Alpha filter peak at {peak_freq} Hz, expected 8-13 Hz"
    
    def test_delta_filter_removes_high_frequencies(self, synthetic_signal):
        """Delta filter should remove frequencies above 4 Hz."""
        result = filtered(synthetic_signal, "Delta", sfreq=500)
        
        fft = np.fft.fft(result[0, :])
        freqs = np.fft.fftfreq(result.shape[1], 1/500)
        power = np.abs(fft)
        
        # Check power above 4 Hz is much lower than below
        low_freq_power = np.sum(power[(freqs > 0) & (freqs < 4)])
        high_freq_power = np.sum(power[(freqs > 10)])
        
        # Low freq power should dominate
        assert low_freq_power > high_freq_power * 10, \
            "Delta filter should suppress high frequencies"
    
    def test_all_bands_available(self):
        """All expected bands should be in freq_bands."""
        expected = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
        for band in expected:
            assert band in freq_bands, f"Missing band: {band}"
            assert len(freq_bands[band]) == 2, f"Band {band} should have [low, high]"
    
    def test_filter_handles_multichannel(self):
        """Filter should work with multiple channels."""
        n_channels = 24
        n_samples = 1000
        signal = np.random.randn(n_channels, n_samples).astype(np.float64)
        
        result = filtered(signal, "Alpha", sfreq=500)
        
        assert result.shape == (n_channels, n_samples)


# =============================================================================
# Test: Hilbert Transform
# =============================================================================

class TestHilbertTransform:
    """Test the hilbert_transform() function."""
    
    @pytest.fixture
    def sine_signal(self):
        """Create simple sine wave for testing Hilbert transform."""
        t = np.linspace(0, 2*np.pi*5, 1000)  # 5 cycles
        return [np.sin(t)]
    
    def test_returns_envelope_and_phase(self, sine_signal):
        """hilbert_transform should return envelope and phase matrices."""
        envelope, phase = hilbert_transform(sine_signal)
        
        assert envelope is not None
        assert phase is not None
    
    def test_output_shapes(self, sine_signal):
        """Output shapes should match input."""
        envelope, phase = hilbert_transform(sine_signal)
        
        input_shape = (len(sine_signal), len(sine_signal[0]))
        assert envelope.shape == input_shape
        assert phase.shape == input_shape
    
    def test_envelope_is_positive(self, sine_signal):
        """Envelope (amplitude) should be non-negative."""
        envelope, _ = hilbert_transform(sine_signal)
        
        assert np.all(envelope >= 0), "Envelope should be >= 0"
    
    def test_phase_in_valid_range(self, sine_signal):
        """Phase should be in [-π, π]."""
        _, phase = hilbert_transform(sine_signal)
        
        assert phase.min() >= -np.pi - 1e-5
        assert phase.max() <= np.pi + 1e-5
    
    def test_envelope_of_sine_is_approximately_constant(self, sine_signal):
        """Envelope of pure sine should be approximately constant."""
        envelope, _ = hilbert_transform(sine_signal)
        
        # Envelope should be close to 1 for sine with amplitude 1
        # (with some edge effects)
        middle = envelope[0, 100:-100]  # Avoid edges
        assert np.std(middle) < 0.1, "Envelope of sine should be nearly constant"
        assert np.abs(np.mean(middle) - 1.0) < 0.2, "Envelope should be ~1 for unit sine"
    
    def test_trim_option(self, sine_signal):
        """Trim option should remove samples from both ends."""
        trim = 100
        envelope, phase = hilbert_transform(sine_signal, trim=trim)
        
        original_samples = len(sine_signal[0])
        expected_samples = original_samples - 2 * trim
        
        assert envelope.shape[1] == expected_samples
        assert phase.shape[1] == expected_samples
    
    def test_multichannel_hilbert(self):
        """Should work with multiple channels."""
        n_channels = 10
        n_samples = 500
        signals = [np.sin(np.linspace(0, 10*np.pi, n_samples)) for _ in range(n_channels)]
        
        envelope, phase = hilbert_transform(signals)
        
        assert envelope.shape == (n_channels, n_samples)
        assert phase.shape == (n_channels, n_samples)


# =============================================================================
# Test: Synchronization Calculation
# =============================================================================

class TestSynchronizationCalculation:
    """Test the calculate_syncro() function."""
    
    @pytest.fixture
    def identical_phases(self):
        """Create identical phase signals (perfect sync)."""
        n_channels = 5
        n_samples = 100
        phase = np.zeros((n_channels, n_samples), dtype=np.float32)
        return phase
    
    @pytest.fixture
    def random_phases(self):
        """Create random phase signals (low sync)."""
        np.random.seed(42)
        n_channels = 5
        n_samples = 100
        phase = np.random.uniform(-np.pi, np.pi, (n_channels, n_samples)).astype(np.float32)
        return phase
    
    def test_output_shape_is_square(self, random_phases):
        """Syncro matrix should be (n_channels, n_channels)."""
        result = calculate_syncro(random_phases)
        
        n_channels = random_phases.shape[0]
        assert result.shape == (n_channels, n_channels)
    
    def test_output_is_symmetric(self, random_phases):
        """Syncro matrix should be symmetric."""
        result = calculate_syncro(random_phases)
        
        assert np.allclose(result, result.T), "Syncro matrix not symmetric"
    
    def test_diagonal_is_one(self, random_phases):
        """Diagonal should be 1 (self-synchronization)."""
        result = calculate_syncro(random_phases)
        
        assert np.allclose(np.diag(result), 1.0), "Diagonal should be 1"
    
    def test_values_in_valid_range(self, random_phases):
        """All values should be in [0, 1]."""
        result = calculate_syncro(random_phases)
        
        assert result.min() >= 0, f"Min syncro value {result.min()} < 0"
        assert result.max() <= 1, f"Max syncro value {result.max()} > 1"
    
    def test_identical_phases_give_high_sync(self, identical_phases):
        """Identical phases should give sync close to 1."""
        result = calculate_syncro(identical_phases)
        
        # Off-diagonal elements should be close to 1
        off_diag = result[~np.eye(result.shape[0], dtype=bool)]
        assert off_diag.mean() > 0.9, f"Identical phases should have high sync: {off_diag.mean()}"
    
    def test_opposite_phases_give_low_sync(self):
        """Opposite phases should give low sync."""
        n_samples = 100
        # Create two channels with opposite phases
        phase = np.zeros((2, n_samples), dtype=np.float32)
        phase[0, :] = 0
        phase[1, :] = np.pi
        
        result = calculate_syncro(phase)
        
        # Off-diagonal should be low
        assert result[0, 1] < 0.5, f"Opposite phases should have low sync: {result[0, 1]}"


# =============================================================================
# Test: Kuramoto Order Parameter
# =============================================================================

class TestKuramotoOrderParameter:
    """Test the order_parameter() function."""
    
    def test_output_shape(self):
        """Order parameter should have shape (n_samples,)."""
        n_channels = 10
        n_samples = 100
        phase = np.random.uniform(-np.pi, np.pi, (n_channels, n_samples))
        
        result = order_parameter(phase)
        
        assert result.shape == (n_samples,)
    
    def test_values_in_valid_range(self):
        """Order parameter should be in [0, 1]."""
        n_channels = 10
        n_samples = 100
        phase = np.random.uniform(-np.pi, np.pi, (n_channels, n_samples))
        
        result = order_parameter(phase)
        
        assert result.min() >= 0, f"Min r value {result.min()} < 0"
        assert result.max() <= 1, f"Max r value {result.max()} > 1"
    
    def test_identical_phases_give_r_one(self):
        """Identical phases should give r = 1."""
        n_channels = 10
        n_samples = 100
        phase = np.zeros((n_channels, n_samples))  # All phases = 0
        
        result = order_parameter(phase)
        
        assert np.allclose(result, 1.0), f"Identical phases should give r=1: {result.mean()}"
    
    def test_uniformly_distributed_phases_give_low_r(self):
        """Uniformly distributed phases should give r close to 0."""
        n_channels = 100  # Need many channels for good approximation
        n_samples = 10
        
        # Create uniformly distributed phases
        phase = np.linspace(-np.pi, np.pi, n_channels).reshape(-1, 1)
        phase = np.tile(phase, (1, n_samples))
        
        result = order_parameter(phase)
        
        assert result.mean() < 0.2, f"Uniform phases should give low r: {result.mean()}"
    
    def test_kuramoto_formula_is_correct(self):
        """Verify the mathematical formula: r = |mean(e^(i*phase))|."""
        n_channels = 5
        n_samples = 50
        np.random.seed(42)
        phase = np.random.uniform(-np.pi, np.pi, (n_channels, n_samples))
        
        # Manual calculation
        complex_phases = np.exp(1j * phase)
        expected = np.abs(complex_phases.mean(axis=0))
        
        result = order_parameter(phase)
        
        assert np.allclose(result, expected, atol=1e-5), \
            f"Kuramoto formula mismatch: max diff = {np.max(np.abs(result - expected))}"


# =============================================================================
# Test: Angular Difference
# =============================================================================

class TestAngularDifference:
    """Test the diff_ang() function."""
    
    def test_zero_difference(self):
        """Same angles should have zero difference."""
        theta1 = np.array([0, np.pi/2, np.pi])
        theta2 = np.array([0, np.pi/2, np.pi])
        
        result = diff_ang(theta1, theta2)
        
        assert np.allclose(result, 0), "Same angles should have 0 difference"
    
    def test_opposite_angles(self):
        """Opposite angles should have π difference."""
        theta1 = np.array([0, 0, 0])
        theta2 = np.array([np.pi, -np.pi, np.pi])
        
        result = diff_ang(theta1, theta2)
        
        assert np.allclose(result, np.pi), "Opposite angles should have π difference"
    
    def test_wrapping_around_pi(self):
        """Should handle wrapping correctly around ±π."""
        theta1 = np.array([np.pi - 0.1])
        theta2 = np.array([-np.pi + 0.1])
        
        result = diff_ang(theta1, theta2)
        
        # These are actually close (0.2 apart)
        assert result[0] < 0.3, f"Should recognize close angles: {result[0]}"
    
    def test_absolute_option(self):
        """abso=True should return absolute difference."""
        theta1 = np.array([0])
        theta2 = np.array([1.0])
        
        result_abs = diff_ang(theta1, theta2, abso=True)
        result_signed = diff_ang(theta1, theta2, abso=False)
        
        assert result_abs[0] >= 0
        assert result_abs[0] == abs(result_signed[0])


# =============================================================================
# Test: Full Processing Pipeline (Mini)
# =============================================================================

class TestMiniPipeline:
    """Test a mini version of the full pipeline."""
    
    @pytest.fixture
    def synthetic_eeg(self):
        """Create synthetic EEG-like data."""
        np.random.seed(42)
        sfreq = 500
        duration = 2  # seconds
        n_channels = 24
        n_samples = sfreq * duration
        
        t = np.linspace(0, duration, n_samples)
        
        # Create signals with different dominant frequencies per channel
        data = np.zeros((n_channels, n_samples))
        for i in range(n_channels):
            # Mix of frequencies with some noise
            data[i, :] = (
                np.sin(2 * np.pi * 10 * t + np.random.rand() * 2 * np.pi) +  # 10 Hz
                0.5 * np.sin(2 * np.pi * 6 * t + np.random.rand() * 2 * np.pi) +  # 6 Hz
                0.2 * np.random.randn(n_samples)  # noise
            )
        
        return data.astype(np.float64)
    
    def test_full_mini_pipeline(self, synthetic_eeg):
        """Run mini pipeline and verify all outputs."""
        band = "Alpha"
        
        # Step 1: Filter
        filtered_signal = filtered(synthetic_eeg, band, sfreq=500)
        assert filtered_signal.shape == synthetic_eeg.shape
        
        # Step 2: Hilbert transform
        amplitude, phase = hilbert_transform([filtered_signal[i] for i in range(filtered_signal.shape[0])], trim=100)
        assert amplitude.shape[0] == 24  # channels
        assert phase.shape[0] == 24
        assert amplitude.shape[1] == filtered_signal.shape[1] - 200  # trimmed
        
        # Step 3: Synchronization
        syncro = calculate_syncro(phase)
        assert syncro.shape == (24, 24)
        assert np.allclose(syncro, syncro.T)  # symmetric
        assert np.allclose(np.diag(syncro), 1.0)  # diagonal is 1
        assert syncro.min() >= 0 and syncro.max() <= 1  # valid range
        
        # Step 4: Kuramoto order parameter
        r = order_parameter(phase)
        assert r.shape == (phase.shape[1],)
        assert r.min() >= 0 and r.max() <= 1
    
    def test_pipeline_reproducibility(self, synthetic_eeg):
        """Same input should produce same output."""
        band = "Alpha"
        
        # Run twice
        filtered1 = filtered(synthetic_eeg, band, sfreq=500)
        amp1, phase1 = hilbert_transform([filtered1[i] for i in range(24)], trim=100)
        syncro1 = calculate_syncro(phase1)
        r1 = order_parameter(phase1)
        
        filtered2 = filtered(synthetic_eeg, band, sfreq=500)
        amp2, phase2 = hilbert_transform([filtered2[i] for i in range(24)], trim=100)
        syncro2 = calculate_syncro(phase2)
        r2 = order_parameter(phase2)
        
        assert np.allclose(filtered1, filtered2)
        assert np.allclose(phase1, phase2)
        assert np.allclose(syncro1, syncro2)
        assert np.allclose(r1, r2)
    
    def test_different_bands_give_different_results(self, synthetic_eeg):
        """Different frequency bands should give different results."""
        results = {}
        
        for band in ["Delta", "Alpha", "Gamma"]:
            filtered_signal = filtered(synthetic_eeg, band, sfreq=500)
            _, phase = hilbert_transform([filtered_signal[i] for i in range(24)], trim=100)
            r = order_parameter(phase)
            results[band] = r.mean()
        
        # Results should be different
        values = list(results.values())
        assert not np.allclose(values[0], values[1]) or not np.allclose(values[1], values[2]), \
            f"Different bands gave same results: {results}"








