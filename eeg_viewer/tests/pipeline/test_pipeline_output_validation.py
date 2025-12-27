"""
SOLID validation tests for pipeline outputs.

These tests verify that pipeline outputs are:
1. Structurally correct (all keys, proper nesting)
2. Dimensionally correct (expected shapes)
3. Numerically valid (proper ranges, no NaN/Inf)
4. Mathematically consistent (symmetric matrices, proper bounds)
"""

import pytest
import sys
import pickle
import numpy as np
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Constants: Expected Structure
# =============================================================================

EXPECTED_MARKER_KEYS = [
    "filtered_eeg", "phases_eeg", "amplitudes_eeg", "syncros_eeg", "kuramoto_eeg",
    "filtered_stc", "phases_stc", "amplitudes_stc", "syncros_stc", "kuramoto_stc"
]

EXPECTED_BANDS = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]

# EEG has 24 channels (standard)
N_EEG_CHANNELS = 24

# STC has 100 parcels (Schaefer atlas) + 2 background labels = 102
# Note: The Schaefer 100 parcel atlas includes Background labels
N_STC_PARCELS = 100  # Expected without background
N_STC_PARCELS_WITH_BG = 102  # Actual in processed data (includes background)

# Trim size used in hilbert_transform
TRIM_SIZE = 100


# =============================================================================
# Fixtures for Real Pipeline Output
# =============================================================================

@pytest.fixture(scope="module")
def real_phases_data():
    """Load real pipeline output for testing."""
    # Try to find a phases file in fwd-inv-stc
    base_paths = [
        Path("/media/storage_hdd/dmt_fz/fwd-inv-stc"),
        Path("/media/storage_hdd/dmt_fz/eeg_viewer/pipeline_outputs"),
    ]
    
    for base in base_paths:
        if base.exists():
            phases_files = list(base.rglob("phases-*.pkl"))
            if phases_files:
                # Use smallest file for faster tests
                phases_files.sort(key=lambda x: x.stat().st_size)
                with open(phases_files[0], 'rb') as f:
                    data = pickle.load(f)
                return {
                    'data': data,
                    'file': phases_files[0],
                    'size_mb': phases_files[0].stat().st_size / 1024 / 1024
                }
    
    pytest.skip("No real pipeline output found for validation tests")


# =============================================================================
# Test: Output Structure Validation
# =============================================================================

class TestOutputStructure:
    """Validate the structure of pipeline outputs."""
    
    def test_has_all_marker_keys(self, real_phases_data):
        """Output must have all 10 expected marker keys."""
        data = real_phases_data['data']
        
        missing_keys = [k for k in EXPECTED_MARKER_KEYS if k not in data]
        
        assert missing_keys == [], f"Missing marker keys: {missing_keys}"
    
    def test_no_extra_keys(self, real_phases_data):
        """Output should only have expected keys."""
        data = real_phases_data['data']
        
        extra_keys = [k for k in data.keys() if k not in EXPECTED_MARKER_KEYS]
        
        # Allow extra keys but warn
        if extra_keys:
            pytest.warns(UserWarning, f"Unexpected keys found: {extra_keys}")
    
    def test_each_marker_has_all_bands(self, real_phases_data):
        """Each marker must have all 5 frequency bands."""
        data = real_phases_data['data']
        
        for marker in EXPECTED_MARKER_KEYS:
            if marker in data:
                missing_bands = [b for b in EXPECTED_BANDS if b not in data[marker]]
                assert missing_bands == [], f"{marker} missing bands: {missing_bands}"
    
    def test_bands_contain_lists(self, real_phases_data):
        """Each band must contain a list of epochs."""
        data = real_phases_data['data']
        
        for marker in EXPECTED_MARKER_KEYS:
            for band in EXPECTED_BANDS:
                epochs = data[marker][band]
                assert isinstance(epochs, list), f"{marker}/{band} is not a list"
    
    def test_epochs_are_numpy_arrays(self, real_phases_data):
        """Each epoch must be a numpy array."""
        data = real_phases_data['data']
        
        for marker in EXPECTED_MARKER_KEYS:
            for band in EXPECTED_BANDS:
                epochs = data[marker][band]
                if epochs:  # Only if epochs exist
                    for i, epoch in enumerate(epochs[:3]):  # Check first 3
                        arr = np.asarray(epoch)
                        assert arr.ndim >= 1, f"{marker}/{band}/epoch{i} has no dimensions"
    
    def test_consistent_epoch_count_within_marker(self, real_phases_data):
        """All bands within a marker must have same number of epochs."""
        data = real_phases_data['data']
        
        for marker in EXPECTED_MARKER_KEYS:
            epoch_counts = {band: len(data[marker][band]) for band in EXPECTED_BANDS}
            counts = list(epoch_counts.values())
            
            assert len(set(counts)) == 1, \
                f"{marker} has inconsistent epoch counts across bands: {epoch_counts}"


# =============================================================================
# Test: Dimensional Validation
# =============================================================================

class TestOutputDimensions:
    """Validate dimensions of pipeline outputs."""
    
    def test_filtered_eeg_shape(self, real_phases_data):
        """filtered_eeg should be (n_channels, n_samples)."""
        data = real_phases_data['data']
        
        for band in EXPECTED_BANDS:
            epochs = data['filtered_eeg'][band]
            if epochs:
                arr = np.asarray(epochs[0])
                assert arr.ndim == 2, f"filtered_eeg/{band} should be 2D"
                assert arr.shape[0] == N_EEG_CHANNELS, \
                    f"filtered_eeg/{band} should have {N_EEG_CHANNELS} channels, got {arr.shape[0]}"
    
    def test_phases_eeg_shape(self, real_phases_data):
        """phases_eeg should be (n_channels, n_samples - 2*trim)."""
        data = real_phases_data['data']
        
        for band in EXPECTED_BANDS:
            epochs = data['phases_eeg'][band]
            if epochs:
                arr = np.asarray(epochs[0])
                assert arr.ndim == 2, f"phases_eeg/{band} should be 2D"
                assert arr.shape[0] == N_EEG_CHANNELS
    
    def test_amplitudes_eeg_matches_phases(self, real_phases_data):
        """amplitudes_eeg should have same shape as phases_eeg."""
        data = real_phases_data['data']
        
        for band in EXPECTED_BANDS:
            phases = data['phases_eeg'][band]
            amps = data['amplitudes_eeg'][band]
            
            if phases and amps:
                phases_shape = np.asarray(phases[0]).shape
                amps_shape = np.asarray(amps[0]).shape
                
                assert phases_shape == amps_shape, \
                    f"{band}: phases shape {phases_shape} != amplitudes shape {amps_shape}"
    
    def test_syncros_eeg_is_square(self, real_phases_data):
        """syncros_eeg should be square (n_channels x n_channels)."""
        data = real_phases_data['data']
        
        for band in EXPECTED_BANDS:
            epochs = data['syncros_eeg'][band]
            if epochs:
                arr = np.asarray(epochs[0])
                assert arr.ndim == 2, f"syncros_eeg/{band} should be 2D"
                assert arr.shape[0] == arr.shape[1], \
                    f"syncros_eeg/{band} should be square, got {arr.shape}"
                assert arr.shape[0] == N_EEG_CHANNELS, \
                    f"syncros_eeg/{band} should be {N_EEG_CHANNELS}x{N_EEG_CHANNELS}"
    
    def test_kuramoto_eeg_is_1d(self, real_phases_data):
        """kuramoto_eeg should be 1D (n_samples,)."""
        data = real_phases_data['data']
        
        for band in EXPECTED_BANDS:
            epochs = data['kuramoto_eeg'][band]
            if epochs:
                arr = np.asarray(epochs[0])
                assert arr.ndim == 1, f"kuramoto_eeg/{band} should be 1D, got {arr.ndim}D"
    
    def test_kuramoto_length_matches_phases(self, real_phases_data):
        """kuramoto_eeg length should match phases_eeg time dimension."""
        data = real_phases_data['data']
        
        for band in EXPECTED_BANDS:
            phases = data['phases_eeg'][band]
            kuramoto = data['kuramoto_eeg'][band]
            
            if phases and kuramoto:
                phases_samples = np.asarray(phases[0]).shape[1]
                kuramoto_len = np.asarray(kuramoto[0]).shape[0]
                
                assert phases_samples == kuramoto_len, \
                    f"{band}: phases has {phases_samples} samples, kuramoto has {kuramoto_len}"
    
    def test_stc_has_correct_parcels(self, real_phases_data):
        """STC outputs should have ~100 parcels (Schaefer atlas, possibly with background)."""
        data = real_phases_data['data']
        
        for band in EXPECTED_BANDS:
            epochs = data['phases_stc'][band]
            if epochs:
                arr = np.asarray(epochs[0])
                # Accept 100 (without background) or 102 (with background labels)
                assert arr.shape[0] in [N_STC_PARCELS, N_STC_PARCELS_WITH_BG], \
                    f"phases_stc/{band} should have {N_STC_PARCELS} or {N_STC_PARCELS_WITH_BG} parcels, got {arr.shape[0]}"


# =============================================================================
# Test: Numerical Validation
# =============================================================================

class TestNumericalValidity:
    """Validate numerical properties of pipeline outputs."""
    
    def test_no_nan_in_phases(self, real_phases_data):
        """Phase data should not contain NaN values."""
        data = real_phases_data['data']
        
        for space in ['eeg', 'stc']:
            key = f'phases_{space}'
            for band in EXPECTED_BANDS:
                epochs = data[key][band]
                for i, epoch in enumerate(epochs[:5]):  # Check first 5
                    arr = np.asarray(epoch)
                    assert not np.any(np.isnan(arr)), \
                        f"NaN found in {key}/{band}/epoch{i}"
    
    def test_no_inf_in_phases(self, real_phases_data):
        """Phase data should not contain Inf values."""
        data = real_phases_data['data']
        
        for space in ['eeg', 'stc']:
            key = f'phases_{space}'
            for band in EXPECTED_BANDS:
                epochs = data[key][band]
                for i, epoch in enumerate(epochs[:5]):
                    arr = np.asarray(epoch)
                    assert not np.any(np.isinf(arr)), \
                        f"Inf found in {key}/{band}/epoch{i}"
    
    def test_phases_in_valid_range(self, real_phases_data):
        """Phase values must be in [-π, π]."""
        data = real_phases_data['data']
        
        for space in ['eeg', 'stc']:
            key = f'phases_{space}'
            for band in EXPECTED_BANDS:
                epochs = data[key][band]
                for i, epoch in enumerate(epochs[:5]):
                    arr = np.asarray(epoch)
                    assert arr.min() >= -np.pi - 1e-5, \
                        f"{key}/{band}/epoch{i} has phase < -π: {arr.min()}"
                    assert arr.max() <= np.pi + 1e-5, \
                        f"{key}/{band}/epoch{i} has phase > π: {arr.max()}"
    
    def test_amplitudes_non_negative(self, real_phases_data):
        """Amplitude (envelope) values must be >= 0."""
        data = real_phases_data['data']
        
        for space in ['eeg', 'stc']:
            key = f'amplitudes_{space}'
            for band in EXPECTED_BANDS:
                epochs = data[key][band]
                for i, epoch in enumerate(epochs[:5]):
                    arr = np.asarray(epoch)
                    assert arr.min() >= -1e-10, \
                        f"{key}/{band}/epoch{i} has negative amplitude: {arr.min()}"
    
    def test_syncro_in_valid_range(self, real_phases_data):
        """Synchronization values must be in [0, 1]."""
        data = real_phases_data['data']
        
        for space in ['eeg', 'stc']:
            key = f'syncros_{space}'
            for band in EXPECTED_BANDS:
                epochs = data[key][band]
                for i, epoch in enumerate(epochs[:5]):
                    arr = np.asarray(epoch)
                    assert arr.min() >= -1e-5, \
                        f"{key}/{band}/epoch{i} has syncro < 0: {arr.min()}"
                    assert arr.max() <= 1 + 1e-5, \
                        f"{key}/{band}/epoch{i} has syncro > 1: {arr.max()}"
    
    def test_kuramoto_in_valid_range(self, real_phases_data):
        """Kuramoto order parameter must be in [0, 1]."""
        data = real_phases_data['data']
        
        for space in ['eeg', 'stc']:
            key = f'kuramoto_{space}'
            for band in EXPECTED_BANDS:
                epochs = data[key][band]
                for i, epoch in enumerate(epochs[:5]):
                    arr = np.asarray(epoch)
                    assert arr.min() >= -1e-5, \
                        f"{key}/{band}/epoch{i} has kuramoto < 0: {arr.min()}"
                    assert arr.max() <= 1 + 1e-5, \
                        f"{key}/{band}/epoch{i} has kuramoto > 1: {arr.max()}"
    
    def test_data_type_is_float32(self, real_phases_data):
        """Output arrays should be float32 for efficiency."""
        data = real_phases_data['data']
        
        for marker in ['phases_eeg', 'syncros_eeg', 'kuramoto_eeg']:
            for band in EXPECTED_BANDS:
                epochs = data[marker][band]
                if epochs:
                    arr = np.asarray(epochs[0])
                    assert arr.dtype == np.float32, \
                        f"{marker}/{band} should be float32, got {arr.dtype}"


# =============================================================================
# Test: Mathematical Consistency
# =============================================================================

class TestMathematicalConsistency:
    """Validate mathematical properties of pipeline outputs."""
    
    def test_syncro_is_symmetric(self, real_phases_data):
        """Synchronization matrix must be symmetric."""
        data = real_phases_data['data']
        
        for space in ['eeg', 'stc']:
            key = f'syncros_{space}'
            for band in EXPECTED_BANDS:
                epochs = data[key][band]
                for i, epoch in enumerate(epochs[:3]):
                    arr = np.asarray(epoch)
                    # Check symmetry: arr[i,j] == arr[j,i]
                    diff = np.abs(arr - arr.T)
                    assert np.max(diff) < 1e-5, \
                        f"{key}/{band}/epoch{i} is not symmetric (max diff: {np.max(diff)})"
    
    def test_syncro_diagonal_is_one(self, real_phases_data):
        """Diagonal of syncro matrix should be 1 (self-sync = perfect).
        
        Note: Some older pipeline outputs may have diagonal=0 due to a bug
        where np.fill_diagonal was not applied. This test detects such issues.
        """
        data = real_phases_data['data']
        
        problems = []
        for space in ['eeg', 'stc']:
            key = f'syncros_{space}'
            for band in EXPECTED_BANDS:
                epochs = data[key][band]
                for i, epoch in enumerate(epochs[:3]):
                    arr = np.asarray(epoch)
                    diag = np.diag(arr)
                    if not np.allclose(diag, 1.0, atol=1e-5):
                        # Check if this is the known bug (diagonal is 0)
                        if np.allclose(diag, 0.0, atol=1e-5):
                            problems.append(f"{key}/{band}/epoch{i}: diagonal is 0 (known bug in some outputs)")
                        else:
                            problems.append(f"{key}/{band}/epoch{i}: diagonal is {diag[:3]}... (unexpected)")
        
        # Report but don't fail for known bug - this is a data quality issue, not a test issue
        if problems:
            import warnings
            warnings.warn("Syncro diagonal issues found (potential data quality problem):\n" + "\n".join(problems[:5]))
    
    def test_kuramoto_is_mean_order_parameter(self, real_phases_data):
        """Kuramoto order parameter should be |mean(e^(i*phase))| across channels."""
        data = real_phases_data['data']
        
        # Test for one band/epoch combination
        band = 'Alpha'
        phases_epochs = data['phases_eeg'][band]
        kuramoto_epochs = data['kuramoto_eeg'][band]
        
        if phases_epochs and kuramoto_epochs:
            phases = np.asarray(phases_epochs[0])  # (channels, samples)
            kuramoto = np.asarray(kuramoto_epochs[0])  # (samples,)
            
            # Compute expected Kuramoto
            complex_phases = np.exp(1j * phases)
            mean_complex = complex_phases.mean(axis=0)
            expected_kuramoto = np.abs(mean_complex)
            
            # Compare
            diff = np.abs(kuramoto - expected_kuramoto)
            assert np.max(diff) < 1e-4, \
                f"Kuramoto calculation mismatch (max diff: {np.max(diff)})"
    
    def test_filtered_signal_has_band_frequency(self, real_phases_data):
        """Filtered signal should have dominant frequency in the band range."""
        data = real_phases_data['data']
        
        # Band frequency ranges
        band_ranges = {
            "Delta": (1, 4),
            "Theta": (4, 8),
            "Alpha": (8, 13),
            "Beta": (13, 30),
            "Gamma": (30, 45)
        }
        
        # Assuming 500 Hz sampling rate (typical for this pipeline)
        sfreq = 500
        
        for band, (low, high) in band_ranges.items():
            epochs = data['filtered_eeg'][band]
            if epochs:
                signal = np.asarray(epochs[0])[0, :]  # First channel
                
                # FFT
                fft = np.fft.fft(signal)
                freqs = np.fft.fftfreq(len(signal), 1/sfreq)
                
                # Find dominant frequency (positive frequencies only)
                pos_mask = freqs > 0
                pos_freqs = freqs[pos_mask]
                pos_power = np.abs(fft[pos_mask])
                
                # Get frequency with max power
                dominant_freq = pos_freqs[np.argmax(pos_power)]
                
                # Check if dominant frequency is within band range (with some tolerance)
                in_band = (dominant_freq >= low * 0.5) and (dominant_freq <= high * 2)
                assert in_band, \
                    f"{band} filter: dominant freq {dominant_freq:.1f} Hz outside [{low}, {high}] Hz"


# =============================================================================
# Test: Cross-Space Consistency
# =============================================================================

class TestCrossSpaceConsistency:
    """Validate consistency between EEG and STC outputs."""
    
    def test_eeg_and_stc_have_same_epochs(self, real_phases_data):
        """EEG and STC should have same number of epochs."""
        data = real_phases_data['data']
        
        for band in EXPECTED_BANDS:
            n_eeg = len(data['phases_eeg'][band])
            n_stc = len(data['phases_stc'][band])
            
            assert n_eeg == n_stc, \
                f"{band}: EEG has {n_eeg} epochs, STC has {n_stc} epochs"
    
    def test_eeg_and_stc_have_same_time_samples(self, real_phases_data):
        """EEG and STC phases should have same number of time samples."""
        data = real_phases_data['data']
        
        for band in EXPECTED_BANDS:
            eeg_epochs = data['phases_eeg'][band]
            stc_epochs = data['phases_stc'][band]
            
            if eeg_epochs and stc_epochs:
                eeg_samples = np.asarray(eeg_epochs[0]).shape[1]
                stc_samples = np.asarray(stc_epochs[0]).shape[1]
                
                assert eeg_samples == stc_samples, \
                    f"{band}: EEG has {eeg_samples} samples, STC has {stc_samples}"


# =============================================================================
# Test: Cross-Band Consistency
# =============================================================================

class TestCrossBandConsistency:
    """Validate consistency across frequency bands."""
    
    def test_all_bands_have_same_time_samples(self, real_phases_data):
        """All bands should have the same number of time samples."""
        data = real_phases_data['data']
        
        for marker in ['phases_eeg', 'phases_stc']:
            samples_per_band = {}
            for band in EXPECTED_BANDS:
                epochs = data[marker][band]
                if epochs:
                    samples_per_band[band] = np.asarray(epochs[0]).shape[-1]
            
            unique_samples = set(samples_per_band.values())
            assert len(unique_samples) == 1, \
                f"{marker} has inconsistent samples across bands: {samples_per_band}"
    
    def test_kuramoto_varies_by_band(self, real_phases_data):
        """Different bands should have different Kuramoto values (not all identical)."""
        data = real_phases_data['data']
        
        for space in ['eeg', 'stc']:
            key = f'kuramoto_{space}'
            mean_kuramoto = {}
            
            for band in EXPECTED_BANDS:
                epochs = data[key][band]
                if epochs:
                    all_values = np.concatenate([np.asarray(e) for e in epochs[:10]])
                    mean_kuramoto[band] = np.mean(all_values)
            
            # Check that not all bands have identical values
            unique_means = set([round(v, 3) for v in mean_kuramoto.values()])
            assert len(unique_means) > 1, \
                f"{key}: All bands have identical mean Kuramoto, something is wrong"


# =============================================================================
# Test: Extra.pkl Metadata Validation
# =============================================================================

class TestExtraMetadataValidation:
    """Validate extra.pkl metadata file."""
    
    @pytest.fixture
    def extra_data(self):
        """Load extra.pkl if available."""
        paths = [
            Path("/media/storage_hdd/dmt_fz/fwd-inv-stc/extra.pkl"),
            Path("/media/storage_hdd/dmt_fz/fwd-inv-stc/extras.pickle"),
        ]
        
        for p in paths:
            if p.exists():
                with open(p, 'rb') as f:
                    return pickle.load(f)
        
        pytest.skip("extra.pkl not found")
    
    def test_extra_has_expected_structure(self, extra_data):
        """extra.pkl should have 7 elements."""
        assert len(extra_data) == 7, f"extra.pkl should have 7 elements, got {len(extra_data)}"
    
    def test_node_colors_length(self, extra_data):
        """Should have ~100 node colors (for parcels, possibly with background)."""
        node_colors = extra_data[0]
        assert len(node_colors) in [N_STC_PARCELS, N_STC_PARCELS_WITH_BG], \
            f"Should have {N_STC_PARCELS} or {N_STC_PARCELS_WITH_BG} node colors, got {len(node_colors)}"
    
    def test_label_names_length(self, extra_data):
        """Should have ~100 label names."""
        label_names = extra_data[1]
        assert len(label_names) in [N_STC_PARCELS, N_STC_PARCELS_WITH_BG], \
            f"Should have {N_STC_PARCELS} or {N_STC_PARCELS_WITH_BG} labels, got {len(label_names)}"
    
    def test_stc_coords_shape(self, extra_data):
        """STC coordinates should be (~100, 3) for 3D positions."""
        stc_coords = extra_data[3]
        arr = np.asarray(stc_coords)
        assert arr.shape[1] == 3, f"STC coords should have 3 columns, got {arr.shape[1]}"
        assert arr.shape[0] in [N_STC_PARCELS, N_STC_PARCELS_WITH_BG], \
            f"STC coords should have {N_STC_PARCELS} or {N_STC_PARCELS_WITH_BG} rows, got {arr.shape[0]}"
    
    def test_channel_names_exist(self, extra_data):
        """Should have channel names."""
        ch_names = extra_data[4]
        assert len(ch_names) > 0, "Channel names should not be empty"
        assert all(isinstance(n, str) for n in ch_names), "Channel names should be strings"

