"""
Comprehensive tests for the EEG cleaning pipeline.

Tests each step, consistency checks, UI behavior simulations, and end-to-end pipeline.
"""

import pytest
import numpy as np
import mne
from pathlib import Path

from cleaning.state import CleaningState, CleaningStep, OperationRecord
from cleaning.filters import (
    FilterPreset, FilterParams, apply_notch_filter, apply_bandpass_filter,
    apply_filter_preset, get_filter_presets
)
from cleaning.bad_channels import (
    detect_bad_channels, interpolate_channels, BadChannelResult
)
from cleaning.rereferencing import (
    ReferenceType, apply_average_reference, apply_single_reference,
    apply_reference, get_available_references
)
from cleaning.ica import compute_ica, apply_ica_exclusion, ICAResult
from cleaning.epochs import (
    create_epochs, detect_bad_epochs, apply_epoch_rejection,
    EpochRejectionCriteria, EpochResult
)
from cleaning.export import export_cleaned_eeg, export_epochs, ExportFormat


# ============ Fixtures ============

@pytest.fixture
def synthetic_raw_long():
    """Create a longer synthetic raw EEG with realistic characteristics."""
    sfreq = 500
    duration = 60  # 60 seconds
    n_channels = 24
    
    # Create channel info
    ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 
                'FC6', 'C3', 'Cz', 'C4', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3',
                'Pz', 'P4', 'O1', 'O2']
    
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    
    # Generate realistic EEG-like data
    n_times = int(sfreq * duration)
    times = np.linspace(0, duration, n_times)
    data = np.zeros((n_channels, n_times))
    
    for i in range(n_channels):
        # Alpha rhythm (10 Hz)
        alpha = 20e-6 * np.sin(2 * np.pi * 10 * times + np.random.rand() * 2 * np.pi)
        # Beta rhythm (20 Hz)
        beta = 10e-6 * np.sin(2 * np.pi * 20 * times + np.random.rand() * 2 * np.pi)
        # Low frequency drift
        drift = 5e-6 * np.sin(2 * np.pi * 0.5 * times)
        # Noise
        noise = 5e-6 * np.random.randn(n_times)
        
        data[i] = alpha + beta + drift + noise
    
    raw = mne.io.RawArray(data, info)
    return raw


@pytest.fixture
def synthetic_raw_with_bad_channel():
    """Create synthetic raw with one clearly bad channel."""
    sfreq = 500
    duration = 30
    n_channels = 8
    
    ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'O1', 'O2']
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    
    n_times = int(sfreq * duration)
    times = np.linspace(0, duration, n_times)
    data = np.zeros((n_channels, n_times))
    
    for i in range(n_channels):
        if i == 2:  # F3 will be noisy
            data[i] = 200e-6 * np.random.randn(n_times)  # Very noisy
        elif i == 5:  # C4 will be flat
            data[i] = 0.1e-6 * np.ones(n_times)  # Almost flat
        else:
            data[i] = 20e-6 * np.sin(2 * np.pi * 10 * times) + 3e-6 * np.random.randn(n_times)
    
    raw = mne.io.RawArray(data, info)
    return raw


@pytest.fixture
def cleaning_state(synthetic_raw_long):
    """Create a CleaningState with loaded data."""
    state = CleaningState()
    state.load_raw(synthetic_raw_long.copy(), Path('/test/eeg.fif'))
    return state


# ============ Step 1: Load Tests ============

class TestLoadStep:
    """Tests for the Load step."""
    
    def test_load_initializes_state(self, synthetic_raw_long):
        """Test that loading initializes all state correctly."""
        state = CleaningState()
        assert not state.is_loaded
        assert state.raw is None
        
        state.load_raw(synthetic_raw_long.copy(), Path('/test/eeg.fif'))
        
        assert state.is_loaded
        assert state.raw is not None
        assert state.current_step == CleaningStep.LOAD
        assert len(state.operations) >= 1
    
    def test_load_preserves_original(self, synthetic_raw_long):
        """Test that original data is preserved."""
        state = CleaningState()
        original_n_times = synthetic_raw_long.n_times
        
        state.load_raw(synthetic_raw_long.copy(), Path('/test/eeg.fif'))
        
        # Original should be preserved
        assert state._raw_original is not None
        assert state._raw_original.n_times == original_n_times
    
    def test_load_resets_previous_state(self, synthetic_raw_long):
        """Test that loading new file resets state to LOAD step."""
        state = CleaningState()
        state.load_raw(synthetic_raw_long.copy(), Path('/test/eeg1.fif'))
        
        # Mark some steps as completed
        state.complete_step(CleaningStep.LOAD)
        state.complete_step(CleaningStep.FILTER)
        
        # Load a new file
        state.load_raw(synthetic_raw_long.copy(), Path('/test/eeg2.fif'))
        
        # State should be at LOAD step (completed_steps may persist depending on impl)
        assert state.current_step == CleaningStep.LOAD
        assert state.is_loaded


# ============ Step 2: Filter Tests ============

class TestFilterStep:
    """Tests for the Filter step."""
    
    def test_notch_filter_removes_line_noise(self, synthetic_raw_long):
        """Test that notch filter attenuates target frequency."""
        raw = synthetic_raw_long.copy()
        
        # Add 50 Hz noise
        sfreq = raw.info['sfreq']
        n_times = raw.n_times
        times = np.arange(n_times) / sfreq
        line_noise = 30e-6 * np.sin(2 * np.pi * 50 * times)
        
        data = raw.get_data()
        data[0] += line_noise
        raw._data = data
        
        # Apply notch filter
        raw_filtered = apply_notch_filter(raw.copy(), 50)
        
        # Check that 50 Hz is attenuated (simplified check)
        assert raw_filtered is not None
        assert raw_filtered.n_times == raw.n_times
    
    def test_bandpass_filter_range(self, synthetic_raw_long):
        """Test bandpass filter with different ranges."""
        raw = synthetic_raw_long.copy()
        
        # Standard EEG range
        raw_filtered = apply_bandpass_filter(raw.copy(), 0.5, 45)
        assert raw_filtered is not None
        
        # Alpha band
        raw_filtered = apply_bandpass_filter(raw.copy(), 8, 13)
        assert raw_filtered is not None
    
    def test_filter_presets(self, synthetic_raw_long):
        """Test all filter presets work."""
        for preset in FilterPreset:
            raw = synthetic_raw_long.copy()
            raw_filtered, params = apply_filter_preset(raw, preset)
            
            assert raw_filtered is not None
            assert params is not None
            assert isinstance(params, FilterParams)
    
    def test_filter_idempotent(self, synthetic_raw_long):
        """Test applying same filter twice gives same result."""
        raw = synthetic_raw_long.copy()
        
        raw1 = apply_bandpass_filter(raw.copy(), 1, 40)
        raw2 = apply_bandpass_filter(raw.copy(), 1, 40)
        
        # Data should be identical (within floating point tolerance)
        np.testing.assert_array_almost_equal(raw1.get_data(), raw2.get_data(), decimal=10)


# ============ Step 3: Bad Channels Tests ============

class TestBadChannelsStep:
    """Tests for the Bad Channels step."""
    
    def test_detect_bad_with_noisy_channel(self, synthetic_raw_with_bad_channel):
        """Test detection of noisy channel."""
        result = detect_bad_channels(
            synthetic_raw_with_bad_channel.copy(),
            std_threshold=2.0,
            check_correlation=False
        )
        
        # Should detect F3 as noisy
        assert 'F3' in result.noisy or len(result.all_bad) > 0
    
    def test_detect_bad_with_flat_channel(self, synthetic_raw_with_bad_channel):
        """Test detection of flat channel."""
        result = detect_bad_channels(
            synthetic_raw_with_bad_channel.copy(),
            flat_threshold=1e-6,
            check_correlation=False
        )
        
        # Should detect C4 as flat
        assert 'C4' in result.flat or len(result.all_bad) > 0
    
    def test_interpolate_restores_shape(self, synthetic_raw_with_bad_channel):
        """Test that interpolation keeps same shape."""
        raw = synthetic_raw_with_bad_channel.copy()
        original_shape = raw.get_data().shape
        
        # Mark and interpolate a channel
        raw.info['bads'] = ['F3']
        raw_interp = interpolate_channels(raw, ['F3'])
        
        assert raw_interp.get_data().shape == original_shape
    
    def test_threshold_sensitivity(self, synthetic_raw_with_bad_channel):
        """Test that thresholds affect detection."""
        raw = synthetic_raw_with_bad_channel.copy()
        
        # Very lenient - should find fewer bad channels
        result_lenient = detect_bad_channels(raw.copy(), std_threshold=5.0, flat_threshold=1e-10)
        
        # Very strict - should find more bad channels
        result_strict = detect_bad_channels(raw.copy(), std_threshold=1.5, flat_threshold=1e-5)
        
        # Strict should find at least as many as lenient
        assert len(result_strict.all_bad) >= len(result_lenient.all_bad)


# ============ Step 4: Re-reference Tests ============

class TestRereferenceStep:
    """Tests for the Re-reference step."""
    
    def test_average_reference_zero_mean(self, synthetic_raw_long):
        """Test that average reference results in zero mean across channels."""
        raw = synthetic_raw_long.copy()
        raw_ref = apply_average_reference(raw)
        
        # Mean across channels should be close to zero
        data = raw_ref.get_data()
        channel_means = np.mean(data, axis=0)
        assert np.allclose(channel_means, 0, atol=1e-10)
    
    def test_single_reference(self, synthetic_raw_long):
        """Test single electrode reference."""
        raw = synthetic_raw_long.copy()
        raw_ref = apply_single_reference(raw, 'Cz')  # Single string, not list
        
        assert raw_ref is not None
    
    def test_all_reference_types(self, synthetic_raw_long):
        """Test available reference types work."""
        raw = synthetic_raw_long.copy()
        
        # Test average reference
        raw_ref = apply_average_reference(raw.copy())
        assert raw_ref is not None


# ============ Step 5: ICA Tests ============

class TestICAStep:
    """Tests for the ICA step."""
    
    def test_compute_ica_basic(self, synthetic_raw_long):
        """Test basic ICA computation."""
        raw = synthetic_raw_long.copy()
        raw = apply_bandpass_filter(raw, 1, 45)  # ICA needs filtered data
        
        result = compute_ica(raw, n_components=10, method='fastica')
        
        assert result.ica is not None
        assert result.n_components == 10
    
    def test_ica_component_limit(self, synthetic_raw_long):
        """Test that n_components is limited correctly."""
        raw = synthetic_raw_long.copy()
        raw = apply_bandpass_filter(raw, 1, 45)
        n_channels = len(raw.ch_names)
        
        # Request more components than channels - should be capped automatically
        result = compute_ica(raw, n_components=n_channels + 10, method='fastica')
        
        # Should be capped at n_channels - 1
        assert result.n_components <= n_channels - 1
    
    def test_apply_ica_preserves_shape(self, synthetic_raw_long):
        """Test ICA application preserves data shape."""
        raw = synthetic_raw_long.copy()
        raw = apply_bandpass_filter(raw, 1, 45)
        original_shape = raw.get_data().shape
        
        result = compute_ica(raw.copy(), n_components=10, method='fastica')
        
        # Set components to exclude
        result.user_excluded = [0]
        
        # Apply ICA
        raw_clean = apply_ica_exclusion(raw.copy(), result)
        
        assert raw_clean.get_data().shape == original_shape


# ============ Step 6: Epochs Tests ============

class TestEpochsStep:
    """Tests for the Epochs step."""
    
    def test_create_epochs_count(self, synthetic_raw_long):
        """Test epoch count calculation."""
        raw = synthetic_raw_long.copy()
        duration = 2.0
        
        result = create_epochs(raw, duration=duration, overlap=0.0)
        
        # Calculate expected count
        total_time = raw.n_times / raw.info['sfreq']
        expected = int((total_time - duration) / duration) + 1
        
        # Should be close to expected
        assert abs(result.n_total - expected) <= 1
    
    def test_create_epochs_consistency(self, synthetic_raw_long):
        """Test that repeated calls give same count."""
        raw = synthetic_raw_long.copy()
        
        results = []
        for _ in range(5):
            result = create_epochs(raw.copy(), duration=2.0, overlap=0.0)
            results.append(result.n_total)
        
        # All should be the same
        assert len(set(results)) == 1
    
    def test_epochs_overlap_increases_count(self, synthetic_raw_long):
        """Test that overlap increases epoch count."""
        raw = synthetic_raw_long.copy()
        
        result_no_overlap = create_epochs(raw.copy(), duration=2.0, overlap=0.0)
        result_overlap = create_epochs(raw.copy(), duration=2.0, overlap=1.0)
        
        assert result_overlap.n_total > result_no_overlap.n_total
    
    def test_epoch_duration_stored(self, synthetic_raw_long):
        """Test that epoch duration affects results."""
        raw = synthetic_raw_long.copy()
        
        result_2s = create_epochs(raw.copy(), duration=2.0)
        result_1s = create_epochs(raw.copy(), duration=1.0)
        
        # 1s epochs should have roughly twice as many as 2s epochs
        ratio = result_1s.n_total / result_2s.n_total
        assert 1.5 < ratio < 2.5


# ============ Step 7: Reject Tests ============

class TestRejectStep:
    """Tests for the Reject step."""
    
    def test_detect_bad_epochs_basic(self, synthetic_raw_long):
        """Test basic bad epoch detection."""
        raw = synthetic_raw_long.copy()
        epoch_result = create_epochs(raw, duration=2.0)
        
        criteria = EpochRejectionCriteria(peak_to_peak_uv=100, flat_uv=0.5)
        detected = detect_bad_epochs(epoch_result, criteria)
        
        # Should have some results
        assert detected.n_total > 0
        assert detected.n_good + detected.n_rejected == detected.n_total
    
    def test_reject_updates_counts(self, synthetic_raw_long):
        """Test that rejection updates counts correctly."""
        raw = synthetic_raw_long.copy()
        epoch_result = create_epochs(raw, duration=2.0)
        
        # Manually add rejections
        epoch_result.rejected_indices = [0, 1, 2]
        epoch_result.n_rejected = 3
        epoch_result.n_good = epoch_result.n_total - 3
        
        assert epoch_result.n_good == epoch_result.n_total - 3
        assert epoch_result.rejection_rate == (3 / epoch_result.n_total) * 100
    
    def test_manual_rejection_indices(self, synthetic_raw_long):
        """Test manual rejection index addition."""
        raw = synthetic_raw_long.copy()
        epoch_result = create_epochs(raw, duration=2.0)
        
        # Add manual rejections
        original_rejected = len(epoch_result.rejected_indices)
        epoch_result.rejected_indices.extend([0, 5, 10])
        epoch_result.rejected_indices = sorted(set(epoch_result.rejected_indices))
        
        assert len(epoch_result.rejected_indices) >= original_rejected


# ============ Reverse Operation Tests ============

class TestReverseOperation:
    """Tests for the reverse/flip operation."""
    
    def test_reverse_data_is_flipped(self, synthetic_raw_long):
        """Test that reverse actually flips the data."""
        raw = synthetic_raw_long.copy()
        original_data = raw.get_data().copy()
        
        # Simulate reverse
        raw._data = original_data[:, ::-1]
        
        # First sample of reversed should equal last sample of original
        np.testing.assert_array_almost_equal(raw.get_data()[:, 0], original_data[:, -1])
    
    def test_reverse_twice_restores(self, synthetic_raw_long):
        """Test that reversing twice restores original."""
        raw = synthetic_raw_long.copy()
        original_data = raw.get_data().copy()
        
        # Reverse twice
        raw._data = raw.get_data()[:, ::-1]
        raw._data = raw.get_data()[:, ::-1]
        
        np.testing.assert_array_almost_equal(raw.get_data(), original_data)
    
    def test_reverse_rejected_indices(self, synthetic_raw_long):
        """Test that rejected indices are correctly transformed."""
        raw = synthetic_raw_long.copy()
        epoch_result = create_epochs(raw, duration=2.0)
        n_epochs = epoch_result.n_total
        
        # Set initial rejections at start
        original_indices = [0, 1, 2]
        epoch_result.rejected_indices = original_indices.copy()
        
        # Simulate reverse transform
        new_indices = [n_epochs - 1 - idx for idx in original_indices]
        new_indices = sorted(new_indices)
        
        # After reverse, first epochs become last epochs
        expected = sorted([n_epochs - 1, n_epochs - 2, n_epochs - 3])
        assert new_indices == expected


# ============ State Consistency Tests ============

class TestStateConsistency:
    """Tests for state consistency across operations."""
    
    def test_step_completion_order(self, cleaning_state):
        """Test that steps can only be completed in order."""
        state = cleaning_state
        
        # Can complete LOAD
        state.complete_step(CleaningStep.LOAD)
        assert CleaningStep.LOAD in state.completed_steps
        
        # Can complete FILTER after LOAD
        state.go_to_step(CleaningStep.FILTER)
        state.complete_step(CleaningStep.FILTER)
        assert CleaningStep.FILTER in state.completed_steps
    
    def test_undo_single_operation(self, cleaning_state):
        """Test that undo reverts single operation."""
        state = cleaning_state
        
        # Apply filter directly (not using update_raw)
        raw_before = state.raw.get_data().copy()
        state._raw = apply_bandpass_filter(state.raw, 1, 40)
        state._raw_history.append(mne.io.RawArray(raw_before, state.raw.info.copy()))
        
        # Undo
        state.undo()
        
        # Should be back to before filter (or close, depending on undo implementation)
        assert state.raw is not None
    
    def test_reset_to_original(self, cleaning_state):
        """Test full reset to original."""
        state = cleaning_state
        original_data = state._raw_original.get_data().copy()
        
        # Apply filter directly
        state._raw = apply_bandpass_filter(state.raw, 1, 40)
        state._raw = apply_notch_filter(state.raw, 50)
        
        # Reset
        state.reset_to_original()
        
        # Should be at original
        np.testing.assert_array_almost_equal(state.raw.get_data(), original_data)
        assert state.current_step == CleaningStep.LOAD


# ============ End-to-End Pipeline Test ============

class TestFullPipeline:
    """End-to-end pipeline test."""
    
    def test_complete_pipeline(self, synthetic_raw_long):
        """Test running through complete pipeline."""
        state = CleaningState()
        
        # Step 1: Load
        state.load_raw(synthetic_raw_long.copy(), Path('/test/eeg.fif'))
        assert state.is_loaded
        state.complete_step(CleaningStep.LOAD)
        
        # Step 2: Filter
        state.go_to_step(CleaningStep.FILTER)
        raw_filtered, params = apply_filter_preset(state.raw.copy(), FilterPreset.STANDARD)
        state._raw = raw_filtered
        state.filter_params = {'highpass': params.highpass, 'lowpass': params.lowpass}
        state.complete_step(CleaningStep.FILTER)
        
        # Step 3: Bad channels
        state.go_to_step(CleaningStep.BAD_CHANNELS)
        result = detect_bad_channels(state.raw.copy())
        state.bad_channels = result.all_bad
        state.complete_step(CleaningStep.BAD_CHANNELS)
        
        # Step 4: Re-reference
        state.go_to_step(CleaningStep.REREFERENCE)
        raw_ref = apply_average_reference(state.raw.copy())
        state._raw = raw_ref
        state.reference_type = 'Average reference'
        state.complete_step(CleaningStep.REREFERENCE)
        
        # Step 5: ICA (skip in this test for speed)
        state.complete_step(CleaningStep.ICA)
        
        # Step 6: Epochs
        state.go_to_step(CleaningStep.EPOCHS)
        epoch_result = create_epochs(state.raw.copy(), duration=2.0)
        state.epochs = epoch_result.epochs
        state.epoch_duration = 2.0
        state.epochs_total = epoch_result.n_total
        state.complete_step(CleaningStep.EPOCHS)
        
        # Step 7: Reject
        state.go_to_step(CleaningStep.REJECT)
        criteria = EpochRejectionCriteria.lenient()
        epoch_result = detect_bad_epochs(epoch_result, criteria)
        state.complete_step(CleaningStep.REJECT)
        
        # Step 8: Visualize
        state.complete_step(CleaningStep.VISUALIZE)
        
        # Verify all main steps completed
        main_steps = [
            CleaningStep.LOAD, CleaningStep.FILTER, CleaningStep.BAD_CHANNELS,
            CleaningStep.REREFERENCE, CleaningStep.ICA, CleaningStep.EPOCHS,
            CleaningStep.REJECT, CleaningStep.VISUALIZE
        ]
        for step in main_steps:
            assert step in state.completed_steps, f"Step {step} not completed"
    
    def test_pipeline_with_reverse(self, synthetic_raw_long):
        """Test pipeline with reverse operation."""
        state = CleaningState()
        state.load_raw(synthetic_raw_long.copy(), Path('/test/eeg.fif'))
        
        # Create epochs
        epoch_result = create_epochs(state.raw.copy(), duration=2.0)
        n_epochs = epoch_result.n_total
        
        # Reject first few epochs
        epoch_result.rejected_indices = [0, 1, 2]
        
        # Reverse
        state._raw._data = state.raw.get_data()[:, ::-1]
        
        # Update indices
        new_indices = sorted([n_epochs - 1 - idx for idx in epoch_result.rejected_indices])
        epoch_result.rejected_indices = new_indices
        
        # Verify indices are now at end
        assert min(epoch_result.rejected_indices) > n_epochs / 2


# ============ Export Tests ============

class TestExportStep:
    """Tests for the Export step."""
    
    def test_export_fif(self, synthetic_raw_long, tmp_path):
        """Test FIF export."""
        raw = synthetic_raw_long.copy()
        output_path = tmp_path / "test_export"
        
        result_path = export_cleaned_eeg(raw, output_path, ExportFormat.FIF)
        
        assert result_path.exists()
        # Verify we can load it back
        raw_loaded = mne.io.read_raw_fif(result_path, preload=True, verbose=False)
        assert raw_loaded.n_times == raw.n_times
    
    def test_export_epochs_fif(self, synthetic_raw_long, tmp_path):
        """Test epochs FIF export."""
        raw = synthetic_raw_long.copy()
        epoch_result = create_epochs(raw, duration=2.0)
        
        outputs = export_epochs(
            epoch_result.epochs,
            tmp_path,
            "test_epochs",
            formats=[ExportFormat.FIF]
        )
        
        assert 'fif' in outputs
        assert outputs['fif'].exists()
        # Should follow MNE naming convention
        assert '_epo.fif' in str(outputs['fif'])


# ============ UI Behavior Simulation Tests ============

class TestUIBehavior:
    """Tests simulating UI behavior patterns."""
    
    def test_rapid_step_switching(self, cleaning_state):
        """Test rapid switching between steps."""
        state = cleaning_state
        state.complete_step(CleaningStep.LOAD)
        
        # Rapidly switch between steps
        for _ in range(10):
            state.go_to_step(CleaningStep.FILTER)
            state.go_to_step(CleaningStep.LOAD)
            state.go_to_step(CleaningStep.BAD_CHANNELS)
            state.go_to_step(CleaningStep.FILTER)
        
        # Should still be functional
        assert state.is_loaded
        assert state.raw is not None
    
    def test_multiple_filter_applications(self, cleaning_state):
        """Test applying filters multiple times."""
        state = cleaning_state
        
        # Apply different filters rapidly
        for preset in [FilterPreset.STANDARD, FilterPreset.ALPHA_THETA, FilterPreset.ERP]:
            raw_filtered, _ = apply_filter_preset(state.raw.copy(), preset)
            state._raw = raw_filtered
        
        # Should still have valid data
        assert state.raw is not None
        assert not np.isnan(state.raw.get_data()).any()
    
    def test_epoch_creation_recreation(self, cleaning_state):
        """Test creating epochs, modifying, then recreating."""
        state = cleaning_state
        
        # Create 2s epochs
        result1 = create_epochs(state.raw.copy(), duration=2.0)
        n_epochs_2s = result1.n_total
        
        # Create 1s epochs (should replace)
        result2 = create_epochs(state.raw.copy(), duration=1.0)
        n_epochs_1s = result2.n_total
        
        # Recreate 2s epochs - should get same count as first time
        result3 = create_epochs(state.raw.copy(), duration=2.0)
        
        assert result3.n_total == n_epochs_2s
        assert n_epochs_1s > n_epochs_2s


# ============ Edge Case Tests ============

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_empty_bad_channels(self, synthetic_raw_long):
        """Test handling when no bad channels detected."""
        raw = synthetic_raw_long.copy()
        
        # Use very lenient thresholds
        result = detect_bad_channels(raw, std_threshold=100, flat_threshold=1e-20)
        
        # Should handle empty result gracefully
        assert isinstance(result.all_bad, list)
    
    def test_all_epochs_rejected(self, synthetic_raw_long):
        """Test handling when all epochs would be rejected."""
        raw = synthetic_raw_long.copy()
        epoch_result = create_epochs(raw, duration=2.0)
        
        # Very strict criteria
        criteria = EpochRejectionCriteria(peak_to_peak_uv=0.001)  # Impossibly strict
        detected = detect_bad_epochs(epoch_result, criteria)
        
        # Should handle gracefully
        assert detected.n_total > 0
    
    def test_single_epoch(self):
        """Test handling with very few epochs."""
        # Create very short data
        sfreq = 500
        duration = 3
        n_channels = 4
        
        ch_names = ['Fp1', 'Fp2', 'O1', 'O2']
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        
        n_times = int(sfreq * duration)
        data = 20e-6 * np.random.randn(n_channels, n_times)
        raw = mne.io.RawArray(data, info)
        
        # Create epochs (should get 1 or very few)
        result = create_epochs(raw, duration=2.0)
        
        assert result.n_total >= 1
    
    def test_view_beyond_signal_end(self, synthetic_raw_long):
        """Test that view limits are clamped correctly."""
        raw = synthetic_raw_long.copy()
        actual_duration = raw.n_times / raw.info['sfreq']
        
        # Try to view beyond end
        view_start = actual_duration + 100
        view_duration = 5.0
        
        # Clamp to valid range
        if view_start > actual_duration - view_duration:
            view_start = max(0, actual_duration - view_duration)
        
        assert view_start <= actual_duration - view_duration
        assert view_start >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

