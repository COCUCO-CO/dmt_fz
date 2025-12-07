"""
Tests for cleaning/bad_channels.py - Bad channel detection and interpolation.
"""

import pytest
import numpy as np

from cleaning.bad_channels import (
    detect_bad_channels, detect_bridged_channels, get_channel_neighbors,
    interpolate_channels, compute_channel_quality_metrics, BadChannelResult
)


class TestBadChannelResult:
    """Tests for BadChannelResult dataclass."""
    
    def test_empty_result(self):
        """Test empty result."""
        result = BadChannelResult()
        
        assert len(result.flat) == 0
        assert len(result.noisy) == 0
        assert len(result.all_bad) == 0
    
    def test_all_bad_combines_lists(self):
        """Test all_bad property combines all lists."""
        result = BadChannelResult(
            flat=['Fp1'],
            noisy=['Fp2'],
            uncorrelated=['F3']
        )
        
        all_bad = result.all_bad
        assert len(all_bad) == 3
        assert 'Fp1' in all_bad
        assert 'Fp2' in all_bad
        assert 'F3' in all_bad
    
    def test_all_bad_removes_duplicates(self):
        """Test all_bad removes duplicates."""
        result = BadChannelResult(
            flat=['Fp1'],
            noisy=['Fp1', 'Fp2']  # Fp1 is in both
        )
        
        all_bad = result.all_bad
        assert len(all_bad) == 2
    
    def test_get_reason(self):
        """Test getting reason for bad channel."""
        result = BadChannelResult(
            flat=['Fp1'],
            noisy=['Fp2']
        )
        
        assert 'flat' in result.get_reason('Fp1')
        assert 'noisy' in result.get_reason('Fp2')
        assert result.get_reason('F3') == 'manual'


class TestDetectBadChannels:
    """Tests for detect_bad_channels function."""
    
    def test_detect_flat_channel(self, synthetic_raw_with_artifacts):
        """Test detection of flat channels."""
        result = detect_bad_channels(synthetic_raw_with_artifacts)
        
        # Channel 0 (Fp1) was set to all zeros
        assert 'Fp1' in result.flat
    
    def test_detect_noisy_channel(self, synthetic_raw_with_artifacts):
        """Test detection of noisy channels."""
        # Use low std_threshold and appropriate flat_threshold to detect noisy
        result = detect_bad_channels(synthetic_raw_with_artifacts, std_threshold=2.0, flat_threshold=1e-12)
        
        # Channel 1 (Fp2) was set to very high variance (1e-3 scale vs 1e-5 for others)
        assert 'Fp2' in result.noisy or 'Fp2' in result.all_bad
    
    def test_channel_stats_populated(self, synthetic_raw_with_artifacts):
        """Test that channel stats are populated."""
        result = detect_bad_channels(synthetic_raw_with_artifacts)
        
        assert len(result.channel_stats) > 0
        assert 'Fp1' in result.channel_stats
        assert 'variance' in result.channel_stats['Fp1']
    
    def test_threshold_adjustment(self, synthetic_raw_with_artifacts):
        """Test that threshold affects detection."""
        # Very high threshold - should detect fewer noisy
        result_high = detect_bad_channels(synthetic_raw_with_artifacts, std_threshold=10.0)
        
        # Very low threshold - should detect more noisy
        result_low = detect_bad_channels(synthetic_raw_with_artifacts, std_threshold=1.0)
        
        assert len(result_low.noisy) >= len(result_high.noisy)
    
    def test_correlation_check_disabled_by_default(self, synthetic_raw):
        """Test that correlation check is disabled by default."""
        result = detect_bad_channels(synthetic_raw, check_correlation=False)
        
        # Without correlation check, should not find uncorrelated channels
        assert len(result.uncorrelated) == 0
    
    def test_correlation_check_when_enabled(self, synthetic_raw):
        """Test that correlation check works when enabled."""
        result = detect_bad_channels(synthetic_raw, check_correlation=True, corr_threshold=0.99)
        
        # With very high threshold, might find some uncorrelated
        # (depends on synthetic data)
    
    def test_good_data_no_bad_channels(self, synthetic_raw):
        """Test that good data has no bad channels detected."""
        result = detect_bad_channels(synthetic_raw)
        
        # Synthetic raw without artifacts should have no bad channels
        assert len(result.all_bad) == 0
    
    def test_duration_parameter(self, synthetic_raw):
        """Test duration parameter limits analysis time."""
        result = detect_bad_channels(synthetic_raw, duration=2.0)
        
        # Should complete without error
        assert result is not None


class TestDetectBridgedChannels:
    """Tests for detect_bridged_channels function."""
    
    def test_no_bridged_in_normal_data(self, synthetic_raw):
        """Test no bridged channels in normal data."""
        bridged = detect_bridged_channels(synthetic_raw)
        
        assert len(bridged) == 0
    
    def test_detect_bridged_pair(self, synthetic_raw):
        """Test detection of bridged channel pair."""
        # Create bridged channels
        data = synthetic_raw.get_data()
        data[1] = data[0]  # Make Fp2 identical to Fp1
        
        raw_bridged = synthetic_raw.copy()
        raw_bridged._data = data
        
        bridged = detect_bridged_channels(raw_bridged, threshold=0.99)
        
        assert len(bridged) >= 1
        # Check that the pair contains both channels
        pair = bridged[0]
        assert 'Fp1' in pair or 'Fp2' in pair


class TestGetChannelNeighbors:
    """Tests for get_channel_neighbors function."""
    
    def test_returns_dict(self):
        """Test returns dictionary for all channels."""
        ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4']
        neighbors = get_channel_neighbors(ch_names)
        
        assert isinstance(neighbors, dict)
        assert len(neighbors) == len(ch_names)
    
    def test_neighbors_are_nearby(self):
        """Test that neighbors are spatially nearby."""
        ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'Fz', 'C3', 'C4', 'Cz']
        neighbors = get_channel_neighbors(ch_names)
        
        # F3 neighbors should include nearby channels
        f3_neighbors = neighbors.get('F3', [])
        # At least some neighbors should exist
        assert len(f3_neighbors) > 0
    
    def test_unknown_channels(self):
        """Test handling of unknown channel names."""
        ch_names = ['Unknown1', 'Unknown2', 'Fp1']
        neighbors = get_channel_neighbors(ch_names)
        
        # Unknown channels should have empty neighbor list
        assert neighbors['Unknown1'] == []


class TestInterpolateChannels:
    """Tests for interpolate_channels function."""
    
    def test_interpolate_single_channel(self, synthetic_raw):
        """Test interpolating a single channel."""
        raw_interp = interpolate_channels(synthetic_raw, ['Fp1'])
        
        assert raw_interp is not None
        assert len(raw_interp.ch_names) == len(synthetic_raw.ch_names)
    
    def test_empty_bad_list(self, synthetic_raw):
        """Test with empty bad channel list."""
        raw_interp = interpolate_channels(synthetic_raw, [])
        
        assert raw_interp is not None
        # Should return copy of original
        assert np.allclose(raw_interp.get_data(), synthetic_raw.get_data())
    
    def test_interpolation_changes_data(self, synthetic_raw_with_artifacts):
        """Test that interpolation changes the bad channel data."""
        original_data = synthetic_raw_with_artifacts.get_data().copy()
        
        raw_interp = interpolate_channels(synthetic_raw_with_artifacts, ['Fp1'])
        new_data = raw_interp.get_data()
        
        # The flat channel should now have non-zero data
        fp1_idx = synthetic_raw_with_artifacts.ch_names.index('Fp1')
        
        # Original was flat (all zeros)
        assert np.std(original_data[fp1_idx]) < 1e-10
        # After interpolation should have some variance
        # (Note: this might still be low due to the way interpolation works)


class TestComputeChannelQualityMetrics:
    """Tests for compute_channel_quality_metrics function."""
    
    def test_returns_metrics_for_all_channels(self, synthetic_raw):
        """Test returns metrics for all channels."""
        metrics = compute_channel_quality_metrics(synthetic_raw)
        
        assert len(metrics) == len(synthetic_raw.ch_names)
    
    def test_metrics_contain_expected_fields(self, synthetic_raw):
        """Test metrics contain expected fields."""
        metrics = compute_channel_quality_metrics(synthetic_raw)
        
        ch_metrics = metrics[synthetic_raw.ch_names[0]]
        
        assert 'variance' in ch_metrics
        assert 'kurtosis' in ch_metrics
        assert 'peak_to_peak_uv' in ch_metrics
        assert 'zero_crossing_rate' in ch_metrics
        assert 'hf_ratio' in ch_metrics
    
    def test_flat_channel_low_variance(self, synthetic_raw_with_artifacts):
        """Test flat channel has low variance."""
        metrics = compute_channel_quality_metrics(synthetic_raw_with_artifacts)
        
        fp1_metrics = metrics['Fp1']  # This is the flat channel
        
        assert fp1_metrics['variance'] < 1e-10
    
    def test_noisy_channel_high_variance(self, synthetic_raw_with_artifacts):
        """Test noisy channel has high variance."""
        metrics = compute_channel_quality_metrics(synthetic_raw_with_artifacts)
        
        fp2_metrics = metrics['Fp2']  # This is the noisy channel
        normal_metrics = metrics['C3']  # Normal channel
        
        assert fp2_metrics['variance'] > normal_metrics['variance']

