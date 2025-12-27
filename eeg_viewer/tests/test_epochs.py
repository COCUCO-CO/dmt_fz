"""
Tests for cleaning/epochs.py - Epoch creation and rejection.
"""


from cleaning.epochs import (
    EpochRejectionCriteria, EpochResult, create_epochs, detect_bad_epochs,
    apply_epoch_rejection, get_epoch_data, get_rejection_stats
)


class TestEpochRejectionCriteria:
    """Tests for EpochRejectionCriteria dataclass."""
    
    def test_default_values(self):
        """Test default rejection criteria."""
        criteria = EpochRejectionCriteria()
        
        assert criteria.peak_to_peak_uv == 150.0
        assert criteria.flat_uv == 0.5
        assert criteria.gradient_uv_ms == 10.0
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        criteria = EpochRejectionCriteria(peak_to_peak_uv=200)
        d = criteria.to_dict()
        
        assert d['peak_to_peak_uv'] == 200
    
    def test_from_dict(self):
        """Test creating from dictionary."""
        d = {'peak_to_peak_uv': 100, 'flat_uv': 1.0}
        criteria = EpochRejectionCriteria.from_dict(d)
        
        assert criteria.peak_to_peak_uv == 100
        assert criteria.flat_uv == 1.0
    
    def test_preset_default(self):
        """Test default preset."""
        criteria = EpochRejectionCriteria.default()
        assert criteria.peak_to_peak_uv == 150.0
    
    def test_preset_lenient(self):
        """Test lenient preset."""
        criteria = EpochRejectionCriteria.lenient()
        assert criteria.peak_to_peak_uv == 200.0
    
    def test_preset_strict(self):
        """Test strict preset."""
        criteria = EpochRejectionCriteria.strict()
        assert criteria.peak_to_peak_uv == 100.0


class TestEpochResult:
    """Tests for EpochResult dataclass."""
    
    def test_empty_result(self):
        """Test empty result."""
        result = EpochResult()
        
        assert result.epochs is None
        assert result.n_total == 0
        assert result.n_good == 0
        assert result.n_rejected == 0
    
    def test_rejection_rate_zero(self):
        """Test rejection rate with no epochs."""
        result = EpochResult()
        
        assert result.rejection_rate == 0.0
    
    def test_rejection_rate(self):
        """Test rejection rate calculation."""
        result = EpochResult(n_total=100, n_rejected=20)
        
        assert result.rejection_rate == 20.0
    
    def test_get_rejection_summary(self):
        """Test getting rejection summary."""
        result = EpochResult(
            rejection_reasons={
                0: ['peak_to_peak'],
                1: ['flat'],
                2: ['peak_to_peak', 'gradient']
            }
        )
        
        summary = result.get_rejection_summary()
        
        assert summary['peak_to_peak'] == 2
        assert summary['flat'] == 1
        assert summary['gradient'] == 1


class TestCreateEpochs:
    """Tests for create_epochs function."""
    
    def test_create_basic_epochs(self, synthetic_raw):
        """Test creating basic epochs."""
        result = create_epochs(synthetic_raw, duration=1.0)
        
        assert result.epochs is not None
        assert result.n_total > 0
        assert result.n_good == result.n_total
        assert result.n_rejected == 0
    
    def test_epoch_duration(self, synthetic_raw):
        """Test epoch duration is respected."""
        result = create_epochs(synthetic_raw, duration=1.0)
        
        # Each epoch should be 1 second
        epoch_samples = len(result.epochs.times)
        expected_samples = int(1.0 * synthetic_raw.info['sfreq']) + 1  # +1 for inclusive
        
        assert abs(epoch_samples - expected_samples) <= 1
    
    def test_epoch_overlap(self, synthetic_raw):
        """Test epoch overlap."""
        result_no_overlap = create_epochs(synthetic_raw, duration=1.0, overlap=0.0)
        result_overlap = create_epochs(synthetic_raw, duration=1.0, overlap=0.5)
        
        # With overlap, should have more epochs
        assert result_overlap.n_total >= result_no_overlap.n_total
    
    def test_epoch_stats_populated(self, synthetic_raw):
        """Test that epoch stats are populated."""
        result = create_epochs(synthetic_raw, duration=1.0)
        
        assert len(result.epoch_stats) == result.n_total
        assert 'peak_to_peak' in result.epoch_stats[0]


class TestDetectBadEpochs:
    """Tests for detect_bad_epochs function."""
    
    def test_detect_with_none_epochs(self):
        """Test with no epochs."""
        result = EpochResult()
        criteria = EpochRejectionCriteria()
        
        detected = detect_bad_epochs(result, criteria)
        
        assert detected.n_rejected == 0
    
    def test_detect_with_good_data(self, synthetic_raw):
        """Test detection with good synthetic data."""
        epoch_result = create_epochs(synthetic_raw, duration=1.0)
        # Use lenient criteria since synthetic data has 10Hz oscillation which causes high gradient
        criteria = EpochRejectionCriteria.lenient()
        
        detected = detect_bad_epochs(epoch_result, criteria)
        
        # Synthetic data should be mostly good with lenient criteria
        assert detected.n_good > 0
    
    def test_detect_with_artifacts(self, synthetic_raw_with_artifacts):
        """Test detection with artifact data."""
        epoch_result = create_epochs(synthetic_raw_with_artifacts, duration=1.0)
        criteria = EpochRejectionCriteria(peak_to_peak_uv=50)  # Strict
        
        detected = detect_bad_epochs(epoch_result, criteria)
        
        # Should reject some epochs due to noisy channel
        assert detected.n_rejected > 0
    
    def test_rejection_reasons_populated(self, synthetic_raw_with_artifacts):
        """Test that rejection reasons are populated."""
        epoch_result = create_epochs(synthetic_raw_with_artifacts, duration=1.0)
        criteria = EpochRejectionCriteria(peak_to_peak_uv=50)
        
        detected = detect_bad_epochs(epoch_result, criteria)
        
        if detected.n_rejected > 0:
            # Check that reasons are recorded
            assert len(detected.rejection_reasons) > 0


class TestApplyEpochRejection:
    """Tests for apply_epoch_rejection function."""
    
    def test_apply_with_none_epochs(self):
        """Test with no epochs."""
        result = EpochResult()
        
        applied = apply_epoch_rejection(result)
        
        assert applied.epochs is None
    
    def test_apply_no_rejections(self, synthetic_raw):
        """Test with no rejections."""
        epoch_result = create_epochs(synthetic_raw, duration=1.0)
        original_n = epoch_result.n_total
        
        applied = apply_epoch_rejection(epoch_result)
        
        # Should keep all epochs
        assert len(applied.epochs) == original_n
    
    def test_apply_with_rejections(self, synthetic_raw):
        """Test with some rejections."""
        epoch_result = create_epochs(synthetic_raw, duration=1.0)
        epoch_result.rejected_indices = [0, 1]  # Manually mark first two
        
        applied = apply_epoch_rejection(epoch_result)
        
        assert applied.n_rejected == 2
        assert len(applied.epochs) == epoch_result.n_total - 2
    
    def test_apply_additional_rejects(self, synthetic_raw):
        """Test with additional rejections."""
        epoch_result = create_epochs(synthetic_raw, duration=1.0)
        epoch_result.rejected_indices = [0]
        
        applied = apply_epoch_rejection(epoch_result, additional_rejects=[1, 2])
        
        assert applied.n_rejected == 3


class TestGetEpochData:
    """Tests for get_epoch_data function."""
    
    def test_get_with_none_epochs(self):
        """Test with no epochs."""
        result = EpochResult()
        data, times, ch_names = get_epoch_data(result, 0)
        
        assert len(data) == 0
        assert len(times) == 0
        assert len(ch_names) == 0
    
    def test_get_valid_epoch(self, synthetic_raw):
        """Test getting a valid epoch."""
        epoch_result = create_epochs(synthetic_raw, duration=1.0)
        data, times, ch_names = get_epoch_data(epoch_result, 0)
        
        assert data.shape[0] == len(synthetic_raw.ch_names)  # n_channels
        assert len(times) == data.shape[1]  # n_samples
        assert len(ch_names) == data.shape[0]
    
    def test_get_invalid_index(self, synthetic_raw):
        """Test getting invalid epoch index."""
        epoch_result = create_epochs(synthetic_raw, duration=1.0)
        data, times, ch_names = get_epoch_data(epoch_result, 9999)
        
        assert len(data) == 0


class TestGetRejectionStats:
    """Tests for get_rejection_stats function."""
    
    def test_stats_structure(self):
        """Test stats dictionary structure."""
        result = EpochResult(
            n_total=100,
            n_good=80,
            n_rejected=20,
            rejection_reasons={0: ['peak_to_peak']}
        )
        
        stats = get_rejection_stats(result)
        
        assert 'n_total' in stats
        assert 'n_good' in stats
        assert 'n_rejected' in stats
        assert 'rejection_rate_percent' in stats
        assert 'rejection_by_reason' in stats
    
    def test_stats_values(self):
        """Test stats values are correct."""
        result = EpochResult(n_total=100, n_good=75, n_rejected=25)
        
        stats = get_rejection_stats(result)
        
        assert stats['n_total'] == 100
        assert stats['n_good'] == 75
        assert stats['n_rejected'] == 25
        assert stats['rejection_rate_percent'] == 25.0


class TestEpochConsistency:
    """Tests for epoch creation consistency - epochs should not accumulate."""
    
    def test_recreate_epochs_same_count(self, synthetic_raw):
        """Test that recreating epochs with same duration gives same count."""
        result1 = create_epochs(synthetic_raw, duration=2.0)
        result2 = create_epochs(synthetic_raw, duration=2.0)
        result3 = create_epochs(synthetic_raw, duration=2.0)
        
        # All should have exactly the same number of epochs
        assert result1.n_total == result2.n_total == result3.n_total
    
    def test_recreate_after_different_duration(self, synthetic_raw):
        """Test that changing duration then changing back gives same count."""
        # Create with 2s
        result_2s_first = create_epochs(synthetic_raw, duration=2.0)
        
        # Create with 1s (different)
        result_1s = create_epochs(synthetic_raw, duration=1.0)
        
        # Create with 2s again - should match first 2s result
        result_2s_second = create_epochs(synthetic_raw, duration=2.0)
        
        assert result_2s_first.n_total == result_2s_second.n_total
        # 1s epochs should have more epochs than 2s
        assert result_1s.n_total > result_2s_first.n_total
    
    def test_epochs_independent_of_previous_calls(self, synthetic_raw):
        """Test that epoch counts are computed fresh each time, not accumulated."""
        # Calculate expected number for 2s epochs
        sfreq = synthetic_raw.info['sfreq']
        total_time = synthetic_raw.n_times / sfreq
        expected_2s = int((total_time - 2.0) / 2.0) + 1  # Approximate
        
        # Call create_epochs multiple times
        for _ in range(5):
            result = create_epochs(synthetic_raw, duration=2.0)
            # Should never be double the expected amount
            assert result.n_total < expected_2s * 2, \
                f"Epochs seem to be accumulating: got {result.n_total}, expected ~{expected_2s}"
    
    def test_epoch_count_formula(self, synthetic_raw):
        """Test that epoch count follows expected formula."""
        duration = 2.0
        overlap = 0.0
        
        result = create_epochs(synthetic_raw, duration=duration, overlap=overlap)
        
        sfreq = synthetic_raw.info['sfreq']
        total_time = synthetic_raw.n_times / sfreq
        step = duration - overlap
        expected = int((total_time - duration) / step) + 1
        
        # Allow for rounding differences
        assert abs(result.n_total - expected) <= 1, \
            f"Expected ~{expected} epochs, got {result.n_total}"

