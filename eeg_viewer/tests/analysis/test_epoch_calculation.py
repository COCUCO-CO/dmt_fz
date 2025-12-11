"""Tests for epoch time calculation utilities."""
import pytest


class TestEpochToTime:
    """Tests for converting epoch index to time position."""
    
    def test_first_epoch_starts_at_zero(self):
        """Test that first epoch (idx=0) starts at time 0."""
        from app.visualization.components.eeg_sync import epoch_to_time
        
        result = epoch_to_time(epoch_idx=0, epoch_duration=2.0)
        
        assert result == 0.0
    
    def test_second_epoch_correct_offset(self):
        """Test second epoch starts at epoch_duration."""
        from app.visualization.components.eeg_sync import epoch_to_time
        
        result = epoch_to_time(epoch_idx=1, epoch_duration=2.0)
        
        assert result == 2.0
    
    def test_arbitrary_epoch_calculation(self):
        """Test arbitrary epoch index calculation."""
        from app.visualization.components.eeg_sync import epoch_to_time
        
        result = epoch_to_time(epoch_idx=42, epoch_duration=2.0)
        
        assert result == 84.0  # 42 * 2.0
    
    def test_different_epoch_duration(self):
        """Test with different epoch durations."""
        from app.visualization.components.eeg_sync import epoch_to_time
        
        # 1 second epochs
        assert epoch_to_time(epoch_idx=5, epoch_duration=1.0) == 5.0
        
        # 5 second epochs
        assert epoch_to_time(epoch_idx=5, epoch_duration=5.0) == 25.0
        
        # 0.5 second epochs
        assert epoch_to_time(epoch_idx=10, epoch_duration=0.5) == 5.0
    
    def test_float_epoch_duration(self):
        """Test with fractional epoch duration."""
        from app.visualization.components.eeg_sync import epoch_to_time
        
        result = epoch_to_time(epoch_idx=3, epoch_duration=2.5)
        
        assert result == pytest.approx(7.5)


class TestClampViewToEEG:
    """Tests for clamping view position to valid EEG range."""
    
    def test_clamp_within_range(self, mock_eeg_data):
        """Test position within valid range is unchanged."""
        from app.visualization.components.eeg_sync import clamp_view_to_eeg
        
        result = clamp_view_to_eeg(
            view_start=50.0,
            view_duration=2.0,
            eeg_duration=300.0
        )
        
        assert result == 50.0
    
    def test_clamp_negative_to_zero(self, mock_eeg_data):
        """Test negative position is clamped to zero."""
        from app.visualization.components.eeg_sync import clamp_view_to_eeg
        
        result = clamp_view_to_eeg(
            view_start=-10.0,
            view_duration=2.0,
            eeg_duration=300.0
        )
        
        assert result == 0.0
    
    def test_clamp_beyond_end(self, mock_eeg_data):
        """Test position beyond EEG end is clamped."""
        from app.visualization.components.eeg_sync import clamp_view_to_eeg
        
        result = clamp_view_to_eeg(
            view_start=350.0,  # Beyond 300s
            view_duration=2.0,
            eeg_duration=300.0
        )
        
        assert result == 298.0  # 300 - 2
    
    def test_clamp_at_boundary(self, mock_eeg_data):
        """Test position at exact boundary."""
        from app.visualization.components.eeg_sync import clamp_view_to_eeg
        
        result = clamp_view_to_eeg(
            view_start=298.0,
            view_duration=2.0,
            eeg_duration=300.0
        )
        
        assert result == 298.0
    
    def test_clamp_short_eeg(self):
        """Test clamping with very short EEG."""
        from app.visualization.components.eeg_sync import clamp_view_to_eeg
        
        # EEG shorter than view duration
        result = clamp_view_to_eeg(
            view_start=5.0,
            view_duration=10.0,
            eeg_duration=8.0
        )
        
        assert result == 0.0  # Max is max(0, 8-10) = 0


class TestEpochCount:
    """Tests for calculating number of epochs in EEG."""
    
    def test_exact_division(self):
        """Test when EEG duration divides evenly by epoch duration."""
        from app.visualization.components.eeg_sync import get_epoch_count
        
        result = get_epoch_count(eeg_duration=300.0, epoch_duration=2.0)
        
        assert result == 150
    
    def test_partial_last_epoch(self):
        """Test when last epoch is partial (floor division)."""
        from app.visualization.components.eeg_sync import get_epoch_count
        
        result = get_epoch_count(eeg_duration=301.0, epoch_duration=2.0)
        
        assert result == 150  # Floor, not 151
    
    def test_single_epoch(self):
        """Test EEG with only one epoch."""
        from app.visualization.components.eeg_sync import get_epoch_count
        
        result = get_epoch_count(eeg_duration=2.0, epoch_duration=2.0)
        
        assert result == 1
    
    def test_zero_epochs_short_eeg(self):
        """Test EEG shorter than one epoch."""
        from app.visualization.components.eeg_sync import get_epoch_count
        
        result = get_epoch_count(eeg_duration=1.5, epoch_duration=2.0)
        
        assert result == 0

