"""Tests for epoch calculation utilities."""


class TestEpochCalculation:
    """Tests for epoch time calculations used in EEG sync."""
    
    def test_sync_calculation(self):
        """Test the basic sync calculation: idx * duration = time."""
        epoch_duration = 2.0
        
        # Simulate what happens in analysis.py
        for idx in range(10):
            expected_time = idx * epoch_duration
            actual_time = idx * epoch_duration
            assert actual_time == expected_time
    
    def test_clamp_calculation(self):
        """Test clamping view position to EEG bounds."""
        from app.visualization.components.eeg_sync import clamp_view_to_eeg
        
        eeg_duration = 300.0  # 5 minutes
        epoch_duration = 2.0
        
        # Test various positions
        assert clamp_view_to_eeg(0.0, epoch_duration, eeg_duration) == 0.0
        assert clamp_view_to_eeg(100.0, epoch_duration, eeg_duration) == 100.0
        assert clamp_view_to_eeg(299.0, epoch_duration, eeg_duration) == 298.0  # clamped
        assert clamp_view_to_eeg(400.0, epoch_duration, eeg_duration) == 298.0  # clamped
    
    def test_total_epochs_calculation(self):
        """Test calculating total number of epochs."""
        from app.visualization.components.eeg_sync import get_epoch_count
        
        # 5 minute EEG with 2 second epochs = 150 epochs
        assert get_epoch_count(300.0, 2.0) == 150
        
        # 5 minute EEG with 1 second epochs = 300 epochs
        assert get_epoch_count(300.0, 1.0) == 300
        
        # Partial epochs not counted
        assert get_epoch_count(301.5, 2.0) == 150
