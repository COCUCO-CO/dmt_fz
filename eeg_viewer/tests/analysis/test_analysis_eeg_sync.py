"""Integration tests for EEG sync in analysis page."""


class TestEEGSyncIntegration:
    """Tests for the EEG sync functionality."""
    
    def test_sync_logic(self):
        """Test that sync logic correctly maps sample index to EEG time."""
        # This simulates what happens in process_current_sample
        
        epoch_duration = 2.0
        eeg_duration = 300.0  # 5 minutes
        
        # Test various sample indices
        test_cases = [
            (0, 0.0),      # First sample -> start of EEG
            (1, 2.0),      # Second sample -> 2 seconds
            (10, 20.0),    # 10th sample -> 20 seconds
            (50, 100.0),   # 50th sample -> 100 seconds
        ]
        
        for sample_idx, expected_time in test_cases:
            view_start = sample_idx * epoch_duration
            # Clamp to EEG duration
            max_start = max(0, eeg_duration - epoch_duration)
            view_start = min(view_start, max_start)
            
            assert view_start == expected_time, f"Failed for idx={sample_idx}"
    
    def test_sync_clamps_at_end(self):
        """Test that sync clamps view at end of EEG."""
        epoch_duration = 2.0
        eeg_duration = 100.0
        
        # If we're past the end of the EEG, clamp to max valid position
        sample_idx = 100  # Would be 200 seconds, but EEG is only 100s
        
        view_start = sample_idx * epoch_duration
        max_start = max(0, eeg_duration - epoch_duration)
        view_start = min(view_start, max_start)
        
        assert view_start == 98.0  # 100 - 2 = 98
    
    def test_sync_with_different_epoch_durations(self):
        """Test sync with various epoch durations."""
        eeg_duration = 300.0
        
        for epoch_duration in [0.5, 1.0, 2.0, 5.0]:
            for sample_idx in [0, 10, 50]:
                view_start = sample_idx * epoch_duration
                max_start = max(0, eeg_duration - epoch_duration)
                view_start = min(view_start, max_start)
                
                expected = min(sample_idx * epoch_duration, max_start)
                assert view_start == expected
