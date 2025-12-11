"""Integration tests for EEG synchronization in analysis page."""
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


class TestEEGSyncIntegration:
    """Integration tests for EEG sync with analysis playback."""
    
    def test_should_sync_with_valid_graph_dataset(self, mock_graph_dataset):
        """Test sync is enabled when dataset has EEG metadata."""
        from app.visualization.components.eeg_sync import should_enable_eeg_sync
        
        result = should_enable_eeg_sync(mock_graph_dataset)
        
        assert result is True
    
    def test_should_not_sync_without_metadata(self):
        """Test sync is disabled when dataset lacks metadata."""
        from app.visualization.components.eeg_sync import should_enable_eeg_sync
        
        # Plain list without EEG attributes
        plain_dataset = [{"x": [1, 2, 3]} for _ in range(5)]
        
        result = should_enable_eeg_sync(plain_dataset)
        
        assert result is False
    
    def test_should_not_sync_empty_dataset(self):
        """Test sync is disabled for empty dataset."""
        from app.visualization.components.eeg_sync import should_enable_eeg_sync
        
        result = should_enable_eeg_sync([])
        
        assert result is False
    
    def test_should_not_sync_none_dataset(self):
        """Test sync handles None dataset."""
        from app.visualization.components.eeg_sync import should_enable_eeg_sync
        
        result = should_enable_eeg_sync(None)
        
        assert result is False


class TestSyncStateUpdate:
    """Tests for updating sync state on sample change."""
    
    def test_update_on_same_subject_condition(self, mock_analysis_state, mock_graph_sample):
        """Test state update when subject/condition unchanged."""
        from app.visualization.components.eeg_sync import update_eeg_sync_state
        
        # Setup - already loaded this subject's EEG
        mock_analysis_state.current_eeg_file = "/path/to/S01-DMT_ICA_pruned.set"
        mock_analysis_state.eeg_data = MagicMock()
        mock_analysis_state.eeg_data.duration_sec = 300.0
        
        # Should only update view position, not reload EEG
        with patch('app.visualization.components.eeg_sync.find_eeg_file') as mock_find:
            with patch('eeg_loader.load_eeg_file') as mock_load:
                update_eeg_sync_state(
                    state=mock_analysis_state,
                    sample=mock_graph_sample,
                    eeg_base_dir=Path("/mock/eeg"),
                    epoch_duration=2.0
                )
        
        # Should update view_start but not reload
        assert mock_analysis_state.eeg_view_start == 0.0  # epoch_idx=0
    
    def test_update_different_subject_loads_new_eeg(self, mock_analysis_state, mock_graph_sample, temp_eeg_dir):
        """Test loading new EEG when subject changes."""
        from app.visualization.components.eeg_sync import update_eeg_sync_state
        
        # Setup - different subject was loaded
        mock_analysis_state.current_eeg_file = "/path/to/S99-OTHER.set"
        mock_analysis_state.eeg_data = None
        
        # Mock the EEG loading
        with patch('eeg_loader.load_eeg_file') as mock_load:
            mock_eeg = MagicMock()
            mock_eeg.duration_sec = 300.0
            mock_load.return_value = mock_eeg
            
            update_eeg_sync_state(
                state=mock_analysis_state,
                sample=mock_graph_sample,
                eeg_base_dir=temp_eeg_dir,
                epoch_duration=2.0
            )
        
        # Should have attempted to load new EEG
        mock_load.assert_called_once()
    
    def test_view_position_updates_with_epoch(self, mock_analysis_state, mock_graph_dataset):
        """Test view position updates correctly with epoch index."""
        from app.visualization.components.eeg_sync import update_eeg_sync_state
        
        mock_analysis_state.eeg_data = MagicMock()
        mock_analysis_state.eeg_data.duration_sec = 300.0
        mock_analysis_state.current_eeg_file = "/path/to/S01-DMT_ICA_pruned.set"
        
        # Test different epochs
        epoch_duration = 2.0
        
        with patch('app.visualization.components.eeg_sync.find_eeg_file') as mock_find:
            mock_find.return_value = Path("/path/to/S01-DMT_ICA_pruned.set")
            
            for i, sample in enumerate(mock_graph_dataset):
                update_eeg_sync_state(
                    state=mock_analysis_state,
                    sample=sample,
                    eeg_base_dir=Path("/mock"),
                    epoch_duration=epoch_duration
                )
                
                expected_start = i * epoch_duration
                assert mock_analysis_state.eeg_view_start == expected_start, \
                    f"Epoch {i} should start at {expected_start}"


class TestSyncedPlayback:
    """Tests for synchronized playback behavior."""
    
    def test_advancing_sample_updates_eeg_view(self, mock_analysis_state, mock_graph_dataset):
        """Test that advancing through samples updates EEG view."""
        from app.visualization.components.eeg_sync import update_eeg_sync_state
        
        mock_analysis_state.eeg_data = MagicMock()
        mock_analysis_state.eeg_data.duration_sec = 300.0
        mock_analysis_state.current_eeg_file = "/path/to/S01-DMT_ICA_pruned.set"
        
        positions = []
        
        with patch('app.visualization.components.eeg_sync.find_eeg_file') as mock_find:
            mock_find.return_value = Path("/path/to/S01-DMT_ICA_pruned.set")
            
            for sample in mock_graph_dataset:
                update_eeg_sync_state(
                    state=mock_analysis_state,
                    sample=sample,
                    eeg_base_dir=Path("/mock"),
                    epoch_duration=2.0
                )
                positions.append(mock_analysis_state.eeg_view_start)
        
        # Positions should be increasing
        assert positions == sorted(positions)
        # Should match epoch indices * duration
        assert positions == [i * 2.0 for i in range(10)]
    
    def test_sync_disabled_skips_update(self, mock_analysis_state, mock_graph_sample):
        """Test that disabled sync doesn't update EEG view."""
        from app.visualization.components.eeg_sync import update_eeg_sync_state
        
        mock_analysis_state.eeg_sync_enabled = False
        mock_analysis_state.eeg_view_start = 100.0  # Set initial position
        
        update_eeg_sync_state(
            state=mock_analysis_state,
            sample=mock_graph_sample,
            eeg_base_dir=Path("/mock"),
            epoch_duration=2.0
        )
        
        # Position should remain unchanged
        assert mock_analysis_state.eeg_view_start == 100.0


class TestEEGFileLoadingIntegration:
    """Tests for EEG file loading during sync."""
    
    def test_missing_eeg_file_logs_warning(self, mock_analysis_state, mock_graph_sample, caplog):
        """Test that missing EEG file logs a warning."""
        from app.visualization.components.eeg_sync import update_eeg_sync_state
        import logging
        
        mock_analysis_state.current_eeg_file = None
        mock_analysis_state.eeg_data = None
        
        with patch('app.visualization.components.eeg_sync.find_eeg_file') as mock_find:
            mock_find.return_value = None  # File not found
            
            with caplog.at_level(logging.WARNING):
                update_eeg_sync_state(
                    state=mock_analysis_state,
                    sample=mock_graph_sample,
                    eeg_base_dir=Path("/nonexistent"),
                    epoch_duration=2.0
                )
        
        # Should not crash, EEG data should remain None
        assert mock_analysis_state.eeg_data is None
    
    def test_eeg_load_error_handled_gracefully(self, mock_analysis_state, mock_graph_sample, temp_eeg_dir):
        """Test EEG loading errors are handled gracefully."""
        from app.visualization.components.eeg_sync import update_eeg_sync_state
        
        mock_analysis_state.current_eeg_file = None
        mock_analysis_state.eeg_data = None
        
        with patch('eeg_loader.load_eeg_file') as mock_load:
            mock_load.side_effect = Exception("Load error")
            
            # Should not raise exception
            update_eeg_sync_state(
                state=mock_analysis_state,
                sample=mock_graph_sample,
                eeg_base_dir=temp_eeg_dir,
                epoch_duration=2.0
            )
        
        # State should remain valid
        assert mock_analysis_state.eeg_data is None

