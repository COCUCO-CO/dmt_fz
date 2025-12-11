"""Tests for EEG sync utilities."""
import pytest
from pathlib import Path


class TestEpochToTime:
    """Tests for epoch to time conversion."""
    
    def test_epoch_to_time_basic(self):
        """Test basic epoch to time conversion."""
        from app.visualization.components.eeg_sync import epoch_to_time
        
        assert epoch_to_time(0, 2.0) == 0.0
        assert epoch_to_time(1, 2.0) == 2.0
        assert epoch_to_time(5, 2.0) == 10.0
        assert epoch_to_time(10, 1.5) == 15.0
    
    def test_epoch_to_time_fractional(self):
        """Test with fractional epoch duration."""
        from app.visualization.components.eeg_sync import epoch_to_time
        
        assert epoch_to_time(3, 0.5) == 1.5
        assert epoch_to_time(4, 2.5) == 10.0


class TestClampViewToEEG:
    """Tests for view clamping."""
    
    def test_clamp_within_range(self):
        """Test that valid positions are not changed."""
        from app.visualization.components.eeg_sync import clamp_view_to_eeg
        
        result = clamp_view_to_eeg(10.0, 2.0, 100.0)
        assert result == 10.0
    
    def test_clamp_at_start(self):
        """Test clamping at start of EEG."""
        from app.visualization.components.eeg_sync import clamp_view_to_eeg
        
        result = clamp_view_to_eeg(-5.0, 2.0, 100.0)
        assert result == 0.0
    
    def test_clamp_at_end(self):
        """Test clamping at end of EEG."""
        from app.visualization.components.eeg_sync import clamp_view_to_eeg
        
        result = clamp_view_to_eeg(99.0, 2.0, 100.0)
        assert result == 98.0  # max_start = 100 - 2 = 98
    
    def test_clamp_short_eeg(self):
        """Test with EEG shorter than view duration."""
        from app.visualization.components.eeg_sync import clamp_view_to_eeg
        
        result = clamp_view_to_eeg(0.0, 10.0, 5.0)
        assert result == 0.0  # max_start = 0 (can't have negative)


class TestGetEpochCount:
    """Tests for epoch count calculation."""
    
    def test_get_epoch_count_exact(self):
        """Test with exact division."""
        from app.visualization.components.eeg_sync import get_epoch_count
        
        assert get_epoch_count(100.0, 2.0) == 50
        assert get_epoch_count(60.0, 2.0) == 30
    
    def test_get_epoch_count_remainder(self):
        """Test with remainder (partial epoch not counted)."""
        from app.visualization.components.eeg_sync import get_epoch_count
        
        assert get_epoch_count(101.0, 2.0) == 50  # 1 second left over
        assert get_epoch_count(61.5, 2.0) == 30   # 1.5 seconds left over
    
    def test_get_epoch_count_zero_duration(self):
        """Test with zero epoch duration."""
        from app.visualization.components.eeg_sync import get_epoch_count
        
        assert get_epoch_count(100.0, 0.0) == 0
        assert get_epoch_count(100.0, -1.0) == 0


class TestFindEEGFilesInDir:
    """Tests for finding EEG files."""
    
    def test_find_eeg_files_existing(self, temp_eeg_dir):
        """Test finding existing EEG files."""
        from app.visualization.components.eeg_sync import find_eeg_files_in_dir
        
        files = find_eeg_files_in_dir(temp_eeg_dir / 'DMT')
        assert len(files) > 0
        assert all(f.suffix == '.set' for f in files)
    
    def test_find_eeg_files_empty_dir(self, tmp_path):
        """Test with empty directory."""
        from app.visualization.components.eeg_sync import find_eeg_files_in_dir
        
        empty_dir = tmp_path / 'empty'
        empty_dir.mkdir()
        files = find_eeg_files_in_dir(empty_dir)
        assert files == []
    
    def test_find_eeg_files_nonexistent_dir(self):
        """Test with non-existent directory."""
        from app.visualization.components.eeg_sync import find_eeg_files_in_dir
        
        files = find_eeg_files_in_dir(Path('/nonexistent/path'))
        assert files == []
