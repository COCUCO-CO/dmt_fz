"""Tests for EEG file finder utility."""
import pytest
from pathlib import Path


class TestFindEEGFile:
    """Tests for finding EEG files from graph metadata."""
    
    def test_find_eeg_file_standard_format(self, temp_eeg_dir):
        """Test finding EEG file with standard S01-DMT format."""
        from app.visualization.components.eeg_sync import find_eeg_file
        
        result = find_eeg_file(
            subject_id="S01",
            condition="DMT",
            eeg_base_dir=temp_eeg_dir
        )
        
        assert result is not None
        assert "S01" in result.name
        assert "DMT" in result.name
    
    def test_find_eeg_file_underscore_format(self, temp_eeg_dir):
        """Test finding EEG file with S01_DMT format."""
        from app.visualization.components.eeg_sync import find_eeg_file
        
        result = find_eeg_file(
            subject_id="S02",
            condition="EC",
            eeg_base_dir=temp_eeg_dir
        )
        
        assert result is not None
        assert "S02" in result.name
    
    def test_find_eeg_file_missing_returns_none(self, temp_eeg_dir):
        """Test that missing file returns None."""
        from app.visualization.components.eeg_sync import find_eeg_file
        
        result = find_eeg_file(
            subject_id="S99",  # Non-existent
            condition="DMT",
            eeg_base_dir=temp_eeg_dir
        )
        
        assert result is None
    
    def test_find_eeg_file_missing_condition_returns_none(self, temp_eeg_dir):
        """Test missing condition directory returns None."""
        from app.visualization.components.eeg_sync import find_eeg_file
        
        result = find_eeg_file(
            subject_id="S01",
            condition="UNKNOWN",  # Non-existent condition
            eeg_base_dir=temp_eeg_dir
        )
        
        assert result is None
    
    def test_find_eeg_file_invalid_base_dir(self):
        """Test invalid base directory returns None."""
        from app.visualization.components.eeg_sync import find_eeg_file
        
        result = find_eeg_file(
            subject_id="S01",
            condition="DMT",
            eeg_base_dir=Path("/nonexistent/path")
        )
        
        assert result is None
    
    def test_find_eeg_file_all_conditions(self, temp_eeg_dir):
        """Test finding files for all conditions."""
        from app.visualization.components.eeg_sync import find_eeg_file
        
        for condition in ["DMT", "EC", "EO"]:
            result = find_eeg_file(
                subject_id="S01",
                condition=condition,
                eeg_base_dir=temp_eeg_dir
            )
            assert result is not None, f"Failed for condition {condition}"


class TestExtractMetadataFromSample:
    """Tests for extracting EEG metadata from graph samples."""
    
    def test_extract_metadata_valid_sample(self, mock_graph_sample):
        """Test extracting metadata from valid sample."""
        from app.visualization.components.eeg_sync import extract_eeg_metadata
        
        metadata = extract_eeg_metadata(mock_graph_sample)
        
        assert metadata is not None
        assert metadata['subject_id'] == "S01"
        assert metadata['condition'] == "DMT"
        assert metadata['epoch_idx'] == 0
        assert metadata['band'] == "Alpha"
    
    def test_extract_metadata_missing_attributes(self):
        """Test handling sample without EEG metadata."""
        from app.visualization.components.eeg_sync import extract_eeg_metadata
        
        # Object without expected attributes
        class PlainSample:
            x = [1, 2, 3]
        
        metadata = extract_eeg_metadata(PlainSample())
        
        assert metadata is None
    
    def test_extract_metadata_partial_attributes(self):
        """Test handling sample with partial metadata."""
        from app.visualization.components.eeg_sync import extract_eeg_metadata
        
        class PartialSample:
            subject_id = "S01"
            # Missing: condition, epoch_idx
        
        metadata = extract_eeg_metadata(PartialSample())
        
        # Should return None because required fields are missing
        assert metadata is None
    
    def test_extract_metadata_different_epoch_indices(self, mock_graph_dataset):
        """Test extracting metadata from samples with different epoch indices."""
        from app.visualization.components.eeg_sync import extract_eeg_metadata
        
        for i, sample in enumerate(mock_graph_dataset):
            metadata = extract_eeg_metadata(sample)
            assert metadata is not None
            assert metadata['epoch_idx'] == i

