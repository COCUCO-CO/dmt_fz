"""
Tests for dataset metadata extraction from input files.

Verifies that when scanning input directory, we can extract
real metadata like sample rate, epoch duration, channel count, etc.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.state import PS

# Path to real test data
REAL_EEG_PATH = Path(__file__).parent.parent.parent.parent / "EEG_CLEAN" / "DMT" / "S01-DMT_ICA_pruned.set"


class TestDatasetMetadataExtraction:
    """Test extraction of metadata from .set files."""
    
    @pytest.mark.skipif(not REAL_EEG_PATH.exists(), reason="Real EEG data not available")
    def test_can_extract_sfreq_from_set_file(self):
        """Should extract sample frequency from .set file."""
        from app.pages.pipeline.utils import extract_dataset_metadata
        
        metadata = extract_dataset_metadata(REAL_EEG_PATH)
        
        assert 'sfreq' in metadata
        assert isinstance(metadata['sfreq'], (int, float))
        assert metadata['sfreq'] > 0
    
    @pytest.mark.skipif(not REAL_EEG_PATH.exists(), reason="Real EEG data not available")
    def test_can_extract_n_channels_from_set_file(self):
        """Should extract number of channels from .set file."""
        from app.pages.pipeline.utils import extract_dataset_metadata
        
        metadata = extract_dataset_metadata(REAL_EEG_PATH)
        
        assert 'n_channels' in metadata
        assert isinstance(metadata['n_channels'], int)
        assert metadata['n_channels'] > 0
    
    @pytest.mark.skipif(not REAL_EEG_PATH.exists(), reason="Real EEG data not available")
    def test_can_extract_n_epochs_from_set_file(self):
        """Should extract number of epochs from .set file."""
        from app.pages.pipeline.utils import extract_dataset_metadata
        
        metadata = extract_dataset_metadata(REAL_EEG_PATH)
        
        assert 'n_epochs' in metadata
        assert isinstance(metadata['n_epochs'], int)
        assert metadata['n_epochs'] > 0
    
    @pytest.mark.skipif(not REAL_EEG_PATH.exists(), reason="Real EEG data not available")
    def test_can_extract_epoch_duration_from_set_file(self):
        """Should extract epoch duration in seconds from .set file."""
        from app.pages.pipeline.utils import extract_dataset_metadata
        
        metadata = extract_dataset_metadata(REAL_EEG_PATH)
        
        assert 'epoch_duration' in metadata
        assert isinstance(metadata['epoch_duration'], (int, float))
        assert metadata['epoch_duration'] > 0
    
    @pytest.mark.skipif(not REAL_EEG_PATH.exists(), reason="Real EEG data not available")
    def test_can_extract_n_times_from_set_file(self):
        """Should extract samples per epoch from .set file."""
        from app.pages.pipeline.utils import extract_dataset_metadata
        
        metadata = extract_dataset_metadata(REAL_EEG_PATH)
        
        assert 'n_times' in metadata
        assert isinstance(metadata['n_times'], int)
        assert metadata['n_times'] > 0
    
    @pytest.mark.skipif(not REAL_EEG_PATH.exists(), reason="Real EEG data not available")
    def test_metadata_is_consistent(self):
        """n_times should equal sfreq * epoch_duration."""
        from app.pages.pipeline.utils import extract_dataset_metadata
        
        metadata = extract_dataset_metadata(REAL_EEG_PATH)
        
        expected_n_times = int(metadata['sfreq'] * metadata['epoch_duration'])
        # Allow small tolerance for rounding
        assert abs(metadata['n_times'] - expected_n_times) <= 1


class TestDatasetMetadataInPS:
    """Test that metadata is stored in PS."""
    
    def test_ps_has_dataset_metadata_attr(self):
        """PS should have dataset_metadata attribute."""
        assert hasattr(PS, 'dataset_metadata')
    
    def test_dataset_metadata_defaults_to_none_or_empty(self):
        """dataset_metadata should default to None or empty dict."""
        from app.state.global_state import PipelineState
        ps = PipelineState()
        assert ps.dataset_metadata is None or ps.dataset_metadata == {}
    
    @pytest.mark.skipif(not REAL_EEG_PATH.exists(), reason="Real EEG data not available")
    def test_metadata_can_be_stored_in_ps(self):
        """Should be able to store extracted metadata in PS."""
        from app.pages.pipeline.utils import extract_dataset_metadata
        
        original = PS.dataset_metadata
        try:
            metadata = extract_dataset_metadata(REAL_EEG_PATH)
            PS.dataset_metadata = metadata
            
            assert PS.dataset_metadata is not None
            assert 'sfreq' in PS.dataset_metadata
        finally:
            PS.dataset_metadata = original


class TestScanInputUpdatesMetadata:
    """Test that scanning input directory extracts metadata."""
    
    @pytest.mark.skipif(not REAL_EEG_PATH.exists(), reason="Real EEG data not available")
    def test_scan_extracts_metadata_from_first_file(self):
        """Scanning should extract metadata from first .set file found."""
        from app.pages.pipeline.utils import scan_input_directory
        
        input_dir = REAL_EEG_PATH.parent.parent  # EEG_CLEAN
        result = scan_input_directory(input_dir)
        
        assert 'metadata' in result
        assert result['metadata'] is not None
        assert 'sfreq' in result['metadata']


class TestMetadataFormatting:
    """Test formatting of metadata for display."""
    
    def test_format_sfreq(self):
        """Sample rate should format nicely."""
        from app.pages.pipeline.utils import format_dataset_metadata
        
        metadata = {'sfreq': 500.0, 'n_channels': 24, 'n_epochs': 150, 
                   'epoch_duration': 2.0, 'n_times': 1000}
        formatted = format_dataset_metadata(metadata)
        
        assert '500' in formatted or '500.0' in formatted
    
    def test_format_epoch_duration(self):
        """Epoch duration should show seconds."""
        from app.pages.pipeline.utils import format_dataset_metadata
        
        metadata = {'sfreq': 500.0, 'n_channels': 24, 'n_epochs': 150, 
                   'epoch_duration': 2.0, 'n_times': 1000}
        formatted = format_dataset_metadata(metadata)
        
        assert '2' in formatted  # 2 seconds
    
    def test_format_returns_string(self):
        """format_dataset_metadata should return a string."""
        from app.pages.pipeline.utils import format_dataset_metadata
        
        metadata = {'sfreq': 500.0, 'n_channels': 24}
        formatted = format_dataset_metadata(metadata)
        
        assert isinstance(formatted, str)
    
    def test_format_handles_none(self):
        """Should handle None metadata gracefully."""
        from app.pages.pipeline.utils import format_dataset_metadata
        
        formatted = format_dataset_metadata(None)
        assert isinstance(formatted, str)
        assert 'No' in formatted or 'no' in formatted or 'disponible' in formatted.lower()


