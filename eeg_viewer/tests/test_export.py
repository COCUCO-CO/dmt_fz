"""
Tests for cleaning/export.py - Export functionality.
"""

import json
from pathlib import Path
import tempfile
import numpy as np

from cleaning.export import (
    ExportFormat, export_cleaned_eeg, export_epochs,
    export_preprocessing_log, create_export_bundle
)
from cleaning.epochs import create_epochs


class TestExportFormat:
    """Tests for ExportFormat enum."""
    
    def test_all_formats_exist(self):
        """Verify all formats exist."""
        formats = list(ExportFormat)
        assert ExportFormat.FIF in formats
        assert ExportFormat.SET in formats
        assert ExportFormat.EDF in formats
        assert ExportFormat.NPY in formats
        assert ExportFormat.CSV in formats
    
    def test_extensions(self):
        """Test file extensions are correct."""
        assert ExportFormat.FIF.extension == ".fif"
        assert ExportFormat.SET.extension == ".set"
        assert ExportFormat.EDF.extension == ".edf"
        assert ExportFormat.NPY.extension == ".npy"
    
    def test_display_names(self):
        """Test display names are set."""
        assert "FIF" in ExportFormat.FIF.display_name
        assert "EEGLAB" in ExportFormat.SET.display_name


class TestExportCleanedEEG:
    """Tests for export_cleaned_eeg function."""
    
    def test_export_fif(self, synthetic_raw):
        """Test exporting to FIF format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.fif"
            
            result_path = export_cleaned_eeg(
                synthetic_raw, 
                output_path, 
                format=ExportFormat.FIF,
                overwrite=True
            )
            
            assert result_path.exists()
            assert result_path.suffix == ".fif"
    
    def test_export_adds_extension(self, synthetic_raw):
        """Test that extension is added if missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test"  # No extension
            
            result_path = export_cleaned_eeg(
                synthetic_raw,
                output_path,
                format=ExportFormat.FIF,
                overwrite=True
            )
            
            assert result_path.suffix == ".fif"
    
    def test_export_creates_directory(self, synthetic_raw):
        """Test that output directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "test.fif"
            
            result_path = export_cleaned_eeg(
                synthetic_raw,
                output_path,
                format=ExportFormat.FIF,
                overwrite=True
            )
            
            assert result_path.exists()
            assert result_path.parent.exists()


class TestExportEpochs:
    """Tests for export_epochs function."""
    
    def test_export_epochs_fif(self, synthetic_raw):
        """Test exporting epochs to FIF."""
        epoch_result = create_epochs(synthetic_raw, duration=1.0)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = export_epochs(
                epoch_result.epochs,
                Path(tmpdir),
                "test_epochs",
                formats=[ExportFormat.FIF]
            )
            
            assert 'fif' in outputs
            assert outputs['fif'].exists()
    
    def test_export_epochs_npy(self, synthetic_raw):
        """Test exporting epochs to NPY."""
        epoch_result = create_epochs(synthetic_raw, duration=1.0)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = export_epochs(
                epoch_result.epochs,
                Path(tmpdir),
                "test_epochs",
                formats=[ExportFormat.NPY]
            )
            
            assert 'npy' in outputs
            assert outputs['npy'].exists()
            
            # Verify NPY content
            data = np.load(outputs['npy'])
            assert data.shape[0] == len(epoch_result.epochs)
    
    def test_export_epochs_metadata(self, synthetic_raw):
        """Test that metadata CSV is created."""
        epoch_result = create_epochs(synthetic_raw, duration=1.0)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = export_epochs(
                epoch_result.epochs,
                Path(tmpdir),
                "test_epochs",
                include_metadata=True
            )
            
            assert 'metadata' in outputs
            assert outputs['metadata'].exists()


class TestExportPreprocessingLog:
    """Tests for export_preprocessing_log function."""
    
    def test_export_log(self, loaded_state):
        """Test exporting preprocessing log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "log.json"
            
            result_path = export_preprocessing_log(loaded_state, output_path)
            
            assert result_path.exists()
    
    def test_log_content(self, loaded_state):
        """Test log content is valid JSON."""
        loaded_state.bad_channels = ['Fp1']
        loaded_state.reference_type = "Average"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "log.json"
            export_preprocessing_log(loaded_state, output_path)
            
            with open(output_path) as f:
                log = json.load(f)
            
            assert 'filename' in log
            assert 'bad_channels' in log
            assert 'export_timestamp' in log
            assert log['bad_channels'] == ['Fp1']


class TestCreateExportBundle:
    """Tests for create_export_bundle function."""
    
    def test_bundle_with_raw_only(self, loaded_state):
        """Test creating bundle with raw data only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = create_export_bundle(
                loaded_state,
                Path(tmpdir),
                include_raw=True,
                include_epochs=False
            )
            
            assert 'raw' in outputs
            assert 'log' in outputs
            assert outputs['raw'].exists()
            assert outputs['log'].exists()
    
    def test_bundle_with_epochs(self, loaded_state):
        """Test creating bundle with epochs."""
        # Create epochs first
        epoch_result = create_epochs(loaded_state.raw, duration=1.0)
        loaded_state.epochs = epoch_result.epochs
        
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = create_export_bundle(
                loaded_state,
                Path(tmpdir),
                include_raw=True,
                include_epochs=True
            )
            
            assert 'raw' in outputs
            # Check for epoch files
            epoch_files = [k for k in outputs if 'epochs' in k]
            assert len(epoch_files) > 0
    
    def test_bundle_channel_info(self, loaded_state):
        """Test that channel info file is created."""
        loaded_state.bad_channels = ['Fp1']
        
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = create_export_bundle(
                loaded_state,
                Path(tmpdir)
            )
            
            assert 'channels' in outputs
            assert outputs['channels'].exists()
            
            # Verify content mentions bad channel
            content = outputs['channels'].read_text()
            assert 'Fp1' in content
    
    def test_bundle_creates_timestamped_files(self, loaded_state):
        """Test that files have timestamp in name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = create_export_bundle(
                loaded_state,
                Path(tmpdir)
            )
            
            # Check that files contain timestamp pattern
            for path in outputs.values():
                # Files should contain date pattern like 20231207
                name = path.stem
                assert any(c.isdigit() for c in name)






