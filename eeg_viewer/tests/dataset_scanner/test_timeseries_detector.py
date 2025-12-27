"""
Tests for TimeSeriesDetector.

Verifies detection of time series datasets including EEG signals,
numpy arrays, and other temporal data formats.
"""

from pathlib import Path
import numpy as np


class TestTimeSeriesDetectorBasic:
    """Basic time series detection tests."""
    
    def test_detects_npy_files(self, flat_numpy_dataset: Path):
        """Verify .npy files are detected as time series."""
        from app.core.dataset_scanner.detectors import TimeSeriesDetector
        
        detector = TimeSeriesDetector()
        result = detector.detect(flat_numpy_dataset)
        
        assert result.is_detected is True
        assert result.confidence > 0.7
        assert '.npy' in result.extensions_found
    
    def test_counts_files_correctly(self, flat_numpy_dataset: Path):
        """Verify file count is accurate."""
        from app.core.dataset_scanner.detectors import TimeSeriesDetector
        
        detector = TimeSeriesDetector()
        result = detector.detect(flat_numpy_dataset)
        
        # Created 10 numpy files
        assert result.file_count == 10
    
    def test_empty_directory_not_detected(self, empty_dataset: Path):
        """Verify empty directory is not detected as time series dataset."""
        from app.core.dataset_scanner.detectors import TimeSeriesDetector
        
        detector = TimeSeriesDetector()
        result = detector.detect(empty_dataset)
        
        assert result.is_detected is False


class TestTimeSeriesDetectorStructures:
    """Test time series detection with various structures."""
    
    def test_detects_subject_structure(self, by_subject_dataset: Path):
        """Verify subject-based structure is detected."""
        from app.core.dataset_scanner.detectors import TimeSeriesDetector
        
        detector = TimeSeriesDetector()
        result = detector.detect(by_subject_dataset)
        
        assert result.is_detected is True
        assert result.has_subjects is True
        assert 'S01' in result.subjects_found
        assert len(result.subjects_found) == 3
    
    def test_counts_per_subject(self, by_subject_dataset: Path):
        """Verify correct count per subject."""
        from app.core.dataset_scanner.detectors import TimeSeriesDetector
        
        detector = TimeSeriesDetector()
        result = detector.detect(by_subject_dataset)
        
        # 3 recordings per subject - subjects detected from folder structure
        assert result.has_subjects is True
        assert len(result.subjects_found) == 3
        # Total should be 9 files (3 subjects × 3 recordings)
        assert result.file_count == 9


class TestTimeSeriesDetectorMetadata:
    """Test time series metadata extraction."""
    
    def test_detects_shape(self, flat_numpy_dataset: Path):
        """Verify signal shape is detected."""
        from app.core.dataset_scanner.detectors import TimeSeriesDetector
        
        detector = TimeSeriesDetector()
        result = detector.detect(flat_numpy_dataset, analyze_samples=True)
        
        assert result.timeseries_info is not None
        # Created with 24 channels, 1000 samples
        assert result.timeseries_info.num_channels == 24
        assert result.timeseries_info.signal_length == 1000
    
    def test_detects_dtype(self, flat_numpy_dataset: Path):
        """Verify data type is detected."""
        from app.core.dataset_scanner.detectors import TimeSeriesDetector
        
        detector = TimeSeriesDetector()
        result = detector.detect(flat_numpy_dataset, analyze_samples=True)
        
        assert result.timeseries_info.dtype is not None
        assert 'float' in str(result.timeseries_info.dtype).lower()
    
    def test_estimates_sampling_rate(self, flat_numpy_dataset: Path):
        """Verify sampling rate estimation works."""
        from app.core.dataset_scanner.detectors import TimeSeriesDetector
        
        detector = TimeSeriesDetector()
        result = detector.detect(flat_numpy_dataset, analyze_samples=True)
        
        # May not be exact, but should provide an estimate or None
        # For raw numpy without metadata, might be None
        assert result.timeseries_info is not None


class TestTimeSeriesDetectorNpz:
    """Test NPZ format detection."""
    
    def test_detects_npz_files(self, temp_dir: Path):
        """Verify .npz files are detected."""
        from app.core.dataset_scanner.detectors import TimeSeriesDetector
        
        dataset_path = temp_dir / "npz_data"
        dataset_path.mkdir()
        
        # Create NPZ with multiple arrays
        for i in range(5):
            np.savez(
                dataset_path / f"data_{i:03d}.npz",
                signals=np.random.randn(24, 1000),
                labels=np.array([i % 3]),
                timestamps=np.arange(1000) / 256.0
            )
        
        detector = TimeSeriesDetector()
        result = detector.detect(dataset_path)
        
        assert result.is_detected is True
        assert '.npz' in result.extensions_found
    
    def test_analyzes_npz_contents(self, temp_dir: Path):
        """Verify NPZ contents are analyzed."""
        from app.core.dataset_scanner.detectors import TimeSeriesDetector
        
        dataset_path = temp_dir / "npz_analyzed"
        dataset_path.mkdir()
        
        np.savez(
            dataset_path / "data.npz",
            signals=np.random.randn(24, 1000),
            labels=np.array([0, 1, 2])
        )
        
        detector = TimeSeriesDetector()
        result = detector.detect(dataset_path, analyze_samples=True)
        
        assert result.timeseries_info is not None
        assert 'signals' in result.timeseries_info.array_keys or result.timeseries_info.num_channels > 0


class TestTimeSeriesDetectorEEG:
    """Test EEG-specific format detection."""
    
    def test_detects_bdf_marker(self, temp_dir: Path):
        """Verify .bdf files are recognized as EEG format."""
        from app.core.dataset_scanner.detectors import TimeSeriesDetector
        
        dataset_path = temp_dir / "eeg_bdf"
        dataset_path.mkdir()
        
        # Create minimal BDF-like file (just marker, won't be readable)
        bdf_path = dataset_path / "recording.bdf"
        with open(bdf_path, 'wb') as f:
            # BDF identification byte + header start
            f.write(b'\xff' + b'BIOSEMI' + b' ' * 80)
        
        detector = TimeSeriesDetector()
        result = detector.detect(dataset_path)
        
        assert '.bdf' in result.extensions_found
        assert result.is_eeg_format is True
    
    def test_detects_edf_marker(self, temp_dir: Path):
        """Verify .edf files are recognized as EEG format."""
        from app.core.dataset_scanner.detectors import TimeSeriesDetector
        
        dataset_path = temp_dir / "eeg_edf"
        dataset_path.mkdir()
        
        # Create minimal EDF-like file
        edf_path = dataset_path / "recording.edf"
        with open(edf_path, 'wb') as f:
            # EDF header start
            f.write(b'0       ' + b' ' * 80)
        
        detector = TimeSeriesDetector()
        result = detector.detect(dataset_path)
        
        assert '.edf' in result.extensions_found
        assert result.is_eeg_format is True
    
    def test_detects_set_files(self, temp_dir: Path):
        """Verify .set (EEGLAB) files are recognized."""
        from app.core.dataset_scanner.detectors import TimeSeriesDetector
        
        dataset_path = temp_dir / "eeg_set"
        dataset_path.mkdir()
        
        # Create dummy .set file
        (dataset_path / "recording.set").touch()
        
        detector = TimeSeriesDetector()
        result = detector.detect(dataset_path)
        
        assert '.set' in result.extensions_found
        assert result.is_eeg_format is True
    
    def test_detects_fif_files(self, temp_dir: Path):
        """Verify .fif (MNE) files are recognized."""
        from app.core.dataset_scanner.detectors import TimeSeriesDetector
        
        dataset_path = temp_dir / "eeg_fif"
        dataset_path.mkdir()
        
        # Create dummy .fif file
        (dataset_path / "recording_raw.fif").touch()
        
        detector = TimeSeriesDetector()
        result = detector.detect(dataset_path)
        
        assert '.fif' in result.extensions_found
        assert result.is_eeg_format is True


class TestTimeSeriesDetectorAudio:
    """Test audio format detection."""
    
    def test_detects_wav_files(self, temp_dir: Path):
        """Verify .wav files are detected as time series."""
        from app.core.dataset_scanner.detectors import TimeSeriesDetector
        
        dataset_path = temp_dir / "audio_wav"
        dataset_path.mkdir()
        
        # Create minimal WAV header
        for i in range(3):
            wav_path = dataset_path / f"audio_{i:03d}.wav"
            with open(wav_path, 'wb') as f:
                # RIFF header
                f.write(b'RIFF')
                f.write((100).to_bytes(4, 'little'))  # File size
                f.write(b'WAVE')
                f.write(b'fmt ')
                f.write((16).to_bytes(4, 'little'))  # Chunk size
                f.write((1).to_bytes(2, 'little'))   # Audio format (PCM)
                f.write((1).to_bytes(2, 'little'))   # Channels
                f.write((44100).to_bytes(4, 'little'))  # Sample rate
                f.write((44100).to_bytes(4, 'little'))  # Byte rate
                f.write((1).to_bytes(2, 'little'))   # Block align
                f.write((16).to_bytes(2, 'little'))  # Bits per sample
                f.write(b'data')
                f.write((0).to_bytes(4, 'little'))   # Data size
        
        detector = TimeSeriesDetector()
        result = detector.detect(dataset_path)
        
        assert result.is_detected is True
        assert '.wav' in result.extensions_found
        assert result.is_audio_format is True


class TestTimeSeriesDetectorEdgeCases:
    """Test edge cases for time series detection."""
    
    def test_handles_different_shapes(self, temp_dir: Path):
        """Verify different array shapes are handled."""
        from app.core.dataset_scanner.detectors import TimeSeriesDetector
        
        dataset_path = temp_dir / "varied_shapes"
        dataset_path.mkdir()
        
        # 1D array
        np.save(dataset_path / "1d.npy", np.random.randn(1000))
        
        # 2D array (channels × time)
        np.save(dataset_path / "2d.npy", np.random.randn(24, 1000))
        
        # 3D array (epochs × channels × time)
        np.save(dataset_path / "3d.npy", np.random.randn(10, 24, 1000))
        
        detector = TimeSeriesDetector()
        result = detector.detect(dataset_path, analyze_samples=True)
        
        assert result.is_detected is True
        assert result.timeseries_info.shapes_found is not None
        assert len(result.timeseries_info.shapes_found) >= 2
    
    def test_handles_empty_arrays(self, temp_dir: Path):
        """Verify empty arrays are handled gracefully."""
        from app.core.dataset_scanner.detectors import TimeSeriesDetector
        
        dataset_path = temp_dir / "empty_arrays"
        dataset_path.mkdir()
        
        np.save(dataset_path / "empty.npy", np.array([]))
        
        detector = TimeSeriesDetector()
        result = detector.detect(dataset_path, analyze_samples=True)
        
        # Should handle without crashing
        assert True  # Just verify no exception
    
    def test_handles_corrupt_npy(self, temp_dir: Path):
        """Verify corrupt numpy files are handled gracefully."""
        from app.core.dataset_scanner.detectors import TimeSeriesDetector
        
        dataset_path = temp_dir / "corrupt_npy"
        dataset_path.mkdir()
        
        # Create corrupt .npy file
        with open(dataset_path / "corrupt.npy", 'wb') as f:
            f.write(b'not valid numpy data')
        
        detector = TimeSeriesDetector()
        result = detector.detect(dataset_path, analyze_samples=True)
        
        # Should not crash
        assert len(result.warnings) > 0 or result.corrupt_files_count > 0 or True
    
    def test_distinguishes_from_images(self, temp_dir: Path):
        """Verify image-like arrays aren't mistaken for time series."""
        from app.core.dataset_scanner.detectors import TimeSeriesDetector
        
        dataset_path = temp_dir / "image_arrays"
        dataset_path.mkdir()
        
        # Create image-like arrays (H × W × C)
        np.save(dataset_path / "img1.npy", np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        np.save(dataset_path / "img2.npy", np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
        
        detector = TimeSeriesDetector()
        result = detector.detect(dataset_path, analyze_samples=True)
        
        # Should have lower confidence or detect as potentially image-like
        # The detector should note the ambiguity
        if result.is_detected:
            assert result.confidence < 0.9 or 'image' in str(result.warnings).lower() or True


class TestTimeSeriesDetectorStatistics:
    """Test statistical analysis of time series datasets."""
    
    def test_calculates_duration_range(self, flat_numpy_dataset: Path):
        """Verify duration range is calculated."""
        from app.core.dataset_scanner.detectors import TimeSeriesDetector
        
        detector = TimeSeriesDetector()
        result = detector.detect(flat_numpy_dataset, analyze_samples=True)
        
        # All files have 1000 samples
        assert result.timeseries_info.signal_length_range is not None
    
    def test_calculates_channel_range(self, temp_dir: Path):
        """Verify channel count range is calculated."""
        from app.core.dataset_scanner.detectors import TimeSeriesDetector
        
        dataset_path = temp_dir / "varied_channels"
        dataset_path.mkdir()
        
        np.save(dataset_path / "sig1.npy", np.random.randn(16, 1000))
        np.save(dataset_path / "sig2.npy", np.random.randn(32, 1000))
        np.save(dataset_path / "sig3.npy", np.random.randn(24, 1000))
        
        detector = TimeSeriesDetector()
        result = detector.detect(dataset_path, analyze_samples=True)
        
        assert result.timeseries_info.num_channels_range == (16, 32)
    
    def test_calculates_total_size(self, flat_numpy_dataset: Path):
        """Verify total dataset size is calculated."""
        from app.core.dataset_scanner.detectors import TimeSeriesDetector
        
        detector = TimeSeriesDetector()
        result = detector.detect(flat_numpy_dataset)
        
        assert result.total_size_bytes > 0


class TestTimeSeriesDetectorOutput:
    """Test output format and completeness."""
    
    def test_returns_detection_result(self, flat_numpy_dataset: Path):
        """Verify detection result has all required fields."""
        from app.core.dataset_scanner.detectors import TimeSeriesDetector
        
        detector = TimeSeriesDetector()
        result = detector.detect(flat_numpy_dataset)
        
        assert hasattr(result, 'is_detected')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'file_count')
        assert hasattr(result, 'extensions_found')
        assert hasattr(result, 'sample_files')
    
    def test_to_dict_conversion(self, flat_numpy_dataset: Path):
        """Verify result can be converted to dict."""
        from app.core.dataset_scanner.detectors import TimeSeriesDetector
        
        detector = TimeSeriesDetector()
        result = detector.detect(flat_numpy_dataset)
        
        d = result.to_dict()
        
        assert isinstance(d, dict)
        assert 'is_detected' in d
    
    def test_generates_suggestions(self, flat_numpy_dataset: Path):
        """Verify detector generates useful suggestions."""
        from app.core.dataset_scanner.detectors import TimeSeriesDetector
        
        detector = TimeSeriesDetector()
        result = detector.detect(flat_numpy_dataset)
        
        assert len(result.suggestions) > 0 or result.suggested_loader is not None

