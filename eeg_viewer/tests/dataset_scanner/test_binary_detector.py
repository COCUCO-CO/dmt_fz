"""Tests for BinaryDatasetDetector (IDX/ubyte format)."""

import pytest
import struct
import gzip
from pathlib import Path
import numpy as np

from app.core.dataset_scanner.detectors.binary import BinaryDatasetDetector


def create_idx_file(path: Path, data: np.ndarray, compressed: bool = False):
    """
    Create an IDX format file (MNIST-style).
    
    IDX format:
    - magic[0:2] = 0x0000
    - magic[2] = dtype (0x08 = ubyte)
    - magic[3] = number of dimensions
    - Next 4*ndims bytes = dimension sizes (big-endian)
    - Data follows
    """
    dtype_map = {
        np.dtype('uint8'): 0x08,
        np.dtype('int8'): 0x09,
        np.dtype('int16'): 0x0B,
        np.dtype('int32'): 0x0C,
        np.dtype('float32'): 0x0D,
        np.dtype('float64'): 0x0E,
    }
    
    ndims = len(data.shape)
    dtype_code = dtype_map.get(data.dtype, 0x08)
    
    # Build header
    header = struct.pack('>BBBB', 0x00, 0x00, dtype_code, ndims)
    for dim in data.shape:
        header += struct.pack('>I', dim)
    
    # Write file
    content = header + data.tobytes()
    
    if compressed:
        with gzip.open(path, 'wb') as f:
            f.write(content)
    else:
        with open(path, 'wb') as f:
            f.write(content)


class TestBinaryDetectorBasics:
    """Test basic binary/IDX detection."""
    
    def test_detects_idx3_ubyte_files(self, tmp_path):
        """Test detection of idx3-ubyte (image) files."""
        # Create MNIST-like images
        images = np.random.randint(0, 255, (100, 28, 28), dtype=np.uint8)
        create_idx_file(tmp_path / 'train-images-idx3-ubyte', images)
        
        detector = BinaryDatasetDetector()
        result = detector.detect(tmp_path, analyze_samples=True)
        
        assert result.is_detected is True
        assert result.file_count >= 1
        assert result.confidence > 0.7
    
    def test_detects_idx1_ubyte_files(self, tmp_path):
        """Test detection of idx1-ubyte (label) files."""
        # Create MNIST-like labels
        labels = np.random.randint(0, 10, (100,), dtype=np.uint8)
        create_idx_file(tmp_path / 'train-labels-idx1-ubyte', labels)
        
        detector = BinaryDatasetDetector()
        result = detector.detect(tmp_path, analyze_samples=True)
        
        assert result.is_detected is True
    
    def test_detects_gzipped_idx_files(self, tmp_path):
        """Test detection of gzipped IDX files."""
        images = np.random.randint(0, 255, (50, 28, 28), dtype=np.uint8)
        create_idx_file(tmp_path / 'train-images-idx3-ubyte.gz', images, compressed=True)
        
        detector = BinaryDatasetDetector()
        result = detector.detect(tmp_path, analyze_samples=True)
        
        assert result.is_detected is True
        assert any('.gz' in str(f) for f in result.sample_files)
    
    def test_returns_false_for_non_idx(self, tmp_path):
        """Test that non-IDX files are not detected."""
        # Create a regular text file
        (tmp_path / 'readme.txt').write_text("This is not IDX format")
        (tmp_path / 'data.csv').write_text("a,b,c\n1,2,3")
        
        detector = BinaryDatasetDetector()
        result = detector.detect(tmp_path)
        
        assert result.is_detected is False


class TestBinaryDetectorMNIST:
    """Test MNIST-specific detection patterns."""
    
    def test_detects_train_test_splits(self, tmp_path):
        """Test detection of train/test splits in MNIST-style dataset."""
        # Train data
        train_images = np.random.randint(0, 255, (60000, 28, 28), dtype=np.uint8)
        train_labels = np.random.randint(0, 10, (60000,), dtype=np.uint8)
        create_idx_file(tmp_path / 'train-images-idx3-ubyte', train_images[:100])
        create_idx_file(tmp_path / 'train-labels-idx1-ubyte', train_labels[:100])
        
        # Test data
        test_images = np.random.randint(0, 255, (10000, 28, 28), dtype=np.uint8)
        test_labels = np.random.randint(0, 10, (10000,), dtype=np.uint8)
        create_idx_file(tmp_path / 't10k-images-idx3-ubyte', test_images[:100])
        create_idx_file(tmp_path / 't10k-labels-idx1-ubyte', test_labels[:100])
        
        detector = BinaryDatasetDetector()
        result = detector.detect(tmp_path, analyze_samples=True)
        
        assert result.is_detected is True
        assert result.has_splits is True
        assert 'train' in result.splits_found
        assert 'test' in result.splits_found
    
    def test_extracts_image_dimensions(self, tmp_path):
        """Test extraction of image dimensions from IDX files."""
        # Create 32x32 images (like CIFAR but in IDX format)
        images = np.random.randint(0, 255, (100, 32, 32), dtype=np.uint8)
        create_idx_file(tmp_path / 'images-idx3-ubyte', images)
        
        detector = BinaryDatasetDetector()
        result = detector.detect(tmp_path, analyze_samples=True)
        
        assert result.is_detected is True
        assert result.image_info is not None
        assert (32, 32) in result.image_info.image_sizes
    
    def test_suggests_torchvision_loader(self, tmp_path):
        """Test that torchvision.datasets.MNIST is suggested."""
        images = np.random.randint(0, 255, (100, 28, 28), dtype=np.uint8)
        create_idx_file(tmp_path / 'train-images-idx3-ubyte', images)
        
        detector = BinaryDatasetDetector()
        result = detector.detect(tmp_path, analyze_samples=True)
        
        assert any('MNIST' in s for s in result.suggestions)


class TestBinaryDetectorEdgeCases:
    """Test edge cases for binary detector."""
    
    def test_handles_corrupt_file(self, tmp_path):
        """Test handling of corrupt IDX file."""
        # Create corrupt file with invalid header
        corrupt_file = tmp_path / 'corrupt-idx3-ubyte'
        corrupt_file.write_bytes(b'\x00\x00\xFF\xFF' + b'\x00' * 100)
        
        detector = BinaryDatasetDetector()
        result = detector.detect(tmp_path, analyze_samples=True)
        
        # Should still detect but with lower confidence
        assert result.is_detected is True
        assert result.confidence < 0.95  # Lower than valid IDX
    
    def test_handles_empty_directory(self, tmp_path):
        """Test handling of empty directory."""
        detector = BinaryDatasetDetector()
        result = detector.detect(tmp_path)
        
        assert result.is_detected is False
        assert result.file_count == 0
    
    def test_handles_mixed_file_types(self, tmp_path):
        """Test handling of IDX files mixed with other files."""
        # Create IDX file
        images = np.random.randint(0, 255, (100, 28, 28), dtype=np.uint8)
        create_idx_file(tmp_path / 'images-idx3-ubyte', images)
        
        # Create other files
        (tmp_path / 'readme.txt').write_text("README")
        (tmp_path / 'data.json').write_text('{"key": "value"}')
        
        detector = BinaryDatasetDetector()
        result = detector.detect(tmp_path, analyze_samples=True)
        
        assert result.is_detected is True
        # Should only count IDX files
        assert result.file_count == 1


class TestBinaryDetectorVariants:
    """Test various IDX format variants."""
    
    def test_detects_float_idx(self, tmp_path):
        """Test detection of float32 IDX files with ubyte suffix."""
        data = np.random.rand(100, 10).astype(np.float32)
        # Note: IDX files typically have -ubyte suffix even for float data
        create_idx_file(tmp_path / 'features-idx2-ubyte', data)
        
        detector = BinaryDatasetDetector()
        result = detector.detect(tmp_path, analyze_samples=True)
        
        # Should detect (has ubyte pattern in name)
        assert result.is_detected is True
    
    def test_detects_2d_idx(self, tmp_path):
        """Test detection of 2D IDX files (like feature vectors)."""
        vectors = np.random.randint(0, 255, (1000, 64), dtype=np.uint8)
        create_idx_file(tmp_path / 'features-idx2-ubyte', vectors)
        
        detector = BinaryDatasetDetector()
        result = detector.detect(tmp_path, analyze_samples=True)
        
        assert result.is_detected is True
    
    def test_nested_directory_structure(self, tmp_path):
        """Test detection with nested directory structure."""
        # Create nested structure
        train_dir = tmp_path / 'train'
        test_dir = tmp_path / 'test'
        train_dir.mkdir()
        test_dir.mkdir()
        
        images = np.random.randint(0, 255, (100, 28, 28), dtype=np.uint8)
        create_idx_file(train_dir / 'images-idx3-ubyte', images)
        create_idx_file(test_dir / 'images-idx3-ubyte', images)
        
        detector = BinaryDatasetDetector()
        result = detector.detect(tmp_path, analyze_samples=True)
        
        assert result.is_detected is True
        assert result.file_count >= 2

