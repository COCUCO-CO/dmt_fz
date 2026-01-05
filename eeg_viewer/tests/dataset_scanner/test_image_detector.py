"""
Tests for ImageDetector.

Verifies detection of image datasets in various structures and formats.
"""

from pathlib import Path


class TestImageDetectorBasic:
    """Basic image detection tests."""
    
    def test_detects_png_files(self, flat_image_dataset: Path):
        """Verify PNG files are detected as images."""
        from app.core.dataset_scanner.detectors import ImageDetector
        
        detector = ImageDetector()
        result = detector.detect(flat_image_dataset)
        
        assert result.is_detected is True
        assert result.confidence > 0.8
        assert '.png' in result.extensions_found
    
    def test_detects_jpg_files(self, flat_image_dataset: Path):
        """Verify JPG files are detected as images."""
        from app.core.dataset_scanner.detectors import ImageDetector
        
        detector = ImageDetector()
        result = detector.detect(flat_image_dataset)
        
        assert result.is_detected is True
        assert '.jpg' in result.extensions_found or '.jpeg' in result.extensions_found
    
    def test_counts_files_correctly(self, flat_image_dataset: Path):
        """Verify file count is accurate."""
        from app.core.dataset_scanner.detectors import ImageDetector
        
        detector = ImageDetector()
        result = detector.detect(flat_image_dataset)
        
        # Created 10 PNG + 5 JPG = 15 total
        assert result.file_count == 15
    
    def test_returns_sample_files(self, flat_image_dataset: Path):
        """Verify sample files are returned."""
        from app.core.dataset_scanner.detectors import ImageDetector
        
        detector = ImageDetector()
        result = detector.detect(flat_image_dataset)
        
        assert len(result.sample_files) > 0
        assert all(isinstance(f, Path) for f in result.sample_files)
    
    def test_empty_directory_not_detected(self, empty_dataset: Path):
        """Verify empty directory is not detected as image dataset."""
        from app.core.dataset_scanner.detectors import ImageDetector
        
        detector = ImageDetector()
        result = detector.detect(empty_dataset)
        
        assert result.is_detected is False
        assert result.file_count == 0


class TestImageDetectorStructures:
    """Test image detection with various directory structures."""
    
    def test_detects_train_test_split(self, train_test_image_dataset: Path):
        """Verify train/test split is detected."""
        from app.core.dataset_scanner.detectors import ImageDetector
        
        detector = ImageDetector()
        result = detector.detect(train_test_image_dataset)
        
        assert result.is_detected is True
        assert result.has_splits is True
        assert 'train' in result.splits_found
        assert 'test' in result.splits_found
    
    def test_detects_train_val_test_split(self, train_val_test_image_dataset: Path):
        """Verify train/val/test split is detected."""
        from app.core.dataset_scanner.detectors import ImageDetector
        
        detector = ImageDetector()
        result = detector.detect(train_val_test_image_dataset)
        
        assert result.is_detected is True
        assert result.has_splits is True
        assert 'train' in result.splits_found
        assert 'val' in result.splits_found or 'validation' in result.splits_found
        assert 'test' in result.splits_found
    
    def test_detects_class_folders(self, by_class_image_dataset: Path):
        """Verify class-based folder structure is detected."""
        from app.core.dataset_scanner.detectors import ImageDetector
        
        detector = ImageDetector()
        result = detector.detect(by_class_image_dataset)
        
        assert result.is_detected is True
        assert result.has_classes is True
        assert set(result.classes_found) == {'cat', 'dog', 'bird'}
    
    def test_detects_imagenet_style(self, imagenet_style_dataset: Path):
        """Verify ImageNet-style structure is detected."""
        from app.core.dataset_scanner.detectors import ImageDetector
        
        detector = ImageDetector()
        result = detector.detect(imagenet_style_dataset)
        
        assert result.is_detected is True
        assert result.has_splits is True
        assert result.has_classes is True
        # Should detect the class IDs
        assert len(result.classes_found) == 3
    
    def test_counts_per_split(self, train_val_test_image_dataset: Path):
        """Verify correct count per split."""
        from app.core.dataset_scanner.detectors import ImageDetector
        
        detector = ImageDetector()
        result = detector.detect(train_val_test_image_dataset)
        
        # Created: train=50, val=15, test=15
        assert result.split_counts['train'] == 50
        assert result.split_counts['val'] == 15 or result.split_counts.get('validation') == 15
        assert result.split_counts['test'] == 15
    
    def test_counts_per_class(self, by_class_image_dataset: Path):
        """Verify correct count per class."""
        from app.core.dataset_scanner.detectors import ImageDetector
        
        detector = ImageDetector()
        result = detector.detect(by_class_image_dataset)
        
        # Created 15 images per class
        assert result.class_counts['cat'] == 15
        assert result.class_counts['dog'] == 15
        assert result.class_counts['bird'] == 15


class TestImageDetectorMetadata:
    """Test image metadata extraction."""
    
    def test_extracts_image_dimensions(self, flat_image_dataset: Path):
        """Verify image dimensions are extracted."""
        from app.core.dataset_scanner.detectors import ImageDetector
        
        detector = ImageDetector()
        result = detector.detect(flat_image_dataset, analyze_samples=True)
        
        assert result.image_info is not None
        assert len(result.image_info.sizes_found) > 0
        # All images created are 64x64
        assert (64, 64) in result.image_info.sizes_found
    
    def test_detects_color_mode(self, flat_image_dataset: Path):
        """Verify color mode is detected."""
        from app.core.dataset_scanner.detectors import ImageDetector
        
        detector = ImageDetector()
        result = detector.detect(flat_image_dataset, analyze_samples=True)
        
        assert result.image_info is not None
        assert result.image_info.color_mode in ['RGB', 'RGBA', 'L', 'grayscale']
    
    def test_detects_channels(self, flat_image_dataset: Path):
        """Verify number of channels is detected."""
        from app.core.dataset_scanner.detectors import ImageDetector
        
        detector = ImageDetector()
        result = detector.detect(flat_image_dataset, analyze_samples=True)
        
        assert result.image_info is not None
        assert result.image_info.channels in [1, 3, 4]


class TestImageDetectorEdgeCases:
    """Test edge cases for image detection."""
    
    def test_handles_hidden_files(self, dataset_with_hidden_files: Path):
        """Verify hidden files are ignored."""
        from app.core.dataset_scanner.detectors import ImageDetector
        
        detector = ImageDetector()
        result = detector.detect(dataset_with_hidden_files)
        
        # Should only count the 5 visible images, not hidden ones
        assert result.file_count == 5
    
    def test_handles_corrupt_files(self, dataset_with_corrupt_files: Path):
        """Verify corrupt files are handled gracefully."""
        from app.core.dataset_scanner.detectors import ImageDetector
        
        detector = ImageDetector()
        result = detector.detect(dataset_with_corrupt_files, analyze_samples=True)
        
        # Should detect valid images
        assert result.is_detected is True
        # Should report corrupt files in warnings or separate count
        assert result.corrupt_files_count > 0 or len(result.warnings) > 0
    
    def test_handles_deeply_nested(self, deeply_nested_dataset: Path):
        """Verify deeply nested structures are handled."""
        from app.core.dataset_scanner.detectors import ImageDetector
        
        detector = ImageDetector()
        result = detector.detect(deeply_nested_dataset)
        
        assert result.is_detected is True
        assert result.file_count > 0
        assert result.hierarchy_depth >= 4  # data/category/subcategory/split
    
    def test_single_file_detected(self, temp_dir: Path):
        """Verify single image file is detected."""
        from app.core.dataset_scanner.detectors import ImageDetector
        from tests.dataset_scanner.conftest import create_minimal_png
        
        dataset_path = temp_dir / "single_image"
        dataset_path.mkdir()
        create_minimal_png(dataset_path / "only_image.png")
        
        detector = ImageDetector()
        result = detector.detect(dataset_path)
        
        assert result.is_detected is True
        assert result.file_count == 1
    
    def test_mixed_extensions(self, temp_dir: Path):
        """Verify mixed image extensions are all detected."""
        from app.core.dataset_scanner.detectors import ImageDetector
        from tests.dataset_scanner.conftest import create_minimal_png, create_minimal_jpg
        
        dataset_path = temp_dir / "mixed_ext"
        dataset_path.mkdir()
        
        create_minimal_png(dataset_path / "image1.png")
        create_minimal_png(dataset_path / "image2.PNG")  # Uppercase
        create_minimal_jpg(dataset_path / "image3.jpg")
        create_minimal_jpg(dataset_path / "image4.jpeg")
        
        detector = ImageDetector()
        result = detector.detect(dataset_path)
        
        assert result.file_count == 4
        # Should handle case-insensitive extensions


class TestImageDetectorStatistics:
    """Test statistical analysis of image datasets."""
    
    def test_calculates_total_size(self, flat_image_dataset: Path):
        """Verify total dataset size is calculated."""
        from app.core.dataset_scanner.detectors import ImageDetector
        
        detector = ImageDetector()
        result = detector.detect(flat_image_dataset)
        
        assert result.total_size_bytes > 0
    
    def test_size_distribution(self, imagenet_style_dataset: Path):
        """Verify size distribution is analyzed."""
        from app.core.dataset_scanner.detectors import ImageDetector
        
        detector = ImageDetector()
        result = detector.detect(imagenet_style_dataset, analyze_samples=True)
        
        if result.image_info:
            assert result.image_info.size_distribution is not None or len(result.image_info.sizes_found) > 0
    
    def test_class_balance_analysis(self, by_class_image_dataset: Path):
        """Verify class balance is analyzed."""
        from app.core.dataset_scanner.detectors import ImageDetector
        
        detector = ImageDetector()
        result = detector.detect(by_class_image_dataset)
        
        # All classes have same count (15 each), should be balanced
        assert result.is_balanced is True or all(
            count == 15 for count in result.class_counts.values()
        )
    
    def test_detects_imbalanced_classes(self, temp_dir: Path):
        """Verify imbalanced classes are detected."""
        from app.core.dataset_scanner.detectors import ImageDetector
        from tests.dataset_scanner.conftest import create_minimal_png
        
        dataset_path = temp_dir / "imbalanced"
        
        # Create imbalanced dataset: 100 in class_a, 10 in class_b
        for i in range(100):
            create_minimal_png(dataset_path / "class_a" / f"img_{i:03d}.png")
        for i in range(10):
            create_minimal_png(dataset_path / "class_b" / f"img_{i:03d}.png")
        
        detector = ImageDetector()
        result = detector.detect(dataset_path)
        
        assert result.is_balanced is False or 'imbalance' in str(result.warnings).lower()


class TestImageDetectorOutput:
    """Test output format and completeness."""
    
    def test_returns_detection_result(self, flat_image_dataset: Path):
        """Verify detection result has all required fields."""
        from app.core.dataset_scanner.detectors import ImageDetector
        
        detector = ImageDetector()
        result = detector.detect(flat_image_dataset)
        
        # Check all required fields exist
        assert hasattr(result, 'is_detected')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'file_count')
        assert hasattr(result, 'extensions_found')
        assert hasattr(result, 'sample_files')
        assert hasattr(result, 'total_size_bytes')
    
    def test_to_dict_conversion(self, flat_image_dataset: Path):
        """Verify result can be converted to dict."""
        from app.core.dataset_scanner.detectors import ImageDetector
        
        detector = ImageDetector()
        result = detector.detect(flat_image_dataset)
        
        d = result.to_dict()
        
        assert isinstance(d, dict)
        assert 'is_detected' in d
        assert 'file_count' in d
    
    def test_generates_suggestions(self, by_class_image_dataset: Path):
        """Verify detector generates useful suggestions."""
        from app.core.dataset_scanner.detectors import ImageDetector
        
        detector = ImageDetector()
        result = detector.detect(by_class_image_dataset)
        
        # Should suggest ImageFolder loader for class-based dataset
        assert len(result.suggestions) > 0 or result.suggested_loader is not None








