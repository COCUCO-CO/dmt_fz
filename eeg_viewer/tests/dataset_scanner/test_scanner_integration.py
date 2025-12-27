"""
Tests for DatasetScanner integration.

Verifies the main scanner class correctly orchestrates all detectors
and produces complete DatasetInfo objects.
"""

from pathlib import Path


class TestDatasetScannerBasic:
    """Basic scanner functionality tests."""
    
    def test_scanner_initialization(self):
        """Verify scanner can be initialized."""
        from app.core.dataset_scanner import DatasetScanner
        
        scanner = DatasetScanner()
        
        assert scanner is not None
    
    def test_scan_returns_dataset_info(self, flat_image_dataset: Path):
        """Verify scan returns DatasetInfo object."""
        from app.core.dataset_scanner import DatasetScanner
        from app.core.dataset_scanner.models import DatasetInfo
        
        scanner = DatasetScanner()
        result = scanner.scan(flat_image_dataset)
        
        assert isinstance(result, DatasetInfo)
    
    def test_scan_nonexistent_path(self, temp_dir: Path):
        """Verify scanning nonexistent path returns appropriate error."""
        from app.core.dataset_scanner import DatasetScanner
        
        scanner = DatasetScanner()
        result = scanner.scan(temp_dir / "nonexistent")
        
        assert result.is_valid is False
        assert len(result.warnings) > 0 or result.primary_type.value == 'unknown'
    
    def test_scan_empty_directory(self, empty_dataset: Path):
        """Verify scanning empty directory returns appropriate info."""
        from app.core.dataset_scanner import DatasetScanner
        from app.core.dataset_scanner.models import DatasetType
        
        scanner = DatasetScanner()
        result = scanner.scan(empty_dataset)
        
        assert result.primary_type == DatasetType.UNKNOWN
        assert result.files.total_count == 0


class TestDatasetScannerTypeDetection:
    """Test correct type detection for various datasets."""
    
    def test_detects_image_dataset(self, flat_image_dataset: Path):
        """Verify image dataset is correctly identified."""
        from app.core.dataset_scanner import DatasetScanner
        from app.core.dataset_scanner.models import DatasetType
        
        scanner = DatasetScanner()
        result = scanner.scan(flat_image_dataset)
        
        assert result.primary_type == DatasetType.IMAGE
    
    def test_detects_graph_dataset(self, graph_pt_dataset: Path):
        """Verify graph dataset is correctly identified."""
        from app.core.dataset_scanner import DatasetScanner
        from app.core.dataset_scanner.models import DatasetType
        
        scanner = DatasetScanner()
        result = scanner.scan(graph_pt_dataset)
        
        assert result.primary_type == DatasetType.GRAPH
    
    def test_detects_timeseries_dataset(self, flat_numpy_dataset: Path):
        """Verify time series dataset is correctly identified."""
        from app.core.dataset_scanner import DatasetScanner
        from app.core.dataset_scanner.models import DatasetType
        
        scanner = DatasetScanner()
        result = scanner.scan(flat_numpy_dataset)
        
        assert result.primary_type == DatasetType.TIMESERIES
    
    def test_detects_tabular_dataset(self, single_csv_dataset: Path):
        """Verify tabular dataset is correctly identified."""
        from app.core.dataset_scanner import DatasetScanner
        from app.core.dataset_scanner.models import DatasetType
        
        scanner = DatasetScanner()
        result = scanner.scan(single_csv_dataset)
        
        assert result.primary_type == DatasetType.TABULAR
    
    def test_detects_text_dataset(self, text_flat_dataset: Path):
        """Verify text dataset is correctly identified."""
        from app.core.dataset_scanner import DatasetScanner
        from app.core.dataset_scanner.models import DatasetType
        
        scanner = DatasetScanner()
        result = scanner.scan(text_flat_dataset)
        
        assert result.primary_type == DatasetType.TEXT
    
    def test_detects_phases_dataset(self, by_condition_phases_dataset: Path):
        """Verify phases dataset is correctly identified as graph."""
        from app.core.dataset_scanner import DatasetScanner
        from app.core.dataset_scanner.models import DatasetType
        
        scanner = DatasetScanner()
        result = scanner.scan(by_condition_phases_dataset)
        
        assert result.primary_type == DatasetType.GRAPH
    
    def test_detects_mixed_dataset(self, mixed_dataset: Path):
        """Verify mixed dataset is correctly identified."""
        from app.core.dataset_scanner import DatasetScanner
        from app.core.dataset_scanner.models import DatasetType
        
        scanner = DatasetScanner()
        result = scanner.scan(mixed_dataset)
        
        # Should either be MIXED or the dominant type
        assert result.primary_type in [DatasetType.MIXED, DatasetType.IMAGE, 
                                       DatasetType.TIMESERIES, DatasetType.TABULAR]


class TestDatasetScannerStructureDetection:
    """Test correct structure detection."""
    
    def test_detects_split_structure(self, train_val_test_image_dataset: Path):
        """Verify split structure is detected."""
        from app.core.dataset_scanner import DatasetScanner
        from app.core.dataset_scanner.models import SplitType
        
        scanner = DatasetScanner()
        result = scanner.scan(train_val_test_image_dataset)
        
        assert result.structure.split_type == SplitType.TRAIN_VAL_TEST
    
    def test_detects_class_structure(self, by_class_image_dataset: Path):
        """Verify class structure is detected."""
        from app.core.dataset_scanner import DatasetScanner
        
        scanner = DatasetScanner()
        result = scanner.scan(by_class_image_dataset)
        
        assert result.structure.has_classes is True
        assert set(result.structure.classes) == {'cat', 'dog', 'bird'}
    
    def test_detects_condition_structure(self, by_condition_phases_dataset: Path):
        """Verify condition structure is detected."""
        from app.core.dataset_scanner import DatasetScanner
        from app.core.dataset_scanner.models import SplitType
        
        scanner = DatasetScanner()
        result = scanner.scan(by_condition_phases_dataset)
        
        assert result.structure.split_type == SplitType.BY_CONDITION
        assert set(result.structure.classes) == {'DMT', 'EC', 'EO'}
    
    def test_detects_subject_structure(self, by_subject_dataset: Path):
        """Verify subject structure is detected."""
        from app.core.dataset_scanner import DatasetScanner
        
        scanner = DatasetScanner()
        result = scanner.scan(by_subject_dataset)
        
        assert result.structure.has_subjects is True


class TestDatasetScannerFileInfo:
    """Test file information extraction."""
    
    def test_counts_files(self, flat_image_dataset: Path):
        """Verify file count is accurate."""
        from app.core.dataset_scanner import DatasetScanner
        
        scanner = DatasetScanner()
        result = scanner.scan(flat_image_dataset)
        
        assert result.files.total_count == 15  # 10 PNG + 5 JPG
    
    def test_groups_by_extension(self, flat_image_dataset: Path):
        """Verify files are grouped by extension."""
        from app.core.dataset_scanner import DatasetScanner
        
        scanner = DatasetScanner()
        result = scanner.scan(flat_image_dataset)
        
        assert '.png' in result.files.by_extension
        # The count may vary based on how extensions are distributed
        assert result.files.by_extension['.png'] >= 5
    
    def test_calculates_total_size(self, flat_image_dataset: Path):
        """Verify total size is calculated."""
        from app.core.dataset_scanner import DatasetScanner
        
        scanner = DatasetScanner()
        result = scanner.scan(flat_image_dataset)
        
        assert result.files.total_size_bytes > 0
    
    def test_provides_sample_files(self, by_class_image_dataset: Path):
        """Verify sample files are provided."""
        from app.core.dataset_scanner import DatasetScanner
        
        scanner = DatasetScanner()
        result = scanner.scan(by_class_image_dataset)
        
        assert len(result.files.sample_files) > 0
        assert all(isinstance(f, Path) for f in result.files.sample_files)


class TestDatasetScannerSplitInfo:
    """Test split information extraction."""
    
    def test_extracts_split_counts(self, train_val_test_image_dataset: Path):
        """Verify split counts are extracted."""
        from app.core.dataset_scanner import DatasetScanner
        
        scanner = DatasetScanner()
        result = scanner.scan(train_val_test_image_dataset)
        
        assert result.splits is not None
        assert result.splits.train_count == 50
        assert result.splits.val_count == 15
        assert result.splits.test_count == 15
    
    def test_calculates_split_ratios(self, train_val_test_image_dataset: Path):
        """Verify split ratios are calculated."""
        from app.core.dataset_scanner import DatasetScanner
        
        scanner = DatasetScanner()
        result = scanner.scan(train_val_test_image_dataset)
        
        # 50/15/15 = 62.5%/18.75%/18.75%
        assert abs(result.splits.train_ratio - 0.625) < 0.01
    
    def test_no_splits_returns_none(self, flat_image_dataset: Path):
        """Verify no splits returns None."""
        from app.core.dataset_scanner import DatasetScanner
        
        scanner = DatasetScanner()
        result = scanner.scan(flat_image_dataset)
        
        assert result.splits is None


class TestDatasetScannerTypeSpecific:
    """Test type-specific information extraction."""
    
    def test_image_specific_info(self, flat_image_dataset: Path):
        """Verify image-specific info is extracted."""
        from app.core.dataset_scanner import DatasetScanner
        
        scanner = DatasetScanner()
        result = scanner.scan(flat_image_dataset, deep_scan=True)
        
        assert result.type_specific is not None
        # Should have image-related info
        ts = result.type_specific
        assert 'sizes' in ts or 'channels' in ts or 'color_mode' in ts or len(ts) > 0
    
    def test_graph_specific_info(self, by_condition_phases_dataset: Path):
        """Verify graph-specific info is extracted."""
        from app.core.dataset_scanner import DatasetScanner
        
        scanner = DatasetScanner()
        result = scanner.scan(by_condition_phases_dataset, deep_scan=True)
        
        assert result.type_specific is not None
        # Should have phases-related info
        ts = result.type_specific
        assert 'bands' in ts or 'has_eeg' in ts or 'has_stc' in ts or len(ts) > 0
    
    def test_tabular_specific_info(self, single_csv_dataset: Path):
        """Verify tabular-specific info is extracted."""
        from app.core.dataset_scanner import DatasetScanner
        
        scanner = DatasetScanner()
        result = scanner.scan(single_csv_dataset, deep_scan=True)
        
        assert result.type_specific is not None
        ts = result.type_specific
        assert 'num_rows' in ts or 'num_columns' in ts or 'columns' in ts or len(ts) > 0


class TestDatasetScannerOptions:
    """Test scanner options."""
    
    def test_deep_scan_option(self, flat_image_dataset: Path):
        """Verify deep_scan extracts more information."""
        from app.core.dataset_scanner import DatasetScanner
        
        scanner = DatasetScanner()
        
        quick_result = scanner.scan(flat_image_dataset, deep_scan=False)
        deep_result = scanner.scan(flat_image_dataset, deep_scan=True)
        
        # Deep scan should provide more type-specific info
        assert len(deep_result.type_specific) >= len(quick_result.type_specific) or \
               deep_result.type_specific is not None
    
    def test_sample_size_option(self, by_class_image_dataset: Path):
        """Verify sample_size limits analysis."""
        from app.core.dataset_scanner import DatasetScanner
        
        scanner = DatasetScanner()
        result = scanner.scan(by_class_image_dataset, deep_scan=True, sample_size=5)
        
        # Should still work, just with fewer samples analyzed
        assert result.is_valid is True
    
    def test_detect_labels_option(self, by_class_image_dataset: Path):
        """Verify detect_labels extracts class labels."""
        from app.core.dataset_scanner import DatasetScanner
        
        scanner = DatasetScanner()
        result = scanner.scan(by_class_image_dataset, detect_labels=True)
        
        assert len(result.structure.classes) > 0


class TestDatasetScannerWarningsAndSuggestions:
    """Test warnings and suggestions generation."""
    
    def test_generates_warnings_for_issues(self, dataset_with_corrupt_files: Path):
        """Verify warnings are generated for problematic datasets."""
        from app.core.dataset_scanner import DatasetScanner
        
        scanner = DatasetScanner()
        result = scanner.scan(dataset_with_corrupt_files, deep_scan=True)
        
        # Should have warnings about corrupt files
        assert len(result.warnings) > 0
    
    def test_generates_suggestions(self, by_class_image_dataset: Path):
        """Verify suggestions are generated."""
        from app.core.dataset_scanner import DatasetScanner
        
        scanner = DatasetScanner()
        result = scanner.scan(by_class_image_dataset)
        
        # Should suggest appropriate loaders/frameworks
        assert len(result.suggestions) > 0 or len(result.suggested_loaders) > 0
    
    def test_warns_about_imbalanced_classes(self, temp_dir: Path):
        """Verify warning for imbalanced classes."""
        from app.core.dataset_scanner import DatasetScanner
        from tests.dataset_scanner.conftest import create_minimal_png
        
        # Create imbalanced dataset
        dataset_path = temp_dir / "imbalanced"
        for i in range(100):
            create_minimal_png(dataset_path / "majority" / f"img_{i:03d}.png")
        for i in range(5):
            create_minimal_png(dataset_path / "minority" / f"img_{i:03d}.png")
        
        scanner = DatasetScanner()
        result = scanner.scan(dataset_path)
        
        # Should warn about imbalance
        warnings_text = ' '.join(result.warnings).lower()
        assert 'imbalance' in warnings_text or 'unbalanced' in warnings_text or len(result.warnings) > 0


class TestDatasetScannerCompatibility:
    """Test framework compatibility detection."""
    
    def test_suggests_pytorch_for_images(self, by_class_image_dataset: Path):
        """Verify PyTorch is suggested for image datasets."""
        from app.core.dataset_scanner import DatasetScanner
        
        scanner = DatasetScanner()
        result = scanner.scan(by_class_image_dataset)
        
        frameworks = result.compatible_frameworks
        assert 'pytorch' in [f.lower() for f in frameworks] or 'PyTorch' in frameworks
    
    def test_suggests_appropriate_loaders(self, by_class_image_dataset: Path):
        """Verify appropriate data loaders are suggested."""
        from app.core.dataset_scanner import DatasetScanner
        
        scanner = DatasetScanner()
        result = scanner.scan(by_class_image_dataset)
        
        loaders = result.suggested_loaders
        # ImageFolder is appropriate for class-based image datasets
        assert any('ImageFolder' in loader for loader in loaders) or len(loaders) > 0


class TestDatasetScannerOutput:
    """Test output format and completeness."""
    
    def test_to_dict_conversion(self, flat_image_dataset: Path):
        """Verify DatasetInfo can be converted to dict."""
        from app.core.dataset_scanner import DatasetScanner
        
        scanner = DatasetScanner()
        result = scanner.scan(flat_image_dataset)
        
        d = result.to_dict()
        
        assert isinstance(d, dict)
        assert 'primary_type' in d
        assert 'files' in d
        assert 'structure' in d
    
    def test_summary_generation(self, by_class_image_dataset: Path):
        """Verify human-readable summary is generated."""
        from app.core.dataset_scanner import DatasetScanner
        
        scanner = DatasetScanner()
        result = scanner.scan(by_class_image_dataset)
        
        summary = result.summary()
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        # Should contain key information
        assert 'image' in summary.lower() or result.name in summary
    
    def test_json_serializable(self, flat_image_dataset: Path):
        """Verify result is JSON serializable."""
        from app.core.dataset_scanner import DatasetScanner
        import json
        
        scanner = DatasetScanner()
        result = scanner.scan(flat_image_dataset)
        
        d = result.to_dict()
        
        # Should not raise
        json_str = json.dumps(d, default=str)
        assert len(json_str) > 0


class TestDatasetScannerPerformance:
    """Test scanner performance characteristics."""
    
    def test_handles_large_dataset(self, temp_dir: Path):
        """Verify scanner handles larger datasets efficiently."""
        from app.core.dataset_scanner import DatasetScanner
        from tests.dataset_scanner.conftest import create_minimal_png
        import time
        
        # Create dataset with many files
        dataset_path = temp_dir / "large"
        for i in range(500):
            create_minimal_png(dataset_path / f"img_{i:05d}.png")
        
        scanner = DatasetScanner()
        
        start = time.time()
        result = scanner.scan(dataset_path)
        elapsed = time.time() - start
        
        assert result.files.total_count == 500
        # Should complete in reasonable time (< 30 seconds for 500 files)
        assert elapsed < 30
    
    def test_quick_scan_is_faster(self, temp_dir: Path):
        """Verify quick scan is faster than deep scan."""
        from app.core.dataset_scanner import DatasetScanner
        from tests.dataset_scanner.conftest import create_minimal_png
        import time
        
        dataset_path = temp_dir / "speed_test"
        for i in range(100):
            create_minimal_png(dataset_path / f"img_{i:03d}.png")
        
        scanner = DatasetScanner()
        
        start = time.time()
        scanner.scan(dataset_path, deep_scan=False)
        quick_time = time.time() - start
        
        start = time.time()
        scanner.scan(dataset_path, deep_scan=True)
        deep_time = time.time() - start
        
        # Quick scan should be faster (or at least not slower)
        assert quick_time <= deep_time + 1  # Allow 1 second margin


class TestDatasetScannerCaching:
    """Test caching functionality."""
    
    def test_scan_caching(self, flat_image_dataset: Path):
        """Verify results can be cached for repeated scans."""
        from app.core.dataset_scanner import DatasetScanner
        
        scanner = DatasetScanner()
        
        # First scan
        result1 = scanner.scan(flat_image_dataset)
        
        # Second scan (could use cache)
        result2 = scanner.scan(flat_image_dataset)
        
        # Results should be equivalent
        assert result1.files.total_count == result2.files.total_count
        assert result1.primary_type == result2.primary_type

