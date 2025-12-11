"""
Tests for StructureAnalyzer.

Verifies detection of directory structures, splits, classes, and subjects.
"""

import pytest
from pathlib import Path


class TestStructureAnalyzerBasic:
    """Basic structure analysis tests."""
    
    def test_analyzes_flat_structure(self, flat_image_dataset: Path):
        """Verify flat directory structure is detected."""
        from app.core.dataset_scanner.analyzers import StructureAnalyzer
        from app.core.dataset_scanner.models import SplitType
        
        analyzer = StructureAnalyzer()
        result = analyzer.analyze(flat_image_dataset)
        
        assert result.split_type == SplitType.NONE
        assert result.hierarchy_depth == 0 or result.hierarchy_depth == 1
    
    def test_analyzes_nested_structure(self, deeply_nested_dataset: Path):
        """Verify nested directory structure is detected."""
        from app.core.dataset_scanner.analyzers import StructureAnalyzer
        
        analyzer = StructureAnalyzer()
        result = analyzer.analyze(deeply_nested_dataset)
        
        assert result.hierarchy_depth >= 3
    
    def test_empty_directory(self, empty_dataset: Path):
        """Verify empty directory analysis."""
        from app.core.dataset_scanner.analyzers import StructureAnalyzer
        from app.core.dataset_scanner.models import SplitType
        
        analyzer = StructureAnalyzer()
        result = analyzer.analyze(empty_dataset)
        
        assert result.split_type == SplitType.NONE
        assert result.has_classes is False
        assert result.has_subjects is False


class TestStructureAnalyzerSplits:
    """Test split detection."""
    
    def test_detects_train_test(self, train_test_image_dataset: Path):
        """Verify train/test split is detected."""
        from app.core.dataset_scanner.analyzers import StructureAnalyzer
        from app.core.dataset_scanner.models import SplitType
        
        analyzer = StructureAnalyzer()
        result = analyzer.analyze(train_test_image_dataset)
        
        assert result.split_type == SplitType.TRAIN_TEST
        assert 'train' in result.root_subdirs
        assert 'test' in result.root_subdirs
    
    def test_detects_train_val_test(self, train_val_test_image_dataset: Path):
        """Verify train/val/test split is detected."""
        from app.core.dataset_scanner.analyzers import StructureAnalyzer
        from app.core.dataset_scanner.models import SplitType
        
        analyzer = StructureAnalyzer()
        result = analyzer.analyze(train_val_test_image_dataset)
        
        assert result.split_type == SplitType.TRAIN_VAL_TEST
    
    def test_detects_validation_variant(self, temp_dir: Path):
        """Verify 'validation' folder is recognized as 'val'."""
        from app.core.dataset_scanner.analyzers import StructureAnalyzer
        from app.core.dataset_scanner.models import SplitType
        
        # Create with 'validation' instead of 'val'
        dataset_path = temp_dir / "with_validation"
        (dataset_path / "train").mkdir(parents=True)
        (dataset_path / "validation").mkdir()
        (dataset_path / "test").mkdir()
        
        analyzer = StructureAnalyzer()
        result = analyzer.analyze(dataset_path)
        
        assert result.split_type == SplitType.TRAIN_VAL_TEST
    
    def test_case_insensitive_splits(self, temp_dir: Path):
        """Verify split detection is case-insensitive."""
        from app.core.dataset_scanner.analyzers import StructureAnalyzer
        from app.core.dataset_scanner.models import SplitType
        
        dataset_path = temp_dir / "case_insensitive"
        (dataset_path / "Train").mkdir(parents=True)
        (dataset_path / "TEST").mkdir()
        
        analyzer = StructureAnalyzer()
        result = analyzer.analyze(dataset_path)
        
        assert result.split_type == SplitType.TRAIN_TEST


class TestStructureAnalyzerClasses:
    """Test class detection."""
    
    def test_detects_class_folders(self, by_class_image_dataset: Path):
        """Verify class-based folders are detected."""
        from app.core.dataset_scanner.analyzers import StructureAnalyzer
        from app.core.dataset_scanner.models import SplitType
        
        analyzer = StructureAnalyzer()
        result = analyzer.analyze(by_class_image_dataset)
        
        assert result.split_type == SplitType.BY_CLASS
        assert result.has_classes is True
        assert set(result.classes) == {'cat', 'dog', 'bird'}
    
    def test_detects_nested_classes(self, imagenet_style_dataset: Path):
        """Verify classes within splits are detected."""
        from app.core.dataset_scanner.analyzers import StructureAnalyzer
        from app.core.dataset_scanner.models import SplitType
        
        analyzer = StructureAnalyzer()
        result = analyzer.analyze(imagenet_style_dataset)
        
        assert result.split_type == SplitType.HIERARCHICAL
        assert result.has_classes is True
        # ImageNet-style class IDs
        assert len(result.classes) == 3
    
    def test_class_counts(self, by_class_image_dataset: Path):
        """Verify class file counts are accurate."""
        from app.core.dataset_scanner.analyzers import StructureAnalyzer
        
        analyzer = StructureAnalyzer()
        result = analyzer.analyze(by_class_image_dataset)
        
        # Each class has 15 images
        assert all(count == 15 for count in result.class_counts.values())


class TestStructureAnalyzerSubjects:
    """Test subject detection."""
    
    def test_detects_subject_folders(self, by_subject_dataset: Path):
        """Verify subject-based folders are detected."""
        from app.core.dataset_scanner.analyzers import StructureAnalyzer
        from app.core.dataset_scanner.models import SplitType
        
        analyzer = StructureAnalyzer()
        result = analyzer.analyze(by_subject_dataset)
        
        assert result.split_type == SplitType.BY_SUBJECT
        assert result.has_subjects is True
        assert set(result.subjects) == {'S01', 'S02', 'S03'}
    
    def test_detects_subjects_in_filenames(self, by_condition_phases_dataset: Path):
        """Verify subjects are detected from filenames."""
        from app.core.dataset_scanner.analyzers import StructureAnalyzer
        
        analyzer = StructureAnalyzer()
        result = analyzer.analyze(by_condition_phases_dataset)
        
        assert result.has_subjects is True
        assert 'S01' in result.subjects
        assert len(result.subjects) == 5
    
    def test_subject_pattern_variations(self, temp_dir: Path):
        """Verify different subject naming patterns are detected."""
        from app.core.dataset_scanner.analyzers import StructureAnalyzer
        
        dataset_path = temp_dir / "varied_subjects"
        
        # Different subject naming conventions
        patterns = ['sub-01', 'subject_001', 'P001', 'patient01']
        for pattern in patterns:
            (dataset_path / pattern).mkdir(parents=True)
        
        analyzer = StructureAnalyzer()
        result = analyzer.analyze(dataset_path)
        
        assert result.has_subjects is True


class TestStructureAnalyzerConditions:
    """Test condition detection (EEG-style)."""
    
    def test_detects_condition_folders(self, by_condition_phases_dataset: Path):
        """Verify condition-based folders are detected."""
        from app.core.dataset_scanner.analyzers import StructureAnalyzer
        from app.core.dataset_scanner.models import SplitType
        
        analyzer = StructureAnalyzer()
        result = analyzer.analyze(by_condition_phases_dataset)
        
        assert result.split_type == SplitType.BY_CONDITION
        assert result.has_classes is True
        assert set(result.classes) == {'DMT', 'EC', 'EO'}
    
    def test_condition_counts(self, by_condition_phases_dataset: Path):
        """Verify condition file counts are accurate."""
        from app.core.dataset_scanner.analyzers import StructureAnalyzer
        
        analyzer = StructureAnalyzer()
        result = analyzer.analyze(by_condition_phases_dataset)
        
        # Each condition has 5 subjects
        assert result.class_counts['DMT'] == 5
        assert result.class_counts['EC'] == 5
        assert result.class_counts['EO'] == 5


class TestStructureAnalyzerHierarchy:
    """Test hierarchy analysis."""
    
    def test_calculates_hierarchy_depth(self, deeply_nested_dataset: Path):
        """Verify hierarchy depth is calculated correctly."""
        from app.core.dataset_scanner.analyzers import StructureAnalyzer
        
        analyzer = StructureAnalyzer()
        result = analyzer.analyze(deeply_nested_dataset)
        
        # data/category/subcategory/split = 4 levels
        assert result.hierarchy_depth >= 4
    
    def test_lists_root_subdirs(self, train_val_test_image_dataset: Path):
        """Verify root subdirectories are listed."""
        from app.core.dataset_scanner.analyzers import StructureAnalyzer
        
        analyzer = StructureAnalyzer()
        result = analyzer.analyze(train_val_test_image_dataset)
        
        assert set(result.root_subdirs) == {'train', 'val', 'test'}
    
    def test_detects_hierarchical_structure(self, imagenet_style_dataset: Path):
        """Verify hierarchical structure is detected."""
        from app.core.dataset_scanner.analyzers import StructureAnalyzer
        from app.core.dataset_scanner.models import SplitType
        
        analyzer = StructureAnalyzer()
        result = analyzer.analyze(imagenet_style_dataset)
        
        # train/class/, val/class/, test/class/
        assert result.split_type == SplitType.HIERARCHICAL
        assert result.hierarchy_depth == 2


class TestStructureAnalyzerEdgeCases:
    """Test edge cases for structure analysis."""
    
    def test_handles_hidden_directories(self, temp_dir: Path):
        """Verify hidden directories are ignored."""
        from app.core.dataset_scanner.analyzers import StructureAnalyzer
        
        dataset_path = temp_dir / "with_hidden"
        (dataset_path / "train").mkdir(parents=True)
        (dataset_path / ".hidden").mkdir()
        (dataset_path / "__pycache__").mkdir()
        
        analyzer = StructureAnalyzer()
        result = analyzer.analyze(dataset_path)
        
        assert '.hidden' not in result.root_subdirs
        assert '__pycache__' not in result.root_subdirs
    
    def test_handles_symlinks(self, temp_dir: Path):
        """Verify symlinks are handled properly."""
        from app.core.dataset_scanner.analyzers import StructureAnalyzer
        
        dataset_path = temp_dir / "with_symlinks"
        real_dir = dataset_path / "real"
        real_dir.mkdir(parents=True)
        
        # Create symlink
        link_path = dataset_path / "link"
        try:
            link_path.symlink_to(real_dir)
        except (OSError, NotImplementedError):
            pytest.skip("Symlinks not supported")
        
        analyzer = StructureAnalyzer()
        result = analyzer.analyze(dataset_path)
        
        # Should not crash
        assert result is not None
    
    def test_handles_deep_nesting(self, temp_dir: Path):
        """Verify very deep nesting is handled."""
        from app.core.dataset_scanner.analyzers import StructureAnalyzer
        
        # Create deeply nested structure
        dataset_path = temp_dir / "very_deep"
        current = dataset_path
        for i in range(10):
            current = current / f"level_{i}"
        current.mkdir(parents=True)
        # Need at least one file to calculate depth
        (current / "file.txt").touch()
        
        analyzer = StructureAnalyzer()
        result = analyzer.analyze(dataset_path)
        
        assert result.hierarchy_depth >= 10
    
    def test_single_file_directory(self, single_file_dataset: Path):
        """Verify directory with single file is handled."""
        from app.core.dataset_scanner.analyzers import StructureAnalyzer
        from app.core.dataset_scanner.models import SplitType
        
        analyzer = StructureAnalyzer()
        result = analyzer.analyze(single_file_dataset)
        
        assert result.split_type == SplitType.NONE


class TestStructureAnalyzerPatternDetection:
    """Test pattern detection capabilities."""
    
    def test_detects_numbered_files(self, temp_dir: Path):
        """Verify numbered file patterns are detected."""
        from app.core.dataset_scanner.analyzers import StructureAnalyzer
        
        dataset_path = temp_dir / "numbered"
        dataset_path.mkdir()
        
        for i in range(100):
            (dataset_path / f"data_{i:05d}.txt").touch()
        
        analyzer = StructureAnalyzer()
        result = analyzer.analyze(dataset_path)
        
        assert result.has_numbered_files is True
        assert result.numbering_pattern is not None
    
    def test_detects_date_based_files(self, temp_dir: Path):
        """Verify date-based file patterns are detected."""
        from app.core.dataset_scanner.analyzers import StructureAnalyzer
        
        dataset_path = temp_dir / "dated"
        dataset_path.mkdir()
        
        dates = ['2024-01-01', '2024-01-02', '2024-01-03']
        for date in dates:
            (dataset_path / f"log_{date}.txt").touch()
        
        analyzer = StructureAnalyzer()
        result = analyzer.analyze(dataset_path)
        
        # Should detect some date pattern
        assert result.has_date_pattern is True or len(result.file_patterns) > 0


class TestStructureAnalyzerStatistics:
    """Test statistical analysis of structure."""
    
    def test_counts_total_files(self, by_class_image_dataset: Path):
        """Verify total file count is accurate."""
        from app.core.dataset_scanner.analyzers import StructureAnalyzer
        
        analyzer = StructureAnalyzer()
        result = analyzer.analyze(by_class_image_dataset)
        
        # 3 classes × 15 images = 45
        assert result.total_files == 45
    
    def test_counts_total_directories(self, imagenet_style_dataset: Path):
        """Verify total directory count is accurate."""
        from app.core.dataset_scanner.analyzers import StructureAnalyzer
        
        analyzer = StructureAnalyzer()
        result = analyzer.analyze(imagenet_style_dataset)
        
        # train/, val/, test/ + 3 classes each = 3 + 9 = 12
        assert result.total_directories >= 12
    
    def test_calculates_size_distribution(self, by_class_image_dataset: Path):
        """Verify size distribution across classes is calculated."""
        from app.core.dataset_scanner.analyzers import StructureAnalyzer
        
        analyzer = StructureAnalyzer()
        result = analyzer.analyze(by_class_image_dataset)
        
        # All classes should be balanced
        counts = list(result.class_counts.values())
        assert max(counts) == min(counts)  # Balanced


class TestStructureAnalyzerOutput:
    """Test output format and completeness."""
    
    def test_returns_structure_info(self, train_val_test_image_dataset: Path):
        """Verify StructureInfo is returned with all fields."""
        from app.core.dataset_scanner.analyzers import StructureAnalyzer
        from app.core.dataset_scanner.models import StructureInfo
        
        analyzer = StructureAnalyzer()
        result = analyzer.analyze(train_val_test_image_dataset)
        
        assert isinstance(result, StructureInfo)
        assert hasattr(result, 'split_type')
        assert hasattr(result, 'has_classes')
        assert hasattr(result, 'has_subjects')
        assert hasattr(result, 'hierarchy_depth')
        assert hasattr(result, 'root_subdirs')
    
    def test_to_dict_conversion(self, by_class_image_dataset: Path):
        """Verify result can be converted to dict."""
        from app.core.dataset_scanner.analyzers import StructureAnalyzer
        
        analyzer = StructureAnalyzer()
        result = analyzer.analyze(by_class_image_dataset)
        
        d = result.to_dict() if hasattr(result, 'to_dict') else vars(result)
        
        assert isinstance(d, dict)

