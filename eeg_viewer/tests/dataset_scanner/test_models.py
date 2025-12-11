"""
Tests for dataset_scanner data models (dataclasses).

These tests verify that the data structures used by the scanner
work correctly and provide all expected functionality.
"""

import pytest
from pathlib import Path
from enum import Enum


class TestDatasetType:
    """Tests for DatasetType enum."""
    
    def test_dataset_type_values_exist(self):
        """Verify all expected dataset types are defined."""
        from app.core.dataset_scanner.models import DatasetType
        
        expected_types = ['IMAGE', 'GRAPH', 'TIMESERIES', 'TABULAR', 'TEXT', 'MIXED', 'UNKNOWN']
        
        for type_name in expected_types:
            assert hasattr(DatasetType, type_name), f"Missing DatasetType.{type_name}"
    
    def test_dataset_type_is_enum(self):
        """Verify DatasetType is an Enum."""
        from app.core.dataset_scanner.models import DatasetType
        
        assert issubclass(DatasetType, Enum)
    
    def test_dataset_type_string_value(self):
        """Verify dataset types have proper string values."""
        from app.core.dataset_scanner.models import DatasetType
        
        # Each type should have a lowercase string value
        assert DatasetType.IMAGE.value == 'image'
        assert DatasetType.GRAPH.value == 'graph'
        assert DatasetType.TIMESERIES.value == 'timeseries'


class TestSplitType:
    """Tests for SplitType enum."""
    
    def test_split_type_values_exist(self):
        """Verify all expected split types are defined."""
        from app.core.dataset_scanner.models import SplitType
        
        expected_types = [
            'NONE',           # No splits
            'TRAIN_TEST',     # train/, test/
            'TRAIN_VAL_TEST', # train/, val/, test/
            'BY_CLASS',       # class_a/, class_b/
            'BY_SUBJECT',     # S01/, S02/
            'BY_CONDITION',   # DMT/, EC/, EO/
            'HIERARCHICAL',   # Complex nested structure
        ]
        
        for type_name in expected_types:
            assert hasattr(SplitType, type_name), f"Missing SplitType.{type_name}"


class TestFileInfo:
    """Tests for FileInfo dataclass."""
    
    def test_file_info_creation(self):
        """Verify FileInfo can be created with all fields."""
        from app.core.dataset_scanner.models import FileInfo
        
        info = FileInfo(
            total_count=100,
            by_extension={'.png': 80, '.jpg': 20},
            by_type={'image': 100},
            sample_files=[Path('/test/img.png')],
            total_size_bytes=1024000,
        )
        
        assert info.total_count == 100
        assert info.by_extension['.png'] == 80
        assert info.total_size_bytes == 1024000
    
    def test_file_info_total_size_human(self):
        """Verify human-readable size formatting."""
        from app.core.dataset_scanner.models import FileInfo
        
        # Test various sizes
        info_bytes = FileInfo(total_count=1, by_extension={}, by_type={}, 
                              sample_files=[], total_size_bytes=500)
        assert '500' in info_bytes.total_size_human and 'B' in info_bytes.total_size_human
        
        info_kb = FileInfo(total_count=1, by_extension={}, by_type={},
                           sample_files=[], total_size_bytes=2048)
        assert 'KB' in info_kb.total_size_human or 'kB' in info_kb.total_size_human.upper()
        
        info_mb = FileInfo(total_count=1, by_extension={}, by_type={},
                           sample_files=[], total_size_bytes=5 * 1024 * 1024)
        assert 'MB' in info_mb.total_size_human.upper()
        
        info_gb = FileInfo(total_count=1, by_extension={}, by_type={},
                           sample_files=[], total_size_bytes=2 * 1024 * 1024 * 1024)
        assert 'GB' in info_gb.total_size_human.upper()
    
    def test_file_info_empty_dataset(self):
        """Verify FileInfo handles empty dataset."""
        from app.core.dataset_scanner.models import FileInfo
        
        info = FileInfo(
            total_count=0,
            by_extension={},
            by_type={},
            sample_files=[],
            total_size_bytes=0,
        )
        
        assert info.total_count == 0
        assert info.total_size_human is not None  # Should still return a string


class TestStructureInfo:
    """Tests for StructureInfo dataclass."""
    
    def test_structure_info_creation(self):
        """Verify StructureInfo can be created with all fields."""
        from app.core.dataset_scanner.models import StructureInfo, SplitType
        
        info = StructureInfo(
            split_type=SplitType.TRAIN_VAL_TEST,
            has_classes=True,
            classes=['cat', 'dog', 'bird'],
            has_subjects=False,
            subjects=[],
            hierarchy_depth=2,
            root_subdirs=['train', 'val', 'test'],
        )
        
        assert info.split_type == SplitType.TRAIN_VAL_TEST
        assert info.has_classes is True
        assert len(info.classes) == 3
        assert info.hierarchy_depth == 2
    
    def test_structure_info_with_subjects(self):
        """Verify StructureInfo with subject-based structure."""
        from app.core.dataset_scanner.models import StructureInfo, SplitType
        
        info = StructureInfo(
            split_type=SplitType.BY_SUBJECT,
            has_classes=False,
            classes=[],
            has_subjects=True,
            subjects=['S01', 'S02', 'S03'],
            hierarchy_depth=1,
            root_subdirs=['S01', 'S02', 'S03'],
        )
        
        assert info.has_subjects is True
        assert len(info.subjects) == 3
        assert 'S01' in info.subjects
    
    def test_structure_info_with_conditions(self):
        """Verify StructureInfo with condition-based structure (EEG-style)."""
        from app.core.dataset_scanner.models import StructureInfo, SplitType
        
        info = StructureInfo(
            split_type=SplitType.BY_CONDITION,
            has_classes=True,
            classes=['DMT', 'EC', 'EO'],
            has_subjects=True,
            subjects=['S01', 'S02'],
            hierarchy_depth=2,
            root_subdirs=['DMT', 'EC', 'EO'],
        )
        
        assert info.split_type == SplitType.BY_CONDITION
        assert 'DMT' in info.classes


class TestSplitInfo:
    """Tests for SplitInfo dataclass."""
    
    def test_split_info_creation(self):
        """Verify SplitInfo can be created with all fields."""
        from app.core.dataset_scanner.models import SplitInfo
        
        info = SplitInfo(
            train_count=700,
            val_count=150,
            test_count=150,
        )
        
        assert info.train_count == 700
        assert info.val_count == 150
        assert info.test_count == 150
    
    def test_split_info_ratios(self):
        """Verify split ratios are calculated correctly."""
        from app.core.dataset_scanner.models import SplitInfo
        
        info = SplitInfo(
            train_count=700,
            val_count=150,
            test_count=150,
        )
        
        assert abs(info.train_ratio - 0.70) < 0.01
        assert abs(info.val_ratio - 0.15) < 0.01
        assert abs(info.test_ratio - 0.15) < 0.01
        assert abs(info.train_ratio + info.val_ratio + info.test_ratio - 1.0) < 0.01
    
    def test_split_info_no_val(self):
        """Verify SplitInfo handles train/test only split."""
        from app.core.dataset_scanner.models import SplitInfo
        
        info = SplitInfo(
            train_count=800,
            val_count=0,
            test_count=200,
        )
        
        assert info.val_count == 0
        assert info.val_ratio == 0.0
        assert abs(info.train_ratio - 0.80) < 0.01
    
    def test_split_info_total(self):
        """Verify total property works correctly."""
        from app.core.dataset_scanner.models import SplitInfo
        
        info = SplitInfo(train_count=700, val_count=150, test_count=150)
        
        assert info.total == 1000


class TestTypeSpecificInfo:
    """Tests for type-specific info classes."""
    
    def test_image_specific_info(self):
        """Verify ImageSpecificInfo fields."""
        from app.core.dataset_scanner.models import ImageSpecificInfo
        
        info = ImageSpecificInfo(
            image_sizes=[(64, 64), (128, 128), (64, 64)],
            channels=3,
            color_mode='RGB',
            formats=['.png', '.jpg'],
        )
        
        assert info.channels == 3
        assert info.color_mode == 'RGB'
        assert info.most_common_size == (64, 64)  # Should return most frequent
        assert '.png' in info.formats
    
    def test_graph_specific_info(self):
        """Verify GraphSpecificInfo fields."""
        from app.core.dataset_scanner.models import GraphSpecificInfo
        
        info = GraphSpecificInfo(
            num_nodes_range=(10, 100),
            num_edges_range=(20, 500),
            node_features_dim=64,
            edge_features_dim=1,
            has_node_labels=True,
            has_edge_labels=False,
            is_directed=False,
        )
        
        assert info.num_nodes_range == (10, 100)
        assert info.node_features_dim == 64
        assert info.is_directed is False
    
    def test_timeseries_specific_info(self):
        """Verify TimeSeriesSpecificInfo fields."""
        from app.core.dataset_scanner.models import TimeSeriesSpecificInfo
        
        info = TimeSeriesSpecificInfo(
            sampling_rate=256.0,
            num_channels=24,
            duration_range=(1.0, 10.0),
            signal_length_range=(256, 2560),
            bands=['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'],
            has_stc=True,
            has_eeg=True,
            n_parcels=68,
        )
        
        assert info.sampling_rate == 256.0
        assert info.num_channels == 24
        assert 'Alpha' in info.bands
        assert info.has_stc is True
    
    def test_tabular_specific_info(self):
        """Verify TabularSpecificInfo fields."""
        from app.core.dataset_scanner.models import TabularSpecificInfo
        
        info = TabularSpecificInfo(
            num_rows=1000,
            num_columns=15,
            column_types={'id': 'int64', 'name': 'object', 'value': 'float64'},
            numeric_columns=['id', 'value'],
            categorical_columns=['name'],
            missing_counts={'name': 5, 'value': 10},
            suggested_target='label',
        )
        
        assert info.num_rows == 1000
        assert info.num_columns == 15
        assert 'id' in info.numeric_columns
        assert info.missing_counts['value'] == 10
    
    def test_text_specific_info(self):
        """Verify TextSpecificInfo fields."""
        from app.core.dataset_scanner.models import TextSpecificInfo
        
        info = TextSpecificInfo(
            total_documents=500,
            avg_document_length=150.5,
            vocabulary_size=10000,
            format='txt',
            is_tokenized=False,
        )
        
        assert info.total_documents == 500
        assert info.avg_document_length == 150.5
        assert info.is_tokenized is False


class TestDatasetInfo:
    """Tests for main DatasetInfo dataclass."""
    
    def test_dataset_info_creation(self):
        """Verify DatasetInfo can be created with all fields."""
        from app.core.dataset_scanner.models import (
            DatasetInfo, DatasetType, FileInfo, StructureInfo, SplitType
        )
        
        info = DatasetInfo(
            path=Path('/test/dataset'),
            name='test_dataset',
            primary_type=DatasetType.IMAGE,
            structure=StructureInfo(
                split_type=SplitType.TRAIN_VAL_TEST,
                has_classes=True,
                classes=['a', 'b'],
                has_subjects=False,
                subjects=[],
                hierarchy_depth=2,
                root_subdirs=['train', 'val', 'test'],
            ),
            files=FileInfo(
                total_count=100,
                by_extension={'.png': 100},
                by_type={'image': 100},
                sample_files=[],
                total_size_bytes=1024000,
            ),
            splits=None,
            type_specific={},
            warnings=[],
            suggestions=[],
        )
        
        assert info.name == 'test_dataset'
        assert info.primary_type == DatasetType.IMAGE
        assert info.files.total_count == 100
    
    def test_dataset_info_to_dict(self):
        """Verify DatasetInfo can be converted to dictionary."""
        from app.core.dataset_scanner.models import (
            DatasetInfo, DatasetType, FileInfo, StructureInfo, SplitType
        )
        
        info = DatasetInfo(
            path=Path('/test/dataset'),
            name='test_dataset',
            primary_type=DatasetType.IMAGE,
            structure=StructureInfo(
                split_type=SplitType.NONE,
                has_classes=False,
                classes=[],
                has_subjects=False,
                subjects=[],
                hierarchy_depth=0,
                root_subdirs=[],
            ),
            files=FileInfo(
                total_count=50,
                by_extension={'.png': 50},
                by_type={'image': 50},
                sample_files=[],
                total_size_bytes=512000,
            ),
            splits=None,
            type_specific={},
            warnings=[],
            suggestions=[],
        )
        
        d = info.to_dict()
        
        assert isinstance(d, dict)
        assert d['name'] == 'test_dataset'
        assert d['primary_type'] == 'image'
        assert d['files']['total_count'] == 50
    
    def test_dataset_info_summary(self):
        """Verify DatasetInfo generates readable summary."""
        from app.core.dataset_scanner.models import (
            DatasetInfo, DatasetType, FileInfo, StructureInfo, SplitType
        )
        
        info = DatasetInfo(
            path=Path('/test/dataset'),
            name='test_dataset',
            primary_type=DatasetType.IMAGE,
            structure=StructureInfo(
                split_type=SplitType.TRAIN_VAL_TEST,
                has_classes=True,
                classes=['cat', 'dog'],
                has_subjects=False,
                subjects=[],
                hierarchy_depth=2,
                root_subdirs=['train', 'val', 'test'],
            ),
            files=FileInfo(
                total_count=100,
                by_extension={'.png': 100},
                by_type={'image': 100},
                sample_files=[],
                total_size_bytes=1024000,
            ),
            splits=None,
            type_specific={},
            warnings=['Some images are very small'],
            suggestions=['Consider resizing to uniform size'],
        )
        
        summary = info.summary()
        
        assert isinstance(summary, str)
        assert 'test_dataset' in summary
        assert 'IMAGE' in summary.upper() or 'image' in summary.lower()
        assert '100' in summary  # File count
    
    def test_dataset_info_is_valid(self):
        """Verify is_valid property works correctly."""
        from app.core.dataset_scanner.models import (
            DatasetInfo, DatasetType, FileInfo, StructureInfo, SplitType
        )
        
        # Valid dataset
        valid_info = DatasetInfo(
            path=Path('/test/dataset'),
            name='test',
            primary_type=DatasetType.IMAGE,
            structure=StructureInfo(SplitType.NONE, False, [], False, [], 0, []),
            files=FileInfo(100, {'.png': 100}, {'image': 100}, [], 1024),
            splits=None,
            type_specific={},
            warnings=[],
            suggestions=[],
        )
        assert valid_info.is_valid is True
        
        # Empty dataset (invalid)
        empty_info = DatasetInfo(
            path=Path('/test/empty'),
            name='empty',
            primary_type=DatasetType.UNKNOWN,
            structure=StructureInfo(SplitType.NONE, False, [], False, [], 0, []),
            files=FileInfo(0, {}, {}, [], 0),
            splits=None,
            type_specific={},
            warnings=['No files found'],
            suggestions=[],
        )
        assert empty_info.is_valid is False
    
    def test_dataset_info_compatible_frameworks(self):
        """Verify compatible_frameworks property returns expected frameworks."""
        from app.core.dataset_scanner.models import (
            DatasetInfo, DatasetType, FileInfo, StructureInfo, SplitType
        )
        
        image_info = DatasetInfo(
            path=Path('/test'),
            name='images',
            primary_type=DatasetType.IMAGE,
            structure=StructureInfo(SplitType.BY_CLASS, True, ['a', 'b'], False, [], 1, ['a', 'b']),
            files=FileInfo(100, {'.png': 100}, {'image': 100}, [], 1024),
            splits=None,
            type_specific={},
            warnings=[],
            suggestions=[],
        )
        
        frameworks = image_info.compatible_frameworks
        
        assert isinstance(frameworks, list)
        assert 'pytorch' in frameworks or 'PyTorch' in frameworks
        # Image datasets should be compatible with common frameworks
    
    def test_dataset_info_suggested_loaders(self):
        """Verify suggested_loaders returns appropriate loaders."""
        from app.core.dataset_scanner.models import (
            DatasetInfo, DatasetType, FileInfo, StructureInfo, SplitType
        )
        
        image_info = DatasetInfo(
            path=Path('/test'),
            name='images',
            primary_type=DatasetType.IMAGE,
            structure=StructureInfo(SplitType.BY_CLASS, True, ['a', 'b'], False, [], 1, ['a', 'b']),
            files=FileInfo(100, {'.png': 100}, {'image': 100}, [], 1024),
            splits=None,
            type_specific={},
            warnings=[],
            suggestions=[],
        )
        
        loaders = image_info.suggested_loaders
        
        assert isinstance(loaders, list)
        # Should suggest ImageFolder for class-based image dataset
        assert any('ImageFolder' in loader for loader in loaders) or len(loaders) > 0

