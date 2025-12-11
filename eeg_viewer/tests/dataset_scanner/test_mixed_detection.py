"""Tests for MIXED dataset type detection and warnings."""

import pytest
from pathlib import Path
import numpy as np
from PIL import Image
import json

from app.core.dataset_scanner import DatasetScanner
from app.core.dataset_scanner.models import DatasetType, SplitType


class TestMixedDetection:
    """Test MIXED dataset type detection."""
    
    def test_single_type_not_mixed(self, tmp_path):
        """Test that a single-type dataset is not marked as MIXED."""
        # Create only images
        img_dir = tmp_path / 'images'
        img_dir.mkdir()
        for i in range(50):
            img = Image.new('RGB', (64, 64), color=(i % 256, 0, 0))
            img.save(img_dir / f'image_{i:03d}.png')
        
        scanner = DatasetScanner()
        result = scanner.scan(tmp_path, deep_scan=True)
        
        assert result.primary_type == DatasetType.IMAGE
        assert result.primary_type != DatasetType.MIXED
    
    def test_images_with_few_text_files_not_mixed(self, tmp_path):
        """Test that images with README files are not marked as MIXED."""
        # Create many images
        for i in range(50):
            img = Image.new('RGB', (64, 64), color=(i % 256, 0, 0))
            img.save(tmp_path / f'image_{i:03d}.png')
        
        # Create few text files (typical for README, LICENSE, etc.)
        (tmp_path / 'README.txt').write_text('This is a readme')
        (tmp_path / 'LICENSE').write_text('MIT License')
        
        scanner = DatasetScanner()
        result = scanner.scan(tmp_path, deep_scan=True)
        
        # Should be IMAGE, not MIXED (text files are just metadata)
        assert result.primary_type == DatasetType.IMAGE
    
    def test_significant_mixed_types_detected(self, tmp_path):
        """Test that significant mix of types is detected as MIXED."""
        # Create significant number of images
        img_dir = tmp_path / 'images'
        img_dir.mkdir()
        for i in range(30):
            img = Image.new('RGB', (64, 64), color=(i % 256, 0, 0))
            img.save(img_dir / f'image_{i:03d}.png')
        
        # Create significant number of CSV files (tabular)
        csv_dir = tmp_path / 'tables'
        csv_dir.mkdir()
        for i in range(20):
            content = 'a,b,c\n' + '\n'.join([f'{j},{j+1},{j+2}' for j in range(10)])
            (csv_dir / f'data_{i:03d}.csv').write_text(content)
        
        scanner = DatasetScanner()
        result = scanner.scan(tmp_path, deep_scan=True)
        
        # Should detect MIXED because both types are significant
        assert result.primary_type == DatasetType.MIXED
    
    def test_mixed_generates_warning(self, tmp_path):
        """Test that mixed dataset generates appropriate warning."""
        # Create mix of types
        for i in range(20):
            img = Image.new('RGB', (64, 64), color=(i % 256, 0, 0))
            img.save(tmp_path / f'image_{i:03d}.png')
        
        for i in range(15):
            content = 'a,b,c\n' + '\n'.join([f'{j},{j+1},{j+2}' for j in range(10)])
            (tmp_path / f'data_{i:03d}.csv').write_text(content)
        
        scanner = DatasetScanner()
        result = scanner.scan(tmp_path, deep_scan=True)
        
        # Should have warning about multiple types
        assert any('multiple' in w.lower() or 'type' in w.lower() for w in result.warnings)
    
    def test_mixed_with_three_types(self, tmp_path):
        """Test detection with three significant types."""
        # Images
        img_dir = tmp_path / 'images'
        img_dir.mkdir()
        for i in range(20):
            img = Image.new('RGB', (64, 64), color=(i % 256, 0, 0))
            img.save(img_dir / f'image_{i:03d}.png')
        
        # CSV
        csv_dir = tmp_path / 'data'
        csv_dir.mkdir()
        for i in range(15):
            content = 'a,b,c\n' + '\n'.join([f'{j},{j+1},{j+2}' for j in range(10)])
            (csv_dir / f'data_{i:03d}.csv').write_text(content)
        
        # Text
        text_dir = tmp_path / 'text'
        text_dir.mkdir()
        for i in range(15):
            (text_dir / f'doc_{i:03d}.txt').write_text(f'Document {i} content ' * 50)
        
        scanner = DatasetScanner()
        result = scanner.scan(tmp_path, deep_scan=True)
        
        assert result.primary_type == DatasetType.MIXED


class TestWarnings:
    """Test comprehensive warnings."""
    
    def test_class_imbalance_warning(self, tmp_path):
        """Test warning for class imbalance."""
        # Create imbalanced classes
        class_a = tmp_path / 'class_a'
        class_b = tmp_path / 'class_b'
        class_a.mkdir()
        class_b.mkdir()
        
        # Class A: 100 images
        for i in range(100):
            img = Image.new('RGB', (32, 32))
            img.save(class_a / f'img_{i:03d}.png')
        
        # Class B: only 10 images (10x imbalance)
        for i in range(10):
            img = Image.new('RGB', (32, 32))
            img.save(class_b / f'img_{i:03d}.png')
        
        scanner = DatasetScanner()
        result = scanner.scan(tmp_path, deep_scan=True)
        
        assert any('imbalance' in w.lower() for w in result.warnings)
    
    def test_small_dataset_warning(self, tmp_path):
        """Test warning for small datasets."""
        # Create very few files
        for i in range(5):
            img = Image.new('RGB', (32, 32))
            img.save(tmp_path / f'img_{i:03d}.png')
        
        scanner = DatasetScanner()
        result = scanner.scan(tmp_path, deep_scan=True)
        
        assert any('small' in w.lower() for w in result.warnings)
    
    def test_no_splits_warning(self, tmp_path):
        """Test warning when no train/test splits exist."""
        # Create flat dataset with many files
        for i in range(150):
            img = Image.new('RGB', (32, 32))
            img.save(tmp_path / f'img_{i:03d}.png')
        
        scanner = DatasetScanner()
        result = scanner.scan(tmp_path, deep_scan=True)
        
        assert any('split' in w.lower() for w in result.warnings)
    
    def test_no_warning_with_proper_splits(self, tmp_path):
        """Test no warning when proper splits exist."""
        # Create train/test splits
        train_dir = tmp_path / 'train'
        test_dir = tmp_path / 'test'
        train_dir.mkdir()
        test_dir.mkdir()
        
        for i in range(100):
            img = Image.new('RGB', (32, 32))
            img.save(train_dir / f'img_{i:03d}.png')
        
        for i in range(30):
            img = Image.new('RGB', (32, 32))
            img.save(test_dir / f'img_{i:03d}.png')
        
        scanner = DatasetScanner()
        result = scanner.scan(tmp_path, deep_scan=True)
        
        # Should not warn about splits (has train/test structure)
        assert not any('no train' in w.lower() for w in result.warnings)


class TestMixedThresholds:
    """Test MIXED detection thresholds."""
    
    def test_below_min_files_threshold(self, tmp_path):
        """Test that types below min files threshold don't trigger MIXED."""
        # Create many images
        for i in range(100):
            img = Image.new('RGB', (64, 64), color=(i % 256, 0, 0))
            img.save(tmp_path / f'image_{i:03d}.png')
        
        # Create few CSV (below MIXED_MIN_FILES=10)
        for i in range(5):
            (tmp_path / f'data_{i}.csv').write_text('a,b,c\n1,2,3')
        
        scanner = DatasetScanner()
        result = scanner.scan(tmp_path, deep_scan=True)
        
        # Should NOT be mixed (csv count < 10)
        assert result.primary_type == DatasetType.IMAGE
    
    def test_above_min_files_threshold(self, tmp_path):
        """Test that types above threshold can trigger MIXED."""
        # Create images
        for i in range(30):
            img = Image.new('RGB', (64, 64), color=(i % 256, 0, 0))
            img.save(tmp_path / f'image_{i:03d}.png')
        
        # Create CSV (above MIXED_MIN_FILES=10)
        for i in range(15):
            (tmp_path / f'data_{i}.csv').write_text('a,b,c\n1,2,3\n4,5,6')
        
        scanner = DatasetScanner()
        result = scanner.scan(tmp_path, deep_scan=True)
        
        # Could be MIXED depending on score ratio
        assert result.primary_type in [DatasetType.IMAGE, DatasetType.MIXED]


class TestMixedWithSubsetTypes:
    """Test that artifact files don't trigger MIXED."""
    
    def test_image_dataset_with_metadata_csv(self, tmp_path):
        """Test that a CSV metadata file doesn't make dataset MIXED."""
        # Many images
        img_dir = tmp_path / 'images'
        img_dir.mkdir()
        for i in range(100):
            img = Image.new('RGB', (64, 64))
            img.save(img_dir / f'image_{i:03d}.png')
        
        # One metadata CSV
        (tmp_path / 'labels.csv').write_text('filename,label\n' + 
            '\n'.join([f'image_{i:03d}.png,class_{i%5}' for i in range(100)]))
        
        scanner = DatasetScanner()
        result = scanner.scan(tmp_path, deep_scan=True)
        
        # Should be IMAGE (CSV is just metadata, not separate dataset)
        assert result.primary_type == DatasetType.IMAGE
    
    def test_graph_dataset_with_edge_list_csv(self, tmp_path):
        """Test that edge list CSVs don't make graph dataset MIXED."""
        # Create phases files
        import pickle
        
        phases_data = {
            'phases_eeg': {'Alpha': [np.random.rand(24, 100)]},
            'syncros_eeg': {'Alpha': [np.random.rand(24, 24)]},
        }
        
        for cond in ['DMT', 'EC', 'EO']:
            cond_dir = tmp_path / cond
            cond_dir.mkdir()
            for i in range(5):
                with open(cond_dir / f'phases-S{i:02d}.pkl', 'wb') as f:
                    pickle.dump(phases_data, f)
        
        # Add edge list CSV (artifact of graph processing)
        (tmp_path / 'edge_list.csv').write_text('source,target,weight\n0,1,0.5\n1,2,0.3')
        
        scanner = DatasetScanner()
        result = scanner.scan(tmp_path, deep_scan=True)
        
        # Should be GRAPH
        assert result.primary_type == DatasetType.GRAPH


class TestIntegrationWithModelPage:
    """Test MIXED detection integration with model page workflow."""
    
    def test_mixed_dataset_provides_type_info(self, tmp_path):
        """Test that MIXED dataset still provides useful type-specific info."""
        # Create mix
        for i in range(20):
            img = Image.new('RGB', (64, 64))
            img.save(tmp_path / f'image_{i:03d}.png')
        
        for i in range(15):
            (tmp_path / f'data_{i}.csv').write_text('a,b,c\n1,2,3')
        
        scanner = DatasetScanner()
        result = scanner.scan(tmp_path, deep_scan=True)
        
        # Should still have file info
        assert result.files.total_count > 0
        assert result.files.by_type  # Should have type breakdown
    
    def test_mixed_provides_suggestions(self, tmp_path):
        """Test that MIXED dataset provides helpful suggestions."""
        for i in range(20):
            img = Image.new('RGB', (64, 64))
            img.save(tmp_path / f'image_{i:03d}.png')
        
        for i in range(15):
            (tmp_path / f'data_{i}.csv').write_text('a,b,c\n1,2,3')
        
        scanner = DatasetScanner()
        result = scanner.scan(tmp_path, deep_scan=True)
        
        # Should have suggestions
        assert len(result.suggestions) > 0


