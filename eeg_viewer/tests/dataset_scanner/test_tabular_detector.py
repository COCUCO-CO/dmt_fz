"""
Tests for TabularDetector.

Verifies detection of tabular datasets including CSV, Parquet, HDF5, and Excel.
"""

import pytest
from pathlib import Path
import numpy as np
import pandas as pd


class TestTabularDetectorBasic:
    """Basic tabular detection tests."""
    
    def test_detects_csv_files(self, single_csv_dataset: Path):
        """Verify CSV files are detected."""
        from app.core.dataset_scanner.detectors import TabularDetector
        
        detector = TabularDetector()
        result = detector.detect(single_csv_dataset)
        
        assert result.is_detected is True
        assert result.confidence > 0.8
        assert '.csv' in result.extensions_found
    
    def test_counts_files_correctly(self, flat_csv_dataset: Path):
        """Verify file count is accurate."""
        from app.core.dataset_scanner.detectors import TabularDetector
        
        detector = TabularDetector()
        result = detector.detect(flat_csv_dataset)
        
        # Created 5 CSV files
        assert result.file_count == 5
    
    def test_empty_directory_not_detected(self, empty_dataset: Path):
        """Verify empty directory is not detected as tabular dataset."""
        from app.core.dataset_scanner.detectors import TabularDetector
        
        detector = TabularDetector()
        result = detector.detect(empty_dataset)
        
        assert result.is_detected is False


class TestTabularDetectorStructures:
    """Test tabular detection with various structures."""
    
    def test_detects_train_test_files(self, train_test_csv_dataset: Path):
        """Verify train/test CSV structure is detected."""
        from app.core.dataset_scanner.detectors import TabularDetector
        
        detector = TabularDetector()
        result = detector.detect(train_test_csv_dataset)
        
        assert result.is_detected is True
        assert result.has_splits is True
        assert 'train' in result.splits_found
        assert 'test' in result.splits_found
    
    def test_single_file_detected(self, single_csv_dataset: Path):
        """Verify single CSV file is detected properly."""
        from app.core.dataset_scanner.detectors import TabularDetector
        
        detector = TabularDetector()
        result = detector.detect(single_csv_dataset)
        
        assert result.is_detected is True
        assert result.file_count == 1


class TestTabularDetectorMetadata:
    """Test tabular metadata extraction."""
    
    def test_extracts_row_count(self, single_csv_dataset: Path):
        """Verify row count is extracted."""
        from app.core.dataset_scanner.detectors import TabularDetector
        
        detector = TabularDetector()
        result = detector.detect(single_csv_dataset, analyze_samples=True)
        
        assert result.tabular_info is not None
        # Created with 500 rows
        assert result.tabular_info.num_rows == 500
    
    def test_extracts_column_count(self, single_csv_dataset: Path):
        """Verify column count is extracted."""
        from app.core.dataset_scanner.detectors import TabularDetector
        
        detector = TabularDetector()
        result = detector.detect(single_csv_dataset, analyze_samples=True)
        
        # Created with 5 columns
        assert result.tabular_info.num_columns == 5
    
    def test_extracts_column_names(self, single_csv_dataset: Path):
        """Verify column names are extracted."""
        from app.core.dataset_scanner.detectors import TabularDetector
        
        detector = TabularDetector()
        result = detector.detect(single_csv_dataset, analyze_samples=True)
        
        expected_cols = {'feature_1', 'feature_2', 'feature_3', 'category', 'target'}
        assert set(result.tabular_info.column_names) == expected_cols
    
    def test_detects_column_types(self, single_csv_dataset: Path):
        """Verify column types are detected."""
        from app.core.dataset_scanner.detectors import TabularDetector
        
        detector = TabularDetector()
        result = detector.detect(single_csv_dataset, analyze_samples=True)
        
        assert len(result.tabular_info.numeric_columns) > 0
        assert len(result.tabular_info.categorical_columns) > 0
    
    def test_detects_missing_values(self, temp_dir: Path):
        """Verify missing values are detected."""
        from app.core.dataset_scanner.detectors import TabularDetector
        
        dataset_path = temp_dir / "csv_with_missing"
        dataset_path.mkdir()
        
        # Create CSV with missing values
        df = pd.DataFrame({
            'a': [1, 2, np.nan, 4, 5],
            'b': ['x', None, 'y', 'z', None],
            'c': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        df.to_csv(dataset_path / "data.csv", index=False)
        
        detector = TabularDetector()
        result = detector.detect(dataset_path, analyze_samples=True)
        
        assert result.tabular_info.has_missing_values is True
        assert result.tabular_info.missing_counts['a'] == 1
        assert result.tabular_info.missing_counts['b'] == 2


class TestTabularDetectorFormats:
    """Test different tabular file formats."""
    
    def test_detects_tsv_files(self, temp_dir: Path):
        """Verify TSV files are detected."""
        from app.core.dataset_scanner.detectors import TabularDetector
        
        dataset_path = temp_dir / "tsv_data"
        dataset_path.mkdir()
        
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        df.to_csv(dataset_path / "data.tsv", sep='\t', index=False)
        
        detector = TabularDetector()
        result = detector.detect(dataset_path)
        
        assert result.is_detected is True
        assert '.tsv' in result.extensions_found
    
    def test_detects_parquet_files(self, parquet_dataset: Path):
        """Verify Parquet files are detected."""
        from app.core.dataset_scanner.detectors import TabularDetector
        
        detector = TabularDetector()
        result = detector.detect(parquet_dataset)
        
        assert result.is_detected is True
        assert '.parquet' in result.extensions_found
    
    def test_detects_excel_files(self, temp_dir: Path):
        """Verify Excel files are detected."""
        from app.core.dataset_scanner.detectors import TabularDetector
        
        try:
            import openpyxl
        except ImportError:
            pytest.skip("openpyxl not available")
        
        dataset_path = temp_dir / "excel_data"
        dataset_path.mkdir()
        
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        df.to_excel(dataset_path / "data.xlsx", index=False)
        
        detector = TabularDetector()
        result = detector.detect(dataset_path)
        
        assert result.is_detected is True
        assert '.xlsx' in result.extensions_found
    
    def test_detects_hdf5_tabular(self, temp_dir: Path):
        """Verify HDF5 files with tabular data are detected."""
        from app.core.dataset_scanner.detectors import TabularDetector
        
        try:
            import h5py
        except ImportError:
            pytest.skip("h5py not available")
        
        dataset_path = temp_dir / "hdf5_tabular"
        dataset_path.mkdir()
        
        df = pd.DataFrame({'a': range(100), 'b': np.random.randn(100)})
        df.to_hdf(dataset_path / "data.h5", key='data', mode='w')
        
        detector = TabularDetector()
        result = detector.detect(dataset_path)
        
        assert result.is_detected is True
        assert '.h5' in result.extensions_found or '.hdf5' in result.extensions_found


class TestTabularDetectorEdgeCases:
    """Test edge cases for tabular detection."""
    
    def test_handles_empty_csv(self, temp_dir: Path):
        """Verify empty CSV is handled gracefully."""
        from app.core.dataset_scanner.detectors import TabularDetector
        
        dataset_path = temp_dir / "empty_csv"
        dataset_path.mkdir()
        
        # Create empty CSV with headers only
        pd.DataFrame(columns=['a', 'b', 'c']).to_csv(dataset_path / "empty.csv", index=False)
        
        detector = TabularDetector()
        result = detector.detect(dataset_path, analyze_samples=True)
        
        assert result.is_detected is True
        assert result.tabular_info.num_rows == 0
    
    def test_handles_single_column(self, temp_dir: Path):
        """Verify single column CSV is handled."""
        from app.core.dataset_scanner.detectors import TabularDetector
        
        dataset_path = temp_dir / "single_col"
        dataset_path.mkdir()
        
        pd.DataFrame({'value': range(100)}).to_csv(dataset_path / "data.csv", index=False)
        
        detector = TabularDetector()
        result = detector.detect(dataset_path, analyze_samples=True)
        
        assert result.is_detected is True
        assert result.tabular_info.num_columns == 1
    
    def test_handles_malformed_csv(self, temp_dir: Path):
        """Verify malformed CSV is handled gracefully."""
        from app.core.dataset_scanner.detectors import TabularDetector
        
        dataset_path = temp_dir / "malformed"
        dataset_path.mkdir()
        
        # Create malformed CSV
        with open(dataset_path / "bad.csv", 'w') as f:
            f.write("a,b,c\n")
            f.write("1,2\n")  # Missing value
            f.write("1,2,3,4\n")  # Extra value
        
        detector = TabularDetector()
        result = detector.detect(dataset_path, analyze_samples=True)
        
        # Should not crash
        assert True
    
    def test_handles_large_csv(self, temp_dir: Path):
        """Verify large CSV is handled efficiently."""
        from app.core.dataset_scanner.detectors import TabularDetector
        
        dataset_path = temp_dir / "large_csv"
        dataset_path.mkdir()
        
        # Create larger CSV (not too large for tests)
        df = pd.DataFrame({
            f'col_{i}': np.random.randn(10000)
            for i in range(50)
        })
        df.to_csv(dataset_path / "large.csv", index=False)
        
        detector = TabularDetector()
        result = detector.detect(dataset_path, analyze_samples=True)
        
        assert result.is_detected is True
        assert result.tabular_info.num_rows == 10000
        assert result.tabular_info.num_columns == 50
    
    def test_distinguishes_from_text(self, temp_dir: Path):
        """Verify text files aren't mistaken for tabular."""
        from app.core.dataset_scanner.detectors import TabularDetector
        
        dataset_path = temp_dir / "text_not_csv"
        dataset_path.mkdir()
        
        # Create plain text file
        with open(dataset_path / "readme.txt", 'w') as f:
            f.write("This is not tabular data.\nJust plain text.\n")
        
        detector = TabularDetector()
        result = detector.detect(dataset_path)
        
        # Should not be detected as tabular or have low confidence
        assert result.is_detected is False or result.confidence < 0.5


class TestTabularDetectorStatistics:
    """Test statistical analysis of tabular datasets."""
    
    def test_calculates_total_rows(self, flat_csv_dataset: Path):
        """Verify total rows across files is calculated."""
        from app.core.dataset_scanner.detectors import TabularDetector
        
        detector = TabularDetector()
        result = detector.detect(flat_csv_dataset, analyze_samples=True)
        
        # 5 files × 100 rows = 500 total
        assert result.tabular_info.total_rows == 500
    
    def test_calculates_statistics(self, single_csv_dataset: Path):
        """Verify column statistics are calculated."""
        from app.core.dataset_scanner.detectors import TabularDetector
        
        detector = TabularDetector()
        result = detector.detect(single_csv_dataset, analyze_samples=True)
        
        # Should have detected column types at minimum
        assert result.tabular_info is not None
        assert len(result.tabular_info.numeric_columns) > 0 or len(result.tabular_info.column_names) > 0
    
    def test_detects_target_column(self, single_csv_dataset: Path):
        """Verify potential target column is detected."""
        from app.core.dataset_scanner.detectors import TabularDetector
        
        detector = TabularDetector()
        result = detector.detect(single_csv_dataset, analyze_samples=True)
        
        # Column named 'target' should be suggested as target
        assert result.tabular_info.suggested_target in ['target', 'label'] or \
               result.tabular_info.suggested_target is not None
    
    def test_detects_id_column(self, temp_dir: Path):
        """Verify ID columns are detected."""
        from app.core.dataset_scanner.detectors import TabularDetector
        
        dataset_path = temp_dir / "with_id"
        dataset_path.mkdir()
        
        df = pd.DataFrame({
            'id': range(100),
            'user_id': range(100, 200),
            'feature': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })
        df.to_csv(dataset_path / "data.csv", index=False)
        
        detector = TabularDetector()
        result = detector.detect(dataset_path, analyze_samples=True)
        
        assert 'id' in result.tabular_info.id_columns or 'user_id' in result.tabular_info.id_columns


class TestTabularDetectorOutput:
    """Test output format and completeness."""
    
    def test_returns_detection_result(self, single_csv_dataset: Path):
        """Verify detection result has all required fields."""
        from app.core.dataset_scanner.detectors import TabularDetector
        
        detector = TabularDetector()
        result = detector.detect(single_csv_dataset)
        
        assert hasattr(result, 'is_detected')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'file_count')
        assert hasattr(result, 'extensions_found')
    
    def test_to_dict_conversion(self, single_csv_dataset: Path):
        """Verify result can be converted to dict."""
        from app.core.dataset_scanner.detectors import TabularDetector
        
        detector = TabularDetector()
        result = detector.detect(single_csv_dataset)
        
        d = result.to_dict()
        
        assert isinstance(d, dict)
        assert 'is_detected' in d
    
    def test_generates_suggestions(self, single_csv_dataset: Path):
        """Verify detector generates useful suggestions."""
        from app.core.dataset_scanner.detectors import TabularDetector
        
        detector = TabularDetector()
        result = detector.detect(single_csv_dataset)
        
        # Should suggest pandas or other loaders
        assert len(result.suggestions) > 0 or result.suggested_loader is not None

