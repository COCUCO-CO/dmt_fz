"""
Tabular dataset detector.

Handles CSV, TSV, Parquet, HDF5, and Excel files.
"""

from pathlib import Path
from typing import Set, List, Dict
import csv

from .base import BaseDetector, DetectionResult
from ..models import TabularSpecificInfo


class TabularDetector(BaseDetector):
    """Detector for tabular datasets."""
    
    EXTENSIONS: Set[str] = {
        '.csv', '.tsv',
        '.parquet', '.pq',
        '.h5', '.hdf5', '.hdf',
        '.xlsx', '.xls',
        '.feather',
        '.json',  # Can be tabular
    }
    
    # Column names that suggest ID columns
    ID_COLUMN_PATTERNS = {'id', 'idx', 'index', 'user_id', 'sample_id', 'row_id'}
    
    # Column names that suggest target columns
    TARGET_COLUMN_PATTERNS = {'target', 'label', 'y', 'class', 'output', 'prediction'}
    
    def detect(self, path: Path, analyze_samples: bool = False,
               sample_size: int = 10) -> DetectionResult:
        """Detect tabular dataset."""
        result = DetectionResult()
        
        if not path.exists():
            result.warnings.append(f"Path does not exist: {path}")
            return result
        
        # Find all tabular files
        files = self._find_files(path, self.EXTENSIONS)
        
        if not files:
            return result
        
        # Filter out JSON files that aren't tabular
        tabular_files = []
        for f in files:
            if f.suffix.lower() == '.json':
                # Check if it looks tabular
                if self._is_tabular_json(f):
                    tabular_files.append(f)
            else:
                tabular_files.append(f)
        
        if not tabular_files:
            return result
        
        result.is_detected = True
        result.file_count = len(tabular_files)
        result.sample_files = tabular_files[:sample_size]
        result.total_size_bytes = self._calculate_total_size(tabular_files)
        
        extensions = set(f.suffix.lower() for f in tabular_files)
        result.extensions_found = sorted(extensions)
        
        # Calculate confidence
        if result.file_count >= 1 and '.csv' in extensions:
            result.confidence = 0.90
        elif result.file_count >= 1 and '.parquet' in extensions:
            result.confidence = 0.95
        else:
            result.confidence = 0.75
        
        # Detect train/test splits from filenames
        has_splits, splits, split_counts = self._detect_tabular_splits(tabular_files)
        result.has_splits = has_splits
        result.splits_found = splits
        result.split_counts = split_counts
        
        # Analyze samples
        if analyze_samples:
            result.tabular_info = self._analyze_tabular_files(tabular_files[:sample_size])
        
        result.suggestions = self._generate_suggestions(result)
        result.suggested_loader = self._suggest_loader(result)
        
        return result
    
    def _is_tabular_json(self, path: Path) -> bool:
        """Check if JSON file contains tabular data."""
        try:
            with open(path, 'r') as f:
                # Read first 1000 chars
                content = f.read(1000)
                # Check if it's an array of objects
                return content.strip().startswith('[{')
        except Exception:
            return False
    
    def _detect_tabular_splits(self, files: List[Path]) -> tuple:
        """Detect train/test splits from filenames."""
        splits_found = []
        split_counts = {}
        
        for f in files:
            name_lower = f.stem.lower()
            if 'train' in name_lower:
                splits_found.append('train')
                split_counts['train'] = split_counts.get('train', 0) + 1
            elif 'val' in name_lower or 'valid' in name_lower:
                splits_found.append('val')
                split_counts['val'] = split_counts.get('val', 0) + 1
            elif 'test' in name_lower:
                splits_found.append('test')
                split_counts['test'] = split_counts.get('test', 0) + 1
        
        has_splits = len(set(splits_found)) > 1
        return has_splits, list(set(splits_found)), split_counts
    
    def _analyze_tabular_files(self, files: List[Path]) -> TabularSpecificInfo:
        """Analyze tabular files."""
        info = TabularSpecificInfo()
        
        total_rows = 0
        all_columns = set()
        numeric_cols = set()
        categorical_cols = set()
        id_cols = set()
        missing_counts: Dict[str, int] = {}
        
        for f in files:
            try:
                rows, columns, num_cols, cat_cols, missing = self._analyze_single_file(f)
                total_rows += rows
                all_columns.update(columns)
                numeric_cols.update(num_cols)
                categorical_cols.update(cat_cols)
                for col, count in missing.items():
                    missing_counts[col] = missing_counts.get(col, 0) + count
            except Exception:
                continue
        
        info.total_rows = total_rows
        info.num_rows = total_rows
        info.num_columns = len(all_columns)
        info.column_names = sorted(all_columns)
        info.numeric_columns = sorted(numeric_cols)
        info.categorical_columns = sorted(categorical_cols)
        
        # Detect ID columns
        for col in all_columns:
            if col.lower() in self.ID_COLUMN_PATTERNS:
                id_cols.add(col)
        info.id_columns = sorted(id_cols)
        
        # Detect target column
        for col in all_columns:
            if col.lower() in self.TARGET_COLUMN_PATTERNS:
                info.suggested_target = col
                break
        
        # Missing values
        info.missing_counts = missing_counts
        info.has_missing_values = sum(missing_counts.values()) > 0
        
        return info
    
    def _analyze_single_file(self, path: Path) -> tuple:
        """Analyze a single tabular file."""
        suffix = path.suffix.lower()
        
        if suffix in {'.csv', '.tsv'}:
            return self._analyze_csv(path, delimiter=',' if suffix == '.csv' else '\t')
        elif suffix in {'.parquet', '.pq'}:
            return self._analyze_parquet(path)
        elif suffix in {'.h5', '.hdf5', '.hdf'}:
            return self._analyze_hdf5(path)
        elif suffix in {'.xlsx', '.xls'}:
            return self._analyze_excel(path)
        
        return 0, [], [], [], {}
    
    def _analyze_csv(self, path: Path, delimiter: str = ',') -> tuple:
        """Analyze CSV/TSV file."""
        import pandas as pd
        
        try:
            # Read with pandas for proper type inference
            df = pd.read_csv(path, delimiter=delimiter, nrows=1000)
            
            columns = list(df.columns)
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Count missing values
            missing = df.isnull().sum().to_dict()
            missing = {k: v for k, v in missing.items() if v > 0}
            
            # Get actual row count
            row_count = len(df)
            if row_count == 1000:
                # File might be larger, count lines
                with open(path, 'r') as f:
                    row_count = sum(1 for _ in f) - 1  # Subtract header
            
            return row_count, columns, numeric_cols, categorical_cols, missing
            
        except Exception:
            # Fallback to basic CSV parsing
            try:
                with open(path, 'r', newline='') as f:
                    reader = csv.reader(f, delimiter=delimiter)
                    header = next(reader)
                    row_count = sum(1 for _ in reader)
                return row_count, header, [], [], {}
            except Exception:
                return 0, [], [], [], {}
    
    def _analyze_parquet(self, path: Path) -> tuple:
        """Analyze Parquet file."""
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            
            columns = list(df.columns)
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            missing = df.isnull().sum().to_dict()
            missing = {k: v for k, v in missing.items() if v > 0}
            
            return len(df), columns, numeric_cols, categorical_cols, missing
        except Exception:
            return 0, [], [], [], {}
    
    def _analyze_hdf5(self, path: Path) -> tuple:
        """Analyze HDF5 file."""
        try:
            import pandas as pd
            # Try to read as pandas HDF
            df = pd.read_hdf(path)
            
            columns = list(df.columns)
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            missing = df.isnull().sum().to_dict()
            missing = {k: v for k, v in missing.items() if v > 0}
            
            return len(df), columns, numeric_cols, categorical_cols, missing
        except Exception:
            # Fallback to h5py
            try:
                import h5py
                with h5py.File(path, 'r') as f:
                    keys = list(f.keys())
                    return 0, keys, [], [], {}
            except Exception:
                return 0, [], [], [], {}
    
    def _analyze_excel(self, path: Path) -> tuple:
        """Analyze Excel file."""
        try:
            import pandas as pd
            df = pd.read_excel(path, nrows=1000)
            
            columns = list(df.columns)
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            missing = df.isnull().sum().to_dict()
            missing = {k: v for k, v in missing.items() if v > 0}
            
            return len(df), columns, numeric_cols, categorical_cols, missing
        except Exception:
            return 0, [], [], [], {}
    
    def _generate_suggestions(self, result: DetectionResult) -> List[str]:
        """Generate suggestions."""
        suggestions = []
        
        if '.csv' in result.extensions_found:
            suggestions.append("Use pandas.read_csv for loading")
        if '.parquet' in result.extensions_found:
            suggestions.append("Use pandas.read_parquet for faster loading")
        
        if result.tabular_info:
            if result.tabular_info.has_missing_values:
                suggestions.append("Dataset has missing values - consider imputation")
            if result.tabular_info.suggested_target:
                suggestions.append(f"Detected potential target column: '{result.tabular_info.suggested_target}'")
        
        suggestions.append("Consider using sklearn for preprocessing and modeling")
        
        return suggestions
    
    def _suggest_loader(self, result: DetectionResult) -> str:
        """Suggest data loader."""
        if '.parquet' in result.extensions_found:
            return "pandas.read_parquet"
        return "pandas.read_csv"








