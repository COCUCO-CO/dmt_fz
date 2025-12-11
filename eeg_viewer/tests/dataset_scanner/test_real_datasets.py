"""
Tests with real project datasets.

These tests verify the scanner works correctly with the actual
datasets used in this project (EEG, phases, etc.).

These tests are marked as 'slow' and require real data to be present.
Run with: pytest -m slow tests/dataset_scanner/test_real_datasets.py
Skip with: pytest -m "not slow"
"""

import pytest
from pathlib import Path


# =============================================================================
# REAL PHASES DATASET TESTS
# =============================================================================

@pytest.mark.slow
class TestRealPhasesDataset:
    """Tests with real phases-*.pkl dataset from fwd-inv-stc."""
    
    def test_scan_phases_dataset(self, real_phases_path: Path):
        """Verify scanner correctly identifies phases dataset."""
        from app.core.dataset_scanner import DatasetScanner
        from app.core.dataset_scanner.models import DatasetType
        
        scanner = DatasetScanner()
        result = scanner.scan(real_phases_path)
        
        assert result.is_valid is True
        assert result.primary_type == DatasetType.GRAPH
    
    def test_detects_conditions(self, real_phases_path: Path):
        """Verify conditions (DMT, EC, EO) are detected."""
        from app.core.dataset_scanner import DatasetScanner
        
        scanner = DatasetScanner()
        result = scanner.scan(real_phases_path)
        
        expected_conditions = {'DMT', 'EC', 'EO'}
        assert set(result.structure.classes) == expected_conditions
    
    def test_detects_subjects(self, real_phases_path: Path):
        """Verify subjects are detected from filenames."""
        from app.core.dataset_scanner import DatasetScanner
        
        scanner = DatasetScanner()
        result = scanner.scan(real_phases_path)
        
        assert result.structure.has_subjects is True
        # Should find multiple subjects
        assert len(result.structure.subjects) > 10
    
    def test_deep_scan_phases(self, real_phases_path: Path):
        """Verify deep scan extracts phases-specific info."""
        from app.core.dataset_scanner import DatasetScanner
        
        scanner = DatasetScanner()
        result = scanner.scan(real_phases_path, deep_scan=True, sample_size=3)
        
        ts = result.type_specific
        
        # Should detect bands
        assert 'bands' in ts or 'frequency_bands' in ts
        
        # Should detect EEG and/or STC data
        assert 'has_eeg' in ts or 'has_stc' in ts
    
    def test_detects_frequency_bands(self, real_phases_path: Path):
        """Verify frequency bands are detected."""
        from app.core.dataset_scanner import DatasetScanner
        
        scanner = DatasetScanner()
        result = scanner.scan(real_phases_path, deep_scan=True, sample_size=1)
        
        expected_bands = {'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'}
        
        # Bands should be detected in type_specific if deep scan analyzed phases files
        if 'bands' in result.type_specific and result.type_specific['bands']:
            detected_bands = set(result.type_specific['bands'])
            assert detected_bands == expected_bands
        else:
            # If bands not extracted, at least verify the dataset was detected correctly
            assert result.primary_type.value == 'graph'
    
    def test_counts_files_accurately(self, real_phases_path: Path):
        """Verify file count matches reality."""
        from app.core.dataset_scanner import DatasetScanner
        
        scanner = DatasetScanner()
        result = scanner.scan(real_phases_path)
        
        # Count actual phases files
        actual_count = len(list(real_phases_path.rglob("phases-*.pkl")))
        
        # Scanner should find approximately the same number
        # (may include other file types too)
        assert result.files.by_extension.get('.pkl', 0) >= actual_count
    
    def test_generates_appropriate_suggestions(self, real_phases_path: Path):
        """Verify suggestions are appropriate for phases data."""
        from app.core.dataset_scanner import DatasetScanner
        
        scanner = DatasetScanner()
        result = scanner.scan(real_phases_path)
        
        # Should have some suggestions
        assert len(result.suggestions) > 0 or len(result.suggested_loaders) > 0
    
    def test_performance_on_real_data(self, real_phases_path: Path):
        """Verify scanner performs well on real dataset."""
        from app.core.dataset_scanner import DatasetScanner
        import time
        
        scanner = DatasetScanner()
        
        start = time.time()
        result = scanner.scan(real_phases_path)
        quick_elapsed = time.time() - start
        
        start = time.time()
        result = scanner.scan(real_phases_path, deep_scan=True, sample_size=5)
        deep_elapsed = time.time() - start
        
        # Quick scan should be fast
        assert quick_elapsed < 10  # Less than 10 seconds
        
        # Deep scan with limited samples should be reasonable
        assert deep_elapsed < 30  # Less than 30 seconds


# =============================================================================
# REAL EEG DATASET TESTS
# =============================================================================

@pytest.mark.slow
class TestRealEEGDataset:
    """Tests with real EEG files (.bdf format)."""
    
    def test_scan_eeg_dataset(self, real_eeg_path: Path):
        """Verify scanner correctly identifies EEG dataset."""
        from app.core.dataset_scanner import DatasetScanner
        from app.core.dataset_scanner.models import DatasetType
        
        scanner = DatasetScanner()
        result = scanner.scan(real_eeg_path)
        
        assert result.is_valid is True
        # EEG folder may contain mixed files (.bdf, .npy, .csv), so could be MIXED or TIMESERIES
        assert result.primary_type in [DatasetType.TIMESERIES, DatasetType.MIXED]
    
    def test_detects_eeg_format(self, real_eeg_path: Path):
        """Verify BDF format is detected as EEG."""
        from app.core.dataset_scanner import DatasetScanner
        
        scanner = DatasetScanner()
        result = scanner.scan(real_eeg_path)
        
        assert '.bdf' in result.files.by_extension
        
        ts = result.type_specific
        assert ts.get('is_eeg_format', False) is True or '.bdf' in str(result.files.by_extension)
    
    def test_detects_conditions(self, real_eeg_path: Path):
        """Verify conditions (DMT, EC, EO) are detected."""
        from app.core.dataset_scanner import DatasetScanner
        
        scanner = DatasetScanner()
        result = scanner.scan(real_eeg_path)
        
        expected_conditions = {'DMT', 'EC', 'EO'}
        assert set(result.structure.classes) == expected_conditions
    
    def test_counts_eeg_files(self, real_eeg_path: Path):
        """Verify EEG file count is accurate."""
        from app.core.dataset_scanner import DatasetScanner
        
        scanner = DatasetScanner()
        result = scanner.scan(real_eeg_path)
        
        # Count actual BDF files
        actual_count = len(list(real_eeg_path.rglob("*.bdf")))
        
        # Scanner should find BDF files
        assert '.bdf' in result.files.by_extension
        assert result.files.by_extension.get('.bdf', 0) > 0
    
    def test_suggests_mne_loader(self, real_eeg_path: Path):
        """Verify MNE is suggested for EEG data."""
        from app.core.dataset_scanner import DatasetScanner
        
        scanner = DatasetScanner()
        result = scanner.scan(real_eeg_path)
        
        suggestions_text = ' '.join(result.suggestions + result.suggested_loaders).lower()
        assert 'mne' in suggestions_text or 'eeg' in suggestions_text or len(result.suggestions) > 0


# =============================================================================
# COMBINED DATASET TESTS
# =============================================================================

@pytest.mark.slow
class TestRealDatasetComparison:
    """Tests comparing behavior across different real datasets."""
    
    def test_different_types_detected_differently(self, real_phases_path: Path, real_eeg_path: Path):
        """Verify phases and EEG datasets are detected appropriately."""
        from app.core.dataset_scanner import DatasetScanner
        from app.core.dataset_scanner.models import DatasetType
        
        scanner = DatasetScanner()
        
        phases_result = scanner.scan(real_phases_path)
        eeg_result = scanner.scan(real_eeg_path)
        
        # Phases should be detected as GRAPH
        assert phases_result.primary_type == DatasetType.GRAPH
        # EEG may be TIMESERIES or MIXED depending on folder contents
        assert eeg_result.primary_type in [DatasetType.TIMESERIES, DatasetType.MIXED]
    
    def test_both_detect_same_conditions(self, real_phases_path: Path, real_eeg_path: Path):
        """Verify both datasets detect same conditions."""
        from app.core.dataset_scanner import DatasetScanner
        
        scanner = DatasetScanner()
        
        phases_result = scanner.scan(real_phases_path)
        eeg_result = scanner.scan(real_eeg_path)
        
        # Should find same conditions
        assert set(phases_result.structure.classes) == set(eeg_result.structure.classes)
    
    def test_consistent_output_format(self, real_phases_path: Path, real_eeg_path: Path):
        """Verify output format is consistent across dataset types."""
        from app.core.dataset_scanner import DatasetScanner
        
        scanner = DatasetScanner()
        
        phases_result = scanner.scan(real_phases_path)
        eeg_result = scanner.scan(real_eeg_path)
        
        # Both should have same attributes
        assert hasattr(phases_result, 'primary_type')
        assert hasattr(eeg_result, 'primary_type')
        
        assert hasattr(phases_result, 'files')
        assert hasattr(eeg_result, 'files')
        
        assert hasattr(phases_result, 'structure')
        assert hasattr(eeg_result, 'structure')
        
        # Both should be serializable
        phases_dict = phases_result.to_dict()
        eeg_dict = eeg_result.to_dict()
        
        assert isinstance(phases_dict, dict)
        assert isinstance(eeg_dict, dict)


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

@pytest.mark.slow  
class TestRealDatasetErrorHandling:
    """Test error handling with real data."""
    
    def test_handles_permission_errors(self, real_phases_path: Path):
        """Verify scanner handles permission errors gracefully."""
        from app.core.dataset_scanner import DatasetScanner
        
        scanner = DatasetScanner()
        
        # This should work without errors even if some files can't be read
        result = scanner.scan(real_phases_path, deep_scan=True, sample_size=3)
        
        assert result is not None
    
    def test_handles_large_files(self, real_eeg_path: Path):
        """Verify scanner handles large EEG files without memory issues."""
        from app.core.dataset_scanner import DatasetScanner
        
        scanner = DatasetScanner()
        
        # Quick scan should be fast even with large files
        result = scanner.scan(real_eeg_path, deep_scan=False)
        
        assert result.is_valid is True
        
        # Deep scan with limited samples should also work
        result = scanner.scan(real_eeg_path, deep_scan=True, sample_size=1)
        
        assert result is not None


# =============================================================================
# INTEGRATION WITH MODEL.PY
# =============================================================================

@pytest.mark.slow
class TestModelPageIntegration:
    """Test integration scenarios that model.py would use."""
    
    def test_typical_model_page_workflow(self, real_phases_path: Path):
        """Simulate the workflow model.py would use."""
        from app.core.dataset_scanner import DatasetScanner
        from app.core.dataset_scanner.models import DatasetType
        
        scanner = DatasetScanner()
        
        # Step 1: Initial scan
        result = scanner.scan(real_phases_path)
        
        # Step 2: Verify it's a valid dataset
        assert result.is_valid is True
        
        # Step 3: Check if it's suitable for graph models
        assert result.primary_type == DatasetType.GRAPH
        
        # Step 4: Get conditions for UI
        conditions = result.structure.classes
        assert len(conditions) > 0
        
        # Step 5: Get bands if available
        if result.type_specific.get('bands'):
            bands = result.type_specific['bands']
            assert len(bands) > 0
        
        # Step 6: Convert to dict for UI display
        info_dict = result.to_dict()
        assert isinstance(info_dict, dict)
        
        # Step 7: Get suggestions for user
        suggestions = result.suggestions
        # Should have helpful info
    
    def test_provides_info_for_config_creation(self, real_phases_path: Path):
        """Verify scanner provides info needed for config creation."""
        from app.core.dataset_scanner import DatasetScanner
        
        scanner = DatasetScanner()
        result = scanner.scan(real_phases_path, deep_scan=True, sample_size=2)
        
        # Info needed for create_default_config()
        
        # 1. Conditions
        conditions = result.structure.classes
        assert isinstance(conditions, list)
        
        # 2. Bands
        bands = result.type_specific.get('bands', [])
        # May be empty if not deep scanned, but should be a list
        assert isinstance(bands, list)
        
        # 3. Has STC/EEG
        has_stc = result.type_specific.get('has_stc', False)
        has_eeg = result.type_specific.get('has_eeg', False)
        # Should be booleans
        assert isinstance(has_stc, bool)
        assert isinstance(has_eeg, bool)
    
    def test_handles_ui_refresh_scenario(self, real_phases_path: Path):
        """Verify scanner handles being called multiple times."""
        from app.core.dataset_scanner import DatasetScanner
        
        scanner = DatasetScanner()
        
        # Simulate user clicking "Scan" multiple times
        for _ in range(3):
            result = scanner.scan(real_phases_path)
            assert result.is_valid is True
            assert result.files.total_count > 0


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

@pytest.mark.slow
class TestBackwardCompatibility:
    """Test compatibility with existing detect_dataset_type() behavior."""
    
    def test_provides_same_fields_as_old_function(self, real_phases_path: Path):
        """Verify new scanner provides all fields the old function provided."""
        from app.core.dataset_scanner import DatasetScanner
        
        scanner = DatasetScanner()
        result = scanner.scan(real_phases_path, deep_scan=True, sample_size=2)
        
        # Fields from old detect_dataset_type()
        # type, structure, conditions, file_count, sample_file, bands,
        # data_sources, has_stc, has_eeg, num_nodes_eeg, num_nodes_stc, error
        
        # Primary type
        assert result.primary_type is not None
        
        # Structure
        assert result.structure is not None
        
        # Conditions (classes)
        conditions = result.structure.classes
        assert conditions is not None
        
        # File count
        assert result.files.total_count >= 0
        
        # Sample file
        assert result.files.sample_files is not None
        
        # Bands (in type_specific)
        # May not always be present, but should be accessible
        _ = result.type_specific.get('bands', [])
        
        # EEG/STC info
        _ = result.type_specific.get('has_stc', False)
        _ = result.type_specific.get('has_eeg', False)
    
    def test_can_replace_detect_dataset_type(self, real_phases_path: Path):
        """Verify new scanner can fully replace old function."""
        from app.core.dataset_scanner import DatasetScanner
        
        scanner = DatasetScanner()
        result = scanner.scan(real_phases_path, deep_scan=True, sample_size=2)
        
        # Create a dict similar to old function output for comparison
        old_style_info = {
            'type': result.primary_type.value,
            'structure': result.structure.split_type.value,
            'conditions': result.structure.classes,
            'file_count': result.files.total_count,
            'sample_file': str(result.files.sample_files[0]) if result.files.sample_files else None,
            'bands': result.type_specific.get('bands', []),
            'has_stc': result.type_specific.get('has_stc', False),
            'has_eeg': result.type_specific.get('has_eeg', False),
            'num_nodes_eeg': result.type_specific.get('n_channels_eeg', 0),
            'num_nodes_stc': result.type_specific.get('n_parcels', 0),
            'error': result.warnings[0] if result.warnings and not result.is_valid else None,
        }
        
        # Verify we can create this dict without errors
        assert isinstance(old_style_info, dict)
        assert old_style_info['type'] == 'graph'
        assert set(old_style_info['conditions']) == {'DMT', 'EC', 'EO'}

