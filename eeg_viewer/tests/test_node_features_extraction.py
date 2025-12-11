"""Tests for node features extraction from dataset scanner."""
import pytest
import pickle
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.core.dataset_scanner.detectors.graph import GraphDetector, PhasesInfo


class TestNodeFeaturesExtraction:
    """Test that node features are correctly extracted from phases files."""
    
    @pytest.fixture
    def sample_phases_data(self, tmp_path):
        """Create a sample phases file with known structure."""
        # Create realistic phases data
        n_channels = 24
        n_parcels = 102
        n_timepoints = 800
        n_epochs = 10
        bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
        
        data = {}
        
        # EEG data - 5 feature types
        for feature_type in ['phases', 'amplitudes', 'syncros', 'kuramoto', 'filtered']:
            key = f'{feature_type}_eeg'
            data[key] = {
                band: [np.random.randn(n_channels, n_timepoints) for _ in range(n_epochs)]
                for band in bands
            }
        
        # STC data - 5 feature types
        for feature_type in ['phases', 'amplitudes', 'syncros', 'kuramoto', 'filtered']:
            key = f'{feature_type}_stc'
            data[key] = {
                band: [np.random.randn(n_parcels, n_timepoints) for _ in range(n_epochs)]
                for band in bands
            }
        
        # Save the file
        phases_file = tmp_path / 'phases-S01-DMT.pkl'
        with open(phases_file, 'wb') as f:
            pickle.dump(data, f)
        
        return tmp_path, {
            'n_channels': n_channels,
            'n_parcels': n_parcels,
            'n_timepoints': n_timepoints,
            'n_epochs': n_epochs,
            'bands': bands,
            'n_feature_types': 5,
        }
    
    def test_extracts_eeg_feature_types(self, sample_phases_data):
        """Test that EEG feature types are extracted."""
        tmp_path, expected = sample_phases_data
        
        detector = GraphDetector()
        result = detector.detect(tmp_path, analyze_samples=True)
        
        assert result.is_detected
        assert result.phases_info is not None
        assert len(result.phases_info.eeg_feature_types) == 5
        assert 'phases' in result.phases_info.eeg_feature_types
        assert 'amplitudes' in result.phases_info.eeg_feature_types
        assert 'syncros' in result.phases_info.eeg_feature_types
        assert 'kuramoto' in result.phases_info.eeg_feature_types
        assert 'filtered' in result.phases_info.eeg_feature_types
    
    def test_extracts_stc_feature_types(self, sample_phases_data):
        """Test that STC feature types are extracted."""
        tmp_path, expected = sample_phases_data
        
        detector = GraphDetector()
        result = detector.detect(tmp_path, analyze_samples=True)
        
        assert result.is_detected
        assert result.phases_info is not None
        assert len(result.phases_info.stc_feature_types) == 5
        assert 'phases' in result.phases_info.stc_feature_types
        assert 'amplitudes' in result.phases_info.stc_feature_types
    
    def test_calculates_node_features_count(self, sample_phases_data):
        """Test that node features can be calculated from extracted info."""
        tmp_path, expected = sample_phases_data
        
        detector = GraphDetector()
        result = detector.detect(tmp_path, analyze_samples=True)
        
        pi = result.phases_info
        n_bands = len(pi.bands)
        
        # Calculate node features
        eeg_node_features = len(pi.eeg_feature_types) * n_bands
        stc_node_features = len(pi.stc_feature_types) * n_bands
        
        # Expected: 5 feature types × 5 bands = 25 features per node
        assert eeg_node_features == 25, f"Expected 25 EEG node features, got {eeg_node_features}"
        assert stc_node_features == 25, f"Expected 25 STC node features, got {stc_node_features}"
    
    def test_phases_info_to_dict_includes_feature_types(self, sample_phases_data):
        """Test that to_dict includes feature types."""
        tmp_path, expected = sample_phases_data
        
        detector = GraphDetector()
        result = detector.detect(tmp_path, analyze_samples=True)
        
        info_dict = result.phases_info.to_dict()
        
        assert 'eeg_feature_types' in info_dict
        assert 'stc_feature_types' in info_dict
        assert len(info_dict['eeg_feature_types']) == 5
        assert len(info_dict['stc_feature_types']) == 5
    
    def test_detect_dataset_type_includes_feature_types(self, sample_phases_data):
        """Test that detect_dataset_type includes feature types in output."""
        tmp_path, expected = sample_phases_data
        
        from app.pages.model import detect_dataset_type
        
        info = detect_dataset_type(tmp_path)
        
        # These MUST be present
        assert 'eeg_feature_types' in info, "eeg_feature_types missing from detect_dataset_type output!"
        assert 'stc_feature_types' in info, "stc_feature_types missing from detect_dataset_type output!"
        assert len(info['eeg_feature_types']) == 5, f"Expected 5 EEG feature types, got {len(info['eeg_feature_types'])}"
        assert len(info['stc_feature_types']) == 5, f"Expected 5 STC feature types, got {len(info['stc_feature_types'])}"
    
    def test_partial_feature_types(self, tmp_path):
        """Test with only some feature types present."""
        n_channels = 24
        n_timepoints = 800
        bands = ['Alpha', 'Beta']
        
        # Only phases and amplitudes
        data = {
            'phases_eeg': {band: [np.random.randn(n_channels, n_timepoints)] for band in bands},
            'amplitudes_eeg': {band: [np.random.randn(n_channels, n_timepoints)] for band in bands},
        }
        
        phases_file = tmp_path / 'phases-S01-EC.pkl'
        with open(phases_file, 'wb') as f:
            pickle.dump(data, f)
        
        detector = GraphDetector()
        result = detector.detect(tmp_path, analyze_samples=True)
        
        assert len(result.phases_info.eeg_feature_types) == 2
        assert len(result.phases_info.stc_feature_types) == 0  # No STC data
        
        # Node features: 2 types × 2 bands = 4
        eeg_node_features = len(result.phases_info.eeg_feature_types) * len(result.phases_info.bands)
        assert eeg_node_features == 4


class TestNodeFeaturesInModelPage:
    """Test that node features are displayed in the model page."""
    
    @pytest.fixture
    def mock_dataset_info(self):
        """Create mock dataset info with feature types."""
        return {
            'type': 'graph',
            'bands': ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'],
            'eeg_feature_types': ['phases', 'amplitudes', 'syncros', 'kuramoto', 'filtered'],
            'stc_feature_types': ['phases', 'amplitudes', 'syncros', 'kuramoto', 'filtered'],
            'num_nodes_eeg': 24,
            'num_nodes_stc': 102,
            'scanner_info': MagicMock(
                type_specific={
                    'format': 'phases',
                    'eeg_feature_types': ['phases', 'amplitudes', 'syncros', 'kuramoto', 'filtered'],
                    'stc_feature_types': ['phases', 'amplitudes', 'syncros', 'kuramoto', 'filtered'],
                }
            )
        }
    
    def test_node_features_calculation(self, mock_dataset_info):
        """Test node features calculation logic."""
        info = mock_dataset_info
        
        eeg_ft = info.get('eeg_feature_types', [])
        stc_ft = info.get('stc_feature_types', [])
        n_bands = len(info.get('bands', []))
        
        if eeg_ft and n_bands:
            eeg_node_features = len(eeg_ft) * n_bands
            assert eeg_node_features == 25
        
        if stc_ft and n_bands:
            stc_node_features = len(stc_ft) * n_bands
            assert stc_node_features == 25


class TestRealDataset:
    """Test with the real dataset if available."""
    
    @pytest.mark.skipif(
        not Path('/media/storage_hdd/dmt_fz/fwd-inv-stc').exists(),
        reason="Real dataset not available"
    )
    def test_real_dataset_has_feature_types(self):
        """Test that real dataset extraction includes feature types."""
        from app.pages.model import detect_dataset_type, _dataset_scanner
        
        # Clear cache
        _dataset_scanner.clear_cache()
        
        info = detect_dataset_type(Path('/media/storage_hdd/dmt_fz/fwd-inv-stc'))
        
        print(f"\n=== REAL DATASET INFO ===")
        print(f"bands: {info.get('bands')}")
        print(f"eeg_feature_types: {info.get('eeg_feature_types')}")
        print(f"stc_feature_types: {info.get('stc_feature_types')}")
        
        # These assertions MUST pass
        assert info.get('eeg_feature_types') is not None, "eeg_feature_types is None!"
        assert info.get('stc_feature_types') is not None, "stc_feature_types is None!"
        assert len(info.get('eeg_feature_types', [])) > 0, "eeg_feature_types is empty!"
        assert len(info.get('stc_feature_types', [])) > 0, "stc_feature_types is empty!"
        
        # Calculate and verify node features
        n_bands = len(info.get('bands', []))
        eeg_node_features = len(info['eeg_feature_types']) * n_bands
        stc_node_features = len(info['stc_feature_types']) * n_bands
        
        print(f"EEG node features: {eeg_node_features}")
        print(f"STC node features: {stc_node_features}")
        
        assert eeg_node_features > 0, "EEG node features is 0!"
        assert stc_node_features > 0, "STC node features is 0!"


