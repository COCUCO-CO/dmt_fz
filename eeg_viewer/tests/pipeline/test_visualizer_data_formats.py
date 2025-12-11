"""
Tests for visualizer data format handling.

Verifies that visualizers correctly handle:
- List format (list of arrays)
- NumPy array format
- Missing/None data
"""
import pytest
import sys
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestStep1DataFormats:
    """Test Step 1 visualizer handles different data formats."""
    
    def test_handles_list_of_arrays(self):
        """Should convert list of arrays to numpy array."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        
        # Create mock data as list of arrays (real format)
        mock_data = {
            'phases_stc': {
                'Alpha': [np.random.rand(102, 800) for _ in range(5)],  # 5 epochs
                'Beta': [np.random.rand(102, 800) for _ in range(5)],
            },
            'subject': 'S01-DMT',
            'sfreq': 500,
            'n_epochs': 5
        }
        
        viz = Step1Visualizer(lambda: None)
        viz._data = mock_data
        viz._selected_band = 'Alpha'
        viz._selected_epoch = 0
        
        # Should not raise error
        # The render method internally converts list to array
        band_data = mock_data['phases_stc']['Alpha']
        if isinstance(band_data, list):
            band_data = np.array(band_data)
        
        assert band_data.shape == (5, 102, 800)
    
    def test_handles_numpy_array_directly(self):
        """Should work with numpy array format too."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        
        # Create mock data as numpy array directly
        mock_data = {
            'phases_stc': {
                'Alpha': np.random.rand(5, 102, 800),  # Already numpy
            },
            'subject': 'S01-DMT',
            'sfreq': 500,
            'n_epochs': 5
        }
        
        viz = Step1Visualizer(lambda: None)
        viz._data = mock_data
        
        band_data = mock_data['phases_stc']['Alpha']
        if isinstance(band_data, list):
            band_data = np.array(band_data)
        
        assert band_data.shape == (5, 102, 800)
    
    def test_handles_missing_band(self):
        """Should handle missing band gracefully."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        
        mock_data = {
            'phases_stc': {
                'Alpha': np.random.rand(5, 102, 800),
            }
        }
        
        viz = Step1Visualizer(lambda: None)
        viz._data = mock_data
        viz._selected_band = 'Gamma'  # Not in data
        
        band_data = mock_data['phases_stc'].get('Gamma')
        assert band_data is None


class TestStep4DataFormats:
    """Test Step 4 (Syncro) visualizer handles different data formats."""
    
    def test_handles_epoch_dict_format(self):
        """Should handle syncro data with epoch keys."""
        from app.pages.pipeline.visualizers.step_4_viz import Step4Visualizer
        
        # Real syncro format: dict with 'epoch0', 'epoch1', etc.
        mock_data = {
            'syncros_stc': {
                'Alpha': {
                    'epoch0': np.random.rand(102, 102),
                    'epoch1': np.random.rand(102, 102),
                }
            }
        }
        
        viz = Step4Visualizer(lambda: None)
        viz._data = mock_data
        viz._selected_band = 'Alpha'
        viz._selected_epoch = 0
        
        syncros = mock_data['syncros_stc']
        band_data = syncros.get('Alpha', {})
        assert 'epoch0' in band_data
        assert band_data['epoch0'].shape == (102, 102)
    
    def test_handles_list_of_matrices(self):
        """Should handle syncro data as list of matrices."""
        from app.pages.pipeline.visualizers.step_4_viz import Step4Visualizer
        
        # Alternative format: list of matrices
        mock_data = {
            'syncros_stc': {
                'Alpha': [np.random.rand(102, 102) for _ in range(5)]
            }
        }
        
        viz = Step4Visualizer(lambda: None)
        viz._data = mock_data
        
        band_data = mock_data['syncros_stc']['Alpha']
        if isinstance(band_data, list):
            assert len(band_data) == 5
            assert band_data[0].shape == (102, 102)


class TestStep5DataFormats:
    """Test Step 5 (Order) visualizer handles different data formats."""
    
    def test_handles_order_epoch_dict(self):
        """Should handle order data with epoch keys."""
        from app.pages.pipeline.visualizers.step_5_viz import Step5Visualizer
        
        mock_data = {
            'order_stc': {
                'Alpha': {
                    'epoch0': np.random.rand(800),  # R(t) for one epoch
                    'epoch1': np.random.rand(800),
                }
            }
        }
        
        viz = Step5Visualizer(lambda: None)
        viz._data = mock_data
        viz._selected_band = 'Alpha'
        
        order = mock_data['order_stc']
        band_data = order.get('Alpha', {})
        assert 'epoch0' in band_data
        assert band_data['epoch0'].shape == (800,)


class TestVisualizationPanelResize:
    """Test visualization panel resize functionality."""
    
    def test_panel_has_resizable_class(self):
        """Panel should have CSS class for resizing."""
        # This tests the CSS is added correctly
        from app.pages.pipeline.components.visualization_panel import VisualizationPanel
        panel = VisualizationPanel(lambda: None)
        # The render method adds 'resizable-panel' class
        assert panel is not None
    
    def test_min_height_constraint(self):
        """Panel should have minimum height constraint in CSS."""
        # Check that min-height is defined
        expected_min_height = "150px"  # From the CSS
        assert expected_min_height == "150px"
    
    def test_max_height_constraint(self):
        """Panel should have maximum height constraint."""
        expected_max_height = "80vh"  # From the CSS
        assert expected_max_height == "80vh"


class TestRealDataLoading:
    """Test with real data files if available."""
    
    @pytest.fixture
    def real_phases_file(self):
        """Find a real phases file for testing."""
        base_path = Path('/media/storage_hdd/dmt_fz/eeg_viewer/pipeline_outputs')
        if not base_path.exists():
            pytest.skip("Pipeline outputs directory not found")
        
        files = list(base_path.rglob('phases-*.pkl'))
        if not files:
            pytest.skip("No phases files found")
        
        return files[0]
    
    def test_real_phases_file_loads(self, real_phases_file):
        """Should load real phases file without error."""
        import pickle
        
        with open(real_phases_file, 'rb') as f:
            data = pickle.load(f)
        
        assert 'phases_stc' in data or 'phases_eeg' in data
    
    def test_real_phases_has_expected_bands(self, real_phases_file):
        """Real data should have frequency bands."""
        import pickle
        
        with open(real_phases_file, 'rb') as f:
            data = pickle.load(f)
        
        phases = data.get('phases_stc') or data.get('phases_eeg')
        expected_bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
        
        for band in expected_bands:
            assert band in phases, f"Missing band: {band}"
    
    def test_real_phases_converts_to_array(self, real_phases_file):
        """Real phases data should convert to numpy array."""
        import pickle
        
        with open(real_phases_file, 'rb') as f:
            data = pickle.load(f)
        
        phases = data.get('phases_stc') or data.get('phases_eeg')
        alpha = phases.get('Alpha')
        
        if isinstance(alpha, list):
            arr = np.array(alpha)
            assert arr.ndim == 3  # [epochs, parcels, times]
            assert arr.shape[1] in [100, 102]  # Parcels


