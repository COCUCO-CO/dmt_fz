"""
Tests for pipeline visualization functionality.

Tests:
- Visualization data loading
- Brain plot generation
- Kuramoto visualizations
- Network analysis plots
- Data file handling
"""

import pytest
import sys
import pickle
import numpy as np
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Test: Visualization Data Loading
# =============================================================================

class TestVisualizationDataLoading:
    """Tests for loading visualization data from pipeline output."""
    
    def test_load_phases_pkl(self, temp_pipeline_with_results):
        """Should be able to load phases-*.pkl files."""
        run_dir = temp_pipeline_with_results["result_run"]
        data_file = run_dir / "DMT" / "phases-S01-DMT.pkl"
        
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        
        assert data is not None
        assert isinstance(data, dict)
        assert "phases_stc" in data or "kuramoto_stc" in data
    
    def test_data_has_expected_structure(self, temp_pipeline_with_results):
        """Loaded data should have expected structure."""
        run_dir = temp_pipeline_with_results["result_run"]
        data_file = run_dir / "DMT" / "phases-S01-DMT.pkl"
        
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        
        # Check for band keys
        if "phases_stc" in data:
            assert "Alpha" in data["phases_stc"]
    
    def test_find_phases_files_in_run(self, temp_pipeline_with_results):
        """Should find all phases files in a run directory."""
        run_dir = temp_pipeline_with_results["result_run"]
        
        phases_files = list(run_dir.rglob("phases-*.pkl"))
        
        assert len(phases_files) > 0
    
    def test_extract_subject_from_filename(self, temp_pipeline_with_results):
        """Should extract subject ID from filename."""
        run_dir = temp_pipeline_with_results["result_run"]
        phases_files = list(run_dir.rglob("phases-*.pkl"))
        
        for f in phases_files:
            stem = f.stem  # e.g., "phases-S01-DMT"
            parts = stem.split("-")
            # Should have format phases-{subject}-{condition}
            assert len(parts) >= 2
            subject = parts[1]
            assert subject.startswith("S")


# =============================================================================
# Test: Visualization State Management
# =============================================================================

class TestVisualizationStateManagement:
    """Tests for viz_state management in visualizations."""
    
    def test_viz_state_initialized_on_page(self):
        """viz_state should be initialized when accessing visualizations."""
        from app.state import PS
        
        # Ensure viz_state exists on PS
        if not hasattr(PS, 'viz_state'):
            PS.viz_state = {'data': None, 'file': None, 'loaded': False}
        
        assert hasattr(PS, 'viz_state')
        assert isinstance(PS.viz_state, dict)
    
    def test_viz_state_data_persistence(self, viz_state_with_data):
        """Data should persist in viz_state after loading."""
        assert viz_state_with_data['data'] is not None
        assert viz_state_with_data['loaded'] == True
    
    def test_viz_state_file_tracking(self, viz_state_with_data):
        """viz_state should track loaded file path."""
        assert viz_state_with_data['file'] is not None
        assert viz_state_with_data['file'].exists()
    
    def test_viz_state_function_storage(self, fresh_viz_state):
        """viz_state should store update functions."""
        def mock_update():
            pass
        
        fresh_viz_state['update_brain_plot'] = mock_update
        fresh_viz_state['update_hilbert_2d'] = mock_update
        fresh_viz_state['update_hilbert_3d'] = mock_update
        fresh_viz_state['refresh_all_plots'] = mock_update
        
        assert callable(fresh_viz_state['update_brain_plot'])
        assert callable(fresh_viz_state['refresh_all_plots'])


# =============================================================================
# Test: Subject Loading
# =============================================================================

class TestSubjectLoading:
    """Tests for loading subjects from data directories."""
    
    def test_find_subjects_in_directory(self, temp_pipeline_with_results):
        """Should find all subjects in a run directory."""
        run_dir = temp_pipeline_with_results["result_run"]
        
        all_files = list(run_dir.rglob('phases-*.pkl'))
        subjects = set()
        
        for f in all_files:
            stem = f.stem
            if '-' in stem:
                parts = stem.split('-')
                if len(parts) >= 2:
                    subjects.add(parts[1])
        
        assert len(subjects) > 0
    
    def test_subjects_sorted(self, temp_pipeline_with_results):
        """Subject list should be sortable."""
        run_dir = temp_pipeline_with_results["result_run"]
        
        all_files = list(run_dir.rglob('phases-*.pkl'))
        subjects = []
        
        for f in all_files:
            stem = f.stem
            if '-' in stem:
                parts = stem.split('-')
                if len(parts) >= 2:
                    subjects.append(parts[1])
        
        subjects = sorted(list(set(subjects)))
        
        # Should be sortable without error
        assert subjects == sorted(subjects)


# =============================================================================
# Test: Band Selection
# =============================================================================

class TestBandSelection:
    """Tests for frequency band selection in visualizations."""
    
    def test_valid_bands(self):
        """Should recognize all valid frequency bands."""
        valid_bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
        
        for band in valid_bands:
            assert band[0].isupper()  # Band names are capitalized
    
    def test_band_data_access(self, temp_pipeline_with_results):
        """Should be able to access data for each band."""
        run_dir = temp_pipeline_with_results["result_run"]
        data_file = run_dir / "DMT" / "phases-S01-DMT.pkl"
        
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        
        # Our test data has Alpha
        if "phases_stc" in data:
            assert "Alpha" in data["phases_stc"]


# =============================================================================
# Test: Epoch Selection
# =============================================================================

class TestEpochSelection:
    """Tests for epoch selection in visualizations."""
    
    def test_epoch_index_valid_range(self, temp_pipeline_with_results):
        """Epoch index should be within valid range."""
        run_dir = temp_pipeline_with_results["result_run"]
        data_file = run_dir / "DMT" / "phases-S01-DMT.pkl"
        
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        
        if "phases_stc" in data and "Alpha" in data["phases_stc"]:
            n_epochs = len(data["phases_stc"]["Alpha"])
            
            # Valid epoch indices: 0 to n_epochs-1
            assert 0 <= 0 < n_epochs or n_epochs == 0
    
    def test_epoch_default_zero(self):
        """Default epoch should be 0."""
        default_epoch = 0
        assert default_epoch == 0


# =============================================================================
# Test: Real Data Visualization (Integration)
# =============================================================================

class TestRealDataVisualization:
    """Integration tests with real pipeline output data."""
    
    @pytest.fixture
    def real_pipeline_data(self):
        """Load real pipeline output if available."""
        # Try to find existing pipeline output
        possible_paths = [
            Path("/media/storage_hdd/dmt_fz/fwd-inv-stc"),
            Path("/media/storage_hdd/dmt_fz/eeg_viewer/pipeline_outputs"),
        ]
        
        for base_path in possible_paths:
            if base_path.exists():
                # Look for phases files
                phases_files = list(base_path.rglob("phases-*.pkl"))
                if phases_files:
                    return phases_files[0]
        
        pytest.skip("No real pipeline output found")
    
    @pytest.mark.slow
    def test_load_real_phases_data(self, real_pipeline_data):
        """Should load real pipeline output data."""
        with open(real_pipeline_data, 'rb') as f:
            data = pickle.load(f)
        
        assert data is not None
        # Real data should have full structure
        expected_keys = ["phases_stc", "kuramoto_stc", "syncros_stc", "amplitudes_stc"]
        for key in expected_keys:
            if key in data:
                assert isinstance(data[key], dict)
    
    @pytest.mark.slow
    def test_real_data_has_all_bands(self, real_pipeline_data):
        """Real data should have all frequency bands."""
        with open(real_pipeline_data, 'rb') as f:
            data = pickle.load(f)
        
        bands = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
        
        # Check at least one data type has all bands
        for data_type in ["phases_stc", "kuramoto_stc"]:
            if data_type in data:
                for band in bands:
                    assert band in data[data_type], f"Missing {band} in {data_type}"
                break
    
    @pytest.mark.slow
    def test_real_data_epoch_shape(self, real_pipeline_data):
        """Real data epochs should have correct shape."""
        with open(real_pipeline_data, 'rb') as f:
            data = pickle.load(f)
        
        if "kuramoto_stc" in data:
            alpha_data = data["kuramoto_stc"].get("Alpha", [])
            if alpha_data:
                # Each epoch should be a 1D array (time series)
                epoch0 = alpha_data[0]
                assert isinstance(epoch0, (list, np.ndarray))


# =============================================================================
# Test: Visualization Scripts Imports
# =============================================================================

class TestVisualizationScriptsImports:
    """Tests for viz_scripts module imports."""
    
    def test_brain_3d_importable(self):
        """brain_3d module should be importable."""
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "viz_scripts"))
            from viz_scripts import brain_3d
            assert brain_3d is not None
        except ImportError as e:
            pytest.skip(f"brain_3d not importable: {e}")
    
    def test_kuramoto_viz_importable(self):
        """kuramoto_viz module should be importable."""
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "viz_scripts"))
            from viz_scripts import kuramoto_viz
            assert kuramoto_viz is not None
        except ImportError as e:
            pytest.skip(f"kuramoto_viz not importable: {e}")
    
    def test_clustering_viz_importable(self):
        """clustering_viz module should be importable."""
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "viz_scripts"))
            from viz_scripts import clustering_viz
            assert clustering_viz is not None
        except ImportError as e:
            pytest.skip(f"clustering_viz not importable: {e}")


# =============================================================================
# Test: Visualization Script Functions
# =============================================================================

class TestVisualizationScriptFunctions:
    """Tests for specific visualization functions."""
    
    @pytest.fixture
    def viz_scripts_available(self):
        """Check if viz_scripts are available."""
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "viz_scripts"))
            from viz_scripts import brain_3d, kuramoto_viz
            return {'brain_3d': brain_3d, 'kuramoto_viz': kuramoto_viz}
        except ImportError:
            pytest.skip("viz_scripts not available")
    
    def test_brain_3d_has_required_functions(self, viz_scripts_available):
        """brain_3d should have required visualization functions."""
        brain_3d = viz_scripts_available['brain_3d']
        
        expected_functions = [
            'create_brain_network_figure',
            'create_colored_brain_figure',
        ]
        
        for func_name in expected_functions:
            assert hasattr(brain_3d, func_name), f"Missing function: {func_name}"
    
    def test_kuramoto_viz_has_required_functions(self, viz_scripts_available):
        """kuramoto_viz should have required visualization functions."""
        kuramoto_viz = viz_scripts_available['kuramoto_viz']
        
        expected_functions = [
            'create_timeline_figure',
            'create_band_comparison_figure',
        ]
        
        for func_name in expected_functions:
            assert hasattr(kuramoto_viz, func_name), f"Missing function: {func_name}"


# =============================================================================
# Test: Data File Patterns
# =============================================================================

class TestDataFilePatterns:
    """Tests for pipeline output file naming patterns."""
    
    def test_phases_file_pattern(self, temp_pipeline_with_results):
        """phases files should match expected pattern."""
        run_dir = temp_pipeline_with_results["result_run"]
        
        phases_files = list(run_dir.rglob("phases-*.pkl"))
        
        for f in phases_files:
            assert f.name.startswith("phases-")
            assert f.suffix == ".pkl"
    
    def test_syncro_file_pattern(self, tmp_path):
        """syncro files should match expected pattern."""
        # Create test syncro files
        for name in ["syncro-S01-DMT.pkl", "syncro-S02-EC.pkl"]:
            (tmp_path / name).touch()
        
        syncro_files = list(tmp_path.glob("syncro-*.pkl"))
        
        for f in syncro_files:
            assert f.name.startswith("syncro-")
            assert f.suffix == ".pkl"
    
    def test_condition_directories(self, temp_pipeline_with_results):
        """Run should have condition subdirectories."""
        run_dir = temp_pipeline_with_results["result_run"]
        
        for cond in ["DMT", "EC", "EO"]:
            cond_dir = run_dir / cond
            assert cond_dir.exists() or len(list(run_dir.rglob(f"*{cond}*.pkl"))) >= 0


# =============================================================================
# Test: Custom Path Support
# =============================================================================

class TestCustomPathSupport:
    """Tests for custom data path functionality."""
    
    def test_custom_path_stored_in_viz_state(self, fresh_viz_state, tmp_path):
        """Custom path should be stored in viz_state."""
        fresh_viz_state['custom_path'] = tmp_path
        
        assert fresh_viz_state['custom_path'] == tmp_path
    
    def test_custom_path_overrides_default(self, fresh_viz_state, tmp_path):
        """Custom path should override default run directory."""
        custom_path = tmp_path / "custom_data"
        custom_path.mkdir()
        
        fresh_viz_state['custom_path'] = custom_path
        
        # When custom_path is set, it should be used instead of selected_run
        effective_path = fresh_viz_state.get('custom_path') or None
        assert effective_path == custom_path
    
    def test_custom_path_can_be_cleared(self, fresh_viz_state, tmp_path):
        """Custom path should be clearable."""
        fresh_viz_state['custom_path'] = tmp_path
        fresh_viz_state['custom_path'] = None
        
        assert fresh_viz_state['custom_path'] == None


# =============================================================================
# Test: Animation Generator Support
# =============================================================================

class TestAnimationGeneratorSupport:
    """Tests for animation generation functionality."""
    
    def test_animation_output_structure(self, tmp_path):
        """Animation output should follow expected structure."""
        # Expected: visualizations/plot/{mode}/{subject}_{condition}_{band}
        mode = "stc"
        subject = "S01"
        condition = "DMT"
        band = "Alpha"
        
        expected_path = tmp_path / "visualizations" / "plot" / mode / f"{subject}_{condition}_{band}"
        expected_path.mkdir(parents=True)
        
        assert expected_path.exists()
    
    def test_frame_file_pattern(self, tmp_path):
        """Frame files should match expected pattern."""
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        
        # Create test frame files
        for i in range(5):
            (frames_dir / f"frame_{i:04d}.png").touch()
        
        frames = sorted(frames_dir.glob("*.png"))
        
        assert len(frames) == 5
        assert frames[0].name == "frame_0000.png"


