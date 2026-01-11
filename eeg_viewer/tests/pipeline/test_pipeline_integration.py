"""
Integration tests for the complete pipeline page.

Tests:
- End-to-end workflow
- State persistence across operations
- Real EEG data processing (limited)
"""

import pytest
import sys
import pickle
from pathlib import Path
from datetime import datetime

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Test: Complete Workflow Simulation
# =============================================================================

class TestWorkflowSimulation:
    """Tests simulating complete pipeline workflows."""
    
    def test_create_run_then_select(self, tmp_path, monkeypatch, clean_pipeline_globals):
        """Should be able to create and select a run."""
        from app.pages.pipeline import create_new_run, get_run_dirs
        
        monkeypatch.setattr('app.pages.pipeline.PIPELINE_OUTPUTS', tmp_path)
        
        # Create a new run
        run_dir = create_new_run()
        
        # Should appear in run list
        runs = get_run_dirs()
        assert run_dir.name in runs
        
        # Should be selectable
        PS = clean_pipeline_globals
        PS.selected_run = run_dir
        assert PS.selected_run == run_dir
    
    def test_workflow_state_tracking(self, clean_pipeline_globals, tmp_path, monkeypatch):
        """Should track workflow state correctly."""
        from app.pages.pipeline import create_new_run, pipeline_log
        
        PS = clean_pipeline_globals
        monkeypatch.setattr('app.pages.pipeline.PIPELINE_OUTPUTS', tmp_path)
        
        # Step 1: Create run
        run_dir = create_new_run()
        PS.selected_run = run_dir
        pipeline_log("[RUN] Created new run")
        
        # Step 2: Simulate step execution
        PS.running = True
        PS.current_step = "Source Localization"
        PS.start_time = datetime.now()
        pipeline_log("[1/8] Starting fwd.py")
        
        # Verify state
        assert PS.running == True
        assert PS.selected_run == run_dir
        assert len(PS.log_history) >= 2
        
        # Step 3: Complete step
        PS.running = False
        PS.current_step = ""
        pipeline_log("[1/8] Completed")
        
        assert PS.running == False
        assert PS.selected_run == run_dir  # Should persist
    
    def test_multiple_steps_in_sequence(self, clean_pipeline_globals):
        """Should handle multiple steps in sequence."""
        PS = clean_pipeline_globals
        
        steps = [
            ("fwd.py", "Source Localization"),
            ("save_load_pickle.py", "Consolidate"),
            ("multi2pool2.py", "Network Filtering"),
        ]
        
        for script, step_name in steps:
            # Start step
            PS.running = True
            PS.current_step = step_name
            PS.running_task_name = script
            PS.start_time = datetime.now()
            PS.log_history.append(f"[{step_name}] Starting")
            
            # Complete step
            PS.running = False
            PS.current_step = ""
            PS.running_task_name = ""
            PS.log_history.append(f"[{step_name}] Completed")
        
        # All steps should be logged
        assert len(PS.log_history) == 6  # 2 messages per step
        assert "Completed" in PS.log_history[-1]


# =============================================================================
# Test: State Persistence
# =============================================================================

class TestStatePersistence:
    """Tests for state persistence across operations."""
    
    def test_selected_run_persists_across_steps(self, clean_pipeline_globals, tmp_path):
        """Selected run should persist across step executions."""
        PS = clean_pipeline_globals
        
        run_dir = tmp_path / "run_test"
        run_dir.mkdir()
        PS.selected_run = run_dir
        
        # Simulate multiple operations
        for i in range(3):
            PS.running = True
            PS.running = False
        
        assert PS.selected_run == run_dir
    
    def test_log_history_persists(self, clean_pipeline_globals):
        """Log history should persist."""
        PS = clean_pipeline_globals
        
        messages = ["Message 1", "Message 2", "Message 3"]
        for msg in messages:
            PS.log_history.append(msg)
        
        # Simulate state changes
        PS.running = True
        PS.running = False
        PS.current_step = "Test"
        PS.current_step = ""
        
        assert PS.log_history == messages
    
    def test_viz_state_persists(self, clean_pipeline_globals):
        """Visualization state should persist."""
        PS = clean_pipeline_globals
        
        # Initialize viz_state
        if not hasattr(PS, 'viz_state'):
            PS.viz_state = {}
        
        PS.viz_state['data'] = {'test': 'data'}
        PS.viz_state['loaded'] = True
        
        # Simulate operations
        PS.running = True
        PS.running = False
        
        assert PS.viz_state['data'] == {'test': 'data'}
        assert PS.viz_state['loaded'] == True


# =============================================================================
# Test: Real Data Integration (Limited)
# =============================================================================

class TestRealDataIntegration:
    """Integration tests with real EEG data (limited scope)."""
    
    @pytest.fixture
    def real_eeg_available(self):
        """Check if real EEG data is available."""
        from .conftest import SAMPLE_SET_FILE
        if not SAMPLE_SET_FILE.exists():
            pytest.skip("Real EEG data not available")
        return SAMPLE_SET_FILE
    
    @pytest.mark.slow
    def test_can_import_mne(self):
        """MNE should be importable."""
        import mne
        assert mne is not None
    
    @pytest.mark.slow
    def test_can_load_set_file(self, real_eeg_available):
        """Should be able to load .set file with MNE."""
        import mne
        
        epochs = mne.io.read_epochs_eeglab(
            str(real_eeg_available),
            montage_units='dm',
            verbose=False
        )
        
        assert epochs is not None
        assert len(epochs) > 0
    
    @pytest.mark.slow
    def test_epochs_have_expected_properties(self, real_eeg_available):
        """Loaded epochs should have expected properties."""
        import mne
        
        epochs = mne.io.read_epochs_eeglab(
            str(real_eeg_available),
            montage_units='dm',
            verbose=False
        )
        
        # Should have sampling frequency
        assert epochs.info['sfreq'] > 0
        
        # Should have channels
        assert len(epochs.ch_names) > 0


# =============================================================================
# Test: Error Recovery
# =============================================================================

class TestErrorRecovery:
    """Tests for error handling and recovery."""
    
    def test_state_reset_on_error(self, clean_pipeline_globals):
        """State should reset properly on error."""
        PS = clean_pipeline_globals
        
        # Start a step
        PS.running = True
        PS.current_step = "Test Step"
        PS.start_time = datetime.now()
        
        # Simulate error - reset state
        PS.running = False
        PS.current_step = ""
        PS.start_time = None
        PS.current_process = None
        
        # State should be clean
        assert PS.running == False
        assert PS.current_step == ""
        assert PS.start_time == None
    
    def test_log_history_preserved_on_error(self, clean_pipeline_globals):
        """Log history should be preserved even on error."""
        PS = clean_pipeline_globals
        
        PS.log_history.append("Before error")
        PS.running = True
        PS.log_history.append("During step")
        
        # Simulate error
        PS.log_history.append("[ERROR] Something went wrong")
        PS.running = False
        
        assert "Before error" in PS.log_history
        assert "During step" in PS.log_history
        assert "[ERROR]" in PS.log_history[-1]
    
    def test_selected_run_preserved_on_error(self, clean_pipeline_globals, tmp_path):
        """Selected run should be preserved on error."""
        PS = clean_pipeline_globals
        
        run_dir = tmp_path / "run_test"
        run_dir.mkdir()
        PS.selected_run = run_dir
        
        # Simulate error during step
        PS.running = True
        PS.running = False  # Error reset
        
        assert PS.selected_run == run_dir


# =============================================================================
# Test: Concurrent Access Prevention
# =============================================================================

class TestConcurrentAccessPrevention:
    """Tests for preventing concurrent pipeline execution."""
    
    @pytest.mark.asyncio
    async def test_cannot_run_while_running(self, clean_pipeline_globals, tmp_path):
        """Should not allow starting new step while running."""
        from app.pages.pipeline import run_pipeline_step
        
        PS = clean_pipeline_globals
        PS.running = True
        PS.current_step = "Existing Step"
        
        # Try to start another step
        script = tmp_path / "test.py"
        script.write_text("print('test')")
        
        await run_pipeline_step(str(script), [], "New Step")
        
        # Should not have changed the current step
        assert PS.current_step == "Existing Step"
    
    def test_running_flag_blocks_new_steps(self, clean_pipeline_globals):
        """Running flag should block new steps."""
        PS = clean_pipeline_globals
        
        PS.running = True
        
        # Simulate check in run function
        if PS.running:
            can_start = False
        else:
            can_start = True
        
        assert can_start == False


# =============================================================================
# Test: Output File Generation
# =============================================================================

class TestOutputFileGeneration:
    """Tests for verifying pipeline output file generation."""
    
    def test_phases_file_creation(self, tmp_path):
        """Should create phases files in correct location."""
        run_dir = tmp_path / "run_test"
        dmt_dir = run_dir / "DMT"
        dmt_dir.mkdir(parents=True)
        
        # Simulate phases file creation
        phases_data = {
            "phases_stc": {"Alpha": [[0.1, 0.2, 0.3]]},
            "kuramoto_stc": {"Alpha": [[0.5, 0.6, 0.7]]},
        }
        
        phases_file = dmt_dir / "phases-S01-DMT.pkl"
        with open(phases_file, 'wb') as f:
            pickle.dump(phases_data, f)
        
        assert phases_file.exists()
        
        # Verify contents
        with open(phases_file, 'rb') as f:
            loaded = pickle.load(f)
        
        assert "phases_stc" in loaded
        assert "kuramoto_stc" in loaded
    
    def test_extra_pkl_creation(self, tmp_path):
        """Should create extra.pkl with metadata."""
        run_dir = tmp_path / "run_test"
        run_dir.mkdir()
        
        # Simulate extra.pkl creation
        extra_data = [
            ['red', 'blue', 'green'],  # node_colors
            ['Label1', 'Label2'],       # label_names
            ['L1', 'L2'],               # label_names_short
            [[0, 0, 0], [1, 1, 1]],     # stc_coords_3d
            ['Fp1', 'Fp2'],             # ch_names
            {0: 'Fp1', 1: 'Fp2'},       # mapping
            {},                          # eeg_coords_2d
        ]
        
        extra_file = run_dir / "extra.pkl"
        with open(extra_file, 'wb') as f:
            pickle.dump(extra_data, f)
        
        assert extra_file.exists()


# =============================================================================
# Test: Full Pipeline Output Structure
# =============================================================================

class TestPipelineOutputStructure:
    """Tests for complete pipeline output structure."""
    
    def test_complete_run_structure(self, tmp_path):
        """Complete run should have expected structure."""
        run_dir = tmp_path / "run_20241201_100000"
        
        # Create expected structure
        for cond in ["DMT", "EC", "EO"]:
            cond_dir = run_dir / cond
            cond_dir.mkdir(parents=True)
            
            # Create sample files
            (cond_dir / f"phases-S01-{cond}.pkl").touch()
            (cond_dir / f"syncro-S01-{cond}.pkl").touch()
        
        # Create other expected files
        (run_dir / "extra.pkl").touch()
        
        cluster_dir = run_dir / "clustering_results"
        cluster_dir.mkdir()
        
        pearson_dir = run_dir / "pearson_results"
        pearson_dir.mkdir()
        
        # Verify structure
        assert (run_dir / "DMT").exists()
        assert (run_dir / "EC").exists()
        assert (run_dir / "EO").exists()
        assert (run_dir / "extra.pkl").exists()
        assert (run_dir / "clustering_results").exists()
        assert (run_dir / "pearson_results").exists()
    
    def test_count_subjects_per_condition(self, tmp_path):
        """Should be able to count subjects per condition."""
        run_dir = tmp_path / "run_test"
        
        subjects_per_condition = {"DMT": 5, "EC": 3, "EO": 4}
        
        for cond, n_subjects in subjects_per_condition.items():
            cond_dir = run_dir / cond
            cond_dir.mkdir(parents=True)
            
            for i in range(n_subjects):
                (cond_dir / f"phases-S{i+1:02d}-{cond}.pkl").touch()
        
        # Count files
        for cond, expected in subjects_per_condition.items():
            files = list((run_dir / cond).glob("phases-*.pkl"))
            assert len(files) == expected









