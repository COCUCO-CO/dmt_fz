"""
Tests for PipelineState management.

Tests:
- PipelineState class attributes
- State persistence across operations
- State transitions (running -> complete)
- viz_state dictionary management
"""

import sys
from pathlib import Path
from datetime import datetime

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Test: PipelineState Initialization
# =============================================================================

class TestPipelineStateInit:
    """Tests for PipelineState initialization."""
    
    def test_default_running_state(self):
        """Should initialize with running=False."""
        from app.state import PipelineState
        
        state = PipelineState()
        
        assert state.running == False
        assert state.current_step == ""
        assert state.running_task_name == ""
        assert state.start_time == None
        assert state.current_process == None
    
    def test_default_ui_references(self):
        """Should initialize UI references as None."""
        from app.state import PipelineState
        
        state = PipelineState()
        
        assert state.log_container == None
        assert state.log_scroll == None
        assert state.status_label == None
        assert state.refresh_files == None
    
    def test_default_persistence_attributes(self):
        """Should initialize persistence attributes correctly."""
        from app.state import PipelineState
        
        state = PipelineState()
        
        assert state.log_history == []
        assert state.selected_run == None
    
    def test_default_pipeline_parameters(self):
        """Should have sensible default pipeline parameters."""
        from app.state import PipelineState
        
        state = PipelineState()
        
        assert state.max_subjects == 0  # 0 means all
        assert state.conditions == ["DMT", "EC", "EO"]
        assert state.jobs == 0  # 0 means auto
        assert state.workers == 7
        assert state.bands == ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
        assert state.min_k == 2
        assert state.max_k == 15
        assert state.min_comps == 2
        assert state.max_comps == 10
    
    def test_progress_default(self):
        """Should initialize progress at 0."""
        from app.state import PipelineState
        
        state = PipelineState()
        assert state.progress == 0


# =============================================================================
# Test: PipelineState Running Transitions
# =============================================================================

class TestPipelineStateTransitions:
    """Tests for state transitions during pipeline execution."""
    
    def test_transition_to_running(self, fresh_pipeline_state):
        """Should properly transition to running state."""
        state = fresh_pipeline_state
        
        state.running = True
        state.current_step = "Source Localization"
        state.running_task_name = "fwd.py"
        state.start_time = datetime.now()
        
        assert state.running == True
        assert state.current_step == "Source Localization"
        assert state.running_task_name == "fwd.py"
        assert state.start_time is not None
    
    def test_transition_to_complete(self, pipeline_state_running):
        """Should properly reset state after completion."""
        state = pipeline_state_running
        
        # Simulate completion
        state.running = False
        state.current_step = ""
        state.running_task_name = ""
        state.start_time = None
        state.current_process = None
        
        assert state.running == False
        assert state.current_step == ""
        assert state.running_task_name == ""
        assert state.start_time == None
    
    def test_elapsed_time_calculation(self, fresh_pipeline_state):
        """Should be able to calculate elapsed time."""
        import time
        
        state = fresh_pipeline_state
        state.start_time = datetime.now()
        
        time.sleep(0.1)
        
        elapsed = (datetime.now() - state.start_time).total_seconds()
        assert elapsed >= 0.1


# =============================================================================
# Test: PipelineState Log History
# =============================================================================

class TestPipelineStateLogHistory:
    """Tests for log history management."""
    
    def test_append_to_log_history(self, fresh_pipeline_state):
        """Should be able to append messages to log_history."""
        state = fresh_pipeline_state
        
        state.log_history.append("Message 1")
        state.log_history.append("Message 2")
        
        assert len(state.log_history) == 2
        assert state.log_history[0] == "Message 1"
    
    def test_clear_log_history(self, pipeline_state_with_history):
        """Should be able to clear log history."""
        state = pipeline_state_with_history
        
        assert len(state.log_history) > 0
        
        state.log_history.clear()
        
        assert len(state.log_history) == 0
    
    def test_log_history_preserves_order(self, fresh_pipeline_state):
        """Should preserve message order."""
        state = fresh_pipeline_state
        
        messages = ["First", "Second", "Third", "Fourth"]
        for msg in messages:
            state.log_history.append(msg)
        
        assert state.log_history == messages
    
    def test_log_history_persistence(self, fresh_pipeline_state):
        """Log history should persist across operations."""
        state = fresh_pipeline_state
        
        state.log_history.append("Before running")
        state.running = True
        state.log_history.append("During running")
        state.running = False
        state.log_history.append("After running")
        
        assert len(state.log_history) == 3
        assert "During running" in state.log_history


# =============================================================================
# Test: PipelineState Selected Run
# =============================================================================

class TestPipelineStateSelectedRun:
    """Tests for selected_run persistence."""
    
    def test_set_selected_run(self, fresh_pipeline_state, tmp_path):
        """Should be able to set selected run path."""
        state = fresh_pipeline_state
        
        run_dir = tmp_path / "run_20241201_100000"
        run_dir.mkdir()
        
        state.selected_run = run_dir
        
        assert state.selected_run == run_dir
    
    def test_selected_run_persists(self, fresh_pipeline_state, tmp_path):
        """Selected run should persist across state changes."""
        state = fresh_pipeline_state
        
        run_dir = tmp_path / "run_20241201_100000"
        run_dir.mkdir()
        
        state.selected_run = run_dir
        
        # Simulate running pipeline
        state.running = True
        state.current_step = "Test"
        
        assert state.selected_run == run_dir
        
        # Complete pipeline
        state.running = False
        state.current_step = ""
        
        assert state.selected_run == run_dir
    
    def test_selected_run_can_be_cleared(self, fresh_pipeline_state, tmp_path):
        """Should be able to clear selected run."""
        state = fresh_pipeline_state
        
        run_dir = tmp_path / "run_20241201_100000"
        run_dir.mkdir()
        
        state.selected_run = run_dir
        state.selected_run = None
        
        assert state.selected_run == None


# =============================================================================
# Test: PipelineState Conditions
# =============================================================================

class TestPipelineStateConditions:
    """Tests for pipeline condition parameters."""
    
    def test_default_conditions(self, fresh_pipeline_state):
        """Should have all conditions by default."""
        state = fresh_pipeline_state
        
        assert "DMT" in state.conditions
        assert "EC" in state.conditions
        assert "EO" in state.conditions
    
    def test_modify_conditions(self, fresh_pipeline_state):
        """Should be able to modify conditions."""
        state = fresh_pipeline_state
        
        state.conditions = ["DMT"]
        
        assert state.conditions == ["DMT"]
        assert "EC" not in state.conditions
    
    def test_conditions_independent_of_running(self, fresh_pipeline_state):
        """Conditions should persist regardless of running state."""
        state = fresh_pipeline_state
        
        state.conditions = ["DMT", "EO"]
        state.running = True
        state.running = False
        
        assert state.conditions == ["DMT", "EO"]


# =============================================================================
# Test: PipelineState Bands
# =============================================================================

class TestPipelineStateBands:
    """Tests for frequency band parameters."""
    
    def test_default_bands(self, fresh_pipeline_state):
        """Should have all frequency bands by default."""
        state = fresh_pipeline_state
        
        expected_bands = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
        assert state.bands == expected_bands
    
    def test_modify_bands(self, fresh_pipeline_state):
        """Should be able to modify bands."""
        state = fresh_pipeline_state
        
        state.bands = ["Alpha", "Beta"]
        
        assert state.bands == ["Alpha", "Beta"]
        assert "Delta" not in state.bands


# =============================================================================
# Test: PipelineState Workers/Jobs
# =============================================================================

class TestPipelineStateWorkers:
    """Tests for worker and job parameters."""
    
    def test_default_workers(self, fresh_pipeline_state):
        """Should have sensible default workers."""
        state = fresh_pipeline_state
        
        assert state.workers == 7
        assert state.jobs == 0  # 0 = auto
    
    def test_modify_workers(self, fresh_pipeline_state):
        """Should be able to modify worker count."""
        state = fresh_pipeline_state
        
        state.workers = 4
        state.jobs = 8
        
        assert state.workers == 4
        assert state.jobs == 8
    
    def test_workers_accept_various_values(self, fresh_pipeline_state):
        """Workers should accept various valid values."""
        state = fresh_pipeline_state
        
        for workers in [1, 2, 4, 8, 16, 32]:
            state.workers = workers
            assert state.workers == workers


# =============================================================================
# Test: PipelineState Clustering Parameters
# =============================================================================

class TestPipelineStateClusteringParams:
    """Tests for clustering-specific parameters."""
    
    def test_default_k_range(self, fresh_pipeline_state):
        """Should have sensible default k range."""
        state = fresh_pipeline_state
        
        assert state.min_k == 2
        assert state.max_k == 15
        assert state.min_k < state.max_k
    
    def test_default_pca_range(self, fresh_pipeline_state):
        """Should have sensible default PCA component range."""
        state = fresh_pipeline_state
        
        assert state.min_comps == 2
        assert state.max_comps == 10
        assert state.min_comps < state.max_comps
    
    def test_modify_clustering_params(self, fresh_pipeline_state):
        """Should be able to modify clustering parameters."""
        state = fresh_pipeline_state
        
        state.min_k = 3
        state.max_k = 20
        state.min_comps = 5
        state.max_comps = 15
        
        assert state.min_k == 3
        assert state.max_k == 20
        assert state.min_comps == 5
        assert state.max_comps == 15


# =============================================================================
# Test: Global PS Instance
# =============================================================================

class TestGlobalPipelineState:
    """Tests for global PS instance behavior."""
    
    def test_global_ps_exists(self):
        """Global PS instance should be importable."""
        from app.state import PS
        
        assert PS is not None
    
    def test_global_ps_is_pipeline_state(self):
        """Global PS should be PipelineState instance."""
        from app.state import PS, PipelineState
        
        assert isinstance(PS, PipelineState)
    
    def test_global_ps_singleton_behavior(self):
        """Multiple imports should return same instance."""
        from app.state import PS as PS1
        from app.state import PS as PS2
        
        assert PS1 is PS2
        
        # Modifying one should affect the other
        original = PS1.workers
        PS1.workers = 999
        assert PS2.workers == 999
        PS1.workers = original


# =============================================================================
# Test: viz_state Dictionary
# =============================================================================

class TestVizState:
    """Tests for visualization state dictionary."""
    
    def test_viz_state_default_structure(self, fresh_viz_state):
        """viz_state should have expected keys."""
        assert 'data' in fresh_viz_state
        assert 'file' in fresh_viz_state
        assert 'loaded' in fresh_viz_state
    
    def test_viz_state_stores_data(self, fresh_viz_state):
        """viz_state should be able to store loaded data."""
        test_data = {"phases_stc": {"Alpha": [[1, 2, 3]]}}
        
        fresh_viz_state['data'] = test_data
        fresh_viz_state['loaded'] = True
        
        assert fresh_viz_state['data'] == test_data
        assert fresh_viz_state['loaded'] == True
    
    def test_viz_state_stores_file_path(self, fresh_viz_state, tmp_path):
        """viz_state should store file path."""
        test_file = tmp_path / "test.pkl"
        test_file.touch()
        
        fresh_viz_state['file'] = test_file
        
        assert fresh_viz_state['file'] == test_file
    
    def test_viz_state_custom_path(self, fresh_viz_state, tmp_path):
        """viz_state should support custom_path."""
        fresh_viz_state['custom_path'] = tmp_path
        
        assert fresh_viz_state['custom_path'] == tmp_path
    
    def test_viz_state_stores_functions(self, fresh_viz_state):
        """viz_state should be able to store function references."""
        def dummy_update():
            return "updated"
        
        fresh_viz_state['update_brain_plot'] = dummy_update
        
        assert callable(fresh_viz_state['update_brain_plot'])
        assert fresh_viz_state['update_brain_plot']() == "updated"
    
    def test_viz_state_current_brain_plot(self, fresh_viz_state):
        """viz_state should track current brain plot type."""
        fresh_viz_state['current_brain_plot'] = 'network'
        assert fresh_viz_state['current_brain_plot'] == 'network'
        
        fresh_viz_state['current_brain_plot'] = 'sync'
        assert fresh_viz_state['current_brain_plot'] == 'sync'


# =============================================================================
# Test: State Isolation
# =============================================================================

class TestStateIsolation:
    """Tests to ensure different state instances are isolated."""
    
    def test_fresh_instances_are_independent(self):
        """Each PipelineState instance should be independent."""
        from app.state import PipelineState
        
        state1 = PipelineState()
        state2 = PipelineState()
        
        state1.workers = 1
        state2.workers = 2
        
        assert state1.workers == 1
        assert state2.workers == 2
    
    def test_log_history_not_shared(self):
        """Log history should not be shared between instances."""
        from app.state import PipelineState
        
        state1 = PipelineState()
        state2 = PipelineState()
        
        state1.log_history.append("State 1 message")
        
        assert "State 1 message" in state1.log_history
        assert "State 1 message" not in state2.log_history









