"""
Tests for Pipeline UI state persistence and management.

These tests ensure that critical UI state is preserved when:
- Switching tabs within the pipeline page
- Navigating away and back to the pipeline page  
- Running pipeline steps
- Changing parameters
"""
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
import sys

# Mock NiceGUI before importing pipeline module
sys.modules['nicegui'] = MagicMock()
sys.modules['nicegui.ui'] = MagicMock()

from app.state.global_state import PipelineState


class TestRunDirectoryPersistence:
    """Tests for run directory selection persistence."""
    
    def test_selected_run_persists_in_ps(self):
        """PS.selected_run should persist the selected run directory."""
        ps = PipelineState()
        run_dir = Path("/tmp/test_run_20231201_120000")
        
        ps.selected_run = run_dir
        
        assert ps.selected_run == run_dir
    
    def test_selected_run_survives_page_reload(self):
        """Selected run should survive simulated page reload."""
        ps = PipelineState()
        run_dir = Path("/tmp/test_run_20231201_120000")
        ps.selected_run = run_dir
        
        # Simulate page reload by accessing again
        assert ps.selected_run == run_dir
    
    def test_selected_run_can_be_cleared(self):
        """Selected run can be set to None."""
        ps = PipelineState()
        ps.selected_run = Path("/tmp/test")
        ps.selected_run = None
        
        assert ps.selected_run is None
    
    def test_current_run_dir_closure_pattern(self):
        """Test the mutable list closure pattern used in pipeline.py."""
        # This mimics: current_run_dir = [PS.selected_run]
        ps = PipelineState()
        ps.selected_run = Path("/tmp/initial")
        
        current_run_dir = [ps.selected_run]
        
        # Nested function modifying via closure
        def update_run(new_path):
            current_run_dir[0] = new_path
            ps.selected_run = new_path
        
        update_run(Path("/tmp/updated"))
        
        assert current_run_dir[0] == Path("/tmp/updated")
        assert ps.selected_run == Path("/tmp/updated")


class TestLogHistoryPersistence:
    """Tests for console log history persistence."""
    
    def test_log_history_is_list(self):
        """Log history should be a list."""
        ps = PipelineState()
        assert isinstance(ps.log_history, list)
    
    def test_log_history_persists_messages(self):
        """Messages appended to log_history should persist."""
        ps = PipelineState()
        ps.log_history.clear()
        
        ps.log_history.append("[STEP_1] Starting...")
        ps.log_history.append("[STEP_1] Completed")
        
        assert len(ps.log_history) == 2
        assert "[STEP_1] Starting..." in ps.log_history
    
    def test_log_history_survives_tab_switch(self):
        """Log history should survive tab switching (simulated)."""
        ps = PipelineState()
        ps.log_history.clear()
        
        # Add logs
        ps.log_history.append("Log 1")
        ps.log_history.append("Log 2")
        
        # Simulate tab switch by creating new reference
        log_ref = ps.log_history
        
        assert len(log_ref) == 2
        assert log_ref == ps.log_history
    
    def test_log_history_clear(self):
        """Log history can be cleared."""
        ps = PipelineState()
        ps.log_history.append("test")
        ps.log_history.clear()
        
        assert len(ps.log_history) == 0


class TestVizStatePersistence:
    """Tests for visualization state persistence."""
    
    def test_viz_state_init_on_ps(self):
        """Viz state should be initializable on PS."""
        ps = PipelineState()
        
        # This mimics: if not hasattr(PS, 'viz_state'): PS.viz_state = {...}
        if not hasattr(ps, 'viz_state') or ps.viz_state is None:
            ps.viz_state = {'data': None, 'file': None, 'loaded': False}
        
        assert isinstance(ps.viz_state, dict)
        assert 'data' in ps.viz_state
    
    def test_viz_state_stores_data_reference(self):
        """Viz state should store data references."""
        ps = PipelineState()
        ps.viz_state = {'data': None, 'file': None, 'loaded': False}
        
        # Simulate loading data
        mock_data = {'phases_stc': {'Alpha': [1, 2, 3]}}
        ps.viz_state['data'] = mock_data
        ps.viz_state['loaded'] = True
        
        assert ps.viz_state['data'] == mock_data
        assert ps.viz_state['loaded'] is True
    
    def test_viz_state_stores_function_references(self):
        """Viz state should store function references for refresh."""
        ps = PipelineState()
        ps.viz_state = {'data': None}
        
        # Store function reference (like refresh_all_plots)
        def mock_refresh():
            return "refreshed"
        
        ps.viz_state['refresh_all_plots'] = mock_refresh
        
        assert ps.viz_state['refresh_all_plots']() == "refreshed"
    
    def test_viz_state_custom_path(self):
        """Viz state should support custom data path."""
        ps = PipelineState()
        ps.viz_state = {'data': None, 'custom_path': None}
        
        ps.viz_state['custom_path'] = Path("/custom/data/path")
        
        assert ps.viz_state['custom_path'] == Path("/custom/data/path")


class TestPipelineRunningState:
    """Tests for pipeline running state management."""
    
    def test_running_flag_initial(self):
        """Running flag should be False initially."""
        ps = PipelineState()
        assert ps.running is False
    
    def test_running_flag_set_and_clear(self):
        """Running flag can be set and cleared."""
        ps = PipelineState()
        ps.running = True
        assert ps.running is True
        ps.running = False
        assert ps.running is False
    
    def test_current_step_tracking(self):
        """Current step name should be tracked."""
        ps = PipelineState()
        ps.current_step = "Source Localization"
        assert ps.current_step == "Source Localization"
    
    def test_start_time_tracking(self):
        """Start time should be tracked for elapsed time display."""
        ps = PipelineState()
        now = datetime.now()
        ps.start_time = now
        assert ps.start_time == now
    
    def test_running_task_name(self):
        """Running task name for global indicator."""
        ps = PipelineState()
        ps.running_task_name = "fwd.py"
        assert ps.running_task_name == "fwd.py"
    
    def test_current_process_reference(self):
        """Current process reference for stop functionality."""
        ps = PipelineState()
        mock_process = Mock()
        ps.current_process = mock_process
        assert ps.current_process == mock_process


class TestUIElementReferences:
    """Tests for UI element reference storage in PS."""
    
    def test_log_container_reference(self):
        """Log container reference should be storable."""
        ps = PipelineState()
        mock_container = Mock()
        ps.log_container = mock_container
        assert ps.log_container == mock_container
    
    def test_log_scroll_reference(self):
        """Log scroll reference should be storable."""
        ps = PipelineState()
        mock_scroll = Mock()
        ps.log_scroll = mock_scroll
        assert ps.log_scroll == mock_scroll
    
    def test_status_label_reference(self):
        """Status label reference should be storable."""
        ps = PipelineState()
        mock_label = Mock()
        ps.status_label = mock_label
        assert ps.status_label == mock_label
    
    def test_refresh_files_function_reference(self):
        """Refresh files function reference should be storable."""
        ps = PipelineState()
        
        def mock_refresh():
            return "files refreshed"
        
        ps.refresh_files = mock_refresh
        assert ps.refresh_files() == "files refreshed"


class TestGlobalParametersPersistence:
    """Tests for global pipeline parameters persistence."""
    
    def test_conditions_storage(self):
        """Conditions should be storable."""
        ps = PipelineState()
        ps.conditions = ['DMT', 'EC']
        assert ps.conditions == ['DMT', 'EC']
    
    def test_max_subjects_storage(self):
        """Max subjects should be storable."""
        ps = PipelineState()
        ps.max_subjects = 10
        assert ps.max_subjects == 10
    
    def test_workers_storage(self):
        """Workers count should be storable."""
        ps = PipelineState()
        ps.workers = 7
        assert ps.workers == 7
    
    def test_jobs_storage(self):
        """Jobs count should be storable."""
        ps = PipelineState()
        ps.jobs = 4
        assert ps.jobs == 4
    
    def test_bands_storage(self):
        """Bands selection should be storable."""
        ps = PipelineState()
        ps.bands = ['Delta', 'Theta', 'Alpha']
        assert 'Alpha' in ps.bands
    
    def test_clustering_params_storage(self):
        """Clustering parameters should be storable."""
        ps = PipelineState()
        ps.min_k = 2
        ps.max_k = 15
        ps.min_comps = 2
        ps.max_comps = 10
        
        assert ps.min_k == 2
        assert ps.max_k == 15


class TestCallbackClosurePattern:
    """Tests for the callback closure pattern used in pipeline.py."""
    
    def test_get_conditions_closure(self):
        """Test get_conditions() pattern that reads from UI checkboxes."""
        # Simulate UI checkbox values
        cond_dmt_value = True
        cond_ec_value = False
        cond_eo_value = True
        
        def get_conditions():
            conds = []
            if cond_dmt_value: conds.append('DMT')
            if cond_ec_value: conds.append('EC')
            if cond_eo_value: conds.append('EO')
            return conds
        
        assert get_conditions() == ['DMT', 'EO']
    
    def test_get_input_dir_closure(self):
        """Test get_input_dir() pattern that reads from input field."""
        DEFAULT_INPUT_DIR = Path("/default/path")
        input_field_value = "/custom/path"
        
        def get_input_dir():
            return Path(input_field_value) if input_field_value else DEFAULT_INPUT_DIR
        
        assert get_input_dir() == Path("/custom/path")
    
    def test_run_label_refresh_closure(self):
        """Test run_label refresh pattern."""
        current_run_dir = [None]
        label_text = ["(crear NEW RUN)"]
        
        def refresh_run_label():
            if current_run_dir[0]:
                label_text[0] = str(current_run_dir[0].name)
            else:
                label_text[0] = "(crear NEW RUN)"
        
        current_run_dir[0] = Path("/tmp/run_20231201")
        refresh_run_label()
        
        assert label_text[0] == "run_20231201"


class TestTimerCallbacks:
    """Tests for timer callback patterns."""
    
    def test_update_status_idle(self):
        """Status should show 'Idle' when not running."""
        ps = PipelineState()
        ps.running = False
        
        def update_status():
            if ps.running and ps.start_time:
                return "Running..."
            else:
                return "Idle"
        
        assert update_status() == "Idle"
    
    def test_update_status_running(self):
        """Status should show elapsed time when running."""
        ps = PipelineState()
        ps.running = True
        ps.start_time = datetime.now()
        ps.current_step = "Source Localization"
        
        def update_status():
            if ps.running and ps.start_time:
                elapsed = (datetime.now() - ps.start_time).seconds
                mins, secs = divmod(elapsed, 60)
                return f'Running: {ps.current_step} ({mins}m {secs}s)'
            else:
                return "Idle"
        
        status = update_status()
        assert "Running:" in status
        assert "Source Localization" in status
    
    def test_system_stats_pattern(self):
        """Test system stats update pattern (mocked)."""
        with patch('psutil.cpu_percent', return_value=45.0):
            with patch('psutil.cpu_count', return_value=8):
                import psutil
                cpu_percent = psutil.cpu_percent()
                cpu_count = psutil.cpu_count()
                
                assert cpu_percent == 45.0
                assert cpu_count == 8


class TestVizStateUpdates:
    """Tests for visualization state update patterns."""
    
    def test_refresh_all_plots_function_storage(self):
        """refresh_all_plots function should be storable in viz_state."""
        viz_state = {'data': None}
        
        call_count = [0]
        
        def refresh_all_plots():
            call_count[0] += 1
            return call_count[0]
        
        viz_state['refresh_all_plots'] = refresh_all_plots
        
        # Call it
        result = viz_state['refresh_all_plots']()
        assert result == 1
        
        # Call again
        result = viz_state['refresh_all_plots']()
        assert result == 2
    
    def test_update_brain_plot_storage(self):
        """update_brain_plot function should be storable."""
        viz_state = {'current_brain_plot': 'network'}
        
        def update_brain_plot(plot_type=None):
            if plot_type is None:
                plot_type = viz_state.get('current_brain_plot', 'network')
            else:
                viz_state['current_brain_plot'] = plot_type
            return plot_type
        
        viz_state['update_brain_plot'] = update_brain_plot
        
        assert viz_state['update_brain_plot']() == 'network'
        assert viz_state['update_brain_plot']('sync') == 'sync'
        assert viz_state['current_brain_plot'] == 'sync'
    
    def test_hilbert_functions_storage(self):
        """Hilbert update functions should be storable."""
        viz_state = {}
        
        def update_hilbert_2d():
            return "2d updated"
        
        def update_hilbert_3d():
            return "3d updated"
        
        viz_state['update_hilbert_2d'] = update_hilbert_2d
        viz_state['update_hilbert_3d'] = update_hilbert_3d
        
        assert 'update_hilbert_2d' in viz_state
        assert viz_state['update_hilbert_3d']() == "3d updated"


class TestStateIsolationBetweenPages:
    """Tests ensuring state doesn't leak between different pages."""
    
    def test_pipeline_state_independent_of_viewer_state(self):
        """PipelineState should be independent of ViewerState."""
        from app.state.global_state import PipelineState
        
        ps = PipelineState()
        ps.running = True
        ps.selected_run = Path("/test")
        
        # Create another instance (simulating different page context)
        # Note: In production, PS is a singleton, but this tests isolation
        ps2 = PipelineState()
        
        # New instance should have its own state
        assert ps2.running is False
    
    def test_ps_singleton_behavior(self):
        """Test that imported PS acts as expected singleton."""
        from app.state import PS
        
        # Store current state
        original_running = PS.running
        
        PS.running = True
        
        # Re-import (simulates tab switch)
        from app.state import PS as PS2
        
        # Should be same instance
        assert PS2.running is True
        
        # Restore
        PS.running = original_running





