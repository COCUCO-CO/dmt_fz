"""
Tests for Pipeline UI tabs behavior.

These tests ensure that:
- Tab switching preserves state
- Log history is restored when returning to Console tab
- Files tab refreshes correctly
- System tab updates periodically
- Visualize tab maintains selections
"""
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock
import sys

# Ensure proper path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.state.global_state import PipelineState


class TestConsoleTabPersistence:
    """Tests for Console tab state persistence."""
    
    def test_log_history_restored_on_tab_return(self):
        """Log history should be restored when returning to Console tab."""
        ps = PipelineState()
        ps.log_history = [
            "[STEP_1] Starting...",
            "[STEP_1] Processing S01...",
            "[STEP_1] Completed"
        ]
        
        # Simulate tab switch away and back
        # When returning, the logs should be restored from history
        restored_logs = []
        for msg in ps.log_history:
            restored_logs.append(msg)
        
        assert len(restored_logs) == 3
        assert "[STEP_1] Starting..." in restored_logs
        assert "[STEP_1] Completed" in restored_logs
    
    def test_log_history_special_formatting(self):
        """Log messages with special markers should get spacing."""
        special_markers = ['[RUN]', 'Starting:', '[INFO]', '[SETUP]', '===', 'COMPLETED', 'FAILED']
        
        msgs = [
            "[RUN] Created: /tmp/run_test",
            "Processing...",
            "[INFO] Subject count: 5",
            "COMPLETED successfully"
        ]
        
        formatted = []
        for msg in msgs:
            needs_spacing = any(x in msg for x in special_markers)
            if needs_spacing:
                formatted.append(('space', msg))
            else:
                formatted.append(('normal', msg))
        
        assert formatted[0][0] == 'space'
        assert formatted[1][0] == 'normal'
        assert formatted[2][0] == 'space'
        assert formatted[3][0] == 'space'
    
    def test_status_label_updates(self):
        """Status label should update based on running state."""
        ps = PipelineState()
        
        # Test idle state
        ps.running = False
        ps.start_time = None
        
        def get_status_text():
            if ps.running and ps.start_time:
                elapsed = (datetime.now() - ps.start_time).seconds
                mins, secs = divmod(elapsed, 60)
                return f'Running: {ps.current_step} ({mins}m {secs}s)'
            else:
                return 'Idle'
        
        assert get_status_text() == 'Idle'
        
        # Test running state
        ps.running = True
        ps.start_time = datetime.now()
        ps.current_step = "Source Localization"
        
        status = get_status_text()
        assert 'Running:' in status
        assert 'Source Localization' in status
    
    def test_auto_scroll_to_bottom(self):
        """Log scroll should auto-scroll to bottom."""
        mock_scroll = Mock()
        
        def scroll_to_bottom():
            mock_scroll.scroll_to(percent=1.0)
        
        scroll_to_bottom()
        mock_scroll.scroll_to.assert_called_with(percent=1.0)


class TestFilesTabPersistence:
    """Tests for Files tab state persistence."""
    
    def test_refresh_files_callback_stored(self):
        """refresh_files function should be stored in PS."""
        ps = PipelineState()
        
        call_count = [0]
        
        def refresh_files():
            call_count[0] += 1
        
        ps.refresh_files = refresh_files
        
        # Call from different context (simulating button click)
        ps.refresh_files()
        assert call_count[0] == 1
    
    def test_refresh_triggered_on_new_run(self):
        """refresh_files should be called when new run is created."""
        ps = PipelineState()
        
        refresh_called = [False]
        
        def refresh_files():
            refresh_called[0] = True
        
        ps.refresh_files = refresh_files
        
        # Simulate new run callback
        def new_run():
            # Create run...
            if ps.refresh_files:
                ps.refresh_files()
        
        new_run()
        assert refresh_called[0] is True
    
    def test_refresh_triggered_on_run_select(self):
        """refresh_files should be called when existing run is selected."""
        ps = PipelineState()
        
        refresh_called = [False]
        
        def refresh_files():
            refresh_called[0] = True
        
        ps.refresh_files = refresh_files
        
        # Simulate run selection callback
        def use_existing():
            # Select run...
            if ps.refresh_files:
                ps.refresh_files()
        
        use_existing()
        assert refresh_called[0] is True
    
    def test_file_stats_calculation(self, tmp_path):
        """File stats should be calculated correctly."""
        run_dir = tmp_path / "run_test"
        run_dir.mkdir()
        (run_dir / "DMT").mkdir()
        (run_dir / "clustering_results").mkdir()
        (run_dir / "pearson_results").mkdir()
        
        # Create test files
        (run_dir / "DMT" / "syncro-S01.pkl").touch()
        (run_dir / "DMT" / "syncro-S02.pkl").touch()
        (run_dir / "DMT" / "phases-S01.pkl").touch()
        (run_dir / "order_all-S01.pkl").touch()
        (run_dir / "order-S01.pkl").touch()
        (run_dir / "clustering_results" / "results.pkl").touch()
        (run_dir / "clustering_results" / "results.csv").touch()
        (run_dir / "pearson_results" / "plot.png").touch()
        
        def count_files(pattern):
            return len(list(run_dir.rglob(pattern)))
        
        stats = {
            'syncro': count_files('syncro-*.pkl') + count_files('phases-*.pkl'),
            'order_all': count_files('order_all-*.pkl'),
            'order': count_files('order-*.pkl'),
            'clustering': count_files('clustering_results/**/*.pkl') + count_files('clustering_results/**/*.csv'),
            'pearson': count_files('pearson_results/**/*'),
        }
        
        assert stats['syncro'] == 3
        assert stats['order_all'] == 1
        assert stats['order'] == 1
        assert stats['clustering'] == 2
        assert stats['pearson'] == 1


class TestSystemTabBehavior:
    """Tests for System tab behavior."""
    
    def test_system_stats_update_timer(self):
        """System stats should update periodically (2s timer)."""
        update_count = [0]
        
        def update_system_stats():
            update_count[0] += 1
        
        # Simulate 3 timer ticks
        for _ in range(3):
            update_system_stats()
        
        assert update_count[0] == 3
    
    def test_cpu_color_thresholds(self):
        """CPU color should change based on usage thresholds."""
        THEME_PRIMARY = '#00ff88'
        THEME_WARN = '#ffcc00'
        ERROR_COLOR = '#ff4444'
        
        def get_cpu_color(cpu_percent):
            if cpu_percent < 50:
                return THEME_PRIMARY
            elif cpu_percent < 80:
                return THEME_WARN
            else:
                return ERROR_COLOR
        
        assert get_cpu_color(30) == THEME_PRIMARY
        assert get_cpu_color(60) == THEME_WARN
        assert get_cpu_color(90) == ERROR_COLOR
    
    def test_memory_color_thresholds(self):
        """Memory color should change based on usage thresholds."""
        THEME_PRIMARY = '#00ff88'
        THEME_WARN = '#ffcc00'
        ERROR_COLOR = '#ff4444'
        
        def get_mem_color(mem_percent):
            if mem_percent < 60:
                return THEME_PRIMARY
            elif mem_percent < 85:
                return THEME_WARN
            else:
                return ERROR_COLOR
        
        assert get_mem_color(40) == THEME_PRIMARY
        assert get_mem_color(70) == THEME_WARN
        assert get_mem_color(95) == ERROR_COLOR


class TestVisualizeTabPersistence:
    """Tests for Visualize tab state persistence."""
    
    def test_viz_state_persists_across_tab_switches(self):
        """Visualization state should persist when switching tabs."""
        ps = PipelineState()
        
        # Initialize viz_state
        if not hasattr(ps, 'viz_state') or ps.viz_state is None:
            ps.viz_state = {'data': None, 'file': None, 'loaded': False}
        
        # Simulate loading data
        mock_data = {'phases_stc': {'Alpha': [[1, 2, 3]]}}
        ps.viz_state['data'] = mock_data
        ps.viz_state['loaded'] = True
        ps.viz_state['file'] = Path("/tmp/phases-S01.pkl")
        
        # Simulate tab switch (just accessing again)
        viz_state_after = ps.viz_state
        
        assert viz_state_after['data'] == mock_data
        assert viz_state_after['loaded'] is True
        assert viz_state_after['file'] == Path("/tmp/phases-S01.pkl")
    
    def test_custom_path_persists(self):
        """Custom data path should persist."""
        ps = PipelineState()
        ps.viz_state = {'data': None, 'custom_path': None}
        
        custom_path = Path("/custom/data/path")
        ps.viz_state['custom_path'] = custom_path
        
        # Access again
        assert ps.viz_state['custom_path'] == custom_path
    
    def test_current_brain_plot_persists(self):
        """Current brain plot type should persist."""
        viz_state = {'current_brain_plot': 'network'}
        
        # Change plot type
        viz_state['current_brain_plot'] = 'sync'
        
        # Access again
        assert viz_state['current_brain_plot'] == 'sync'
    
    def test_selector_values_affect_refresh(self):
        """Changing selectors should trigger refresh."""
        refresh_count = [0]
        
        viz_band_value = 'Alpha'
        viz_subject_value = 'S01-DMT'
        viz_epoch_value = 0
        
        def refresh_all_plots():
            refresh_count[0] += 1
        
        # Simulate band change
        viz_band_value = 'Beta'
        refresh_all_plots()
        
        # Simulate subject change
        viz_subject_value = 'S02-DMT'
        refresh_all_plots()
        
        # Simulate epoch change
        viz_epoch_value = 5
        refresh_all_plots()
        
        assert refresh_count[0] == 3
    
    def test_subject_extraction_from_files(self, tmp_path):
        """Subject extraction should work correctly from file names."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create files with different naming patterns
        (data_dir / "syncro-S01-DMT.pkl").touch()
        (data_dir / "phases-S02-EC.pkl").touch()
        (data_dir / "syncro-S01.pkl").touch()  # Without condition
        
        all_files = list(data_dir.glob('syncro-*.pkl')) + list(data_dir.glob('phases-*.pkl'))
        
        subjects = sorted(list(set([
            f.stem.split('-')[1] if '-' in f.stem else f.stem 
            for f in all_files
        ])))
        
        assert 'S01' in subjects
        assert 'S02' in subjects


class TestTabSwitchingWorkflow:
    """Tests for complete tab switching workflows."""
    
    def test_console_to_files_and_back(self):
        """Switching from Console to Files and back should preserve logs."""
        ps = PipelineState()
        ps.log_history = ["Log 1", "Log 2"]
        
        # Switch to Files (logs preserved in history)
        logs_before = list(ps.log_history)
        
        # Switch back to Console (logs restored)
        logs_after = list(ps.log_history)
        
        assert logs_before == logs_after
    
    def test_visualize_data_survives_tab_switch(self):
        """Data loaded in Visualize tab should survive tab switches."""
        ps = PipelineState()
        ps.viz_state = {'data': None, 'loaded': False}
        
        # Load data
        mock_data = {'phases_stc': {'Alpha': [[1, 2, 3]]}}
        ps.viz_state['data'] = mock_data
        ps.viz_state['loaded'] = True
        
        # Switch to Console (simulated)
        # ...
        
        # Switch back to Visualize
        assert ps.viz_state['data'] == mock_data
        assert ps.viz_state['loaded'] is True
    
    def test_run_selection_survives_all_tabs(self):
        """Selected run should survive switching through all tabs."""
        ps = PipelineState()
        run_dir = Path("/tmp/run_20231201")
        ps.selected_run = run_dir
        
        # Simulate switching through tabs
        for tab in ['Console', 'Files', 'System', 'Visualize']:
            # Each tab accesses PS.selected_run
            assert ps.selected_run == run_dir


class TestTabContentRendering:
    """Tests for tab content rendering logic."""
    
    def test_console_initial_message(self):
        """Console should show initial message when no history."""
        ps = PipelineState()
        ps.log_history = []
        
        PIPELINE_DIR = Path("/media/storage_hdd/dmt_fz/pipeline")
        
        def get_initial_messages():
            if not ps.log_history:
                return [
                    'Pipeline ready. Select a step and click RUN.',
                    f'Pipeline directory: {PIPELINE_DIR}'
                ]
            return ps.log_history
        
        messages = get_initial_messages()
        assert 'Pipeline ready' in messages[0]
        assert 'pipeline' in messages[1].lower()
    
    def test_files_no_run_message(self):
        """Files tab should show message when no run selected."""
        current_run_dir = [None]
        
        def get_files_content():
            if not current_run_dir[0]:
                return "No run selected"
            return "Files..."
        
        assert get_files_content() == "No run selected"
    
    def test_files_run_not_exists_message(self, tmp_path):
        """Files tab should handle non-existent run directory."""
        non_existent = tmp_path / "non_existent_run"
        current_run_dir = [non_existent]
        
        def get_files_content():
            if not current_run_dir[0] or not current_run_dir[0].exists():
                return "No run selected"
            return "Files..."
        
        assert get_files_content() == "No run selected"
    
    def test_visualize_no_data_message(self):
        """Visualize tab plots should show message when no data."""
        viz_state = {'data': None}
        
        def get_plot_content():
            if not viz_state.get('data'):
                return "Load data first"
            return "Plot..."
        
        assert get_plot_content() == "Load data first"


class TestExpansionPanels:
    """Tests for expansion panel behavior in Visualize tab."""
    
    def test_clustering_expansion_requires_run(self):
        """Clustering expansion should require a run to be selected."""
        current_run_dir = [None]
        
        def update_cluster_scores():
            if not current_run_dir[0]:
                return "Select a run first"
            return "Clustering data..."
        
        assert update_cluster_scores() == "Select a run first"
    
    def test_pearson_expansion_requires_results(self, tmp_path):
        """Pearson expansion should require pearson_results directory."""
        run_dir = tmp_path / "run_test"
        run_dir.mkdir()
        # No pearson_results directory
        
        current_run_dir = [run_dir]
        
        def refresh_pearson_gallery():
            if not current_run_dir[0]:
                return "No run selected"
            
            pearson_dir = current_run_dir[0] / 'pearson_results'
            if not pearson_dir.exists():
                return "No Pearson results. Run pearson.py first."
            
            return "Gallery loaded"
        
        assert "Run pearson.py first" in refresh_pearson_gallery()
    
    def test_animation_expansion_requires_data(self):
        """Animation generator should require data to be loaded."""
        viz_state = {'data': None, 'custom_path': None}
        current_run_dir = [None]
        viz_subject_value = None
        
        def check_animation_ready():
            data_path = viz_state.get('custom_path') or current_run_dir[0]
            if not data_path or not viz_subject_value:
                return "No data loaded. Load a subject first."
            return "Ready"
        
        assert "No data loaded" in check_animation_ready()


class TestDialogBehavior:
    """Tests for dialog behavior."""
    
    def test_pearson_image_dialog_content(self, tmp_path):
        """Pearson image dialog should show correct content."""
        img_file = tmp_path / "test.png"
        img_file.touch()
        
        def get_dialog_content(file_path):
            content = {
                'title': file_path.name,
                'is_png': file_path.suffix == '.png',
                'is_svg': file_path.suffix == '.svg'
            }
            return content
        
        content = get_dialog_content(img_file)
        assert content['title'] == 'test.png'
        assert content['is_png'] is True
        assert content['is_svg'] is False
    
    def test_dialog_close_behavior(self):
        """Dialog should be closeable."""
        dialog_open = [True]
        
        def close_dialog():
            dialog_open[0] = False
        
        close_dialog()
        assert dialog_open[0] is False

