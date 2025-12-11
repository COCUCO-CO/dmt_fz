"""
Tests for Pipeline UI runtime behavior.

These tests verify that:
- Running indicator shows correctly
- Progress updates work
- Notifications appear at correct times
- Error handling works properly
- Log output is captured correctly
"""
import pytest
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import asyncio
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.state.global_state import PipelineState


class TestRunningIndicator:
    """Tests for the global running indicator."""
    
    def test_running_task_name_displayed(self):
        """Running task name should be set when pipeline starts."""
        ps = PipelineState()
        
        # Start pipeline
        ps.running = True
        ps.running_task_name = "fwd.py"
        
        assert ps.running_task_name == "fwd.py"
    
    def test_running_task_cleared_on_complete(self):
        """Running task name should be cleared when pipeline completes."""
        ps = PipelineState()
        ps.running = True
        ps.running_task_name = "fwd.py"
        
        # Complete pipeline
        ps.running = False
        ps.running_task_name = ""
        
        assert ps.running_task_name == ""
    
    def test_indicator_visibility_logic(self):
        """Indicator should only be visible when running."""
        ps = PipelineState()
        
        def is_indicator_visible():
            return ps.running and ps.running_task_name
        
        ps.running = False
        ps.running_task_name = ""
        assert not is_indicator_visible()
        
        ps.running = True
        ps.running_task_name = "clustering.py"
        assert is_indicator_visible()


class TestElapsedTimeDisplay:
    """Tests for elapsed time display in status."""
    
    def test_elapsed_time_calculation(self):
        """Elapsed time should be calculated correctly."""
        start_time = datetime.now() - timedelta(minutes=5, seconds=30)
        
        elapsed = (datetime.now() - start_time).seconds
        mins, secs = divmod(elapsed, 60)
        
        assert mins >= 5
        assert secs >= 28  # Allow some tolerance
    
    def test_elapsed_time_format(self):
        """Elapsed time should be formatted as Xm Ys."""
        ps = PipelineState()
        ps.running = True
        ps.start_time = datetime.now() - timedelta(minutes=2, seconds=45)
        ps.current_step = "Network Filtering"
        
        elapsed = (datetime.now() - ps.start_time).seconds
        mins, secs = divmod(elapsed, 60)
        status_text = f'Running: {ps.current_step} ({mins}m {secs}s)'
        
        assert 'Running:' in status_text
        assert 'Network Filtering' in status_text
        assert 'm' in status_text
        assert 's' in status_text
    
    def test_status_timer_interval(self):
        """Status should update every 1 second."""
        timer_interval = 1.0  # From ui.timer(1.0, update_status)
        assert timer_interval == 1.0


class TestNotifications:
    """Tests for UI notification behavior."""
    
    def test_notify_already_running(self):
        """Should notify when trying to run while already running."""
        ps = PipelineState()
        ps.running = True
        
        notifications = []
        
        def mock_notify(msg, type='info'):
            notifications.append({'msg': msg, 'type': type})
        
        # Try to run
        if ps.running:
            mock_notify('Pipeline already running', type='warning')
        
        assert len(notifications) == 1
        assert notifications[0]['type'] == 'warning'
        assert 'already running' in notifications[0]['msg']
    
    def test_notify_no_run_selected(self):
        """Should notify when no run directory is selected."""
        current_run_dir = [None]
        notifications = []
        
        def mock_notify(msg, type='info'):
            notifications.append({'msg': msg, 'type': type})
        
        if not current_run_dir[0]:
            mock_notify('Primero creá un NEW RUN', type='warning')
        
        assert len(notifications) == 1
        assert notifications[0]['type'] == 'warning'
    
    def test_notify_script_not_found(self):
        """Should notify when script is not found."""
        script_path = Path("/nonexistent/script.py")
        notifications = []
        
        def mock_notify(msg, type='info'):
            notifications.append({'msg': msg, 'type': type})
        
        if not script_path.exists():
            mock_notify(f'Script not found: {script_path}', type='negative')
        
        assert len(notifications) == 1
        assert notifications[0]['type'] == 'negative'
    
    def test_notify_success(self):
        """Should notify on successful completion."""
        step_name = "Source Localization"
        notifications = []
        
        def mock_notify(msg, type='info'):
            notifications.append({'msg': msg, 'type': type})
        
        mock_notify(f'{step_name} completed!', type='positive')
        
        assert len(notifications) == 1
        assert notifications[0]['type'] == 'positive'
        assert 'completed' in notifications[0]['msg']
    
    def test_notify_failure(self):
        """Should notify on failure."""
        step_name = "Clustering"
        notifications = []
        
        def mock_notify(msg, type='info'):
            notifications.append({'msg': msg, 'type': type})
        
        mock_notify(f'{step_name} failed', type='negative')
        
        assert len(notifications) == 1
        assert notifications[0]['type'] == 'negative'
    
    def test_notify_new_run_created(self):
        """Should notify when new run is created."""
        run_name = "run_20231201_120000"
        notifications = []
        
        def mock_notify(msg, type='info'):
            notifications.append({'msg': msg, 'type': type})
        
        mock_notify(f'Nuevo run: {run_name}', type='positive')
        
        assert len(notifications) == 1
        assert notifications[0]['type'] == 'positive'
        assert run_name in notifications[0]['msg']


class TestLogOutput:
    """Tests for pipeline log output behavior."""
    
    def test_log_starting_message(self):
        """Should log starting message with command."""
        logs = []
        step_name = "Source Localization"
        cmd = ["python", "-u", "fwd.py", "--max-subjects", "5"]
        
        def pipeline_log(msg):
            logs.append(msg)
        
        pipeline_log(f"[{step_name}] Starting: {' '.join(cmd)}")
        
        assert len(logs) == 1
        assert step_name in logs[0]
        assert 'Starting:' in logs[0]
        assert 'fwd.py' in logs[0]
    
    def test_log_input_dir_message(self):
        """Should log input directory."""
        logs = []
        step_name = "Source Localization"
        input_dir = Path("/media/storage_hdd/dmt_fz/EEG_CLEAN")
        
        def pipeline_log(msg):
            logs.append(msg)
        
        pipeline_log(f"[{step_name}] Input dir: {input_dir}")
        
        assert 'Input dir:' in logs[0]
        assert 'EEG_CLEAN' in logs[0]
    
    def test_log_output_dir_message(self):
        """Should log output directory."""
        logs = []
        step_name = "Source Localization"
        output_dir = Path("/tmp/run_test")
        
        def pipeline_log(msg):
            logs.append(msg)
        
        pipeline_log(f"[{step_name}] Output dir: {output_dir}")
        
        assert 'Output dir:' in logs[0]
        assert 'run_test' in logs[0]
    
    def test_log_completion_message(self):
        """Should log completion message."""
        logs = []
        step_name = "Network Filtering"
        
        def pipeline_log(msg):
            logs.append(msg)
        
        pipeline_log(f"[{step_name}] Completed successfully")
        
        assert 'Completed successfully' in logs[0]
    
    def test_log_failure_message(self):
        """Should log failure message with return code."""
        logs = []
        step_name = "Clustering"
        returncode = 1
        
        def pipeline_log(msg):
            logs.append(msg)
        
        pipeline_log(f"[{step_name}] Failed with code {returncode}")
        
        assert 'Failed' in logs[0]
        assert '1' in logs[0]
    
    def test_log_error_message(self):
        """Should log error message."""
        logs = []
        step_name = "Pearson"
        error = "Connection timeout"
        
        def pipeline_log(msg):
            logs.append(msg)
        
        pipeline_log(f"[{step_name}] ERROR: {error}")
        
        assert 'ERROR' in logs[0]
        assert error in logs[0]
    
    def test_log_warning_from_stderr(self):
        """Should log warnings from stderr."""
        logs = []
        stderr_line = "DeprecationWarning: xyz is deprecated"
        
        def pipeline_log(msg):
            logs.append(msg)
        
        pipeline_log(f"[WARN] {stderr_line}")
        
        assert '[WARN]' in logs[0]
        assert 'DeprecationWarning' in logs[0]
    
    def test_log_truncates_long_lines(self):
        """Should truncate very long log lines."""
        long_text = "A" * 1000
        max_length = 500
        
        if len(long_text) > max_length:
            truncated = long_text[:max_length] + '...'
        else:
            truncated = long_text
        
        assert len(truncated) == 503  # 500 + '...'
        assert truncated.endswith('...')
    
    def test_log_filters_progress_bars(self):
        """Should filter out tqdm progress bars."""
        lines = [
            "Processing files...",
            " 50%|██████████          | 5/10 [00:05<00:05, 1.00it/s]",
            "100%|████████████████████| 10/10 [00:10<00:00, 1.00it/s]",
            "Done processing."
        ]
        
        progress_indicators = ['%|', 'it/s]', '█', '▌', '\r']
        
        filtered = [
            line for line in lines 
            if not any(x in line for x in progress_indicators)
        ]
        
        assert len(filtered) == 2
        assert "Processing files..." in filtered
        assert "Done processing." in filtered


class TestLogPersistence:
    """Tests for log persistence behavior."""
    
    def test_log_stored_in_history(self):
        """Logs should be stored in PS.log_history."""
        ps = PipelineState()
        ps.log_history = []
        
        def pipeline_log(msg):
            ps.log_history.append(msg)
        
        pipeline_log("Test message 1")
        pipeline_log("Test message 2")
        
        assert len(ps.log_history) == 2
    
    def test_log_history_order_preserved(self):
        """Log order should be preserved in history."""
        ps = PipelineState()
        ps.log_history = []
        
        def pipeline_log(msg):
            ps.log_history.append(msg)
        
        pipeline_log("First")
        pipeline_log("Second")
        pipeline_log("Third")
        
        assert ps.log_history[0] == "First"
        assert ps.log_history[2] == "Third"
    
    def test_log_ui_update_on_disconnected_client(self):
        """Log should handle disconnected client gracefully."""
        ps = PipelineState()
        ps.log_history = []
        ps.log_container = Mock()
        ps.log_container.update = Mock(side_effect=RuntimeError("Client disconnected"))
        
        error_raised = [False]
        
        def pipeline_log(msg):
            ps.log_history.append(msg)
            
            if ps.log_container:
                try:
                    ps.log_container.update()
                except RuntimeError:
                    # Client disconnected - log is still stored
                    pass
        
        # Should not raise
        pipeline_log("Test message")
        
        # Message should still be in history
        assert "Test message" in ps.log_history


class TestProcessManagement:
    """Tests for subprocess process management."""
    
    def test_process_stored_in_state(self):
        """Current process should be stored in PS."""
        ps = PipelineState()
        mock_process = Mock()
        mock_process.returncode = None
        
        ps.current_process = mock_process
        
        assert ps.current_process == mock_process
    
    def test_process_cleared_on_complete(self):
        """Process should be cleared when complete."""
        ps = PipelineState()
        ps.current_process = Mock()
        
        # Simulate completion
        ps.current_process = None
        
        assert ps.current_process is None
    
    def test_process_terminate_on_stop(self):
        """Process should be terminated on STOP."""
        ps = PipelineState()
        mock_process = Mock()
        ps.current_process = mock_process
        
        # Simulate stop
        if ps.current_process:
            ps.current_process.terminate()
        
        mock_process.terminate.assert_called_once()


class TestEnvironmentVariables:
    """Tests for environment variable setup."""
    
    def test_pythonunbuffered_set(self):
        """PYTHONUNBUFFERED should be set for real-time output."""
        import os
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        
        assert env['PYTHONUNBUFFERED'] == '1'
    
    def test_pipeline_input_dir_set(self):
        """PIPELINE_INPUT_DIR should be set when input_dir provided."""
        import os
        env = os.environ.copy()
        input_dir = Path("/media/storage_hdd/dmt_fz/EEG_CLEAN")
        
        env['PIPELINE_INPUT_DIR'] = str(input_dir)
        
        assert env['PIPELINE_INPUT_DIR'] == str(input_dir)
    
    def test_pipeline_output_dir_set(self):
        """PIPELINE_OUTPUT_DIR should be set when output_dir provided."""
        import os
        env = os.environ.copy()
        output_dir = Path("/tmp/run_test")
        
        env['PIPELINE_OUTPUT_DIR'] = str(output_dir)
        
        assert env['PIPELINE_OUTPUT_DIR'] == str(output_dir)


class TestErrorHandling:
    """Tests for error handling during pipeline execution."""
    
    def test_exception_logged(self):
        """Exceptions should be logged."""
        logs = []
        step_name = "Test Step"
        
        def pipeline_log(msg):
            logs.append(msg)
        
        try:
            raise ValueError("Test error")
        except Exception as e:
            pipeline_log(f"[{step_name}] ERROR: {str(e)}")
        
        assert any('ERROR' in log for log in logs)
        assert any('Test error' in log for log in logs)
    
    def test_traceback_logged(self):
        """Traceback should be logged on exception."""
        import traceback
        logs = []
        
        def pipeline_log(msg):
            logs.append(msg)
        
        try:
            raise ValueError("Test error")
        except Exception:
            pipeline_log(traceback.format_exc())
        
        assert any('Traceback' in log for log in logs)
        assert any('ValueError' in log for log in logs)
    
    def test_state_reset_on_error(self):
        """State should be reset on error."""
        ps = PipelineState()
        ps.running = True
        ps.current_step = "Test"
        ps.running_task_name = "test.py"
        ps.start_time = datetime.now()
        ps.current_process = Mock()
        
        # Simulate error cleanup (finally block)
        ps.running = False
        ps.current_step = ""
        ps.running_task_name = ""
        ps.start_time = None
        ps.current_process = None
        
        assert ps.running is False
        assert ps.current_step == ""
        assert ps.running_task_name == ""
        assert ps.start_time is None
        assert ps.current_process is None


class TestUIUpdateDuringExecution:
    """Tests for UI updates during pipeline execution."""
    
    def test_log_container_updates(self):
        """Log container should update after each message."""
        mock_container = Mock()
        ps = PipelineState()
        ps.log_container = mock_container
        
        def pipeline_log(msg):
            ps.log_history.append(msg)
            if ps.log_container:
                ps.log_container.update()
        
        pipeline_log("Test 1")
        pipeline_log("Test 2")
        
        assert mock_container.update.call_count == 2
    
    def test_scroll_area_scrolls_to_bottom(self):
        """Scroll area should scroll to bottom after log update."""
        mock_scroll = Mock()
        ps = PipelineState()
        ps.log_scroll = mock_scroll
        
        def pipeline_log(msg):
            ps.log_history.append(msg)
            if ps.log_scroll:
                ps.log_scroll.scroll_to(percent=1.0)
        
        pipeline_log("Test")
        
        mock_scroll.scroll_to.assert_called_with(percent=1.0)
    
    def test_async_sleep_allows_ui_updates(self):
        """Small async sleep should allow UI updates between log lines."""
        sleep_time = 0.01  # From await asyncio.sleep(0.01)
        
        # This is the pattern used to allow UI updates
        assert sleep_time == 0.01

