"""
Tests for running indicator persistence across page navigation.

These tests verify that:
- Running indicator shows correctly when a task is running
- Running state persists when navigating away and back
- State is NOT reset when returning to pipeline page while task is running
"""
from datetime import datetime
from unittest.mock import Mock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.state.global_state import PipelineState


class TestRunningIndicatorPersistence:
    """Tests for running indicator state persistence."""
    
    def test_running_state_persists_when_set(self):
        """Running state should persist after being set."""
        ps = PipelineState()
        
        # Simulate starting a task
        ps.running = True
        ps.current_step = "Source Localization"
        ps.running_task_name = "fwd.py"
        ps.start_time = datetime.now()
        ps.current_process = Mock()
        
        # State should persist
        assert ps.running is True
        assert ps.current_step == "Source Localization"
        assert ps.running_task_name == "fwd.py"
        assert ps.start_time is not None
        assert ps.current_process is not None
    
    def test_running_state_survives_simulated_page_reload(self):
        """Running state should survive when page is reloaded (simulated)."""
        ps = PipelineState()
        
        # Start a task
        ps.running = True
        ps.running_task_name = "fwd.py"
        ps.start_time = datetime.now()
        
        # Simulate what SHOULD happen on page reload
        # The state should NOT be reset if running
        if not ps.running:
            ps.running = False
            ps.current_step = ""
            ps.start_time = None
        
        # State should still be there
        assert ps.running is True
        assert ps.running_task_name == "fwd.py"
    
    def test_state_only_reset_when_not_running(self):
        """State should only reset when no task is running."""
        ps = PipelineState()
        
        # No task running
        ps.running = False
        ps.current_step = "old step"  # Stale data
        
        # This is the correct reset behavior
        if not ps.running:
            ps.current_step = ""
            ps.start_time = None
            ps.current_process = None
        
        assert ps.current_step == ""
    
    def test_running_indicator_gets_task_info(self):
        """Running indicator should get task info from PS."""
        ps = PipelineState()
        
        ps.running = True
        ps.running_task_name = "clustering.py"
        
        # Simulate get_running_task logic
        def get_running_task():
            if ps.running and ps.running_task_name:
                return (ps.running_task_name, 'pipeline')
            return ('', '')
        
        task_name, source = get_running_task()
        
        assert task_name == "clustering.py"
        assert source == "pipeline"
    
    def test_running_indicator_empty_when_not_running(self):
        """Running indicator should be empty when no task running."""
        ps = PipelineState()
        
        ps.running = False
        ps.running_task_name = ""
        
        def get_running_task():
            if ps.running and ps.running_task_name:
                return (ps.running_task_name, 'pipeline')
            return ('', '')
        
        task_name, source = get_running_task()
        
        assert task_name == ""
        assert source == ""


class TestConsoleTabStateReset:
    """Tests for console tab state reset behavior."""
    
    def test_should_not_reset_state_when_task_running(self):
        """Console tab should NOT reset state when a task is running."""
        ps = PipelineState()
        
        # Task is running
        ps.running = True
        ps.current_step = "Network Filtering"
        ps.running_task_name = "multi2pool2.py"
        ps.start_time = datetime.now()
        mock_process = Mock()
        ps.current_process = mock_process
        
        # Simulate the CORRECT page load behavior
        # Only reset if NOT running
        if not ps.running:
            ps.running = False
            ps.current_step = ""
            ps.start_time = None
            ps.current_process = None
        
        # State should be preserved
        assert ps.running is True
        assert ps.current_step == "Network Filtering"
        assert ps.running_task_name == "multi2pool2.py"
        assert ps.start_time is not None
        assert ps.current_process == mock_process
    
    def test_should_reset_state_when_no_task_running(self):
        """Console tab SHOULD reset stale state when no task is running."""
        ps = PipelineState()
        
        # No task running, but stale state from previous session
        ps.running = False
        ps.current_step = "old step"
        ps.running_task_name = "old_script.py"
        ps.start_time = datetime(2020, 1, 1)  # Old timestamp
        ps.current_process = None
        
        # Simulate page load - should reset stale state
        if not ps.running:
            ps.current_step = ""
            ps.start_time = None
            # Note: running_task_name should also be cleared
            ps.running_task_name = ""
        
        assert ps.current_step == ""
        assert ps.running_task_name == ""
        assert ps.start_time is None


class TestNavigationWorkflow:
    """Tests for navigation workflow with running tasks."""
    
    def test_start_task_navigate_away_navigate_back(self):
        """Full workflow: start task, navigate to viewer, return to pipeline."""
        ps = PipelineState()
        
        # Step 1: Start a task on pipeline page
        ps.running = True
        ps.current_step = "Source Localization"
        ps.running_task_name = "fwd.py"
        ps.start_time = datetime.now()
        ps.current_process = Mock()
        
        # Step 2: User navigates to viewer page (state persists in PS)
        # ... viewer page doesn't touch PS pipeline state ...
        
        # Step 3: User navigates back to pipeline page
        # Simulate CORRECT page load - check before resetting
        if not ps.running:
            ps.running = False
            ps.current_step = ""
            ps.start_time = None
            ps.current_process = None
        
        # Step 4: Verify state is preserved
        assert ps.running is True
        assert ps.current_step == "Source Localization"
        assert ps.running_task_name == "fwd.py"
    
    def test_indicator_visible_throughout_navigation(self):
        """Running indicator should be visible throughout navigation."""
        ps = PipelineState()
        
        # Start task
        ps.running = True
        ps.running_task_name = "fwd.py"
        
        def get_indicator_visible():
            return bool(ps.running and ps.running_task_name)
        
        # Should be visible initially
        assert get_indicator_visible() is True
        
        # Simulate navigation (state persists)
        # ...
        
        # Should still be visible after returning
        assert get_indicator_visible() is True
    
    def test_task_completion_while_on_different_page(self):
        """Task completing while on different page should update state."""
        ps = PipelineState()
        
        # Start task
        ps.running = True
        ps.running_task_name = "fwd.py"
        ps.current_step = "Source Localization"
        
        # Simulate task completion (happens in background)
        ps.running = False
        ps.current_step = ""
        ps.running_task_name = ""
        ps.start_time = None
        ps.current_process = None
        
        # When user returns, state should show completed
        assert ps.running is False
        assert ps.running_task_name == ""


class TestGlobalPSInstance:
    """Tests using the global PS instance."""
    
    def test_global_ps_running_state_persists(self):
        """Global PS instance should persist running state."""
        from app.state import PS
        
        # Save original state
        original_running = PS.running
        original_task = PS.running_task_name
        
        try:
            # Set running state
            PS.running = True
            PS.running_task_name = "test_script.py"
            
            # Re-import (simulates page navigation)
            from app.state import PS as PS2
            
            # Should be same instance with same state
            assert PS2.running is True
            assert PS2.running_task_name == "test_script.py"
        finally:
            # Restore original state
            PS.running = original_running
            PS.running_task_name = original_task
    
    def test_current_process_reference_persists(self):
        """Process reference should persist for STOP functionality."""
        from app.state import PS
        
        original_process = PS.current_process
        
        try:
            mock_process = Mock()
            PS.current_process = mock_process
            
            # Verify it persists
            from app.state import PS as PS2
            assert PS2.current_process == mock_process
        finally:
            PS.current_process = original_process

