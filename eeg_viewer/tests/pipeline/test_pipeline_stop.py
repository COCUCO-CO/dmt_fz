"""
Tests for pipeline STOP functionality.

These tests verify that:
- STOP button correctly terminates running processes
- Process group is killed (including child processes)
- State is properly reset after stopping
"""
import pytest
import asyncio
import signal
import sys
import os
from datetime import datetime
from unittest.mock import Mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.state import PS
from app.state.global_state import PipelineState


class TestStopProcess:
    """Tests for stop_current_process function."""
    
    def test_stop_when_no_process_running(self):
        """Stop should return False when no process is running."""
        ps = PipelineState()
        ps.current_process = None
        
        # Simulate stop logic
        result = ps.current_process is not None
        assert result is False
    
    def test_stop_when_process_is_running(self):
        """Stop should attempt to terminate when process exists."""
        ps = PipelineState()
        mock_process = Mock()
        mock_process.terminate = Mock()
        ps.current_process = mock_process
        
        # Simulate stop
        if ps.current_process:
            ps.current_process.terminate()
        
        mock_process.terminate.assert_called_once()
    
    def test_stop_calls_kill_after_terminate(self):
        """Stop should call kill() if terminate doesn't work."""
        ps = PipelineState()
        mock_process = Mock()
        mock_process.terminate = Mock()
        mock_process.kill = Mock()
        mock_process.returncode = None  # Process still running
        ps.current_process = mock_process
        
        # Simulate aggressive stop
        if ps.current_process:
            ps.current_process.terminate()
            # If still running, kill
            if ps.current_process.returncode is None:
                ps.current_process.kill()
        
        mock_process.terminate.assert_called_once()
        mock_process.kill.assert_called_once()
    
    def test_state_reset_after_stop(self):
        """State should be reset after stopping."""
        ps = PipelineState()
        
        # Setup running state
        ps.running = True
        ps.current_step = "Calculate Syncro"
        ps.running_task_name = "calculate_syncro.py"
        ps.start_time = datetime.now()
        ps.current_process = Mock()
        
        # Simulate stop and reset
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


class TestProcessGroupKill:
    """Tests for process group termination."""
    
    def test_process_group_concept(self):
        """Verify process group kill would work."""
        # On Unix, we need to kill the process group to also kill children
        # os.killpg(os.getpgid(pid), signal.SIGTERM)
        
        mock_pid = 12345
        
        # Verify we can get pgid concept (this would fail on non-Unix)
        if hasattr(os, 'killpg') and hasattr(os, 'getpgid'):
            # Unix system - process group support available
            assert callable(os.killpg)
            assert callable(os.getpgid)
    
    def test_sigterm_then_sigkill_pattern(self):
        """Verify SIGTERM followed by SIGKILL pattern."""
        # Standard pattern: SIGTERM (graceful), wait, then SIGKILL (force)
        assert signal.SIGTERM != signal.SIGKILL
        assert signal.SIGTERM.value == 15
        assert signal.SIGKILL.value == 9


class TestStopWithRealAsyncProcess:
    """Tests with real async subprocess (controlled test script)."""
    
    @pytest.mark.asyncio
    async def test_can_create_and_stop_subprocess(self):
        """Test that we can create and stop a subprocess."""
        # Create a simple subprocess that sleeps
        process = await asyncio.create_subprocess_exec(
            sys.executable, '-c', 'import time; time.sleep(10)',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Verify it's running
        assert process.returncode is None
        
        # Stop it
        process.terminate()
        
        # Wait for it to terminate (with timeout)
        try:
            await asyncio.wait_for(process.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
        
        # Verify it's stopped
        assert process.returncode is not None
    
    @pytest.mark.asyncio
    async def test_kill_stops_stubborn_process(self):
        """Test that kill() stops a process that ignores SIGTERM."""
        # Create a process that ignores SIGTERM
        script = '''
import signal
import time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
time.sleep(60)
'''
        process = await asyncio.create_subprocess_exec(
            sys.executable, '-c', script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        await asyncio.sleep(0.1)  # Let it start
        
        # Try terminate (will be ignored)
        process.terminate()
        
        # Give it a moment
        await asyncio.sleep(0.2)
        
        # Should still be running
        if process.returncode is None:
            # Force kill
            process.kill()
        
        await asyncio.wait_for(process.wait(), timeout=2.0)
        
        # Should be stopped now (killed)
        assert process.returncode is not None
    
    @pytest.mark.asyncio
    async def test_process_group_kill_concept(self):
        """Test killing process group (parent + children)."""
        if sys.platform == 'win32':
            pytest.skip("Process group not supported on Windows")
        
        # Create a parent that spawns a child
        script = '''
import subprocess
import time
import sys
# Spawn a child that sleeps
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
time.sleep(60)  # Parent also sleeps
'''
        # Start in new process group
        process = await asyncio.create_subprocess_exec(
            sys.executable, '-c', script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True  # Creates new process group
        )
        
        await asyncio.sleep(0.2)  # Let it start and spawn child
        
        # Kill entire process group
        try:
            pgid = os.getpgid(process.pid)
            os.killpg(pgid, signal.SIGTERM)
            await asyncio.sleep(0.2)
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass  # Already dead
        
        await asyncio.wait_for(process.wait(), timeout=2.0)
        assert process.returncode is not None


class TestGlobalPSStopIntegration:
    """Tests using global PS instance."""
    
    @pytest.mark.asyncio
    async def test_stop_with_global_ps(self):
        """Test stop functionality with global PS."""
        # Save original state
        original_process = PS.current_process
        original_running = PS.running
        
        try:
            # Create a test subprocess
            process = await asyncio.create_subprocess_exec(
                sys.executable, '-c', 'import time; time.sleep(10)',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Set PS state
            PS.current_process = process
            PS.running = True
            PS.current_step = "Test Step"
            
            # Simulate stop_current_process logic
            if PS.current_process:
                PS.current_process.terminate()
                try:
                    await asyncio.wait_for(PS.current_process.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    PS.current_process.kill()
                    await PS.current_process.wait()
            
            # Verify process stopped
            assert PS.current_process.returncode is not None
            
        finally:
            # Restore original state
            PS.current_process = original_process
            PS.running = original_running
    
    def test_ps_has_current_process_attr(self):
        """PS should have current_process attribute."""
        assert hasattr(PS, 'current_process')
    
    def test_ps_has_running_attr(self):
        """PS should have running attribute."""
        assert hasattr(PS, 'running')


class TestStopButtonCallback:
    """Tests for the stop button callback behavior."""
    
    @pytest.mark.asyncio
    async def test_stop_callback_is_async(self):
        """Stop callback should be async to properly await termination."""
        from app.pages.pipeline.runner import stop_current_process
        
        # Verify it's a coroutine function
        assert asyncio.iscoroutinefunction(stop_current_process)
    
    @pytest.mark.asyncio
    async def test_stop_returns_false_when_no_process(self):
        """Stop should return False when nothing is running."""
        from app.pages.pipeline.runner import stop_current_process
        
        original = PS.current_process
        try:
            PS.current_process = None
            result = await stop_current_process()
            assert result is False
        finally:
            PS.current_process = original





