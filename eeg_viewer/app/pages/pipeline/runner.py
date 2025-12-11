"""
Pipeline execution engine.

Single Responsibility: Handles subprocess execution for pipeline scripts.
Open/Closed: Execution logic is isolated and can be extended without modification.
"""
import asyncio
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from nicegui import ui

from app.state import PS

from .config import (
    PIPELINE_DIR,
    PROGRESS_BAR_PATTERNS,
    MAX_LOG_LINE_LENGTH,
)
from .utils import pipeline_log


async def run_pipeline_step(
    script_name: str,
    args_list: list[str],
    step_name: str,
    output_dir: Path = None,
    input_dir: Path = None
) -> bool:
    """
    Execute a pipeline script as a subprocess with real-time logging.
    
    This function:
    1. Checks if pipeline is already running
    2. Sets up environment variables
    3. Executes the script with unbuffered output
    4. Captures and logs stdout/stderr in real-time
    5. Handles errors and cleanup
    
    Args:
        script_name: Name of the script (e.g., 'fwd.py')
        args_list: List of command-line arguments
        step_name: Human-readable step name for logging
        output_dir: Optional output directory (sets PIPELINE_OUTPUT_DIR env var)
        input_dir: Optional input directory (sets PIPELINE_INPUT_DIR env var)
        
    Returns:
        True if execution succeeded, False otherwise
    """
    # Check if already running
    if PS.running:
        try:
            ui.notify('Pipeline already running', type='warning')
        except RuntimeError:
            pass
        return False
    
    # Update state
    PS.running = True
    PS.current_step = step_name
    PS.running_task_name = script_name
    PS.start_time = datetime.now()
    
    # Validate script exists
    script_path = PIPELINE_DIR / script_name
    if not script_path.exists():
        try:
            ui.notify(f'Script not found: {script_path}', type='negative')
        except RuntimeError:
            pass
        _reset_state()
        return False
    
    # Build command
    cmd = ["python", "-u", str(script_path)] + args_list
    pipeline_log(f"[{step_name}] Starting: {' '.join(cmd)}")
    
    # Set up environment
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    
    if input_dir:
        env['PIPELINE_INPUT_DIR'] = str(input_dir)
        pipeline_log(f"[{step_name}] Input dir: {input_dir}")
    
    if output_dir:
        env['PIPELINE_OUTPUT_DIR'] = str(output_dir)
        pipeline_log(f"[{step_name}] Output dir: {output_dir}")
    
    try:
        # Create subprocess with larger buffer for tqdm progress bars
        # Use start_new_session=True to create a process group for proper termination
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(PIPELINE_DIR.parent.parent),
            env=env,
            limit=1024 * 1024,  # 1MB buffer limit
            start_new_session=(sys.platform != 'win32')  # Create process group on Unix
        )
        
        PS.current_process = process
        
        # Read both streams concurrently
        await asyncio.gather(
            _read_stream(process.stdout, step_name, is_stderr=False),
            _read_stream(process.stderr, step_name, is_stderr=True)
        )
        
        await process.wait()
        
        # Handle completion
        if process.returncode == 0:
            pipeline_log(f"[{step_name}] Completed successfully")
            try:
                ui.notify(f'{step_name} completed!', type='positive')
            except RuntimeError:
                pass
            return True
        else:
            pipeline_log(f"[{step_name}] Failed with code {process.returncode}")
            try:
                ui.notify(f'{step_name} failed', type='negative')
            except RuntimeError:
                pass
            return False
            
    except Exception as e:
        import traceback
        pipeline_log(f"[{step_name}] ERROR: {str(e)}")
        pipeline_log(traceback.format_exc())
        try:
            ui.notify(f'Error: {e}', type='negative')
        except RuntimeError:
            pass
        return False
    finally:
        _reset_state()


async def _read_stream(stream, step_name: str, is_stderr: bool = False) -> None:
    """
    Read a stream in chunks and log output in real-time.
    
    Handles:
    - Chunked reading for large outputs
    - Line buffering
    - Progress bar filtering
    - Line truncation
    
    Args:
        stream: asyncio stream to read
        step_name: Step name for log context
        is_stderr: Whether this is stderr (adds [WARN] prefix)
    """
    buffer = b''
    
    while True:
        try:
            chunk = await stream.read(8192)  # 8KB chunks
            
            if not chunk:
                # Process remaining buffer
                if buffer:
                    _process_log_line(buffer, is_stderr)
                break
            
            buffer += chunk
            
            # Process complete lines
            while b'\n' in buffer:
                line, buffer = buffer.split(b'\n', 1)
                _process_log_line(line, is_stderr)
                await asyncio.sleep(0.01)  # Allow UI updates
            
            # Prevent buffer overflow
            if len(buffer) > 100 * 1024:
                buffer = b''
                
        except Exception:
            await asyncio.sleep(0.1)
            continue


def _process_log_line(line: bytes, is_stderr: bool) -> None:
    """
    Process and log a single line of output.
    
    Args:
        line: Raw bytes from subprocess
        is_stderr: Whether from stderr
    """
    try:
        text = line.decode('utf-8', errors='replace').strip()
        
        if not text:
            return
            
        # Skip progress bars
        if any(pattern in text for pattern in PROGRESS_BAR_PATTERNS):
            return
        
        # Truncate long lines
        if len(text) > MAX_LOG_LINE_LENGTH:
            text = text[:MAX_LOG_LINE_LENGTH] + '...'
        
        # Log with appropriate prefix
        if is_stderr:
            pipeline_log(f"[WARN] {text}")
        else:
            pipeline_log(text)
            
    except Exception:
        pass


def _reset_state() -> None:
    """Reset pipeline state after execution."""
    PS.running = False
    PS.current_step = ""
    PS.running_task_name = ""
    PS.start_time = None
    PS.current_process = None


async def stop_current_process() -> bool:
    """
    Stop the currently running pipeline process.
    
    This function:
    1. Tries SIGTERM on the process group (kills parent + children)
    2. Waits briefly for graceful shutdown
    3. Forces SIGKILL if process doesn't respond
    4. Resets pipeline state
    
    Returns:
        True if a process was stopped, False if nothing was running
    """
    if not PS.current_process:
        return False
    
    process = PS.current_process
    step_name = PS.current_step or "Unknown"
    
    try:
        # Try to kill the entire process group (Unix only)
        if sys.platform != 'win32' and process.pid:
            try:
                pgid = os.getpgid(process.pid)
                os.killpg(pgid, signal.SIGTERM)
                pipeline_log(f"[{step_name}] Sending SIGTERM to process group {pgid}")
            except (ProcessLookupError, OSError):
                # Process already dead or no permission
                pass
        else:
            # Windows or fallback: just terminate the main process
            process.terminate()
        
        # Wait briefly for graceful shutdown
        try:
            await asyncio.wait_for(process.wait(), timeout=2.0)
            pipeline_log(f"[{step_name}] Process terminated gracefully")
        except asyncio.TimeoutError:
            # Process didn't respond to SIGTERM, force kill
            pipeline_log(f"[{step_name}] Process didn't respond, sending SIGKILL")
            
            if sys.platform != 'win32' and process.pid:
                try:
                    pgid = os.getpgid(process.pid)
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
            else:
                process.kill()
            
            # Wait for kill to complete
            try:
                await asyncio.wait_for(process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pipeline_log(f"[{step_name}] WARNING: Process may still be running")
        
        pipeline_log(f"[{step_name}] STOPPED by user")
        
        try:
            ui.notify('Pipeline stopped', type='warning')
        except RuntimeError:
            pass
        
        # Reset state after stopping
        _reset_state()
        
        return True
        
    except Exception as e:
        pipeline_log(f"[{step_name}] Error stopping: {e}")
        _reset_state()
        return False

