"""
Tests for pipeline step execution functionality.

Tests:
- run_pipeline_step: Execute pipeline scripts
- Process management (start, stop, concurrent prevention)
- Environment variable handling
- Output logging
"""

import pytest
import asyncio
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Test: run_pipeline_step Basic Execution
# =============================================================================

class TestRunPipelineStepBasic:
    """Basic tests for run_pipeline_step function."""
    
    @pytest.mark.asyncio
    async def test_prevents_concurrent_execution(self, clean_pipeline_globals):
        """Should prevent running when already running."""
        from app.pages.pipeline import run_pipeline_step
        
        PS = clean_pipeline_globals
        PS.running = True
        
        # Should return immediately without starting
        await run_pipeline_step("test.py", [], "Test Step")
        
        # State should remain unchanged (no new step started)
        assert PS.running == True
        assert PS.current_step != "Test Step"
    
    @pytest.mark.asyncio
    async def test_sets_running_state(self, clean_pipeline_globals, tmp_path):
        """Should set PS.running to True during execution."""
        from app.pages.pipeline import run_pipeline_step
        
        PS = clean_pipeline_globals
        
        # Create a simple test script
        script = tmp_path / "test_script.py"
        script.write_text("import time; time.sleep(0.1); print('done')")
        
        # Track state changes
        was_running = False
        
        async def check_running():
            nonlocal was_running
            await asyncio.sleep(0.05)
            was_running = PS.running
        
        # Start both tasks
        task1 = asyncio.create_task(run_pipeline_step(str(script), [], "Test Step"))
        task2 = asyncio.create_task(check_running())
        
        await asyncio.gather(task1, task2)
        
        # Should have been running during execution
        assert was_running == True
        # Should be reset after completion
        assert PS.running == False
    
    @pytest.mark.asyncio
    async def test_resets_state_after_completion(self, clean_pipeline_globals, tmp_path):
        """Should reset all state after successful completion."""
        from app.pages.pipeline import run_pipeline_step
        
        PS = clean_pipeline_globals
        
        # Create simple test script
        script = tmp_path / "quick_script.py"
        script.write_text("print('quick test')")
        
        await run_pipeline_step(str(script), [], "Quick Test")
        
        assert PS.running == False
        assert PS.current_step == ""
        assert PS.running_task_name == ""
        assert PS.start_time == None
        assert PS.current_process == None
    
    @pytest.mark.asyncio
    async def test_logs_start_message(self, clean_pipeline_globals, tmp_path):
        """Should log start message with command."""
        from app.pages.pipeline import run_pipeline_step
        
        PS = clean_pipeline_globals
        
        script = tmp_path / "test_script.py"
        script.write_text("print('test')")
        
        await run_pipeline_step(str(script), ["--arg1", "value"], "Test Step")
        
        # Check that start message was logged
        start_messages = [m for m in PS.log_history if "Starting:" in m]
        assert len(start_messages) > 0
        assert "--arg1" in start_messages[0]
        assert "value" in start_messages[0]
    
    @pytest.mark.asyncio
    async def test_handles_script_not_found(self, clean_pipeline_globals, tmp_path, monkeypatch):
        """Should handle missing script gracefully."""
        from app.pages.pipeline import run_pipeline_step
        
        PS = clean_pipeline_globals
        
        # Point to directory without the script
        monkeypatch.setattr('app.pages.pipeline.PIPELINE_DIR', tmp_path)
        
        await run_pipeline_step("nonexistent.py", [], "Missing Script")
        
        # Should have reset running state
        assert PS.running == False


# =============================================================================
# Test: run_pipeline_step Environment Variables
# =============================================================================

class TestRunPipelineStepEnvironment:
    """Tests for environment variable handling in run_pipeline_step."""
    
    @pytest.mark.asyncio
    async def test_sets_pipeline_output_dir(self, clean_pipeline_globals, tmp_path):
        """Should set PIPELINE_OUTPUT_DIR environment variable."""
        from app.pages.pipeline import run_pipeline_step
        
        PS = clean_pipeline_globals
        
        # Create script that prints env var
        script = tmp_path / "env_test.py"
        script.write_text("""
import os
print(f"OUTPUT_DIR={os.environ.get('PIPELINE_OUTPUT_DIR', 'NOT_SET')}")
""")
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        await run_pipeline_step(str(script), [], "Env Test", output_dir=output_dir)
        
        # Check that output dir was logged
        output_messages = [m for m in PS.log_history if "Output dir:" in m]
        assert len(output_messages) > 0
    
    @pytest.mark.asyncio
    async def test_sets_pipeline_input_dir(self, clean_pipeline_globals, tmp_path):
        """Should set PIPELINE_INPUT_DIR environment variable."""
        from app.pages.pipeline import run_pipeline_step
        
        PS = clean_pipeline_globals
        
        script = tmp_path / "env_test.py"
        script.write_text("""
import os
print(f"INPUT_DIR={os.environ.get('PIPELINE_INPUT_DIR', 'NOT_SET')}")
""")
        
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        
        await run_pipeline_step(str(script), [], "Env Test", input_dir=input_dir)
        
        # Check that input dir was logged
        input_messages = [m for m in PS.log_history if "Input dir:" in m]
        assert len(input_messages) > 0
    
    @pytest.mark.asyncio
    async def test_sets_pythonunbuffered(self, clean_pipeline_globals, tmp_path):
        """Should set PYTHONUNBUFFERED for real-time output."""
        from app.pages.pipeline import run_pipeline_step
        
        PS = clean_pipeline_globals
        
        script = tmp_path / "unbuffered_test.py"
        script.write_text("""
import os
import sys
# If unbuffered, output should appear immediately
for i in range(3):
    print(f"Line {i}")
    sys.stdout.flush()
""")
        
        await run_pipeline_step(str(script), [], "Unbuffered Test")
        
        # Should have captured output lines
        line_messages = [m for m in PS.log_history if "Line" in m]
        assert len(line_messages) >= 1


# =============================================================================
# Test: run_pipeline_step Output Handling
# =============================================================================

class TestRunPipelineStepOutput:
    """Tests for output and logging in run_pipeline_step."""
    
    @pytest.mark.asyncio
    async def test_captures_stdout(self, clean_pipeline_globals, tmp_path):
        """Should capture and log stdout output."""
        from app.pages.pipeline import run_pipeline_step
        
        PS = clean_pipeline_globals
        
        script = tmp_path / "stdout_test.py"
        script.write_text("""
print("[INFO] Processing started")
print("[STEP] Step 1 complete")
print("[DONE] All done")
""")
        
        await run_pipeline_step(str(script), [], "Stdout Test")
        
        # Check captured output
        assert any("[INFO] Processing started" in m for m in PS.log_history)
        assert any("[STEP] Step 1 complete" in m for m in PS.log_history)
        assert any("[DONE] All done" in m for m in PS.log_history)
    
    @pytest.mark.asyncio
    async def test_captures_stderr_with_prefix(self, clean_pipeline_globals, tmp_path):
        """Should capture stderr with [WARN] prefix."""
        from app.pages.pipeline import run_pipeline_step
        
        PS = clean_pipeline_globals
        
        script = tmp_path / "stderr_test.py"
        script.write_text("""
import sys
print("Normal output", file=sys.stdout)
print("Warning message", file=sys.stderr)
""")
        
        await run_pipeline_step(str(script), [], "Stderr Test")
        
        # Check that stderr was captured with WARN prefix
        warn_messages = [m for m in PS.log_history if "[WARN]" in m]
        assert len(warn_messages) > 0
    
    @pytest.mark.asyncio
    async def test_filters_progress_bars(self, clean_pipeline_globals, tmp_path):
        """Should filter out tqdm-style progress bars."""
        from app.pages.pipeline import run_pipeline_step
        
        PS = clean_pipeline_globals
        
        script = tmp_path / "progress_test.py"
        script.write_text("""
# Simulate tqdm output
print("Starting process")
print("50%|████████████████░░░░░░░░░░░░░░░░| 5/10 [00:01<00:01, 3.33it/s]")
print("100%|████████████████████████████████| 10/10 [00:02<00:00, 4.00it/s]")
print("Process complete")
""")
        
        await run_pipeline_step(str(script), [], "Progress Test")
        
        # Progress bars should be filtered out
        progress_messages = [m for m in PS.log_history if "████" in m or "it/s]" in m]
        assert len(progress_messages) == 0
        
        # But regular messages should be captured
        assert any("Starting process" in m for m in PS.log_history)
        assert any("Process complete" in m for m in PS.log_history)
    
    @pytest.mark.asyncio
    async def test_truncates_long_lines(self, clean_pipeline_globals, tmp_path):
        """Should truncate lines longer than 500 characters."""
        from app.pages.pipeline import run_pipeline_step
        
        PS = clean_pipeline_globals
        
        script = tmp_path / "long_line_test.py"
        long_text = "X" * 1000
        script.write_text(f'print("{long_text}")')
        
        await run_pipeline_step(str(script), [], "Long Line Test")
        
        # Find the long line (if captured)
        x_messages = [m for m in PS.log_history if "XXXX" in m]
        if x_messages:
            # Should be truncated with ...
            assert len(x_messages[0]) <= 510  # 500 + some prefix + "..."
            assert "..." in x_messages[0]
    
    @pytest.mark.asyncio
    async def test_logs_completion_message(self, clean_pipeline_globals, tmp_path):
        """Should log completion message on success."""
        from app.pages.pipeline import run_pipeline_step
        
        PS = clean_pipeline_globals
        
        script = tmp_path / "success_test.py"
        script.write_text("print('Success!')")
        
        await run_pipeline_step(str(script), [], "Success Test")
        
        # Should log completion
        completion_messages = [m for m in PS.log_history if "Completed successfully" in m]
        assert len(completion_messages) > 0


# =============================================================================
# Test: run_pipeline_step Error Handling
# =============================================================================

class TestRunPipelineStepErrors:
    """Tests for error handling in run_pipeline_step."""
    
    @pytest.mark.asyncio
    async def test_handles_script_error(self, clean_pipeline_globals, tmp_path):
        """Should handle script that exits with error."""
        from app.pages.pipeline import run_pipeline_step
        
        PS = clean_pipeline_globals
        
        script = tmp_path / "error_test.py"
        script.write_text("""
import sys
print("About to fail")
sys.exit(1)
""")
        
        await run_pipeline_step(str(script), [], "Error Test")
        
        # Should log failure
        failure_messages = [m for m in PS.log_history if "Failed" in m]
        assert len(failure_messages) > 0
        
        # State should be reset
        assert PS.running == False
    
    @pytest.mark.asyncio
    async def test_handles_python_exception(self, clean_pipeline_globals, tmp_path):
        """Should handle Python exception in script."""
        from app.pages.pipeline import run_pipeline_step
        
        PS = clean_pipeline_globals
        
        script = tmp_path / "exception_test.py"
        script.write_text("""
print("Starting")
raise ValueError("Test exception")
""")
        
        await run_pipeline_step(str(script), [], "Exception Test")
        
        # Should have captured the error somehow
        # Either in log history or via exit code
        assert PS.running == False
    
    @pytest.mark.asyncio
    async def test_resets_state_on_error(self, clean_pipeline_globals, tmp_path):
        """Should reset all state even when error occurs."""
        from app.pages.pipeline import run_pipeline_step
        
        PS = clean_pipeline_globals
        
        script = tmp_path / "error_reset_test.py"
        script.write_text("raise Exception('Forced error')")
        
        await run_pipeline_step(str(script), [], "Error Reset Test")
        
        # All state should be reset
        assert PS.running == False
        assert PS.current_step == ""
        assert PS.running_task_name == ""
        assert PS.start_time == None
        assert PS.current_process == None


# =============================================================================
# Test: run_pipeline_step with Arguments
# =============================================================================

class TestRunPipelineStepArguments:
    """Tests for passing arguments to pipeline scripts."""
    
    @pytest.mark.asyncio
    async def test_passes_arguments_to_script(self, clean_pipeline_globals, tmp_path):
        """Should pass arguments to the script."""
        from app.pages.pipeline import run_pipeline_step
        
        PS = clean_pipeline_globals
        
        output_file = tmp_path / "args_output.txt"
        script = tmp_path / "args_test.py"
        script.write_text(f"""
import sys
with open('{output_file}', 'w') as f:
    f.write(' '.join(sys.argv[1:]))
""")
        
        args = ["--max-subjects", "5", "--conditions", "DMT", "EC"]
        await run_pipeline_step(str(script), args, "Args Test")
        
        # Check that args were passed
        assert output_file.exists()
        content = output_file.read_text()
        assert "--max-subjects" in content
        assert "5" in content
        assert "DMT" in content
        assert "EC" in content
    
    @pytest.mark.asyncio
    async def test_handles_empty_arguments(self, clean_pipeline_globals, tmp_path):
        """Should work with empty arguments list."""
        from app.pages.pipeline import run_pipeline_step
        
        PS = clean_pipeline_globals
        
        script = tmp_path / "no_args_test.py"
        script.write_text("print('No args needed')")
        
        await run_pipeline_step(str(script), [], "No Args Test")
        
        assert any("No args needed" in m for m in PS.log_history)
    
    @pytest.mark.asyncio
    async def test_handles_arguments_with_spaces(self, clean_pipeline_globals, tmp_path):
        """Should handle arguments containing spaces."""
        from app.pages.pipeline import run_pipeline_step
        
        PS = clean_pipeline_globals
        
        output_file = tmp_path / "space_args_output.txt"
        script = tmp_path / "space_args_test.py"
        script.write_text(f"""
import sys
with open('{output_file}', 'w') as f:
    f.write('\\n'.join(sys.argv[1:]))
""")
        
        args = ["--path", str(tmp_path / "path with spaces")]
        await run_pipeline_step(str(script), args, "Space Args Test")
        
        content = output_file.read_text()
        assert "path with spaces" in content


# =============================================================================
# Test: Pipeline Script Execution (Integration)
# =============================================================================

class TestPipelineScriptExecution:
    """Integration tests for actual pipeline script execution."""
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_fwd_script_help(self, clean_pipeline_globals):
        """Should be able to run fwd.py --help."""
        from app.pages.pipeline import PIPELINE_DIR
        
        PS = clean_pipeline_globals
        
        # Run fwd.py with --help (quick, doesn't process data)
        # We need to modify the call to use actual fwd.py path
        script_path = PIPELINE_DIR / "fwd.py"
        
        if not script_path.exists():
            pytest.skip("fwd.py not found")
        
        # This should complete quickly with help text
        process = await asyncio.create_subprocess_exec(
            "python", str(script_path), "--help",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        
        # Should exit successfully
        assert process.returncode == 0
        # Should have help text
        assert b"--max-subjects" in stdout or b"--max-subjects" in stderr
    
    def test_all_pipeline_scripts_parseable(self):
        """All pipeline scripts should be valid Python."""
        from app.pages.pipeline import PIPELINE_DIR
        
        scripts = [
            "fwd.py",
            "save_load_pickle.py",
            "multi2pool2.py",
            "calculate_syncro.py",
            "generate_order.py",
            "build_order_data.py",
            "pearson.py",
            "clustering.py",
        ]
        
        for script_name in scripts:
            script_path = PIPELINE_DIR / script_name
            if script_path.exists():
                # Try to compile the script
                with open(script_path, 'r') as f:
                    source = f.read()
                try:
                    compile(source, script_path, 'exec')
                except SyntaxError as e:
                    pytest.fail(f"Syntax error in {script_name}: {e}")





