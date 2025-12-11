"""
Tests for pipeline utility functions.

Tests:
- get_run_dirs: List existing pipeline runs
- create_new_run: Create new run directory
- scan_files: Scan for EEG files
- pipeline_log: Log message handling
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime
import time

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Test: get_run_dirs
# =============================================================================

class TestGetRunDirs:
    """Tests for get_run_dirs function."""
    
    def test_returns_empty_when_no_directory(self, tmp_path, monkeypatch):
        """Should return empty list when PIPELINE_OUTPUTS doesn't exist."""
        from app.pages.pipeline import get_run_dirs, PIPELINE_OUTPUTS
        
        # Point to non-existent directory
        non_existent = tmp_path / "does_not_exist"
        monkeypatch.setattr('app.pages.pipeline.PIPELINE_OUTPUTS', non_existent)
        
        result = get_run_dirs()
        assert result == []
    
    def test_returns_empty_when_no_runs(self, tmp_path, monkeypatch):
        """Should return empty list when directory has no run_* folders."""
        from app.pages.pipeline import get_run_dirs
        
        # Create empty output directory
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        monkeypatch.setattr('app.pages.pipeline.PIPELINE_OUTPUTS', output_dir)
        
        result = get_run_dirs()
        assert result == []
    
    def test_returns_only_run_directories(self, tmp_path, monkeypatch):
        """Should only return directories starting with 'run_'."""
        from app.pages.pipeline import get_run_dirs
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # Create various directories
        (output_dir / "run_20241201_100000").mkdir()
        (output_dir / "run_20241202_120000").mkdir()
        (output_dir / "not_a_run").mkdir()
        (output_dir / "other_folder").mkdir()
        (output_dir / "some_file.txt").touch()
        
        monkeypatch.setattr('app.pages.pipeline.PIPELINE_OUTPUTS', output_dir)
        
        result = get_run_dirs()
        assert len(result) == 2
        assert all(r.startswith("run_") for r in result)
    
    def test_returns_sorted_descending(self, tmp_path, monkeypatch):
        """Should return runs sorted newest first."""
        from app.pages.pipeline import get_run_dirs
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # Create runs in random order
        (output_dir / "run_20241202_120000").mkdir()
        (output_dir / "run_20241201_100000").mkdir()
        (output_dir / "run_20241203_150000").mkdir()
        
        monkeypatch.setattr('app.pages.pipeline.PIPELINE_OUTPUTS', output_dir)
        
        result = get_run_dirs()
        assert result == [
            "run_20241203_150000",
            "run_20241202_120000",
            "run_20241201_100000",
        ]
    
    def test_ignores_files(self, tmp_path, monkeypatch):
        """Should ignore files, only return directories."""
        from app.pages.pipeline import get_run_dirs
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # Create a run directory and a file with run_ prefix
        (output_dir / "run_20241201_100000").mkdir()
        (output_dir / "run_20241202_log.txt").touch()
        
        monkeypatch.setattr('app.pages.pipeline.PIPELINE_OUTPUTS', output_dir)
        
        result = get_run_dirs()
        assert result == ["run_20241201_100000"]


# =============================================================================
# Test: create_new_run
# =============================================================================

class TestCreateNewRun:
    """Tests for create_new_run function."""
    
    def test_creates_directory_with_timestamp(self, tmp_path, monkeypatch):
        """Should create directory with run_YYYYMMDD_HHMMSS format."""
        from app.pages.pipeline import create_new_run
        
        monkeypatch.setattr('app.pages.pipeline.PIPELINE_OUTPUTS', tmp_path)
        
        before = datetime.now()
        result = create_new_run()
        after = datetime.now()
        
        # Check directory was created
        assert result.exists()
        assert result.is_dir()
        
        # Check name format
        name = result.name
        assert name.startswith("run_")
        
        # Parse timestamp
        ts_str = name[4:]  # Remove "run_"
        ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
        
        # Should be between before and after
        assert before.replace(microsecond=0) <= ts <= after.replace(microsecond=0) + datetime.resolution
    
    def test_creates_condition_subdirectories(self, tmp_path, monkeypatch):
        """Should create DMT/, EC/, EO/ subdirectories."""
        from app.pages.pipeline import create_new_run
        
        monkeypatch.setattr('app.pages.pipeline.PIPELINE_OUTPUTS', tmp_path)
        
        result = create_new_run()
        
        for cond in ["DMT", "EC", "EO"]:
            cond_dir = result / cond
            assert cond_dir.exists()
            assert cond_dir.is_dir()
    
    def test_creates_parent_directories(self, tmp_path, monkeypatch):
        """Should create parent directories if they don't exist."""
        from app.pages.pipeline import create_new_run
        
        # Point to nested non-existent path
        nested_output = tmp_path / "deeply" / "nested" / "output"
        monkeypatch.setattr('app.pages.pipeline.PIPELINE_OUTPUTS', nested_output)
        
        result = create_new_run()
        
        assert result.exists()
        assert nested_output.exists()
    
    def test_multiple_runs_have_unique_names(self, tmp_path, monkeypatch):
        """Creating multiple runs should generate unique names."""
        from app.pages.pipeline import create_new_run
        
        monkeypatch.setattr('app.pages.pipeline.PIPELINE_OUTPUTS', tmp_path)
        
        runs = []
        for _ in range(3):
            runs.append(create_new_run())
            time.sleep(1.1)  # Ensure different timestamp
        
        # All names should be unique
        names = [r.name for r in runs]
        assert len(names) == len(set(names))


# =============================================================================
# Test: pipeline_log
# =============================================================================

class TestPipelineLog:
    """Tests for pipeline_log function."""
    
    def test_appends_to_log_history(self, clean_pipeline_globals):
        """Should append message to PS.log_history."""
        from app.pages.pipeline import pipeline_log
        
        PS = clean_pipeline_globals
        
        pipeline_log("Test message 1")
        pipeline_log("Test message 2")
        
        assert len(PS.log_history) == 2
        assert PS.log_history[0] == "Test message 1"
        assert PS.log_history[1] == "Test message 2"
    
    def test_handles_missing_log_container(self, clean_pipeline_globals):
        """Should not crash when PS.log_container is None."""
        from app.pages.pipeline import pipeline_log
        
        PS = clean_pipeline_globals
        PS.log_container = None
        
        # Should not raise
        pipeline_log("Test message")
        
        assert "Test message" in PS.log_history
    
    def test_preserves_message_order(self, clean_pipeline_globals):
        """Should preserve order of logged messages."""
        from app.pages.pipeline import pipeline_log
        
        PS = clean_pipeline_globals
        
        messages = [
            "[RUN] Starting",
            "[SETUP] Init",
            "[1/4] Step 1",
            "[INFO] Done",
        ]
        
        for msg in messages:
            pipeline_log(msg)
        
        assert PS.log_history == messages
    
    def test_handles_special_characters(self, clean_pipeline_globals):
        """Should handle messages with special characters."""
        from app.pages.pipeline import pipeline_log
        
        PS = clean_pipeline_globals
        
        special_messages = [
            "Path: /media/storage/data → output",
            "Progress: 50% ████░░░░",
            "Unicode: αβγδ θ",
            "Special: <>&\"'",
        ]
        
        for msg in special_messages:
            pipeline_log(msg)
        
        assert PS.log_history == special_messages
    
    def test_handles_empty_message(self, clean_pipeline_globals):
        """Should handle empty messages."""
        from app.pages.pipeline import pipeline_log
        
        PS = clean_pipeline_globals
        
        pipeline_log("")
        pipeline_log("Normal message")
        pipeline_log("")
        
        assert len(PS.log_history) == 3
        assert PS.log_history[0] == ""
        assert PS.log_history[1] == "Normal message"


# =============================================================================
# Test: scan_files (from pipeline/fwd.py)
# =============================================================================

class TestScanFiles:
    """Tests for scan_files function that discovers EEG files."""
    
    def test_scans_subdirectory_structure(self, temp_pipeline_dir):
        """Should find files in DMT/, EC/, EO/ subdirectories."""
        # Import from pipeline module (not app.pages.pipeline)
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "pipeline"))
        from fwd import scan_files
        
        input_dir = temp_pipeline_dir["input"]
        
        # Create dummy .set files
        (input_dir / "DMT" / "S01-DMT.set").touch()
        (input_dir / "DMT" / "S02-DMT.set").touch()
        (input_dir / "EC" / "S01-EC.set").touch()
        (input_dir / "EO" / "S01-EO.set").touch()
        
        result = scan_files(input_dir)
        
        assert len(result["DMT"]) == 2
        assert len(result["EC"]) == 1
        assert len(result["EO"]) == 1
    
    def test_scans_flat_structure(self, temp_pipeline_dir):
        """Should find files with condition in filename when no subdirs."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "pipeline"))
        from fwd import scan_files
        
        # Create flat directory (no DMT/EC/EO subdirs)
        flat_dir = temp_pipeline_dir["root"] / "flat_input"
        flat_dir.mkdir()
        
        # Create files with condition in name
        (flat_dir / "S01_DMT_cleaned.set").touch()
        (flat_dir / "S02_DMT_cleaned.set").touch()
        (flat_dir / "S01_EC_cleaned.set").touch()
        (flat_dir / "S01_EO_cleaned.set").touch()
        
        result = scan_files(flat_dir)
        
        assert len(result["DMT"]) == 2
        assert len(result["EC"]) == 1
        assert len(result["EO"]) == 1
    
    def test_returns_empty_when_no_files(self, temp_pipeline_dir):
        """Should return empty lists when no .set files found."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "pipeline"))
        from fwd import scan_files
        
        result = scan_files(temp_pipeline_dir["input"])
        
        assert result["DMT"] == []
        assert result["EC"] == []
        assert result["EO"] == []
    
    def test_returns_sorted_files(self, temp_pipeline_dir):
        """Should return files sorted alphabetically."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "pipeline"))
        from fwd import scan_files
        
        input_dir = temp_pipeline_dir["input"]
        
        # Create files in random order
        (input_dir / "DMT" / "S10-DMT.set").touch()
        (input_dir / "DMT" / "S01-DMT.set").touch()
        (input_dir / "DMT" / "S05-DMT.set").touch()
        
        result = scan_files(input_dir)
        
        names = [f.name for f in result["DMT"]]
        assert names == ["S01-DMT.set", "S05-DMT.set", "S10-DMT.set"]
    
    def test_ignores_non_set_files(self, temp_pipeline_dir):
        """Should only include .set files."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "pipeline"))
        from fwd import scan_files
        
        input_dir = temp_pipeline_dir["input"]
        
        # Create various file types
        (input_dir / "DMT" / "S01-DMT.set").touch()
        (input_dir / "DMT" / "S01-DMT.fdt").touch()
        (input_dir / "DMT" / "S01-DMT.txt").touch()
        (input_dir / "DMT" / "readme.md").touch()
        
        result = scan_files(input_dir)
        
        assert len(result["DMT"]) == 1
        assert result["DMT"][0].suffix == ".set"


# =============================================================================
# Test: Path Constants
# =============================================================================

class TestPipelineConstants:
    """Tests for pipeline path constants."""
    
    def test_pipeline_dir_exists(self):
        """PIPELINE_DIR should point to existing directory."""
        from app.pages.pipeline import PIPELINE_DIR
        
        assert PIPELINE_DIR.exists()
        assert PIPELINE_DIR.is_dir()
        assert PIPELINE_DIR.name == "pipeline"
    
    def test_pipeline_dir_contains_scripts(self):
        """PIPELINE_DIR should contain pipeline scripts."""
        from app.pages.pipeline import PIPELINE_DIR
        
        expected_scripts = [
            "fwd.py",
            "save_load_pickle.py",
            "multi2pool2.py",
            "calculate_syncro.py",
            "generate_order.py",
            "build_order_data.py",
            "pearson.py",
            "clustering.py",
        ]
        
        for script in expected_scripts:
            script_path = PIPELINE_DIR / script
            assert script_path.exists(), f"Missing pipeline script: {script}"
    
    def test_default_input_dir_path(self):
        """DEFAULT_INPUT_DIR should be defined correctly."""
        from app.pages.pipeline import DEFAULT_INPUT_DIR
        
        # Should point to EEG_CLEAN
        assert DEFAULT_INPUT_DIR.name == "EEG_CLEAN"
    
    def test_pipeline_outputs_path(self):
        """PIPELINE_OUTPUTS should point to correct location."""
        from app.pages.pipeline import PIPELINE_OUTPUTS
        
        # Should be under eeg_viewer
        assert "eeg_viewer" in str(PIPELINE_OUTPUTS)
        assert PIPELINE_OUTPUTS.name == "pipeline_outputs"

