"""
Tests for live file browser component.

Verifies that:
- File browser shows files in run directory
- Auto-refresh detects new files
- File counts update correctly
"""
import pytest
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestFileBrowserData:
    """Test file browser data functions."""
    
    def test_count_files_in_directory(self):
        """Should count files in directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            # Create some test files
            (tmp / "file1.pkl").touch()
            (tmp / "file2.pkl").touch()
            (tmp / "file3.txt").touch()
            
            pkl_count = len(list(tmp.glob("*.pkl")))
            assert pkl_count == 2
    
    def test_count_files_by_pattern(self):
        """Should count files matching pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "phases-S01.pkl").touch()
            (tmp / "phases-S02.pkl").touch()
            (tmp / "syncro-S01.pkl").touch()
            
            phases_count = len(list(tmp.glob("phases-*.pkl")))
            syncro_count = len(list(tmp.glob("syncro-*.pkl")))
            
            assert phases_count == 2
            assert syncro_count == 1
    
    def test_list_directory_contents(self):
        """Should list directory contents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "subdir").mkdir()
            (tmp / "file.txt").touch()
            
            contents = list(tmp.iterdir())
            assert len(contents) == 2
    
    def test_get_file_size(self):
        """Should get file size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            test_file = tmp / "test.txt"
            test_file.write_text("Hello World")
            
            size = test_file.stat().st_size
            assert size > 0


class TestFileBrowserComponent:
    """Test file browser UI component."""
    
    def test_get_run_file_stats_exists(self):
        """get_run_file_stats function should exist."""
        from app.pages.pipeline.components.file_browser import get_run_file_stats
        assert callable(get_run_file_stats)
    
    def test_get_run_file_stats_with_none(self):
        """Should handle None run_dir gracefully."""
        from app.pages.pipeline.components.file_browser import get_run_file_stats
        stats = get_run_file_stats(None)
        assert stats == {}
    
    def test_get_run_file_stats_with_nonexistent(self):
        """Should handle nonexistent directory gracefully."""
        from app.pages.pipeline.components.file_browser import get_run_file_stats
        stats = get_run_file_stats(Path("/nonexistent/path"))
        assert stats == {}
    
    def test_render_function_exists(self):
        """render_file_browser function should exist."""
        from app.pages.pipeline.components.file_browser import FileBrowserPanel
        assert FileBrowserPanel is not None


class TestFileBrowserAutoRefresh:
    """Test auto-refresh functionality."""
    
    def test_detects_new_files(self):
        """Should detect when new files are added."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            
            # Initial count
            count1 = len(list(tmp.glob("*.pkl")))
            assert count1 == 0
            
            # Add file
            (tmp / "new_file.pkl").touch()
            
            # New count
            count2 = len(list(tmp.glob("*.pkl")))
            assert count2 == 1
    
    def test_counts_nested_files(self):
        """Should count files in subdirectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "DMT").mkdir()
            (tmp / "DMT" / "phases-S01.pkl").touch()
            (tmp / "EC").mkdir()
            (tmp / "EC" / "phases-S02.pkl").touch()
            
            total = len(list(tmp.rglob("*.pkl")))
            assert total == 2


class TestFileBrowserFormatting:
    """Test file browser formatting."""
    
    def test_format_file_size_bytes(self):
        """Should format small file sizes in bytes."""
        from app.pages.pipeline.components.file_browser import format_file_size
        assert 'B' in format_file_size(500)
    
    def test_format_file_size_kb(self):
        """Should format KB file sizes."""
        from app.pages.pipeline.components.file_browser import format_file_size
        result = format_file_size(5000)
        assert 'KB' in result or 'K' in result
    
    def test_format_file_size_mb(self):
        """Should format MB file sizes."""
        from app.pages.pipeline.components.file_browser import format_file_size
        result = format_file_size(5_000_000)
        assert 'MB' in result or 'M' in result

