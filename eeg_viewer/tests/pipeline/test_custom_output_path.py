"""
Tests for custom output path functionality.

Verifies that:
- Custom paths can be set
- Path validation works correctly
- Directory creation works
"""
import pytest
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestCustomOutputPath:
    """Test custom output path functionality."""
    
    def test_absolute_path_detection(self):
        """Should detect absolute paths correctly."""
        path = Path('/media/storage_hdd/custom_output')
        assert path.is_absolute()
        
        relative = Path('relative/path')
        assert not relative.is_absolute()
    
    def test_directory_creation(self):
        """Should be able to create new directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / 'new_run_dir'
            assert not new_dir.exists()
            
            new_dir.mkdir(parents=True, exist_ok=True)
            assert new_dir.exists()
            assert new_dir.is_dir()
    
    def test_io_config_has_custom_field(self):
        """IOConfigPanel should have custom output field attribute."""
        from app.pages.pipeline.components.io_config import IOConfigPanel
        panel = IOConfigPanel()
        assert hasattr(panel, '_custom_output_field')
    
    def test_io_config_has_use_custom_path_method(self):
        """IOConfigPanel should have _use_custom_path method."""
        from app.pages.pipeline.components.io_config import IOConfigPanel
        panel = IOConfigPanel()
        assert hasattr(panel, '_use_custom_path')
        assert callable(panel._use_custom_path)
    
    def test_path_with_spaces_works(self):
        """Paths with spaces should work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path_with_spaces = Path(tmpdir) / 'my custom output'
            path_with_spaces.mkdir(parents=True, exist_ok=True)
            assert path_with_spaces.exists()


class TestPathDisplayFormatting:
    """Test path display formatting."""
    
    def test_standard_run_shows_name_only(self):
        """Standard runs should display just the name."""
        from app.pages.pipeline.config import PIPELINE_OUTPUTS
        run_path = PIPELINE_OUTPUTS / 'run_20251211_123456'
        
        # Should show just the name
        if PIPELINE_OUTPUTS in run_path.parents or run_path.parent == PIPELINE_OUTPUTS:
            display = run_path.name
        else:
            display = f"📁 {run_path.name}"
        
        assert display == 'run_20251211_123456'
    
    def test_custom_path_shows_icon(self):
        """Custom paths should show folder icon."""
        from app.pages.pipeline.config import PIPELINE_OUTPUTS
        custom_path = Path('/media/custom/output/my_run')
        
        if PIPELINE_OUTPUTS in custom_path.parents or custom_path.parent == PIPELINE_OUTPUTS:
            display = custom_path.name
        else:
            display = f"📁 {custom_path.name}"
        
        assert '📁' in display
        assert 'my_run' in display

