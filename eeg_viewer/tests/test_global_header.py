"""
Tests for global header component.

Verifies that the system monitor (CPU/RAM/GPU) is available in all pages.
"""
import pytest
import inspect


class TestGlobalHeader:
    """Test global header is used in all pages."""
    
    def test_global_header_exists(self):
        """Global header component should exist."""
        from app.visualization.components.global_header import render_global_header
        
        assert callable(render_global_header)
    
    def test_global_header_has_system_monitor(self):
        """Global header should include system monitor."""
        from app.visualization.components.global_header import render_global_header
        
        source = inspect.getsource(render_global_header)
        
        assert 'render_system_monitor_compact' in source
    
    def test_global_header_has_running_indicator(self):
        """Global header should include running indicator."""
        from app.visualization.components.global_header import render_global_header
        
        source = inspect.getsource(render_global_header)
        
        assert 'render_running_indicator' in source
    
    def test_global_header_has_navigation(self):
        """Global header should include navigation buttons."""
        from app.visualization.components.global_header import render_global_header
        
        source = inspect.getsource(render_global_header)
        
        assert 'VIEWER' in source
        assert 'PIPELINE' in source
        assert 'MODEL' in source
        assert 'ANALYSIS' in source


class TestPagesUseGlobalHeader:
    """Test that all pages use the global header."""
    
    def test_viewer_uses_global_header(self):
        """Viewer page should use global header."""
        from app.pages import viewer
        
        source = inspect.getsource(viewer)
        
        assert 'render_global_header' in source or 'global_header' in source
    
    def test_pipeline_uses_global_header(self):
        """Pipeline page should use global header."""
        from app.pages.pipeline.components import header
        
        source = inspect.getsource(header)
        
        assert 'render_global_header' in source or 'global_header' in source
    
    def test_model_uses_global_header(self):
        """Model page should use global header."""
        from app.pages import model
        
        source = inspect.getsource(model)
        
        assert 'render_global_header' in source or 'global_header' in source
    
    def test_analysis_uses_global_header(self):
        """Analysis page should use global header."""
        from app.pages import analysis
        
        source = inspect.getsource(analysis)
        
        assert 'render_global_header' in source or 'global_header' in source


class TestSystemMonitorGlobalVisibility:
    """Test that system monitor is globally visible."""
    
    def test_system_monitor_in_global_header(self):
        """System monitor should be rendered in global header."""
        from app.visualization.components import global_header
        
        source = inspect.getsource(global_header)
        
        # System monitor should be imported and used
        assert 'render_system_monitor_compact' in source
        assert 'system_monitor' in source.lower()
    
    def test_all_page_configs_exist(self):
        """All page configurations should exist in global header."""
        from app.visualization.components.global_header import PAGE_CONFIG
        
        expected_pages = ['viewer', 'cleaner', 'pipeline', 'model', 'analysis']
        
        for page in expected_pages:
            assert page in PAGE_CONFIG, f"Page '{page}' missing from PAGE_CONFIG"

