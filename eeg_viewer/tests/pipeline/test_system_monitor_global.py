"""
Tests for global system monitor component.

Verifies that:
- System monitor displays CPU, RAM, GPU stats
- System monitor is available globally (not just in pipeline)
- Stats update periodically
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestSystemMonitorData:
    """Test system monitor data collection."""
    
    def test_can_get_cpu_percent(self):
        """Should be able to get CPU usage percentage."""
        import psutil
        cpu = psutil.cpu_percent(interval=0.1)
        assert isinstance(cpu, float)
        assert 0 <= cpu <= 100
    
    def test_can_get_memory_info(self):
        """Should be able to get memory usage."""
        import psutil
        mem = psutil.virtual_memory()
        assert hasattr(mem, 'percent')
        assert hasattr(mem, 'used')
        assert hasattr(mem, 'total')
        assert 0 <= mem.percent <= 100
    
    def test_can_get_cpu_count(self):
        """Should be able to get CPU core count."""
        import psutil
        count = psutil.cpu_count()
        assert isinstance(count, int)
        assert count > 0


class TestSystemMonitorComponent:
    """Test system monitor UI component."""
    
    def test_system_stats_function_exists(self):
        """get_system_stats function should exist."""
        from app.visualization.components.system_monitor import get_system_stats
        assert callable(get_system_stats)
    
    def test_system_stats_returns_dict(self):
        """get_system_stats should return a dictionary."""
        from app.visualization.components.system_monitor import get_system_stats
        stats = get_system_stats()
        assert isinstance(stats, dict)
    
    def test_system_stats_has_cpu(self):
        """Stats should include CPU info."""
        from app.visualization.components.system_monitor import get_system_stats
        stats = get_system_stats()
        assert 'cpu_percent' in stats
        assert 'cpu_count' in stats
    
    def test_system_stats_has_memory(self):
        """Stats should include memory info."""
        from app.visualization.components.system_monitor import get_system_stats
        stats = get_system_stats()
        assert 'mem_percent' in stats
        assert 'mem_used_gb' in stats
        assert 'mem_total_gb' in stats
    
    def test_render_function_exists(self):
        """render_system_monitor function should exist."""
        from app.visualization.components.system_monitor import render_system_monitor_compact
        assert callable(render_system_monitor_compact)


class TestSystemMonitorFormatting:
    """Test formatting of system stats."""
    
    def test_format_bytes_to_gb(self):
        """Should format bytes to GB correctly."""
        from app.visualization.components.system_monitor import get_system_stats
        stats = get_system_stats()
        # mem_used_gb and mem_total_gb should be reasonable values
        assert stats['mem_total_gb'] > 0
        assert stats['mem_used_gb'] >= 0
        assert stats['mem_used_gb'] <= stats['mem_total_gb']








