"""
System tab for resource monitoring.

Displays CPU, RAM, and GPU usage with color-coded indicators.
"""
from nicegui import ui

from config import THEME_PRIMARY, THEME_WARN, THEME_TEXT_DIM


class SystemTab:
    """
    System monitoring tab component.
    
    Responsibilities:
    - Display CPU usage (overall and per-core)
    - Display RAM usage
    - Display GPU usage (if available)
    - Auto-update every 2 seconds
    """
    
    ERROR_COLOR = '#ff4444'
    
    def __init__(self):
        """Initialize system tab."""
        self._container = None
    
    def render(self) -> None:
        """Render the system tab content."""
        with ui.tab_panel('system').classes('p-2'):
            ui.label('// SYSTEM_MONITOR').classes('terminal-header mb-2')
            self._container = ui.column().classes('w-full gap-4')
            
            # Update timer
            ui.timer(2.0, self._update_stats)
            
            # Initial render
            self._update_stats()
    
    def _update_stats(self) -> None:
        """Update system statistics display."""
        import psutil
        
        self._container.clear()
        
        with self._container:
            self._render_cpu(psutil)
            self._render_memory(psutil)
            self._render_gpu()
    
    def _render_cpu(self, psutil) -> None:
        """Render CPU usage."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()
        cpu_color = self._get_color(cpu_percent, thresholds=(50, 80))
        
        with ui.row().classes('gap-4 items-center'):
            with ui.column().classes('gap-0'):
                ui.label(f'CPU {cpu_percent:.0f}%').style(
                    f'color:{cpu_color}; font-family: JetBrains Mono; '
                    f'font-size: 1rem; font-weight: bold;'
                )
                ui.label(f'{cpu_count} cores').style(
                    f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;'
                )
            
            # Per-core usage
            per_cpu = psutil.cpu_percent(percpu=True)
            with ui.row().classes('gap-1 flex-wrap'):
                for i, pct in enumerate(per_cpu[:16]):  # Limit to 16 cores
                    color = self._get_color(pct, thresholds=(50, 80))
                    ui.label(f'{pct:.0f}').style(
                        f'color:{color}; font-family: JetBrains Mono; '
                        f'font-size: 0.65rem; min-width: 22px; text-align: center;'
                    )
    
    def _render_memory(self, psutil) -> None:
        """Render memory usage."""
        mem = psutil.virtual_memory()
        mem_used_gb = mem.used / (1024**3)
        mem_total_gb = mem.total / (1024**3)
        mem_color = self._get_color(mem.percent, thresholds=(60, 85))
        
        with ui.row().classes('gap-4 items-center'):
            ui.label(f'RAM {mem.percent:.0f}%').style(
                f'color:{mem_color}; font-family: JetBrains Mono; '
                f'font-size: 1rem; font-weight: bold;'
            )
            ui.label(f'{mem_used_gb:.1f} / {mem_total_gb:.0f} GB').style(
                f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;'
            )
    
    def _render_gpu(self) -> None:
        """Render GPU usage if available."""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_load = gpu.load * 100
                gpu_color = self._get_color(gpu_load, thresholds=(50, 80))
                
                with ui.row().classes('gap-4 items-center'):
                    ui.label(f'GPU {gpu_load:.0f}%').style(
                        f'color:{gpu_color}; font-family: JetBrains Mono; '
                        f'font-size: 1rem; font-weight: bold;'
                    )
                    ui.label(f'{gpu.memoryUsed:.0f} / {gpu.memoryTotal:.0f} MB').style(
                        f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;'
                    )
            else:
                self._render_gpu_unavailable()
        except ImportError:
            self._render_gpu_unavailable()
        except Exception:
            self._render_gpu_unavailable()
    
    def _render_gpu_unavailable(self) -> None:
        """Render GPU not available message."""
        ui.label('GPU: N/A').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
    
    def _get_color(self, value: float, thresholds: tuple = (50, 80)) -> str:
        """
        Get color based on value and thresholds.
        
        Args:
            value: Current value (0-100)
            thresholds: (low_threshold, high_threshold)
            
        Returns:
            Color hex string
        """
        low, high = thresholds
        if value < low:
            return THEME_PRIMARY
        elif value < high:
            return THEME_WARN
        else:
            return self.ERROR_COLOR









