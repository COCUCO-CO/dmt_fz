"""
Global system monitor component.

Displays CPU, RAM, and GPU usage in a compact format
suitable for placement in the header of any page.
"""
import psutil
from nicegui import ui

from config import THEME_PRIMARY, THEME_WARN, THEME_TEXT_DIM


def get_system_stats() -> dict:
    """
    Get current system statistics.
    
    Returns:
        Dictionary with CPU, memory, and GPU stats
    """
    # CPU
    cpu_percent = psutil.cpu_percent(interval=0.1)
    cpu_count = psutil.cpu_count()
    
    # Memory
    mem = psutil.virtual_memory()
    mem_percent = mem.percent
    mem_used_gb = mem.used / (1024**3)
    mem_total_gb = mem.total / (1024**3)
    
    stats = {
        'cpu_percent': cpu_percent,
        'cpu_count': cpu_count,
        'mem_percent': mem_percent,
        'mem_used_gb': mem_used_gb,
        'mem_total_gb': mem_total_gb,
        'gpu_available': False,
        'gpu_percent': 0,
        'gpu_mem_used': 0,
        'gpu_mem_total': 0,
    }
    
    # GPU (optional)
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            stats['gpu_available'] = True
            stats['gpu_percent'] = gpu.load * 100
            stats['gpu_mem_used'] = gpu.memoryUsed
            stats['gpu_mem_total'] = gpu.memoryTotal
    except:
        pass
    
    return stats


def _get_color_for_percent(percent: float) -> str:
    """Get color based on usage percentage."""
    if percent < 50:
        return THEME_PRIMARY
    elif percent < 80:
        return THEME_WARN
    else:
        return '#ff4444'


def render_system_monitor_compact():
    """
    Render a compact system monitor suitable for the header.
    
    Shows CPU, RAM, and GPU (if available) in a single row.
    Updates every 2 seconds.
    """
    with ui.row().classes('items-center gap-3'):
        # CPU indicator
        cpu_label = ui.label('CPU: --%').style(
            f'font-family: JetBrains Mono; font-size: 0.7rem; color: {THEME_TEXT_DIM};'
        )
        
        # RAM indicator  
        ram_label = ui.label('RAM: --%').style(
            f'font-family: JetBrains Mono; font-size: 0.7rem; color: {THEME_TEXT_DIM};'
        )
        
        # GPU indicator (will be hidden if no GPU)
        gpu_container = ui.element('span')
        gpu_label = None
        
        def update_stats():
            nonlocal gpu_label
            try:
                stats = get_system_stats()
                
                # Update CPU
                cpu_color = _get_color_for_percent(stats['cpu_percent'])
                cpu_label.text = f"CPU: {stats['cpu_percent']:.0f}%"
                cpu_label.style(f'font-family: JetBrains Mono; font-size: 0.7rem; color: {cpu_color};')
                
                # Update RAM
                ram_color = _get_color_for_percent(stats['mem_percent'])
                ram_label.text = f"RAM: {stats['mem_percent']:.0f}%"
                ram_label.style(f'font-family: JetBrains Mono; font-size: 0.7rem; color: {ram_color};')
                
                # Update GPU if available
                if stats['gpu_available']:
                    if gpu_label is None:
                        with gpu_container:
                            gpu_label = ui.label('GPU: --%').style(
                                f'font-family: JetBrains Mono; font-size: 0.7rem; color: {THEME_TEXT_DIM};'
                            )
                    gpu_color = _get_color_for_percent(stats['gpu_percent'])
                    gpu_label.text = f"GPU: {stats['gpu_percent']:.0f}%"
                    gpu_label.style(f'font-family: JetBrains Mono; font-size: 0.7rem; color: {gpu_color};')
            except:
                pass
        
        # Initial update
        update_stats()
        
        # Timer for periodic updates
        ui.timer(2.0, update_stats)


def render_system_monitor_detailed():
    """
    Render a more detailed system monitor with progress bars.
    
    Used in dedicated system monitoring views.
    """
    with ui.column().classes('gap-2 w-full'):
        # Stats container that will be updated
        stats_container = ui.column().classes('w-full gap-3')
        
        def update_detailed():
            stats_container.clear()
            with stats_container:
                stats = get_system_stats()
                
                # CPU
                cpu_color = _get_color_for_percent(stats['cpu_percent'])
                with ui.row().classes('items-center gap-2 w-full'):
                    ui.label('CPU').style(
                        f'color: {THEME_TEXT_DIM}; font-family: JetBrains Mono; '
                        f'font-size: 0.75rem; min-width: 40px;'
                    )
                    with ui.element('div').classes('flex-1').style(
                        'background: #1a1a1a; height: 8px; border-radius: 4px; overflow: hidden;'
                    ):
                        ui.element('div').style(
                            f'background: {cpu_color}; width: {stats["cpu_percent"]}%; '
                            f'height: 100%; transition: width 0.3s;'
                        )
                    ui.label(f'{stats["cpu_percent"]:.0f}%').style(
                        f'color: {cpu_color}; font-family: JetBrains Mono; '
                        f'font-size: 0.75rem; min-width: 45px; text-align: right;'
                    )
                
                # RAM
                ram_color = _get_color_for_percent(stats['mem_percent'])
                with ui.row().classes('items-center gap-2 w-full'):
                    ui.label('RAM').style(
                        f'color: {THEME_TEXT_DIM}; font-family: JetBrains Mono; '
                        f'font-size: 0.75rem; min-width: 40px;'
                    )
                    with ui.element('div').classes('flex-1').style(
                        'background: #1a1a1a; height: 8px; border-radius: 4px; overflow: hidden;'
                    ):
                        ui.element('div').style(
                            f'background: {ram_color}; width: {stats["mem_percent"]}%; '
                            f'height: 100%; transition: width 0.3s;'
                        )
                    ui.label(f'{stats["mem_used_gb"]:.1f}/{stats["mem_total_gb"]:.0f}G').style(
                        f'color: {ram_color}; font-family: JetBrains Mono; '
                        f'font-size: 0.75rem; min-width: 70px; text-align: right;'
                    )
                
                # GPU (if available)
                if stats['gpu_available']:
                    gpu_color = _get_color_for_percent(stats['gpu_percent'])
                    with ui.row().classes('items-center gap-2 w-full'):
                        ui.label('GPU').style(
                            f'color: {THEME_TEXT_DIM}; font-family: JetBrains Mono; '
                            f'font-size: 0.75rem; min-width: 40px;'
                        )
                        with ui.element('div').classes('flex-1').style(
                            'background: #1a1a1a; height: 8px; border-radius: 4px; overflow: hidden;'
                        ):
                            ui.element('div').style(
                                f'background: {gpu_color}; width: {stats["gpu_percent"]}%; '
                                f'height: 100%; transition: width 0.3s;'
                            )
                        ui.label(f'{stats["gpu_percent"]:.0f}%').style(
                            f'color: {gpu_color}; font-family: JetBrains Mono; '
                            f'font-size: 0.75rem; min-width: 45px; text-align: right;'
                        )
        
        update_detailed()
        ui.timer(2.0, update_detailed)

