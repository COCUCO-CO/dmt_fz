"""
Step 2 Visualizer: Consolidate Data (save_load_pickle.py)

Shows consolidated subject data overview - compact layout.
"""
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from nicegui import ui

from config import THEME_PRIMARY, THEME_SECONDARY, THEME_TEXT_DIM
from .base import BaseVisualizer


class Step2Visualizer(BaseVisualizer):
    """Visualizer for Step 2: Consolidate Data."""
    
    step_number = 2
    step_name = "Consolidar Datos"
    step_description = "Agrupa archivos por condición"
    file_patterns = ["subject_phases_*.pkl", "*/subject_phases_*.pkl", "**/subject_phases_*.pkl"]
    supported_backends = ['matplotlib']
    
    def get_controls(self) -> Dict[str, Any]:
        return {}
    
    def _render_visualization(self) -> None:
        files = self.find_data_files()
        
        if not files:
            self._render_no_data_message("No hay datos consolidados")
            return
        
        # Compact horizontal layout
        with ui.row().classes('w-full gap-4 items-start'):
            # Left: File list (compact)
            with ui.column().style('min-width: 180px;'):
                ui.label('📦 Consolidados').style(
                    f'color: {THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.8rem;'
                )
                
                conditions = []
                counts = []
                
                for f in files:
                    cond = f.stem.replace('subject_phases_', '')
                    size_mb = f.stat().st_size / (1024 * 1024)
                    
                    n_subjects = '?'
                    try:
                        self.load_data(f)
                        if self._data and isinstance(self._data, dict):
                            n_subjects = len(self._data)
                            conditions.append(cond)
                            counts.append(n_subjects)
                    except:
                        pass
                    
                    with ui.row().classes('items-center gap-2 p-1').style(
                        'background: #1a1a1a; border-radius: 4px; margin: 2px 0;'
                    ):
                        ui.label(cond).style(
                            f'color: {THEME_SECONDARY}; font-weight: bold; min-width: 40px; font-size: 0.75rem;'
                        )
                        ui.label(f'{n_subjects}').style(
                            f'color: {THEME_PRIMARY}; font-size: 0.75rem;'
                        )
                        ui.label(f'{size_mb:.0f}MB').style(
                            f'color: {THEME_TEXT_DIM}; font-size: 0.65rem; margin-left: auto;'
                        )
            
            # Right: Compact bar chart
            if conditions:
                with ui.column().classes('flex-1'):
                    self._render_condition_chart(conditions, counts)
    
    def _render_condition_chart(self, conditions: List[str], counts: List[int]) -> None:
        """Render compact bar chart."""
        fig, ax = self.create_matplotlib_figure(figsize=(4, 2.5))
        colors = ['#00d4aa', '#f59e0b', '#a78bfa'][:len(conditions)]
        bars = ax.bar(conditions, counts, color=colors)
        ax.set_ylabel('Sujetos', color='#888888', fontsize=9)
        ax.set_title('Por Condición', color=THEME_PRIMARY, fontsize=9)
        ax.tick_params(labelsize=8)
        
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                   str(count), ha='center', va='bottom', color='#888888', fontsize=8)
        
        self.show_matplotlib(fig)
