"""
Step 6 Visualizer: Aggregate Metrics (build_order_data.py)

Shows aggregated metrics - handles large files carefully.
"""
from typing import Dict, Any, List
import numpy as np
from nicegui import ui

from config import THEME_PRIMARY, THEME_SECONDARY, THEME_TEXT_DIM, THEME_WARN
from .base import BaseVisualizer


class Step6Visualizer(BaseVisualizer):
    """Visualizer for Step 6: Aggregate Metrics."""
    
    step_number = 6
    step_name = "Agregar Métricas"
    step_description = "Resumen de todos los sujetos"
    # order_all-*.pkl in subdirs OR order_all*.pkl in root
    file_patterns = ["order_all*.pkl", "*/order_all*.pkl", "**/order_all-*.pkl"]
    supported_backends = ['matplotlib']
    
    def get_controls(self) -> Dict[str, Any]:
        return {}
    
    def _render_visualization(self) -> None:
        files = self.find_data_files()
        
        if not files:
            self._render_no_data_message("No hay datos agregados (order_all*.pkl)")
            return
        
        # Show file summary without loading (files can be huge)
        with ui.row().classes('w-full gap-4 items-start'):
            # Left: File list
            with ui.column().style('min-width: 200px;'):
                ui.label('📊 Archivos Agregados').style(
                    f'color: {THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.8rem;'
                )
                
                # Group by condition
                by_condition = {'DMT': [], 'EC': [], 'EO': [], 'Other': []}
                
                for f in files:
                    name = f.name.upper()
                    size_mb = f.stat().st_size / (1024 * 1024)
                    
                    if 'DMT' in name:
                        by_condition['DMT'].append((f, size_mb))
                    elif 'EC' in name and 'PEC' not in name:
                        by_condition['EC'].append((f, size_mb))
                    elif 'EO' in name:
                        by_condition['EO'].append((f, size_mb))
                    else:
                        by_condition['Other'].append((f, size_mb))
                
                colors = {'DMT': '#00d4aa', 'EC': '#f59e0b', 'EO': '#a78bfa', 'Other': '#888888'}
                
                for cond, file_list in by_condition.items():
                    if not file_list:
                        continue
                    
                    total_size = sum(s for _, s in file_list)
                    
                    with ui.card().classes('p-2 my-1').style(
                        f'background: {colors[cond]}15; border: 1px solid {colors[cond]}40;'
                    ):
                        with ui.row().classes('items-center justify-between w-full'):
                            ui.label(cond).style(
                                f'color: {colors[cond]}; font-weight: bold; font-size: 0.8rem;'
                            )
                            ui.label(f'{len(file_list)} archivos').style(
                                f'color: {THEME_TEXT_DIM}; font-size: 0.7rem;'
                            )
                        ui.label(f'{total_size:.1f} MB total').style(
                            f'color: {THEME_TEXT_DIM}; font-size: 0.65rem;'
                        )
            
            # Right: Summary chart (counts only, no loading)
            with ui.column().classes('flex-1'):
                self._render_summary_chart(by_condition)
    
    def _render_summary_chart(self, by_condition: dict) -> None:
        """Render summary bar chart of file counts per condition."""
        conditions = []
        counts = []
        colors_list = []
        colors = {'DMT': '#00d4aa', 'EC': '#f59e0b', 'EO': '#a78bfa', 'Other': '#888888'}
        
        for cond, file_list in by_condition.items():
            if file_list:
                conditions.append(cond)
                counts.append(len(file_list))
                colors_list.append(colors[cond])
        
        if not conditions:
            return
        
        fig, ax = self.create_matplotlib_figure(figsize=(3.5, 2.5))
        bars = ax.bar(conditions, counts, color=colors_list)
        ax.set_ylabel('Archivos', color='#888888', fontsize=8)
        ax.set_title('Archivos por Condición', color=THEME_PRIMARY, fontsize=9)
        ax.tick_params(labelsize=8)
        
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   str(count), ha='center', va='bottom', color='#888888', fontsize=8)
        
        self.show_matplotlib(fig)
