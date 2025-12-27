"""
Step 6 Visualizer: Aggregate Metrics (build_order_data.py)

Shows aggregated metrics - file summary only (no chart to avoid duplication with Step 2).
"""
from typing import Dict, Any
from nicegui import ui

from config import THEME_PRIMARY, THEME_SECONDARY, THEME_TEXT_DIM
from .base import BaseVisualizer


class Step6Visualizer(BaseVisualizer):
    """Visualizer for Step 6: Aggregate Metrics."""
    
    step_number = 6
    step_name = "Agregar Métricas"
    step_description = "Resumen de todos los sujetos"
    # order_all-*.pkl in subdirs OR order_all*.pkl in root
    file_patterns = ["order_all*.pkl", "*/order_all*.pkl", "**/order_all-*.pkl"]
    supported_backends = ['plotly']
    
    def get_controls(self) -> Dict[str, Any]:
        return {}
    
    def _render_visualization(self) -> None:
        files = self.find_data_files()
        
        if not files:
            self._render_no_data_message("No hay datos agregados (order_all*.pkl)")
            return
        
        # Group by condition
        by_condition = {'DMT': [], 'EC': [], 'EO': [], 'Other': []}
        total_size = 0
        
        for f in files:
            name = f.name.upper()
            size_mb = f.stat().st_size / (1024 * 1024)
            total_size += size_mb
            
            if 'DMT' in name:
                by_condition['DMT'].append((f, size_mb))
            elif 'EC' in name and 'PEC' not in name:
                by_condition['EC'].append((f, size_mb))
            elif 'EO' in name:
                by_condition['EO'].append((f, size_mb))
            else:
                by_condition['Other'].append((f, size_mb))
        
        colors = {'DMT': '#00d4aa', 'EC': '#f59e0b', 'EO': '#a78bfa', 'Other': '#888888'}
        
        # Compact file summary (no chart - that's Step 2's job)
        with ui.column().classes('w-full gap-2'):
            ui.label('📊 Métricas Agregadas').style(
                f'color: {THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.85rem;'
            )
            
            # Summary row
            with ui.row().classes('gap-4 items-center mb-2'):
                ui.label(f'{len(files)} archivos').style(
                    f'color: {THEME_SECONDARY}; font-size: 0.8rem;'
                )
                ui.label(f'{total_size:.1f} MB total').style(
                    f'color: {THEME_TEXT_DIM}; font-size: 0.75rem;'
                )
            
            # Compact cards grid
            with ui.element('div').style(
                'display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 8px;'
            ):
                for cond, file_list in by_condition.items():
                    if not file_list:
                        continue
                    
                    cond_size = sum(s for _, s in file_list)
                    
                    with ui.card().classes('p-2').style(
                        f'background: {colors[cond]}15; border: 1px solid {colors[cond]}40;'
                    ):
                        with ui.row().classes('items-center gap-2'):
                            ui.element('div').style(
                                f'width: 10px; height: 10px; border-radius: 50%; background: {colors[cond]};'
                            )
                            ui.label(cond).style(
                                f'color: {colors[cond]}; font-weight: bold; font-size: 0.8rem;'
                            )
                        ui.label(f'{len(file_list)} archivos • {cond_size:.1f} MB').style(
                            f'color: {THEME_TEXT_DIM}; font-size: 0.7rem;'
                        )
            
            # Info message
            ui.separator().classes('my-2')
            with ui.row().classes('items-center gap-2'):
                ui.icon('info', size='xs').style(f'color: {THEME_TEXT_DIM};')
                ui.label(
                    'Archivos order_all*.pkl contienen métricas Kuramoto agregadas de todos los sujetos. '
                    'Son usados por los pasos de Pearson y Clustering.'
                ).style(f'color: {THEME_TEXT_DIM}; font-size: 0.7rem;')
