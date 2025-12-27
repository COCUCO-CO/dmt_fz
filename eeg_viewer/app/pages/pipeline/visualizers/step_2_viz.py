"""
Step 2 Visualizer: Consolidate Data (save_load_pickle.py)

Shows consolidated subject data overview - compact layout with Plotly.
"""
from typing import Dict, Any, List
from nicegui import ui
import plotly.graph_objects as go

from config import THEME_PRIMARY, THEME_SECONDARY, THEME_TEXT_DIM
from .base import BaseVisualizer


class Step2Visualizer(BaseVisualizer):
    """Visualizer for Step 2: Consolidate Data."""
    
    step_number = 2
    step_name = "Consolidar Datos"
    step_description = "Agrupa archivos por condición"
    file_patterns = ["subject_phases_*.pkl", "*/subject_phases_*.pkl", "**/subject_phases_*.pkl"]
    supported_backends = ['plotly']
    
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
            with ui.column().style('min-width: 180px; max-width: 200px;'):
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
            
            # Right: Compact Plotly bar chart
            if conditions:
                with ui.column().classes('flex-1').style('max-width: 350px;'):
                    self._render_condition_chart(conditions, counts)
    
    def _render_condition_chart(self, conditions: List[str], counts: List[int]) -> None:
        """Render compact bar chart using Plotly."""
        colors = {'DMT': '#00d4aa', 'EC': '#f59e0b', 'EO': '#a78bfa'}
        bar_colors = [colors.get(c, '#888888') for c in conditions]
        
        fig = go.Figure(data=[
            go.Bar(
                x=conditions,
                y=counts,
                marker_color=bar_colors,
                text=counts,
                textposition='outside',
                textfont=dict(size=12, color='#888888'),
                hovertemplate='%{x}: %{y} sujetos<extra></extra>'
            )
        ])
        
        fig.update_layout(
            template='plotly_dark',
            height=200,
            margin=dict(l=40, r=20, t=30, b=30),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title=dict(
                text='Sujetos por Condición',
                font=dict(size=11, color=THEME_PRIMARY),
                x=0.5
            ),
            xaxis=dict(
                tickfont=dict(size=10, color='#888888'),
                showgrid=False
            ),
            yaxis=dict(
                title='Sujetos',
                titlefont=dict(size=9, color='#888888'),
                tickfont=dict(size=9, color='#888888'),
                gridcolor='rgba(100,100,100,0.2)'
            ),
            showlegend=False,
            bargap=0.3
        )
        
        ui.plotly(fig).classes('w-full')
