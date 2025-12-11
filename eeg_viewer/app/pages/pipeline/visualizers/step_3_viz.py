"""
Step 3 Visualizer: Network Filter (multi2pool2.py)

Shows brain network separation - compact horizontal layout with Plotly.
"""
from typing import Dict, Any
import numpy as np
from nicegui import ui
import plotly.graph_objects as go

from config import THEME_PRIMARY, THEME_SECONDARY, THEME_TEXT_DIM
from .base import BaseVisualizer


# Schaefer 7-network parcellation info
NETWORKS = {
    'Vis': {'color': '#ff6b6b', 'parcels': 14, 'name': 'Visual'},
    'SomMot': {'color': '#4ecdc4', 'parcels': 14, 'name': 'Somatomotor'},
    'DorsAttn': {'color': '#45b7d1', 'parcels': 15, 'name': 'Dorsal Attn'},
    'SalVentAttn': {'color': '#96ceb4', 'parcels': 14, 'name': 'Salience'},
    'Limbic': {'color': '#ffeaa7', 'parcels': 14, 'name': 'Limbic'},
    'Cont': {'color': '#dfe6e9', 'parcels': 15, 'name': 'Control'},
    'Default': {'color': '#a29bfe', 'parcels': 14, 'name': 'DMN'},
}


class Step3Visualizer(BaseVisualizer):
    """Visualizer for Step 3: Network Filter."""
    
    step_number = 3
    step_name = "Filtrar por Redes"
    step_description = "7 redes cerebrales funcionales"
    file_patterns = ["network_*.pkl", "*/network_*.pkl", "**/network_*.pkl"]
    supported_backends = ['plotly']
    
    def get_controls(self) -> Dict[str, Any]:
        return {}
    
    def _render_visualization(self) -> None:
        # Horizontal layout: networks on left, pie on right
        with ui.row().classes('w-full gap-4 items-start'):
            # Left: Compact network cards in 2 columns
            with ui.column().style('min-width: 280px; max-width: 300px;'):
                ui.label('🧠 7 Redes Funcionales (Schaefer)').style(
                    f'color: {THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.8rem; margin-bottom: 4px;'
                )
                
                # Grid 2 columns
                with ui.element('div').style(
                    'display: grid; grid-template-columns: 1fr 1fr; gap: 4px;'
                ):
                    for net_id, info in NETWORKS.items():
                        with ui.element('div').classes('p-2').style(
                            f'background: {info["color"]}15; border: 1px solid {info["color"]}40; '
                            f'border-radius: 4px;'
                        ):
                            with ui.row().classes('items-center gap-1'):
                                ui.element('div').style(
                                    f'width: 8px; height: 8px; border-radius: 50%; '
                                    f'background: {info["color"]};'
                                )
                                ui.label(net_id).style(
                                    f'color: {info["color"]}; font-weight: bold; font-size: 0.7rem;'
                                )
                            ui.label(f'{info["parcels"]} parcelas').style(
                                f'color: {THEME_TEXT_DIM}; font-size: 0.6rem;'
                            )
            
            # Right: Compact Plotly pie chart
            with ui.column().style('flex: 1; max-width: 280px;'):
                self._render_network_pie()
    
    def _render_network_pie(self) -> None:
        """Render compact pie chart using Plotly."""
        labels = list(NETWORKS.keys())
        values = [info['parcels'] for info in NETWORKS.values()]
        colors = [info['color'] for info in NETWORKS.values()]
        
        fig = go.Figure(data=[
            go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=colors, line=dict(color='#1a1a1a', width=1)),
                textinfo='percent',
                textfont=dict(size=10, color='#000000'),
                hovertemplate='%{label}<br>%{value} parcelas<br>%{percent}<extra></extra>',
                hole=0.3
            )
        ])
        
        fig.update_layout(
            template='plotly_dark',
            height=220,
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title=dict(
                text='Distribución de Parcelas',
                font=dict(size=10, color=THEME_PRIMARY),
                x=0.5
            ),
            showlegend=False,
            font=dict(size=9)
        )
        
        ui.plotly(fig).classes('w-full')
