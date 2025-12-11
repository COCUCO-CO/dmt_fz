"""
Step 3 Visualizer: Network Filter (multi2pool2.py)

Shows brain network separation - compact horizontal layout.
"""
from typing import Dict, Any
import numpy as np
from nicegui import ui

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
    supported_backends = ['matplotlib']
    
    def get_controls(self) -> Dict[str, Any]:
        return {}
    
    def _render_visualization(self) -> None:
        # Horizontal layout: networks on left, pie on right
        with ui.row().classes('w-full gap-4 items-start'):
            # Left: Compact network cards in 2 columns
            with ui.column().style('min-width: 300px;'):
                ui.label('🧠 7 Redes (Schaefer)').style(
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
                            ui.label(f'{info["parcels"]}p').style(
                                f'color: {THEME_TEXT_DIM}; font-size: 0.6rem;'
                            )
            
            # Right: Compact pie chart
            with ui.column().classes('flex-1'):
                self._render_network_pie()
    
    def _render_network_pie(self) -> None:
        """Render compact pie chart."""
        fig, ax = self.create_matplotlib_figure(figsize=(3.5, 3))
        
        labels = list(NETWORKS.keys())
        sizes = [info['parcels'] for info in NETWORKS.values()]
        colors = [info['color'] for info in NETWORKS.values()]
        
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors, autopct='%1.0f%%',
            startangle=90, pctdistance=0.75, textprops={'fontsize': 7}
        )
        
        for text in texts:
            text.set_color('#888888')
            text.set_fontsize(7)
        for autotext in autotexts:
            autotext.set_color('#000000')
            autotext.set_fontsize(6)
        
        self.show_matplotlib(fig)
