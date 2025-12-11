"""
Step 7 Visualizer: Pearson Correlations (pearson.py)

Shows correlation analysis results with image gallery.
"""
from typing import Dict, Any, List
from pathlib import Path
import numpy as np
from nicegui import ui

from config import THEME_PRIMARY, THEME_SECONDARY, THEME_TEXT_DIM, THEME_WARN
from .base import BaseVisualizer, extract_event_value


class Step7Visualizer(BaseVisualizer):
    """Visualizer for Step 7: Pearson Correlations."""
    
    step_number = 7
    step_name = "Correlaciones"
    step_description = "Análisis de correlación entre condiciones"
    file_patterns = ["pearson_results/*.pkl", "pearson_results/**/*.pkl"]
    supported_backends = ['matplotlib', 'plotly']
    
    def __init__(self, get_run_dir):
        super().__init__(get_run_dir)
        self._filter_metric = 'All'
        self._filter_condition = 'All'
        self._filter_band = 'All'
    
    def get_controls(self) -> Dict[str, Any]:
        return {
            'metric': ['All', 'Coherence', 'Metastability'],
            'condition': ['All', 'DMT', 'EC', 'EO'],
            'band': ['All', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
        }
    
    def _render_controls(self) -> None:
        with ui.row().classes('items-center gap-3 mb-2 flex-wrap'):
            ui.select(
                ['All', 'Coherence', 'Metastability'],
                value=self._filter_metric, label='Métrica'
            ).props('dense dark').classes('w-32').on(
                'update:model-value',
                lambda e: self._on_filter_change('metric', e.args)
            )
            
            ui.select(
                ['All', 'DMT', 'EC', 'EO'],
                value=self._filter_condition, label='Condición'
            ).props('dense dark').classes('w-24').on(
                'update:model-value',
                lambda e: self._on_filter_change('condition', e.args)
            )
            
            ui.select(
                ['All', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'],
                value=self._filter_band, label='Banda'
            ).props('dense dark').classes('w-24').on(
                'update:model-value',
                lambda e: self._on_filter_change('band', e.args)
            )
    
    def _on_filter_change(self, filter_type: str, value):
        val = extract_event_value(value)
        if filter_type == 'metric':
            self._filter_metric = val
        elif filter_type == 'condition':
            self._filter_condition = val
        elif filter_type == 'band':
            self._filter_band = val
        self.render(self._container)
    
    def _get_pearson_dir(self) -> Path:
        if self.run_dir:
            return self.run_dir / 'pearson_results'
        return None
    
    def _find_images(self) -> List[Path]:
        """Find all image files matching filters."""
        pearson_dir = self._get_pearson_dir()
        if not pearson_dir or not pearson_dir.exists():
            return []
        
        all_files = list(pearson_dir.glob('*.png')) + list(pearson_dir.glob('*.svg'))
        
        # Apply filters
        filtered = []
        for f in all_files:
            name_lower = f.name.lower()
            
            # Metric filter
            if self._filter_metric != 'All':
                if self._filter_metric.lower() not in name_lower:
                    continue
            
            # Condition filter
            if self._filter_condition != 'All':
                if self._filter_condition.lower() not in name_lower:
                    continue
            
            # Band filter
            if self._filter_band != 'All':
                if self._filter_band.lower() not in name_lower:
                    continue
            
            filtered.append(f)
        
        return sorted(filtered)
    
    def _render_visualization(self) -> None:
        pearson_dir = self._get_pearson_dir()
        
        if not pearson_dir or not pearson_dir.exists():
            self._render_no_data_message("No hay resultados de Pearson")
            ui.label("Ejecutá pearson.py para generar análisis").style(
                f'color: {THEME_TEXT_DIM}; font-size: 0.7rem;'
            )
            return
        
        images = self._find_images()
        pkl_files = list(pearson_dir.glob('*.pkl'))
        
        with ui.column().classes('w-full gap-3'):
            # Stats header
            with ui.row().classes('items-center gap-2'):
                ui.label(f'📊 {len(images)} imágenes').style(
                    f'color: {THEME_PRIMARY}; font-size: 0.8rem;'
                )
                ui.label(f'| {len(pkl_files)} archivos pkl').style(
                    f'color: {THEME_TEXT_DIM}; font-size: 0.75rem;'
                )
            
            # Image gallery
            if images:
                self._render_gallery(images[:20])  # Limit to 20 for performance
            else:
                ui.label('No hay imágenes que coincidan con los filtros').style(
                    f'color: {THEME_TEXT_DIM}; font-size: 0.8rem;'
                )
    
    def _render_gallery(self, images: List[Path]) -> None:
        """Render image gallery."""
        with ui.element('div').classes('grid gap-2').style(
            'grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));'
        ):
            for img_path in images:
                self._render_thumbnail(img_path)
    
    def _render_thumbnail(self, img_path: Path) -> None:
        """Render clickable thumbnail."""
        dialog = ui.dialog()
        
        with ui.card().classes('cursor-pointer p-2 hover:opacity-80').style(
            'background: #1a1a1a; border: 1px solid #333;'
        ).on('click', dialog.open):
            # Label
            label = img_path.stem[:25] + ('...' if len(img_path.stem) > 25 else '')
            ui.label(label).style(
                f'color: {THEME_TEXT_DIM}; font-size: 0.65rem; '
                f'white-space: nowrap; overflow: hidden;'
            )
            
            # Thumbnail (only for PNG)
            if img_path.suffix == '.png':
                ui.image(str(img_path)).classes('w-full').style(
                    'max-height: 100px; object-fit: contain;'
                )
        
        # Full image dialog
        with dialog:
            with ui.card().classes('p-4').style('background: #0a0a0a; max-width: 90vw; max-height: 90vh;'):
                ui.label(img_path.name).style(
                    f'color: {THEME_PRIMARY}; font-size: 0.9rem; margin-bottom: 10px;'
                )
                if img_path.suffix == '.png':
                    ui.image(str(img_path)).classes('w-full').style('max-height: 70vh;')
                else:
                    # For SVG, use object tag with sanitize=False for proper rendering
                    ui.html(
                        f'<object data="{img_path}" type="image/svg+xml" '
                        f'style="width:100%; max-height: 70vh;"></object>',
                        sanitize=False
                    )
                ui.button('Cerrar', on_click=dialog.close).props('flat').classes('mt-3')

