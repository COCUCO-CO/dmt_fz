"""
Pearson Results Gallery Component.

Displays correlation analysis images in a filterable gallery.
"""
from pathlib import Path
from typing import Callable, Optional, List
from nicegui import ui

from config import THEME_PRIMARY, THEME_SECONDARY, THEME_TEXT_DIM


class PearsonGallery:
    """
    Gallery for Pearson correlation result images.
    
    Features:
    - Filter by metric, condition, band
    - Thumbnail grid with click-to-expand
    - Supports PNG and SVG images
    """
    
    def __init__(self, get_run_dir: Callable[[], Optional[Path]]):
        """
        Initialize gallery.
        
        Args:
            get_run_dir: Function that returns current run directory
        """
        self._get_run_dir = get_run_dir
    
    @property
    def pearson_dir(self) -> Optional[Path]:
        """Get pearson results directory."""
        run_dir = self._get_run_dir()
        if run_dir:
            return run_dir / 'pearson_results'
        return None
    
    def find_images(self, metric: str = None, condition: str = None, 
                   band: str = None) -> List[Path]:
        """
        Find images matching filters.
        
        Args:
            metric: Filter by metric (Coherence, Metastability)
            condition: Filter by condition (DMT, EC, EO)
            band: Filter by band (Delta, Theta, Alpha, Beta, Gamma)
            
        Returns:
            List of matching image paths
        """
        if not self.pearson_dir or not self.pearson_dir.exists():
            return []
        
        # Get all images
        images = list(self.pearson_dir.glob('*.png')) + list(self.pearson_dir.glob('*.svg'))
        
        # Apply filters
        filtered = []
        for img in images:
            name_lower = img.name.lower()
            
            if metric and metric.lower() not in name_lower:
                continue
            if condition and condition.lower() not in name_lower:
                continue
            if band and band.lower() not in name_lower:
                continue
            
            filtered.append(img)
        
        return sorted(filtered)
    
    def render(self, max_images: int = 30) -> None:
        """
        Render the gallery.
        
        Args:
            max_images: Maximum images to display
        """
        images = self.find_images()
        
        if not images:
            ui.label('No hay imágenes de Pearson').style(
                f'color: {THEME_TEXT_DIM}; font-size: 0.8rem;'
            )
            return
        
        ui.label(f'📊 {len(images)} resultados de correlación').style(
            f'color: {THEME_SECONDARY}; font-size: 0.8rem; margin-bottom: 8px;'
        )
        
        # Grid of thumbnails
        with ui.element('div').classes('grid gap-2').style(
            'grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));'
        ):
            for img_path in images[:max_images]:
                self._render_thumbnail(img_path)
    
    def _render_thumbnail(self, img_path: Path) -> None:
        """Render a single thumbnail card."""
        dialog = ui.dialog()
        
        with ui.card().classes('cursor-pointer p-1 hover:opacity-80').style(
            'background: #1a1a1a;'
        ).on('click', dialog.open):
            label = img_path.stem[:20] + ('...' if len(img_path.stem) > 20 else '')
            ui.label(label).style(
                f'color: {THEME_TEXT_DIM}; font-size: 0.6rem;'
            )
            if img_path.suffix == '.png':
                ui.image(str(img_path)).style('max-height: 80px; object-fit: contain;')
        
        # Full size dialog
        with dialog:
            with ui.card().classes('p-4').style('background: #0a0a0a;'):
                ui.label(img_path.name).style(f'color: {THEME_PRIMARY}; margin-bottom: 10px;')
                if img_path.suffix == '.png':
                    ui.image(str(img_path)).style('max-height: 70vh; max-width: 80vw;')
                ui.button('Cerrar', on_click=dialog.close).props('flat').classes('mt-3')

