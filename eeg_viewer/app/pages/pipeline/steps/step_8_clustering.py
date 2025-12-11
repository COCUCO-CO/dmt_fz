"""
Step 8: Clustering (clustering.py)

Identifies brain states through clustering analysis.
This step has additional controls for bands and cluster parameters.
All parameters persist in PS across page navigations.
"""
from nicegui import ui

from config import THEME_TEXT_DIM

from app.state import PS
from .base import BasePipelineStep, StepContext


class Step8Clustering(BasePipelineStep):
    """Clustering analysis step using clustering.py."""
    
    script_name = "clustering.py"
    display_name = "Clustering"
    title = "// STEP_8: CLUSTERING (OPTIONAL)"
    description = "clustering.py - Brain state identification"
    output_pattern = "→ run_*/clustering_results/"
    color = "#ff6b9d"  # Pink
    estimated_time = "Quick: ~30min, Full: ~4h"
    
    # UI element references (set during render)
    _band_checkboxes: dict = None
    _min_k: ui.number = None
    _max_k: ui.number = None
    _min_pca: ui.number = None
    _max_pca: ui.number = None
    _search_mode: ui.toggle = None
    
    def render_extra_controls(self, container) -> None:
        """Render clustering-specific controls with values from PS."""
        with container:
            # Band selection - initialize from PS
            with ui.row().classes('gap-4 items-center mt-2'):
                ui.label('Bands:').style(
                    f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.7rem;'
                )
                self._band_checkboxes = {
                    'Delta': ui.checkbox('δ', value='Delta' in PS.bands).props('dense'),
                    'Theta': ui.checkbox('θ', value='Theta' in PS.bands).props('dense'),
                    'Alpha': ui.checkbox('α', value='Alpha' in PS.bands).props('dense'),
                    'Beta': ui.checkbox('β', value='Beta' in PS.bands).props('dense'),
                    'Gamma': ui.checkbox('γ', value='Gamma' in PS.bands).props('dense'),
                }
                
                # Update PS when bands change
                def update_bands():
                    PS.bands = [
                        band for band, cb in self._band_checkboxes.items()
                        if cb.value
                    ]
                
                for cb in self._band_checkboxes.values():
                    cb.on('update:model-value', lambda: update_bands())
            
            # Cluster and PCA range - initialize from PS
            with ui.row().classes('gap-3 items-center mt-2'):
                ui.label('Clusters:').style(
                    f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.7rem;'
                )
                self._min_k = ui.number(value=PS.min_k, min=2, max=20).props('dense').classes('w-16')
                ui.label('-').style(f'color:{THEME_TEXT_DIM};')
                self._max_k = ui.number(value=PS.max_k, min=2, max=30).props('dense').classes('w-16')
                
                ui.label('PCA:').style(
                    f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.7rem;'
                )
                self._min_pca = ui.number(value=PS.min_comps, min=2, max=20).props('dense').classes('w-16')
                ui.label('-').style(f'color:{THEME_TEXT_DIM};')
                self._max_pca = ui.number(value=PS.max_comps, min=2, max=30).props('dense').classes('w-16')
                
                # Update PS when values change - use args for the new value
                self._min_k.on('update:model-value', lambda e: setattr(PS, 'min_k', int(e.args if e.args else 2)))
                self._max_k.on('update:model-value', lambda e: setattr(PS, 'max_k', int(e.args if e.args else 15)))
                self._min_pca.on('update:model-value', lambda e: setattr(PS, 'min_comps', int(e.args if e.args else 2)))
                self._max_pca.on('update:model-value', lambda e: setattr(PS, 'max_comps', int(e.args if e.args else 10)))
            
            # Search mode toggle - initialize from PS
            self._search_mode = ui.toggle(['Quick', 'Full'], value=PS.clustering_search_mode).props('dense')
            self._search_mode.on('update:model-value', lambda e: setattr(PS, 'clustering_search_mode', e.args if e.args else 'Quick'))
    
    def _get_selected_bands(self) -> list[str]:
        """Get list of selected bands."""
        if not self._band_checkboxes:
            return ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
        return [
            band for band, checkbox in self._band_checkboxes.items()
            if checkbox.value
        ]
    
    def build_args(self) -> list[str]:
        """Build arguments for clustering.py."""
        ctx = self.context
        
        # Get values from UI elements
        bands = self._get_selected_bands()
        mode_arg = '--quick-search' if self._search_mode and self._search_mode.value == 'Quick' else '--full-search'
        
        min_k = int(self._min_k.value) if self._min_k else ctx.min_k
        max_k = int(self._max_k.value) if self._max_k else ctx.max_k
        min_comps = int(self._min_pca.value) if self._min_pca else ctx.min_comps
        max_comps = int(self._max_pca.value) if self._max_pca else ctx.max_comps
        
        # Build output directory path
        cluster_out = str(ctx.run_dir / "clustering_results") if ctx.run_dir else ""
        
        return [
            mode_arg,
            '--conditions', *ctx.conditions,
            '--bands', *bands,
            '--min-k', str(min_k),
            '--max-k', str(max_k),
            '--min-comps', str(min_comps),
            '--max-comps', str(max_comps),
            '--workers', str(ctx.workers),
            '--output-dir', cluster_out,
        ]

