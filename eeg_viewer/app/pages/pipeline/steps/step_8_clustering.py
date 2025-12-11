"""
Step 8: Clustering (clustering.py)

Identifies brain states through clustering analysis.
This step has additional controls for bands and cluster parameters.
"""
from nicegui import ui

from config import THEME_TEXT_DIM

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
        """Render clustering-specific controls."""
        with container:
            # Band selection
            with ui.row().classes('gap-4 items-center mt-2'):
                ui.label('Bands:').style(
                    f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.7rem;'
                )
                self._band_checkboxes = {
                    'Delta': ui.checkbox('δ', value=True).props('dense'),
                    'Theta': ui.checkbox('θ', value=True).props('dense'),
                    'Alpha': ui.checkbox('α', value=True).props('dense'),
                    'Beta': ui.checkbox('β', value=True).props('dense'),
                    'Gamma': ui.checkbox('γ', value=True).props('dense'),
                }
            
            # Cluster and PCA range
            with ui.row().classes('gap-3 items-center mt-2'):
                ui.label('Clusters:').style(
                    f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.7rem;'
                )
                self._min_k = ui.number(value=2, min=2, max=20).props('dense').classes('w-16')
                ui.label('-').style(f'color:{THEME_TEXT_DIM};')
                self._max_k = ui.number(value=15, min=2, max=30).props('dense').classes('w-16')
                
                ui.label('PCA:').style(
                    f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.7rem;'
                )
                self._min_pca = ui.number(value=2, min=2, max=20).props('dense').classes('w-16')
                ui.label('-').style(f'color:{THEME_TEXT_DIM};')
                self._max_pca = ui.number(value=10, min=2, max=30).props('dense').classes('w-16')
            
            # Search mode toggle
            self._search_mode = ui.toggle(['Quick', 'Full'], value='Quick').props('dense')
    
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

