"""
Visualize tab for pipeline data visualization.

Comprehensive visualization dashboard for pipeline outputs.
This is the most complex tab with multiple plot types and data sources.
"""
import pickle
import asyncio
from pathlib import Path
from typing import Callable, Optional
from nicegui import ui

from config import (
    THEME_PRIMARY, THEME_SECONDARY, THEME_WARN, THEME_ERROR,
    THEME_TEXT_DIM, THEME_CARD, THEME_BORDER, THEME_TEXT
)
from app.state import PS
from ..utils import find_subjects_in_directory


class VisualizeTab:
    """
    Visualization tab component.
    
    Responsibilities:
    - Data loading and management
    - Multiple visualization types (brain, kuramoto, hilbert, etc.)
    - Clustering and Pearson result galleries
    - Animation generation
    
    State is stored in PS.viz_state for persistence across tab switches.
    """
    
    def __init__(self, get_run_dir: Callable[[], Optional[Path]]):
        """
        Initialize visualize tab.
        
        Args:
            get_run_dir: Function that returns current run directory
        """
        self._get_run_dir = get_run_dir
        
        # Initialize viz_state on PS if not present
        if not hasattr(PS, 'viz_state') or PS.viz_state is None:
            PS.viz_state = {'data': None, 'file': None, 'loaded': False}
        
        # UI element references
        self._viz_band = None
        self._viz_subject = None
        self._viz_epoch = None
        self._viz_status = None
        self._custom_data_path = None
        
        # Plot containers
        self._containers = {}
    
    @property
    def viz_state(self) -> dict:
        """Get visualization state."""
        return PS.viz_state
    
    def render(self) -> None:
        """Render the visualize tab content."""
        with ui.tab_panel('visualize').classes('p-0').style(
            'height: 100%; overflow: hidden;'
        ):
            with ui.column().classes('w-full h-full').style(
                'display: flex; flex-direction: column; overflow: hidden;'
            ):
                self._render_header()
                self._render_content()
    
    def _render_header(self) -> None:
        """Render data selection header."""
        with ui.card().classes('dark-card p-3 w-full').style('flex-shrink: 0;'):
            # Custom data path
            with ui.row().classes('items-center gap-2 w-full mb-2'):
                ui.label('Data Path:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                self._custom_data_path = ui.input(
                    placeholder='/path/to/data or leave empty for run dir'
                ).props('dense dark').classes('flex-1')
                ui.button('Load', on_click=self._browse_path, icon='folder_open').props('dense flat')
            
            # Selectors
            with ui.row().classes('items-center gap-4 flex-wrap'):
                ui.label('▌VISUALIZATION').style(
                    f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; '
                    f'font-size: 0.9rem; letter-spacing: 1px;'
                )
                
                self._viz_band = ui.select(
                    ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'],
                    value='Alpha', label='Band'
                ).props('dense dark').classes('w-24')
                
                self._viz_subject = ui.select([], label='Subject').props('dense dark').classes('w-32')
                self._viz_epoch = ui.number(value=0, min=0, max=100, label='Epoch').props('dense').classes('w-20')
                
                ui.button('Load Subjects', on_click=self._load_subjects, icon='refresh').props('dense flat')
                
                self._viz_status = ui.label('No data loaded').style(
                    f'color:{THEME_TEXT_DIM}; font-size: 0.7rem; margin-left: auto;'
                )
                
                # Store refresh function
                self.viz_state['refresh_all_plots'] = self._refresh_all_plots
                
                # Connect selectors to auto-refresh
                self._viz_band.on('update:model-value', lambda e: self._refresh_all_plots())
                self._viz_subject.on('update:model-value', lambda e: self._refresh_all_plots())
                self._viz_epoch.on('update:model-value', lambda e: self._refresh_all_plots())
    
    def _render_content(self) -> None:
        """Render scrollable visualization content."""
        with ui.scroll_area().classes('w-full flex-1').style('min-height: 0;'):
            with ui.column().classes('w-full p-3 gap-3'):
                self._render_brain_row()
                self._render_kuramoto_row()
                self._render_analysis_row()
                self._render_hilbert_row()
                self._render_clustering_expansion()
                self._render_pearson_expansion()
                self._render_animation_expansion()
    
    # =========================================================================
    # DATA LOADING
    # =========================================================================
    
    def _browse_path(self) -> None:
        """Load data from custom path."""
        path = self._custom_data_path.value.strip() if self._custom_data_path.value else None
        if path:
            p = Path(path)
            if p.exists():
                self.viz_state['custom_path'] = p
                self._load_subjects_from_path(p)
            else:
                ui.notify(f'Path not found: {path}', type='warning')
        elif self._get_run_dir():
            self.viz_state['custom_path'] = None
            self._load_subjects_from_path(self._get_run_dir())
    
    def _load_subjects(self) -> None:
        """Load subjects from current path."""
        path = self.viz_state.get('custom_path') or self._get_run_dir()
        if path:
            self._load_subjects_from_path(path)
    
    def _load_subjects_from_path(self, path: Path) -> None:
        """Load subjects from given path and auto-refresh plots."""
        subjects = find_subjects_in_directory(path)
        self._viz_subject.options = subjects
        
        if subjects:
            self._viz_subject.value = subjects[0]
            ui.notify(f'Found {len(subjects)} subjects in {path.name}', type='info')
            if 'refresh_all_plots' in self.viz_state:
                self.viz_state['refresh_all_plots']()
        else:
            ui.notify(f'No subjects found in {path.name}', type='warning')
    
    def _load_data_for_subject(self) -> bool:
        """Load pickle data for current subject."""
        data_path = self.viz_state.get('custom_path') or self._get_run_dir()
        
        if not data_path or not self._viz_subject.value:
            return False
        
        try:
            p = Path(data_path)
            subj = self._viz_subject.value
            
            # Prefer phases-*.pkl (has phases_stc)
            phases_files = list(p.rglob(f'phases-*{subj}*.pkl'))
            syncro_files = list(p.rglob(f'syncro-*{subj}*.pkl'))
            
            if phases_files:
                with open(phases_files[0], 'rb') as f:
                    self.viz_state['data'] = pickle.load(f)
                self.viz_state['file'] = phases_files[0]
                self._viz_status.text = f'Loaded: {phases_files[0].name}'
                self._viz_status.style(f'color:{THEME_PRIMARY}; font-size: 0.7rem;')
                return True
            elif syncro_files:
                with open(syncro_files[0], 'rb') as f:
                    self.viz_state['data'] = pickle.load(f)
                self.viz_state['file'] = syncro_files[0]
                self._viz_status.text = f'Loaded: {syncro_files[0].name} (no phases)'
                self._viz_status.style(f'color:{THEME_WARN}; font-size: 0.7rem;')
                return True
            else:
                self._viz_status.text = f'No files for {subj}'
                self._viz_status.style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                return False
        except Exception as e:
            self._viz_status.text = f'Error: {e}'
            return False
    
    def _refresh_all_plots(self) -> None:
        """Refresh all visualizations with current parameters."""
        self._load_data_for_subject()
        
        try:
            # Update all plots
            for name in ['brain', 'network', 'kuramoto_timeline', 'band_comp',
                        'phase', 'sync_matrix', 'connectivity', 'hilbert_2d', 'hilbert_3d']:
                if name in self._containers:
                    getattr(self, f'_update_{name}_plot', lambda: None)()
            
            # Also call stored update functions
            if 'update_brain_plot' in self.viz_state:
                self.viz_state['update_brain_plot']()
            if 'update_hilbert_2d' in self.viz_state:
                self.viz_state['update_hilbert_2d']()
            if 'update_hilbert_3d' in self.viz_state:
                self.viz_state['update_hilbert_3d']()
        except Exception:
            pass
    
    # =========================================================================
    # VISUALIZATION ROWS
    # =========================================================================
    
    def _render_brain_row(self) -> None:
        """Render 3D brain and network stats row."""
        with ui.row().classes('w-full gap-3'):
            # 3D Brain
            with ui.card().classes('dark-card p-3').style('flex: 2; min-width: 400px;'):
                ui.label('▌3D BRAIN NETWORK').style(
                    f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.8rem;'
                ).classes('mb-2')
                
                self._containers['brain'] = ui.column().classes('w-full')
                self.viz_state['current_brain_plot'] = 'network'
                
                # Brain plot type buttons
                brain_buttons = {}
                with ui.row().classes('gap-2 mb-2'):
                    for btn_type, btn_label in [
                        ('network', 'Networks'), ('colored', 'Parcellation'),
                        ('sync', 'Sync Map'), ('all_bands', 'All Bands')
                    ]:
                        brain_buttons[btn_type] = ui.button(
                            btn_label,
                            on_click=lambda t=btn_type: self._update_brain_plot(t)
                        ).props('dense')
                
                def update_brain_buttons(active_type):
                    for bt, btn in brain_buttons.items():
                        if bt == active_type:
                            btn.style(f'background:{THEME_PRIMARY}; color:black;')
                        else:
                            btn.style(f'background:transparent; color:{THEME_TEXT_DIM}; border: 1px solid {THEME_BORDER};')
                
                self._brain_buttons_update = update_brain_buttons
                update_brain_buttons('network')
                self._update_brain_plot('network')
                self.viz_state['update_brain_plot'] = self._update_brain_plot
            
            # Network Stats
            with ui.card().classes('dark-card p-3').style('flex: 1; min-width: 300px;'):
                ui.label('▌NETWORK SYNC').style(
                    f'color:{THEME_SECONDARY}; font-family: JetBrains Mono; font-size: 0.8rem;'
                ).classes('mb-2')
                self._containers['network'] = ui.column().classes('w-full')
                if self.viz_state.get('data'):
                    self._update_network_plot()
    
    def _render_kuramoto_row(self) -> None:
        """Render Kuramoto analysis row."""
        with ui.row().classes('w-full gap-3'):
            # Timeline
            with ui.card().classes('dark-card p-3 flex-1'):
                ui.label('▌KURAMOTO TIMELINE').style(
                    f'color:{THEME_WARN}; font-family: JetBrains Mono; font-size: 0.8rem;'
                ).classes('mb-2')
                ui.button('All Bands', on_click=self._show_all_bands).props('dense flat size=sm').classes('mb-1')
                self._containers['kuramoto_timeline'] = ui.column().classes('w-full')
                if self.viz_state.get('data'):
                    self._update_kuramoto_timeline_plot()
            
            # Band Comparison
            with ui.card().classes('dark-card p-3 flex-1'):
                ui.label('▌BAND COMPARISON').style(
                    f'color:{THEME_SECONDARY}; font-family: JetBrains Mono; font-size: 0.8rem;'
                ).classes('mb-2')
                self._containers['band_comp'] = ui.column().classes('w-full')
                if self.viz_state.get('data'):
                    self._update_band_comp_plot()
    
    def _render_analysis_row(self) -> None:
        """Render phase/sync/connectivity row."""
        with ui.row().classes('w-full gap-3'):
            # Phase Distribution
            with ui.card().classes('dark-card p-3 flex-1'):
                ui.label('▌PHASE DISTRIBUTION').style(
                    'color:#a78bfa; font-family: JetBrains Mono; font-size: 0.8rem;'
                ).classes('mb-2')
                self._containers['phase'] = ui.column().classes('w-full')
                if self.viz_state.get('data'):
                    self._update_phase_plot()
            
            # Sync Matrix
            with ui.card().classes('dark-card p-3 flex-1'):
                ui.label('▌SYNC MATRIX').style(
                    'color:#60a5fa; font-family: JetBrains Mono; font-size: 0.8rem;'
                ).classes('mb-2')
                self._containers['sync_matrix'] = ui.column().classes('w-full')
                if self.viz_state.get('data'):
                    self._update_sync_matrix_plot()
            
            # Connectivity
            with ui.card().classes('dark-card p-3 flex-1'):
                ui.label('▌ROI CONNECTIVITY').style(
                    'color:#22c55e; font-family: JetBrains Mono; font-size: 0.8rem;'
                ).classes('mb-2')
                self._conn_threshold = ui.slider(min=0.3, max=0.9, step=0.1, value=0.5).props('label-always').classes('w-full')
                self._conn_threshold.on('update:model-value', lambda e: self._update_connectivity_plot())
                self._containers['connectivity'] = ui.column().classes('w-full')
                if self.viz_state.get('data'):
                    self._update_connectivity_plot()
    
    def _render_hilbert_row(self) -> None:
        """Render Hilbert transform visualizations."""
        with ui.row().classes('w-full gap-3'):
            # Hilbert 2D
            with ui.card().classes('dark-card p-3 flex-1'):
                ui.label('▌HILBERT 2D').style(
                    'color:#f472b6; font-family: JetBrains Mono; font-size: 0.8rem;'
                ).classes('mb-2')
                self._containers['hilbert_2d'] = ui.column().classes('w-full')
                self.viz_state['update_hilbert_2d'] = self._update_hilbert_2d_plot
                if self.viz_state.get('data'):
                    self._update_hilbert_2d_plot()
            
            # Hilbert 3D
            with ui.card().classes('dark-card p-3 flex-1'):
                ui.label('▌HILBERT 3D PHASE SPACE').style(
                    'color:#c084fc; font-family: JetBrains Mono; font-size: 0.8rem;'
                ).classes('mb-2')
                self._containers['hilbert_3d'] = ui.column().classes('w-full')
                self.viz_state['update_hilbert_3d'] = self._update_hilbert_3d_plot
                if self.viz_state.get('data'):
                    self._update_hilbert_3d_plot()
    
    # =========================================================================
    # EXPANSION PANELS
    # =========================================================================
    
    def _render_clustering_expansion(self) -> None:
        """Render clustering analysis expansion."""
        with ui.expansion('CLUSTERING ANALYSIS', icon='analytics').classes('w-full').style(
            f'background:{THEME_CARD};'
        ):
            with ui.row().classes('w-full gap-3 p-2'):
                # Cluster Scores
                with ui.card().classes('dark-card p-3 flex-1'):
                    ui.label('▌CLUSTER SCORES').style(
                        'color:#ec4899; font-family: JetBrains Mono; font-size: 0.8rem;'
                    ).classes('mb-2')
                    ui.button('Load', on_click=self._update_cluster_scores).props('dense flat size=sm').classes('mb-2')
                    self._containers['cluster_scores'] = ui.column().classes('w-full')
                
                # PCA Scatter
                with ui.card().classes('dark-card p-3 flex-1'):
                    ui.label('▌PCA CLUSTERS').style(
                        'color:#f97316; font-family: JetBrains Mono; font-size: 0.8rem;'
                    ).classes('mb-2')
                    ui.button('Load', on_click=self._update_pca_scatter).props('dense flat size=sm').classes('mb-2')
                    self._containers['pca_scatter'] = ui.column().classes('w-full')
    
    def _render_pearson_expansion(self) -> None:
        """Render Pearson correlations expansion."""
        with ui.expansion('PEARSON CORRELATIONS', icon='insights').classes('w-full').style(
            f'background:{THEME_CARD};'
        ):
            with ui.column().classes('w-full p-2'):
                with ui.row().classes('gap-2 items-center mb-2'):
                    self._prs_metric = ui.select(['All', 'Coherence', 'Metastability'], value='All', label='Metric').props('dense').classes('w-28')
                    self._prs_cond = ui.select(['All', 'DMT', 'EC', 'EO'], value='All', label='Condition').props('dense').classes('w-20')
                    self._prs_band = ui.select(['All', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'], value='All', label='Band').props('dense').classes('w-24')
                
                self._pearson_gallery = ui.column().classes('w-full')
                self._pearson_dialog = ui.dialog().classes('w-full max-w-4xl')
                
                for sel in [self._prs_metric, self._prs_cond, self._prs_band]:
                    sel.on('update:model-value', lambda e: self._refresh_pearson_gallery())
                
                ui.button('Load Images', on_click=self._refresh_pearson_gallery, icon='refresh').props('dense flat').classes('mt-2')
    
    def _render_animation_expansion(self) -> None:
        """Render animation generator expansion."""
        with ui.expansion('ANIMATION GENERATOR', icon='movie').classes('w-full').style(
            f'background:{THEME_CARD};'
        ):
            with ui.column().classes('w-full p-3 gap-3'):
                ui.label('Generate Kuramoto visualization frames and animations').style(
                    f'color:{THEME_TEXT_DIM}; font-size: 0.75rem;'
                )
                
                with ui.row().classes('gap-4 items-center flex-wrap'):
                    self._anim_mode = ui.select(['stc', 'eeg', 'all', 'advanced'], value='stc', label='Mode').props('dense').classes('w-28')
                    self._anim_quality = ui.select(['high', 'medium', 'low'], value='medium', label='Quality').props('dense').classes('w-24')
                    self._anim_format = ui.select(['png', 'jpg'], value='png', label='Format').props('dense').classes('w-20')
                    self._anim_start_epoch = ui.number(value=0, min=0, max=100, label='Start').props('dense').classes('w-20')
                    self._anim_end_epoch = ui.number(value=10, min=1, max=200, label='End').props('dense').classes('w-20')
                    self._anim_fps = ui.number(value=5, min=1, max=30, label='FPS').props('dense').classes('w-16')
                
                with ui.row().classes('gap-4 items-center'):
                    self._anim_output_dir = ui.input(value='', placeholder='/path/to/output or auto').props('dense').classes('flex-1')
                    ui.label('Output dir (leave empty for auto)').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                
                self._anim_log = ui.column().classes('w-full').style(
                    'max-height: 800px; overflow-y: auto; background: #050505; '
                    'border-radius: 4px; padding: 8px;'
                )
                
                with ui.row().classes('gap-2'):
                    ui.button('Generate Frames', on_click=self._generate_frames, icon='photo_library').props('dense').style(
                        f'background:{THEME_PRIMARY}; color:black;'
                    )
                    ui.button('Create Video', on_click=self._generate_video, icon='movie').props('dense').style(
                        f'background:{THEME_SECONDARY}; color:black;'
                    )
                    ui.button('Clear Log', on_click=lambda: self._anim_log.clear(), icon='delete').props('dense flat')
    
    # =========================================================================
    # PLOT UPDATE METHODS
    # =========================================================================
    
    def _update_brain_plot(self, plot_type: str = None) -> None:
        """Update 3D brain plot."""
        if plot_type is None:
            plot_type = self.viz_state.get('current_brain_plot', 'network')
        else:
            self.viz_state['current_brain_plot'] = plot_type
        
        container = self._containers.get('brain')
        if not container:
            return
            
        container.clear()
        try:
            from viz_scripts import brain_3d
            data = self.viz_state.get('data')
            
            with container:
                if plot_type == 'network':
                    fig = brain_3d.create_brain_network_figure()
                elif plot_type == 'colored':
                    fig = brain_3d.create_colored_brain_figure()
                elif plot_type == 'sync':
                    fig = brain_3d.create_sync_brain_figure(
                        data, self._viz_band.value, int(self._viz_epoch.value or 0)
                    )
                elif plot_type == 'all_bands':
                    fig = brain_3d.create_all_bands_brain_figure(data)
                else:
                    fig = brain_3d.create_brain_network_figure()
                
                ui.plotly(fig).classes('w-full').style('height: 450px;')
            
            if hasattr(self, '_brain_buttons_update'):
                self._brain_buttons_update(plot_type)
        except Exception as e:
            with container:
                ui.label(f'Error: {e}').style(f'color:{THEME_ERROR}; font-size: 0.75rem;')
    
    def _update_network_plot(self) -> None:
        """Update network comparison plot."""
        self._update_generic_plot(
            'network', 'brain_3d', 'create_network_comparison_figure',
            lambda: [self.viz_state.get('data'), self._viz_band.value],
            'height: 350px;'
        )
    
    def _update_kuramoto_timeline_plot(self) -> None:
        """Update Kuramoto timeline plot."""
        self._update_generic_plot(
            'kuramoto_timeline', 'kuramoto_viz', 'create_timeline_figure',
            lambda: [self.viz_state.get('data'), self._viz_band.value],
            'height: 280px;'
        )
    
    def _show_all_bands(self) -> None:
        """Show all bands in timeline."""
        container = self._containers.get('kuramoto_timeline')
        if not container:
            return
        container.clear()
        try:
            from viz_scripts import kuramoto_viz
            with container:
                fig = kuramoto_viz.create_all_bands_timeline(self.viz_state.get('data'))
                ui.plotly(fig).classes('w-full').style('height: 280px;')
        except Exception as e:
            with container:
                ui.label(f'Error: {e}').style(f'color:{THEME_TEXT_DIM};')
    
    def _update_band_comp_plot(self) -> None:
        """Update band comparison plot."""
        self._update_generic_plot(
            'band_comp', 'kuramoto_viz', 'create_band_comparison_figure',
            lambda: [self.viz_state.get('data')],
            'height: 280px;'
        )
    
    def _update_phase_plot(self) -> None:
        """Update phase distribution plot."""
        self._update_generic_plot(
            'phase', 'kuramoto_viz', 'create_phase_distribution_figure',
            lambda: [self.viz_state.get('data'), self._viz_band.value, int(self._viz_epoch.value or 0)],
            'height: 280px;'
        )
    
    def _update_sync_matrix_plot(self) -> None:
        """Update sync matrix plot."""
        self._update_generic_plot(
            'sync_matrix', 'kuramoto_viz', 'create_heatmap_figure',
            lambda: [self.viz_state.get('data'), self._viz_band.value, int(self._viz_epoch.value or 0)],
            'height: 280px;'
        )
    
    def _update_connectivity_plot(self) -> None:
        """Update connectivity plot."""
        self._update_generic_plot(
            'connectivity', 'kuramoto_viz', 'create_roi_connectivity_figure',
            lambda: [self.viz_state.get('data'), self._viz_band.value, self._conn_threshold.value],
            'height: 250px;'
        )
    
    def _update_hilbert_2d_plot(self) -> None:
        """Update Hilbert 2D plot."""
        self._update_generic_plot(
            'hilbert_2d', 'kuramoto_viz', 'create_hilbert_2d_figure',
            lambda: [self.viz_state.get('data'), self._viz_band.value, int(self._viz_epoch.value or 0)],
            'height: 500px;'
        )
    
    def _update_hilbert_3d_plot(self) -> None:
        """Update Hilbert 3D plot."""
        container = self._containers.get('hilbert_3d')
        if not container:
            return
        container.clear()
        data = self.viz_state.get('data')
        if not data:
            with container:
                ui.label('Load data first').style(f'color:{THEME_TEXT_DIM}; font-size: 0.75rem;')
            return
        try:
            from viz_scripts import kuramoto_viz
            subj = self._viz_subject.value or 'S01'
            cond = 'DMT'
            if '-' in str(subj):
                cond = str(subj).split('-')[-1]
            
            with container:
                fig = kuramoto_viz.create_hilbert_3d_figure(
                    data, self._viz_band.value, int(self._viz_epoch.value or 0),
                    roi_idx=0, subject=subj, condition=cond
                )
                ui.plotly(fig).classes('w-full').style('height: 500px;')
        except Exception as e:
            with container:
                ui.label(f'Error: {e}').style(f'color:{THEME_TEXT_DIM};')
    
    def _update_generic_plot(self, container_name: str, module_name: str, func_name: str,
                            args_getter: Callable, style: str) -> None:
        """Generic plot update helper."""
        container = self._containers.get(container_name)
        if not container:
            return
        container.clear()
        data = self.viz_state.get('data')
        if not data:
            with container:
                ui.label('Load data first').style(f'color:{THEME_TEXT_DIM}; font-size: 0.75rem;')
            return
        try:
            module = __import__(f'viz_scripts.{module_name}', fromlist=[func_name])
            func = getattr(module, func_name)
            with container:
                fig = func(*args_getter())
                ui.plotly(fig).classes('w-full').style(style)
        except Exception as e:
            with container:
                ui.label(f'Error: {e}').style(f'color:{THEME_TEXT_DIM};')
    
    def _update_cluster_scores(self) -> None:
        """Update cluster scores plot."""
        container = self._containers.get('cluster_scores')
        if not container:
            return
        container.clear()
        try:
            from viz_scripts import clustering_viz
            run_dir = self._get_run_dir()
            if run_dir:
                with container:
                    fig = clustering_viz.create_band_comparison_figure(run_dir)
                    ui.plotly(fig).classes('w-full').style('height: 280px;')
            else:
                with container:
                    ui.label('Select a run first').style(f'color:{THEME_TEXT_DIM};')
        except Exception as e:
            with container:
                ui.label(f'Error: {e}').style(f'color:{THEME_TEXT_DIM};')
    
    def _update_pca_scatter(self) -> None:
        """Update PCA scatter plot."""
        container = self._containers.get('pca_scatter')
        if not container:
            return
        container.clear()
        try:
            from viz_scripts import clustering_viz
            run_dir = self._get_run_dir()
            if run_dir:
                with container:
                    fig = clustering_viz.create_pca_scatter_figure(run_dir, self._viz_band.value)
                    ui.plotly(fig).classes('w-full').style('height: 280px;')
            else:
                with container:
                    ui.label('Select a run first').style(f'color:{THEME_TEXT_DIM};')
        except Exception as e:
            with container:
                ui.label(f'Error: {e}').style(f'color:{THEME_TEXT_DIM};')
    
    def _refresh_pearson_gallery(self) -> None:
        """Refresh Pearson results gallery."""
        self._pearson_gallery.clear()
        run_dir = self._get_run_dir()
        
        if not run_dir:
            with self._pearson_gallery:
                ui.label('No run selected').style(f'color:{THEME_TEXT_DIM};')
            return
        
        pearson_dir = run_dir / 'pearson_results'
        if not pearson_dir.exists():
            with self._pearson_gallery:
                ui.label('No Pearson results. Run pearson.py first.').style(f'color:{THEME_TEXT_DIM};')
            return
        
        all_files = list(pearson_dir.glob('*.png')) + list(pearson_dir.glob('*.svg'))
        filtered = [f for f in all_files if 
            (self._prs_metric.value == 'All' or self._prs_metric.value.lower() in f.name.lower()) and
            (self._prs_cond.value == 'All' or self._prs_cond.value.lower() in f.name.lower()) and
            (self._prs_band.value == 'All' or self._prs_band.value.lower() in f.name.lower())]
        
        with self._pearson_gallery:
            ui.label(f'{len(filtered)} images').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
            with ui.element('div').classes('grid gap-3 mt-2').style(
                'grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));'
            ):
                for img_file in sorted(filtered)[:30]:
                    def show_full_image(p=img_file):
                        self._pearson_dialog.clear()
                        with self._pearson_dialog:
                            with ui.column().classes('w-full items-center'):
                                ui.label(p.name).style(f'color:{THEME_PRIMARY}; font-size: 0.9rem; margin-bottom: 10px;')
                                if p.suffix == '.png':
                                    ui.image(str(p)).classes('w-full').style('max-height: 70vh;')
                                else:
                                    ui.html(f'<object data="{p}" type="image/svg+xml" style="width:100%; max-height: 70vh;"></object>', sanitize=False)
                                ui.button('Close', on_click=self._pearson_dialog.close).props('flat').classes('mt-3')
                        self._pearson_dialog.open()
                    
                    with ui.card().classes('cursor-pointer p-2').style(f'background:{THEME_CARD};').on('click', show_full_image):
                        label = img_file.stem[:30] + ('...' if len(img_file.stem) > 30 else '')
                        ui.label(label).style(f'color:{THEME_TEXT}; font-size: 0.65rem;')
    
    # =========================================================================
    # ANIMATION METHODS
    # =========================================================================
    
    async def _generate_frames(self) -> None:
        """Generate visualization frames."""
        self._anim_log.clear()
        with self._anim_log:
            ui.label('Starting frame generation...').style(f'color:{THEME_PRIMARY}; font-size: 0.7rem;')
        
        data_path = self.viz_state.get('custom_path') or self._get_run_dir()
        if not data_path or not self._viz_subject.value:
            with self._anim_log:
                ui.label('Error: No data loaded. Load a subject first.').style(f'color:{THEME_ERROR}; font-size: 0.7rem;')
            return
        
        import os
        
        script_path = Path('/media/storage_hdd/dmt_fz/viz_scripts/plot.py')
        conda_prefix = os.environ.get('CONDA_PREFIX', os.path.expanduser('~/anaconda3/envs/dmt_fz'))
        python_path = Path(conda_prefix) / 'bin' / 'python'
        if not python_path.exists():
            python_path = 'python'
        
        subj = self._viz_subject.value or 'S01'
        cond = 'DMT'
        if '-' in str(subj):
            parts = str(subj).split('-')
            subj = parts[0]
            cond = parts[1] if len(parts) > 1 else 'DMT'
        
        mode = self._anim_mode.value
        start_ep = int(self._anim_start_epoch.value or 0)
        end_ep = int(self._anim_end_epoch.value or 10)
        
        cmd = [
            str(python_path), str(script_path),
            '--subject', subj,
            '--condition', cond,
            '--band', self._viz_band.value,
            '--mode', mode,
            '--epochs', f'{start_ep}:{end_ep}',
        ]
        
        band = self._viz_band.value or 'Alpha'
        expected_output = Path('/media/storage_hdd/dmt_fz/visualizations/plot') / mode / f'{subj}_{cond}_{band}'
        
        with self._anim_log:
            ui.label(f'Command: {" ".join(cmd)}').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
            ui.label(f'Frames will be saved to: {expected_output}').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
        
        try:
            env = os.environ.copy()
            env['PIPELINE_OUTPUT_DIR'] = str(data_path)
            base_path = Path('/media/storage_hdd/dmt_fz')
            pythonpath = [str(base_path / 'viz_scripts'), str(base_path / 'pipeline'), str(base_path)]
            existing_pythonpath = env.get('PYTHONPATH', '')
            env['PYTHONPATH'] = ':'.join(pythonpath) + (':' + existing_pythonpath if existing_pythonpath else '')
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            
            async def read_output(stream):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    text = line.decode().strip()
                    if text:
                        with self._anim_log:
                            ui.label(text).style(f'color:{THEME_TEXT}; font-size: 0.65rem;')
            
            await asyncio.gather(read_output(process.stdout), read_output(process.stderr))
            await process.wait()
            
            with self._anim_log:
                if process.returncode == 0:
                    ui.label('✓ Frames generated successfully!').style(f'color:{THEME_PRIMARY}; font-size: 0.75rem;')
                    frames = sorted(expected_output.glob(f'*.{self._anim_format.value}'))
                    if frames:
                        ui.label(f'Generated {len(frames)} frames').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                        with ui.card().classes('mt-2 p-2').style('background: #1a1a1a;'):
                            ui.label('Preview (first frame):').style(f'color:{THEME_TEXT_DIM}; font-size: 0.6rem;')
                            ui.image(str(frames[0])).classes('w-full').style('max-height: 1024px; max-width: 1024px; object-fit: contain;')
                else:
                    ui.label(f'Process exited with code {process.returncode}').style(f'color:{THEME_WARN}; font-size: 0.7rem;')
        except Exception as e:
            with self._anim_log:
                ui.label(f'Error: {e}').style(f'color:{THEME_ERROR}; font-size: 0.7rem;')
    
    async def _generate_video(self) -> None:
        """Generate video from frames."""
        self._anim_log.clear()
        with self._anim_log:
            ui.label('🎬 Generating video from frames...').style(f'color:{THEME_PRIMARY}; font-size: 0.7rem;')
        
        subj = self._viz_subject.value or 'S01'
        cond = 'DMT'
        if '-' in str(subj):
            parts = str(subj).split('-')
            subj = parts[0]
            cond = parts[1] if len(parts) > 1 else 'DMT'
        
        mode = self._anim_mode.value or 'stc'
        band = self._viz_band.value or 'Alpha'
        
        if self._anim_output_dir.value and self._anim_output_dir.value.strip():
            frames_dir = Path(self._anim_output_dir.value.strip())
        else:
            frames_dir = Path('/media/storage_hdd/dmt_fz/visualizations/plot') / mode / f'{subj}_{cond}_{band}'
        
        with self._anim_log:
            ui.label(f'Looking for frames in: {frames_dir}').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
        
        if not frames_dir.exists():
            with self._anim_log:
                ui.label(f'Frames directory not found: {frames_dir}').style(f'color:{THEME_ERROR}; font-size: 0.7rem;')
            return
        
        output_video = frames_dir.parent / f'{self._viz_subject.value}_{self._viz_band.value}_animation.mp4'
        
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(int(self._anim_fps.value or 5)),
            '-pattern_type', 'glob',
            '-i', str(frames_dir / f'*.{self._anim_format.value}'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            str(output_video)
        ]
        
        with self._anim_log:
            ui.label(f'Running: ffmpeg -> {output_video.name}').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await process.communicate()
            
            with self._anim_log:
                if process.returncode == 0:
                    ui.label(f'✓ Video saved: {output_video}').style(f'color:{THEME_PRIMARY}; font-size: 0.75rem;')
                    if output_video.exists():
                        with ui.card().classes('mt-2 p-2 w-full').style('background: #1a1a1a;'):
                            ui.label('Generated video:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.6rem;')
                            ui.video(str(output_video)).classes('w-full').style('max-height: 1024px; max-width: 1024px;')
                else:
                    ui.label(f'ffmpeg error: {stderr.decode()[:200]}').style(f'color:{THEME_ERROR}; font-size: 0.65rem;')
        except Exception as e:
            with self._anim_log:
                ui.label(f'Error: {e}').style(f'color:{THEME_ERROR}; font-size: 0.7rem;')

