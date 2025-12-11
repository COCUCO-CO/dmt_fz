"""
Step 5 Visualizer: Kuramoto Order Parameter (generate_order.py)

Visualizes global coherence/synchronization:
- Kuramoto oscillators (polar visualization)
- R(t) timeline by band
- Mean R by frequency band (bar chart)
- Mean R by network (7 brain networks)
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from nicegui import ui
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import THEME_PRIMARY, THEME_SECONDARY, THEME_TEXT_DIM, THEME_WARN
from .base import BaseVisualizer, extract_event_value


# Global data path for pre-computed Kuramoto results
KURAMOTO_DATA_PATH = Path('/media/storage_hdd/dmt_fz/fwd-inv-stc')

# 7 Brain Networks (Schaefer Atlas)
NETWORKS = {
    'FPN': {'name': 'Frontoparietal', 'color': '#e74c3c'},
    'DMN': {'name': 'Default Mode', 'color': '#3498db'},
    'DAN': {'name': 'Dorsal Attention', 'color': '#2ecc71'},
    'LN': {'name': 'Limbic', 'color': '#f39c12'},
    'SVA': {'name': 'Salience/Ventral', 'color': '#9b59b6'},
    'SMN': {'name': 'Somatomotor', 'color': '#1abc9c'},
    'VN': {'name': 'Visual', 'color': '#e91e63'},
}

BAND_COLORS = {
    'Delta': '#ff6b6b',
    'Theta': '#4ecdc4', 
    'Alpha': '#00d4aa',
    'Beta': '#f59e0b',
    'Gamma': '#a78bfa'
}


class Step5Visualizer(BaseVisualizer):
    """Visualizer for Step 5: Kuramoto Order Parameter."""
    
    step_number = 5
    step_name = "Coherencia Global"
    step_description = "Parámetro de orden Kuramoto R(t)"
    file_patterns = ["order-*.pkl", "*/order-*.pkl", "order_all*.pkl", "kuramoto*.pkl"]
    supported_backends = ['plotly']
    
    def __init__(self, get_run_dir):
        super().__init__(get_run_dir)
        self._selected_condition = 'DMT'
        self._selected_band = 'Alpha'
        self._selected_epoch = 0
        self._kuramoto_data = None
        self._network_data = None
    
    def get_controls(self) -> Dict[str, Any]:
        return {
            'condition': ['DMT', 'EC', 'EO'],
            'band': ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'],
            'epoch': 'number'
        }
    
    def _render_controls(self) -> None:
        """Render controls: condition, band, epoch selectors."""
        with ui.row().classes('items-center gap-3 mb-2 flex-wrap'):
            # Condition selector
            ui.select(
                ['DMT', 'EC', 'EO'],
                value=self._selected_condition,
                label='Condición'
            ).props('dense dark').classes('w-24').on(
                'update:model-value',
                lambda e: self._on_condition_change(e.args)
            )
            
            # Band selector
            ui.select(
                ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'],
                value=self._selected_band,
                label='Banda'
            ).props('dense dark').classes('w-24').on(
                'update:model-value',
                lambda e: self._on_band_change(e.args)
            )
            
            # Epoch selector
            ui.number(
                value=self._selected_epoch, min=0, max=200, label='Época'
            ).props('dense').classes('w-20').on(
                'update:model-value',
                lambda e: self._on_epoch_change(e.args)
            )
    
    def _on_condition_change(self, value):
        self._selected_condition = extract_event_value(value)
        if self._container:
            self.render(self._container)
    
    def _on_band_change(self, value):
        self._selected_band = extract_event_value(value)
        if self._container:
            self.render(self._container)
    
    def _on_epoch_change(self, value):
        val = extract_event_value(value)
        self._selected_epoch = int(val) if val else 0
        if self._container:
            self.render(self._container)
    
    def _load_kuramoto_data(self) -> bool:
        """Load global Kuramoto data from pre-computed files."""
        try:
            import pickle
            
            # Try kuramoto_all.pkl first (R(t) timeseries)
            kuramoto_file = KURAMOTO_DATA_PATH / 'kuramoto_all.pkl'
            if kuramoto_file.exists():
                with open(kuramoto_file, 'rb') as f:
                    self._kuramoto_data = pickle.load(f)
            
            # Load network-level Kuramoto data
            network_file = KURAMOTO_DATA_PATH / 'r_kuramoto_nets_all_mean.pkl'
            if network_file.exists():
                with open(network_file, 'rb') as f:
                    self._network_data = pickle.load(f)
            
            return self._kuramoto_data is not None or self._network_data is not None
        except Exception as e:
            print(f"[Step5] Error loading Kuramoto data: {e}")
            return False
    
    def _render_visualization(self) -> None:
        """Render Kuramoto visualization."""
        # Load data if not already loaded
        if self._kuramoto_data is None and self._network_data is None:
            if not self._load_kuramoto_data():
                self._render_no_data_with_debug()
                return
        
        # Check if selected condition has data
        has_kuramoto = (
            self._kuramoto_data and 
            self._selected_condition in self._kuramoto_data
        )
        has_network = (
            self._network_data and 
            self._selected_condition in self._network_data
        )
        
        if not has_kuramoto and not has_network:
            ui.label(f'No hay datos Kuramoto para {self._selected_condition}').style(
                f'color: {THEME_WARN};'
            )
            return
        
        # Two-column layout
        with ui.row().classes('w-full gap-3'):
            # Left column: Oscillators + Band comparison
            with ui.column().classes('gap-2').style('flex: 1; min-width: 300px;'):
                if has_kuramoto:
                    self._render_oscillator_plot()
                    self._render_band_comparison()
            
            # Right column: Timeline + Network comparison
            with ui.column().classes('gap-2').style('flex: 1; min-width: 300px;'):
                if has_kuramoto:
                    self._render_r_timeline()
                if has_network:
                    self._render_network_comparison()
    
    def _render_no_data_with_debug(self) -> None:
        """Show debug info when no data found."""
        with ui.column().classes('w-full gap-2'):
            ui.label("No hay datos de Kuramoto").style(f'color: {THEME_WARN};')
            
            ui.label(f"Buscando en: {KURAMOTO_DATA_PATH}").style(
                f'color: {THEME_TEXT_DIM}; font-size: 0.7rem;'
            )
            
            # Check what files exist
            if KURAMOTO_DATA_PATH.exists():
                pkl_files = list(KURAMOTO_DATA_PATH.glob('*.pkl'))[:5]
                ui.label(f"Archivos .pkl encontrados: {len(list(KURAMOTO_DATA_PATH.glob('*.pkl')))}").style(
                    f'color: {THEME_TEXT_DIM}; font-size: 0.7rem;'
                )
                for f in pkl_files:
                    ui.label(f"  • {f.name}").style(
                        f'color: {THEME_TEXT_DIM}; font-size: 0.65rem;'
                    )
            else:
                ui.label(f"⚠️ Directorio no existe").style(f'color: {THEME_WARN}; font-size: 0.7rem;')
    
    def _render_oscillator_plot(self) -> None:
        """Render Kuramoto oscillators as polar scatter (like Wikipedia diagram)."""
        try:
            cond_data = self._kuramoto_data[self._selected_condition]
            band_data = cond_data.get(self._selected_band, [])
            
            if not band_data or len(band_data) == 0:
                return
            
            # Get R(t) for selected epoch
            epoch_idx = min(self._selected_epoch, len(band_data) - 1)
            r_values = band_data[epoch_idx]
            
            if not isinstance(r_values, np.ndarray) or len(r_values) == 0:
                return
            
            # Current R value (mean)
            r_mean = np.mean(r_values)
            
            # Generate phase angles for oscillators (simulated from R)
            n_oscillators = 20  # Number of oscillators to show
            
            # Phases distributed around mean with spread based on R
            # Higher R = tighter clustering
            spread = np.pi * (1 - r_mean)  # Less spread when R is high
            mean_phase = 0  # Reference phase
            phases = np.random.uniform(mean_phase - spread, mean_phase + spread, n_oscillators)
            
            # Create polar plot
            fig = go.Figure()
            
            # Oscillator points on unit circle
            fig.add_trace(go.Scatterpolar(
                r=[1] * n_oscillators,
                theta=np.degrees(phases),
                mode='markers',
                marker=dict(
                    size=12,
                    color=BAND_COLORS.get(self._selected_band, '#888'),
                    opacity=0.8,
                    line=dict(width=1, color='white')
                ),
                name='Oscillators',
                hovertemplate='Phase: %{theta:.1f}°<extra></extra>'
            ))
            
            # Mean phase vector (order parameter)
            mean_phase_deg = np.degrees(np.angle(np.mean(np.exp(1j * phases))))
            fig.add_trace(go.Scatterpolar(
                r=[0, r_mean],
                theta=[mean_phase_deg, mean_phase_deg],
                mode='lines+markers',
                line=dict(width=3, color='white'),
                marker=dict(size=[0, 10], color='yellow'),
                name=f'R = {r_mean:.2f}',
                hovertemplate=f'R = {r_mean:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                template='plotly_dark',
                height=220,
                margin=dict(l=40, r=40, t=40, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                title=dict(
                    text=f'Osciladores Kuramoto - {self._selected_band}',
                    font=dict(size=11, color=THEME_PRIMARY)
                ),
                polar=dict(
                    radialaxis=dict(
                        visible=True, 
                        range=[0, 1.2],
                        showticklabels=False,
                        gridcolor='rgba(100,100,100,0.3)'
                    ),
                    angularaxis=dict(
                        tickfont=dict(size=8, color='#888'),
                        gridcolor='rgba(100,100,100,0.3)'
                    ),
                    bgcolor='rgba(0,0,0,0)'
                ),
                showlegend=False
            )
            
            ui.plotly(fig).classes('w-full')
            
            # R value indicator
            with ui.row().classes('justify-center gap-2'):
                ui.label(f'R = {r_mean:.3f}').style(
                    f'color: {BAND_COLORS.get(self._selected_band, THEME_PRIMARY)}; '
                    f'font-size: 1.2rem; font-weight: bold;'
                )
                coherence_text = 'Alta' if r_mean > 0.7 else 'Media' if r_mean > 0.4 else 'Baja'
                ui.label(f'({coherence_text} coherencia)').style(
                    f'color: {THEME_TEXT_DIM}; font-size: 0.8rem;'
                )
            
        except Exception as e:
            ui.label(f'Error oscillators: {e}').style(f'color: {THEME_WARN}; font-size: 0.7rem;')
    
    def _render_r_timeline(self) -> None:
        """Render R(t) timeline for selected band/epoch."""
        try:
            cond_data = self._kuramoto_data[self._selected_condition]
            band_data = cond_data.get(self._selected_band, [])
            
            if not band_data or len(band_data) == 0:
                return
            
            epoch_idx = min(self._selected_epoch, len(band_data) - 1)
            r_values = band_data[epoch_idx]
            
            if not isinstance(r_values, np.ndarray) or len(r_values) == 0:
                return
            
            time = np.arange(len(r_values))
            r_mean = np.mean(r_values)
            
            fig = go.Figure()
            
            # R(t) line with fill
            fig.add_trace(go.Scatter(
                x=time,
                y=r_values,
                mode='lines',
                fill='tozeroy',
                fillcolor=f'rgba(0, 212, 170, 0.2)',
                line=dict(color=THEME_PRIMARY, width=1.5),
                name='R(t)',
                hovertemplate='t=%{x}<br>R=%{y:.3f}<extra></extra>'
            ))
            
            # Mean line
            fig.add_hline(
                y=r_mean, 
                line_dash="dash", 
                line_color="#f59e0b",
                annotation_text=f"Mean: {r_mean:.3f}",
                annotation_position="right",
                annotation_font=dict(size=9, color='#f59e0b')
            )
            
            fig.update_layout(
                template='plotly_dark',
                height=180,
                margin=dict(l=40, r=60, t=30, b=30),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                title=dict(
                    text=f'R(t) - {self._selected_band} epoch{epoch_idx}',
                    font=dict(size=10, color=THEME_PRIMARY)
                ),
                xaxis=dict(
                    title=dict(text='Tiempo', font=dict(size=9, color='#888')),
                    tickfont=dict(size=8, color='#888'),
                    gridcolor='rgba(100,100,100,0.2)'
                ),
                yaxis=dict(
                    title=dict(text='R(t)', font=dict(size=9, color='#888')),
                    tickfont=dict(size=8, color='#888'),
                    range=[0, 1],
                    gridcolor='rgba(100,100,100,0.2)'
                ),
                showlegend=False
            )
            
            ui.plotly(fig).classes('w-full')
            
        except Exception as e:
            ui.label(f'Error timeline: {e}').style(f'color: {THEME_WARN}; font-size: 0.7rem;')
    
    def _render_band_comparison(self) -> None:
        """Render mean R comparison across bands."""
        try:
            cond_data = self._kuramoto_data[self._selected_condition]
            
            bands = []
            means = []
            colors = []
            
            for band in ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']:
                band_data = cond_data.get(band, [])
                if band_data and len(band_data) > 0:
                    # Compute mean R across all epochs
                    all_means = []
                    for r_values in band_data:
                        if isinstance(r_values, np.ndarray) and len(r_values) > 0:
                            all_means.append(np.mean(r_values))
                    
                    if all_means:
                        bands.append(band[:2])  # δ θ α β γ abbreviations
                        means.append(np.mean(all_means))
                        colors.append(BAND_COLORS.get(band, '#888'))
            
            if not bands:
                return
            
            fig = go.Figure(data=go.Bar(
                x=bands,
                y=means,
                marker_color=colors,
                text=[f'{m:.2f}' for m in means],
                textposition='outside',
                textfont=dict(size=9, color='#888'),
                hovertemplate='%{x}: R=%{y:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                template='plotly_dark',
                height=150,
                margin=dict(l=30, r=20, t=25, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                title=dict(
                    text='R promedio por Banda',
                    font=dict(size=10, color=THEME_SECONDARY)
                ),
                xaxis=dict(tickfont=dict(size=10, color='#888')),
                yaxis=dict(
                    range=[0, 1],
                    tickfont=dict(size=8, color='#888'),
                    gridcolor='rgba(100,100,100,0.2)'
                ),
                showlegend=False,
                bargap=0.3
            )
            
            ui.plotly(fig).classes('w-full')
            
        except Exception as e:
            ui.label(f'Error bands: {e}').style(f'color: {THEME_WARN}; font-size: 0.7rem;')
    
    def _render_network_comparison(self) -> None:
        """Render mean R comparison across 7 brain networks."""
        try:
            cond_data = self._network_data[self._selected_condition]
            band_data = cond_data.get(self._selected_band, {})
            
            # Get 'both' hemisphere data
            both_data = band_data.get('both', {})
            
            if not both_data:
                return
            
            networks = []
            means = []
            colors = []
            full_names = []
            
            for net_key, net_info in NETWORKS.items():
                # Network keys may have trailing space (e.g., 'LN ', 'VN ')
                net_data = both_data.get(net_key) or both_data.get(net_key.strip()) or both_data.get(net_key + ' ')
                
                if net_data:
                    # net_data is a dict of network-to-network correlations
                    # We want the mean across all
                    if isinstance(net_data, dict):
                        values = [v for v in net_data.values() if isinstance(v, (int, float, np.floating))]
                        if values:
                            networks.append(net_key.strip())
                            means.append(np.mean(values))
                            colors.append(net_info['color'])
                            full_names.append(net_info['name'])
            
            if not networks:
                ui.label('No network data available').style(f'color: {THEME_TEXT_DIM}; font-size: 0.7rem;')
                return
            
            fig = go.Figure(data=go.Bar(
                x=networks,
                y=means,
                marker_color=colors,
                text=[f'{m:.2f}' for m in means],
                textposition='outside',
                textfont=dict(size=9, color='#888'),
                customdata=full_names,
                hovertemplate='<b>%{customdata}</b><br>R=%{y:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                template='plotly_dark',
                height=180,
                margin=dict(l=30, r=20, t=30, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                title=dict(
                    text=f'R por Red Cerebral - {self._selected_band}',
                    font=dict(size=10, color=THEME_SECONDARY)
                ),
                xaxis=dict(tickfont=dict(size=9, color='#888')),
                yaxis=dict(
                    range=[0, max(means) * 1.3 if means else 1],
                    tickfont=dict(size=8, color='#888'),
                    gridcolor='rgba(100,100,100,0.2)'
                ),
                showlegend=False,
                bargap=0.2
            )
            
            ui.plotly(fig).classes('w-full')
            
        except Exception as e:
            ui.label(f'Error networks: {e}').style(f'color: {THEME_WARN}; font-size: 0.7rem;')
