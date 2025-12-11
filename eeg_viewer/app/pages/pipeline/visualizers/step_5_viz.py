"""
Step 5 Visualizer: Kuramoto Order (generate_order.py)

Visualizes global coherence:
- Order parameter timeline R(t)
- Mean order by band comparison
"""
from typing import Dict, Any, List
import numpy as np
from nicegui import ui

from config import THEME_PRIMARY, THEME_SECONDARY, THEME_TEXT_DIM, THEME_WARN
from .base import BaseVisualizer, extract_event_value


class Step5Visualizer(BaseVisualizer):
    """Visualizer for Step 5: Kuramoto Order."""
    
    step_number = 5
    step_name = "Coherencia Global"
    step_description = "Parámetro de orden Kuramoto R(t)"
    # Search in root and subdirectories - order-*.pkl NOT order_all
    file_patterns = ["order-*.pkl", "*/order-*.pkl", "**/order-*.pkl"]
    supported_backends = ['matplotlib', 'plotly']
    
    def __init__(self, get_run_dir):
        super().__init__(get_run_dir)
        self._selected_band = 'Alpha'
        self._selected_epoch = 0
        self._selected_subject = None
        self._show_all_bands = False
    
    def get_controls(self) -> Dict[str, Any]:
        return {
            'band': ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'],
            'epoch': 'number',
            'subject': 'select'
        }
    
    def _render_controls(self) -> None:
        with ui.row().classes('items-center gap-3 mb-2 flex-wrap'):
            subjects = self._get_subjects()
            if subjects:
                ui.select(
                    subjects, value=subjects[0] if subjects else None,
                    label='Sujeto'
                ).props('dense dark').classes('w-28').on(
                    'update:model-value', 
                    lambda e: self._on_subject_change(e.args)
                )
            
            ui.select(
                ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'],
                value=self._selected_band, label='Banda'
            ).props('dense dark').classes('w-24').on(
                'update:model-value',
                lambda e: self._on_band_change(e.args)
            )
            
            ui.number(
                value=self._selected_epoch, min=0, max=200, label='Época'
            ).props('dense').classes('w-20').on(
                'update:model-value',
                lambda e: self._on_epoch_change(e.args)
            )
            
            ui.button(
                '📊 All Bands', 
                on_click=self._toggle_all_bands
            ).props('dense flat').style(
                f'color: {"#00d4aa" if self._show_all_bands else THEME_TEXT_DIM};'
            )
    
    def _get_subjects(self) -> List[str]:
        files = self.find_data_files()
        subjects = []
        for f in files:
            name = f.stem
            if 'order-' in name:
                subj = name.replace('order-', '')
                subjects.append(subj)
        return sorted(set(subjects))
    
    def _on_subject_change(self, value):
        self._selected_subject = extract_event_value(value)
        self._load_subject_data()
        self.render(self._container)
    
    def _on_band_change(self, value):
        self._selected_band = extract_event_value(value)
        self.render(self._container)
    
    def _on_epoch_change(self, value):
        val = extract_event_value(value)
        self._selected_epoch = int(val) if val else 0
        self.render(self._container)
    
    def _toggle_all_bands(self):
        self._show_all_bands = not self._show_all_bands
        self.render(self._container)
    
    def _load_subject_data(self) -> bool:
        if not self._selected_subject:
            return False
        files = self.find_data_files()
        for f in files:
            if self._selected_subject in f.name:
                return self.load_data(f)
        return False
    
    def _render_visualization(self) -> None:
        if not self._data:
            files = self.find_data_files()
            if files:
                self.load_data(files[0])
        
        if not self._data:
            self._render_no_data_message("No hay datos de orden Kuramoto")
            return
        
        with ui.row().classes('w-full gap-3'):
            with ui.column().classes('flex-1'):
                if self._show_all_bands:
                    self._render_all_bands_timeline()
                else:
                    self._render_order_timeline()
            
            with ui.column().style('width: 180px;'):
                self._render_order_stats()
                self._render_band_comparison()
    
    def _render_order_timeline(self) -> None:
        """Render R(t) timeline for selected band."""
        try:
            order = self._data.get('order_stc') or self._data.get('order_eeg', {})
            band_data = order.get(self._selected_band, {})
            
            epoch_key = f'epoch{self._selected_epoch}'
            if epoch_key not in band_data:
                if band_data:
                    epoch_key = list(band_data.keys())[0]
                else:
                    ui.label(f'No data for {self._selected_band}').style(f'color: {THEME_TEXT_DIM};')
                    return
            
            r_values = band_data[epoch_key]
            
            fig, ax = self.create_matplotlib_figure(figsize=(7, 3))
            
            time = np.arange(len(r_values))
            ax.plot(time, r_values, color=THEME_PRIMARY, linewidth=1)
            ax.fill_between(time, r_values, alpha=0.3, color=THEME_PRIMARY)
            ax.axhline(y=np.mean(r_values), color='#f59e0b', linestyle='--', 
                      linewidth=1, label=f'Mean: {np.mean(r_values):.3f}')
            
            ax.set_xlabel('Tiempo', color='#888888')
            ax.set_ylabel('R(t)', color='#888888')
            ax.set_ylim(0, 1)
            ax.set_title(f'Kuramoto {self._selected_band} - {epoch_key}', 
                        color=THEME_PRIMARY, fontsize=10)
            ax.legend(facecolor='#0a0a0a', edgecolor='#333333', labelcolor='#888888')
            
            self.show_matplotlib(fig)
            
        except Exception as e:
            ui.label(f'Error: {e}').style(f'color: {THEME_WARN}; font-size: 0.75rem;')
    
    def _render_all_bands_timeline(self) -> None:
        """Render R(t) timeline for all bands."""
        try:
            order = self._data.get('order_stc') or self._data.get('order_eeg', {})
            
            fig, ax = self.create_matplotlib_figure(figsize=(7, 4))
            
            colors = {'Delta': '#ff6b6b', 'Theta': '#4ecdc4', 'Alpha': '#00d4aa',
                     'Beta': '#f59e0b', 'Gamma': '#a78bfa'}
            
            for band, band_data in order.items():
                epoch_key = f'epoch{self._selected_epoch}'
                if epoch_key not in band_data:
                    if band_data:
                        epoch_key = list(band_data.keys())[0]
                    else:
                        continue
                
                r_values = band_data[epoch_key]
                time = np.arange(len(r_values))
                ax.plot(time, r_values, color=colors.get(band, '#888888'), 
                       linewidth=1, label=band, alpha=0.8)
            
            ax.set_xlabel('Tiempo', color='#888888')
            ax.set_ylabel('R(t)', color='#888888')
            ax.set_ylim(0, 1)
            ax.set_title('Kuramoto - Todas las Bandas', color=THEME_PRIMARY, fontsize=10)
            ax.legend(facecolor='#0a0a0a', edgecolor='#333333', labelcolor='#888888', 
                     fontsize=8, loc='upper right')
            
            self.show_matplotlib(fig)
            
        except Exception as e:
            ui.label(f'Error: {e}').style(f'color: {THEME_WARN}; font-size: 0.75rem;')
    
    def _render_order_stats(self) -> None:
        """Render order statistics."""
        try:
            order = self._data.get('order_stc') or self._data.get('order_eeg', {})
            band_data = order.get(self._selected_band, {})
            
            if not band_data:
                return
            
            # Aggregate across epochs
            all_means = []
            for epoch_key, r_values in band_data.items():
                all_means.append(np.mean(r_values))
            
            stats = {
                'R Mean': f'{np.mean(all_means):.3f}',
                'R Std': f'{np.std(all_means):.3f}',
                'Épocas': len(band_data),
            }
            
            self._render_stats_card(stats, "📊 Kuramoto Stats")
            
        except:
            pass
    
    def _render_band_comparison(self) -> None:
        """Render band comparison bar chart."""
        try:
            order = self._data.get('order_stc') or self._data.get('order_eeg', {})
            
            bands = []
            means = []
            colors = []
            color_map = {'Delta': '#ff6b6b', 'Theta': '#4ecdc4', 'Alpha': '#00d4aa',
                        'Beta': '#f59e0b', 'Gamma': '#a78bfa'}
            
            for band, band_data in order.items():
                all_means = [np.mean(r) for r in band_data.values()]
                if all_means:
                    bands.append(band)
                    means.append(np.mean(all_means))
                    colors.append(color_map.get(band, '#888888'))
            
            if not bands:
                return
            
            fig, ax = self.create_matplotlib_figure(figsize=(3, 2.5))
            bars = ax.bar(range(len(bands)), means, color=colors)
            ax.set_xticks(range(len(bands)))
            ax.set_xticklabels([b[:2] for b in bands], fontsize=7)
            ax.set_ylabel('R mean', color='#888888', fontsize=8)
            ax.set_ylim(0, 1)
            ax.tick_params(labelsize=7)
            
            self.show_matplotlib(fig)
            
        except:
            pass

