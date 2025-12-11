"""
Step 4 Visualizer: Synchronization (calculate_syncro.py)

Visualizes PLV synchronization matrices:
- Heatmap of connectivity
- Network connectivity diagram
- Histogram of PLV values
"""
from typing import Dict, Any, List, Optional
import numpy as np
from nicegui import ui

from config import THEME_PRIMARY, THEME_SECONDARY, THEME_TEXT_DIM, THEME_WARN
from .base import BaseVisualizer, extract_event_value


class Step4Visualizer(BaseVisualizer):
    """Visualizer for Step 4: Synchronization."""
    
    step_number = 4
    step_name = "Sincronización"
    step_description = "Matrices PLV de conectividad"
    # Search in root and all subdirectories (DMT/, EC/, EO/)
    file_patterns = ["syncro-*.pkl", "*/syncro-*.pkl", "**/syncro-*.pkl"]
    supported_backends = ['matplotlib', 'plotly']
    
    def __init__(self, get_run_dir):
        super().__init__(get_run_dir)
        self._selected_band = 'Alpha'
        self._selected_epoch = 0
        self._selected_subject = None
    
    def get_controls(self) -> Dict[str, Any]:
        return {
            'band': ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'],
            'epoch': 'number',
            'subject': 'select'
        }
    
    def _render_controls(self) -> None:
        with ui.row().classes('items-center gap-3 mb-2 flex-wrap'):
            # Subject selector
            subjects = self._get_subjects()
            if subjects:
                ui.select(
                    subjects, value=subjects[0] if subjects else None,
                    label='Sujeto'
                ).props('dense dark').classes('w-28').on(
                    'update:model-value', 
                    lambda e: self._on_subject_change(e.args)
                )
            
            # Band selector
            ui.select(
                ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'],
                value=self._selected_band, label='Banda'
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
            
            # Backend toggle
            with ui.row().classes('gap-1'):
                ui.button('📊 Static', on_click=lambda: self._set_backend('matplotlib')).props(
                    'dense flat size=sm'
                ).style(f'color: {"#00d4aa" if self._backend == "matplotlib" else THEME_TEXT_DIM};')
                ui.button('🔄 Interactive', on_click=lambda: self._set_backend('plotly')).props(
                    'dense flat size=sm'
                ).style(f'color: {"#00d4aa" if self._backend == "plotly" else THEME_TEXT_DIM};')
    
    def _get_subjects(self) -> List[str]:
        files = self.find_data_files()
        subjects = []
        for f in files:
            name = f.stem
            if 'syncro-' in name:
                subj = name.replace('syncro-', '')
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
    
    def _set_backend(self, backend):
        self._backend = backend
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
            self._render_no_data_message("No hay matrices de sincronización")
            return
        
        with ui.row().classes('w-full gap-3'):
            # Main heatmap
            with ui.column().classes('flex-1'):
                self._render_syncro_heatmap()
            
            # Stats panel
            with ui.column().style('width: 180px;'):
                self._render_syncro_stats()
                self._render_histogram()
    
    def _render_syncro_heatmap(self) -> None:
        """Render synchronization matrix heatmap."""
        try:
            syncros = self._data.get('syncros_stc') or self._data.get('syncros_eeg', {})
            band_data = syncros.get(self._selected_band)
            
            if band_data is None:
                ui.label(f'No data for {self._selected_band}').style(f'color: {THEME_TEXT_DIM};')
                return
            
            # Handle different data formats
            matrix = None
            epoch_key = f'epoch{self._selected_epoch}'
            
            if isinstance(band_data, dict):
                # Format: {'epoch0': matrix, 'epoch1': matrix, ...}
                if epoch_key in band_data:
                    matrix = band_data[epoch_key]
                elif band_data:
                    epoch_key = list(band_data.keys())[0]
                    matrix = band_data[epoch_key]
            elif isinstance(band_data, list):
                # Format: list of epochs or nested list
                if len(band_data) > 0:
                    first_elem = band_data[0]
                    if isinstance(first_elem, np.ndarray) and first_elem.ndim == 2:
                        # Simple list of matrices: [matrix1, matrix2, ...]
                        idx = min(self._selected_epoch, len(band_data) - 1)
                        matrix = band_data[idx]
                        epoch_key = f'epoch{idx}'
                    elif isinstance(first_elem, list):
                        # Nested list: [[matrices...], [matrices...], ...]
                        # Flatten or take first subject's epochs
                        flat_matrices = first_elem  # Take first subject
                        if flat_matrices and len(flat_matrices) > 0:
                            idx = min(self._selected_epoch, len(flat_matrices) - 1)
                            matrix = flat_matrices[idx]
                            if isinstance(matrix, np.ndarray):
                                epoch_key = f'epoch{idx}'
                            else:
                                matrix = None
            elif isinstance(band_data, np.ndarray):
                # Direct numpy array
                if band_data.ndim == 3:
                    idx = min(self._selected_epoch, band_data.shape[0] - 1)
                    matrix = band_data[idx]
                    epoch_key = f'epoch{idx}'
                elif band_data.ndim == 2:
                    matrix = band_data
                    epoch_key = 'single'
            
            if matrix is None or not isinstance(matrix, np.ndarray):
                ui.label(f'Could not parse data for {self._selected_band}').style(f'color: {THEME_TEXT_DIM};')
                return
            
            if self._backend == 'matplotlib':
                # Compact heatmap
                fig, ax = self.create_matplotlib_figure(figsize=(4.5, 4))
                
                im = ax.imshow(matrix, cmap='viridis', vmin=0, vmax=1, aspect='equal')
                ax.set_xlabel('Parcela', color='#888888', fontsize=8)
                ax.set_ylabel('Parcela', color='#888888', fontsize=8)
                ax.set_title(f'PLV {self._selected_band} - {epoch_key}', 
                           color=THEME_PRIMARY, fontsize=9)
                ax.tick_params(labelsize=7)
                
                cbar = fig.colorbar(im, ax=ax, shrink=0.8)
                cbar.ax.yaxis.set_tick_params(color='#888888', labelsize=7)
                cbar.ax.set_ylabel('PLV', color='#888888', fontsize=8)
                
                self.show_matplotlib(fig)
            else:
                import plotly.graph_objects as go
                fig = go.Figure(data=go.Heatmap(
                    z=matrix,
                    colorscale='Viridis',
                    zmin=0, zmax=1,
                    colorbar=dict(title='PLV', len=0.8)
                ))
                fig.update_layout(
                    template='plotly_dark',
                    title=dict(text=f'PLV {self._selected_band} - {epoch_key}', font=dict(size=12)),
                    xaxis_title='Parcela',
                    yaxis_title='Parcela',
                    height=300,
                    margin=dict(l=40, r=40, t=40, b=40),
                    yaxis=dict(scaleanchor='x', scaleratio=1)
                )
                ui.plotly(fig).classes('w-full').style('max-height: 320px;')
                
        except Exception as e:
            ui.label(f'Error: {e}').style(f'color: {THEME_WARN}; font-size: 0.75rem;')
    
    def _render_syncro_stats(self) -> None:
        """Render synchronization statistics."""
        try:
            syncros = self._data.get('syncros_stc') or self._data.get('syncros_eeg', {})
            band_data = syncros.get(self._selected_band)
            
            if not band_data:
                return
            
            # Get matrix using same logic as heatmap
            matrix = self._get_matrix_from_band_data(band_data)
            
            if matrix is None:
                return
            
            # Get upper triangle (excluding diagonal)
            upper = matrix[np.triu_indices_from(matrix, k=1)]
            
            # Count epochs
            n_epochs = self._count_epochs(band_data)
            
            stats = {
                'Mean PLV': f'{np.mean(upper):.3f}',
                'Std': f'{np.std(upper):.3f}',
                'Max': f'{np.max(upper):.3f}',
                'Min': f'{np.min(upper):.3f}',
                'Size': f'{matrix.shape[0]}x{matrix.shape[1]}',
                'Épocas': n_epochs,
            }
            
            self._render_stats_card(stats, "📊 PLV Stats")
            
        except Exception as e:
            ui.label(f'Stats error: {e}').style(f'color: {THEME_TEXT_DIM}; font-size: 0.7rem;')
    
    def _get_matrix_from_band_data(self, band_data) -> Optional[np.ndarray]:
        """Extract a matrix from band data in any format."""
        if isinstance(band_data, dict):
            epoch_key = f'epoch{self._selected_epoch}'
            if epoch_key in band_data:
                return band_data[epoch_key]
            elif band_data:
                return band_data[list(band_data.keys())[0]]
        elif isinstance(band_data, list) and len(band_data) > 0:
            first_elem = band_data[0]
            if isinstance(first_elem, np.ndarray) and first_elem.ndim == 2:
                idx = min(self._selected_epoch, len(band_data) - 1)
                return band_data[idx]
            elif isinstance(first_elem, list) and first_elem:
                idx = min(self._selected_epoch, len(first_elem) - 1)
                return first_elem[idx] if isinstance(first_elem[idx], np.ndarray) else None
        elif isinstance(band_data, np.ndarray):
            if band_data.ndim == 3:
                idx = min(self._selected_epoch, band_data.shape[0] - 1)
                return band_data[idx]
            elif band_data.ndim == 2:
                return band_data
        return None
    
    def _count_epochs(self, band_data) -> int:
        """Count number of epochs in band data."""
        if isinstance(band_data, dict):
            return len(band_data)
        elif isinstance(band_data, list):
            if band_data and isinstance(band_data[0], list):
                return len(band_data[0])  # Nested: first subject's epochs
            return len(band_data)
        elif isinstance(band_data, np.ndarray) and band_data.ndim == 3:
            return band_data.shape[0]
        return 1
    
    def _render_histogram(self) -> None:
        """Render PLV distribution histogram."""
        try:
            syncros = self._data.get('syncros_stc') or self._data.get('syncros_eeg', {})
            band_data = syncros.get(self._selected_band)
            
            if not band_data:
                return
            
            matrix = self._get_matrix_from_band_data(band_data)
            if matrix is None:
                return
            
            upper = matrix[np.triu_indices_from(matrix, k=1)]
            
            fig, ax = self.create_matplotlib_figure(figsize=(3, 2))
            ax.hist(upper, bins=30, color=THEME_PRIMARY, alpha=0.7, edgecolor='none')
            ax.set_xlabel('PLV', color='#888888', fontsize=8)
            ax.set_ylabel('Count', color='#888888', fontsize=8)
            ax.tick_params(labelsize=7)
            
            self.show_matplotlib(fig)
            
        except:
            pass

