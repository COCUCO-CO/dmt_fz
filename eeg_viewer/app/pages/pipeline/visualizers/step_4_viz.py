"""
Step 4 Visualizer: Synchronization (calculate_syncro.py)

Visualizes PLV synchronization matrices:
- Select subject and multiple conditions
- Show matrices side by side (up to 3)
- Stats and histogram below each matrix
"""
from typing import Dict, Any, List, Optional
import numpy as np
from nicegui import ui
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import THEME_PRIMARY, THEME_SECONDARY, THEME_TEXT_DIM, THEME_WARN
from .base import BaseVisualizer, extract_event_value


class Step4Visualizer(BaseVisualizer):
    """Visualizer for Step 4: Synchronization."""
    
    step_number = 4
    step_name = "Sincronización"
    step_description = "Matrices PLV de conectividad"
    file_patterns = ["syncro-*.pkl", "*/syncro-*.pkl", "**/syncro-*.pkl"]
    supported_backends = ['matplotlib', 'plotly']
    
    def __init__(self, get_run_dir):
        super().__init__(get_run_dir)
        self._selected_subject = None
        self._selected_conditions = ['DMT']  # Default to DMT
        self._selected_epoch = 0
        self._available_conditions = []
        self._loaded_data_by_condition = {}  # Cache data per condition
    
    def get_controls(self) -> Dict[str, Any]:
        return {
            'subject': 'select',
            'conditions': 'checkboxes',
            'epoch': 'number'
        }
    
    def _render_controls(self) -> None:
        """Render controls: subject selector + condition checkboxes."""
        # Get available subjects and conditions
        subjects, conditions_by_subject = self._get_subjects_and_conditions()
        self._available_conditions = list(set(c for conds in conditions_by_subject.values() for c in conds))
        
        with ui.row().classes('items-center gap-3 mb-2 flex-wrap'):
            # Subject selector
            if subjects:
                initial_subject = self._selected_subject or (subjects[0] if subjects else None)
                ui.select(
                    subjects, value=initial_subject, label='Sujeto'
                ).props('dense dark').classes('w-28').on(
                    'update:model-value', 
                    lambda e: self._on_subject_change(e.args)
                )
            
            # Condition checkboxes
            ui.label('Condiciones:').style(f'color: {THEME_TEXT_DIM}; font-size: 0.75rem;')
            
            cond_colors = {'DMT': '#00d4aa', 'EC': '#f59e0b', 'EO': '#a78bfa'}
            for cond in ['DMT', 'EC', 'EO']:
                is_available = cond in self._available_conditions
                is_checked = cond in self._selected_conditions
                
                cb = ui.checkbox(
                    cond, 
                    value=is_checked,
                    on_change=lambda e, c=cond: self._toggle_condition(c, e.value)
                ).props('dense')
                
                if is_available:
                    cb.style(f'color: {cond_colors.get(cond, THEME_TEXT_DIM)};')
                else:
                    cb.props('disable')
                    cb.style(f'color: {THEME_TEXT_DIM}; opacity: 0.5;')
            
            # Epoch selector
            ui.number(
                value=self._selected_epoch, min=0, max=200, label='Época'
            ).props('dense').classes('w-20').on(
                'update:model-value',
                lambda e: self._on_epoch_change(e.args)
            )
            
            # Backend toggle
            with ui.row().classes('gap-1 ml-2'):
                ui.button('Static', on_click=lambda: self._set_backend('matplotlib')).props(
                    'dense flat size=sm'
                ).style(f'color: {"#00d4aa" if self._backend == "matplotlib" else THEME_TEXT_DIM};')
                ui.button('Interactive', on_click=lambda: self._set_backend('plotly')).props(
                    'dense flat size=sm'
                ).style(f'color: {"#00d4aa" if self._backend == "plotly" else THEME_TEXT_DIM};')
    
    def _get_subjects_and_conditions(self) -> tuple:
        """Get available subjects and their conditions."""
        files = self.find_data_files()
        subjects = []
        conditions_by_subject = {}
        
        for f in files:
            name = f.stem
            if 'syncro-' in name:
                # Extract subject ID (e.g., "S01-DMT" -> "S01")
                subj_cond = name.replace('syncro-', '')
                
                if '-' in subj_cond:
                    parts = subj_cond.rsplit('-', 1)
                    subj = parts[0]
                    cond = parts[1] if len(parts) > 1 else 'Unknown'
                else:
                    subj = subj_cond
                    # Try to infer condition from parent folder
                    if f.parent.name in ['DMT', 'EC', 'EO']:
                        cond = f.parent.name
                    else:
                        cond = 'Unknown'
                
                if subj not in conditions_by_subject:
                    conditions_by_subject[subj] = []
                    subjects.append(subj)
                
                if cond not in conditions_by_subject[subj]:
                    conditions_by_subject[subj].append(cond)
        
        return sorted(set(subjects)), conditions_by_subject
    
    def _toggle_condition(self, condition: str, is_checked: bool) -> None:
        """Toggle a condition on/off."""
        if is_checked and condition not in self._selected_conditions:
            self._selected_conditions.append(condition)
        elif not is_checked and condition in self._selected_conditions:
            self._selected_conditions.remove(condition)
        self._selected_conditions.sort()
        self.render(self._container)
    
    def _on_subject_change(self, value):
        new_subject = extract_event_value(value)
        if new_subject != self._selected_subject:
            self._selected_subject = new_subject
            self._loaded_data_by_condition.clear()  # Clear cache on subject change
            self._data = None  # Also clear base data
            if self._container:
                self.render(self._container)
    
    def _on_epoch_change(self, value):
        val = extract_event_value(value)
        self._selected_epoch = int(val) if val else 0
        self.render(self._container)
    
    def _set_backend(self, backend):
        self._backend = backend
        self.render(self._container)
    
    def _load_condition_data(self, subject: str, condition: str) -> Optional[dict]:
        """Load data for a specific subject-condition pair."""
        cache_key = f"{subject}-{condition}"
        if cache_key in self._loaded_data_by_condition:
            return self._loaded_data_by_condition[cache_key]
        
        files = self.find_data_files()
        for f in files:
            # Check if file matches subject and condition
            is_match = (
                (subject in f.name and condition in f.name) or
                (subject in f.name and f.parent.name == condition)
            )
            if is_match:
                try:
                    import pickle
                    with open(f, 'rb') as handle:
                        data = pickle.load(handle)
                    self._loaded_data_by_condition[cache_key] = data
                    return data
                except:
                    pass
        return None
    
    def _render_visualization(self) -> None:
        """Render PLV matrices for selected conditions."""
        if not self._selected_conditions:
            ui.label('Seleccioná al menos una condición').style(f'color: {THEME_WARN};')
            return
        
        # Get subject (use first available if not selected)
        subjects, _ = self._get_subjects_and_conditions()
        if not self._selected_subject and subjects:
            self._selected_subject = subjects[0]
        
        if not self._selected_subject:
            self._render_no_data_message("No hay sujetos disponibles")
            return
        
        # Load data for each selected condition
        condition_data = {}
        for cond in self._selected_conditions:
            data = self._load_condition_data(self._selected_subject, cond)
            if data:
                condition_data[cond] = data
        
        if not condition_data:
            self._render_no_data_message(f"No hay datos para {self._selected_subject}")
            return
        
        # Render matrices side by side
        n_conditions = len(condition_data)
        
        with ui.row().classes('w-full gap-2 flex-wrap'):
            for cond, data in condition_data.items():
                # Each condition in a column
                col_width = f'{min(100 // n_conditions, 48)}%'
                with ui.column().style(f'flex: 1; min-width: 280px; max-width: {col_width};'):
                    self._render_single_condition(cond, data)
    
    def _render_single_condition(self, condition: str, data: dict) -> None:
        """Render matrix + stats + histogram for one condition."""
        cond_colors = {'DMT': '#00d4aa', 'EC': '#f59e0b', 'EO': '#a78bfa'}
        color = cond_colors.get(condition, '#888888')
        
        # Get matrix
        syncros = data.get('syncros_stc') or data.get('syncros_eeg', {})
        band = 'Alpha'  # Default band
        band_data = syncros.get(band)
        
        if band_data is None:
            ui.label(f'{condition}: No data').style(f'color: {color};')
            return
        
        matrix = self._get_matrix_from_band_data(band_data)
        if matrix is None:
            ui.label(f'{condition}: Invalid data').style(f'color: {color};')
            return
        
        # Header
        ui.label(f'{condition}').style(
            f'color: {color}; font-weight: bold; font-size: 0.85rem; margin-bottom: 4px;'
        )
        
        # Heatmap
        if self._backend == 'plotly':
            self._render_plotly_heatmap(matrix, condition, color)
        else:
            self._render_matplotlib_heatmap(matrix, condition, color)
        
        # Stats + Histogram row
        with ui.row().classes('w-full gap-2 mt-1'):
            self._render_compact_stats(matrix, color)
            self._render_compact_histogram(matrix, color)
    
    def _render_plotly_heatmap(self, matrix: np.ndarray, condition: str, color: str) -> None:
        """Render compact, high-quality Plotly heatmap."""
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            colorscale='Viridis',
            zmin=0, zmax=1,
            showscale=True,
            colorbar=dict(
                len=0.8, 
                thickness=12, 
                tickfont=dict(size=9, color='#cccccc'),  # Lighter color for visibility
                title=dict(text='PLV', font=dict(size=9, color='#cccccc')),
                tickcolor='#cccccc',
                outlinecolor='#333333',
            ),
            hovertemplate='<b>ROI %{x} ↔ ROI %{y}</b><br>PLV: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            template='plotly_dark',
            height=220,  # Reduced height for better fit
            margin=dict(l=35, r=45, t=30, b=30),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title=dict(
                text=f'PLV Alpha - {condition}', 
                font=dict(size=10, color=color, family='JetBrains Mono')
            ),
            xaxis=dict(
                title=dict(text='ROI', font=dict(size=9, color='#888888')),
                tickfont=dict(size=8, color='#888888'), 
                showgrid=False,
                dtick=25
            ),
            yaxis=dict(
                title=dict(text='ROI', font=dict(size=9, color='#888888')),
                tickfont=dict(size=8, color='#888888'), 
                showgrid=False, 
                scaleanchor='x',
                dtick=25
            ),
        )
        
        ui.plotly(fig).classes('w-full').style('max-height: 240px;')
    
    def _render_matplotlib_heatmap(self, matrix: np.ndarray, condition: str, color: str) -> None:
        """Render compact Matplotlib heatmap."""
        fig, ax = self.create_matplotlib_figure(figsize=(3, 2.8))
        
        im = ax.imshow(matrix, cmap='viridis', vmin=0, vmax=1, aspect='equal')
        ax.set_title(f'PLV Alpha - {condition}', color=color, fontsize=9)
        ax.tick_params(labelsize=6)
        
        cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
        cbar.ax.tick_params(labelsize=6)
        
        self.show_matplotlib(fig)
    
    def _render_compact_stats(self, matrix: np.ndarray, color: str) -> None:
        """Render compact stats card."""
        upper = matrix[np.triu_indices_from(matrix, k=1)]
        
        with ui.card().classes('p-2').style(
            f'background: {color}10; border: 1px solid {color}40; flex: 1;'
        ):
            ui.label('Stats').style(f'color: {color}; font-size: 0.7rem; font-weight: bold;')
            
            stats = [
                ('Mean', f'{np.mean(upper):.3f}'),
                ('Std', f'{np.std(upper):.3f}'),
                ('Max', f'{np.max(upper):.3f}'),
            ]
            
            for label, value in stats:
                with ui.row().classes('justify-between'):
                    ui.label(label).style(f'color: {THEME_TEXT_DIM}; font-size: 0.65rem;')
                    ui.label(value).style(f'color: {color}; font-size: 0.65rem;')
    
    def _render_compact_histogram(self, matrix: np.ndarray, color: str) -> None:
        """Render compact histogram."""
        upper = matrix[np.triu_indices_from(matrix, k=1)]
        
        fig, ax = self.create_matplotlib_figure(figsize=(2, 1.5))
        ax.hist(upper, bins=20, color=color, alpha=0.7, edgecolor='none')
        ax.set_xlabel('PLV', color='#666', fontsize=6)
        ax.set_ylabel('', fontsize=6)
        ax.tick_params(labelsize=5)
        ax.set_xlim(0, 1)
        
        self.show_matplotlib(fig)
    
    def _get_matrix_from_band_data(self, band_data) -> Optional[np.ndarray]:
        """Extract a matrix from band data in any format."""
        epoch_idx = self._selected_epoch
        
        if isinstance(band_data, dict):
            epoch_key = f'epoch{epoch_idx}'
            if epoch_key in band_data:
                return band_data[epoch_key]
            elif band_data:
                return band_data[list(band_data.keys())[0]]
        elif isinstance(band_data, list) and len(band_data) > 0:
            first_elem = band_data[0]
            if isinstance(first_elem, np.ndarray) and first_elem.ndim == 2:
                idx = min(epoch_idx, len(band_data) - 1)
                return band_data[idx]
            elif isinstance(first_elem, list) and first_elem:
                idx = min(epoch_idx, len(first_elem) - 1)
                return first_elem[idx] if isinstance(first_elem[idx], np.ndarray) else None
        elif isinstance(band_data, np.ndarray):
            if band_data.ndim == 3:
                idx = min(epoch_idx, band_data.shape[0] - 1)
                return band_data[idx]
            elif band_data.ndim == 2:
                return band_data
        return None
