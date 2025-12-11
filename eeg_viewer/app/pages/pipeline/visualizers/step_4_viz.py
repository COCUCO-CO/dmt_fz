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
import io
import base64

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
        self._is_loading = False
        self._cached_files = None  # Cache file list
        self._cached_subjects = None  # Cache subjects
    
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
    
    def _get_subjects_and_conditions(self) -> tuple:
        """Get available subjects and their conditions - CACHED."""
        # Return cached result if available
        if self._cached_subjects is not None:
            return self._cached_subjects
        
        # Only scan files once
        if self._cached_files is None:
            self._cached_files = self.find_data_files()
        
        subjects = []
        conditions_by_subject = {}
        
        for f in self._cached_files:
            name = f.stem
            if 'syncro-' in name:
                subj_cond = name.replace('syncro-', '')
                
                if '-' in subj_cond:
                    parts = subj_cond.rsplit('-', 1)
                    subj = parts[0]
                    cond = parts[1] if len(parts) > 1 else 'Unknown'
                else:
                    subj = subj_cond
                    if f.parent.name in ['DMT', 'EC', 'EO']:
                        cond = f.parent.name
                    else:
                        cond = 'Unknown'
                
                if subj not in conditions_by_subject:
                    conditions_by_subject[subj] = []
                    subjects.append(subj)
                
                if cond not in conditions_by_subject[subj]:
                    conditions_by_subject[subj].append(cond)
        
        # Cache the result
        self._cached_subjects = (sorted(set(subjects)), conditions_by_subject)
        return self._cached_subjects
    
    def _toggle_condition(self, condition: str, is_checked: bool) -> None:
        """Toggle a condition on/off."""
        if is_checked and condition not in self._selected_conditions:
            self._selected_conditions.append(condition)
        elif not is_checked and condition in self._selected_conditions:
            self._selected_conditions.remove(condition)
        self._selected_conditions.sort()
        if self._container:
            self.render(self._container)
    
    def _on_subject_change(self, value):
        """Handle subject change - render directly without timer."""
        new_subject = extract_event_value(value)
        if new_subject != self._selected_subject:
            self._selected_subject = new_subject
            # DON'T clear _cached_files or _cached_subjects - they don't change
            # Only clear data cache for this subject (others might be reused)
            # self._loaded_data_by_condition.clear()  # Keep cache for other subjects
            self._data = None
            # Render directly
            if self._container:
                self.render(self._container)
    
    def _on_epoch_change(self, value):
        val = extract_event_value(value)
        self._selected_epoch = int(val) if val else 0
        if self._container:
            self.render(self._container)
    
    def _load_condition_data(self, subject: str, condition: str) -> Optional[dict]:
        """Load data for a specific subject-condition pair - FAST direct path."""
        import pickle
        from pathlib import Path
        
        cache_key = f"{subject}-{condition}"
        if cache_key in self._loaded_data_by_condition:
            return self._loaded_data_by_condition[cache_key]
        
        run_dir = self.run_dir
        if not run_dir:
            return None
        
        # Try direct paths first (FAST - no searching)
        possible_paths = [
            # Pattern: {run_dir}/{condition}/syncro-{subject}-{condition}.pkl
            run_dir / condition / f"syncro-{subject}-{condition}.pkl",
            # Pattern: {run_dir}/syncro-{subject}-{condition}.pkl  
            run_dir / f"syncro-{subject}-{condition}.pkl",
            # Pattern: {run_dir}/{condition}/syncro-{subject}.pkl
            run_dir / condition / f"syncro-{subject}.pkl",
        ]
        
        for path in possible_paths:
            if path.exists():
                try:
                    with open(path, 'rb') as handle:
                        data = pickle.load(handle)
                    self._loaded_data_by_condition[cache_key] = data
                    return data
                except Exception:
                    pass
        
        # Fallback: search in cached files (slower but thorough)
        if self._cached_files is None:
            self._cached_files = self.find_data_files()
        
        for f in self._cached_files:
            is_match = (
                (subject in f.name and condition in f.name) or
                (subject in f.name and f.parent.name == condition)
            )
            if is_match:
                try:
                    with open(f, 'rb') as handle:
                        data = pickle.load(handle)
                    self._loaded_data_by_condition[cache_key] = data
                    return data
                except Exception:
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
        
        with ui.row().classes('w-full gap-3 flex-wrap justify-center'):
            for cond, data in condition_data.items():
                # Each condition in a column - responsive sizing
                if n_conditions == 1:
                    col_style = 'flex: 0 0 auto; width: 500px; max-width: 100%;'
                elif n_conditions == 2:
                    col_style = 'flex: 0 0 auto; width: 450px; max-width: 48%;'
                else:
                    col_style = 'flex: 0 0 auto; width: 380px; max-width: 32%;'
                
                with ui.column().style(col_style):
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
        
        # Always use high-quality matplotlib
        self._render_hq_matplotlib_heatmap(matrix, condition, color)
        
        # Stats + Histogram row
        with ui.row().classes('w-full gap-2 mt-1'):
            self._render_compact_stats(matrix, color)
            self._render_compact_histogram(matrix, color)
    
    def _render_hq_matplotlib_heatmap(self, matrix: np.ndarray, condition: str, color: str) -> None:
        """Render HIGH QUALITY matplotlib heatmap with white text."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # High DPI for crisp rendering
        dpi = 150
        fig, ax = plt.subplots(figsize=(4.5, 4), dpi=dpi)
        
        # Dark background
        fig.patch.set_facecolor('#0a0a0a')
        ax.set_facecolor('#0a0a0a')
        
        # Plot matrix
        im = ax.imshow(matrix, cmap='viridis', vmin=0, vmax=1, aspect='equal', 
                       interpolation='nearest')  # 'nearest' for sharp pixels
        
        # Title in condition color
        ax.set_title(f'PLV Alpha - {condition}', color=color, fontsize=12, 
                     fontweight='bold', pad=10)
        
        # White axis labels
        ax.set_xlabel('ROI', color='white', fontsize=11)
        ax.set_ylabel('ROI', color='white', fontsize=11)
        
        # White tick labels
        ax.tick_params(colors='white', labelsize=10)
        
        # Set ticks
        tick_positions = [0, 20, 40, 60, 80, 100]
        ax.set_xticks([t for t in tick_positions if t < matrix.shape[1]])
        ax.set_yticks([t for t in tick_positions if t < matrix.shape[0]])
        
        # Colorbar with WHITE text
        cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label('PLV', color='white', fontsize=11)
        cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white', labelsize=10)
        cbar.outline.set_edgecolor('#444444')
        
        # Set specific ticks on colorbar
        cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        
        # Make spine colors dark
        for spine in ax.spines.values():
            spine.set_color('#333333')
        
        plt.tight_layout()
        
        # Save to buffer with high quality
        buf = io.BytesIO()
        fig.savefig(buf, format='png', facecolor=fig.get_facecolor(), 
                    edgecolor='none', bbox_inches='tight', dpi=dpi)
        buf.seek(0)
        img_data = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        # Display
        ui.image(f'data:image/png;base64,{img_data}').style(
            'max-width: 100%; height: auto; border-radius: 4px;'
        )
    
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
        """Render compact histogram with high quality."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        upper = matrix[np.triu_indices_from(matrix, k=1)]
        
        dpi = 100
        fig, ax = plt.subplots(figsize=(2.5, 1.8), dpi=dpi)
        fig.patch.set_facecolor('#0a0a0a')
        ax.set_facecolor('#0a0a0a')
        
        ax.hist(upper, bins=20, color=color, alpha=0.8, edgecolor='none')
        ax.set_xlabel('PLV', color='white', fontsize=8)
        ax.set_ylabel('Count', color='white', fontsize=8)
        ax.tick_params(colors='white', labelsize=7)
        ax.set_xlim(0, 1)
        
        for spine in ax.spines.values():
            spine.set_color('#333333')
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', facecolor=fig.get_facecolor(), 
                    edgecolor='none', bbox_inches='tight', dpi=dpi)
        buf.seek(0)
        img_data = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        ui.image(f'data:image/png;base64,{img_data}').style(
            'max-width: 100%; height: auto;'
        )
    
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
