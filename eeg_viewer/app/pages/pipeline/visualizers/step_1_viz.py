"""
Step 1 Visualizer: Source Localization (fwd.py)

Shows:
- 3D Brain network view (default)
- Hilbert 2D and 3D visualizations (toggle)
- Summary stats of generated data
"""
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
from nicegui import ui

from config import THEME_PRIMARY, THEME_SECONDARY, THEME_TEXT_DIM, THEME_WARN
from .base import BaseVisualizer, extract_event_value

# Import plotly only when needed
try:
    import plotly.graph_objects as go
except ImportError:
    go = None


class Step1Visualizer(BaseVisualizer):
    """Visualizer for Step 1: Source Localization."""
    
    step_number = 1
    step_name = "Localización de Fuentes"
    step_description = "EEG → 100 regiones cerebrales"
    # Search in root and all subdirectories (DMT/, EC/, EO/)
    file_patterns = ["phases-*.pkl", "*/phases-*.pkl", "**/phases-*.pkl"]
    supported_backends = ['plotly']  # 3D always uses plotly
    
    def __init__(self, get_run_dir):
        super().__init__(get_run_dir)
        self._stats = None
        self._show_hilbert = False
        self._selected_band = 'Alpha'
        self._selected_epoch = 0
        self._selected_roi = 0
        self._selected_conditions = ['DMT']  # Support multiple conditions
        self._selected_subject = None  # Selected subject
        self._loaded_data = {}  # Cache data by subject
        self._subject_select = None  # UI reference
    
    def get_controls(self) -> Dict[str, Any]:
        # Minimal controls for Hilbert view
        return {'hilbert_toggle': 'button'}
    
    def find_data_files(self) -> List[Path]:
        """
        Find data files matching this step's patterns.
        Overridden to search properly in subdirectories.
        """
        if not self.run_dir or not self.run_dir.exists():
            return []
        
        files = set()
        
        # Direct files in root
        files.update(self.run_dir.glob('phases-*.pkl'))
        
        # Files in condition subdirectories (DMT, EC, EO)
        for subdir in ['DMT', 'EC', 'EO']:
            subdir_path = self.run_dir / subdir
            if subdir_path.exists():
                files.update(subdir_path.glob('phases-*.pkl'))
        
        # Also try recursive search
        files.update(self.run_dir.rglob('phases-*.pkl'))
        
        return sorted(files)
    
    def _render_controls(self) -> None:
        # Hilbert toggle button
        with ui.row().classes('gap-2 mb-2'):
            ui.button(
                '📈 Hilbert 2D/3D',
                on_click=self._toggle_hilbert
            ).props('dense flat size=sm').style(
                f'color: {"#00d4aa" if self._show_hilbert else THEME_TEXT_DIM};'
            )
    
    def _toggle_hilbert(self) -> None:
        self._show_hilbert = not self._show_hilbert
        self.render(self._container)
    
    def _render_visualization(self) -> None:
        # Gather stats from all generated files
        self._gather_stats()
        
        if self._show_hilbert:
            self._render_hilbert_view()
        else:
            with ui.row().classes('w-full gap-4'):
                # Left: 3D Brain Network (main visualization)
                with ui.column().classes('flex-1'):
                    self._render_brain_3d()
                
                # Right: Summary stats
                with ui.column().style('width: 220px;'):
                    self._render_summary_stats()
    
    def _get_available_conditions(self) -> List[str]:
        """Get list of available conditions from files."""
        files = self.find_data_files()
        conditions = set()
        
        for f in files:
            name = f.name.upper()
            if 'DMT' in name:
                conditions.add('DMT')
            elif 'EC' in name and 'PEC' not in name:
                conditions.add('EC')
            elif 'EO' in name:
                conditions.add('EO')
        
        # Also check parent directories
        for f in files:
            parent = f.parent.name.upper()
            if parent in ['DMT', 'EC', 'EO']:
                conditions.add(parent)
        
        return sorted(conditions) if conditions else ['DMT']
    
    def _get_subjects_for_condition(self, condition: str) -> List[str]:
        """Get subjects for a specific condition (just IDs like S01, S02)."""
        files = self.find_data_files()
        subjects = []
        
        for f in files:
            # Check if file belongs to this condition
            in_condition = (
                condition.lower() in f.name.lower() or 
                f.parent.name.upper() == condition
            )
            
            if in_condition:
                # Extract subject ID (e.g., phases-S01-DMT -> S01)
                name = f.stem.replace('phases-', '')
                if '-' in name:
                    subj_id = name.split('-')[0]
                    subjects.append(subj_id)
                else:
                    subjects.append(name)
        
        return sorted(set(subjects))
    
    def _get_all_subjects(self) -> List[str]:
        """Get all available subjects across all conditions (just IDs like S01, S02)."""
        files = self.find_data_files()
        subjects = set()
        
        for f in files:
            # Extract subject ID from filename (e.g., phases-S01-DMT -> S01)
            name = f.stem.replace('phases-', '')
            # Extract just the subject ID (S01, S02, etc.) without condition
            if '-' in name:
                subj_id = name.split('-')[0]  # S01-DMT -> S01
                subjects.add(subj_id)
            else:
                subjects.add(name)
        
        return sorted(subjects)
    
    def _on_subject_change(self, value) -> None:
        """Handle subject selection change."""
        self._selected_subject = extract_event_value(value)
        self._loaded_data.clear()  # Clear cache to reload with new subject
        self.render(self._container)
    
    def _render_hilbert_view(self) -> None:
        """Render Hilbert 2D and 3D visualizations with condition selection."""
        available_conditions = self._get_available_conditions()
        
        # Get all available subjects
        all_subjects = self._get_all_subjects()
        if not self._selected_subject and all_subjects:
            self._selected_subject = all_subjects[0]
        
        # Header controls
        with ui.row().classes('items-center gap-3 mb-3 flex-wrap w-full'):
            # Condition checkboxes
            ui.label('Condiciones:').style(f'color: {THEME_TEXT_DIM}; font-size: 0.75rem;')
            
            for cond in available_conditions:
                is_selected = cond in self._selected_conditions
                ui.checkbox(
                    cond, 
                    value=is_selected,
                    on_change=lambda e, c=cond: self._toggle_condition(c, e.value)
                ).props('dense').style(
                    f'color: {"#00d4aa" if is_selected else THEME_TEXT_DIM};'
                )
            
            ui.element('div').classes('flex-grow')
            
            # Subject selector
            self._subject_select = ui.select(
                all_subjects,
                value=self._selected_subject, label='Sujeto'
            ).props('dense dark').classes('w-32').on(
                'update:model-value',
                lambda e: self._on_subject_change(e.args)
            )
            
            # Band selector
            ui.select(
                ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'],
                value=self._selected_band, label='Banda'
            ).props('dense dark').classes('w-24').on(
                'update:model-value',
                lambda e: self._on_hilbert_band_change(e.args)
            )
            
            # Epoch selector
            ui.number(
                value=self._selected_epoch, min=0, max=500, label='Época'
            ).props('dense').classes('w-20').on(
                'update:model-value',
                lambda e: setattr(self, '_selected_epoch', int(e.args or 0))
            )
            
            # ROI selector  
            ui.number(
                value=self._selected_roi, min=0, max=101, label='ROI'
            ).props('dense').classes('w-16').on(
                'update:model-value',
                lambda e: setattr(self, '_selected_roi', int(e.args or 0))
            )
            
            ui.button('Actualizar', on_click=lambda: self.render(self._container)).props(
                'dense size=sm'
            ).style(f'background: {THEME_PRIMARY}; color: black;')
        
        # Render plots for each selected condition
        if not self._selected_conditions:
            ui.label('Seleccioná al menos una condición').style(
                f'color: {THEME_WARN}; font-size: 0.8rem;'
            )
            return
        
        # Create grid of plots - one row per condition
        for cond in self._selected_conditions:
            if cond not in available_conditions:
                continue
            
            subjects = self._get_subjects_for_condition(cond)
            if not subjects:
                continue
            
            # Use selected subject if available in this condition, else first
            current_subject = self._selected_subject if self._selected_subject in subjects else subjects[0]
            
            # Load subject's data for this condition
            subject_data = self._load_condition_data(cond, current_subject)
            if not subject_data:
                continue
            
            # Condition header
            cond_colors = {'DMT': '#00d4aa', 'EC': '#f59e0b', 'EO': '#a78bfa'}
            with ui.row().classes('w-full items-center gap-2 mt-2'):
                ui.element('div').style(
                    f'width: 12px; height: 12px; border-radius: 50%; '
                    f'background: {cond_colors.get(cond, "#888")};'
                )
                ui.label(f'{cond} - {current_subject}').style(
                    f'color: {cond_colors.get(cond, "#888")}; font-weight: bold; font-size: 0.85rem;'
                )
                ui.label(f'({len(subjects)} sujetos disponibles)').style(
                    f'color: {THEME_TEXT_DIM}; font-size: 0.7rem;'
                )
            
            # Hilbert plots row - aligned containers
            with ui.row().classes('w-full gap-3 items-stretch'):
                # Hilbert 2D
                with ui.card().classes('flex-1 p-2').style(
                    'background: #0a0a0a; display: flex; flex-direction: column; min-height: 380px;'
                ):
                    ui.label('▌HILBERT 2D').style(
                        'color: #f472b6; font-family: JetBrains Mono; font-size: 0.75rem; margin-bottom: 4px;'
                    )
                    with ui.element('div').classes('flex-1').style('min-height: 350px;'):
                        self._render_hilbert_2d_proper(subject_data, cond)
                
                # Hilbert 3D
                with ui.card().classes('flex-1 p-2').style(
                    'background: #0a0a0a; display: flex; flex-direction: column; min-height: 380px;'
                ):
                    ui.label('▌HILBERT 3D PHASE SPACE').style(
                        'color: #c084fc; font-family: JetBrains Mono; font-size: 0.75rem; margin-bottom: 4px;'
                    )
                    with ui.element('div').classes('flex-1').style('min-height: 350px;'):
                        self._render_hilbert_3d_proper(subject_data, cond, current_subject)
    
    def _toggle_condition(self, condition: str, is_checked: bool) -> None:
        """Toggle a condition selection."""
        if is_checked and condition not in self._selected_conditions:
            self._selected_conditions.append(condition)
        elif not is_checked and condition in self._selected_conditions:
            self._selected_conditions.remove(condition)
        self.render(self._container)
    
    def _load_condition_data(self, condition: str, subject: str) -> Optional[dict]:
        """Load data for a specific condition/subject (subject is just ID like S01)."""
        cache_key = f"{condition}_{subject}"
        if cache_key in self._loaded_data:
            return self._loaded_data[cache_key]
        
        files = self.find_data_files()
        for f in files:
            in_condition = (
                condition.lower() in f.name.lower() or
                f.parent.name.upper() == condition
            )
            # Match subject ID (e.g., S01 matches phases-S01-DMT)
            has_subject = f'-{subject}-' in f.name or f'-{subject}.' in f.name or f.name.startswith(f'phases-{subject}')
            
            if in_condition and has_subject:
                try:
                    import pickle
                    with open(f, 'rb') as handle:
                        data = pickle.load(handle)
                    self._loaded_data[cache_key] = data
                    return data
                except Exception as e:
                    print(f"Error loading {f}: {e}")
                    return None
        
        return None
    
    def _on_hilbert_band_change(self, value):
        self._selected_band = extract_event_value(value)
        self.render(self._container)
    
    def _render_hilbert_2d_proper(self, data: dict, condition: str) -> None:
        """Render proper Hilbert 2D visualization with high quality."""
        try:
            phases = data.get('phases_stc', {})
            amplitudes = data.get('amplitudes_stc', {})
            
            band_phases = phases.get(self._selected_band)
            band_amps = amplitudes.get(self._selected_band) if amplitudes else None
            
            if band_phases is None:
                ui.label(f'No data for {self._selected_band}').style(f'color: {THEME_TEXT_DIM};')
                return
            
            # Convert to array if needed
            if isinstance(band_phases, list):
                band_phases = np.array(band_phases)
            
            n_epochs = band_phases.shape[0]
            n_rois = band_phases.shape[1] if band_phases.ndim > 1 else 1
            epoch_idx = min(self._selected_epoch, n_epochs - 1)
            fs = data.get('sfreq', 500.0)
            
            # Get data for epoch
            phase_data = band_phases[epoch_idx]  # [n_rois, n_times]
            n_times = phase_data.shape[1]
            time = np.arange(n_times) / fs
            
            # Use Plotly for better quality and consistent styling with 3D
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Create 2x2 subplot figure
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Amplitude Envelope', 'Instantaneous Phase', 
                               'Phase Evolution', 'Phase Distribution'),
                specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}],
                       [{'type': 'xy'}, {'type': 'polar'}]],
                horizontal_spacing=0.12,
                vertical_spacing=0.18
            )
            
            # 1. Amplitude Envelope Heatmap (top-left)
            # Shows instantaneous power of the analytic signal for each ROI over time
            if band_amps is not None:
                if isinstance(band_amps, list):
                    band_amps = np.array(band_amps)
                amp_data = band_amps[epoch_idx][:20]  # First 20 ROIs
                fig.add_trace(
                    go.Heatmap(
                        z=amp_data,
                        x=time,
                        y=list(range(20)),
                        colorscale='YlOrRd',
                        showscale=False,
                        hovertemplate=(
                            '<b>Amplitude Envelope</b><br>'
                            'Instantaneous power of the Hilbert transform.<br>'
                            'Warm colors = higher neural activity.<br><br>'
                            'Time: %{x:.2f}s | ROI: %{y} | Amp: %{z:.3f}'
                            '<extra></extra>'
                        )
                    ),
                    row=1, col=1
                )
            
            # 2. Instantaneous Phase Heatmap (top-right)
            # Shows phase angle (-π to +π) for each ROI over time
            fig.add_trace(
                go.Heatmap(
                    z=phase_data[:20],
                    x=time,
                    y=list(range(20)),
                    colorscale='HSV',
                    zmin=-np.pi, zmax=np.pi,
                    colorbar=dict(title='Phase', len=0.4, y=0.8, x=1.02),
                    hovertemplate=(
                        '<b>Instantaneous Phase</b><br>'
                        'Phase angle of the Hilbert transform (-π to +π).<br>'
                        'Cyclic colors (HSV) represent the full phase cycle.<br>'
                        'Useful for detecting phase synchronization.<br><br>'
                        'Time: %{x:.2f}s | ROI: %{y} | Phase: %{z:.2f} rad'
                        '<extra></extra>'
                    )
                ),
                row=1, col=2
            )
            
            # 3. Phase Evolution (bottom-left)
            # Shows how phase oscillates over time for representative ROIs
            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f59e0b', '#a78bfa']
            roi_indices = [0, 6, 12, 18, 24]
            for i, (roi_idx, color) in enumerate(zip(roi_indices, colors)):
                if roi_idx < n_rois:
                    fig.add_trace(
                        go.Scatter(
                            x=time,
                            y=phase_data[roi_idx],
                            mode='lines',
                            name=f'ROI {roi_idx}',
                            line=dict(color=color, width=1),
                            hovertemplate=(
                                '<b>Phase Evolution</b><br>'
                                'Temporal evolution of phase for 5 representative ROIs.<br>'
                                'Parallel lines indicate synchronized regions.<br><br>'
                                f'ROI {roi_idx} | Time: %{{x:.2f}}s | Phase: %{{y:.2f}} rad'
                                '<extra></extra>'
                            )
                        ),
                        row=2, col=1
                    )
            
            # 4. Polar Phase Distribution (bottom-right)
            # Shows phase distribution of all ROIs at a single time point
            t_idx = n_times // 2
            phases_at_t = phase_data[:, t_idx]
            r_values = np.ones_like(phases_at_t)
            
            fig.add_trace(
                go.Scatterpolar(
                    r=r_values,
                    theta=np.degrees(phases_at_t),
                    mode='markers',
                    marker=dict(color='#f59e0b', size=6, opacity=0.8),
                    name='Phases',
                    hovertemplate=(
                        '<b>Phase Distribution</b><br>'
                        'Polar distribution of all ROI phases at t=T/2.<br>'
                        'Each dot is one ROI. White arrow = mean phase.<br>'
                        'Arrow length = order parameter (coherence).<br><br>'
                        'Phase: %{theta:.1f}°'
                        '<extra></extra>'
                    )
                ),
                row=2, col=2
            )
            
            # Add mean phase vector
            mean_phase = np.angle(np.mean(np.exp(1j * phases_at_t)))
            r_mean = np.abs(np.mean(np.exp(1j * phases_at_t)))
            fig.add_trace(
                go.Scatterpolar(
                    r=[0, r_mean],
                    theta=[0, np.degrees(mean_phase)],
                    mode='lines+markers',
                    line=dict(color='white', width=3),
                    marker=dict(color='white', size=[0, 8]),
                    name='Mean',
                    hovertemplate=(
                        '<b>Mean Phase Vector</b><br>'
                        f'Mean phase: {np.degrees(mean_phase):.1f}°<br>'
                        f'Order parameter (R): {r_mean:.3f}<br>'
                        'R close to 1 = high synchronization'
                        '<extra></extra>'
                    )
                ),
                row=2, col=2
            )
            
            # Update layout for dark theme
            fig.update_layout(
                template='plotly_dark',
                height=350,
                margin=dict(l=50, r=70, t=40, b=50),
                showlegend=False,
                paper_bgcolor='rgba(10,10,10,1)',
                plot_bgcolor='rgba(10,10,10,1)',
                font=dict(size=9, color='#888888'),
            )
            
            # Update axes
            fig.update_xaxes(title_text='', tickfont_size=8, row=1, col=1)
            fig.update_yaxes(title_text='ROI', title_font_size=9, tickfont_size=8, row=1, col=1)
            fig.update_xaxes(title_text='', tickfont_size=8, row=1, col=2)
            fig.update_yaxes(title_text='ROI', title_font_size=9, tickfont_size=8, row=1, col=2)
            fig.update_xaxes(title_text='Time (s)', title_font_size=9, tickfont_size=8, row=2, col=1)
            fig.update_yaxes(title_text='Phase (rad)', title_font_size=9, tickfont_size=8, 
                            range=[-np.pi, np.pi], row=2, col=1)
            
            # Update polar subplot
            fig.update_polars(
                radialaxis=dict(visible=True, range=[0, 1.2], showticklabels=False),
                angularaxis=dict(tickfont_size=8),
                bgcolor='rgba(10,10,10,1)'
            )
            
            # Update subplot titles
            for annotation in fig['layout']['annotations']:
                annotation['font'] = dict(size=10, color='#f472b6')
            
            ui.plotly(fig).classes('w-full')
            
        except Exception as e:
            ui.label(f'Error: {e}').style(f'color: {THEME_WARN}; font-size: 0.75rem;')
    
    def _render_hilbert_3d_proper(self, data: dict, condition: str, subject: str) -> None:
        """Render proper Hilbert 3D phase space like the original visualize_hilbert_improved.py."""
        try:
            phases = data.get('phases_stc', {})
            amplitudes = data.get('amplitudes_stc', {})
            
            band_phases = phases.get(self._selected_band)
            band_amps = amplitudes.get(self._selected_band)
            
            if band_phases is None:
                ui.label(f'No data for {self._selected_band}').style(f'color: {THEME_TEXT_DIM};')
                return
            
            # Convert to array if needed
            if isinstance(band_phases, list):
                band_phases = np.array(band_phases)
            if band_amps is not None and isinstance(band_amps, list):
                band_amps = np.array(band_amps)
            
            n_epochs = band_phases.shape[0]
            n_rois = band_phases.shape[1]
            epoch_idx = min(self._selected_epoch, n_epochs - 1)
            roi_idx = min(self._selected_roi, n_rois - 1)
            
            # Get phase and amplitude for selected ROI and epoch
            phase_seg = band_phases[epoch_idx, roi_idx, :]
            
            if band_amps is not None:
                amp_seg = band_amps[epoch_idx, roi_idx, :]
            else:
                # Use unit amplitude if no amplitude data
                amp_seg = np.ones_like(phase_seg)
            
            # Time vector (assuming 500 Hz)
            fs = data.get('sfreq', 500.0)
            time_full = np.arange(len(phase_seg)) / fs
            
            # Trim edges to avoid filter artifacts (15% each side)
            trim = 0.15
            start = int(len(time_full) * trim)
            end = int(len(time_full) * (1 - trim))
            
            time_seg = time_full[start:end]
            phase_seg = phase_seg[start:end]
            amp_seg = amp_seg[start:end]
            
            # Analytical signal
            analytic_seg = amp_seg * np.exp(1j * phase_seg)
            real_seg = analytic_seg.real
            imag_seg = analytic_seg.imag
            
            # Create proper 3D figure like visualize_hilbert_improved.py
            import plotly.graph_objects as go
            
            # Main trajectory (colored by time)
            trajectory = go.Scatter3d(
                x=time_seg,
                y=real_seg,
                z=imag_seg,
                mode="lines",
                line=dict(
                    color=time_seg,
                    colorscale="Viridis",
                    width=4,
                    colorbar=dict(title="Tiempo (s)", len=0.6, y=0.5)
                ),
                name="Trayectoria",
                hovertemplate="t=%{x:.3f}s<br>Real=%{y:.4f}<br>Imag=%{z:.4f}<extra></extra>"
            )
            
            # Projections
            z_floor = np.full_like(time_seg, imag_seg.min() - 0.02)
            proj_time_real = go.Scatter3d(
                x=time_seg, y=real_seg, z=z_floor,
                mode="lines",
                line=dict(color="rgba(31, 119, 180, 0.5)", width=2, dash="dot"),
                name="Proyección tiempo-real",
                hoverinfo="skip"
            )
            
            y_wall = np.full_like(time_seg, real_seg.min() - 0.02)
            proj_time_imag = go.Scatter3d(
                x=time_seg, y=y_wall, z=imag_seg,
                mode="lines",
                line=dict(color="rgba(214, 39, 40, 0.5)", width=2, dash="dot"),
                name="Proyección tiempo-imag",
                hoverinfo="skip"
            )
            
            x_wall = np.full_like(real_seg, time_seg.max() + 0.02)
            proj_real_imag = go.Scatter3d(
                x=x_wall, y=real_seg, z=imag_seg,
                mode="lines",
                line=dict(color="rgba(148, 103, 189, 0.5)", width=2, dash="longdash"),
                name="Proyección plano complejo",
                hoverinfo="skip"
            )
            
            # Start/end markers
            markers = go.Scatter3d(
                x=[time_seg[0], time_seg[-1]],
                y=[real_seg[0], real_seg[-1]],
                z=[imag_seg[0], imag_seg[-1]],
                mode="markers",
                marker=dict(
                    size=6, 
                    color=["#2ca02c", "#ff7f0e"], 
                    symbol=["circle", "square"],
                    line=dict(width=1.5, color="black")
                ),
                name="Inicio / Final",
                hoverinfo="skip"
            )
            
            fig = go.Figure(data=[trajectory, proj_time_real, proj_time_imag, 
                                  proj_real_imag, markers])
            
            fig.update_layout(
                template="plotly_dark",
                scene=dict(
                    xaxis=dict(
                        title=dict(text="Tiempo (s)", font=dict(size=10)),
                        tickfont=dict(size=8),
                        backgroundcolor="rgba(0,0,0,0)",
                        gridcolor="rgba(100,100,100,0.2)"
                    ),
                    yaxis=dict(
                        title=dict(text="Re(z) = A·cos(φ)", font=dict(size=10)),
                        tickfont=dict(size=8),
                        backgroundcolor="rgba(0,0,0,0)",
                        gridcolor="rgba(100,100,100,0.2)"
                    ),
                    zaxis=dict(
                        title=dict(text="Im(z) = A·sin(φ)", font=dict(size=10)),
                        tickfont=dict(size=8),
                        backgroundcolor="rgba(0,0,0,0)",
                        gridcolor="rgba(100,100,100,0.2)"
                    ),
                    camera=dict(eye=dict(x=1.6, y=1.4, z=0.9)),
                    aspectmode="manual",
                    aspectratio=dict(x=1.5, y=1, z=1)
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.15,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=8)
                ),
                margin=dict(l=0, r=0, b=40, t=60),
                title=dict(
                    text=(
                        f"<b>Transformada de Hilbert - Señal Analítica 3D</b><br>"
                        f"<span style='font-size:10px;'>{subject} | {condition} | "
                        f"Banda {self._selected_band} | Época #{epoch_idx + 1} | ROI {roi_idx}</span>"
                    ),
                    x=0.5,
                    xanchor="center",
                    font=dict(size=12)
                ),
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            ui.plotly(fig).classes('w-full')
            
        except Exception as e:
            ui.label(f'Error: {e}').style(f'color: {THEME_WARN}; font-size: 0.75rem;')
    
    def _gather_stats(self) -> None:
        """Gather statistics from all phases files."""
        files = self.find_data_files()
        
        if not files:
            self._stats = None
            return
        
        # Count subjects by condition
        conditions = {'DMT': 0, 'EC': 0, 'EO': 0, 'Other': 0}
        total_epochs = 0
        sfreq = None
        n_parcels = None
        n_times = None
        
        for f in files:
            name = f.name.upper()
            parent = f.parent.name.upper()
            
            # Check both filename and parent directory
            if 'DMT' in name or parent == 'DMT':
                conditions['DMT'] += 1
            elif ('EC' in name or parent == 'EC') and 'PEC' not in name:
                conditions['EC'] += 1
            elif 'EO' in name or parent == 'EO':
                conditions['EO'] += 1
            else:
                conditions['Other'] += 1
            
            # Load one file to get metadata
            if sfreq is None:
                try:
                    self.load_data(f)
                    if self._data:
                        sfreq = self._data.get('sfreq', 0)
                        n_epochs = self._data.get('n_epochs', 0)
                        total_epochs += n_epochs if n_epochs else 0
                        
                        # Get shape from phases_stc
                        phases = self._data.get('phases_stc', {})
                        if phases:
                            first_band = list(phases.values())[0]
                            if isinstance(first_band, list) and first_band:
                                arr = np.array(first_band)
                                n_parcels = arr.shape[1] if arr.ndim > 1 else None
                                n_times = arr.shape[2] if arr.ndim > 2 else None
                            elif isinstance(first_band, np.ndarray):
                                n_parcels = first_band.shape[1] if first_band.ndim > 1 else None
                                n_times = first_band.shape[2] if first_band.ndim > 2 else None
                except:
                    pass
        
        # Remove zero counts
        conditions = {k: v for k, v in conditions.items() if v > 0}
        
        self._stats = {
            'total_files': len(files),
            'conditions': conditions,
            'total_epochs': total_epochs,
            'sfreq': sfreq,
            'n_parcels': n_parcels,
            'n_times': n_times,
        }
    
    def _render_brain_3d(self) -> None:
        """Render 3D brain network visualization."""
        try:
            # Try to load from viz_scripts
            try:
                from viz_scripts import brain_3d
                fig = brain_3d.create_brain_network_figure()
                fig.update_layout(
                    height=450,
                    margin=dict(l=0, r=0, t=30, b=0),
                    autosize=True
                )
                ui.plotly(fig).classes('w-full').style('height: 100%; min-height: 400px;')
            except ImportError:
                # Fallback: create our own 3D brain visualization
                self._render_fallback_brain_3d()
                
        except Exception as e:
            ui.label(f'Error rendering 3D brain: {e}').style(
                f'color: {THEME_WARN}; font-size: 0.75rem;'
            )
    
    def _render_fallback_brain_3d(self) -> None:
        """Render fallback 3D brain when viz_scripts not available."""
        import plotly.graph_objects as go
        
        # Schaefer 7-network positions (approximate)
        np.random.seed(42)
        n_parcels = 102
        
        # Generate brain-like positions
        theta = np.random.uniform(0, 2*np.pi, n_parcels)
        phi = np.random.uniform(0.2, np.pi - 0.2, n_parcels)
        r = 0.85 + np.random.uniform(0, 0.15, n_parcels)
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        
        # 7 networks
        network_names = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']
        network_full = ['Visual Network (VN)', 'Somatomotor Network (SMN)', 
                       'Dorsal Attention Network (DAN)', 'Salience/Ventral Attention (SVAN)',
                       'Limbic Network (LN)', 'Frontoparietal Network (FPN)', 'Default Mode Network (DMN)']
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dfe6e9', '#a29bfe']
        
        parcels_per_net = [15, 15, 15, 14, 14, 15, 14]
        networks = np.repeat(np.arange(7), parcels_per_net)
        
        fig = go.Figure()
        
        for i, (net, full_name, color) in enumerate(zip(network_names, network_full, colors)):
            mask = networks == i
            fig.add_trace(go.Scatter3d(
                x=x[mask], y=y[mask], z=z[mask],
                mode='markers',
                marker=dict(size=5, color=color, opacity=0.8),
                name=full_name,
                text=[f'{net} - Parcel {j+1}' for j in np.where(mask)[0]],
                hoverinfo='text'
            ))
        
        fig.update_layout(
            template='plotly_dark',
            title=dict(text='Brain Networks (7 Networks - Schaefer Atlas)', 
                      font=dict(size=12, color='#00d4aa')),
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
                aspectmode='cube'
            ),
            height=450,
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(
                yanchor="top", y=0.99,
                xanchor="left", x=0.01,
                bgcolor='rgba(0,0,0,0.5)',
                font=dict(size=9, color='#888888')
            ),
            autosize=True
        )
        ui.plotly(fig).classes('w-full').style('height: 100%; min-height: 400px;')
    
    def _scan_directory_structure(self) -> dict:
        """Scan directory for subdirectories and any pkl files."""
        if not self.run_dir or not self.run_dir.exists():
            return {}
        
        result = {
            'has_subdirs': False,
            'subdirs': {},
            'root_files': 0,
        }
        
        # Check for condition subdirectories
        for subdir_name in ['DMT', 'EC', 'EO']:
            subdir_path = self.run_dir / subdir_name
            if subdir_path.exists() and subdir_path.is_dir():
                result['has_subdirs'] = True
                # Count any pkl files in subdir
                pkl_files = list(subdir_path.glob('*.pkl'))
                phases_files = [f for f in pkl_files if 'phases' in f.name.lower()]
                order_files = [f for f in pkl_files if 'order' in f.name.lower()]
                syncro_files = [f for f in pkl_files if 'syncro' in f.name.lower()]
                
                result['subdirs'][subdir_name] = {
                    'total_pkl': len(pkl_files),
                    'phases': len(phases_files),
                    'order': len(order_files),
                    'syncro': len(syncro_files),
                }
        
        # Count root pkl files
        result['root_files'] = len(list(self.run_dir.glob('*.pkl')))
        
        return result
    
    def _render_summary_stats(self) -> None:
        """Render summary statistics of generated data."""
        with ui.card().classes('p-3 w-full').style(
            'background: #1a1a1a; border: 1px solid #333;'
        ):
            ui.label('📊 Datos Generados').style(
                f'color: {THEME_PRIMARY}; font-family: JetBrains Mono; '
                f'font-size: 0.85rem; font-weight: bold; margin-bottom: 8px;'
            )
            
            # Scan directory structure first
            dir_info = self._scan_directory_structure()
            
            if not self._stats or self._stats['total_files'] == 0:
                # No phases files, but check if there are subdirectories with other files
                if dir_info.get('has_subdirs'):
                    ui.label('📂 Estructura detectada:').style(
                        f'color: {THEME_SECONDARY}; font-size: 0.75rem; margin-bottom: 4px;'
                    )
                    
                    cond_colors = {'DMT': '#00d4aa', 'EC': '#f59e0b', 'EO': '#a78bfa'}
                    
                    for subdir, info in dir_info['subdirs'].items():
                        with ui.row().classes('items-center gap-2'):
                            ui.element('div').style(
                                f'width: 10px; height: 10px; border-radius: 50%; '
                                f'background: {cond_colors.get(subdir, "#888")};'
                            )
                            ui.label(f'{subdir}/').style(
                                f'color: {cond_colors.get(subdir, "#888")}; font-weight: bold; font-size: 0.75rem;'
                            )
                            ui.label(f'{info["total_pkl"]} archivos').style(
                                f'color: {THEME_TEXT_DIM}; font-size: 0.7rem;'
                            )
                        
                        # Show file breakdown if any
                        if info['phases'] > 0 or info['order'] > 0 or info['syncro'] > 0:
                            with ui.row().classes('ml-4 gap-2'):
                                if info['phases'] > 0:
                                    ui.label(f'phases: {info["phases"]}').style(
                                        f'color: {THEME_TEXT_DIM}; font-size: 0.65rem;'
                                    )
                                if info['order'] > 0:
                                    ui.label(f'order: {info["order"]}').style(
                                        f'color: {THEME_TEXT_DIM}; font-size: 0.65rem;'
                                    )
                                if info['syncro'] > 0:
                                    ui.label(f'syncro: {info["syncro"]}').style(
                                        f'color: {THEME_TEXT_DIM}; font-size: 0.65rem;'
                                    )
                    
                    if dir_info['root_files'] > 0:
                        ui.label(f'+ {dir_info["root_files"]} archivos en raíz').style(
                            f'color: {THEME_TEXT_DIM}; font-size: 0.7rem; margin-top: 4px;'
                        )
                    
                    ui.separator().classes('my-2')
                    ui.label('No hay phases-*.pkl aún').style(
                        f'color: {THEME_WARN}; font-size: 0.7rem;'
                    )
                    ui.label('Ejecutá "Localización de Fuentes"').style(
                        f'color: {THEME_SECONDARY}; font-size: 0.65rem;'
                    )
                else:
                    ui.label('No hay archivos phases-*.pkl').style(
                        f'color: {THEME_TEXT_DIM}; font-size: 0.75rem;'
                    )
                    ui.label('Ejecutá "Localización de Fuentes"').style(
                        f'color: {THEME_SECONDARY}; font-size: 0.7rem;'
                    )
                return
            
            # Total files
            ui.label(f'📁 {self._stats["total_files"]} archivos').style(
                f'color: {THEME_PRIMARY}; font-size: 0.8rem; margin-bottom: 4px;'
            )
            
            # Subjects by condition
            ui.label('Sujetos por condición:').style(
                f'color: {THEME_TEXT_DIM}; font-size: 0.7rem; margin-top: 8px;'
            )
            
            cond_colors = {'DMT': '#00d4aa', 'EC': '#f59e0b', 'EO': '#a78bfa', 'Other': '#888888'}
            for cond, count in self._stats['conditions'].items():
                with ui.row().classes('items-center gap-2'):
                    ui.element('div').style(
                        f'width: 10px; height: 10px; border-radius: 50%; '
                        f'background: {cond_colors.get(cond, "#888")};'
                    )
                    ui.label(f'{cond}: {count}').style(
                        f'color: {cond_colors.get(cond, "#888")}; font-size: 0.75rem;'
                    )
            
            ui.separator().classes('my-2')
            
            # Metadata
            ui.label('Metadatos:').style(
                f'color: {THEME_TEXT_DIM}; font-size: 0.7rem;'
            )
            
            if self._stats['sfreq']:
                ui.label(f'Freq: {self._stats["sfreq"]:.0f} Hz').style(
                    f'color: {THEME_TEXT_DIM}; font-size: 0.7rem;'
                )
            
            if self._stats['n_parcels']:
                ui.label(f'Parcelas: {self._stats["n_parcels"]}').style(
                    f'color: {THEME_TEXT_DIM}; font-size: 0.7rem;'
                )
            
            if self._stats['n_times']:
                ui.label(f'Muestras/época: {self._stats["n_times"]}').style(
                    f'color: {THEME_TEXT_DIM}; font-size: 0.7rem;'
                )
            
            # List some files
            ui.separator().classes('my-2')
            ui.label('Archivos:').style(
                f'color: {THEME_TEXT_DIM}; font-size: 0.7rem;'
            )
            
            files = self.find_data_files()[:5]  # Show first 5
            for f in files:
                ui.label(f'📄 {f.name}').style(
                    f'color: {THEME_SECONDARY}; font-size: 0.65rem;'
                )
            
            if len(self.find_data_files()) > 5:
                ui.label(f'... y {len(self.find_data_files()) - 5} más').style(
                    f'color: {THEME_TEXT_DIM}; font-size: 0.6rem;'
                )
