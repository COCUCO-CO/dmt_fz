"""
Step 8 Visualizer: Clustering (clustering.py)

Shows brain state clustering results:
- PCA scatter plot
- Silhouette scores
- Cluster occupancy
"""
from typing import Dict, Any, List
from pathlib import Path
import numpy as np
from nicegui import ui

from config import THEME_PRIMARY, THEME_SECONDARY, THEME_TEXT_DIM, THEME_WARN
from .base import BaseVisualizer, extract_event_value


class Step8Visualizer(BaseVisualizer):
    """Visualizer for Step 8: Clustering."""
    
    step_number = 8
    step_name = "Estados Cerebrales"
    step_description = "Clustering de patrones de conectividad"
    file_patterns = ["clustering_results/*.pkl", "clustering_results/**/*.pkl"]
    supported_backends = ['matplotlib', 'plotly']
    
    def __init__(self, get_run_dir):
        super().__init__(get_run_dir)
        self._selected_band = 'Alpha'
        self._show_3d = False
    
    def get_controls(self) -> Dict[str, Any]:
        return {
            'band': ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
        }
    
    def _render_controls(self) -> None:
        with ui.row().classes('items-center gap-3 mb-2 flex-wrap'):
            ui.select(
                ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'],
                value=self._selected_band, label='Banda'
            ).props('dense dark').classes('w-24').on(
                'update:model-value',
                lambda e: self._on_band_change(e.args)
            )
            
            ui.button(
                '🔮 3D PCA', 
                on_click=self._toggle_3d
            ).props('dense flat').style(
                f'color: {"#a78bfa" if self._show_3d else THEME_TEXT_DIM};'
            )
    
    def _on_band_change(self, value):
        self._selected_band = extract_event_value(value)
        self.render(self._container)
    
    def _toggle_3d(self):
        self._show_3d = not self._show_3d
        self.render(self._container)
    
    def _get_clustering_dir(self) -> Path:
        if self.run_dir:
            return self.run_dir / 'clustering_results'
        return None
    
    def _find_images(self) -> List[Path]:
        """Find clustering visualization images."""
        clust_dir = self._get_clustering_dir()
        if not clust_dir or not clust_dir.exists():
            return []
        
        images = []
        for pattern in ['*.png', '*.svg', '**/*.png', '**/*.svg']:
            images.extend(clust_dir.glob(pattern))
        
        # Filter by band if specified
        if self._selected_band != 'All':
            images = [f for f in images if self._selected_band.lower() in f.name.lower()]
        
        return sorted(set(images))
    
    def _render_visualization(self) -> None:
        clust_dir = self._get_clustering_dir()
        
        if not clust_dir or not clust_dir.exists():
            self._render_no_data_message("No hay resultados de clustering")
            ui.label("Ejecutá clustering.py para identificar estados").style(
                f'color: {THEME_TEXT_DIM}; font-size: 0.7rem;'
            )
            return
        
        with ui.row().classes('w-full gap-3'):
            # Left: Main visualization
            with ui.column().classes('flex-1'):
                if self._show_3d:
                    self._render_pca_3d()
                else:
                    self._render_pca_2d()
            
            # Right: Stats and scores
            with ui.column().style('width: 200px;'):
                self._render_silhouette_scores()
                self._render_cluster_stats()
    
    def _render_pca_2d(self) -> None:
        """Render 2D PCA scatter plot."""
        try:
            # Try to find PCA data
            clust_dir = self._get_clustering_dir()
            pca_files = list(clust_dir.glob(f'*{self._selected_band}*pca*.pkl'))
            
            if not pca_files:
                pca_files = list(clust_dir.glob('*pca*.pkl'))
            
            if not pca_files:
                # Render placeholder
                fig, ax = self.create_matplotlib_figure(figsize=(6, 4))
                ax.text(0.5, 0.5, 'No PCA data found\nRun clustering first', 
                       ha='center', va='center', color='#888888', fontsize=10)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                self.show_matplotlib(fig)
                return
            
            self.load_data(pca_files[0])
            
            if not self._data:
                return
            
            # Extract PCA components and labels
            pca_coords = self._data.get('pca_coords', self._data.get('X_pca'))
            labels = self._data.get('labels', self._data.get('cluster_labels'))
            
            if pca_coords is None:
                ui.label('PCA coords not found in file').style(f'color: {THEME_TEXT_DIM};')
                return
            
            fig, ax = self.create_matplotlib_figure(figsize=(6, 4))
            
            # Color by cluster
            if labels is not None:
                unique_labels = np.unique(labels)
                colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', 
                         '#dfe6e9', '#a29bfe', '#fd79a8']
                
                for i, label in enumerate(unique_labels):
                    mask = labels == label
                    ax.scatter(pca_coords[mask, 0], pca_coords[mask, 1],
                              c=colors[i % len(colors)], s=20, alpha=0.6,
                              label=f'Cluster {label}')
                ax.legend(facecolor='#0a0a0a', edgecolor='#333', labelcolor='#888', 
                         fontsize=7, loc='upper right')
            else:
                ax.scatter(pca_coords[:, 0], pca_coords[:, 1],
                          c=THEME_PRIMARY, s=20, alpha=0.6)
            
            ax.set_xlabel('PC1', color='#888888')
            ax.set_ylabel('PC2', color='#888888')
            ax.set_title(f'PCA Clustering - {self._selected_band}', 
                        color=THEME_PRIMARY, fontsize=10)
            
            self.show_matplotlib(fig)
            
        except Exception as e:
            ui.label(f'Error: {e}').style(f'color: {THEME_WARN}; font-size: 0.75rem;')
    
    def _render_pca_3d(self) -> None:
        """Render 3D PCA scatter plot using plotly."""
        try:
            clust_dir = self._get_clustering_dir()
            pca_files = list(clust_dir.glob(f'*{self._selected_band}*pca*.pkl'))
            
            if not pca_files:
                pca_files = list(clust_dir.glob('*pca*.pkl'))
            
            if not pca_files:
                ui.label('No PCA data for 3D visualization').style(f'color: {THEME_TEXT_DIM};')
                return
            
            self.load_data(pca_files[0])
            
            pca_coords = self._data.get('pca_coords', self._data.get('X_pca'))
            labels = self._data.get('labels', self._data.get('cluster_labels'))
            
            if pca_coords is None or pca_coords.shape[1] < 3:
                ui.label('Need 3 PCA components for 3D plot').style(f'color: {THEME_TEXT_DIM};')
                return
            
            import plotly.graph_objects as go
            
            if labels is not None:
                fig = go.Figure()
                unique_labels = np.unique(labels)
                colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', 
                         '#dfe6e9', '#a29bfe', '#fd79a8']
                
                for i, label in enumerate(unique_labels):
                    mask = labels == label
                    fig.add_trace(go.Scatter3d(
                        x=pca_coords[mask, 0],
                        y=pca_coords[mask, 1],
                        z=pca_coords[mask, 2],
                        mode='markers',
                        marker=dict(size=4, color=colors[i % len(colors)], opacity=0.7),
                        name=f'Cluster {label}'
                    ))
            else:
                fig = go.Figure(data=[go.Scatter3d(
                    x=pca_coords[:, 0],
                    y=pca_coords[:, 1],
                    z=pca_coords[:, 2],
                    mode='markers',
                    marker=dict(size=4, color=THEME_PRIMARY, opacity=0.7)
                )])
            
            fig.update_layout(
                template='plotly_dark',
                title=f'3D PCA - {self._selected_band}',
                scene=dict(
                    xaxis_title='PC1',
                    yaxis_title='PC2',
                    zaxis_title='PC3'
                ),
                height=350,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            ui.plotly(fig).classes('w-full')
            
        except Exception as e:
            ui.label(f'Error: {e}').style(f'color: {THEME_WARN}; font-size: 0.75rem;')
    
    def _render_silhouette_scores(self) -> None:
        """Render silhouette score chart."""
        try:
            clust_dir = self._get_clustering_dir()
            
            # Look for silhouette CSV or scores in pkl
            csv_files = list(clust_dir.glob('*silhouette*.csv'))
            
            if csv_files:
                import pandas as pd
                df = pd.read_csv(csv_files[0])
                
                if 'k' in df.columns and 'silhouette' in df.columns:
                    fig, ax = self.create_matplotlib_figure(figsize=(3.5, 2.5))
                    ax.plot(df['k'], df['silhouette'], 'o-', color=THEME_PRIMARY, markersize=4)
                    ax.set_xlabel('K', color='#888888', fontsize=8)
                    ax.set_ylabel('Silhouette', color='#888888', fontsize=8)
                    ax.tick_params(labelsize=7)
                    ax.set_title('Silhouette Scores', color=THEME_SECONDARY, fontsize=9)
                    self.show_matplotlib(fig)
                    return
            
            # Fallback message
            with ui.card().classes('p-2').style('background: #1a1a1a;'):
                ui.label('Silhouette').style(f'color: {THEME_SECONDARY}; font-size: 0.75rem;')
                ui.label('No scores found').style(f'color: {THEME_TEXT_DIM}; font-size: 0.7rem;')
                
        except Exception as e:
            ui.label(f'Error: {e}').style(f'color: {THEME_TEXT_DIM}; font-size: 0.7rem;')
    
    def _render_cluster_stats(self) -> None:
        """Render cluster statistics."""
        try:
            clust_dir = self._get_clustering_dir()
            pkl_files = list(clust_dir.glob('*.pkl'))
            
            stats = {
                'Files': len(pkl_files),
                'Images': len(self._find_images()),
            }
            
            # Try to get cluster count from data
            if self._data and 'labels' in self._data:
                labels = self._data['labels']
                stats['Clusters'] = len(np.unique(labels))
                stats['Samples'] = len(labels)
            
            self._render_stats_card(stats, "📊 Clustering")
            
        except:
            pass

