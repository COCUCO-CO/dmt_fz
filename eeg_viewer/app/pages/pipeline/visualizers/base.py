"""
Base class for pipeline step visualizers.

Provides common interface and functionality for all step-specific visualizers.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any
import pickle
from nicegui import ui

from config import THEME_PRIMARY, THEME_SECONDARY, THEME_TEXT_DIM, THEME_CARD, THEME_BORDER


def extract_event_value(value: Any) -> Any:
    """
    Extract the actual value from NiceGUI event args.
    
    NiceGUI select events can return either:
    - A simple value (string, int, etc.)
    - A dict like {'value': 1, 'label': 'Theta'}
    
    This helper extracts the usable value in both cases.
    """
    if isinstance(value, dict):
        # Return 'label' for display value, or 'value' as fallback
        return value.get('label', value.get('value', value))
    return value


class BaseVisualizer(ABC):
    """
    Abstract base class for step visualizers.
    
    Each step (1-8) has its own visualizer that knows:
    - What files to look for
    - What visualizations are relevant
    - What controls are needed
    """
    
    # Mapping from internal step number to display number (button number)
    # Steps 2 and 6 are skipped in the UI
    STEP_TO_DISPLAY = {1: 1, 3: 2, 4: 3, 5: 4, 7: 5, 8: 6}
    
    # Class attributes - override in subclasses
    step_number: int = 0
    step_name: str = ""
    step_description: str = ""
    file_patterns: List[str] = []  # Glob patterns for data files
    supported_backends: List[str] = ['matplotlib', 'plotly']
    
    @property
    def display_number(self) -> int:
        """Get the display number (what user sees in buttons)."""
        return self.STEP_TO_DISPLAY.get(self.step_number, self.step_number)
    
    def __init__(self, get_run_dir: Callable[[], Optional[Path]]):
        """
        Initialize visualizer.
        
        Args:
            get_run_dir: Function that returns current run directory
        """
        self._get_run_dir = get_run_dir
        self._data = None
        self._current_file = None
        self._container = None
        self._controls = {}
        self._backend = 'matplotlib'  # Default to faster backend
    
    @property
    def run_dir(self) -> Optional[Path]:
        """Get current run directory."""
        return self._get_run_dir()
    
    def get_controls(self) -> Dict[str, Any]:
        """
        Get available controls for this visualizer.
        
        Returns:
            Dict with control names and their specs
        """
        return {}
    
    def get_3d_backend(self) -> str:
        """Get backend for 3D visualizations (always plotly)."""
        return 'plotly'
    
    def set_backend(self, backend: str) -> None:
        """Set visualization backend (matplotlib or plotly)."""
        if backend in self.supported_backends:
            self._backend = backend
    
    def find_data_files(self) -> List[Path]:
        """
        Find data files matching this step's patterns.
        
        Returns:
            List of unique paths to data files (sorted)
        """
        if not self.run_dir or not self.run_dir.exists():
            return []
        
        files = set()
        for pattern in self.file_patterns:
            files.update(self.run_dir.rglob(pattern))
        return sorted(files)
    
    def load_data(self, file_path: Optional[Path] = None) -> bool:
        """
        Load data from pickle file.
        
        Args:
            file_path: Specific file to load, or auto-detect
            
        Returns:
            True if data loaded successfully
        """
        if file_path is None:
            files = self.find_data_files()
            if not files:
                return False
            file_path = files[0]
        
        try:
            with open(file_path, 'rb') as f:
                self._data = pickle.load(f)
            self._current_file = file_path
            return True
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return False
    
    def render(self, container=None) -> None:
        """
        Render the visualization.
        
        Args:
            container: NiceGUI container to render into
        """
        if container:
            self._container = container
        
        if self._container:
            self._container.clear()
            with self._container:
                self._render_content()
        else:
            self._render_content()
    
    def _render_content(self) -> None:
        """Render the actual content (override in subclass)."""
        # Header - use display_number which matches the button the user clicked
        with ui.row().classes('items-center gap-2 mb-2'):
            ui.label(f'Step {self.display_number}').style(
                f'color: {THEME_PRIMARY}; font-family: JetBrains Mono; '
                f'font-size: 0.8rem; font-weight: bold;'
            )
            ui.label(self.step_name).style(
                f'color: {THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.75rem;'
            )
        
        # Controls
        self._render_controls()
        
        # Main visualization
        self._render_visualization()
    
    def _render_controls(self) -> None:
        """Render control widgets (override in subclass)."""
        pass
    
    @abstractmethod
    def _render_visualization(self) -> None:
        """Render the main visualization (must override)."""
        pass
    
    def _render_no_data_message(self, message: str = "No data available") -> None:
        """Render a message when no data is available."""
        with ui.column().classes('items-center justify-center p-4'):
            ui.icon('info', size='lg').style(f'color: {THEME_TEXT_DIM};')
            ui.label(message).style(
                f'color: {THEME_TEXT_DIM}; font-size: 0.8rem; text-align: center;'
            )
            ui.label(f'Run step {self.step_number} to generate data').style(
                f'color: {THEME_TEXT_DIM}; font-size: 0.7rem;'
            )
    
    def _render_stats_card(self, stats: Dict[str, Any], title: str = "Stats") -> None:
        """Render a compact stats card."""
        with ui.card().classes('p-2').style(f'background: {THEME_CARD}; border: 1px solid {THEME_BORDER};'):
            ui.label(title).style(
                f'color: {THEME_SECONDARY}; font-family: JetBrains Mono; '
                f'font-size: 0.7rem; font-weight: bold; margin-bottom: 4px;'
            )
            for key, value in stats.items():
                with ui.row().classes('items-center gap-2'):
                    ui.label(f'{key}:').style(
                        f'color: {THEME_TEXT_DIM}; font-size: 0.65rem;'
                    )
                    if isinstance(value, float):
                        display_val = f'{value:.4f}'
                    else:
                        display_val = str(value)
                    ui.label(display_val).style(
                        f'color: {THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.7rem;'
                    )
    
    def create_matplotlib_figure(self, figsize=(6, 4), dpi=100):
        """Create a matplotlib figure."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        fig.patch.set_facecolor('#0a0a0a')
        ax.set_facecolor('#0a0a0a')
        ax.tick_params(colors='#888888')
        for spine in ax.spines.values():
            spine.set_color('#333333')
        return fig, ax
    
    def show_matplotlib(self, fig) -> None:
        """Display matplotlib figure in NiceGUI."""
        import io
        import base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', facecolor=fig.get_facecolor(), 
                    edgecolor='none', bbox_inches='tight')
        buf.seek(0)
        img_data = base64.b64encode(buf.read()).decode('utf-8')
        # Use ui.image with base64 data URI instead of ui.html to avoid sanitize issues
        ui.image(f'data:image/png;base64,{img_data}').style('max-width:100%; height:auto;')
        import matplotlib.pyplot as plt
        plt.close(fig)

