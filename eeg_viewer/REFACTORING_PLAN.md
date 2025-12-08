# EEG Viewer - Plan de Refactorización

> **Documento para LLM**: Este documento describe la estructura actual, problemas identificados, y un plan detallado de refactorización siguiendo principios SOLID y patrones de diseño.

---

## 📊 ANÁLISIS DEL ESTADO ACTUAL

### Métricas del archivo `main.py`

| Métrica | Valor | Problema |
|---------|-------|----------|
| Líneas de código | 5094 | CRÍTICO - Monolito inmanejable |
| Clases de estado global | 4 | Alto acoplamiento |
| Funciones de nivel superior | ~80 | Falta organización |
| Páginas UI en un archivo | 5 | Violación de SRP |
| Funciones anidadas | ~100+ | Difícil testing |

### Estructura Actual Identificada

```
main.py (5094 líneas)
├── [1-26]     Imports
├── [27-76]    ELECTRODE_POSITIONS + clase State (estado global)
├── [78-104]   Funciones de procesamiento de señales
├── [106-238]  STYLE (CSS embebido)
├── [240-362]  Creadores de figuras Plotly (make_*_fig)
├── [364-643]  Funciones update_* para EEG1 y EEG2 (duplicación)
├── [645-731]  Gestión de canales y UI relacionada
├── [733-791]  Navegación y Epochs
├── [793-819]  PipelineState (otro estado global)
├── [820-932]  Funciones helper del Pipeline
├── [933-2110] @ui.page('/pipeline') - Pipeline Page completa
├── [2112-2137] ModelState (otro estado global)
├── [2139-2203] AnalysisState (otro estado global)
├── [2205-2516] Funciones de detección y configuración de datasets
├── [2518-3196] @ui.page('/model') - Model Page completa
├── [3198-3907] Funciones de carga de modelos y procesamiento
├── [3910-4873] @ui.page('/analysis') - Analysis Page completa
├── [4875-5084] @ui.page('/') - Main Page con navegación
└── [5086-5094] Entry point
```

---

## 🚨 PROBLEMAS IDENTIFICADOS

### 1. Violación del Single Responsibility Principle (SRP)
**Severidad: CRÍTICA**

El archivo `main.py` maneja:
- Estado de 5 páginas diferentes
- Procesamiento de señales EEG
- Creación de figuras Plotly
- Lógica de UI de NiceGUI
- Estilos CSS
- Carga de modelos ML
- Gestión de pipelines
- Análisis de datos

**Impacto**: Imposible de mantener, testear o extender.

### 2. Estado Global Múltiple
**Severidad: ALTA**

```python
S = State()        # Viewer state
PS = PipelineState()  # Pipeline state
MS = ModelState()     # Model training state
AS = AnalysisState()  # Analysis state
```

**Problemas**:
- Estado mutable global = bugs impredecibles
- Imposible testing unitario
- Race conditions potenciales
- Memoria no liberada

### 3. Código Duplicado Extensivo
**Severidad: ALTA**

Ejemplo de duplicación EEG1/EEG2:

```python
# 200+ líneas duplicadas con mínimas diferencias
def update_eeg():    # ~50 líneas
def update_eeg2():   # ~50 líneas casi idénticas
def update_fft():    # ~30 líneas
def update_fft2():   # ~30 líneas casi idénticas
def update_hilbert():
def update_hilbert2():
def update_brain():
def update_brain2():
```

### 4. Funciones Anidadas Profundas
**Severidad: MEDIA-ALTA**

Dentro de `@ui.page('/pipeline')` hay ~50 funciones anidadas definidas dentro del scope de la página, haciendo imposible:
- Reutilizarlas
- Testearlas
- Importarlas en otros módulos

### 5. Configuración Hardcodeada
**Severidad: MEDIA**

- CSS inline de 130+ líneas
- Colores, tamaños, rutas dispersos
- Constantes mágicas por todo el código

### 6. Falta de Abstracción
**Severidad: MEDIA**

No hay interfaces ni clases base. Ejemplo:
- Todas las figuras se crean con funciones separadas sin clase base
- No hay patrón Factory para figuras
- No hay Strategy para procesamiento de señales

---

## 🎯 PLAN DE REFACTORIZACIÓN

### Nueva Estructura de Directorios

```
eeg_viewer/
├── main.py                          # Entry point mínimo (~30 líneas)
│
├── app/
│   ├── __init__.py                  # Inicialización de la app
│   │
│   ├── state/                       # Gestión de estado
│   │   ├── __init__.py
│   │   ├── base.py                  # BaseState con patrón Observer
│   │   ├── viewer_state.py          # ViewerState (antes State)
│   │   ├── pipeline_state.py        # PipelineState
│   │   ├── model_state.py           # ModelState
│   │   └── analysis_state.py        # AnalysisState
│   │
│   ├── core/                        # Lógica de negocio (sin UI)
│   │   ├── __init__.py
│   │   ├── signal/
│   │   │   ├── __init__.py
│   │   │   ├── filters.py           # apply_notch, apply_bandpass
│   │   │   ├── transforms.py        # compute_fft, compute_hilbert
│   │   │   └── processing.py        # process_data
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   ├── dataset_detector.py  # detect_dataset_type
│   │   │   ├── model_loader.py      # load_trained_model, etc.
│   │   │   └── epoch_generator.py   # gen_epochs, save_epochs
│   │   └── pipeline/
│   │       ├── __init__.py
│   │       ├── runner.py            # run_pipeline_step
│   │       └── config_generator.py  # create_default_config
│   │
│   ├── visualization/               # Todo relacionado a gráficos
│   │   ├── __init__.py
│   │   ├── styles/
│   │   │   ├── __init__.py
│   │   │   ├── theme.py             # Variables de tema
│   │   │   └── css.py               # CSS generado
│   │   ├── figures/
│   │   │   ├── __init__.py
│   │   │   ├── base.py              # BaseFigure (Abstract Factory)
│   │   │   ├── eeg_figure.py        # EEGFigure
│   │   │   ├── fft_figure.py        # FFTFigure
│   │   │   ├── hilbert_figure.py    # HilbertFigure
│   │   │   ├── brain_figure.py      # BrainFigure
│   │   │   ├── loss_figure.py       # LossFigure
│   │   │   └── latent_figure.py     # LatentSpaceFigure
│   │   └── updaters/
│   │       ├── __init__.py
│   │       ├── base.py              # BaseUpdater
│   │       ├── eeg_updater.py       # EEGUpdater (maneja EEG1 y EEG2)
│   │       ├── fft_updater.py
│   │       └── ...
│   │
│   ├── pages/                       # Páginas de la aplicación
│   │   ├── __init__.py
│   │   ├── base_page.py             # BasePage con header común
│   │   ├── viewer/
│   │   │   ├── __init__.py
│   │   │   ├── page.py              # ViewerPage
│   │   │   └── components/
│   │   │       ├── file_browser.py
│   │   │       ├── channel_selector.py
│   │   │       ├── filter_panel.py
│   │   │       ├── navigation.py
│   │   │       └── epoch_panel.py
│   │   ├── pipeline/
│   │   │   ├── __init__.py
│   │   │   ├── page.py              # PipelinePage
│   │   │   └── components/
│   │   │       ├── step_card.py     # Componente reutilizable
│   │   │       ├── console_tab.py
│   │   │       ├── files_tab.py
│   │   │       └── system_tab.py
│   │   ├── model/
│   │   │   ├── __init__.py
│   │   │   ├── page.py              # ModelPage
│   │   │   └── components/
│   │   │       ├── dataset_config.py
│   │   │       ├── model_config.py
│   │   │       └── training_controls.py
│   │   ├── analysis/
│   │   │   ├── __init__.py
│   │   │   ├── page.py              # AnalysisPage
│   │   │   └── components/
│   │   │       ├── model_loader.py
│   │   │       ├── playback_controls.py
│   │   │       └── activation_viewer.py
│   │   └── components/              # Componentes compartidos
│   │       ├── __init__.py
│   │       ├── header.py            # HeaderComponent
│   │       ├── dark_card.py         # DarkCard widget
│   │       └── terminal_label.py    # TerminalLabel widget
│   │
│   └── utils/
│       ├── __init__.py
│       ├── constants.py             # ELECTRODE_POSITIONS, FREQ_BANDS
│       └── async_helpers.py         # Helpers para async
│
├── config.py                        # Configuración (existente)
├── eeg_loader.py                    # Loader de EEG (existente)
└── cleaning/                        # Módulo de limpieza (existente)
```

---

## 📋 MÓDULOS A CREAR (Detallado)

### 1. `app/state/base.py`
**Propósito**: Clase base para estado con patrón Observer

```python
"""
Base state management with Observer pattern.
Allows UI components to subscribe to state changes.
"""
from abc import ABC
from typing import Callable, Dict, List, Any
from dataclasses import dataclass, field

class BaseState(ABC):
    """Base class for application state management."""
    
    def __init__(self):
        self._observers: Dict[str, List[Callable]] = {}
    
    def subscribe(self, property_name: str, callback: Callable) -> None:
        """Subscribe to changes in a specific property."""
        if property_name not in self._observers:
            self._observers[property_name] = []
        self._observers[property_name].append(callback)
    
    def notify(self, property_name: str, value: Any) -> None:
        """Notify all observers of a property change."""
        for callback in self._observers.get(property_name, []):
            callback(value)
    
    def set_property(self, name: str, value: Any) -> None:
        """Set a property and notify observers."""
        setattr(self, f'_{name}', value)
        self.notify(name, value)
```

### 2. `app/state/viewer_state.py`
**Propósito**: Estado del viewer principal

```python
"""
State management for the main EEG viewer page.
Replaces the global State() class.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from .base import BaseState

@dataclass
class EEGSlot:
    """Represents a loaded EEG file slot."""
    data: Optional[Any] = None
    filename: str = ""
    is_loaded: bool = False

class ViewerState(BaseState):
    """State for the EEG viewer page."""
    
    def __init__(self):
        super().__init__()
        # EEG data slots
        self.eeg_slot1 = EEGSlot()
        self.eeg_slot2 = EEGSlot()
        self.compare_mode: bool = False
        
        # Channel selection
        self.selected_channels: List[str] = []
        self.current_amplitudes: Dict[str, float] = {}
        
        # View settings
        self.view_start: float = 0.0
        self.view_duration: float = 5.0
        self.scale_factor: float = 1.0
        self.is_playing: bool = False
        
        # Filters
        self.notch_enabled: bool = False
        self.notch_freq: float = 50.0
        self.bandpass_enabled: bool = False
        self.bandpass_low: float = 1.0
        self.bandpass_high: float = 45.0
        
        # Hilbert
        self.hilbert_channel: str = ""
        
        # Epochs
        self.epoch_duration: float = 2.0
        self.epochs: List[Dict] = []
    
    @property
    def primary_eeg(self) -> Optional[Any]:
        """Get the primary loaded EEG data."""
        return self.eeg_slot1.data
    
    @property
    def secondary_eeg(self) -> Optional[Any]:
        """Get the secondary EEG data for comparison."""
        return self.eeg_slot2.data if self.compare_mode else None
    
    def load_eeg(self, data: Any, slot: int = 1) -> None:
        """Load EEG data into specified slot."""
        target = self.eeg_slot1 if slot == 1 else self.eeg_slot2
        target.data = data
        target.filename = getattr(data, 'filename', 'unknown')
        target.is_loaded = True
        self.notify('eeg_loaded', slot)
    
    def toggle_compare_mode(self) -> None:
        """Toggle comparison mode."""
        self.compare_mode = not self.compare_mode
        self.notify('compare_mode', self.compare_mode)
```

### 3. `app/core/signal/filters.py`
**Propósito**: Funciones de filtrado de señales (puras, sin estado)

```python
"""
Signal filtering functions for EEG data.
All functions are pure - no side effects, no global state.
"""
import numpy as np
from numpy.typing import NDArray
from scipy import signal

def apply_notch_filter(
    data: NDArray[np.float64],
    sfreq: float,
    freq: float = 50.0,
    quality: float = 30.0
) -> NDArray[np.float64]:
    """
    Apply notch filter to remove line noise.
    
    Args:
        data: EEG data array (channels x samples) or (samples,)
        sfreq: Sampling frequency in Hz
        freq: Frequency to notch out (default 50Hz for EU, use 60Hz for US)
        quality: Quality factor of the filter
    
    Returns:
        Filtered data with same shape as input
    """
    b, a = signal.iirnotch(freq, quality, sfreq)
    return signal.filtfilt(b, a, data, axis=-1)


def apply_bandpass_filter(
    data: NDArray[np.float64],
    sfreq: float,
    low_freq: float,
    high_freq: float,
    order: int = 4
) -> NDArray[np.float64]:
    """
    Apply bandpass filter to EEG data.
    
    Args:
        data: EEG data array
        sfreq: Sampling frequency in Hz
        low_freq: Low cutoff frequency
        high_freq: High cutoff frequency
        order: Filter order (default 4)
    
    Returns:
        Bandpass filtered data
    """
    nyquist = sfreq / 2
    low = max(low_freq / nyquist, 0.001)
    high = min(high_freq / nyquist, 0.999)
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data, axis=-1)


def apply_highpass_filter(
    data: NDArray[np.float64],
    sfreq: float,
    cutoff: float,
    order: int = 4
) -> NDArray[np.float64]:
    """Apply highpass filter."""
    nyquist = sfreq / 2
    normalized_cutoff = max(cutoff / nyquist, 0.001)
    b, a = signal.butter(order, normalized_cutoff, btype='high')
    return signal.filtfilt(b, a, data, axis=-1)


def apply_lowpass_filter(
    data: NDArray[np.float64],
    sfreq: float,
    cutoff: float,
    order: int = 4
) -> NDArray[np.float64]:
    """Apply lowpass filter."""
    nyquist = sfreq / 2
    normalized_cutoff = min(cutoff / nyquist, 0.999)
    b, a = signal.butter(order, normalized_cutoff, btype='low')
    return signal.filtfilt(b, a, data, axis=-1)
```

### 4. `app/core/signal/transforms.py`
**Propósito**: Transformaciones de señales (FFT, Hilbert)

```python
"""
Signal transformation functions.
"""
import numpy as np
from numpy.typing import NDArray
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import Tuple

def compute_fft(
    data: NDArray[np.float64],
    sfreq: float
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute FFT of signal data.
    
    Args:
        data: Signal data (samples,) or (channels, samples)
        sfreq: Sampling frequency
    
    Returns:
        Tuple of (frequencies, magnitudes)
    """
    n = data.shape[-1]
    freqs = fftfreq(n, 1/sfreq)[:n//2]
    fft_vals = np.abs(fft(data, axis=-1))[..., :n//2] / n * 2
    return freqs, fft_vals


def compute_hilbert_transform(
    data: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute Hilbert transform for envelope and phase extraction.
    
    Args:
        data: Signal data
    
    Returns:
        Tuple of (amplitude_envelope, instantaneous_phase)
    """
    analytic = signal.hilbert(data, axis=-1)
    amplitude = np.abs(analytic)
    phase = np.angle(analytic)
    return amplitude, phase


def compute_power_spectral_density(
    data: NDArray[np.float64],
    sfreq: float,
    nperseg: int = 256
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute PSD using Welch's method."""
    freqs, psd = signal.welch(data, sfreq, nperseg=nperseg, axis=-1)
    return freqs, psd
```

### 5. `app/visualization/figures/base.py`
**Propósito**: Factory abstracta para figuras

```python
"""
Abstract base class for Plotly figures.
Implements Factory pattern for consistent figure creation.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import plotly.graph_objects as go

class BaseFigure(ABC):
    """Abstract base for all EEG visualization figures."""
    
    # Default plot styling
    PLOT_BG = 'rgba(8,8,8,1)'
    GRID_COLOR = 'rgba(0,255,136,0.08)'
    GRID_COLOR_MINOR = 'rgba(0,255,136,0.03)'
    FONT_FAMILY = 'JetBrains Mono, monospace'
    
    def __init__(self, height: int = 250, use_secondary_style: bool = False):
        """
        Initialize figure.
        
        Args:
            height: Figure height in pixels
            use_secondary_style: If True, use secondary color scheme (for comparison)
        """
        self.height = height
        self.use_secondary_style = use_secondary_style
        self._figure: Optional[go.Figure] = None
    
    @property
    @abstractmethod
    def primary_color(self) -> str:
        """Primary accent color for this figure type."""
        pass
    
    @property
    def accent_color(self) -> str:
        """Get accent color based on style mode."""
        return '#f472b6' if self.use_secondary_style else self.primary_color
    
    @abstractmethod
    def create(self) -> go.Figure:
        """Create and return the figure."""
        pass
    
    @abstractmethod
    def update(self, data: Dict[str, Any]) -> None:
        """Update figure with new data."""
        pass
    
    def get_base_layout(self) -> Dict[str, Any]:
        """Get base layout configuration."""
        return {
            'template': 'plotly_dark',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': self.PLOT_BG,
            'height': self.height,
            'font': {
                'family': self.FONT_FAMILY,
                'size': 10,
                'color': '#e8e8e8'
            },
            'hovermode': 'x unified',
            'hoverlabel': {
                'bgcolor': '#1a1a1a',
                'font': {'family': self.FONT_FAMILY, 'size': 10}
            }
        }
    
    @property
    def figure(self) -> go.Figure:
        """Get or create the figure."""
        if self._figure is None:
            self._figure = self.create()
        return self._figure
```

### 6. `app/visualization/figures/eeg_figure.py`
**Propósito**: Figura de EEG

```python
"""
EEG time series figure.
"""
import numpy as np
from typing import Dict, Any, List
import plotly.graph_objects as go
from .base import BaseFigure

class EEGFigure(BaseFigure):
    """Figure for displaying EEG time series."""
    
    # Signal colors palette
    SIGNAL_COLORS = [
        '#00ff88', '#00d4ff', '#ffcc00', '#ff6b9d', '#a78bfa',
        '#00ffcc', '#ff9f43', '#74b9ff', '#55efc4', '#fd79a8'
    ]
    
    @property
    def primary_color(self) -> str:
        return '#00ff88'
    
    def create(self) -> go.Figure:
        """Create empty EEG figure with proper layout."""
        fig = go.Figure()
        
        layout = self.get_base_layout()
        layout.update({
            'margin': {'l': 70, 'r': 10, 't': 10, 'b': 50},
            'xaxis': {
                'title': {'text': 'TIME [s]', 'font': {'size': 9, 'color': self.accent_color}},
                'gridcolor': self.GRID_COLOR,
                'zerolinecolor': self.GRID_COLOR,
                'tickfont': {'size': 9, 'color': '#999'},
                'fixedrange': False
            },
            'yaxis': {
                'gridcolor': self.GRID_COLOR_MINOR,
                'tickfont': {'size': 9, 'color': self.accent_color},
                'fixedrange': True
            },
            'showlegend': False
        })
        
        fig.update_layout(**layout)
        self._figure = fig
        return fig
    
    def update(
        self,
        times: np.ndarray,
        data: np.ndarray,
        channel_names: List[str],
        scale_factor: float = 1.0
    ) -> None:
        """
        Update EEG figure with new data.
        
        Args:
            times: Time array
            data: EEG data (channels x samples) in µV
            channel_names: List of channel names
            scale_factor: Vertical scale factor
        """
        n_channels = len(channel_names)
        spacing = 2.0 * scale_factor
        
        # Normalize each channel
        normalized = np.zeros_like(data)
        for i in range(n_channels):
            std = np.std(data[i])
            normalized[i] = data[i] / (std * 3) if std > 0 else data[i]
        
        # Clear existing traces
        self.figure.data = []
        
        # Add traces for each channel
        colors = self.SIGNAL_COLORS if not self.use_secondary_style else [
            '#f472b6', '#fb7185', '#fda4af', '#fecdd3', '#ffe4e6'
        ]
        
        for i in range(n_channels):
            offset = (n_channels - 1 - i) * spacing
            self.figure.add_trace(go.Scatter(
                x=times,
                y=normalized[i] + offset,
                name=channel_names[i],
                line={'color': colors[i % len(colors)], 'width': 1},
                hovertemplate=f'{channel_names[i]}: %{{customdata:.1f}} µV<extra></extra>',
                customdata=data[i]
            ))
        
        # Update y-axis
        y_min = -spacing
        y_max = n_channels * spacing
        self.figure.update_layout(
            yaxis={
                'tickmode': 'array',
                'tickvals': [(n_channels - 1 - i) * spacing for i in range(n_channels)],
                'ticktext': channel_names,
                'range': [y_min, y_max],
                'fixedrange': True
            }
        )
```

### 7. `app/pages/components/header.py`
**Propósito**: Header reutilizable con navegación

```python
"""
Reusable header component with navigation.
"""
from nicegui import ui
from typing import Optional

class HeaderComponent:
    """Header with navigation for all pages."""
    
    PAGES = [
        ('VIEWER', '/'),
        ('CLEANER', '/cleaner'),
        ('PIPELINE', '/pipeline'),
        ('MODEL', '/model'),
        ('ANALYSIS', '/analysis'),
    ]
    
    def __init__(
        self,
        current_page: str,
        subtitle: Optional[str] = None,
        subtitle_color: str = '#00ff88'
    ):
        """
        Create header component.
        
        Args:
            current_page: Name of current page (for highlighting)
            subtitle: Optional subtitle text
            subtitle_color: Color for subtitle
        """
        self.current_page = current_page
        self.subtitle = subtitle
        self.subtitle_color = subtitle_color
    
    def render(self) -> None:
        """Render the header."""
        with ui.header().classes('items-center px-4 py-1').style(
            'background: #0a0a0a; border-bottom: 1px solid #1a1a1a;'
        ):
            # Logo
            ui.label('▶').style(
                'color: #00ff88; font-family: JetBrains Mono; '
                'font-size: 0.75rem; letter-spacing: 2px;'
            )
            ui.label('EEG_VIEWER').classes('text-base font-medium ml-2').style(
                'color: #00ff88; font-family: JetBrains Mono; letter-spacing: 1px;'
            )
            ui.label('v1.0').classes('text-xs ml-2').style(
                'color: #666; font-family: JetBrains Mono;'
            )
            
            # Subtitle
            if self.subtitle:
                ui.label(f'// {self.subtitle}').classes('text-sm ml-3').style(
                    f'color: {self.subtitle_color}; font-family: JetBrains Mono;'
                )
            
            # Navigation
            with ui.row().classes('ml-auto gap-2'):
                for name, path in self.PAGES:
                    is_current = name == self.current_page
                    color = '#00ff88' if is_current else '#666'
                    ui.button(
                        name,
                        on_click=lambda p=path: ui.navigate.to(p)
                    ).props('flat dense').style(f'color: {color};')
```

### 8. `app/pages/viewer/page.py`
**Propósito**: Página principal del viewer

```python
"""
Main EEG Viewer page.
"""
from nicegui import ui
from ...state.viewer_state import ViewerState
from ...visualization.figures.eeg_figure import EEGFigure
from ...visualization.figures.fft_figure import FFTFigure
from ...visualization.figures.hilbert_figure import HilbertFigure
from ...visualization.figures.brain_figure import BrainFigure
from ..components.header import HeaderComponent
from .components.file_browser import FileBrowserComponent
from .components.channel_selector import ChannelSelectorComponent
from .components.filter_panel import FilterPanelComponent
from .components.navigation import NavigationComponent

class ViewerPage:
    """Main EEG viewer page."""
    
    def __init__(self, state: ViewerState):
        """
        Initialize viewer page.
        
        Args:
            state: ViewerState instance
        """
        self.state = state
        
        # Create figures
        self.eeg_fig1 = EEGFigure()
        self.eeg_fig2 = EEGFigure(use_secondary_style=True)
        self.fft_fig1 = FFTFigure()
        self.fft_fig2 = FFTFigure(use_secondary_style=True)
        self.hilbert_fig1 = HilbertFigure()
        self.hilbert_fig2 = HilbertFigure(use_secondary_style=True)
        self.brain_fig1 = BrainFigure()
        self.brain_fig2 = BrainFigure(use_secondary_style=True)
        
        # UI references
        self.eeg_plot1 = None
        self.eeg_plot2 = None
    
    def render(self) -> None:
        """Render the viewer page."""
        # Header
        HeaderComponent('VIEWER').render()
        
        # Main layout
        with ui.row().classes('w-full p-4 gap-4').style('min-height: calc(100vh - 50px);'):
            # Left sidebar
            self._render_sidebar()
            
            # Main content
            self._render_main_content()
    
    def _render_sidebar(self) -> None:
        """Render left sidebar."""
        with ui.column().classes('gap-4 shrink-0').style('width: 380px;'):
            FileBrowserComponent(self.state).render()
            ChannelSelectorComponent(self.state).render()
    
    def _render_main_content(self) -> None:
        """Render main content area."""
        with ui.column().classes('flex-1 gap-3'):
            FilterPanelComponent(self.state).render()
            self._render_eeg_row()
            NavigationComponent(self.state).render()
            self._render_topography_row()
            self._render_fft_row()
            self._render_hilbert_row()
    
    def _render_eeg_row(self) -> None:
        """Render EEG plots row."""
        with ui.row().classes('gap-3 w-full'):
            # EEG 1
            with ui.card().classes('dark-card p-3 flex-1'):
                self._render_eeg_header('EEG 1', '#00ff88')
                self.eeg_plot1 = ui.plotly(self.eeg_fig1.figure).classes('w-full')
            
            # EEG 2 (conditional)
            with ui.card().classes('dark-card p-3 flex-1').bind_visibility_from(
                self.state, 'compare_mode'
            ):
                self._render_eeg_header('EEG 2', '#f472b6')
                self.eeg_plot2 = ui.plotly(self.eeg_fig2.figure).classes('w-full')
    
    def _render_eeg_header(self, title: str, color: str) -> None:
        """Render EEG card header with scale controls."""
        with ui.row().classes('items-center justify-between mb-1'):
            ui.label(f'▌{title}').style(
                f'color: {color}; font-family: JetBrains Mono; '
                'font-size: 0.8rem; letter-spacing: 1px;'
            )
            with ui.row().classes('gap-1'):
                ui.button('-', on_click=lambda: self._scale(-1)).props('dense flat size=xs')
                ui.button('1x', on_click=lambda: self._scale(0)).props('dense flat size=xs')
                ui.button('+', on_click=lambda: self._scale(1)).props('dense flat size=xs')
    
    def _scale(self, direction: int) -> None:
        """Adjust scale factor."""
        if direction < 0:
            self.state.scale_factor = max(0.2, self.state.scale_factor * 0.7)
        elif direction > 0:
            self.state.scale_factor = min(5.0, self.state.scale_factor * 1.4)
        else:
            self.state.scale_factor = 1.0
        self.update_eeg()
    
    def update_eeg(self) -> None:
        """Update EEG displays."""
        # Implementation...
        pass
```

### 9. `app/core/signal/processing.py`
**Propósito**: Función principal que combina filtros

```python
"""
High-level signal processing that combines filters.
"""
import numpy as np
from numpy.typing import NDArray
from typing import Optional
from .filters import apply_notch_filter, apply_bandpass_filter

def process_eeg_data(
    data: NDArray[np.float64],
    sfreq: float,
    notch_enabled: bool = False,
    notch_freq: float = 50.0,
    bandpass_enabled: bool = False,
    bandpass_low: float = 1.0,
    bandpass_high: float = 45.0
) -> NDArray[np.float64]:
    """
    Apply configured filters to EEG data.
    
    Args:
        data: Raw EEG data (channels x samples)
        sfreq: Sampling frequency
        notch_enabled: Whether to apply notch filter
        notch_freq: Notch filter frequency
        bandpass_enabled: Whether to apply bandpass filter
        bandpass_low: Low cutoff frequency
        bandpass_high: High cutoff frequency
    
    Returns:
        Filtered data with same shape as input
    """
    result = data.copy()
    
    if notch_enabled:
        result = apply_notch_filter(result, sfreq, notch_freq)
    
    if bandpass_enabled:
        result = apply_bandpass_filter(result, sfreq, bandpass_low, bandpass_high)
    
    return result
```

### 10. `app/visualization/styles/theme.py`
**Propósito**: Constantes de tema centralizadas

```python
"""
Theme constants for consistent styling across the application.
"""

# Colors
THEME_BG = '#0a0a0a'
THEME_CARD = '#121212'
THEME_BORDER = '#1a1a1a'
THEME_PRIMARY = '#00ff88'      # Green
THEME_SECONDARY = '#00d4ff'    # Cyan
THEME_WARN = '#ffcc00'         # Yellow
THEME_ERROR = '#ff4444'        # Red
THEME_TEXT = '#e8e8e8'
THEME_TEXT_DIM = '#666666'

# Signal colors for multi-channel display
SIGNAL_COLORS = [
    '#00ff88', '#00d4ff', '#ffcc00', '#ff6b9d', '#a78bfa',
    '#00ffcc', '#ff9f43', '#74b9ff', '#55efc4', '#fd79a8'
]

# Secondary palette (for EEG 2 in compare mode)
SIGNAL_COLORS_SECONDARY = [
    '#f472b6', '#fb7185', '#fda4af', '#fecdd3', '#ffe4e6'
]

# Plot constants
PLOT_BG = 'rgba(8,8,8,1)'
PLOT_GRID = 'rgba(0,255,136,0.08)'
PLOT_GRID_MINOR = 'rgba(0,255,136,0.03)'

# Font
FONT_FAMILY = 'JetBrains Mono, monospace'
FONT_FAMILY_FALLBACK = 'IBM Plex Mono, SF Mono, monospace'
```

### 11. `app/visualization/styles/css.py`
**Propósito**: CSS generado desde el tema

```python
"""
Generate CSS from theme constants.
"""
from .theme import (
    THEME_BG, THEME_CARD, THEME_BORDER, THEME_PRIMARY,
    THEME_SECONDARY, THEME_TEXT, THEME_TEXT_DIM, FONT_FAMILY
)

def generate_style() -> str:
    """Generate the complete CSS stylesheet."""
    return f"""
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');

:root {{
    --bg: {THEME_BG};
    --card: {THEME_CARD};
    --border: {THEME_BORDER};
    --primary: {THEME_PRIMARY};
    --secondary: {THEME_SECONDARY};
    --text: {THEME_TEXT};
    --text-dim: {THEME_TEXT_DIM};
}}

* {{ scrollbar-width: thin; scrollbar-color: var(--primary) var(--bg); }}
*::-webkit-scrollbar {{ width: 6px; height: 6px; }}
*::-webkit-scrollbar-track {{ background: var(--bg); }}
*::-webkit-scrollbar-thumb {{ background: var(--primary); border-radius: 3px; }}

body {{
    background: var(--bg) !important;
    font-family: {FONT_FAMILY} !important;
    color: var(--text) !important;
}}

.dark-card {{
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4) !important;
}}

.terminal-header {{
    font-size: 0.7rem;
    color: var(--text-dim);
    letter-spacing: 0.5px;
    text-transform: uppercase;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 8px;
}}

.ch-btn {{
    font-size: 0.75rem !important;
    font-family: {FONT_FAMILY} !important;
    padding: 3px 8px !important;
    min-width: 42px !important;
    margin: 2px !important;
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--text-dim) !important;
    border-radius: 2px !important;
}}
.ch-btn:hover {{ border-color: var(--primary) !important; color: var(--primary) !important; }}
.ch-sel {{ background: rgba(0, 255, 136, 0.15) !important; border-color: var(--primary) !important; color: var(--primary) !important; }}
"""

# Pre-generated style for import
STYLE = generate_style()
```

### 12. `app/state/pipeline_state.py`
**Propósito**: Estado del pipeline (extraído de main.py líneas 798-818)

```python
"""
State management for the Pipeline page.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Any
from datetime import datetime
from pathlib import Path
from .base import BaseState

@dataclass
class PipelineConfig:
    """Pipeline execution parameters."""
    max_subjects: int = 0  # 0 = all
    conditions: List[str] = field(default_factory=lambda: ["DMT", "EC", "EO"])
    jobs: int = 0  # 0 = auto
    workers: int = 7
    bands: List[str] = field(default_factory=lambda: ["Delta", "Theta", "Alpha", "Beta", "Gamma"])
    min_k: int = 2
    max_k: int = 15
    min_comps: int = 2
    max_comps: int = 10

class PipelineState(BaseState):
    """State for the pipeline execution page."""
    
    def __init__(self):
        super().__init__()
        self.running: bool = False
        self.current_step: str = ""
        self.start_time: Optional[datetime] = None
        self.current_process: Optional[Any] = None
        self.progress: int = 0
        self.config = PipelineConfig()
        
        # UI state
        self.current_run_dir: Optional[Path] = None
        self.log_messages: List[str] = []
    
    def start_step(self, step_name: str) -> None:
        """Start a pipeline step."""
        self.running = True
        self.current_step = step_name
        self.start_time = datetime.now()
        self.notify('step_started', step_name)
    
    def finish_step(self, success: bool = True) -> None:
        """Finish current pipeline step."""
        self.running = False
        step = self.current_step
        self.current_step = ""
        self.start_time = None
        self.notify('step_finished', {'step': step, 'success': success})
    
    def add_log(self, message: str) -> None:
        """Add message to log."""
        self.log_messages.append(message)
        self.notify('log_added', message)
    
    def clear_log(self) -> None:
        """Clear all log messages."""
        self.log_messages = []
        self.notify('log_cleared', None)
```

### 13. `app/utils/constants.py`
**Propósito**: Constantes globales (extraído de main.py)

```python
"""
Global constants for the EEG viewer application.
"""

# Electrode positions (10-20 system) - for topography plots
ELECTRODE_POSITIONS = {
    'Fp1': (-0.3, 0.9), 'Fp2': (0.3, 0.9), 'Fpz': (0.0, 0.95),
    'F7': (-0.7, 0.6), 'F3': (-0.35, 0.6), 'Fz': (0.0, 0.6), 
    'F4': (0.35, 0.6), 'F8': (0.7, 0.6),
    'FC1': (-0.2, 0.4), 'FC2': (0.2, 0.4), 
    'FC5': (-0.55, 0.4), 'FC6': (0.55, 0.4),
    'T7': (-0.85, 0.2), 'C3': (-0.4, 0.2), 'Cz': (0.0, 0.2), 
    'C4': (0.4, 0.2), 'T8': (0.85, 0.2),
    'T3': (-0.85, 0.2), 'T4': (0.85, 0.2),  # Alternative names
    'CP1': (-0.2, 0.0), 'CP2': (0.2, 0.0), 'CPz': (0.0, 0.0), 
    'CP5': (-0.55, 0.0), 'CP6': (0.55, 0.0),
    'P7': (-0.7, -0.3), 'P3': (-0.35, -0.3), 'Pz': (0.0, -0.3), 
    'P4': (0.35, -0.3), 'P8': (0.7, -0.3),
    'T5': (-0.7, -0.3), 'T6': (0.7, -0.3),  # Alternative names
    'O1': (-0.3, -0.7), 'O2': (0.3, -0.7), 'Oz': (0.0, -0.7),
    'M1': (-0.95, 0.0), 'M2': (0.95, 0.0),  # Mastoids
    'A1': (-0.95, 0.0), 'A2': (0.95, 0.0),  # Alternative names
}

# Frequency bands (Hz)
FREQ_BANDS = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 100),
}

# Default paths (relative to eeg_viewer/)
DEFAULT_PATHS = {
    'eeg_raw': '../EEG',
    'eeg_clean': '../EEG_CLEAN',
    'pipeline_outputs': '../pipeline_outputs',
    'results': '../fwd-inv-stc',
}
```

### 14. Mapeo EXACTO: main.py líneas → nuevos módulos

| Líneas en main.py | Destino | Descripción |
|-------------------|---------|-------------|
| 1-26 | Distribuido | Imports van a cada módulo |
| 27-39 | `app/utils/constants.py` | ELECTRODE_POSITIONS |
| 42-76 | `app/state/viewer_state.py` | clase State → ViewerState |
| 79-81 | `app/core/signal/filters.py` | apply_notch |
| 83-86 | `app/core/signal/filters.py` | apply_bandpass |
| 88-92 | `app/core/signal/transforms.py` | compute_fft |
| 94-96 | `app/core/signal/transforms.py` | compute_hilbert |
| 98-104 | `app/core/signal/processing.py` | process_data |
| 107-238 | `app/visualization/styles/css.py` | STYLE |
| 245-272 | `app/visualization/figures/eeg_figure.py` | make_eeg_fig |
| 274-308 | `app/visualization/figures/fft_figure.py` | make_fft_fig |
| 310-331 | `app/visualization/figures/hilbert_figure.py` | make_hilbert_fig |
| 333-362 | `app/visualization/figures/brain_figure.py` | make_brain_fig |
| 365-411 | `app/visualization/updaters/eeg_updater.py` | update_eeg |
| 413-432 | `app/visualization/updaters/fft_updater.py` | update_fft |
| 434-453 | `app/visualization/updaters/hilbert_updater.py` | update_hilbert |
| 455-498 | `app/visualization/updaters/brain_updater.py` | update_brain |
| 500-643 | `app/visualization/updaters/*.py` | update_*2 (DRY: unificar) |
| 646-731 | `app/pages/viewer/components/channel_selector.py` | Channel management |
| 733-763 | `app/pages/viewer/components/navigation.py` | Navigation |
| 766-791 | `app/core/data/epoch_generator.py` | gen_epochs, save_epochs |
| 798-818 | `app/state/pipeline_state.py` | PipelineState |
| 821-931 | `app/core/pipeline/runner.py` | Pipeline helpers |
| 936-2110 | `app/pages/pipeline/page.py` | @ui.page('/pipeline') |
| 2116-2137 | `app/state/model_state.py` | ModelState |
| 2140-2203 | `app/state/analysis_state.py` | AnalysisState |
| 2210-2350 | `app/core/data/dataset_detector.py` | detect_dataset_type |
| 2379-2516 | `app/core/pipeline/config_generator.py` | create_default_config |
| 2519-3196 | `app/pages/model/page.py` | @ui.page('/model') |
| 3210-3387 | `app/core/data/model_loader.py` | load_trained_model |
| 3494-3595 | `app/core/data/model_loader.py` | process_sample |
| 3910-4873 | `app/pages/analysis/page.py` | @ui.page('/analysis') |
| 4876-5084 | `app/pages/viewer/page.py` | main_content() |
| 5086-5094 | `main.py` (nuevo) | Entry point |

---

## 🔄 ORDEN DE IMPLEMENTACIÓN

### Fase 0: PREPARACIÓN (CRÍTICO - Hacer primero)

**Antes de tocar CUALQUIER código:**

1. **Crear backup**:
```bash
cd /media/storage_hdd/dmt_fz/eeg_viewer
cp main.py main_backup_$(date +%Y%m%d).py
```

2. **Crear tests de humo** (`tests/test_smoke.py`):
```python
"""
Smoke tests - Run BEFORE and AFTER refactoring.
These verify the app still works correctly.
"""
import pytest
import numpy as np

class TestSignalProcessing:
    """Verify signal processing functions work."""
    
    def test_apply_notch(self):
        from main import apply_notch
        data = np.random.randn(10, 1000)
        result = apply_notch(data, 256.0, 50.0)
        assert result.shape == data.shape
        assert not np.allclose(result, data)  # Should be different
    
    def test_apply_bandpass(self):
        from main import apply_bandpass
        data = np.random.randn(10, 1000)
        result = apply_bandpass(data, 256.0, 1.0, 45.0)
        assert result.shape == data.shape
    
    def test_compute_fft(self):
        from main import compute_fft
        data = np.random.randn(10, 1000)
        freqs, power = compute_fft(data, 256.0)
        assert len(freqs) > 0
        assert power.shape[0] == 10
    
    def test_compute_hilbert(self):
        from main import compute_hilbert
        data = np.random.randn(1000)
        amp, phase = compute_hilbert(data)
        assert amp.shape == data.shape
        assert phase.shape == data.shape

class TestFigureCreation:
    """Verify Plotly figures can be created."""
    
    def test_eeg_figure(self):
        from main import make_eeg_fig
        fig = make_eeg_fig()
        assert fig is not None
        assert hasattr(fig, 'data')
    
    def test_fft_figure(self):
        from main import make_fft_fig
        fig = make_fft_fig()
        assert fig is not None
    
    def test_hilbert_figure(self):
        from main import make_hilbert_fig
        fig = make_hilbert_fig()
        assert fig is not None
    
    def test_brain_figure(self):
        from main import make_brain_fig
        fig = make_brain_fig()
        assert fig is not None

class TestStateCreation:
    """Verify state objects can be created."""
    
    def test_viewer_state(self):
        from main import State
        s = State()
        assert s.eeg_data is None
        assert s.selected_channels == []
    
    def test_pipeline_state(self):
        from main import PipelineState
        ps = PipelineState()
        assert ps.running == False
    
    def test_model_state(self):
        from main import ModelState
        ms = ModelState()
        assert ms.training == False
    
    def test_analysis_state(self):
        from main import AnalysisState
        ast = AnalysisState()
        assert ast.model is None

class TestImports:
    """Verify all imports work."""
    
    def test_main_imports(self):
        # This should not raise any ImportError
        from main import (
            State, PipelineState, ModelState, AnalysisState,
            apply_notch, apply_bandpass, compute_fft, compute_hilbert,
            process_data, make_eeg_fig, make_fft_fig, make_hilbert_fig,
            make_brain_fig, ELECTRODE_POSITIONS, STYLE
        )
        assert True
    
    def test_config_imports(self):
        from config import (
            THEME_BG, THEME_PRIMARY, THEME_SECONDARY,
            FREQ_BANDS, SIGNAL_COLORS, EEG_RAW_DIR
        )
        assert THEME_PRIMARY == "#00ff88"
```

3. **Ejecutar tests de humo**:
```bash
cd /media/storage_hdd/dmt_fz/eeg_viewer
pytest tests/test_smoke.py -v
# TODOS deben pasar antes de continuar
```

4. **Verificar que la app funciona**:
```bash
./run.sh
# Navegar a http://localhost:8080 y verificar:
# - [ ] Viewer carga
# - [ ] Cleaner carga  
# - [ ] Pipeline carga
# - [ ] Model carga
# - [ ] Analysis carga
# - [ ] Se puede cargar un archivo EEG
# - [ ] Los gráficos se actualizan
```

### Fase 1: Fundamentos (Día 1-2)
1. Crear estructura de directorios (script create_structure.sh)
2. Implementar `app/state/base.py`
3. Implementar `app/state/viewer_state.py` (SIN referencias UI)
4. Mover `app/core/signal/filters.py`
5. Mover `app/core/signal/transforms.py`
6. Mover `app/core/signal/processing.py`
7. Verificar: `pytest tests/test_smoke.py` sigue pasando

### Fase 2: Visualización (Día 3-4)
1. Implementar `app/visualization/styles/theme.py`
2. Implementar `app/visualization/figures/base.py`
3. Implementar todas las figuras específicas
4. Tests para figuras

### Fase 3: Componentes UI (Día 5-6)
1. Implementar `app/pages/components/header.py`
2. Implementar componentes compartidos
3. Implementar componentes del viewer
4. Tests de integración

### Fase 4: Páginas (Día 7-8)
1. Implementar `ViewerPage`
2. Implementar `PipelinePage`
3. Implementar `ModelPage`
4. Implementar `AnalysisPage`

### Fase 5: Integración (Día 9-10)
1. Nuevo `main.py` mínimo
2. Migrar estados restantes
3. Testing completo
4. Documentación

---

## ✓ CHECKLIST DE VERIFICACIÓN POR FASE

### Fase 1 Checklist
```
[ ] Directorio app/ creado con __init__.py
[ ] app/state/base.py implementado y testeado
[ ] app/state/viewer_state.py implementado y testeado
[ ] app/core/signal/filters.py implementado y testeado
[ ] app/core/signal/transforms.py implementado y testeado
[ ] app/core/signal/processing.py implementado y testeado
[ ] Tests pasan: pytest tests/test_signal.py
[ ] Import desde main.py funciona: from app.core.signal import *
```

### Fase 2 Checklist
```
[ ] app/visualization/styles/theme.py creado
[ ] app/visualization/styles/css.py creado
[ ] app/visualization/figures/base.py implementado
[ ] app/visualization/figures/eeg_figure.py implementado
[ ] app/visualization/figures/fft_figure.py implementado
[ ] app/visualization/figures/hilbert_figure.py implementado
[ ] app/visualization/figures/brain_figure.py implementado
[ ] Tests visuales pasan (crear figura, verificar que no explota)
```

### Fase 3 Checklist
```
[ ] app/pages/components/header.py implementado
[ ] app/pages/components/dark_card.py implementado
[ ] app/pages/viewer/components/file_browser.py implementado
[ ] app/pages/viewer/components/channel_selector.py implementado
[ ] app/pages/viewer/components/filter_panel.py implementado
[ ] app/pages/viewer/components/navigation.py implementado
[ ] Cada componente se puede instanciar sin error
```

### Fase 4 Checklist
```
[ ] app/pages/viewer/page.py implementado
[ ] app/pages/pipeline/page.py implementado
[ ] app/pages/model/page.py implementado
[ ] app/pages/analysis/page.py implementado
[ ] Todas las páginas cargan sin error
[ ] Navegación entre páginas funciona
```

### Fase 5 Checklist
```
[ ] main.py nuevo tiene < 50 líneas
[ ] main.py antiguo renombrado a main_legacy.py
[ ] Todas las rutas funcionan: /, /cleaner, /pipeline, /model, /analysis
[ ] No hay estados globales en main.py
[ ] pytest tests/ pasa todos los tests
[ ] La app funciona igual que antes del refactor
```

---

## 📦 ARCHIVOS `__init__.py` REQUERIDOS

### `app/__init__.py`
```python
"""
EEG Viewer Application Package.

This package contains all modules for the EEG viewer application.
"""
from .state import ViewerState, PipelineState, ModelState, AnalysisState

__version__ = "2.0.0"
```

### `app/state/__init__.py`
```python
"""State management modules."""
from .base import BaseState
from .viewer_state import ViewerState, EEGSlot
from .pipeline_state import PipelineState, PipelineConfig
from .model_state import ModelState
from .analysis_state import AnalysisState

__all__ = [
    'BaseState',
    'ViewerState', 'EEGSlot',
    'PipelineState', 'PipelineConfig',
    'ModelState',
    'AnalysisState',
]
```

### `app/core/__init__.py`
```python
"""Core business logic modules (no UI dependencies)."""
from .signal import apply_notch_filter, apply_bandpass_filter, compute_fft, compute_hilbert_transform
```

### `app/core/signal/__init__.py`
```python
"""Signal processing functions."""
from .filters import (
    apply_notch_filter,
    apply_bandpass_filter,
    apply_highpass_filter,
    apply_lowpass_filter,
)
from .transforms import (
    compute_fft,
    compute_hilbert_transform,
    compute_power_spectral_density,
)
from .processing import process_eeg_data

__all__ = [
    'apply_notch_filter', 'apply_bandpass_filter',
    'apply_highpass_filter', 'apply_lowpass_filter',
    'compute_fft', 'compute_hilbert_transform', 'compute_power_spectral_density',
    'process_eeg_data',
]
```

### `app/visualization/__init__.py`
```python
"""Visualization modules for Plotly figures."""
from .styles import STYLE, THEME_PRIMARY, THEME_SECONDARY
from .figures import EEGFigure, FFTFigure, HilbertFigure, BrainFigure
```

### `app/visualization/figures/__init__.py`
```python
"""Plotly figure classes."""
from .base import BaseFigure
from .eeg_figure import EEGFigure
from .fft_figure import FFTFigure
from .hilbert_figure import HilbertFigure
from .brain_figure import BrainFigure

__all__ = [
    'BaseFigure',
    'EEGFigure', 'FFTFigure', 'HilbertFigure', 'BrainFigure',
]
```

### `app/pages/__init__.py`
```python
"""Page modules for NiceGUI UI."""
from .viewer import ViewerPage
from .pipeline import PipelinePage
from .model import ModelPage
from .analysis import AnalysisPage
```

---

## 🔗 MANEJO DE ESTADOS (Instanciación)

**Problema**: ¿Dónde se crean las instancias de estado?

**Solución**: Singleton por página usando el patrón "state holder"

### `app/state/holders.py`
```python
"""
State instance holders.

Each page has its own state instance. States are created lazily on first access.
This replaces the global S, PS, MS, AS variables.
"""
from typing import Optional
from .viewer_state import ViewerState
from .pipeline_state import PipelineState
from .model_state import ModelState
from .analysis_state import AnalysisState

class StateHolder:
    """Holds singleton instances of application states."""
    
    _viewer: Optional[ViewerState] = None
    _pipeline: Optional[PipelineState] = None
    _model: Optional[ModelState] = None
    _analysis: Optional[AnalysisState] = None
    
    @classmethod
    def get_viewer_state(cls) -> ViewerState:
        """Get or create viewer state."""
        if cls._viewer is None:
            cls._viewer = ViewerState()
        return cls._viewer
    
    @classmethod
    def get_pipeline_state(cls) -> PipelineState:
        """Get or create pipeline state."""
        if cls._pipeline is None:
            cls._pipeline = PipelineState()
        return cls._pipeline
    
    @classmethod
    def get_model_state(cls) -> ModelState:
        """Get or create model state."""
        if cls._model is None:
            cls._model = ModelState()
        return cls._model
    
    @classmethod
    def get_analysis_state(cls) -> AnalysisState:
        """Get or create analysis state."""
        if cls._analysis is None:
            cls._analysis = AnalysisState()
        return cls._analysis
    
    @classmethod
    def reset_all(cls) -> None:
        """Reset all states (useful for testing)."""
        cls._viewer = None
        cls._pipeline = None
        cls._model = None
        cls._analysis = None
```

### Uso en páginas:
```python
# app/pages/viewer/page.py
from ...state.holders import StateHolder

class ViewerPage:
    def __init__(self):
        self.state = StateHolder.get_viewer_state()
```

---

## 🔀 GRAFO DE DEPENDENCIAS

```
main.py
    └── app/pages/*
            ├── app/state/* (estados)
            ├── app/visualization/* (figuras)
            │       └── app/visualization/styles/* (tema)
            └── app/core/* (lógica de negocio)
                    └── (numpy, scipy - sin deps internas)
```

**Regla crítica**: Los módulos en `app/core/` NO pueden importar de `app/pages/` ni `app/visualization/`.
Esto asegura que la lógica de negocio es independiente de la UI.

---

## 📝 CONVENCIONES DE CÓDIGO

### Naming
- Clases: `PascalCase`
- Funciones/métodos: `snake_case`
- Constantes: `UPPER_SNAKE_CASE`
- Archivos: `snake_case.py`

### Docstrings
```python
def function_name(param1: Type1, param2: Type2) -> ReturnType:
    """
    Brief description.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ErrorType: When this error occurs
    """
```

### Type Hints
- Usar type hints en todos los parámetros y returns
- Usar `Optional[T]` para valores que pueden ser None
- Usar `from __future__ import annotations` para forward references

### Imports
```python
# Standard library
from typing import Dict, List, Optional

# Third party
import numpy as np
from nicegui import ui

# Local
from .base import BaseFigure
from ..state import ViewerState
```

---

## ✅ BENEFICIOS ESPERADOS

1. **Mantenibilidad**: Cada módulo tiene una responsabilidad clara
2. **Testabilidad**: Funciones puras y dependencias inyectadas
3. **Reusabilidad**: Componentes y figuras reutilizables
4. **Escalabilidad**: Fácil agregar nuevas páginas/visualizaciones
5. **Legibilidad**: Código organizado y bien documentado
6. **Performance**: Lazy loading de módulos

---

## 🎯 PRÓXIMOS PASOS

Una vez aprobado este plan:

1. Crear la estructura de directorios
2. Comenzar con Fase 1 (Fundamentos)
3. Migrar gradualmente sin romper funcionalidad existente
4. Agregar tests en cada paso

¿Procedemos con la implementación?

---

## 📚 REFERENCIA: MÓDULO `cleaning/` (EJEMPLO A SEGUIR)

El módulo `cleaning/` ya existente es un **excelente ejemplo** de la arquitectura objetivo. Deberías usarlo como referencia:

### Estructura de `cleaning/`

```
cleaning/
├── __init__.py         # Exports claros con __all__
├── state.py            # CleaningState con patrón Observer
├── filters.py          # Funciones puras de filtrado
├── bad_channels.py     # Detección de canales malos
├── rereferencing.py    # Re-referencia
├── ica.py              # ICA
├── epochs.py           # Creación de epochs
├── export.py           # Exportación
├── ui_components.py    # Componentes UI reutilizables
└── cleaner_page.py     # Página principal (2745 líneas, pero BIEN ORGANIZADA)
```

### Patrones Clave Implementados

#### 1. Estado con Observer (`state.py`)

```python
class CleaningState:
    def __init__(self):
        self._on_change_callbacks: List[Callable] = []
    
    def on_change(self, callback: Callable):
        """Register a callback for state changes."""
        self._on_change_callbacks.append(callback)
    
    def _notify_change(self):
        """Notify all listeners of state change."""
        for callback in self._on_change_callbacks:
            callback()
```

#### 2. Exports Explícitos (`__init__.py`)

```python
from .state import CleaningState, CleaningStep
from .filters import FilterPreset, apply_filter_preset
# ...

__all__ = [
    'CleaningState', 'CleaningStep',
    'FilterPreset', 'apply_filter_preset',
    # ...
]
```

#### 3. Funciones Puras (sin side effects)

```python
def apply_bandpass_filter(
    raw: mne.io.Raw,
    low_freq: float,
    high_freq: float
) -> mne.io.Raw:
    """Pure function - no global state."""
    return raw.copy().filter(l_freq=low_freq, h_freq=high_freq)
```

#### 4. Dataclasses para Resultados

```python
@dataclass
class BadChannelResult:
    bad_by_std: List[str]
    bad_by_flat: List[str]
    bad_by_correlation: List[str]
    all_bad: List[str]
    metrics: Dict[str, float]
```

### Diferencia Clave

| Aspecto | `main.py` (actual) | `cleaning/` (objetivo) |
|---------|-------------------|------------------------|
| Estado | 4 clases globales mezcladas | 1 clase encapsulada |
| Funciones | Anidadas en páginas | Módulos separados |
| UI | Inline con lógica | Separada |
| Tests | Imposibles | Fáciles |

---

## 🤖 NOTAS PARA LLM QUE IMPLEMENTE ESTO

### Reglas de Migración

1. **NO romper funcionalidad existente**: Migrar gradualmente
2. **Un módulo a la vez**: No intentar todo de una vez
3. **Tests primero**: Escribir tests antes de mover código
4. **Imports relativos**: Usar imports relativos dentro de `app/`

### Orden de Prioridad

1. **CRÍTICO**: Extraer `signal/filters.py` y `signal/transforms.py` (sin dependencias UI)
2. **ALTO**: Mover estados a módulos separados
3. **MEDIO**: Extraer figuras a clases
4. **BAJO**: Reorganizar páginas

### Comandos Útiles

```bash
# Verificar que no se rompe nada
cd /media/storage_hdd/dmt_fz/eeg_viewer
python -c "from main import *; print('OK')"

# Correr la app
./run.sh

# Tests existentes
pytest tests/
```

### Archivo de Entrada Propuesto (`main.py` nuevo)

```python
#!/usr/bin/env python3
"""
EEG Viewer - Entry Point

This is the minimal entry point. All logic is in app/ modules.
"""
from nicegui import ui
from app.pages.viewer import ViewerPage
from app.pages.pipeline import PipelinePage
from app.pages.model import ModelPage
from app.pages.analysis import AnalysisPage
from app.visualization.styles import STYLE

# Register pages
@ui.page('/')
def viewer_page():
    ui.add_head_html(f'<style>{STYLE}</style>')
    ViewerPage().render()

@ui.page('/pipeline')
def pipeline_page():
    ui.add_head_html(f'<style>{STYLE}</style>')
    PipelinePage().render()

@ui.page('/model')
def model_page():
    ui.add_head_html(f'<style>{STYLE}</style>')
    ModelPage().render()

@ui.page('/analysis')
def analysis_page():
    ui.add_head_html(f'<style>{STYLE}</style>')
    AnalysisPage().render()

# Cleaner page is already separate
from cleaning.cleaner_page import cleaner_page
@ui.page('/cleaner')
def cleaner_page_route():
    ui.add_head_html(f'<style>{STYLE}</style>')
    cleaner_page()

if __name__ in {"__main__", "__mp_main__"}:
    print("EEG VIEWER - http://localhost:8080")
    ui.run(title='EEG Viewer', port=8080, reload=False, show=False, dark=True)
```

---

## 📊 MÉTRICAS DE ÉXITO

| Métrica | Antes | Objetivo |
|---------|-------|----------|
| `main.py` líneas | 5094 | < 50 |
| Archivos de módulo | 1 | ~40 |
| Cobertura de tests | 0% | > 80% |
| Estados globales | 4 | 0 |
| Funciones anidadas | ~100 | 0 |
| Imports circulares | ¿? | 0 |

---

## ⚠️ CABOS SUELTOS Y RIESGOS IDENTIFICADOS

### 1. Funciones Anidadas NO Documentadas (~50 funciones)

Dentro de cada `@ui.page()` hay funciones anidadas que NO se pueden extraer fácilmente:

**En `pipeline_page()` (líneas 936-2110):**
```python
# Funciones anidadas que hay que extraer:
def scan_input_dir()      # Línea 965
def get_input_dir()       # Línea 981
def refresh_run_label()   # Línea 993
async def new_run()       # Línea 1001
def use_existing()        # Línea 1014
def get_conditions()      # Línea 1060
async def run_fwd()       # Línea 1074
async def run_multi()     # Línea 1097
async def run_order()     # Línea 1113
async def run_clustering()  # Línea 1151
async def run_build_order() # Línea 1187
async def run_pearson()   # Línea 1204
def update_status()       # Línea 1238
async def stop_pipeline() # Línea 1250
def clear_log()           # Línea 1261
def refresh_files()       # Línea 1276
def update_system_stats() # Línea 1326
def refresh_all_plots()   # Línea 1430
def update_brain_plot()   # Línea 1486
def update_network_plot() # Línea 1562
def update_kuramoto_timeline()  # Línea 1590
def show_all_bands()      # Línea 1606
def update_band_comparison()    # Línea 1628
def update_phase_plot()   # Línea 1655
def update_sync_matrix()  # Línea 1680
def update_connectivity() # Línea 1706
def update_hilbert_2d()   # Línea 1734
def update_hilbert_3d()   # Línea 1761
async def generate_frames()     # Línea 1917
async def generate_video()      # Línea 2031
# ... y ~20 más
```

**Solución**: Estas funciones usan variables del scope de la página (closures). Hay que:
1. Convertirlas en métodos de una clase `PipelinePage`
2. Pasar el estado explícitamente en lugar de usar closures
3. Mover las funciones async a `app/core/pipeline/runner.py`

### 2. Dependencias Externas No Incluidas

El código importa módulos que están FUERA de `eeg_viewer/`:

```python
# En el tab VISUALIZE del pipeline:
from viz_scripts import brain_3d        # /media/storage_hdd/dmt_fz/viz_scripts/
from viz_scripts import kuramoto_viz    # /media/storage_hdd/dmt_fz/viz_scripts/
from viz_scripts import clustering_viz  # /media/storage_hdd/dmt_fz/viz_scripts/

# El pipeline ejecuta scripts externos:
PIPELINE_DIR = Path(__file__).parent.parent / "dashboard" / "pipeline_backend"
# Ejecuta: fwd.py, multi2pool2.py, generate_order.py, clustering.py, etc.
```

**Riesgo**: Estos módulos deben seguir funcionando después del refactor.

**Solución**: 
- NO mover estos módulos
- Solo refactorizar las IMPORTACIONES dentro de main.py
- Verificar que los paths relativos siguen funcionando

### 3. El Archivo `config.py` Existente

Ya existe `/media/storage_hdd/dmt_fz/eeg_viewer/config.py` con constantes:

```python
THEME_BG = "#0a0a0a"
THEME_PRIMARY = "#00ff88"
FREQ_BANDS = {"δ": (1, 4), "θ": (4, 8), ...}  # Usa símbolos griegos!
```

**Riesgo**: El nuevo `app/visualization/styles/theme.py` podría duplicar estas constantes.

**Solución**: 
- NO crear `app/visualization/styles/theme.py`
- REUSAR el `config.py` existente
- Solo crear `app/visualization/styles/css.py` que importa de `config.py`

### 4. Tests Existentes Solo Cubren `cleaning/`

Los tests en `/media/storage_hdd/dmt_fz/eeg_viewer/tests/` solo prueban el módulo cleaning.
NO hay tests para:
- Las figuras Plotly
- Las funciones de procesamiento de señal
- Las páginas UI
- El estado del viewer

**Riesgo**: No hay forma de verificar que el refactor no rompe nada.

**Solución antes de refactorizar**:
1. Crear tests para las funciones puras (signal processing)
2. Crear tests de integración que verifiquen que la app carga
3. Hacer snapshot del comportamiento actual

### 5. Variables de Estado en Closures

Muchas funciones usan `current_run_dir[0]` como "mutable container":

```python
current_run_dir = [None]  # Lista mutable usada como referencia

async def run_fwd():
    if not current_run_dir[0]:  # Acceso a closure
        ui.notify('Primero creá un NEW RUN', type='warning')
        return
```

**Problema**: Esto es un anti-pattern que dificulta testing.

**Solución**: Mover estas variables al estado de la página:
```python
class PipelinePageState:
    def __init__(self):
        self.current_run_dir: Optional[Path] = None
```

### 6. Referencias UI Guardadas en Estado Global

El código guarda referencias a widgets UI en estados globales:

```python
class State:
    def __init__(self):
        self.eeg_plot = None       # Referencia a widget Plotly
        self.fft_plot = None       # Referencia a widget Plotly
        # ...

class PipelineState:
    def __init__(self):
        self.log_container = None  # Referencia a ui.column()
        self.status_label = None   # Referencia a ui.label()
```

**Problema**: Mezcla estado de datos con estado de UI.

**Solución**: Separar en dos tipos de estado:
- `ViewerState` - Solo datos (EEG cargado, canales seleccionados)
- `ViewerUIState` - Solo UI (referencias a widgets) - local a la página

### 7. El Tab VISUALIZE Tiene 400+ Líneas de Código Complejo

El tab de visualización dentro de `pipeline_page()` (líneas 1369-2110) incluye:
- 3D brain network visualization
- Kuramoto timeline plots
- Phase distribution
- Sync matrix
- ROI connectivity
- Hilbert 2D/3D
- Clustering analysis
- Pearson correlations gallery
- Animation generator

**Riesgo**: Este código es muy complejo y tiene muchas dependencias.

**Solución**: 
1. Extraer a `app/pages/pipeline/components/visualize_tab.py`
2. Cada tipo de visualización en su propia clase
3. Mantener las importaciones a `viz_scripts/` sin cambios

### 8. Funciones de Análisis de Modelo (3000+ líneas)

Las páginas MODEL y ANALYSIS tienen funciones muy específicas:
- `detect_dataset_type()` - 140 líneas
- `create_default_config()` - 140 líneas  
- `load_trained_model()` - Carga modelos PyTorch
- `register_activation_hooks()` - Hooks para visualizar capas
- `process_sample()` - Procesa muestras a través del modelo
- `compute_dataset_ranges()` - Calcula rangos de ejes
- `load_hdf5_image_dataset()` - Carga datasets HDF5

**Riesgo**: Estas funciones son específicas de ML y tienen dependencias pesadas (PyTorch, torch_geometric).

**Solución**: Mover a `app/core/ml/` pero mantener las importaciones lazy:
```python
def load_trained_model(path):
    import torch  # Import lazy para evitar carga inicial pesada
    from torch_geometric.data import Batch
    # ...
```

---

## 🔧 CORRECCIONES AL PLAN ORIGINAL

### Cambio 1: NO crear `app/visualization/styles/theme.py`

Usar el `config.py` existente directamente:

```python
# app/visualization/styles/__init__.py
from config import (
    THEME_BG, THEME_CARD, THEME_BORDER, THEME_PRIMARY, 
    THEME_SECONDARY, THEME_WARN, THEME_ERROR, THEME_TEXT, 
    THEME_TEXT_DIM, SIGNAL_COLORS, FREQ_BANDS
)
from .css import STYLE, generate_style

__all__ = [
    'THEME_BG', 'THEME_CARD', 'THEME_BORDER', 'THEME_PRIMARY',
    'THEME_SECONDARY', 'THEME_WARN', 'THEME_ERROR', 'THEME_TEXT',
    'THEME_TEXT_DIM', 'SIGNAL_COLORS', 'FREQ_BANDS',
    'STYLE', 'generate_style',
]
```

### Cambio 2: Separar Estado de UI

```python
# app/state/viewer_state.py - SOLO datos
class ViewerState(BaseState):
    def __init__(self):
        self.eeg_slot1 = EEGSlot()
        self.selected_channels: List[str] = []
        # NO referencias a widgets UI

# app/pages/viewer/page.py - Referencias UI locales
class ViewerPage:
    def __init__(self):
        self.state = StateHolder.get_viewer_state()
        # Referencias UI locales a la página
        self.eeg_plot = None
        self.fft_plot = None
```

### Cambio 3: Lazy imports para ML

```python
# app/core/ml/model_loader.py
def load_trained_model(model_path: Path):
    """Load model with lazy imports."""
    # Imports pesados solo cuando se necesitan
    import torch
    import sys
    
    # ... resto del código
```

### Cambio 4: Agregar Tests de Humo ANTES de refactorizar

```python
# tests/test_smoke.py - Ejecutar ANTES de refactorizar
"""Smoke tests to verify app works before refactoring."""

def test_main_imports():
    """Verify main.py can be imported without errors."""
    from main import State, PipelineState, ModelState, AnalysisState
    assert State is not None

def test_signal_processing():
    """Verify signal processing functions work."""
    from main import apply_notch, apply_bandpass, compute_fft, compute_hilbert
    import numpy as np
    
    data = np.random.randn(10, 1000)
    sfreq = 256.0
    
    # Should not raise
    filtered = apply_notch(data, sfreq, 50.0)
    filtered = apply_bandpass(data, sfreq, 1.0, 45.0)
    freqs, power = compute_fft(data, sfreq)
    amp, phase = compute_hilbert(data)
    
    assert filtered.shape == data.shape
    assert len(freqs) > 0

def test_figure_creation():
    """Verify figures can be created."""
    from main import make_eeg_fig, make_fft_fig, make_hilbert_fig, make_brain_fig
    
    fig1 = make_eeg_fig()
    fig2 = make_fft_fig()
    fig3 = make_hilbert_fig()
    fig4 = make_brain_fig()
    
    assert fig1 is not None
    assert fig2 is not None
```

---

## 📊 MATRIZ DE RIESGOS

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|--------------|---------|------------|
| Romper pipeline execution | ALTA | CRÍTICO | Tests de humo + backup de main.py |
| Imports circulares | MEDIA | ALTO | Grafo de dependencias estricto |
| viz_scripts deja de funcionar | BAJA | ALTO | No mover, solo cambiar imports |
| Performance degradada | BAJA | MEDIO | Lazy imports para ML |
| Tests fallan | MEDIA | MEDIO | Agregar tests de humo primero |

---

## 🛠️ SCRIPT PARA CREAR ESTRUCTURA

Ejecutar este script para crear todos los directorios y archivos `__init__.py` vacíos:

```bash
#!/bin/bash
# create_structure.sh
# Ejecutar desde /media/storage_hdd/dmt_fz/eeg_viewer/

set -e

echo "Creating app/ directory structure..."

# Create directories
mkdir -p app/state
mkdir -p app/core/signal
mkdir -p app/core/data
mkdir -p app/core/pipeline
mkdir -p app/visualization/styles
mkdir -p app/visualization/figures
mkdir -p app/visualization/updaters
mkdir -p app/pages/components
mkdir -p app/pages/viewer/components
mkdir -p app/pages/pipeline/components
mkdir -p app/pages/model/components
mkdir -p app/pages/analysis/components
mkdir -p app/utils

# Create __init__.py files
touch app/__init__.py
touch app/state/__init__.py
touch app/core/__init__.py
touch app/core/signal/__init__.py
touch app/core/data/__init__.py
touch app/core/pipeline/__init__.py
touch app/visualization/__init__.py
touch app/visualization/styles/__init__.py
touch app/visualization/figures/__init__.py
touch app/visualization/updaters/__init__.py
touch app/pages/__init__.py
touch app/pages/components/__init__.py
touch app/pages/viewer/__init__.py
touch app/pages/viewer/components/__init__.py
touch app/pages/pipeline/__init__.py
touch app/pages/pipeline/components/__init__.py
touch app/pages/model/__init__.py
touch app/pages/model/components/__init__.py
touch app/pages/analysis/__init__.py
touch app/pages/analysis/components/__init__.py
touch app/utils/__init__.py

echo "Structure created successfully!"
echo ""
echo "Created directories:"
find app -type d | sort

echo ""
echo "Next steps:"
echo "1. Implement app/state/base.py"
echo "2. Implement app/core/signal/filters.py"
echo "3. Run: python -c 'import app; print(app)'"
```

Guardar como `create_structure.sh` y ejecutar:
```bash
cd /media/storage_hdd/dmt_fz/eeg_viewer
chmod +x create_structure.sh
./create_structure.sh
```

---

## 📋 RESUMEN EJECUTIVO PARA LLM

### Lo que debes hacer:

1. **LEER** este documento completo antes de empezar
2. **EJECUTAR** el script `create_structure.sh` para crear directorios
3. **IMPLEMENTAR** en orden de fases (1→5)
4. **VERIFICAR** cada checklist antes de pasar a la siguiente fase
5. **NO BORRAR** `main.py` hasta que todo funcione - renombrarlo a `main_legacy.py`

### Lo que NO debes hacer:

1. **NO** implementar todo de una vez
2. **NO** crear módulos que importen de `pages/` en `core/`
3. **NO** dejar estados globales sueltos
4. **NO** copiar/pegar código sin adaptar imports
5. **NO** olvidar los `__init__.py`

### Prioridad de módulos:

```
CRÍTICO (hacer primero):
├── app/core/signal/filters.py
├── app/core/signal/transforms.py
├── app/state/base.py
└── app/state/viewer_state.py

ALTO:
├── app/visualization/styles/theme.py
├── app/visualization/figures/base.py
└── app/visualization/figures/eeg_figure.py

MEDIO:
├── app/pages/components/header.py
├── app/pages/viewer/page.py
└── Resto de figuras

BAJO:
├── app/pages/pipeline/page.py
├── app/pages/model/page.py
└── app/pages/analysis/page.py
```

### Comando de verificación después de cada cambio:

```bash
cd /media/storage_hdd/dmt_fz/eeg_viewer
python -c "from main import *; print('main.py OK')"
python -c "from app.core.signal import *; print('signal OK')"
./run.sh  # Verificar que la app carga
```

---

## 🚀 CONCLUSIÓN

Este documento proporciona una hoja de ruta completa para transformar un monolito de 5000+ líneas en una arquitectura modular, mantenible y testeable.

El módulo `cleaning/` ya existente demuestra que esta arquitectura funciona bien en este proyecto. La refactorización debe seguir ese patrón establecido.

**Siguiente paso**: Aprobar el plan y comenzar la implementación fase por fase.

---

## 🔧 EJEMPLOS ADICIONALES IMPORTANTES

### `app/visualization/figures/fft_figure.py`

```python
"""FFT power spectrum figure."""
import numpy as np
from typing import Dict, Any, List
import plotly.graph_objects as go
from .base import BaseFigure

# Import frequency bands from constants
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config import FREQ_BANDS

class FFTFigure(BaseFigure):
    """Figure for displaying FFT power spectrum with frequency band highlighting."""
    
    BAND_COLORS = [
        'rgba(0,212,255,0.08)',   # Delta
        'rgba(0,255,136,0.08)',   # Theta
        'rgba(255,204,0,0.08)',   # Alpha
        'rgba(255,107,157,0.08)', # Beta
        'rgba(167,139,250,0.08)', # Gamma
    ]
    
    @property
    def primary_color(self) -> str:
        return '#00d4ff'
    
    def create(self) -> go.Figure:
        """Create FFT figure with frequency band regions."""
        fig = go.Figure()
        
        # Add frequency band highlighting
        for i, (band, (lo, hi)) in enumerate(FREQ_BANDS.items()):
            fig.add_vrect(
                x0=lo, x1=hi,
                fillcolor=self.BAND_COLORS[i % len(self.BAND_COLORS)],
                line_width=0
            )
            fig.add_annotation(
                x=(lo + hi) / 2, y=1.02, yref='paper',
                text=band, showarrow=False,
                font=dict(size=11, color='#666', family=self.FONT_FAMILY)
            )
        
        layout = self.get_base_layout()
        layout.update({
            'margin': {'l': 60, 'r': 10, 't': 30, 'b': 50},
            'height': 180,
            'xaxis': {
                'title': {'text': 'FREQ [Hz]', 'font': {'size': 9, 'color': self.accent_color}},
                'gridcolor': self.GRID_COLOR,
                'range': [0, 60],
                'fixedrange': True,
            },
            'yaxis': {
                'title': {'text': 'PWR [µV]', 'font': {'size': 9, 'color': self.accent_color}},
                'gridcolor': self.GRID_COLOR,
                'fixedrange': True,
            },
            'showlegend': True,
            'legend': {'orientation': 'h', 'y': 1.15, 'font': {'size': 8}},
        })
        
        fig.update_layout(**layout)
        self._figure = fig
        return fig
    
    def update(
        self,
        freqs: np.ndarray,
        power: np.ndarray,
        channel_names: List[str],
        max_freq: float = 60.0
    ) -> None:
        """
        Update FFT figure with new data.
        
        Args:
            freqs: Frequency array
            power: Power values (channels x frequencies)
            channel_names: List of channel names
            max_freq: Maximum frequency to display
        """
        # Mask to max frequency
        mask = freqs <= max_freq
        freqs = freqs[mask]
        power = power[:, mask] if power.ndim > 1 else power[mask]
        
        # Keep frequency band vrects (first N traces)
        n_bands = len(FREQ_BANDS)
        self.figure.data = self.figure.data[:n_bands] if len(self.figure.data) >= n_bands else []
        
        colors = self.SIGNAL_COLORS if not self.use_secondary_style else [
            '#f472b6', '#fb7185', '#fda4af', '#fecdd3', '#ffe4e6'
        ]
        fills = [c.replace(')', ', 0.15)').replace('rgb', 'rgba') for c in colors]
        
        for i, ch in enumerate(channel_names[:5]):  # Max 5 channels
            ch_power = power[i] if power.ndim > 1 else power
            self.figure.add_trace(go.Scatter(
                x=freqs, y=ch_power,
                name=ch,
                line={'color': colors[i % len(colors)], 'width': 1.5},
                fill='tozeroy',
                fillcolor=fills[i % len(fills)]
            ))
```

### `app/visualization/figures/brain_figure.py`

```python
"""Brain topography figure with electrode positions."""
import numpy as np
from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
from .base import BaseFigure

# Import electrode positions
from ...utils.constants import ELECTRODE_POSITIONS

class BrainFigure(BaseFigure):
    """Figure for displaying brain topography with electrode activation levels."""
    
    @property
    def primary_color(self) -> str:
        return '#00ff88'
    
    def create(self) -> go.Figure:
        """Create brain topography figure with head outline."""
        fig = go.Figure()
        
        # Head outline
        head_color = 'rgba(244,114,182,0.5)' if self.use_secondary_style else 'rgba(0,255,136,0.5)'
        theta = np.linspace(0, 2 * np.pi, 100)
        
        # Head circle
        fig.add_trace(go.Scatter(
            x=np.cos(theta), y=np.sin(theta),
            mode='lines', line={'color': head_color, 'width': 2},
            showlegend=False, hoverinfo='skip'
        ))
        
        # Nose
        fig.add_trace(go.Scatter(
            x=[-0.08, 0, 0.08], y=[0.98, 1.12, 0.98],
            mode='lines', line={'color': head_color, 'width': 2},
            showlegend=False, hoverinfo='skip'
        ))
        
        # Left ear
        fig.add_trace(go.Scatter(
            x=[-1.02, -1.08, -1.02], y=[0.15, 0, -0.15],
            mode='lines', line={'color': head_color, 'width': 1.5},
            showlegend=False, hoverinfo='skip'
        ))
        
        # Right ear
        fig.add_trace(go.Scatter(
            x=[1.02, 1.08, 1.02], y=[0.15, 0, -0.15],
            mode='lines', line={'color': head_color, 'width': 1.5},
            showlegend=False, hoverinfo='skip'
        ))
        
        layout = self.get_base_layout()
        layout.update({
            'height': 200,
            'margin': {'l': 5, 'r': 5, 't': 5, 'b': 5},
            'plot_bgcolor': '#0a0a0a',
            'xaxis': {
                'range': [-1.25, 1.25], 'showgrid': False,
                'zeroline': False, 'showticklabels': False,
                'scaleanchor': 'y', 'fixedrange': True
            },
            'yaxis': {
                'range': [-0.9, 1.2], 'showgrid': False,
                'zeroline': False, 'showticklabels': False,
                'fixedrange': True
            },
            'showlegend': False,
            'hovermode': 'closest',
        })
        
        fig.update_layout(**layout)
        self._figure = fig
        return fig
    
    def update(
        self,
        selected_channels: List[str],
        amplitudes: Dict[str, float],
    ) -> None:
        """
        Update brain topography with electrode activations.
        
        Args:
            selected_channels: List of selected channel names
            amplitudes: Dict mapping channel names to amplitude values
        """
        # Calculate normalization
        if amplitudes and selected_channels:
            vals = [amplitudes.get(ch, 0) for ch in selected_channels if ch in amplitudes]
            min_a, max_a = (min(vals), max(vals)) if vals else (0, 1)
            rng = max_a - min_a if max_a > min_a else 1
        else:
            min_a, rng = 0, 1
        
        # Separate selected and unselected electrodes
        sel_x, sel_y, sel_colors, sel_text, sel_labels = [], [], [], [], []
        uns_x, uns_y, uns_labels = [], [], []
        
        for ch, (x, y) in ELECTRODE_POSITIONS.items():
            if ch in selected_channels:
                sel_x.append(x)
                sel_y.append(y)
                sel_labels.append(ch)
                amp = amplitudes.get(ch, 0)
                sel_colors.append((amp - min_a) / rng if rng > 0 else 0.5)
                sel_text.append(f'{ch}<br>{amp:.1f} µV')
            else:
                uns_x.append(x)
                uns_y.append(y)
                uns_labels.append(ch)
        
        # Keep head outline traces (first 4)
        self.figure.data = self.figure.data[:4]
        
        # Add unselected electrodes (dim)
        if uns_x:
            self.figure.add_trace(go.Scatter(
                x=uns_x, y=uns_y, mode='markers+text',
                marker={'size': 12, 'color': 'rgba(30,30,30,0.6)',
                        'line': {'width': 1, 'color': 'rgba(60,60,60,0.5)'}},
                text=uns_labels, textposition='top center',
                textfont={'size': 7, 'color': 'rgba(100,100,100,0.6)'},
                hoverinfo='text', hovertext=uns_labels, showlegend=False
            ))
        
        # Add selected electrodes (colored by amplitude)
        if sel_x:
            colorscale = (
                [[0, '#831843'], [0.25, '#be185d'], [0.5, '#f472b6'],
                 [0.75, '#fda4af'], [1, '#ffe4e6']]
                if self.use_secondary_style else
                [[0, '#0d47a1'], [0.25, '#00bcd4'], [0.5, '#00ff88'],
                 [0.75, '#ffcc00'], [1, '#ff5722']]
            )
            
            self.figure.add_trace(go.Scatter(
                x=sel_x, y=sel_y, mode='markers+text',
                marker={
                    'size': 18, 'color': sel_colors, 'colorscale': colorscale,
                    'cmin': 0, 'cmax': 1,
                    'line': {'width': 2, 'color': self.accent_color},
                    'showscale': True,
                    'colorbar': {'title': {'text': 'µV', 'font': {'size': 9}},
                                'len': 0.5, 'thickness': 8, 'x': 1.02}
                },
                text=sel_labels, textposition='top center',
                textfont={'size': 8, 'color': self.accent_color},
                hoverinfo='text', hovertext=sel_text, showlegend=False
            ))
```

### `app/pages/viewer/components/file_browser.py`

```python
"""
File browser component for EEG file selection.
"""
import asyncio
from pathlib import Path
from typing import Callable, Optional
from nicegui import ui

from ....state.viewer_state import ViewerState
from .....config import EEG_RAW_DIR, EEG_CLEAN_DIR
from .....eeg_loader import scan_eeg_directory, load_eeg_file

class FileBrowserComponent:
    """Component for browsing and loading EEG files."""
    
    def __init__(self, state: ViewerState, on_load: Optional[Callable] = None):
        """
        Initialize file browser.
        
        Args:
            state: ViewerState instance
            on_load: Callback to invoke after loading a file
        """
        self.state = state
        self.on_load = on_load
        self.eeg_slot_select = None
    
    def render(self) -> None:
        """Render the file browser component."""
        with ui.card().classes('dark-card p-4 w-full'):
            ui.label('// FILE_BROWSER').classes('terminal-header')
            
            # Slot selector
            with ui.row().classes('items-center gap-2 mb-2'):
                ui.label('Load to:').style('color: #666; font-size: 0.7rem;')
                self.eeg_slot_select = ui.toggle(
                    ['EEG 1', 'EEG 2'], value='EEG 1'
                ).props('dense')
                
                ui.button('Compare', on_click=self._toggle_compare, icon='compare') \
                    .props('dense flat size=sm')
                ui.button('Clear 2', on_click=self._clear_eeg2, icon='close') \
                    .props('dense flat size=sm')
            
            # File list
            with ui.scroll_area().classes('w-full').style('height: 250px;'):
                self._render_directory('raw/', EEG_RAW_DIR)
                self._render_directory('clean/', EEG_CLEAN_DIR)
    
    def _render_directory(self, label: str, directory: Path) -> None:
        """Render a directory section."""
        files = scan_eeg_directory(directory)
        if not files:
            return
        
        ui.label(f'├─ {label}').style(
            'color: #00ff88; font-family: JetBrains Mono; font-size: 0.8rem;'
        ).classes('mt-3 mb-2')
        
        # Group by condition
        conditions = {}
        for f in files:
            conditions.setdefault(f['condition'], []).append(f)
        
        for cond, cond_files in conditions.items():
            with ui.expansion(f'{cond} ({len(cond_files)} files)').classes('w-full'):
                for f in cond_files:
                    self._render_file_item(f)
    
    def _render_file_item(self, file_info: dict) -> None:
        """Render a single file item."""
        with ui.row().classes('file-item items-center w-full gap-3'):
            ui.icon('description', size='sm').classes('opacity-60')
            with ui.column().classes('flex-1'):
                ui.label(file_info['name']).classes('text-sm font-medium')
                ui.label(f"{file_info['size_mb']:.1f} MB").classes('text-xs opacity-50')
            ui.button(
                icon='play_arrow',
                on_click=lambda p=file_info['path']: self._load_file(p)
            ).props('flat dense size=sm color=red')
    
    async def _load_file(self, filepath: str) -> None:
        """Load an EEG file."""
        try:
            slot = 2 if self.eeg_slot_select.value == 'EEG 2' else 1
            loaded = await asyncio.get_event_loop().run_in_executor(
                None, load_eeg_file, Path(filepath)
            )
            
            self.state.load_eeg(loaded, slot)
            
            if slot == 1:
                # Auto-select first 10 EEG channels
                eeg_chs = [ch for ch, t in loaded.channel_types.items() if t == 'eeg']
                self.state.selected_channels = eeg_chs[:10]
                self.state.hilbert_channel = eeg_chs[0] if eeg_chs else ""
            else:
                self.state.compare_mode = True
            
            ui.notify(f'EEG {slot}: {loaded.filename}', type='positive')
            
            if self.on_load:
                self.on_load()
                
        except Exception as e:
            ui.notify(f'Error: {e}', type='negative')
    
    def _toggle_compare(self) -> None:
        """Toggle comparison mode."""
        self.state.toggle_compare_mode()
        ui.notify(f"Compare mode: {'ON' if self.state.compare_mode else 'OFF'}", type='info')
    
    def _clear_eeg2(self) -> None:
        """Clear the second EEG slot."""
        self.state.eeg_slot2.data = None
        self.state.eeg_slot2.is_loaded = False
        self.state.compare_mode = False
        ui.notify('EEG 2 cleared', type='info')
```

---

## ⚠️ PROBLEMAS POTENCIALES Y SOLUCIONES

### 1. Imports Circulares
**Problema**: `app/pages/viewer/page.py` importa `ViewerState`, pero `ViewerState` podría necesitar tipos de `pages`.

**Solución**: 
- Core modules (`state/`, `core/`) nunca importan de `pages/`
- Usar `TYPE_CHECKING` para type hints:
```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..pages.viewer import ViewerPage
```

### 2. NiceGUI y Async
**Problema**: Muchas funciones en main.py usan `async def` para operaciones lentas.

**Solución**: Mantener el mismo patrón. NiceGUI maneja bien async:
```python
async def _load_file(self, filepath: str) -> None:
    # Ejecutar operación lenta en thread pool
    loaded = await asyncio.get_event_loop().run_in_executor(
        None, load_eeg_file, Path(filepath)
    )
```

### 3. Referencias UI Dinámicas
**Problema**: El código actual guarda referencias a widgets UI en el estado (e.g., `S.eeg_plot`).

**Solución**: Mover referencias UI a las clases de página, no al estado:
```python
# MAL (en estado)
class State:
    self.eeg_plot = None  # Referencia a widget

# BIEN (en página)
class ViewerPage:
    def __init__(self):
        self.eeg_plot = None  # Referencia local a widget
```

### 4. Callbacks con Contexto
**Problema**: Callbacks en NiceGUI necesitan acceso al contexto correcto.

**Solución**: Usar closures o pasar self:
```python
# Con closure
ui.button('Load', on_click=lambda: self._load_file())

# Con partial
from functools import partial
ui.button('Load', on_click=partial(self._load_file, slot=1))
```

---

## 📐 DIAGRAMAS DE ARQUITECTURA

### Flujo de Datos (Viewer Page)

```
[Usuario] 
    │
    ▼
[FileBrowser] ──load_file()──► [eeg_loader.py]
    │                               │
    ▼                               ▼
[ViewerState] ◄──────────── EEGData
    │
    │ notify('eeg_loaded')
    ▼
[EEGUpdater] ──────────────► [EEGFigure]
    │                              │
    │ get_channel_data()           │ update()
    ▼                              ▼
[signal/processing.py]        [ui.plotly()]
```

### Flujo de Datos (Pipeline Page)

```
[Usuario]
    │
    ▼
[StepCard] ──run_step()──► [PipelineState]
    │                           │
    ▼                           ▼
[pipeline/runner.py]    [ConsoleTab.add_log()]
    │
    │ subprocess
    ▼
[dashboard/pipeline_backend/*.py]
```

---

## ✅ PROGRESO DE IMPLEMENTACIÓN

### Estado: ✅ COMPLETADO

| Fase | Estado | Fecha |
|------|--------|-------|
| Fase 0: Preparación | ✅ COMPLETADA | 2024-12-08 |
| Fase 1: Infraestructura Base | ✅ COMPLETADA | 2024-12-08 |
| Fase 2: Visualización y Migración | ✅ COMPLETADA | 2024-12-08 |
| Fase 3: Componentes UI | ✅ COMPLETADA | 2024-12-08 |
| Fase 3: Eliminar duplicación EEG1/EEG2 | ✅ COMPLETADA | 2024-12-08 |
| Fase 3: Separar Páginas | ⏸️ POSPUESTA | - |

**Nota:** La extracción de páginas (`/analysis`, `/model`, `/pipeline`) se pospuso porque:
- Tienen dependencias circulares con estado global (AS, MS, PS)
- El beneficio marginal no justifica el riesgo de romper funcionalidad
- El código ya está significativamente mejorado con las funciones genéricas

### Archivos Creados (23 archivos Python, 2390 líneas)

```
eeg_viewer/app/
├── __init__.py                          # Package principal
├── state/
│   ├── __init__.py                      # Exports de state
│   ├── base.py                          # BaseState + Observer + StateHolder
│   └── viewer_state.py                  # ViewerState, PipelineState, ModelState, AnalysisState
├── core/
│   ├── __init__.py                      # Exports de core
│   ├── updaters.py                      # Funciones unificadas update_eeg_plot, update_fft_plot, etc.
│   └── signal/
│       ├── __init__.py                  # Exports de signal
│       ├── filters.py                   # apply_notch, apply_bandpass, apply_lowpass, apply_highpass
│       ├── transforms.py                # compute_fft, compute_hilbert, compute_psd, compute_band_power
│       └── processing.py                # process_data, process_with_state, extract_epochs
├── visualization/
│   ├── __init__.py                      # Exports de visualization
│   ├── figures/
│   │   ├── __init__.py
│   │   ├── base.py                      # BaseFigure, FigureFactory
│   │   ├── eeg_figure.py                # EEGFigure, make_eeg_fig
│   │   ├── fft_figure.py                # FFTFigure, make_fft_fig
│   │   ├── hilbert_figure.py            # HilbertFigure, make_hilbert_fig
│   │   └── brain_figure.py              # BrainFigure, make_brain_fig
│   ├── styles/
│   │   ├── __init__.py
│   │   └── theme.py                     # Constantes de tema (re-exporta config.py)
│   └── components/
│       ├── __init__.py                  # Exports de components
│       └── navigation.py                # NavigationControls, FilterControls, ScaleControls
├── pages/
│   └── __init__.py                      # Páginas UI (pendiente migración)
└── utils/
    └── __init__.py                      # Utilidades (pendiente)
```

### Cambios en main.py

| Antes | Después | Reducción |
|-------|---------|-----------|
| 5094 líneas | 4986 líneas | -108 líneas (~2.1%) |

**Logros principales:**
1. ✅ Duplicación EEG1/EEG2 eliminada (funciones genéricas `_update_*_generic`)
2. ✅ Signal processing extraído a módulos puros y testeables
3. ✅ Figuras Plotly con Factory pattern
4. ✅ Componentes UI reutilizables creados
5. ✅ 97+ tests unitarios para los nuevos módulos

**Funciones migradas a módulos:**
- `apply_notch()` → `app.core.signal.filters`
- `apply_bandpass()` → `app.core.signal.filters`
- `compute_fft()` → `app.core.signal.transforms`
- `compute_hilbert()` → `app.core.signal.transforms`
- `process_data()` → `app.core.signal.processing`
- `make_eeg_fig()` → `app.visualization.figures.eeg_figure`
- `make_fft_fig()` → `app.visualization.figures.fft_figure`
- `make_hilbert_fig()` → `app.visualization.figures.hilbert_figure`
- `make_brain_fig()` → `app.visualization.figures.brain_figure`

### Tests Creados

| Archivo | Líneas | Descripción |
|---------|--------|-------------|
| test_smoke.py | 314 | Tests de humo para main.py |
| test_state.py | 249 | Tests para app/state (Observer, StateHolder) |
| test_signal.py | 300 | Tests para app/core/signal (filtros, transforms) |
| test_visualization.py | 223 | Tests para app/visualization (figuras) |
| **Nuevos tests** | **772** | **69 tests adicionales** |

### Resultado de Tests

| Test Suite | Tests | Resultado |
|------------|-------|-----------|
| test_state.py | 22 | ✅ PASSED |
| test_signal.py | 27 | ✅ PASSED |
| test_visualization.py | 20 | ✅ PASSED |
| test_smoke.py | 28 | ✅ PASSED |
| Otros tests existentes | 216 | ✅ PASSED |
| **TOTAL** | **313** | **313/313 PASSED** |

### Próximos Pasos

1. **Separar páginas** a `app/pages/`:
   - `pipeline_page.py` (~1580 líneas, líneas 857-2440)
   - `model_page.py` (~1390 líneas, líneas 2441-3831)
   - `analysis_page.py` (~965 líneas, líneas 3832-4797)
   - `viewer_page.py` (main page, líneas 4798-5015)

2. **Eliminar duplicación EEG1/EEG2**:
   - Usar `app/core/updaters.py` con parámetro `use_secondary`
   - Unificar `update_eeg()` y `update_eeg2()` 
   - Unificar `update_fft()` y `update_fft2()`
   - Etc.

3. **Migrar estado global**:
   - Reemplazar `S`, `PS`, `MS`, `AS` con `StateHolder`
   - Usar inyección de dependencias en lugar de globals

### Uso de los Nuevos Módulos

```python
# Importar módulos
from app.state import ViewerState, StateHolder
from app.core.signal import apply_notch, compute_fft, process_data
from app.core import update_eeg_plot, update_fft_plot
from app.visualization import make_eeg_fig, FigureFactory
from app.visualization.components import NavigationControls, FilterControls

# Crear estado con Observer
state = ViewerState()
state.subscribe(lambda attr, old, new: update_plots())

# Crear figuras
fig = FigureFactory.create('eeg', use_secondary_style=False)

# Actualizar plots de forma unificada
update_eeg_plot(plot, data, times, channels, use_secondary=False)
```

