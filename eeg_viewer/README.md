# EEG Viewer

> **Comprehensive EEG Processing and Visualization Suite**

A modular web-based application for EEG data visualization, preprocessing, pipeline execution, and neural network analysis. Built with [NiceGUI](https://nicegui.io/) and [MNE-Python](https://mne.tools/).

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Architecture](#architecture)
6. [Module Reference](#module-reference)
7. [Pages](#pages)
8. [Configuration](#configuration)
9. [API Reference](#api-reference)
10. [Testing](#testing)
11. [Troubleshooting](#troubleshooting)

---

## Overview

The EEG Viewer is a multi-page web application providing:

| Page | URL | Purpose |
|------|-----|---------|
| **Viewer** | `/` | Interactive EEG visualization with dual-file comparison |
| **Cleaner** | `/cleaner` | Step-by-step EEG preprocessing pipeline |
| **Pipeline** | `/pipeline` | Batch processing and source localization |
| **Model** | `/model` | Neural network training (VAE/GNN) |
| **Analysis** | `/analysis` | Trained model inspection and latent space exploration |

### Supported File Formats

- **BDF** - BioSemi (primary format)
- **EDF** - European Data Format
- **SET** - EEGLAB
- **FIF** - MNE native format

---

## Features

### Viewer Page (`/`)
- **Dual EEG Comparison**: Load two EEG files side-by-side for direct comparison
- **Real-time Filtering**: Notch (50/60 Hz) and bandpass filters
- **Multi-channel Display**: Select and visualize up to all available EEG channels
- **Time Navigation**: Play, pause, seek through recordings with keyboard shortcuts
- **Analysis Plots**:
  - Time-domain EEG signals with auto-scaling
  - FFT power spectrum with frequency band highlighting (δ, θ, α, β, γ)
  - Hilbert transform (envelope + instantaneous phase)
  - Brain topography with electrode activation visualization
- **Epoch Generation**: Create and export epochs for downstream analysis

### Cleaner Page (`/cleaner`)

A complete EEG preprocessing pipeline with 9 steps:

```
LOAD → FILTER → BAD_CHANNELS → REREFERENCE → ICA → EPOCHS → REJECT → VISUALIZE → EXPORT
```

Each step:
- Has dedicated UI controls with parameter inputs
- Shows real-time visualization of changes
- Supports undo/redo operations
- Saves snapshots for step-by-step comparison

**Key capabilities:**
- **Bad Channel Detection**: Automatic detection (flat, noisy, uncorrelated) + manual selection
- **Interpolation**: Spherical spline interpolation of bad channels
- **Re-referencing**: Average, linked mastoids, single electrode, or REST
- **ICA**: FastICA/Picard with automatic EOG/ECG artifact detection
- **Epoch Rejection**: Peak-to-peak amplitude, flat detection, gradient thresholds
- **Export**: FIF, SET, EDF, NPY, CSV formats with preprocessing logs

### Pipeline Page (`/pipeline`)
- Batch execution of source localization pipeline
- Forward model computation
- Multipool phase extraction
- Kuramoto order parameter calculation
- Clustering analysis
- Real-time progress monitoring and logging

### Model Page (`/model`)
- Dataset detection (graph/image formats)
- VAE/GNN model configuration
- Training with live loss visualization
- Checkpoint management

### Analysis Page (`/analysis`)
- Load trained models for inspection
- Layer-by-layer activation visualization
- Latent space PCA exploration
- Sample playback through the network
- Attention weight visualization (for GNN models)

---

## Installation

### Prerequisites

- **Python**: 3.9 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: 8GB+ RAM recommended for large EEG files

### Linux/macOS

```bash
cd /media/storage_hdd/dmt_fz/eeg_viewer

# Option 1: Using the run script (creates venv automatically)
chmod +x run.sh
./run.sh

# Option 2: Manual installation
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

### Windows

```cmd
cd \path\to\eeg_viewer

REM Option 1: Using the batch file (creates venv automatically)
run.bat

REM Option 2: Manual installation
python -m venv venv
call venv\Scripts\activate.bat
pip install -r requirements.txt
python main.py
```

### Dependencies

Core dependencies (from `requirements.txt`):

```
nicegui>=2.0.0          # Web UI framework
plotly>=5.18.0          # Interactive plots
mne>=1.6.0              # EEG processing
numpy>=1.24.0           # Numerical computing
scipy>=1.11.0           # Signal processing
pandas>=2.0.0           # Data manipulation
nibabel>=5.0.0          # Neuroimaging file I/O
scikit-learn>=1.3.0     # Machine learning utilities
pymatreader>=0.0.32     # MATLAB file reading
tqdm>=4.65.0            # Progress bars
```

For model training (optional):
```
torch>=2.0.0            # PyTorch
torch-geometric>=2.0.0  # Graph neural networks
```

---

## Quick Start

1. **Start the application:**
   ```bash
   ./run.sh  # Linux/macOS
   # or
   run.bat   # Windows
   ```

2. **Open browser:** Navigate to `http://localhost:8080`

3. **Load an EEG file:**
   - In the File Browser panel, click the ▶ button next to any file
   - Files are auto-discovered from `../EEG/` and `../EEG_CLEAN/` directories
   - Or use the path input to browse custom directories

4. **Navigate the recording:**
   - Use ◀ ▶ buttons or keyboard arrows
   - Click window duration buttons (2s, 5s, 10s, 20s)
   - Press Space to play/pause

5. **Apply filters:**
   - Enable Notch filter (50 Hz default) to remove line noise
   - Enable Bandpass filter (1-45 Hz default) to isolate EEG frequencies
   - Click "Apply" to see changes

6. **Compare two EEGs:**
   - Change "Load to:" to "EEG 2"
   - Load a second file
   - Click "Compare" to enable dual view mode

---

## Architecture

### Directory Structure

```
eeg_viewer/
├── main.py                    # Entry point, page registration
├── config.py                  # Global configuration (paths, colors, bands)
├── eeg_loader.py              # EEG file loading utilities
├── requirements.txt           # Python dependencies
├── run.sh                     # Linux/macOS launcher
├── run.bat                    # Windows launcher
├── pytest.ini                 # Test configuration
│
├── app/                       # Main application package
│   ├── __init__.py
│   ├── state/                 # State management
│   │   ├── __init__.py
│   │   ├── base.py            # BaseState with Observer pattern
│   │   ├── viewer_state.py    # ViewerState class (modern)
│   │   └── global_state.py    # Legacy state classes (S, PS, MS, AS)
│   │
│   ├── core/                  # Business logic (no UI dependencies)
│   │   ├── __init__.py
│   │   ├── updaters.py        # Generic plot update functions
│   │   └── signal/            # Signal processing
│   │       ├── __init__.py
│   │       ├── filters.py     # Notch, bandpass, highpass, lowpass
│   │       ├── transforms.py  # FFT, Hilbert, PSD
│   │       └── processing.py  # Combined processing pipeline
│   │
│   ├── visualization/         # Plotly figures and styling
│   │   ├── __init__.py
│   │   ├── figures/           # Figure factory classes
│   │   │   ├── __init__.py
│   │   │   ├── base.py        # BaseFigure, FigureFactory
│   │   │   ├── eeg_figure.py  # EEG time series
│   │   │   ├── fft_figure.py  # FFT spectrum
│   │   │   ├── hilbert_figure.py
│   │   │   └── brain_figure.py
│   │   ├── styles/
│   │   │   ├── __init__.py
│   │   │   ├── css.py         # Generated CSS stylesheet
│   │   │   └── theme.py       # Theme constants
│   │   └── components/
│   │       ├── __init__.py
│   │       ├── navigation.py  # Nav controls, filter controls
│   │       ├── file_browser.py
│   │       └── signal_preview.py
│   │
│   ├── pages/                 # NiceGUI page definitions
│   │   ├── __init__.py
│   │   ├── viewer.py          # Main viewer page (/)
│   │   ├── pipeline.py        # Pipeline page (/pipeline)
│   │   ├── model.py           # Model training (/model)
│   │   └── analysis.py        # Analysis page (/analysis)
│   │
│   └── utils/
│       └── __init__.py
│
├── cleaning/                  # EEG preprocessing module
│   ├── __init__.py            # Public API exports
│   ├── state.py               # CleaningState, CleaningStep enum
│   ├── filters.py             # Filter presets and application
│   ├── bad_channels.py        # Bad channel detection/interpolation
│   ├── rereferencing.py       # Reference application
│   ├── ica.py                 # ICA computation and artifact detection
│   ├── epochs.py              # Epoch creation and rejection
│   ├── export.py              # Multi-format export
│   ├── cleaner_page.py        # NiceGUI page (/cleaner)
│   └── ui_components.py       # Reusable UI widgets
│
├── tests/                     # Test suite
│   ├── conftest.py            # Shared fixtures
│   ├── run_tests.py           # Test runner with options
│   ├── test_smoke.py          # Import and basic function tests
│   ├── test_state.py          # State management tests
│   ├── test_signal.py         # Signal processing tests
│   ├── test_visualization.py  # Figure creation tests
│   ├── test_filters.py        # Cleaning filter tests
│   ├── test_bad_channels.py   # Bad channel detection tests
│   ├── test_ica.py            # ICA tests
│   ├── test_epochs.py         # Epoch tests
│   ├── test_export.py         # Export tests
│   ├── test_rereferencing.py  # Reference tests
│   └── test_integration.py    # End-to-end tests
│
├── viz_scripts/               # Standalone visualization scripts
│   ├── animation_generator.py # Generate brain activity animations
│   ├── brain_3d.py            # 3D brain network visualization
│   ├── clustering_viz.py      # Clustering result plots
│   └── kuramoto_viz.py        # Kuramoto order parameter plots
│
├── cache/                     # Runtime cache (gitignored)
│   └── autoencoder/           # Model checkpoints and dataset cache
│
└── pipeline_outputs/          # Pipeline execution results (gitignored)
    └── run_YYYYMMDD_HHMMSS/   # Timestamped run directories
```

### State Management

The application uses a hybrid state management approach:

#### Legacy Global State (`app/state/global_state.py`)

Used by the current pages for backwards compatibility:

```python
# Global singleton instances
S = State()           # Viewer state
PS = PipelineState()  # Pipeline state
MS = ModelState()     # Model training state
AS = AnalysisState()  # Analysis state
```

Each state class holds:
- Data references (loaded EEG, models, etc.)
- UI widget references (plots, containers)
- User preferences (filters, view settings)

#### Modern ViewerState (`app/state/viewer_state.py`)

New reactive state with Observer pattern support:

```python
class ViewerState(BaseState):
    """Reactive state with change notifications."""
    
    def __init__(self):
        super().__init__()
        # Data is stored in typed containers
        self.eeg_slot1 = EEGSlot()
        self.eeg_slot2 = EEGSlot()
        # ...
    
    def subscribe(self, callback: Callable):
        """Register for change notifications."""
        self._observers.append(callback)
```

#### Cleaning Module State (`cleaning/state.py`)

Dedicated state for the cleaning pipeline:

```python
class CleaningState:
    # Raw data management
    _raw: mne.io.Raw              # Current data
    _raw_original: mne.io.Raw     # Original (for reset)
    _raw_history: List[Raw]       # Undo stack
    
    # Step tracking
    current_step: CleaningStep
    completed_steps: Set[CleaningStep]
    operations: List[OperationRecord]  # Audit log
    
    # Step results
    bad_channels: List[str]
    ica: ICA
    epochs: Epochs
    # ...
```

### Data Flow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  File Load  │────▶│   EEGData    │────▶│    State    │
│ eeg_loader  │     │  dataclass   │     │  (S, PS..)  │
└─────────────┘     └──────────────┘     └──────────────┘
                                                │
                                                ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Plotly    │◀────│   Figures    │◀────│  Updaters   │
│   Plots     │     │  (factory)   │     │  (generic)  │
└─────────────┘     └──────────────┘     └──────────────┘
```

---

## Module Reference

### `eeg_loader.py` - EEG File Loading

```python
from eeg_loader import load_eeg_file, get_channel_data, scan_eeg_directory

# Load an EEG file (auto-detects format)
eeg_data = load_eeg_file("/path/to/file.bdf")

# Access properties
eeg_data.raw          # mne.io.Raw object
eeg_data.sfreq        # Sampling frequency (Hz)
eeg_data.n_channels   # Number of channels
eeg_data.duration_sec # Duration in seconds
eeg_data.channel_names  # List of channel names
eeg_data.channel_types  # Dict: channel -> type ('eeg', 'stim', etc.)

# Extract channel data for visualization
data, times, channels = get_channel_data(
    eeg_data,
    channels=['Fp1', 'Fp2', 'F3'],  # None = all EEG channels
    start_sec=10.0,                  # Start time
    duration_sec=5.0                 # Window duration
)
# data: ndarray (n_channels, n_samples) in Volts
# times: ndarray (n_samples,) in seconds
# channels: List[str] actual channel names

# Scan directory for EEG files
files = scan_eeg_directory("/path/to/eeg/")
# Returns: [{'name': 'S01.bdf', 'path': '...', 'format': '.bdf', 
#            'size_mb': 45.2, 'condition': 'DMT'}, ...]
```

### `app/core/signal/` - Signal Processing

Pure functions with no side effects or global state:

```python
from app.core.signal import (
    apply_notch, apply_bandpass, apply_highpass, apply_lowpass,
    compute_fft, compute_hilbert, compute_psd,
    process_data
)

# Individual filters
filtered = apply_notch(data, sfreq=512, freq=50.0, quality=30.0)
filtered = apply_bandpass(data, sfreq=512, low=1.0, high=45.0, order=4)
filtered = apply_highpass(data, sfreq=512, cutoff=0.1)
filtered = apply_lowpass(data, sfreq=512, cutoff=100.0)

# Transforms
freqs, magnitudes = compute_fft(data, sfreq=512)
amplitude, phase = compute_hilbert(data)
freqs, psd = compute_psd(data, sfreq=512, nperseg=256)

# Combined processing (applies filters based on flags)
processed = process_data(
    data, sfreq=512,
    notch_enabled=True, notch_freq=50.0,
    bandpass_enabled=True, bandpass_low=1.0, bandpass_high=45.0
)
```

### `app/visualization/figures/` - Plotly Figures

Factory pattern for consistent figure creation:

```python
from app.visualization import make_eeg_fig, make_fft_fig, make_hilbert_fig, make_brain_fig

# Create figures (use_eeg2=True for secondary/comparison style)
eeg_fig = make_eeg_fig(use_eeg2=False)    # Green theme
eeg_fig2 = make_eeg_fig(use_eeg2=True)    # Pink theme

fft_fig = make_fft_fig()                  # Cyan theme with band regions
hilbert_fig = make_hilbert_fig()          # Yellow theme, 2-row subplot
brain_fig = make_brain_fig()              # Topography with head outline

# Using the FigureFactory
from app.visualization import FigureFactory
fig = FigureFactory.create('eeg', use_secondary_style=True)
```

### `cleaning/` - Preprocessing Pipeline

Complete EEG preprocessing toolkit:

```python
from cleaning import (
    CleaningState, CleaningStep,
    detect_bad_channels, interpolate_channels,
    apply_reference, ReferenceType,
    compute_ica, detect_eog_components, apply_ica_exclusion,
    create_epochs, detect_bad_epochs, EpochRejectionCriteria,
    export_cleaned_eeg, ExportFormat
)

# Initialize state and load data
state = CleaningState()
raw = mne.io.read_raw_bdf("file.bdf", preload=True)
state.load_raw(raw, Path("file.bdf"))

# Step 2: Filter
from cleaning.filters import apply_filter_preset, FilterPreset
filtered = apply_filter_preset(raw, FilterPreset.STANDARD)
state.update_raw(filtered, 'filter', CleaningStep.FILTER, {...})

# Step 3: Bad channels
result = detect_bad_channels(raw, std_threshold=3.0)
# result.flat_channels, result.noisy_channels, result.all_bad
interpolated = interpolate_channels(raw, result.all_bad)

# Step 4: Re-reference
rereferenced = apply_reference(raw, ReferenceType.AVERAGE)

# Step 5: ICA
ica_result = compute_ica(raw, n_components=20, method='fastica')
eog_indices = detect_eog_components(ica_result.ica, raw)
cleaned = apply_ica_exclusion(raw, ica_result.ica, eog_indices)

# Step 6-7: Epochs
epochs = create_epochs(raw, duration=2.0, overlap=0.0)
criteria = EpochRejectionCriteria(peak_to_peak_uv=150.0)
epoch_result = detect_bad_epochs(epochs, criteria)
clean_epochs = epochs.drop(epoch_result.rejected_indices)

# Step 9: Export
export_cleaned_eeg(raw, "/output/cleaned.fif", ExportFormat.FIF)
export_preprocessing_log(state.get_preprocessing_summary(), "/output/log.json")
```

---

## Pages

### Viewer Page (`/`)

**URL:** `http://localhost:8080/`

**Layout:**
```
┌─────────────────────────────────────────────────────────────────┐
│ ▶ EEG_VIEWER v1.0                    [VIEWER|CLEANER|PIPELINE..]│
├────────────────┬────────────────────────────────────────────────┤
│ // FILE_BROWSER│ FILTERS: [Notch ✓] 50 Hz [Bandpass ✓] 1-45 Hz  │
│ Load to: [EEG1]│────────────────────────────────────────────────│
│ [Compare][Clr2]│ ▌EEG 1                    ▌EEG 2 (compare mode)│
│ ├─ raw/        │ [- 1x +]                                       │
│ │ └─ DMT(5)    │ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   │
│ │   └─ S01.bdf │ [|◀ ◀◀ ◀ ▶ ▶ ▶▶ ▶|] [2s][5s][10s][20s] 0:05/5:23│
│ ├─ clean/      │────────────────────────────────────────────────│
│                │ ▌TOPO 1          ▌TOPO 2                       │
│ // FILE_INFO   │ (brain map)      (brain map)                   │
│ EEG 1: S01.bdf │────────────────────────────────────────────────│
│ 512Hz|24ch|5:23│ ▌FFT 1           ▌FFT 2                        │
│ EEG 2: --      │ (spectrum)       (spectrum)                    │
│                │────────────────────────────────────────────────│
│ // CHANNELS    │ ▌HILBERT 1 [ch:▼]  ▌HILBERT 2                  │
│ [ALL][10][CLR] │ (envelope+phase)   (envelope+phase)            │
│ [Fp1][Fp2][F3] │────────────────────────────────────────────────│
│ [F4][C3][C4].. │ DATASET: Epoch: [2.0] sec [Generate][Save]     │
└────────────────┴────────────────────────────────────────────────┘
```

**Key Functions:**

| Function | Description |
|----------|-------------|
| `load_file(path)` | Load EEG file into slot 1 or 2 |
| `update_all()` | Refresh all plots with current view |
| `toggle_ch(ch)` | Toggle channel selection |
| `nav_fwd()/nav_back()` | Navigate time window |
| `toggle_play()` | Start/stop automatic playback |
| `gen_epochs()` | Generate epochs from current file |

### Cleaner Page (`/cleaner`)

**URL:** `http://localhost:8080/cleaner`

**Pipeline Steps:**

| Step | Module | Key Operations |
|------|--------|----------------|
| 1. LOAD | - | Load BDF/EDF/SET/FIF file |
| 2. FILTER | `filters.py` | Highpass, lowpass, notch, presets |
| 3. BAD_CHANNELS | `bad_channels.py` | Auto-detect, manual mark, interpolate |
| 4. REREFERENCE | `rereferencing.py` | Average, mastoids, single, REST |
| 5. ICA | `ica.py` | Compute, auto-detect EOG/ECG, exclude |
| 6. EPOCHS | `epochs.py` | Create fixed-length epochs |
| 7. REJECT | `epochs.py` | Amplitude, flat, gradient criteria |
| 8. VISUALIZE | - | Preview final result |
| 9. EXPORT | `export.py` | FIF, SET, EDF, NPY, CSV + logs |

**Step Navigation:**
- Sidebar shows all steps with completion indicators
- Each step has "Apply", "Reset", "Previous", "Next" buttons
- Undo available via state history
- Re-running a step invalidates all subsequent steps

### Pipeline Page (`/pipeline`)

**URL:** `http://localhost:8080/pipeline`

Executes batch processing scripts from `pipeline/`:

| Step | Script | Description |
|------|--------|-------------|
| Forward Model | `fwd.py` | Compute source space forward model |
| Multipool | `multi2pool2.py` | Parallel phase extraction |
| Order | `generate_order.py` | Kuramoto order parameter |
| Clustering | `clustering.py` | K-means clustering of brain states |
| Build Order | `build_order_all.py` | Aggregate order metrics |
| Pearson | `pearson_condition_comparison.py` | Cross-condition correlations |

---

## Configuration

### `config.py` - Global Settings

```python
# Directory paths (relative to project root)
BASE_DIR = Path(__file__).resolve().parent.parent
EEG_RAW_DIR = BASE_DIR / "EEG"        # Raw EEG files
EEG_CLEAN_DIR = BASE_DIR / "EEG_CLEAN"  # Cleaned EEG files
RESULTS_DIR = BASE_DIR / "fwd-inv-stc"  # Analysis results

# Supported file formats
SUPPORTED_FORMATS = [".bdf", ".set", ".edf", ".fif"]

# Default sampling frequency
DEFAULT_SFREQ = 512

# UI Theme - Terminal/Konsole style
THEME_BG = "#0a0a0a"           # Deep black background
THEME_CARD = "#111111"         # Card background
THEME_BORDER = "#1e1e1e"       # Subtle borders
THEME_PRIMARY = "#00ff88"      # Terminal green (main accent)
THEME_SECONDARY = "#00d4ff"    # Cyan (secondary accent)
THEME_WARN = "#ffcc00"         # Amber/yellow
THEME_ERROR = "#ff5555"        # Red for errors
THEME_TEXT = "#c8c8c8"         # Light gray text
THEME_TEXT_DIM = "#666666"     # Dimmed text

# Signal colors for multi-channel display
SIGNAL_COLORS = [
    '#00ff88',  # Terminal green
    '#00d4ff',  # Cyan
    '#ffcc00',  # Yellow
    '#ff6b9d',  # Pink
    '#a78bfa',  # Purple
    # ... more colors
]

# Frequency bands (Hz) - using Greek symbols
FREQ_BANDS = {
    "δ": (1, 4),     # Delta
    "θ": (4, 8),     # Theta
    "α": (8, 13),    # Alpha
    "β": (13, 30),   # Beta
    "γ": (30, 45),   # Gamma
}
```

### Customization

**Change EEG directories:**
```python
# In config.py
EEG_RAW_DIR = Path("/your/custom/raw/path")
EEG_CLEAN_DIR = Path("/your/custom/clean/path")
```

**Change port:**
```python
# In main.py, line 47
ui.run(title='EEG Viewer', port=YOUR_PORT, ...)
```

**Add custom file browser paths:**
- Use the path input in the File Browser panel
- Click the folder icon to browse
- Click + to add the path to the list

---

## API Reference

### EEGData (dataclass)

```python
@dataclass
class EEGData:
    raw: Optional[mne.io.Raw]     # MNE Raw object
    filename: str                  # File name
    filepath: Path                 # Full path
    sfreq: float                   # Sampling frequency (Hz)
    n_channels: int                # Number of channels
    n_samples: int                 # Total samples
    duration_sec: float            # Duration in seconds
    channel_names: List[str]       # Channel name list
    channel_types: Dict[str, str]  # Channel name -> type mapping
    info: Dict[str, Any]           # Additional metadata
    
    @property
    def is_loaded(self) -> bool:
        return self.raw is not None
```

### CleaningStep (enum)

```python
class CleaningStep(Enum):
    LOAD = 1
    FILTER = 2
    BAD_CHANNELS = 3
    REREFERENCE = 4
    ICA = 5
    EPOCHS = 6
    REJECT = 7
    VISUALIZE = 8
    EXPORT = 9
    
    @property
    def display_name(self) -> str: ...  # "1. Load", "2. Filter", etc.
    
    @property
    def short_name(self) -> str: ...    # "LOAD", "FILTER", etc.
```

### BadChannelResult (dataclass)

```python
@dataclass
class BadChannelResult:
    flat_channels: List[str]        # Variance < threshold
    noisy_channels: List[str]       # Variance > threshold
    uncorrelated_channels: List[str]  # Low correlation with neighbors
    all_bad: List[str]              # Union of all
    channel_stats: Dict[str, Dict]  # Per-channel statistics
```

### EpochRejectionCriteria (dataclass)

```python
@dataclass
class EpochRejectionCriteria:
    peak_to_peak_uv: float = 150.0   # Max amplitude difference
    flat_uv: float = 0.5             # Min amplitude difference
    gradient_uv_ms: float = 100.0    # Max slope
```

### ExportFormat (enum)

```python
class ExportFormat(Enum):
    FIF = "fif"     # MNE native (recommended)
    SET = "set"     # EEGLAB
    EDF = "edf"     # European Data Format
    NPY = "npy"     # NumPy array
    CSV = "csv"     # Tabular data
```

---

## Testing

### Run All Tests

```bash
cd /media/storage_hdd/dmt_fz/eeg_viewer

# Using pytest directly
pytest tests/ -v

# Using the test runner
python tests/run_tests.py
```

### Run Specific Test Modules

```bash
# Signal processing tests
pytest tests/test_signal.py -v

# Cleaning module tests
pytest tests/test_filters.py tests/test_bad_channels.py -v

# Skip slow tests (uses real EEG data)
pytest tests/ -v -m "not slow"

# Run only smoke tests
pytest tests/test_smoke.py -v
```

### Test Markers

```python
# In tests, mark slow tests:
@pytest.mark.slow
def test_with_real_data(sample_raw):
    ...
```

### Test Fixtures (`tests/conftest.py`)

| Fixture | Scope | Description |
|---------|-------|-------------|
| `sample_raw` | session | Real EEG data (30s) |
| `sample_raw_short` | session | Real EEG data (10s) |
| `synthetic_raw` | function | Fake EEG (fast, no file) |
| `synthetic_raw_with_artifacts` | function | Fake EEG with artifacts |
| `cleaning_state` | function | Fresh CleaningState |
| `loaded_state` | function | CleaningState with data |

---

## Troubleshooting

### Common Issues

**1. "No module named 'mne'" or similar import error**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/macOS
# or
call venv\Scripts\activate  # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

**2. "File not found" for EEG files**
- Check that the EEG directories exist in the parent folder:
  - `../EEG/DMT/`, `../EEG/EC/`, `../EEG/EO/`
  - `../EEG_CLEAN/DMT/`, etc.
- Use the path input to add custom directories

**3. Plots not updating**
- Check browser console for JavaScript errors
- Try refreshing the page
- Ensure the file loaded successfully (check FILE_INFO panel)

**4. "Port 8080 already in use"**
```python
# In main.py, change the port:
ui.run(port=8081, ...)  # or any available port
```

**5. ICA is slow**
- Use fewer components (15-20 instead of 25)
- Use 'picard' method instead of 'fastica'
- Filter data to 1 Hz highpass before ICA

**6. Memory issues with large files**
- Crop files before loading: `raw.crop(tmax=300)` for first 5 minutes
- Use preload=False when scanning directories
- Reduce number of channels displayed

### Windows-Specific

**✓ Windows Compatibility Status:**
- ✅ `run.bat` script provided for Windows launch
- ✅ All paths use `pathlib.Path` for cross-platform compatibility
- ✅ `eeg_loader.py` uses platform-agnostic path handling
- ✅ Signal processing modules are pure Python/NumPy
- ✅ MNE-Python works on Windows

**Potential Issues:**

1. **Pipeline page (`/pipeline`):**
   - Uses `asyncio.create_subprocess_exec()` which works on Windows
   - External scripts in `pipeline/` may have Linux-specific dependencies
   - If pipeline steps fail, check subprocess execution and paths

2. **Batch file not working:**
   - Run from Command Prompt (cmd), not PowerShell
   - Ensure Python is in PATH
   - If venv creation fails, try: `python -m venv venv` manually
   - Check Python version: `python --version` (requires 3.9+)

3. **File paths:**
   - Avoid using `~` (home directory) - use full paths instead
   - Custom paths in File Browser should work if directory exists

**Verified working on Windows:**
- File loading (BDF, EDF, SET, FIF)
- Signal processing and visualization
- Cleaning pipeline (all steps)
- Model training page (requires PyTorch)

### Getting Help

1. Check the browser console (F12) for errors
2. Check the terminal for Python tracebacks
3. Run tests to verify modules work:
   ```bash
   pytest tests/test_smoke.py -v
   ```

---

## Development

### Adding a New Cleaning Step

1. Add to `CleaningStep` enum in `cleaning/state.py`
2. Create processing module `cleaning/new_step.py`
3. Add state fields in `CleaningState`
4. Add UI render function in `cleaner_page.py`
5. Add to render dispatch in `render_step_controls()`
6. Create tests in `tests/test_new_step.py`
7. Export from `cleaning/__init__.py`

### Adding a New Visualization

1. Create figure class in `app/visualization/figures/`
2. Inherit from `BaseFigure` or create standalone
3. Add factory function `make_xxx_fig()`
4. Export from `app/visualization/__init__.py`
5. Use in pages: `ui.plotly(make_xxx_fig())`

### Code Style

- **Python 3.9+** type hints
- **Google-style** docstrings
- **Black** formatter (optional)
- **pytest** for tests
- No circular imports (core/ cannot import pages/)

---

## License

Part of the DMT_FZ research project.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024-12 | Initial modular architecture, cleaning pipeline, dual EEG comparison |

---

*Generated for LLM comprehension. This document describes the complete EEG Viewer module structure, APIs, and usage patterns.*

