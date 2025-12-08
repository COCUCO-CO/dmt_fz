# EEG Cleaning Pipeline

## Overview

This module implements a step-by-step EEG preprocessing pipeline with a NiceGUI web interface. Each step receives the output of the previous step, processes it, and passes the result forward.

## Pipeline Steps

```
LOAD -> FILTER -> BAD_CHANNELS -> REREFERENCE -> ICA -> EPOCHS -> REJECT -> VISUALIZE -> EXPORT
```

| Step | Input | Output | Description |
|------|-------|--------|-------------|
| LOAD | File path | mne.io.Raw | Load BDF/EDF/SET/FIF file |
| FILTER | Raw | Raw | Apply bandpass and notch filters |
| BAD_CHANNELS | Raw | Raw | Detect and interpolate bad channels |
| REREFERENCE | Raw | Raw | Apply reference (average, mastoids, REST) |
| ICA | Raw | Raw | Remove artifact components (EOG, ECG, muscle) |
| EPOCHS | Raw | Epochs | Segment continuous data into fixed-length epochs |
| REJECT | Epochs | Epochs | Detect and remove bad epochs |
| VISUALIZE | Epochs | Epochs | Preview final data before export |
| EXPORT | Epochs/Raw | Files | Save to FIF/SET/EDF/NPY/CSV |

## Architecture

### Core Files

```
cleaning/
  __init__.py          # Module exports
  state.py             # CleaningState, CleaningStep enum, OperationRecord
  filters.py           # Filter functions and presets
  bad_channels.py      # Bad channel detection and interpolation
  rereferencing.py     # Reference application
  ica.py               # ICA computation and artifact detection
  epochs.py            # Epoch creation and rejection
  export.py            # Export functions for multiple formats
  cleaner_page.py      # NiceGUI interface and CleanerPageState
```

### State Management

Two state classes manage the pipeline:

#### CleaningState (state.py)

Core EEG processing state. Holds:

```python
class CleaningState:
    raw: mne.io.Raw              # Current EEG data
    _raw_original: mne.io.Raw    # Original loaded data (never modified)
    _raw_history: List[Raw]      # History stack for undo
    
    current_step: CleaningStep   # Current pipeline position
    completed_steps: Set[Step]   # Steps marked as done
    
    # Step results
    filter_params: FilterParams
    bad_channels: List[str]
    bad_channels_auto: List[str]
    interpolated_channels: List[str]
    reference_type: str
    ica: ICA
    ica_excluded: List[int]
    epochs: mne.Epochs
    epochs_rejected: List[int]
```

Key methods:
- `load_raw(filepath)` - Load EEG file
- `update_raw(raw, op_type, step, params, description)` - Update data and log operation
- `undo()` - Revert last operation
- `reset_to_original()` - Reset to loaded state
- `complete_step(step)` - Mark step as done
- `get_preprocessing_summary()` - Get dict of all applied operations

#### CleanerPageState (cleaner_page.py)

UI-specific state. Holds:

```python
class CleanerPageState:
    cleaning: CleaningState      # Core state
    
    # UI snapshots (raw data at each step completion)
    _step_raw_snapshots: Dict[CleaningStep, Raw]
    
    # Display options
    show_bad_channels_highlight: bool
    show_rejected_epochs_overlay: bool
    
    # Per-step reverse tracking
    reversed_steps: Set[CleaningStep]
    
    # Input field persistence
    epoch_duration: float
    epoch_overlap: float
    reject_ptp_threshold: float
    # ... etc
    
    # Operation result indicators
    _export_all_result: bool
    _rejection_applied: bool
    # ... etc
```

### Data Flow

Each step follows this pattern:

```
1. Get input from previous step: raw = get_step_input(current_step)
2. Apply operation: new_raw = apply_operation(raw, params)
3. Update state: PS.cleaning.update_raw(new_raw, ...)
4. Save snapshot: PS.save_step_raw(current_step)
5. Invalidate subsequent steps: invalidate_subsequent_steps(current_step)
6. Refresh UI: render_step_controls(), update_main_plot()
```

The `get_step_input(step)` function returns the output of the previous step:

```python
def get_step_input(step: CleaningStep) -> mne.io.Raw:
    step_order = list(CleaningStep)
    idx = step_order.index(step)
    if idx == 0:
        return PS.cleaning._raw_original.copy()
    prev_step = step_order[idx - 1]
    if prev_step in PS._step_raw_snapshots:
        return PS._step_raw_snapshots[prev_step].copy()
    return PS.cleaning._raw_original.copy()
```

### Snapshots

When a step completes, its output is saved:

```python
PS._step_raw_snapshots[CleaningStep.FILTER] = filtered_raw.copy()
```

This allows:
- Viewing the correct EEG state when navigating between steps
- Resetting a step to its input state
- Reversing data at specific steps

### Invalidation

When a step is modified, all subsequent steps lose their completed status:

```python
def invalidate_subsequent_steps(step: CleaningStep):
    step_order = list(CleaningStep)
    idx = step_order.index(step)
    for s in step_order[idx:]:
        PS.cleaning.completed_steps.discard(s)
        PS._step_raw_snapshots.pop(s, None)
        PS.reversed_steps.discard(s)
```

## Key Data Types

### FilterParams (filters.py)

```python
@dataclass
class FilterParams:
    highpass: float = 0.1      # Hz
    lowpass: float = 45.0      # Hz
    notch: float = 50.0        # Hz (50 or 60)
    notch_width: float = 2.0   # Hz
```

### BadChannelResult (bad_channels.py)

```python
@dataclass
class BadChannelResult:
    flat_channels: List[str]
    noisy_channels: List[str]
    uncorrelated_channels: List[str]
    all_bad: List[str]
    channel_stats: Dict[str, Dict]  # variance, correlation per channel
```

### ICAResult (ica.py)

```python
@dataclass
class ICAResult:
    ica: mne.preprocessing.ICA
    n_components: int
    eog_indices: List[int]
    ecg_indices: List[int]
    muscle_indices: List[int]
    excluded: List[int]         # User-selected for removal
```

### EpochResult (epochs.py)

```python
@dataclass
class EpochResult:
    epochs: mne.Epochs
    n_total: int
    n_good: int
    n_rejected: int
    rejected_indices: List[int]
    epoch_stats: Dict            # min_std, max_ptp, max_gradient per epoch
    channel_rejection_stats: List[EpochChannelRejectionStats]
```

### EpochRejectionCriteria (epochs.py)

```python
@dataclass
class EpochRejectionCriteria:
    peak_to_peak_uv: float = 150.0   # Max amplitude difference
    flat_uv: float = 0.5             # Min amplitude difference
    gradient_uv_ms: float = 100.0    # Max slope
```

## Adding a New Pipeline Step

### 1. Add to CleaningStep enum (state.py)

```python
class CleaningStep(Enum):
    LOAD = 1
    FILTER = 2
    # ... existing steps ...
    NEW_STEP = 10  # Add here in order
    EXPORT = 11    # Keep EXPORT last

    @property
    def display_name(self):
        names = {
            # ... existing ...
            CleaningStep.NEW_STEP: '10. New Step',
        }
        return names.get(self, self.name)

    @property
    def short_name(self):
        return self.name.replace('_', ' ').title()
```

### 2. Create processing module (new_step.py)

```python
from dataclasses import dataclass
from typing import List, Optional
import mne

@dataclass
class NewStepResult:
    """Result of new step operation."""
    # Define output fields
    processed_data: mne.io.Raw
    stats: dict

def apply_new_step(raw: mne.io.Raw, param1: float, param2: str) -> NewStepResult:
    """
    Apply new step operation.
    
    Args:
        raw: Input EEG data (mne.io.Raw)
        param1: Description
        param2: Description
    
    Returns:
        NewStepResult with processed data and stats
    """
    # Process
    processed = raw.copy()
    # ... do processing ...
    
    return NewStepResult(
        processed_data=processed,
        stats={'key': 'value'}
    )
```

### 3. Add state fields (state.py)

In `CleaningState.__init__`:

```python
# New step results
self.new_step_result: Optional[NewStepResult] = None
self.new_step_param1: float = 1.0
```

### 4. Add UI controls (cleaner_page.py)

```python
def render_new_step_controls():
    """Render new step controls."""
    if not PS.cleaning.is_loaded:
        show_not_loaded_message()
        return
    
    ui.label('Description of this step.').style(
        f'color: {THEME_TEXT_DIM}; font-size: 0.8rem; margin-bottom: 16px;'
    )
    
    # Parameters
    with ui.element('div').classes('control-section'):
        param1 = ui.number('Param 1', value=1.0, min=0, max=10).props('dense outlined')
        param1.tooltip('What this parameter does')
    
    # Results display
    if PS.cleaning.new_step_result:
        with ui.row().classes('gap-4 mt-4'):
            stat_card('Stat', str(PS.cleaning.new_step_result.stats['key']), THEME_PRIMARY)
    
    # Action bar
    with ui.element('div').classes('action-bar'):
        ui.button('Apply', icon='check', on_click=lambda: apply_new_step_async(param1.value)).props('dense')
        ui.button('Reset Step', icon='replay', on_click=lambda: reset_step(CleaningStep.NEW_STEP)).props('flat dense').classes('reset-btn')
        ui.element('div').classes('flex-1')
        ui.button('Previous', icon='arrow_back', on_click=go_to_previous_step).props('flat dense')
        ui.button('Next Step', icon='arrow_forward', on_click=lambda: complete_step(CleaningStep.NEW_STEP)).props('dense')
```

### 5. Add to render dispatch (cleaner_page.py)

In `render_step_controls()`:

```python
elif step == CleaningStep.NEW_STEP:
    render_new_step_controls()
```

### 6. Add async handler if needed (cleaner_page.py)

```python
async def apply_new_step_async(param1: float):
    if not PS.cleaning.is_loaded:
        return
    try:
        # Get input from previous step
        raw_input = get_step_input(CleaningStep.NEW_STEP)
        
        # Run in executor for long operations
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: apply_new_step(raw_input, param1))
        
        # Update state
        PS.cleaning.new_step_result = result
        PS.cleaning.update_raw(result.processed_data, 'new_step', CleaningStep.NEW_STEP,
                               {'param1': param1}, f'Applied new step')
        
        # Save snapshot and invalidate subsequent
        PS.save_step_raw(CleaningStep.NEW_STEP)
        invalidate_subsequent_steps(CleaningStep.NEW_STEP)
        
        # Refresh UI
        render_step_controls()
        update_main_plot()
        safe_notify('New step applied', type='positive')
    except Exception as e:
        safe_notify(f'Error: {e}', type='negative')
```

### 7. Add reset logic (cleaner_page.py)

In `reset_step()` add case:

```python
elif step == CleaningStep.NEW_STEP:
    PS.cleaning.new_step_result = None
```

### 8. Add to load reset (cleaner_page.py)

In `load_file()` after loading:

```python
PS.cleaning.new_step_result = None
```

### 9. Add tests (tests/test_new_step.py)

```python
import pytest
from cleaning.new_step import apply_new_step, NewStepResult

class TestNewStep:
    def test_basic_operation(self, mock_raw):
        result = apply_new_step(mock_raw, param1=1.0)
        assert isinstance(result, NewStepResult)
        assert result.processed_data is not None
    
    def test_invalid_param(self, mock_raw):
        with pytest.raises(ValueError):
            apply_new_step(mock_raw, param1=-1)
```

### 10. Export from __init__.py

```python
from .new_step import apply_new_step, NewStepResult
```

## UI Patterns

### Notification with error handling

```python
def safe_notify(message: str, type: str = 'info'):
    try:
        ui.notify(message, type=type, position='bottom-right', timeout=3000)
    except:
        pass  # UI context may be gone
```

### Operation result indicators

Add state fields:

```python
self._operation_result: Optional[bool] = None  # None=not run, True=success, False=error
```

In UI:

```python
if PS._operation_result is not None:
    if PS._operation_result:
        ui.icon('check_circle', size='xs').style(f'color: {THEME_PRIMARY};')
    else:
        ui.icon('error', size='xs').style(f'color: {THEME_ERROR};')
```

### Input persistence

Store input values in CleanerPageState to survive re-renders:

```python
# In CleanerPageState.__init__
self.my_param: float = 1.0

# In render function
def on_param_change(e):
    PS.my_param = float(e.value)

ui.number('Param', value=PS.my_param, on_change=on_param_change)
```

## Testing

Run all tests:

```bash
cd /media/storage_hdd/dmt_fz/eeg_viewer
conda activate eeg_viewer
python -m pytest tests/ -v
```

Run specific module:

```bash
python tests/run_tests.py --module filters
python tests/run_tests.py --module epochs
```

Fast tests only (synthetic data):

```bash
python tests/run_tests.py --fast
```

## Dependencies

- mne >= 1.0.0
- nicegui >= 1.0.0
- plotly >= 5.0.0
- numpy
- scipy

## File Formats

### Input
- BDF (BioSemi)
- EDF (European Data Format)
- SET (EEGLAB)
- FIF (MNE)

### Output
- FIF (recommended, preserves all metadata)
- SET (EEGLAB compatible)
- EDF (clinical standard)
- NPY (numpy array)
- CSV (tabular)
- JSON (rejection stats, preprocessing log)


