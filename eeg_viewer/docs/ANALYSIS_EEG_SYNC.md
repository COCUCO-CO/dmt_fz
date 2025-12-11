# EEG-Autoencoder Synchronization Feature

## Overview

This document describes the implementation plan to synchronize the EEG visualizer with the autoencoder analysis playback in `analysis.py`.

## Objective

When processing a graph dataset through the autoencoder, show the corresponding EEG segment that was used to create each graph sample. This allows visualization of what EEG data the autoencoder is analyzing in real-time.

## Architecture

### Data Flow

```
Dataset Sample (Graph) → Contains metadata:
  - subject_id: "S01"
  - condition: "DMT" 
  - band: "Alpha"
  - epoch_idx: 42

↓

EEG File Mapping:
  EEG_CLEAN/{condition}/{subject_id}-{condition}_ICA_pruned.set
  Example: EEG_CLEAN/DMT/S01-DMT_ICA_pruned.set

↓

EEG Viewer:
  view_start = epoch_idx * epoch_duration
  view_duration = epoch_duration (default 2.0s)
```

### Components to Modify

1. **`app/state/global_state.py`** - Add EEG sync fields to `AnalysisState`
2. **`app/pages/analysis.py`** - Add EEG visualization panel and sync logic
3. **`app/pages/viewer.py`** - Extract reusable functions for EEG visualization

## Implementation Plan

### Phase 1: State Extension

Add to `AnalysisState`:
- `eeg_data`: Loaded EEG file for current subject/condition
- `eeg_sync_enabled`: Toggle for EEG synchronization
- `epoch_duration`: Duration of each epoch (default 2.0s)
- `current_eeg_file`: Path to currently loaded EEG
- `eeg_channels`: Selected channels for visualization
- UI references for EEG plot

### Phase 2: EEG Visualization Module

Create reusable EEG visualization components by refactoring `viewer.py`:
- Extract `make_eeg_fig()` as standalone
- Extract `_update_eeg_generic()` as pure function
- Create helper to find EEG file from graph metadata

### Phase 3: Sync Integration

In `analysis.py`:
1. When loading dataset, detect if it's a graph dataset with EEG metadata
2. Add EEG panel above the existing visualizations
3. On sample change:
   - Extract `subject_id`, `condition`, `epoch_idx` from sample
   - If different subject/condition: load corresponding EEG file
   - Calculate view_start = epoch_idx * epoch_duration
   - Update EEG visualization

### Phase 4: UI Integration

- Add toggle to enable/disable EEG sync
- Share playback controls (don't duplicate)
- Add epoch duration setting
- Add channel selection for EEG view

## Test Plan

### Unit Tests

1. **test_eeg_file_finder.py**
   - Test finding EEG file from graph metadata
   - Test handling missing files gracefully
   - Test different filename formats (S01-DMT vs S01_DMT)

2. **test_eeg_sync_state.py**
   - Test AnalysisState EEG sync fields initialization
   - Test EEG data loading integration

3. **test_epoch_calculation.py**
   - Test epoch_idx to time conversion
   - Test edge cases (first epoch, last epoch)
   - Test different epoch durations

### Integration Tests

4. **test_analysis_eeg_sync.py**
   - Test EEG sync with mock graph dataset
   - Test sample navigation updates EEG view
   - Test playback synchronization

### Test Files Location

All tests will be in: `eeg_viewer/tests/analysis/`

## Files to Create/Modify

### New Files
- `eeg_viewer/app/visualization/components/eeg_sync.py` - EEG sync utilities
- `eeg_viewer/tests/analysis/__init__.py`
- `eeg_viewer/tests/analysis/conftest.py`
- `eeg_viewer/tests/analysis/test_eeg_file_finder.py`
- `eeg_viewer/tests/analysis/test_eeg_sync_state.py`
- `eeg_viewer/tests/analysis/test_epoch_calculation.py`
- `eeg_viewer/tests/analysis/test_analysis_eeg_sync.py`

### Modified Files
- `eeg_viewer/app/state/global_state.py` - Add EEG sync fields
- `eeg_viewer/app/pages/analysis.py` - Add EEG panel and sync logic

## Implementation Order

1. ✅ Create documentation (this file)
2. ✅ Create test directory structure
3. ✅ Implement unit tests (35 tests)
4. ✅ Implement `eeg_sync.py` utilities
5. ✅ Update `AnalysisState`
6. ✅ Add EEG panel to `analysis.py`
7. ✅ Implement sync logic
8. ✅ Run all tests (55 passed)
9. Manual testing with real data

## Assumptions

- Epoch duration is 2.0 seconds (configurable)
- EEG files are in `EEG_CLEAN/{condition}/` directory
- File naming: `{subject_id}-{condition}_ICA_pruned.set` or similar
- Graph samples have `subject_id`, `condition`, `epoch_idx` attributes

## Edge Cases

1. **Missing EEG file**: Show placeholder, log warning
2. **Invalid epoch_idx**: Clamp to valid range
3. **Different sample rates**: Handle in visualization
4. **No EEG metadata in dataset**: Disable sync, show message

