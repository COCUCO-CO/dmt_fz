"""
Pytest fixtures for pipeline page tests.

Provides fixtures for:
- Temporary directories for pipeline runs
- Real EEG data (cleaned .set files)
- Pipeline state instances
- Mock UI elements
"""

import pytest
import sys
import shutil
from pathlib import Path
from datetime import datetime

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Real EEG data paths
EEG_CLEAN_DIR = Path("/media/storage_hdd/dmt_fz/EEG_CLEAN")
SAMPLE_SET_FILE = EEG_CLEAN_DIR / "DMT" / "S01-DMT_ICA_pruned.set"
SAMPLE_FDT_FILE = EEG_CLEAN_DIR / "DMT" / "S01-DMT_ICA_pruned.fdt"


# =============================================================================
# Directory Fixtures
# =============================================================================

@pytest.fixture
def temp_pipeline_dir(tmp_path):
    """
    Create a temporary directory structure for pipeline tests.
    
    Creates:
        tmp_path/
        ├── input/
        │   ├── DMT/
        │   ├── EC/
        │   └── EO/
        └── output/
    """
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    
    # Create input structure
    for cond in ["DMT", "EC", "EO"]:
        (input_dir / cond).mkdir(parents=True)
    
    output_dir.mkdir(parents=True)
    
    return {
        "root": tmp_path,
        "input": input_dir,
        "output": output_dir,
    }


@pytest.fixture
def temp_pipeline_dir_with_runs(temp_pipeline_dir):
    """
    Temporary directory with existing pipeline runs.
    
    Creates:
        output/
        ├── run_20241201_100000/
        │   ├── DMT/
        │   ├── EC/
        │   └── EO/
        ├── run_20241202_120000/
        │   └── ...
        └── run_20241203_150000/
            └── ...
    """
    output_dir = temp_pipeline_dir["output"]
    
    runs = [
        "run_20241201_100000",
        "run_20241202_120000", 
        "run_20241203_150000",
    ]
    
    for run_name in runs:
        run_dir = output_dir / run_name
        for cond in ["DMT", "EC", "EO"]:
            (run_dir / cond).mkdir(parents=True)
    
    temp_pipeline_dir["runs"] = [output_dir / r for r in runs]
    return temp_pipeline_dir


@pytest.fixture
def temp_pipeline_with_results(temp_pipeline_dir_with_runs):
    """
    Temporary directory with pipeline results (pkl files).
    
    Creates dummy .pkl files simulating pipeline output.
    """
    import pickle
    
    run_dir = temp_pipeline_dir_with_runs["runs"][-1]  # Use latest run
    
    # Create dummy phase files
    dummy_data = {"phases_stc": {"Alpha": [[1, 2, 3]]}, "kuramoto_stc": {"Alpha": [[0.5]]}}
    
    for cond in ["DMT", "EC"]:
        cond_dir = run_dir / cond
        for i in range(3):
            fname = cond_dir / f"phases-S0{i+1}-{cond}.pkl"
            with open(fname, 'wb') as f:
                pickle.dump(dummy_data, f)
    
    temp_pipeline_dir_with_runs["result_run"] = run_dir
    return temp_pipeline_dir_with_runs


# =============================================================================
# Real EEG Data Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def real_eeg_set_file():
    """
    Path to a real cleaned EEG .set file.
    
    Returns the path if it exists, otherwise skips the test.
    """
    if not SAMPLE_SET_FILE.exists():
        pytest.skip(f"Real EEG file not found: {SAMPLE_SET_FILE}")
    return SAMPLE_SET_FILE


@pytest.fixture
def temp_input_with_real_eeg(temp_pipeline_dir, real_eeg_set_file):
    """
    Temporary input directory with a copy of real EEG data.
    
    Copies S01-DMT_ICA_pruned.set/.fdt to temp input/DMT/
    """
    input_dmt = temp_pipeline_dir["input"] / "DMT"
    
    # Copy .set and .fdt files
    set_dest = input_dmt / SAMPLE_SET_FILE.name
    fdt_dest = input_dmt / SAMPLE_FDT_FILE.name
    
    shutil.copy(SAMPLE_SET_FILE, set_dest)
    shutil.copy(SAMPLE_FDT_FILE, fdt_dest)
    
    temp_pipeline_dir["real_eeg"] = set_dest
    return temp_pipeline_dir


@pytest.fixture
def temp_input_with_multiple_subjects(temp_pipeline_dir):
    """
    Temporary input directory with multiple subjects across conditions.
    
    Copies real EEG files for multiple subjects and conditions.
    """
    input_dir = temp_pipeline_dir["input"]
    
    # Copy files for each condition
    conditions_files = {
        "DMT": ["S01-DMT_ICA_pruned", "S02-DMT_ICA_pruned"],
        "EC": [],  # Will be populated if files exist
        "EO": [],
    }
    
    # Check what's available in EC and EO
    ec_dir = EEG_CLEAN_DIR / "EC"
    eo_dir = EEG_CLEAN_DIR / "EO"
    
    if ec_dir.exists():
        ec_files = list(ec_dir.glob("*.set"))[:2]
        for f in ec_files:
            conditions_files["EC"].append(f.stem)
    
    if eo_dir.exists():
        eo_files = list(eo_dir.glob("*.set"))[:2]
        for f in eo_files:
            conditions_files["EO"].append(f.stem)
    
    copied = {}
    for cond, files in conditions_files.items():
        cond_input = input_dir / cond
        cond_source = EEG_CLEAN_DIR / cond
        copied[cond] = []
        
        for fname in files:
            set_src = cond_source / f"{fname}.set"
            fdt_src = cond_source / f"{fname}.fdt"
            
            if set_src.exists() and fdt_src.exists():
                shutil.copy(set_src, cond_input / set_src.name)
                shutil.copy(fdt_src, cond_input / fdt_src.name)
                copied[cond].append(cond_input / set_src.name)
    
    temp_pipeline_dir["copied_files"] = copied
    return temp_pipeline_dir


# =============================================================================
# Pipeline State Fixtures
# =============================================================================

@pytest.fixture
def fresh_pipeline_state():
    """
    Fresh PipelineState instance with default values.
    """
    from app.state import PipelineState
    return PipelineState()


@pytest.fixture
def pipeline_state_running(fresh_pipeline_state):
    """
    PipelineState configured as if a task is running.
    """
    fresh_pipeline_state.running = True
    fresh_pipeline_state.current_step = "Test Step"
    fresh_pipeline_state.running_task_name = "test_script.py"
    fresh_pipeline_state.start_time = datetime.now()
    return fresh_pipeline_state


@pytest.fixture
def pipeline_state_with_history(fresh_pipeline_state):
    """
    PipelineState with log history populated.
    """
    fresh_pipeline_state.log_history = [
        "[RUN] Starting pipeline",
        "[SETUP] Loading data...",
        "[1/4] Processing step 1",
        "[2/4] Processing step 2",
        "[INFO] Completed successfully",
    ]
    return fresh_pipeline_state


@pytest.fixture
def pipeline_state_with_run_selected(fresh_pipeline_state, temp_pipeline_dir_with_runs):
    """
    PipelineState with a run directory selected.
    """
    fresh_pipeline_state.selected_run = temp_pipeline_dir_with_runs["runs"][-1]
    return fresh_pipeline_state


# =============================================================================
# Viz State Fixtures
# =============================================================================

@pytest.fixture
def fresh_viz_state():
    """
    Fresh visualization state dictionary.
    """
    return {
        'data': None,
        'file': None,
        'loaded': False,
        'custom_path': None,
        'current_brain_plot': 'network',
    }


@pytest.fixture
def viz_state_with_data(fresh_viz_state, temp_pipeline_with_results):
    """
    Visualization state with loaded data.
    """
    import pickle
    
    run_dir = temp_pipeline_with_results["result_run"]
    data_file = run_dir / "DMT" / "phases-S01-DMT.pkl"
    
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    fresh_viz_state['data'] = data
    fresh_viz_state['file'] = data_file
    fresh_viz_state['loaded'] = True
    
    return fresh_viz_state


# =============================================================================
# Environment Fixtures  
# =============================================================================

@pytest.fixture
def pipeline_env(temp_pipeline_dir, monkeypatch):
    """
    Set up environment variables for pipeline execution.
    """
    monkeypatch.setenv("PIPELINE_INPUT_DIR", str(temp_pipeline_dir["input"]))
    monkeypatch.setenv("PIPELINE_OUTPUT_DIR", str(temp_pipeline_dir["output"]))
    return temp_pipeline_dir


@pytest.fixture
def clean_pipeline_globals():
    """
    Reset global pipeline state before/after test.
    
    Ensures tests don't affect each other through global state.
    """
    from app.state import PS
    
    # Save original state
    original = {
        'running': PS.running,
        'current_step': PS.current_step,
        'running_task_name': PS.running_task_name,
        'start_time': PS.start_time,
        'current_process': PS.current_process,
        'log_history': PS.log_history.copy(),
        'selected_run': PS.selected_run,
    }
    
    # Reset state
    PS.running = False
    PS.current_step = ""
    PS.running_task_name = ""
    PS.start_time = None
    PS.current_process = None
    PS.log_history = []
    PS.selected_run = None
    PS.log_container = None
    PS.log_scroll = None
    
    yield PS
    
    # Restore original state
    for key, value in original.items():
        setattr(PS, key, value)









