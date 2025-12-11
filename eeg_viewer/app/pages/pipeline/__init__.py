"""
Pipeline page package.

This package implements the Pipeline page for EEG processing following SOLID principles:
- Single Responsibility: Each module handles one concern
- Open/Closed: Steps can be extended without modifying existing code
- Liskov Substitution: All steps inherit from BasePipelineStep
- Interface Segregation: Small, focused interfaces
- Dependency Inversion: High-level modules don't depend on low-level modules

Structure:
    pipeline/
    ├── __init__.py          # Package exports
    ├── config.py            # Constants and paths
    ├── utils.py             # Utility functions
    ├── runner.py            # Pipeline execution engine
    ├── page.py              # Main page orchestrator
    ├── steps/               # Pipeline step definitions
    │   ├── __init__.py
    │   ├── base.py          # Abstract base class
    │   └── step_*.py        # Individual step implementations
    ├── tabs/                # Tab panel implementations
    │   ├── __init__.py
    │   ├── console.py
    │   ├── files.py
    │   ├── system.py
    │   └── visualize.py
    └── components/          # Reusable UI components
        ├── __init__.py
        ├── header.py
        ├── io_config.py
        └── global_params.py
"""

# Page entry points
from .page import pipeline_page, pipeline_page_route

# Config exports (for tests and backwards compatibility)
from .config import (
    PIPELINE_DIR,
    PIPELINE_OUTPUTS,
    RESULTS_BASE,
    DEFAULT_INPUT_DIR,
    STEP_CONFIGS,
)

# Utility exports (for tests and backwards compatibility)
from .utils import (
    get_run_dirs,
    create_new_run,
    pipeline_log,
    scan_input_directory,
    count_output_files,
    find_subjects_in_directory,
)

# Runner exports (for tests and backwards compatibility)
from .runner import run_pipeline_step, stop_current_process


__all__ = [
    # Page
    'pipeline_page',
    'pipeline_page_route',
    # Config
    'PIPELINE_DIR',
    'PIPELINE_OUTPUTS',
    'RESULTS_BASE',
    'DEFAULT_INPUT_DIR',
    'STEP_CONFIGS',
    # Utils
    'get_run_dirs',
    'create_new_run',
    'pipeline_log',
    'scan_input_directory',
    'count_output_files',
    'find_subjects_in_directory',
    # Runner
    'run_pipeline_step',
    'stop_current_process',
]

