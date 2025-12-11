"""
Pipeline configuration constants and paths.

Single Responsibility: Only handles configuration and path definitions.
"""
from pathlib import Path


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Base paths - calculated relative to this file
_PACKAGE_DIR = Path(__file__).parent
_APP_DIR = _PACKAGE_DIR.parent.parent
_EEG_VIEWER_DIR = _APP_DIR.parent
_PROJECT_ROOT = _EEG_VIEWER_DIR.parent

# Pipeline script directory (contains fwd.py, clustering.py, etc.)
PIPELINE_DIR = _PROJECT_ROOT / "pipeline"

# Output directories
PIPELINE_OUTPUTS = _EEG_VIEWER_DIR / "pipeline_outputs"
RESULTS_BASE = _PROJECT_ROOT / "fwd-inv-stc"

# Default input directory
DEFAULT_INPUT_DIR = _PROJECT_ROOT / "EEG_CLEAN"


# =============================================================================
# STEP CONFIGURATION
# =============================================================================

# Step definitions: script_name -> (display_name, description, output_pattern, color, estimated_time)
STEP_CONFIGS = {
    'fwd.py': {
        'name': 'Source Localization',
        'title': '// STEP_1: SOURCE_LOCALIZATION',
        'description': 'fwd.py - Forward/Inverse Solution + Metrics',
        'output': '→ run_*/phases-{subj}.pkl',
        'color': '#00ff88',  # THEME_PRIMARY
        'time': '~3-4h (o menos con max_epochs)',
    },
    'save_load_pickle.py': {
        'name': 'Consolidate Phases',
        'title': '// STEP_2: CONSOLIDATE_PHASES',
        'description': 'save_load_pickle.py - Merge phases-*.pkl into subject_phases_{cond}.pkl',
        'output': '→ run_*/{cond}/subject_phases_{cond}.pkl',
        'color': '#06b6d4',  # Cyan
        'time': '~1-2 min',
    },
    'multi2pool2.py': {
        'name': 'Network Filtering',
        'title': '// STEP_3: NETWORK_FILTERING',
        'description': 'multi2pool2.py - Filter by brain networks (DMN, FPN, etc)',
        'output': '→ run_*/order_all-{subj}.pkl',
        'color': '#00d4ff',  # THEME_SECONDARY
        'time': '~2-5 min',
    },
    'calculate_syncro.py': {
        'name': 'Sync Metrics',
        'title': '// STEP_4: SYNC_METRICS',
        'description': 'calculate_syncro.py - Calculate sync matrices & Kuramoto',
        'output': '→ run_*/{cond}/syncro-{subj}.pkl',
        'color': '#10b981',  # Emerald
        'time': '~10-30 min',
    },
    'generate_order.py': {
        'name': 'Kuramoto Order',
        'title': '// STEP_5: KURAMOTO_ORDER',
        'description': 'generate_order.py - Calculate Kuramoto order parameter',
        'output': '→ run_*/order-{subj}.pkl',
        'color': '#ffcc00',  # THEME_WARN
        'time': '~1 min',
    },
    'build_order_data.py': {
        'name': 'Aggregate Data',
        'title': '// STEP_6: AGGREGATE_DATA',
        'description': 'build_order_data.py - Aggregate Kuramoto metrics',
        'output': '-> r_kuramoto_nets_*.pkl',
        'color': '#60a5fa',  # Blue
        'time': '~2-5 min',
    },
    'pearson.py': {
        'name': 'Correlations',
        'title': '// STEP_7: CORRELATIONS',
        'description': 'pearson.py - Correlate with questionnaires',
        'output': '-> pearson_results/',
        'color': '#a78bfa',  # Purple
        'time': '~3-5 min',
    },
    'clustering.py': {
        'name': 'Clustering',
        'title': '// STEP_8: CLUSTERING (OPTIONAL)',
        'description': 'clustering.py - Brain state identification',
        'output': '→ run_*/clustering_results/',
        'color': '#ff6b9d',  # Pink
        'time': 'Quick: ~30min, Full: ~4h',
    },
}


# =============================================================================
# DEFAULT VALUES
# =============================================================================

# Global parameters defaults
DEFAULT_MAX_SUBJECTS = 0  # 0 = all
DEFAULT_MAX_EPOCHS = 0    # 0 = all
DEFAULT_WORKERS = 7
DEFAULT_JOBS = 0          # 0 = auto

# Clustering defaults
DEFAULT_MIN_K = 2
DEFAULT_MAX_K = 15
DEFAULT_MIN_PCA = 2
DEFAULT_MAX_PCA = 10

# Available conditions
CONDITIONS = ['DMT', 'EC', 'EO']

# Available frequency bands
BANDS = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
BAND_SYMBOLS = {'Delta': 'δ', 'Theta': 'θ', 'Alpha': 'α', 'Beta': 'β', 'Gamma': 'γ'}


# =============================================================================
# UI CONFIGURATION
# =============================================================================

# Panel widths
LEFT_PANEL_WIDTH = '450px'

# Log message markers that need spacing
LOG_SECTION_MARKERS = ['[RUN]', 'Starting:', '[INFO]', '[SETUP]', '===', 'COMPLETED', 'FAILED']

# Progress bar patterns to filter from logs
PROGRESS_BAR_PATTERNS = ['%|', 'it/s]', '0%|', '100%|', '█', '▌', '\r']

# Maximum log line length
MAX_LOG_LINE_LENGTH = 500

