"""
EEG Cleaning Pipeline Module

Provides tools for EEG preprocessing:
- Bad channel detection and interpolation
- Re-referencing
- ICA for artifact removal
- Epoch rejection
- Export utilities
"""

from .state import CleaningState, CleaningStep
from .filters import FilterPreset, apply_filter_preset, get_filter_presets
from .bad_channels import (
    detect_bad_channels,
    interpolate_channels,
    BadChannelResult
)
from .rereferencing import (
    ReferenceType,
    apply_reference,
    get_available_references
)
from .ica import (
    compute_ica,
    detect_eog_components,
    detect_ecg_components,
    apply_ica_exclusion,
    ICAResult
)
from .epochs import (
    create_epochs,
    detect_bad_epochs,
    EpochRejectionCriteria,
    EpochResult
)
from .export import (
    export_cleaned_eeg,
    export_epochs,
    export_preprocessing_log,
    ExportFormat
)

__all__ = [
    # State
    'CleaningState',
    'CleaningStep',
    # Filters
    'FilterPreset',
    'apply_filter_preset',
    'get_filter_presets',
    # Bad channels
    'detect_bad_channels',
    'interpolate_channels',
    'BadChannelResult',
    # Re-referencing
    'ReferenceType',
    'apply_reference',
    'get_available_references',
    # ICA
    'compute_ica',
    'detect_eog_components',
    'detect_ecg_components',
    'apply_ica_exclusion',
    'ICAResult',
    # Epochs
    'create_epochs',
    'detect_bad_epochs',
    'EpochRejectionCriteria',
    'EpochResult',
    # Export
    'export_cleaned_eeg',
    'export_epochs',
    'export_preprocessing_log',
    'ExportFormat',
]






