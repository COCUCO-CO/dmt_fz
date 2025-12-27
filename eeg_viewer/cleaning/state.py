"""
Cleaning Pipeline State Management

Manages the state of the EEG cleaning process including:
- Current step in the pipeline
- Applied operations history
- Undo/redo functionality
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime
from pathlib import Path
import mne


class CleaningStep(Enum):
    """Steps in the cleaning pipeline."""
    LOAD = auto()
    FILTER = auto()
    BAD_CHANNELS = auto()
    REREFERENCE = auto()
    ICA = auto()
    EPOCHS = auto()
    REJECT = auto()
    VISUALIZE = auto()  # New: preview final result
    EXPORT = auto()
    
    @property
    def display_name(self) -> str:
        names = {
            CleaningStep.LOAD: "1. Load",
            CleaningStep.FILTER: "2. Filter",
            CleaningStep.BAD_CHANNELS: "3. Bad Channels",
            CleaningStep.REREFERENCE: "4. Re-Reference",
            CleaningStep.ICA: "5. ICA",
            CleaningStep.EPOCHS: "6. Epochs",
            CleaningStep.REJECT: "7. Reject",
            CleaningStep.VISUALIZE: "8. Visualize",
            CleaningStep.EXPORT: "9. Export",
        }
        return names.get(self, self.name)
    
    @property
    def short_name(self) -> str:
        names = {
            CleaningStep.LOAD: "LOAD",
            CleaningStep.FILTER: "FILTER",
            CleaningStep.BAD_CHANNELS: "BAD_CH",
            CleaningStep.REREFERENCE: "RE-REF",
            CleaningStep.ICA: "ICA",
            CleaningStep.EPOCHS: "EPOCHS",
            CleaningStep.REJECT: "REJECT",
            CleaningStep.VISUALIZE: "VIEW",
            CleaningStep.EXPORT: "EXPORT",
        }
        return names.get(self, self.name)


@dataclass
class OperationRecord:
    """Record of an operation applied to the EEG."""
    step: CleaningStep
    operation: str
    parameters: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'step': self.step.name,
            'operation': self.operation,
            'parameters': self.parameters,
            'timestamp': self.timestamp.isoformat(),
            'description': self.description,
        }


class CleaningState:
    """
    Manages the state of the EEG cleaning pipeline.
    
    Provides:
    - Current raw object management
    - History of operations for undo
    - Step tracking
    - Results storage for each step
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all state to initial values."""
        # Raw data management
        self._raw: Optional[mne.io.Raw] = None
        self._raw_original: Optional[mne.io.Raw] = None  # Backup for full reset
        self._raw_history: List[mne.io.Raw] = []  # For undo
        self._max_history: int = 10
        
        # File info
        self.filepath: Optional[Path] = None
        self.filename: str = ""
        
        # Current step
        self.current_step: CleaningStep = CleaningStep.LOAD
        self.completed_steps: set = set()
        
        # Operation history (for logging)
        self.operations: List[OperationRecord] = []
        
        # Step-specific results
        self.filter_params: Dict[str, Any] = {}
        self.bad_channels: List[str] = []
        self.bad_channels_auto: Optional[List[str]] = None  # Auto-detected (None = not run yet)
        self.bad_channels_manual: List[str] = []  # Manually marked
        self.interpolated_channels: List[str] = []
        self.reference_type: str = ""
        self.reference_channels: List[str] = []
        
        # ICA
        self.ica: Optional[Any] = None  # mne.preprocessing.ICA
        self.ica_excluded: List[int] = []
        self.ica_labels: Dict[int, str] = {}  # Component -> label (blink, heart, etc)
        
        # Epochs
        self.epochs: Optional[Any] = None  # mne.Epochs
        self.epoch_duration: float = 2.0
        self.epochs_rejected: List[int] = []
        self.epochs_total: int = 0
        self.rejection_criteria: Dict[str, float] = {}
        
        # UI callbacks
        self._on_change_callbacks: List[Callable] = []
    
    @property
    def raw(self) -> Optional[mne.io.Raw]:
        """Current raw EEG data."""
        return self._raw
    
    @property
    def is_loaded(self) -> bool:
        """Check if EEG data is loaded."""
        return self._raw is not None
    
    @property
    def sfreq(self) -> float:
        """Sampling frequency."""
        return self._raw.info['sfreq'] if self._raw else 0.0
    
    @property
    def n_channels(self) -> int:
        """Number of channels."""
        return len(self._raw.ch_names) if self._raw else 0
    
    @property
    def ch_names(self) -> List[str]:
        """Channel names."""
        return list(self._raw.ch_names) if self._raw else []
    
    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self._raw.n_times / self._raw.info['sfreq'] if self._raw else 0.0
    
    @property
    def can_undo(self) -> bool:
        """Check if undo is available."""
        return len(self._raw_history) > 0
    
    def load_raw(self, raw: mne.io.Raw, filepath: Optional[Path] = None):
        """Load raw EEG data into the state."""
        self._raw = raw.copy()
        self._raw_original = raw.copy()
        self._raw_history = []
        self.filepath = filepath
        self.filename = filepath.name if filepath else "Unknown"
        self.current_step = CleaningStep.LOAD
        self.completed_steps.add(CleaningStep.LOAD)
        
        self._add_operation(
            CleaningStep.LOAD,
            "load_file",
            {'filepath': str(filepath) if filepath else None},
            f"Loaded {self.filename}"
        )
        self._notify_change()
    
    def _save_to_history(self):
        """Save current state to history for undo."""
        if self._raw is not None:
            self._raw_history.append(self._raw.copy())
            # Keep only last N states
            if len(self._raw_history) > self._max_history:
                self._raw_history.pop(0)
    
    def update_raw(self, new_raw: mne.io.Raw, operation: str, 
                   step: CleaningStep, params: Dict[str, Any],
                   description: str = ""):
        """Update raw data and record the operation."""
        self._save_to_history()
        self._raw = new_raw
        self._add_operation(step, operation, params, description)
        self._notify_change()
    
    def undo(self) -> bool:
        """Undo last operation. Returns True if successful."""
        if not self.can_undo:
            return False
        
        self._raw = self._raw_history.pop()
        if self.operations:
            self.operations.pop()
        self._notify_change()
        return True
    
    def reset_to_original(self):
        """Reset to originally loaded data."""
        if self._raw_original is not None:
            self._raw = self._raw_original.copy()
            self._raw_history = []
            self.operations = [self.operations[0]] if self.operations else []
            self.completed_steps = {CleaningStep.LOAD}
            self.current_step = CleaningStep.LOAD
            
            # Reset step-specific data
            self.bad_channels = []
            self.bad_channels_auto = None  # None means detection not run yet
            self.bad_channels_manual = []
            self.interpolated_channels = []
            self.ica = None
            self.ica_excluded = []
            self.ica_labels = {}
            self.epochs = None
            self.epochs_rejected = []
            
            self._notify_change()
    
    def _add_operation(self, step: CleaningStep, operation: str,
                       params: Dict[str, Any], description: str = ""):
        """Add operation to history."""
        record = OperationRecord(
            step=step,
            operation=operation,
            parameters=params,
            description=description or operation
        )
        self.operations.append(record)
    
    def go_to_step(self, step: CleaningStep):
        """Navigate to a specific step."""
        self.current_step = step
        self._notify_change()
    
    def complete_step(self, step: CleaningStep):
        """Mark a step as completed."""
        self.completed_steps.add(step)
        # Auto-advance to next step
        steps = list(CleaningStep)
        current_idx = steps.index(step)
        if current_idx < len(steps) - 1:
            self.current_step = steps[current_idx + 1]
        self._notify_change()
    
    def on_change(self, callback: Callable):
        """Register a callback for state changes."""
        self._on_change_callbacks.append(callback)
    
    def _notify_change(self):
        """Notify all listeners of state change."""
        for callback in self._on_change_callbacks:
            try:
                callback()
            except Exception as e:
                print(f"Error in state change callback: {e}")
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get summary of all preprocessing steps applied."""
        return {
            'filename': self.filename,
            'original_sfreq': self._raw_original.info['sfreq'] if self._raw_original else None,
            'current_sfreq': self.sfreq,
            'original_channels': len(self._raw_original.ch_names) if self._raw_original else 0,
            'current_channels': self.n_channels,
            'duration_sec': self.duration,
            'filter_params': self.filter_params,
            'bad_channels': self.bad_channels,
            'interpolated_channels': self.interpolated_channels,
            'reference_type': self.reference_type,
            'ica_components_excluded': self.ica_excluded,
            'ica_labels': self.ica_labels,
            'epochs_total': self.epochs_total,
            'epochs_rejected': len(self.epochs_rejected),
            'rejection_criteria': self.rejection_criteria,
            'operations': [op.to_dict() for op in self.operations],
        }

