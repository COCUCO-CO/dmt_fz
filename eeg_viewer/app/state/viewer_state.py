"""
Viewer state for the main EEG visualization page.

Contains all state related to EEG viewing, playback, and filtering.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING
import numpy as np

from .base import BaseState

if TYPE_CHECKING:
    import mne


@dataclass
class ViewerState(BaseState):
    """
    State for the main EEG viewer.
    
    Manages:
    - EEG data references (raw1, raw2)
    - View window (position, duration)
    - Playback state
    - Filter settings
    - Channel selection
    - Display options
    """
    # === EEG Data ===
    raw1: Optional[mne.io.Raw] = field(default=None, repr=False)
    raw2: Optional[mne.io.Raw] = field(default=None, repr=False)
    file_path1: str = ""
    file_path2: str = ""
    
    # === View Window ===
    view_start: float = 0.0
    view_duration: float = 5.0
    
    # === Playback ===
    is_playing: bool = False
    playback_speed: float = 1.0
    
    # === Filters ===
    notch_enabled: bool = False
    notch_freq: float = 50.0
    bandpass_enabled: bool = False
    bandpass_low: float = 1.0
    bandpass_high: float = 40.0
    
    # === Channel Selection ===
    selected_channels: List[str] = field(default_factory=list)
    hilbert_channel: str = ""
    
    # === FFT Settings ===
    fft_log_scale: bool = False
    fft_max_freq: float = 60.0
    
    # === Display Options ===
    show_fft: bool = True
    show_hilbert: bool = True
    show_brain: bool = True
    auto_scale: bool = True
    y_scale: float = 1.0
    
    # === Epoch Generation ===
    epoch_duration: float = 2.0
    epoch_overlap: float = 0.0
    
    def __post_init__(self):
        """Initialize observer set after dataclass init."""
        object.__setattr__(self, '_observers', set())
        object.__setattr__(self, '_notify_enabled', True)
    
    @property
    def sfreq1(self) -> float:
        """Sampling frequency of first EEG file."""
        if self.raw1 is not None:
            return self.raw1.info['sfreq']
        return 250.0
    
    @property
    def sfreq2(self) -> float:
        """Sampling frequency of second EEG file."""
        if self.raw2 is not None:
            return self.raw2.info['sfreq']
        return 250.0
    
    @property
    def duration1(self) -> float:
        """Duration in seconds of first EEG file."""
        if self.raw1 is not None:
            return self.raw1.n_times / self.raw1.info['sfreq']
        return 0.0
    
    @property
    def duration2(self) -> float:
        """Duration in seconds of second EEG file."""
        if self.raw2 is not None:
            return self.raw2.n_times / self.raw2.info['sfreq']
        return 0.0
    
    @property
    def channel_names1(self) -> List[str]:
        """Channel names from first EEG file."""
        if self.raw1 is not None:
            return self.raw1.ch_names
        return []
    
    @property
    def channel_names2(self) -> List[str]:
        """Channel names from second EEG file."""
        if self.raw2 is not None:
            return self.raw2.ch_names
        return []
    
    @property
    def has_data(self) -> bool:
        """Check if at least one EEG file is loaded."""
        return self.raw1 is not None or self.raw2 is not None
    
    @property
    def view_end(self) -> float:
        """End time of current view window."""
        return self.view_start + self.view_duration
    
    def get_view_samples(self, sfreq: float) -> tuple[int, int]:
        """Get sample indices for current view window."""
        start_sample = int(self.view_start * sfreq)
        end_sample = int(self.view_end * sfreq)
        return start_sample, end_sample
    
    def reset(self) -> None:
        """Reset to default viewing state (keeps loaded data)."""
        self.batch_update(
            view_start=0.0,
            view_duration=5.0,
            is_playing=False,
            selected_channels=[],
            hilbert_channel=""
        )
    
    def clear_data(self) -> None:
        """Clear all loaded EEG data."""
        self.batch_update(
            raw1=None,
            raw2=None,
            file_path1="",
            file_path2="",
            selected_channels=[],
            hilbert_channel=""
        )
        self.reset()


@dataclass  
class PipelineState(BaseState):
    """State for the pipeline execution page."""
    running: bool = False
    workers: int = 7
    current_step: str = ""
    progress: float = 0.0
    log_messages: List[str] = field(default_factory=list)
    
    # Pipeline configuration
    input_path: str = ""
    output_path: str = ""
    config_path: str = ""
    
    def __post_init__(self):
        object.__setattr__(self, '_observers', set())
        object.__setattr__(self, '_notify_enabled', True)
    
    def add_log(self, message: str) -> None:
        """Add a log message."""
        self.log_messages = self.log_messages + [message]
    
    def clear_logs(self) -> None:
        """Clear all log messages."""
        self.log_messages = []


@dataclass
class ModelState(BaseState):
    """State for the model training page."""
    training: bool = False
    model_type: str = "vae"
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    
    # Training progress
    current_epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    
    # Paths
    dataset_path: str = ""
    checkpoint_path: str = ""
    
    def __post_init__(self):
        object.__setattr__(self, '_observers', set())
        object.__setattr__(self, '_notify_enabled', True)


@dataclass
class AnalysisState(BaseState):
    """State for the model analysis page."""
    device: str = "cpu"
    model_type: str = "graph"
    
    # Loaded model info
    model_loaded: bool = False
    model_path: str = ""
    
    # Analysis results
    latent_pca: Optional[np.ndarray] = field(default=None, repr=False)
    activations: dict = field(default_factory=dict, repr=False)
    
    def __post_init__(self):
        object.__setattr__(self, '_observers', set())
        object.__setattr__(self, '_notify_enabled', True)

