"""Fixtures for analysis tests."""
import pytest
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from unittest.mock import MagicMock


@dataclass
class MockEEGData:
    """Mock EEG data for testing."""
    raw: Optional[Any] = None
    filename: str = "S01-DMT_ICA_pruned.set"
    filepath: Path = field(default_factory=lambda: Path("/mock/path"))
    sfreq: float = 512.0
    n_channels: int = 24
    n_samples: int = 512 * 300  # 5 minutes
    duration_sec: float = 300.0
    channel_names: List[str] = field(default_factory=lambda: [f"Ch{i}" for i in range(24)])
    channel_types: Dict[str, str] = field(default_factory=lambda: {f"Ch{i}": "eeg" for i in range(24)})
    
    @property
    def is_loaded(self) -> bool:
        return True


@pytest.fixture
def mock_eeg_data():
    """Create mock EEG data."""
    return MockEEGData()


@pytest.fixture
def temp_eeg_dir(tmp_path):
    """Create temporary EEG directory structure."""
    # Create condition directories
    for condition in ["DMT", "EC", "EO"]:
        cond_dir = tmp_path / condition
        cond_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dummy EEG files (just empty files for path testing)
        for subject in ["S01", "S02", "S03"]:
            # Different naming conventions
            (cond_dir / f"{subject}-{condition}_ICA_pruned.set").touch()
            (cond_dir / f"{subject}_{condition}_ICA_pruned.set").touch()
    
    return tmp_path


@pytest.fixture
def mock_analysis_state():
    """Create a mock AnalysisState for testing."""
    state = MagicMock()
    state.model = None
    state.dataset = None
    state.current_idx = 0
    state.total_samples = 0
    state.playing = False
    state.eeg_data = None
    state.eeg_sync_enabled = False
    state.epoch_duration = 2.0
    state.current_eeg_file = None
    state.eeg_view_start = 0.0
    state.eeg_channels = []
    return state
