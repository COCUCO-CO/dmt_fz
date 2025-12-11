"""Fixtures for analysis tests."""
import pytest
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from unittest.mock import MagicMock


@dataclass
class MockGraphSample:
    """Mock graph sample with EEG metadata."""
    x: np.ndarray = field(default_factory=lambda: np.random.randn(24, 10).astype(np.float32))
    edge_index: np.ndarray = field(default_factory=lambda: np.array([[0, 1], [1, 0]]))
    edge_attr: np.ndarray = field(default_factory=lambda: np.array([[0.5], [0.5]]))
    y: int = 0
    graph_attr: np.ndarray = field(default_factory=lambda: np.array([0.7, 0.8, 0.9]))
    subject_id: str = "S01"
    condition: str = "DMT"
    band: str = "Alpha"
    epoch_idx: int = 0
    
    @property
    def num_nodes(self):
        return self.x.shape[0]


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
def mock_graph_sample():
    """Create a mock graph sample with EEG metadata."""
    return MockGraphSample()


@pytest.fixture
def mock_graph_dataset():
    """Create a mock graph dataset with multiple samples."""
    samples = []
    for i in range(10):
        sample = MockGraphSample(
            epoch_idx=i,
            graph_attr=np.array([0.5 + i * 0.05, 0.6, 0.7])
        )
        samples.append(sample)
    return samples


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
    state.eeg_sync_enabled = True
    state.epoch_duration = 2.0
    state.current_eeg_file = None
    state.eeg_view_start = 0.0
    return state

