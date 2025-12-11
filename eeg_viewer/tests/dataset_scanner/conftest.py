"""
Pytest fixtures for dataset_scanner tests.

This module provides realistic mock datasets for testing the scanner.
Each fixture creates actual files on disk that mimic real ML datasets.
"""

import pytest
import numpy as np
import pandas as pd
import pickle
import json
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any
import struct
import sys

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_minimal_png(path: Path, width: int = 64, height: int = 64, 
                       channels: int = 3, color: tuple = None) -> None:
    """
    Create a minimal valid PNG file without PIL dependency.
    Uses raw PNG encoding with zlib compression.
    """
    import zlib
    
    if color is None:
        # Random color
        np.random.seed(hash(str(path)) % (2**32))
        color = tuple(np.random.randint(0, 255, 3))
    
    def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        chunk_len = struct.pack('>I', len(data))
        chunk_crc = struct.pack('>I', zlib.crc32(chunk_type + data) & 0xffffffff)
        return chunk_len + chunk_type + data + chunk_crc
    
    # PNG signature
    signature = b'\x89PNG\r\n\x1a\n'
    
    # IHDR chunk
    if channels == 1:
        color_type = 0  # Grayscale
    elif channels == 3:
        color_type = 2  # RGB
    else:  # channels == 4
        color_type = 6  # RGBA
    
    ihdr_data = struct.pack('>IIBBBBB', width, height, 8, color_type, 0, 0, 0)
    ihdr = png_chunk(b'IHDR', ihdr_data)
    
    # IDAT chunk (image data)
    raw_data = b''
    for y in range(height):
        raw_data += b'\x00'  # Filter byte (none)
        for x in range(width):
            if channels == 1:
                raw_data += bytes([color[0]])
            elif channels == 3:
                raw_data += bytes(color[:3])
            else:
                raw_data += bytes(color[:3]) + b'\xff'
    
    compressed = zlib.compress(raw_data, 9)
    idat = png_chunk(b'IDAT', compressed)
    
    # IEND chunk
    iend = png_chunk(b'IEND', b'')
    
    # Write PNG file
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(signature + ihdr + idat + iend)


def create_minimal_jpg(path: Path, width: int = 64, height: int = 64) -> None:
    """
    Create a minimal valid JPEG file.
    Creates a simple solid color JPEG using raw bytes.
    """
    # Minimal JPEG structure (solid gray image)
    # This is a valid JPEG that most readers will accept
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create using numpy and basic JPEG markers
    # For simplicity, we'll create a raw RGB array and use a minimal encoder
    try:
        from PIL import Image
        img = Image.new('RGB', (width, height), color=(128, 128, 128))
        img.save(path, 'JPEG', quality=50)
    except ImportError:
        # Fallback: create a minimal valid JPEG using hardcoded bytes
        # This is a 1x1 red pixel JPEG, scaled up won't look great but is valid
        minimal_jpeg = bytes([
            0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
            0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
            0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
            0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
            0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
            0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
            0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
            0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01,
            0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00,
            0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
            0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01, 0x03,
            0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D,
            0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06,
            0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
            0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72,
            0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
            0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45,
            0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
            0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75,
            0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
            0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3,
            0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6,
            0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9,
            0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
            0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4,
            0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01,
            0x00, 0x00, 0x3F, 0x00, 0xFB, 0xD5, 0xDB, 0x20, 0xA8, 0xF1, 0x45, 0x00,
            0xFF, 0xD9
        ])
        with open(path, 'wb') as f:
            f.write(minimal_jpeg)


def create_numpy_timeseries(path: Path, n_channels: int = 24, 
                            n_samples: int = 1000, sfreq: float = 256.0) -> None:
    """Create a realistic EEG-like time series as numpy array."""
    np.random.seed(42)
    
    # Generate realistic EEG-like signals
    t = np.arange(n_samples) / sfreq
    data = np.zeros((n_channels, n_samples))
    
    for ch in range(n_channels):
        # Mix of frequencies (alpha, beta, theta)
        alpha = 0.5 * np.sin(2 * np.pi * 10 * t + np.random.uniform(0, 2*np.pi))
        beta = 0.3 * np.sin(2 * np.pi * 20 * t + np.random.uniform(0, 2*np.pi))
        theta = 0.4 * np.sin(2 * np.pi * 6 * t + np.random.uniform(0, 2*np.pi))
        noise = 0.1 * np.random.randn(n_samples)
        
        data[ch] = alpha + beta + theta + noise
    
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, data)


def create_phases_pkl(path: Path, n_channels: int = 24, n_epochs: int = 10,
                      n_samples: int = 500, bands: list = None,
                      include_stc: bool = False, n_parcels: int = 68) -> None:
    """
    Create a realistic phases-*.pkl file mimicking the project's format.
    
    Structure:
    {
        'phases_eeg': {band: [array(n_channels, n_samples) for each epoch]},
        'syncros_eeg': {band: [array(n_channels, n_channels) for each epoch]},
        'kuramoto_eeg': {band: [float for each epoch]},
        'amplitudes_eeg': {band: [array(n_channels, n_samples) for each epoch]},
        # Optional STC data
        'phases_stc': {band: [array(n_parcels, n_samples) for each epoch]},
        ...
    }
    """
    if bands is None:
        bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    
    np.random.seed(hash(str(path)) % (2**32))
    
    data = {
        'phases_eeg': {},
        'syncros_eeg': {},
        'kuramoto_eeg': {},
        'amplitudes_eeg': {},
    }
    
    for band in bands:
        phases_list = []
        syncros_list = []
        kuramoto_list = []
        amplitudes_list = []
        
        for epoch in range(n_epochs):
            # Generate random phases (-π to π)
            phases = np.random.uniform(-np.pi, np.pi, (n_channels, n_samples))
            phases_list.append(phases.astype(np.float32))
            
            # Generate synchronization matrix (symmetric, 0-1)
            sync = np.random.uniform(0, 1, (n_channels, n_channels))
            sync = (sync + sync.T) / 2  # Make symmetric
            np.fill_diagonal(sync, 1.0)
            syncros_list.append(sync.astype(np.float32))
            
            # Generate Kuramoto order parameter
            kuramoto_list.append(float(np.random.uniform(0.3, 0.9)))
            
            # Generate amplitudes
            amplitudes = np.random.uniform(0.5, 2.0, (n_channels, n_samples))
            amplitudes_list.append(amplitudes.astype(np.float32))
        
        data['phases_eeg'][band] = phases_list
        data['syncros_eeg'][band] = syncros_list
        data['kuramoto_eeg'][band] = kuramoto_list
        data['amplitudes_eeg'][band] = amplitudes_list
    
    # Add STC data if requested
    if include_stc:
        data['phases_stc'] = {}
        data['syncros_stc'] = {}
        data['kuramoto_stc'] = {}
        data['amplitudes_stc'] = {}
        
        for band in bands:
            phases_list = []
            syncros_list = []
            kuramoto_list = []
            amplitudes_list = []
            
            for epoch in range(n_epochs):
                phases = np.random.uniform(-np.pi, np.pi, (n_parcels, n_samples))
                phases_list.append(phases.astype(np.float32))
                
                sync = np.random.uniform(0, 1, (n_parcels, n_parcels))
                sync = (sync + sync.T) / 2
                np.fill_diagonal(sync, 1.0)
                syncros_list.append(sync.astype(np.float32))
                
                kuramoto_list.append(float(np.random.uniform(0.3, 0.9)))
                amplitudes = np.random.uniform(0.5, 2.0, (n_parcels, n_samples))
                amplitudes_list.append(amplitudes.astype(np.float32))
            
            data['phases_stc'][band] = phases_list
            data['syncros_stc'][band] = syncros_list
            data['kuramoto_stc'][band] = kuramoto_list
            data['amplitudes_stc'][band] = amplitudes_list
    
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def create_graph_pt(path: Path, n_nodes: int = 24, n_edges: int = 100,
                    node_features: int = 10, edge_features: int = 1,
                    num_classes: int = 3) -> None:
    """
    Create a PyTorch Geometric compatible graph file.
    
    Creates a dict with:
    - x: node features [n_nodes, node_features]
    - edge_index: [2, n_edges]
    - edge_attr: [n_edges, edge_features]
    - y: label
    """
    np.random.seed(hash(str(path)) % (2**32))
    
    # Node features
    x = np.random.randn(n_nodes, node_features).astype(np.float32)
    
    # Edge index (random connections)
    src = np.random.randint(0, n_nodes, n_edges)
    dst = np.random.randint(0, n_nodes, n_edges)
    edge_index = np.stack([src, dst]).astype(np.int64)
    
    # Edge attributes
    edge_attr = np.random.randn(n_edges, edge_features).astype(np.float32)
    
    # Label
    y = np.array([np.random.randint(0, num_classes)], dtype=np.int64)
    
    # Try to save as torch tensors if available
    try:
        import torch
        data = {
            'x': torch.from_numpy(x),
            'edge_index': torch.from_numpy(edge_index),
            'edge_attr': torch.from_numpy(edge_attr),
            'y': torch.from_numpy(y),
            'num_nodes': n_nodes
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, path)
    except ImportError:
        # Fallback to pickle
        data = {
            'x': x,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'y': y,
            'num_nodes': n_nodes
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f)


def create_csv_tabular(path: Path, n_rows: int = 100, 
                       columns: Dict[str, str] = None) -> None:
    """
    Create a CSV file with realistic tabular data.
    
    columns: dict mapping column name to type ('numeric', 'categorical', 'datetime')
    """
    if columns is None:
        columns = {
            'id': 'numeric',
            'age': 'numeric',
            'category': 'categorical',
            'value': 'numeric',
            'label': 'categorical'
        }
    
    np.random.seed(42)
    data = {}
    
    for col, dtype in columns.items():
        if dtype == 'numeric':
            data[col] = np.random.randn(n_rows)
        elif dtype == 'categorical':
            categories = ['A', 'B', 'C', 'D']
            data[col] = np.random.choice(categories, n_rows)
        elif dtype == 'datetime':
            base = pd.Timestamp('2024-01-01')
            data[col] = [base + pd.Timedelta(days=int(x)) for x in np.random.randint(0, 365, n_rows)]
    
    df = pd.DataFrame(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def create_text_file(path: Path, n_lines: int = 50) -> None:
    """Create a text file with sample content."""
    lines = [
        f"This is line {i}. Sample text content for NLP testing."
        for i in range(n_lines)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def create_json_data(path: Path, n_records: int = 20) -> None:
    """Create a JSON file with structured data."""
    records = []
    for i in range(n_records):
        records.append({
            'id': i,
            'text': f'Sample text {i}',
            'label': np.random.choice(['positive', 'negative', 'neutral']),
            'score': float(np.random.uniform(0, 1))
        })
    
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(records, f, indent=2)


def create_jsonl_data(path: Path, n_records: int = 20) -> None:
    """Create a JSONL (JSON Lines) file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for i in range(n_records):
            record = {
                'id': i,
                'text': f'Sample text {i}',
                'label': np.random.choice(['positive', 'negative', 'neutral']),
            }
            f.write(json.dumps(record) + '\n')


def create_hdf5_file(path: Path, datasets: Dict[str, tuple] = None) -> None:
    """
    Create an HDF5 file with datasets.
    
    datasets: dict mapping dataset name to (shape, dtype)
    """
    try:
        import h5py
    except ImportError:
        pytest.skip("h5py not available")
        return
    
    if datasets is None:
        datasets = {
            'images': ((100, 64, 64, 3), np.float32),
            'labels': ((100,), np.int64),
        }
    
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as f:
        for name, (shape, dtype) in datasets.items():
            data = np.random.randn(*shape).astype(dtype)
            f.create_dataset(name, data=data)


def create_parquet_file(path: Path, n_rows: int = 100) -> None:
    """Create a Parquet file with tabular data."""
    try:
        import pyarrow.parquet as pq
        import pyarrow as pa
    except ImportError:
        # Fallback: use pandas if pyarrow available through pandas
        try:
            df = pd.DataFrame({
                'col_a': np.random.randn(n_rows),
                'col_b': np.random.choice(['X', 'Y', 'Z'], n_rows),
                'col_c': np.random.randint(0, 100, n_rows),
            })
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path)
        except Exception:
            pytest.skip("parquet support not available")
            return
        return
    
    df = pd.DataFrame({
        'col_a': np.random.randn(n_rows),
        'col_b': np.random.choice(['X', 'Y', 'Z'], n_rows),
        'col_c': np.random.randint(0, 100, n_rows),
    })
    table = pa.Table.from_pandas(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


# =============================================================================
# FIXTURES - FLAT DATASETS
# =============================================================================

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    tmp = Path(tempfile.mkdtemp())
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def flat_image_dataset(temp_dir: Path) -> Path:
    """Create a flat directory with mixed image files."""
    dataset_path = temp_dir / "flat_images"
    dataset_path.mkdir()
    
    # Create PNG images
    for i in range(10):
        create_minimal_png(dataset_path / f"image_{i:03d}.png")
    
    # Create JPG images
    for i in range(5):
        create_minimal_jpg(dataset_path / f"photo_{i:03d}.jpg")
    
    return dataset_path


@pytest.fixture
def flat_numpy_dataset(temp_dir: Path) -> Path:
    """Create a flat directory with numpy arrays."""
    dataset_path = temp_dir / "flat_numpy"
    dataset_path.mkdir()
    
    for i in range(10):
        create_numpy_timeseries(
            dataset_path / f"signal_{i:03d}.npy",
            n_channels=24,
            n_samples=1000
        )
    
    return dataset_path


@pytest.fixture
def flat_csv_dataset(temp_dir: Path) -> Path:
    """Create a flat directory with CSV files."""
    dataset_path = temp_dir / "flat_csv"
    dataset_path.mkdir()
    
    for i in range(5):
        create_csv_tabular(dataset_path / f"data_{i:03d}.csv", n_rows=100)
    
    return dataset_path


# =============================================================================
# FIXTURES - SPLIT DATASETS
# =============================================================================

@pytest.fixture
def train_test_image_dataset(temp_dir: Path) -> Path:
    """Create train/test split image dataset."""
    dataset_path = temp_dir / "train_test_images"
    
    for split in ['train', 'test']:
        split_dir = dataset_path / split
        n_images = 20 if split == 'train' else 10
        for i in range(n_images):
            create_minimal_png(split_dir / f"img_{i:03d}.png")
    
    return dataset_path


@pytest.fixture
def train_val_test_image_dataset(temp_dir: Path) -> Path:
    """Create train/val/test split image dataset."""
    dataset_path = temp_dir / "train_val_test_images"
    
    splits = {'train': 50, 'val': 15, 'test': 15}
    for split, count in splits.items():
        split_dir = dataset_path / split
        for i in range(count):
            create_minimal_png(split_dir / f"img_{i:03d}.png")
    
    return dataset_path


# =============================================================================
# FIXTURES - CLASS-BASED DATASETS
# =============================================================================

@pytest.fixture
def by_class_image_dataset(temp_dir: Path) -> Path:
    """Create image dataset organized by class folders."""
    dataset_path = temp_dir / "by_class_images"
    
    classes = ['cat', 'dog', 'bird']
    for cls in classes:
        cls_dir = dataset_path / cls
        for i in range(15):
            color = {'cat': (255, 100, 100), 'dog': (100, 255, 100), 'bird': (100, 100, 255)}[cls]
            create_minimal_png(cls_dir / f"{i:03d}.png", color=color)
    
    return dataset_path


@pytest.fixture
def imagenet_style_dataset(temp_dir: Path) -> Path:
    """Create ImageNet-style dataset with train/val/test and class subfolders."""
    dataset_path = temp_dir / "imagenet_style"
    
    classes = ['n01234567', 'n01234568', 'n01234569']  # ImageNet-like class IDs
    splits = {'train': 30, 'val': 10, 'test': 10}
    
    for split, count in splits.items():
        for cls in classes:
            cls_dir = dataset_path / split / cls
            n_images = count // len(classes)
            for i in range(n_images):
                create_minimal_png(cls_dir / f"img_{i:03d}.png")
    
    return dataset_path


# =============================================================================
# FIXTURES - SUBJECT/CONDITION DATASETS (EEG-style)
# =============================================================================

@pytest.fixture
def by_condition_phases_dataset(temp_dir: Path) -> Path:
    """Create phases dataset organized by condition (DMT, EC, EO)."""
    dataset_path = temp_dir / "by_condition_phases"
    
    conditions = ['DMT', 'EC', 'EO']
    subjects = ['S01', 'S02', 'S03', 'S04', 'S05']
    
    for condition in conditions:
        cond_dir = dataset_path / condition
        for subject in subjects:
            create_phases_pkl(
                cond_dir / f"phases-{subject}-{condition}.pkl",
                n_channels=24,
                n_epochs=10,
                bands=['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'],
                include_stc=True,
                n_parcels=68
            )
    
    return dataset_path


@pytest.fixture
def by_subject_dataset(temp_dir: Path) -> Path:
    """Create dataset organized by subject folders."""
    dataset_path = temp_dir / "by_subject"
    
    subjects = ['S01', 'S02', 'S03']
    
    for subject in subjects:
        subj_dir = dataset_path / subject
        # Each subject has multiple recordings
        for i in range(3):
            create_numpy_timeseries(
                subj_dir / f"recording_{i:02d}.npy",
                n_channels=24,
                n_samples=5000
            )
    
    return dataset_path


# =============================================================================
# FIXTURES - GRAPH DATASETS
# =============================================================================

@pytest.fixture
def graph_pt_dataset(temp_dir: Path) -> Path:
    """Create dataset of PyTorch Geometric graph files."""
    dataset_path = temp_dir / "graph_pt"
    dataset_path.mkdir()
    
    for i in range(20):
        n_nodes = np.random.randint(20, 50)
        n_edges = n_nodes * 4  # Sparse graph
        create_graph_pt(
            dataset_path / f"graph_{i:03d}.pt",
            n_nodes=n_nodes,
            n_edges=n_edges,
            node_features=10,
            edge_features=2
        )
    
    return dataset_path


@pytest.fixture
def graph_by_class_dataset(temp_dir: Path) -> Path:
    """Create graph dataset organized by class."""
    dataset_path = temp_dir / "graph_by_class"
    
    classes = ['class_0', 'class_1', 'class_2']
    for idx, cls in enumerate(classes):
        cls_dir = dataset_path / cls
        for i in range(10):
            create_graph_pt(
                cls_dir / f"graph_{i:03d}.pt",
                n_nodes=24,
                n_edges=100,
                num_classes=3
            )
    
    return dataset_path


# =============================================================================
# FIXTURES - TABULAR DATASETS
# =============================================================================

@pytest.fixture
def single_csv_dataset(temp_dir: Path) -> Path:
    """Create a single CSV file dataset."""
    dataset_path = temp_dir / "single_csv"
    dataset_path.mkdir()
    
    create_csv_tabular(
        dataset_path / "data.csv",
        n_rows=500,
        columns={
            'feature_1': 'numeric',
            'feature_2': 'numeric',
            'feature_3': 'numeric',
            'category': 'categorical',
            'target': 'numeric'
        }
    )
    
    return dataset_path


@pytest.fixture
def train_test_csv_dataset(temp_dir: Path) -> Path:
    """Create train/test CSV files."""
    dataset_path = temp_dir / "train_test_csv"
    dataset_path.mkdir()
    
    create_csv_tabular(dataset_path / "train.csv", n_rows=400)
    create_csv_tabular(dataset_path / "test.csv", n_rows=100)
    
    return dataset_path


@pytest.fixture
def parquet_dataset(temp_dir: Path) -> Path:
    """Create a Parquet file dataset."""
    dataset_path = temp_dir / "parquet_data"
    dataset_path.mkdir()
    
    create_parquet_file(dataset_path / "data.parquet", n_rows=500)
    
    return dataset_path


@pytest.fixture
def hdf5_dataset(temp_dir: Path) -> Path:
    """Create an HDF5 dataset."""
    dataset_path = temp_dir / "hdf5_data"
    dataset_path.mkdir()
    
    create_hdf5_file(
        dataset_path / "dataset.h5",
        datasets={
            'train_images': ((100, 32, 32, 3), np.float32),
            'train_labels': ((100,), np.int64),
            'test_images': ((20, 32, 32, 3), np.float32),
            'test_labels': ((20,), np.int64),
        }
    )
    
    return dataset_path


# =============================================================================
# FIXTURES - TEXT DATASETS
# =============================================================================

@pytest.fixture
def text_flat_dataset(temp_dir: Path) -> Path:
    """Create flat text file dataset."""
    dataset_path = temp_dir / "text_flat"
    dataset_path.mkdir()
    
    for i in range(10):
        create_text_file(dataset_path / f"doc_{i:03d}.txt", n_lines=50)
    
    return dataset_path


@pytest.fixture
def text_by_class_dataset(temp_dir: Path) -> Path:
    """Create text dataset organized by class/sentiment."""
    dataset_path = temp_dir / "text_by_class"
    
    classes = ['positive', 'negative', 'neutral']
    for cls in classes:
        cls_dir = dataset_path / cls
        for i in range(10):
            create_text_file(cls_dir / f"review_{i:03d}.txt", n_lines=20)
    
    return dataset_path


@pytest.fixture
def json_dataset(temp_dir: Path) -> Path:
    """Create JSON dataset."""
    dataset_path = temp_dir / "json_data"
    dataset_path.mkdir()
    
    create_json_data(dataset_path / "data.json", n_records=100)
    
    return dataset_path


@pytest.fixture
def jsonl_dataset(temp_dir: Path) -> Path:
    """Create JSONL dataset."""
    dataset_path = temp_dir / "jsonl_data"
    dataset_path.mkdir()
    
    create_jsonl_data(dataset_path / "data.jsonl", n_records=100)
    
    return dataset_path


# =============================================================================
# FIXTURES - MIXED/COMPLEX DATASETS
# =============================================================================

@pytest.fixture
def mixed_dataset(temp_dir: Path) -> Path:
    """Create dataset with multiple data types."""
    dataset_path = temp_dir / "mixed_data"
    dataset_path.mkdir()
    
    # Images
    (dataset_path / "images").mkdir()
    for i in range(5):
        create_minimal_png(dataset_path / "images" / f"img_{i:03d}.png")
    
    # CSV
    create_csv_tabular(dataset_path / "metadata.csv", n_rows=50)
    
    # Numpy
    create_numpy_timeseries(dataset_path / "signals.npy", n_channels=10)
    
    return dataset_path


@pytest.fixture
def deeply_nested_dataset(temp_dir: Path) -> Path:
    """Create deeply nested directory structure."""
    dataset_path = temp_dir / "deeply_nested"
    
    # Create: data/category/subcategory/split/files
    categories = ['cat_A', 'cat_B']
    subcategories = ['sub_1', 'sub_2']
    splits = ['train', 'test']
    
    for cat in categories:
        for subcat in subcategories:
            for split in splits:
                dir_path = dataset_path / "data" / cat / subcat / split
                for i in range(3):
                    create_minimal_png(dir_path / f"img_{i:03d}.png")
    
    return dataset_path


# =============================================================================
# FIXTURES - EDGE CASES
# =============================================================================

@pytest.fixture
def empty_dataset(temp_dir: Path) -> Path:
    """Create an empty directory."""
    dataset_path = temp_dir / "empty"
    dataset_path.mkdir()
    return dataset_path


@pytest.fixture
def single_file_dataset(temp_dir: Path) -> Path:
    """Create dataset with just one file."""
    dataset_path = temp_dir / "single_file"
    dataset_path.mkdir()
    create_csv_tabular(dataset_path / "only_file.csv", n_rows=100)
    return dataset_path


@pytest.fixture
def dataset_with_hidden_files(temp_dir: Path) -> Path:
    """Create dataset with hidden files that should be ignored."""
    dataset_path = temp_dir / "with_hidden"
    dataset_path.mkdir()
    
    # Normal files
    for i in range(5):
        create_minimal_png(dataset_path / f"image_{i:03d}.png")
    
    # Hidden files (should be ignored)
    create_minimal_png(dataset_path / ".hidden_image.png")
    (dataset_path / ".DS_Store").touch()
    (dataset_path / "__MACOSX").mkdir()
    create_minimal_png(dataset_path / "__MACOSX" / "._image.png")
    
    return dataset_path


@pytest.fixture
def dataset_with_corrupt_files(temp_dir: Path) -> Path:
    """Create dataset with some corrupt/invalid files."""
    dataset_path = temp_dir / "with_corrupt"
    dataset_path.mkdir()
    
    # Valid files
    for i in range(5):
        create_minimal_png(dataset_path / f"valid_{i:03d}.png")
    
    # Corrupt PNG (invalid header)
    corrupt_png = dataset_path / "corrupt.png"
    with open(corrupt_png, 'wb') as f:
        f.write(b'not a valid png file at all')
    
    # Empty file with image extension
    (dataset_path / "empty.jpg").touch()
    
    return dataset_path


# =============================================================================
# FIXTURES - REAL PROJECT DATA PATHS (for integration tests)
# =============================================================================

@pytest.fixture
def real_phases_path() -> Path:
    """Path to real phases dataset (if available)."""
    path = Path("/media/storage_hdd/dmt_fz/fwd-inv-stc")
    if not path.exists():
        pytest.skip("Real phases dataset not available")
    return path


@pytest.fixture
def real_eeg_path() -> Path:
    """Path to real EEG dataset (if available)."""
    path = Path("/media/storage_hdd/dmt_fz/EEG")
    if not path.exists():
        pytest.skip("Real EEG dataset not available")
    return path

