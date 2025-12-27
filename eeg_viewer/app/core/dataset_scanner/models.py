"""
Data models for the dataset scanner.

Contains all dataclasses and enums used throughout the module.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter


class DatasetType(Enum):
    """Primary dataset type."""
    IMAGE = 'image'
    GRAPH = 'graph'
    TIMESERIES = 'timeseries'
    TABULAR = 'tabular'
    TEXT = 'text'
    MIXED = 'mixed'
    UNKNOWN = 'unknown'


class SplitType(Enum):
    """Dataset split structure type."""
    NONE = 'none'
    TRAIN_TEST = 'train_test'
    TRAIN_VAL_TEST = 'train_val_test'
    BY_CLASS = 'by_class'
    BY_SUBJECT = 'by_subject'
    BY_CONDITION = 'by_condition'
    HIERARCHICAL = 'hierarchical'


@dataclass
class FileInfo:
    """Information about files in the dataset."""
    total_count: int
    by_extension: Dict[str, int]
    by_type: Dict[str, int]
    sample_files: List[Path]
    total_size_bytes: int
    
    @property
    def total_size_human(self) -> str:
        """Human-readable size string."""
        size = self.total_size_bytes
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} PB"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_count': self.total_count,
            'by_extension': self.by_extension,
            'by_type': self.by_type,
            'sample_files': [str(f) for f in self.sample_files],
            'total_size_bytes': self.total_size_bytes,
            'total_size_human': self.total_size_human,
        }


@dataclass
class StructureInfo:
    """Information about directory structure."""
    split_type: SplitType
    has_classes: bool
    classes: List[str]
    has_subjects: bool
    subjects: List[str]
    hierarchy_depth: int
    root_subdirs: List[str]
    class_counts: Dict[str, int] = field(default_factory=dict)
    subject_counts: Dict[str, int] = field(default_factory=dict)
    total_files: int = 0
    total_directories: int = 0
    has_numbered_files: bool = False
    numbering_pattern: Optional[str] = None
    has_date_pattern: bool = False
    file_patterns: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'split_type': self.split_type.value,
            'has_classes': self.has_classes,
            'classes': self.classes,
            'has_subjects': self.has_subjects,
            'subjects': self.subjects,
            'hierarchy_depth': self.hierarchy_depth,
            'root_subdirs': self.root_subdirs,
            'class_counts': self.class_counts,
            'subject_counts': self.subject_counts,
            'total_files': self.total_files,
            'total_directories': self.total_directories,
        }


@dataclass
class SplitInfo:
    """Information about train/val/test splits."""
    train_count: int
    val_count: int
    test_count: int
    
    @property
    def total(self) -> int:
        """Total number of samples."""
        return self.train_count + self.val_count + self.test_count
    
    @property
    def train_ratio(self) -> float:
        """Training set ratio."""
        if self.total == 0:
            return 0.0
        return self.train_count / self.total
    
    @property
    def val_ratio(self) -> float:
        """Validation set ratio."""
        if self.total == 0:
            return 0.0
        return self.val_count / self.total
    
    @property
    def test_ratio(self) -> float:
        """Test set ratio."""
        if self.total == 0:
            return 0.0
        return self.test_count / self.total
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'train_count': self.train_count,
            'val_count': self.val_count,
            'test_count': self.test_count,
            'total': self.total,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
        }


@dataclass
class ImageSpecificInfo:
    """Image-specific information."""
    image_sizes: List[Tuple[int, int]] = field(default_factory=list)
    channels: int = 0
    color_mode: str = ''
    formats: List[str] = field(default_factory=list)
    size_distribution: Optional[Dict[str, int]] = None
    
    @property
    def sizes_found(self) -> List[Tuple[int, int]]:
        """Alias for image_sizes."""
        return self.image_sizes
    
    @property
    def most_common_size(self) -> Optional[Tuple[int, int]]:
        """Return most common image size."""
        if not self.image_sizes:
            return None
        counter = Counter(self.image_sizes)
        return counter.most_common(1)[0][0]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'image_sizes': self.image_sizes,
            'channels': self.channels,
            'color_mode': self.color_mode,
            'formats': self.formats,
            'most_common_size': self.most_common_size,
        }


@dataclass
class GraphSpecificInfo:
    """Graph-specific information."""
    num_nodes_range: Tuple[int, int] = (0, 0)
    num_edges_range: Tuple[int, int] = (0, 0)
    node_features_dim: int = 0
    edge_features_dim: int = 0
    has_node_labels: bool = False
    has_edge_labels: bool = False
    has_labels: bool = False
    num_classes: int = 0
    is_directed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'num_nodes_range': self.num_nodes_range,
            'num_edges_range': self.num_edges_range,
            'node_features_dim': self.node_features_dim,
            'edge_features_dim': self.edge_features_dim,
            'has_node_labels': self.has_node_labels,
            'has_edge_labels': self.has_edge_labels,
            'has_labels': self.has_labels,
            'num_classes': self.num_classes,
            'is_directed': self.is_directed,
        }


@dataclass
class TimeSeriesSpecificInfo:
    """Time series specific information."""
    sampling_rate: Optional[float] = None
    num_channels: int = 0
    num_channels_range: Tuple[int, int] = (0, 0)
    duration_range: Tuple[float, float] = (0.0, 0.0)
    signal_length: int = 0
    signal_length_range: Tuple[int, int] = (0, 0)
    bands: List[str] = field(default_factory=list)
    has_stc: bool = False
    has_eeg: bool = False
    n_parcels: int = 0
    dtype: Optional[str] = None
    shapes_found: List[Tuple] = field(default_factory=list)
    array_keys: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'sampling_rate': self.sampling_rate,
            'num_channels': self.num_channels,
            'num_channels_range': self.num_channels_range,
            'duration_range': self.duration_range,
            'signal_length': self.signal_length,
            'signal_length_range': self.signal_length_range,
            'bands': self.bands,
            'has_stc': self.has_stc,
            'has_eeg': self.has_eeg,
            'n_parcels': self.n_parcels,
            'dtype': self.dtype,
        }


@dataclass
class TabularSpecificInfo:
    """Tabular data specific information."""
    num_rows: int = 0
    total_rows: int = 0
    num_columns: int = 0
    column_names: List[str] = field(default_factory=list)
    column_types: Dict[str, str] = field(default_factory=dict)
    numeric_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    id_columns: List[str] = field(default_factory=list)
    missing_counts: Dict[str, int] = field(default_factory=dict)
    has_missing_values: bool = False
    suggested_target: Optional[str] = None
    numeric_stats: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'num_rows': self.num_rows,
            'total_rows': self.total_rows,
            'num_columns': self.num_columns,
            'column_names': self.column_names,
            'column_types': self.column_types,
            'numeric_columns': self.numeric_columns,
            'categorical_columns': self.categorical_columns,
            'id_columns': self.id_columns,
            'has_missing_values': self.has_missing_values,
            'missing_counts': self.missing_counts,
            'suggested_target': self.suggested_target,
        }


@dataclass
class TextSpecificInfo:
    """Text data specific information."""
    document_count: int = 0
    total_documents: int = 0
    avg_document_length: float = 0.0
    total_lines: int = 0
    total_words: int = 0
    total_characters: int = 0
    avg_words_per_document: float = 0.0
    estimated_vocabulary_size: int = 0
    vocabulary_size: int = 0
    format: str = ''
    is_tokenized: bool = False
    is_json: bool = False
    is_jsonl: bool = False
    json_keys: List[str] = field(default_factory=list)
    json_structure: str = ''  # 'array', 'object', 'records'
    record_count: int = 0
    is_conversational: bool = False
    is_sentiment_dataset: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'document_count': self.document_count,
            'avg_document_length': self.avg_document_length,
            'total_lines': self.total_lines,
            'total_words': self.total_words,
            'avg_words_per_document': self.avg_words_per_document,
            'estimated_vocabulary_size': self.estimated_vocabulary_size,
            'format': self.format,
            'is_json': self.is_json,
            'is_jsonl': self.is_jsonl,
            'json_keys': self.json_keys,
            'record_count': self.record_count,
        }


@dataclass
class DatasetInfo:
    """Complete dataset information."""
    path: Path
    name: str
    primary_type: DatasetType
    structure: StructureInfo
    files: FileInfo
    splits: Optional[SplitInfo]
    type_specific: Dict[str, Any]
    warnings: List[str]
    suggestions: List[str]
    
    @property
    def is_valid(self) -> bool:
        """Check if dataset is valid (has files and known type)."""
        return (
            self.files.total_count > 0 and 
            self.primary_type != DatasetType.UNKNOWN
        )
    
    @property
    def compatible_frameworks(self) -> List[str]:
        """Return list of compatible ML frameworks."""
        frameworks = []
        
        if self.primary_type == DatasetType.IMAGE:
            frameworks = ['pytorch', 'tensorflow', 'keras', 'scikit-learn']
        elif self.primary_type == DatasetType.GRAPH:
            frameworks = ['pytorch', 'pytorch_geometric', 'dgl', 'networkx']
        elif self.primary_type == DatasetType.TIMESERIES:
            frameworks = ['pytorch', 'tensorflow', 'mne', 'scipy']
        elif self.primary_type == DatasetType.TABULAR:
            frameworks = ['pandas', 'scikit-learn', 'pytorch', 'xgboost', 'lightgbm']
        elif self.primary_type == DatasetType.TEXT:
            frameworks = ['pytorch', 'tensorflow', 'huggingface', 'spacy', 'nltk']
        else:
            frameworks = ['pytorch', 'tensorflow', 'scikit-learn']
        
        return frameworks
    
    @property
    def suggested_loaders(self) -> List[str]:
        """Return suggested data loaders."""
        loaders = []
        
        if self.primary_type == DatasetType.IMAGE:
            if self.structure.has_classes:
                loaders.append('torchvision.datasets.ImageFolder')
            loaders.append('torch.utils.data.DataLoader')
            loaders.append('tf.keras.utils.image_dataset_from_directory')
            
        elif self.primary_type == DatasetType.GRAPH:
            loaders.append('torch_geometric.loader.DataLoader')
            if 'phases' in str(self.type_specific.get('format', '')).lower():
                loaders.append('Custom phases loader')
                
        elif self.primary_type == DatasetType.TIMESERIES:
            if self.type_specific.get('is_eeg_format'):
                loaders.append('mne.io.read_raw_*')
            loaders.append('numpy.load')
            loaders.append('torch.utils.data.DataLoader')
            
        elif self.primary_type == DatasetType.TABULAR:
            loaders.append('pandas.read_csv')
            loaders.append('pandas.read_parquet')
            
        elif self.primary_type == DatasetType.TEXT:
            if self.type_specific.get('is_json') or self.type_specific.get('is_jsonl'):
                loaders.append('json.load / jsonlines')
            loaders.append('datasets.load_dataset (HuggingFace)')
        
        return loaders
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Dataset: {self.name}",
            f"Type: {self.primary_type.value.upper()}",
            f"Files: {self.files.total_count} ({self.files.total_size_human})",
            f"Structure: {self.structure.split_type.value}",
        ]
        
        if self.structure.has_classes:
            lines.append(f"Classes: {', '.join(self.structure.classes[:5])}" + 
                        (f" (+{len(self.structure.classes)-5} more)" if len(self.structure.classes) > 5 else ""))
        
        if self.structure.has_subjects:
            lines.append(f"Subjects: {len(self.structure.subjects)}")
        
        if self.splits:
            lines.append(f"Splits: train={self.splits.train_count}, val={self.splits.val_count}, test={self.splits.test_count}")
        
        if self.warnings:
            lines.append(f"Warnings: {len(self.warnings)}")
        
        return '\n'.join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (JSON-serializable)."""
        return {
            'path': str(self.path),
            'name': self.name,
            'primary_type': self.primary_type.value,
            'structure': self.structure.to_dict(),
            'files': self.files.to_dict(),
            'splits': self.splits.to_dict() if self.splits else None,
            'type_specific': self.type_specific,
            'is_valid': self.is_valid,
            'compatible_frameworks': self.compatible_frameworks,
            'suggested_loaders': self.suggested_loaders,
            'warnings': self.warnings,
            'suggestions': self.suggestions,
        }





