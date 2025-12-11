"""
Base detector class and detection result dataclass.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Set


@dataclass
class DetectionResult:
    """Result from a detector."""
    is_detected: bool = False
    confidence: float = 0.0
    file_count: int = 0
    extensions_found: List[str] = field(default_factory=list)
    sample_files: List[Path] = field(default_factory=list)
    total_size_bytes: int = 0
    
    # Structure info
    has_splits: bool = False
    splits_found: List[str] = field(default_factory=list)
    split_counts: Dict[str, int] = field(default_factory=dict)
    has_classes: bool = False
    classes_found: List[str] = field(default_factory=list)
    class_counts: Dict[str, int] = field(default_factory=dict)
    has_subjects: bool = False
    subjects_found: List[str] = field(default_factory=list)
    subject_counts: Dict[str, int] = field(default_factory=dict)
    has_conditions: bool = False
    conditions_found: List[str] = field(default_factory=list)
    condition_counts: Dict[str, int] = field(default_factory=dict)
    hierarchy_depth: int = 0
    
    # Type-specific info (filled by specific detectors)
    image_info: Optional[Any] = None
    graph_info: Optional[Any] = None
    timeseries_info: Optional[Any] = None
    tabular_info: Optional[Any] = None
    text_info: Optional[Any] = None
    phases_info: Optional[Any] = None
    
    # Metadata
    is_phases_format: bool = False
    is_eeg_format: bool = False
    is_audio_format: bool = False
    is_balanced: bool = True
    is_sentiment_dataset: bool = False
    total_graphs: int = 0
    corrupt_files_count: int = 0
    
    # Output
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    suggested_loader: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'is_detected': self.is_detected,
            'confidence': self.confidence,
            'file_count': self.file_count,
            'extensions_found': self.extensions_found,
            'sample_files': [str(f) for f in self.sample_files],
            'total_size_bytes': self.total_size_bytes,
            'has_splits': self.has_splits,
            'splits_found': self.splits_found,
            'split_counts': self.split_counts,
            'has_classes': self.has_classes,
            'classes_found': self.classes_found,
            'class_counts': self.class_counts,
            'warnings': self.warnings,
            'suggestions': self.suggestions,
        }


class BaseDetector(ABC):
    """Base class for dataset type detectors."""
    
    # Extensions this detector handles
    EXTENSIONS: Set[str] = set()
    
    # Known split folder names
    SPLIT_FOLDERS = {'train', 'training', 'val', 'valid', 'validation', 'test', 'testing', 'dev'}
    
    # Known condition folders (EEG specific)
    CONDITION_FOLDERS = {'DMT', 'EC', 'EO', 'REST', 'TASK'}
    
    # Subject patterns
    SUBJECT_PATTERNS = [
        r'^S\d+$',      # S01, S02, etc.
        r'^sub-\d+$',   # sub-01, sub-02 (BIDS format)
        r'^subject_?\d+$',  # subject_01, subject01
        r'^P\d+$',      # P001, P01
        r'^patient\d+$',  # patient01
    ]
    
    # Hidden/ignored patterns
    IGNORED_PATTERNS = {
        '.', '__', '.DS_Store', '__MACOSX', 'Thumbs.db',
        '.git', '.svn', '__pycache__', '.ipynb_checkpoints'
    }
    
    def __init__(self):
        self._file_cache: Dict[Path, List[Path]] = {}
    
    @abstractmethod
    def detect(self, path: Path, analyze_samples: bool = False, 
               sample_size: int = 10) -> DetectionResult:
        """
        Detect if path contains this type of dataset.
        
        Args:
            path: Directory to scan
            analyze_samples: Whether to analyze sample files in detail
            sample_size: Number of files to sample for analysis
            
        Returns:
            DetectionResult with detection info
        """
        pass
    
    def _is_ignored(self, name: str) -> bool:
        """Check if file/folder should be ignored."""
        for pattern in self.IGNORED_PATTERNS:
            if name.startswith(pattern):
                return True
        return False
    
    def _find_files(self, path: Path, extensions: Set[str] = None,
                    recursive: bool = True) -> List[Path]:
        """
        Find files with given extensions.
        
        Args:
            path: Directory to search
            extensions: Set of extensions (with dot, e.g. {'.png', '.jpg'})
            recursive: Search recursively
            
        Returns:
            List of file paths
        """
        if extensions is None:
            extensions = self.EXTENSIONS
        
        files = []
        
        if not path.exists():
            return files
        
        if recursive:
            for ext in extensions:
                # Case-insensitive search
                files.extend(path.rglob(f"*{ext}"))
                files.extend(path.rglob(f"*{ext.upper()}"))
        else:
            for item in path.iterdir():
                if item.is_file() and not self._is_ignored(item.name):
                    if item.suffix.lower() in extensions:
                        files.append(item)
        
        # Remove duplicates and filter ignored
        seen = set()
        result = []
        for f in files:
            if f not in seen and not self._is_ignored(f.name):
                seen.add(f)
                result.append(f)
        
        return sorted(result)
    
    def _calculate_total_size(self, files: List[Path]) -> int:
        """Calculate total size of files."""
        total = 0
        for f in files:
            try:
                total += f.stat().st_size
            except (OSError, IOError):
                pass
        return total
    
    def _detect_splits(self, path: Path) -> tuple:
        """
        Detect train/val/test splits.
        
        Returns:
            (has_splits, splits_found, split_counts)
        """
        has_splits = False
        splits_found = []
        split_counts = {}
        
        for item in path.iterdir():
            if item.is_dir() and not self._is_ignored(item.name):
                name_lower = item.name.lower()
                if name_lower in self.SPLIT_FOLDERS:
                    has_splits = True
                    # Normalize name
                    if 'train' in name_lower:
                        normalized = 'train'
                    elif 'val' in name_lower:
                        normalized = 'val'
                    elif 'test' in name_lower:
                        normalized = 'test'
                    elif 'dev' in name_lower:
                        normalized = 'dev'
                    else:
                        normalized = name_lower
                    
                    splits_found.append(normalized)
                    # Count files in split
                    count = sum(1 for _ in item.rglob('*') if _.is_file())
                    split_counts[normalized] = count
        
        return has_splits, splits_found, split_counts
    
    def _detect_classes(self, path: Path, exclude_splits: bool = True) -> tuple:
        """
        Detect class folders.
        
        Returns:
            (has_classes, classes_found, class_counts)
        """
        has_classes = False
        classes_found = []
        class_counts = {}
        
        # Check for class-like subfolders
        subdirs = [d for d in path.iterdir() if d.is_dir() and not self._is_ignored(d.name)]
        
        if exclude_splits:
            subdirs = [d for d in subdirs if d.name.lower() not in self.SPLIT_FOLDERS]
        
        # If we have multiple subdirs with data files, treat as classes
        if len(subdirs) >= 2:
            for subdir in subdirs:
                file_count = sum(1 for _ in subdir.rglob('*') if _.is_file())
                if file_count > 0:
                    has_classes = True
                    classes_found.append(subdir.name)
                    class_counts[subdir.name] = file_count
        
        return has_classes, sorted(classes_found), class_counts
    
    def _detect_conditions(self, path: Path) -> tuple:
        """
        Detect condition folders (EEG-style: DMT, EC, EO).
        
        Returns:
            (has_conditions, conditions_found, condition_counts)
        """
        has_conditions = False
        conditions_found = []
        condition_counts = {}
        
        for item in path.iterdir():
            if item.is_dir() and item.name in self.CONDITION_FOLDERS:
                has_conditions = True
                conditions_found.append(item.name)
                count = sum(1 for _ in item.rglob('*') if _.is_file())
                condition_counts[item.name] = count
        
        return has_conditions, sorted(conditions_found), condition_counts
    
    def _detect_subjects_from_files(self, files: List[Path]) -> tuple:
        """
        Detect subjects from filenames.
        
        Returns:
            (has_subjects, subjects_found, subject_counts)
        """
        import re
        
        subjects = {}
        
        for f in files:
            # Try to extract subject ID from filename
            name = f.stem
            
            # Pattern: S01, S02, etc.
            match = re.search(r'S(\d+)', name)
            if match:
                subj_id = f"S{match.group(1)}"
                subjects[subj_id] = subjects.get(subj_id, 0) + 1
                continue
            
            # Pattern: sub-01, sub-02 (BIDS)
            match = re.search(r'sub-(\d+)', name)
            if match:
                subj_id = f"sub-{match.group(1)}"
                subjects[subj_id] = subjects.get(subj_id, 0) + 1
        
        has_subjects = len(subjects) > 1
        subjects_found = sorted(subjects.keys())
        
        return has_subjects, subjects_found, subjects
    
    def _check_balance(self, counts: Dict[str, int]) -> bool:
        """Check if class counts are balanced (within 2:1 ratio)."""
        if not counts or len(counts) < 2:
            return True
        
        values = list(counts.values())
        if min(values) == 0:
            return False
        
        ratio = max(values) / min(values)
        return ratio <= 2.0
    
    def _get_hierarchy_depth(self, path: Path) -> int:
        """Calculate maximum directory depth."""
        max_depth = 0
        
        for item in path.rglob('*'):
            if item.is_file():
                depth = len(item.relative_to(path).parts) - 1
                max_depth = max(max_depth, depth)
        
        return max_depth

