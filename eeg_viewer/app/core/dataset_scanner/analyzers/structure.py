"""
Structure analyzer for dataset directories.

Analyzes directory structure to detect splits, classes, subjects, etc.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import re

from ..models import StructureInfo, SplitType


class StructureAnalyzer:
    """Analyzer for dataset directory structure."""
    
    # Known split folder names
    SPLIT_FOLDERS = {'train', 'training', 'val', 'valid', 'validation', 'test', 'testing', 'dev'}
    
    # Known condition folders (EEG specific)
    CONDITION_FOLDERS = {'DMT', 'EC', 'EO', 'REST', 'TASK'}
    
    # Subject patterns
    SUBJECT_PATTERNS = [
        r'^S\d+$',
        r'^sub-\d+$',
        r'^subject_?\d+$',
        r'^P\d+$',
        r'^patient\d+$',
    ]
    
    # Hidden/ignored patterns
    IGNORED_PATTERNS = {'.', '__', '.DS_Store', '__MACOSX', 'Thumbs.db'}
    
    def analyze(self, path: Path) -> StructureInfo:
        """
        Analyze directory structure.
        
        Args:
            path: Directory to analyze
            
        Returns:
            StructureInfo with detected structure
        """
        if not path.exists() or not path.is_dir():
            return StructureInfo(
                split_type=SplitType.NONE,
                has_classes=False,
                classes=[],
                has_subjects=False,
                subjects=[],
                hierarchy_depth=0,
                root_subdirs=[],
            )
        
        # Get root subdirectories
        root_subdirs = self._get_subdirs(path)
        
        # Calculate hierarchy depth
        hierarchy_depth = self._calculate_depth(path)
        
        # Count files and directories
        total_files, total_directories = self._count_items(path)
        
        # Detect split type
        split_type, has_classes, classes, class_counts = self._detect_structure(path, root_subdirs)
        
        # Detect subjects
        has_subjects, subjects, subject_counts = self._detect_subjects(path, root_subdirs)
        
        # Detect file patterns
        has_numbered, numbering_pattern = self._detect_numbered_files(path)
        has_date = self._detect_date_pattern(path)
        
        return StructureInfo(
            split_type=split_type,
            has_classes=has_classes,
            classes=classes,
            has_subjects=has_subjects,
            subjects=subjects,
            hierarchy_depth=hierarchy_depth,
            root_subdirs=root_subdirs,
            class_counts=class_counts,
            subject_counts=subject_counts,
            total_files=total_files,
            total_directories=total_directories,
            has_numbered_files=has_numbered,
            numbering_pattern=numbering_pattern,
            has_date_pattern=has_date,
        )
    
    def _is_ignored(self, name: str) -> bool:
        """Check if name should be ignored."""
        for pattern in self.IGNORED_PATTERNS:
            if name.startswith(pattern):
                return True
        return False
    
    def _get_subdirs(self, path: Path) -> List[str]:
        """Get non-hidden subdirectory names."""
        subdirs = []
        for item in path.iterdir():
            if item.is_dir() and not self._is_ignored(item.name):
                subdirs.append(item.name)
        return sorted(subdirs)
    
    def _calculate_depth(self, path: Path) -> int:
        """Calculate maximum directory depth."""
        max_depth = 0
        for item in path.rglob('*'):
            if item.is_file():
                depth = len(item.relative_to(path).parts) - 1
                max_depth = max(max_depth, depth)
        return max_depth
    
    def _count_items(self, path: Path) -> Tuple[int, int]:
        """Count total files and directories."""
        files = 0
        dirs = 0
        for item in path.rglob('*'):
            if self._is_ignored(item.name):
                continue
            if item.is_file():
                files += 1
            elif item.is_dir():
                dirs += 1
        return files, dirs
    
    def _detect_structure(self, path: Path, root_subdirs: List[str]) -> Tuple[SplitType, bool, List[str], Dict[str, int]]:
        """
        Detect the structure type.
        
        Returns:
            (split_type, has_classes, classes, class_counts)
        """
        subdirs_lower = {s.lower(): s for s in root_subdirs}
        
        # Check for train/val/test splits
        has_train = any(s in subdirs_lower for s in ['train', 'training'])
        has_val = any(s in subdirs_lower for s in ['val', 'valid', 'validation'])
        has_test = any(s in subdirs_lower for s in ['test', 'testing'])
        
        if has_train and has_val and has_test:
            # Check for classes within splits
            train_dir = path / (subdirs_lower.get('train') or subdirs_lower.get('training'))
            if train_dir and train_dir.exists():
                classes, class_counts = self._get_classes_in_dir(train_dir)
                if classes:
                    return SplitType.HIERARCHICAL, True, classes, class_counts
            return SplitType.TRAIN_VAL_TEST, False, [], {}
        
        if has_train and has_test:
            train_dir = path / (subdirs_lower.get('train') or subdirs_lower.get('training'))
            if train_dir and train_dir.exists():
                classes, class_counts = self._get_classes_in_dir(train_dir)
                if classes:
                    return SplitType.HIERARCHICAL, True, classes, class_counts
            return SplitType.TRAIN_TEST, False, [], {}
        
        # Check for conditions (EEG style) - detect even single conditions
        conditions = [s for s in root_subdirs if s in self.CONDITION_FOLDERS]
        if len(conditions) >= 1:
            class_counts = {}
            for cond in conditions:
                cond_dir = path / cond
                count = sum(1 for _ in cond_dir.rglob('*') if _.is_file())
                class_counts[cond] = count
            return SplitType.BY_CONDITION, True, sorted(conditions), class_counts
        
        # Check for subject folders
        subject_folders = [s for s in root_subdirs if self._is_subject_folder(s)]
        if len(subject_folders) >= 2:
            subject_counts = {}
            for subj in subject_folders:
                subj_dir = path / subj
                count = sum(1 for _ in subj_dir.rglob('*') if _.is_file())
                subject_counts[subj] = count
            return SplitType.BY_SUBJECT, False, [], subject_counts
        
        # Check for class folders (generic)
        if len(root_subdirs) >= 2:
            # Check if subdirs contain data files
            valid_class_dirs = []
            class_counts = {}
            for subdir in root_subdirs:
                subdir_path = path / subdir
                file_count = sum(1 for _ in subdir_path.rglob('*') if _.is_file())
                if file_count > 0:
                    valid_class_dirs.append(subdir)
                    class_counts[subdir] = file_count
            
            if len(valid_class_dirs) >= 2:
                return SplitType.BY_CLASS, True, sorted(valid_class_dirs), class_counts
        
        return SplitType.NONE, False, [], {}
    
    def _get_classes_in_dir(self, path: Path) -> Tuple[List[str], Dict[str, int]]:
        """Get class folders within a directory."""
        classes = []
        class_counts = {}
        
        for item in path.iterdir():
            if item.is_dir() and not self._is_ignored(item.name):
                file_count = sum(1 for _ in item.rglob('*') if _.is_file())
                if file_count > 0:
                    classes.append(item.name)
                    class_counts[item.name] = file_count
        
        return sorted(classes), class_counts
    
    def _is_subject_folder(self, name: str) -> bool:
        """Check if folder name looks like a subject ID."""
        for pattern in self.SUBJECT_PATTERNS:
            if re.match(pattern, name, re.IGNORECASE):
                return True
        return False
    
    def _detect_subjects(self, path: Path, root_subdirs: List[str]) -> Tuple[bool, List[str], Dict[str, int]]:
        """
        Detect subjects from folder names and filenames.
        
        Returns:
            (has_subjects, subjects, subject_counts)
        """
        subjects = set()
        subject_counts: Dict[str, int] = {}
        
        # Check folder names
        for subdir in root_subdirs:
            if self._is_subject_folder(subdir):
                subjects.add(subdir)
                subdir_path = path / subdir
                count = sum(1 for _ in subdir_path.rglob('*') if _.is_file())
                subject_counts[subdir] = count
        
        # Check filenames
        for f in path.rglob('*'):
            if f.is_file():
                # Try to extract subject ID from filename
                match = re.search(r'(S\d+|sub-\d+)', f.stem, re.IGNORECASE)
                if match:
                    subj_id = match.group(1).upper()
                    if subj_id.startswith('SUB-'):
                        subj_id = 'sub-' + subj_id[4:]
                    subjects.add(subj_id)
                    subject_counts[subj_id] = subject_counts.get(subj_id, 0) + 1
        
        has_subjects = len(subjects) > 1
        return has_subjects, sorted(subjects), subject_counts
    
    def _detect_numbered_files(self, path: Path) -> Tuple[bool, str]:
        """
        Detect numbered file patterns.
        
        Returns:
            (has_numbered, pattern)
        """
        numbered_patterns = [
            r'.*_\d{3,}\..*$',  # name_001.ext
            r'.*\d{5,}\..*$',   # name00001.ext
        ]
        
        sample_files = list(path.rglob('*'))[:100]
        
        for pattern in numbered_patterns:
            matches = sum(1 for f in sample_files if f.is_file() and re.match(pattern, f.name))
            if matches > len(sample_files) * 0.5:
                return True, pattern
        
        return False, None
    
    def _detect_date_pattern(self, path: Path) -> bool:
        """Detect date-based file naming."""
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{8}',              # YYYYMMDD
            r'\d{4}_\d{2}_\d{2}',  # YYYY_MM_DD
        ]
        
        sample_files = list(path.rglob('*'))[:100]
        
        for pattern in date_patterns:
            matches = sum(1 for f in sample_files if f.is_file() and re.search(pattern, f.name))
            if matches > len(sample_files) * 0.3:
                return True
        
        return False


