"""
Time series dataset detector.

Handles numpy arrays, EEG formats (BDF, EDF, FIF, SET), and audio files.
"""

from pathlib import Path
from typing import Set, List, Optional, Tuple
import struct

from .base import BaseDetector, DetectionResult
from ..models import TimeSeriesSpecificInfo


class TimeSeriesDetector(BaseDetector):
    """Detector for time series datasets."""
    
    EXTENSIONS: Set[str] = {
        # NumPy
        '.npy', '.npz',
        # EEG formats
        '.bdf', '.edf', '.fif', '.set',
        # Audio
        '.wav', '.mp3', '.flac', '.ogg',
        # Other
        '.mat',  # MATLAB
    }
    
    EEG_EXTENSIONS = {'.bdf', '.edf', '.fif', '.set'}
    AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg'}
    
    def detect(self, path: Path, analyze_samples: bool = False,
               sample_size: int = 10) -> DetectionResult:
        """Detect time series dataset."""
        result = DetectionResult()
        
        if not path.exists():
            result.warnings.append(f"Path does not exist: {path}")
            return result
        
        # Find all time series files
        files = self._find_files(path, self.EXTENSIONS)
        
        if not files:
            return result
        
        result.is_detected = True
        result.file_count = len(files)
        result.sample_files = files[:sample_size]
        result.total_size_bytes = self._calculate_total_size(files)
        
        # Get extensions found
        extensions = set(f.suffix.lower() for f in files)
        result.extensions_found = sorted(extensions)
        
        # Determine if EEG or audio format
        result.is_eeg_format = bool(extensions & self.EEG_EXTENSIONS)
        result.is_audio_format = bool(extensions & self.AUDIO_EXTENSIONS)
        
        # Calculate confidence
        if result.file_count >= 50:
            result.confidence = 0.90
        elif result.file_count >= 10:
            result.confidence = 0.80
        else:
            result.confidence = 0.65
        
        # Detect structure
        has_subjects, subjects, subject_counts = self._detect_subjects_from_files(files)
        result.has_subjects = has_subjects
        result.subjects_found = subjects
        result.subject_counts = subject_counts
        
        # Also check for subject folders
        subdirs = [d for d in path.iterdir() if d.is_dir() and not self._is_ignored(d.name)]
        subject_dirs = [d for d in subdirs if self._is_subject_dir(d.name)]
        if subject_dirs:
            result.has_subjects = True
            result.subjects_found = sorted(set(result.subjects_found + [d.name for d in subject_dirs]))
        
        # Detect conditions
        has_conditions, conditions, condition_counts = self._detect_conditions(path)
        if has_conditions:
            result.has_conditions = has_conditions
            result.conditions_found = conditions
            result.condition_counts = condition_counts
            result.has_classes = True
            result.classes_found = conditions
            result.class_counts = condition_counts
        
        result.hierarchy_depth = self._get_hierarchy_depth(path)
        
        # Analyze samples
        if analyze_samples:
            result.timeseries_info = self._analyze_timeseries(files[:sample_size])
            corrupt = self._count_corrupt_files(files[:sample_size])
            if corrupt > 0:
                result.corrupt_files_count = corrupt
                result.warnings.append(f"{corrupt} corrupt/unreadable files")
        
        # Generate suggestions
        result.suggestions = self._generate_suggestions(result)
        result.suggested_loader = self._suggest_loader(result)
        
        return result
    
    def _is_subject_dir(self, name: str) -> bool:
        """Check if directory name looks like a subject ID."""
        import re
        patterns = [
            r'^S\d+$',
            r'^sub-\d+$',
            r'^subject_?\d+$',
            r'^P\d+$',
        ]
        for pattern in patterns:
            if re.match(pattern, name, re.IGNORECASE):
                return True
        return False
    
    def _analyze_timeseries(self, files: List[Path]) -> TimeSeriesSpecificInfo:
        """Analyze time series files."""
        info = TimeSeriesSpecificInfo()
        
        shapes = []
        channels_list = []
        lengths_list = []
        dtypes = set()
        array_keys = set()
        
        for f in files:
            try:
                shape, dtype, keys = self._get_array_info(f)
                if shape:
                    shapes.append(shape)
                    # Infer channels and length from shape
                    if len(shape) == 2:
                        channels_list.append(shape[0])
                        lengths_list.append(shape[1])
                    elif len(shape) == 1:
                        lengths_list.append(shape[0])
                if dtype:
                    dtypes.add(str(dtype))
                if keys:
                    array_keys.update(keys)
            except Exception:
                continue
        
        info.shapes_found = shapes
        info.array_keys = list(array_keys)
        
        if dtypes:
            info.dtype = list(dtypes)[0]
        
        if channels_list:
            info.num_channels = max(set(channels_list), key=channels_list.count)
            info.num_channels_range = (min(channels_list), max(channels_list))
        
        if lengths_list:
            info.signal_length = max(set(lengths_list), key=lengths_list.count)
            info.signal_length_range = (min(lengths_list), max(lengths_list))
        
        return info
    
    def _get_array_info(self, path: Path) -> Tuple[Optional[tuple], Optional[str], Optional[List[str]]]:
        """Get array shape and dtype without full loading."""
        import numpy as np
        
        suffix = path.suffix.lower()
        
        try:
            if suffix == '.npy':
                # Read numpy header only
                with open(path, 'rb') as f:
                    version = np.lib.format.read_magic(f)
                    shape, fortran, dtype = np.lib.format._read_array_header(f, version)
                return shape, str(dtype), None
            
            elif suffix == '.npz':
                # List keys and get first array info
                npz = np.load(path)
                keys = list(npz.keys())
                if keys:
                    first_arr = npz[keys[0]]
                    return first_arr.shape, str(first_arr.dtype), keys
                npz.close()
                return None, None, keys
            
            elif suffix in self.EEG_EXTENSIONS:
                # For EEG files, we just note the format
                return None, None, None
            
            elif suffix == '.wav':
                # Parse WAV header
                with open(path, 'rb') as f:
                    header = f.read(44)
                    if header[:4] == b'RIFF' and header[8:12] == b'WAVE':
                        channels = struct.unpack('<H', header[22:24])[0]
                        sample_rate = struct.unpack('<I', header[24:28])[0]
                        # Estimate samples from file size
                        f.seek(0, 2)  # End of file
                        file_size = f.tell()
                        bits = struct.unpack('<H', header[34:36])[0]
                        samples = (file_size - 44) // (channels * bits // 8)
                        return (channels, samples), 'int16', None
            
        except Exception:
            pass
        
        return None, None, None
    
    def _count_corrupt_files(self, files: List[Path]) -> int:
        """Count corrupt/unreadable files."""
        import numpy as np
        
        corrupt = 0
        for f in files:
            try:
                suffix = f.suffix.lower()
                if suffix == '.npy':
                    # Try to read header
                    with open(f, 'rb') as file:
                        np.lib.format.read_magic(file)
                elif suffix == '.npz':
                    npz = np.load(f)
                    _ = list(npz.keys())
                    npz.close()
                # Other formats - just check file is readable
                else:
                    with open(f, 'rb') as file:
                        file.read(100)
            except Exception:
                corrupt += 1
        
        return corrupt
    
    def _generate_suggestions(self, result: DetectionResult) -> List[str]:
        """Generate suggestions."""
        suggestions = []
        
        if result.is_eeg_format:
            suggestions.append("Use MNE-Python for EEG data loading and preprocessing")
            suggestions.append("mne.io.read_raw_bdf/edf for BDF/EDF files")
        
        if result.is_audio_format:
            suggestions.append("Use librosa or torchaudio for audio processing")
        
        if '.npy' in result.extensions_found or '.npz' in result.extensions_found:
            suggestions.append("Use numpy.load for direct loading")
            suggestions.append("Consider torch.from_numpy for PyTorch")
        
        if not result.has_subjects and result.file_count > 10:
            suggestions.append("Consider organizing by subject for reproducible splits")
        
        return suggestions
    
    def _suggest_loader(self, result: DetectionResult) -> str:
        """Suggest data loader."""
        if result.is_eeg_format:
            return "mne.io.read_raw_* with custom Dataset"
        if result.is_audio_format:
            return "torchaudio.load or librosa.load"
        return "numpy.load with torch.utils.data.DataLoader"








