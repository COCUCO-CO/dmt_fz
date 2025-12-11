"""
Binary dataset detector.

Handles IDX/ubyte format (MNIST-style) and other binary formats.
"""

from pathlib import Path
from typing import Set, List, Optional, Tuple
import struct
import gzip

from .base import BaseDetector, DetectionResult
from ..models import ImageSpecificInfo


class BinaryDatasetDetector(BaseDetector):
    """
    Detector for binary dataset formats.
    
    Supports:
    - IDX format (MNIST-style): idx1-ubyte, idx3-ubyte (labels and images)
    - Gzipped variants: .gz files containing IDX data
    - Raw binary image files
    """
    
    # Extensions for IDX/ubyte format
    EXTENSIONS: Set[str] = {
        '.ubyte', '.idx', '.idx1-ubyte', '.idx3-ubyte',
        '-ubyte',  # For files like train-images-idx3-ubyte
    }
    
    # Also check files without extension or with .gz
    MNIST_PATTERNS = [
        'train-images', 'train-labels', 't10k-images', 't10k-labels',
        'images-idx3-ubyte', 'labels-idx1-ubyte',
    ]
    
    def detect(self, path: Path, analyze_samples: bool = False,
               sample_size: int = 10) -> DetectionResult:
        """Detect binary/IDX dataset."""
        result = DetectionResult()
        
        if not path.exists():
            result.warnings.append(f"Path does not exist: {path}")
            return result
        
        # Find IDX/ubyte files
        files = self._find_idx_files(path)
        
        if not files:
            return result
        
        result.is_detected = True
        result.file_count = len(files)
        result.sample_files = files[:sample_size]
        result.total_size_bytes = self._calculate_total_size(files)
        
        # Get extensions found
        extensions = set()
        for f in files:
            if f.suffix:
                extensions.add(f.suffix.lower())
            # Check for idx patterns
            name = f.name.lower()
            if 'idx3' in name:
                extensions.add('.idx3-ubyte')
            elif 'idx1' in name:
                extensions.add('.idx1-ubyte')
        result.extensions_found = sorted(extensions) if extensions else ['.ubyte']
        
        # High confidence for recognized IDX format
        result.confidence = 0.95 if self._has_valid_idx_header(files[0]) else 0.75
        
        # Detect splits
        has_train = any('train' in f.name.lower() for f in files)
        has_test = any('test' in f.name.lower() or 't10k' in f.name.lower() for f in files)
        
        if has_train or has_test:
            result.has_splits = True
            result.splits_found = []
            if has_train:
                result.splits_found.append('train')
                result.split_counts['train'] = sum(1 for f in files if 'train' in f.name.lower())
            if has_test:
                result.splits_found.append('test')
                result.split_counts['test'] = sum(1 for f in files if 'test' in f.name.lower() or 't10k' in f.name.lower())
        
        # Analyze if requested
        if analyze_samples:
            result.image_info = self._analyze_idx_files(files)
        
        # Generate suggestions
        result.suggestions = self._generate_suggestions(result)
        result.suggested_loader = "torchvision.datasets.MNIST or custom IDX loader"
        
        return result
    
    def _find_idx_files(self, path: Path) -> List[Path]:
        """Find all IDX/ubyte format files."""
        files = []
        
        # Search for files with IDX-related extensions
        for ext in ['.gz', '']:  # Check both compressed and uncompressed
            for f in path.rglob(f'*{ext}'):
                if f.is_file():
                    name = f.name.lower()
                    # Check for IDX patterns in filename
                    if any(pattern in name for pattern in ['idx1', 'idx3', 'ubyte', '-images', '-labels']):
                        files.append(f)
                    # Check for MNIST-style names
                    elif any(pattern in name for pattern in self.MNIST_PATTERNS):
                        files.append(f)
        
        # Also check for explicit ubyte extension
        for ext in ['.ubyte', '.idx']:
            files.extend(path.rglob(f'*{ext}'))
        
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for f in files:
            if f not in seen:
                seen.add(f)
                unique.append(f)
        
        return sorted(unique, key=lambda x: x.name)
    
    def _has_valid_idx_header(self, file: Path) -> bool:
        """Check if file has valid IDX magic number."""
        try:
            open_func = gzip.open if file.suffix == '.gz' else open
            with open_func(file, 'rb') as f:
                magic = f.read(4)
                if len(magic) < 4:
                    return False
                
                # IDX magic number format: 0x00 0x00 [dtype] [ndims]
                # dtype: 0x08 = unsigned byte, 0x09 = signed byte, etc.
                # ndims: 1 = vector, 3 = 3D array (images)
                if magic[0:2] == b'\x00\x00':
                    dtype = magic[2]
                    ndims = magic[3]
                    if dtype in (0x08, 0x09, 0x0B, 0x0C, 0x0D, 0x0E) and ndims in (1, 2, 3, 4):
                        return True
                
        except Exception:
            pass
        
        return False
    
    def _analyze_idx_files(self, files: List[Path]) -> ImageSpecificInfo:
        """Analyze IDX files to extract metadata."""
        info = ImageSpecificInfo()
        sizes = []
        
        for f in files:
            try:
                metadata = self._read_idx_header(f)
                if metadata:
                    dims, dtype, shape = metadata
                    if dims == 3:  # Image file (n_images x rows x cols)
                        if len(shape) >= 3:
                            h, w = shape[1], shape[2]
                            sizes.append((w, h))
                            info.channels = 1  # IDX images are grayscale
                            info.color_mode = 'grayscale'
            except Exception:
                continue
        
        if sizes:
            info.image_sizes = list(set(sizes))
        info.formats = ['idx-ubyte']
        
        return info
    
    def _read_idx_header(self, file: Path) -> Optional[Tuple[int, int, tuple]]:
        """
        Read IDX file header and return (ndims, dtype, shape).
        
        IDX header format:
        - magic[0:2] = 0x0000
        - magic[2] = dtype (0x08 = ubyte, 0x09 = sbyte, 0x0B = short, 0x0C = int, 0x0D = float, 0x0E = double)
        - magic[3] = number of dimensions
        - Next 4*ndims bytes = dimension sizes (big-endian)
        """
        try:
            open_func = gzip.open if file.suffix == '.gz' else open
            with open_func(file, 'rb') as f:
                magic = f.read(4)
                if len(magic) < 4 or magic[0:2] != b'\x00\x00':
                    return None
                
                dtype = magic[2]
                ndims = magic[3]
                
                # Read dimension sizes
                shape = []
                for _ in range(ndims):
                    dim_bytes = f.read(4)
                    if len(dim_bytes) < 4:
                        return None
                    dim = struct.unpack('>I', dim_bytes)[0]
                    shape.append(dim)
                
                return ndims, dtype, tuple(shape)
                
        except Exception:
            return None
    
    def _generate_suggestions(self, result: DetectionResult) -> List[str]:
        """Generate suggestions for IDX/ubyte datasets."""
        suggestions = [
            "Use torchvision.datasets.MNIST for standard MNIST loading",
            "For custom IDX: use struct module to parse headers",
        ]
        
        if result.has_splits:
            suggestions.append("Dataset already has train/test splits")
        else:
            suggestions.append("Consider splitting into train/test sets")
        
        if any('.gz' in str(f) for f in result.sample_files):
            suggestions.append("Use gzip.open for compressed files")
        
        return suggestions


