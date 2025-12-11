"""
Image dataset detector.
"""

from pathlib import Path
from typing import Set, List, Tuple, Optional
import struct

from .base import BaseDetector, DetectionResult
from ..models import ImageSpecificInfo


class ImageDetector(BaseDetector):
    """Detector for image datasets."""
    
    EXTENSIONS: Set[str] = {
        '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif',
        '.webp', '.ico', '.ppm', '.pgm', '.pbm'
    }
    
    def detect(self, path: Path, analyze_samples: bool = False,
               sample_size: int = 10) -> DetectionResult:
        """Detect image dataset."""
        result = DetectionResult()
        
        if not path.exists():
            result.warnings.append(f"Path does not exist: {path}")
            return result
        
        # Find all image files
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
        
        # Calculate confidence based on file count
        if result.file_count >= 100:
            result.confidence = 0.95
        elif result.file_count >= 10:
            result.confidence = 0.85
        else:
            result.confidence = 0.70
        
        # Detect structure
        has_splits, splits_found, split_counts = self._detect_splits(path)
        result.has_splits = has_splits
        result.splits_found = splits_found
        result.split_counts = split_counts
        
        # Detect classes
        if has_splits:
            # Look for classes within splits
            for split in splits_found:
                split_path = path / split
                if split_path.exists():
                    _, classes, counts = self._detect_classes(split_path, exclude_splits=False)
                    if classes:
                        result.has_classes = True
                        result.classes_found = classes
                        # Aggregate class counts across splits
                        for cls, count in counts.items():
                            result.class_counts[cls] = result.class_counts.get(cls, 0) + count
                    break
        else:
            has_classes, classes_found, class_counts = self._detect_classes(path)
            result.has_classes = has_classes
            result.classes_found = classes_found
            result.class_counts = class_counts
        
        # Check balance
        if result.class_counts:
            result.is_balanced = self._check_balance(result.class_counts)
            if not result.is_balanced:
                result.warnings.append("Class imbalance detected")
        
        # Get hierarchy depth
        result.hierarchy_depth = self._get_hierarchy_depth(path)
        
        # Analyze samples if requested
        if analyze_samples:
            result.image_info = self._analyze_images(result.sample_files)
            result.corrupt_files_count = self._count_corrupt_images(result.sample_files)
            if result.corrupt_files_count > 0:
                result.warnings.append(f"{result.corrupt_files_count} corrupt/unreadable images found")
        
        # Generate suggestions
        result.suggestions = self._generate_suggestions(result)
        result.suggested_loader = self._suggest_loader(result)
        
        return result
    
    def _analyze_images(self, files: List[Path]) -> ImageSpecificInfo:
        """Analyze sample images for metadata."""
        info = ImageSpecificInfo()
        sizes = []
        channels_found = set()
        formats = set()
        
        for f in files:
            try:
                size, channels, fmt = self._get_image_info(f)
                if size:
                    sizes.append(size)
                if channels:
                    channels_found.add(channels)
                if fmt:
                    formats.add(fmt)
            except Exception:
                continue
        
        info.image_sizes = sizes
        info.formats = list(formats)
        
        if channels_found:
            # Most common channel count
            info.channels = max(set(channels_found), key=list(channels_found).count)
            if info.channels == 1:
                info.color_mode = 'grayscale'
            elif info.channels == 3:
                info.color_mode = 'RGB'
            elif info.channels == 4:
                info.color_mode = 'RGBA'
        
        return info
    
    def _get_image_info(self, path: Path) -> Tuple[Optional[Tuple[int, int]], Optional[int], Optional[str]]:
        """
        Get image dimensions and channel count without full loading.
        
        Returns:
            (width, height), channels, format
        """
        suffix = path.suffix.lower()
        
        try:
            with open(path, 'rb') as f:
                header = f.read(32)
            
            if not header:
                return None, None, None
            
            # PNG
            if header[:8] == b'\x89PNG\r\n\x1a\n':
                if len(header) >= 24:
                    w = struct.unpack('>I', header[16:20])[0]
                    h = struct.unpack('>I', header[20:24])[0]
                    # Color type at byte 25
                    if len(header) >= 26:
                        color_type = header[25]
                        channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color_type, 3)
                    else:
                        channels = 3
                    return (w, h), channels, 'png'
            
            # JPEG
            if header[:2] == b'\xff\xd8':
                # Parse JPEG to find SOF marker
                with open(path, 'rb') as f:
                    f.read(2)  # Skip SOI
                    while True:
                        marker = f.read(2)
                        if len(marker) < 2:
                            break
                        if marker[0] != 0xff:
                            break
                        if marker[1] in (0xc0, 0xc1, 0xc2):  # SOF markers
                            f.read(3)  # Skip length and precision
                            h = struct.unpack('>H', f.read(2))[0]
                            w = struct.unpack('>H', f.read(2))[0]
                            channels = struct.unpack('B', f.read(1))[0]
                            return (w, h), channels, 'jpeg'
                        else:
                            length = struct.unpack('>H', f.read(2))[0]
                            f.read(length - 2)
                return None, 3, 'jpeg'
            
            # GIF
            if header[:6] in (b'GIF87a', b'GIF89a'):
                w = struct.unpack('<H', header[6:8])[0]
                h = struct.unpack('<H', header[8:10])[0]
                return (w, h), 3, 'gif'
            
            # BMP
            if header[:2] == b'BM':
                w = struct.unpack('<I', header[18:22])[0]
                h = abs(struct.unpack('<i', header[22:26])[0])
                return (w, h), 3, 'bmp'
            
            # Try PIL as fallback
            try:
                from PIL import Image
                with Image.open(path) as img:
                    w, h = img.size
                    mode = img.mode
                    channels = {'L': 1, 'RGB': 3, 'RGBA': 4, 'P': 1}.get(mode, 3)
                    return (w, h), channels, suffix[1:]
            except ImportError:
                pass
            except Exception:
                pass
        
        except Exception:
            pass
        
        return None, None, suffix[1:] if suffix else None
    
    def _count_corrupt_images(self, files: List[Path]) -> int:
        """Count number of corrupt/unreadable images."""
        corrupt = 0
        for f in files:
            try:
                with open(f, 'rb') as file:
                    header = file.read(8)
                    if len(header) < 2:
                        corrupt += 1
                    elif f.suffix.lower() == '.png' and header[:8] != b'\x89PNG\r\n\x1a\n':
                        corrupt += 1
                    elif f.suffix.lower() in ('.jpg', '.jpeg') and header[:2] != b'\xff\xd8':
                        corrupt += 1
            except Exception:
                corrupt += 1
        return corrupt
    
    def _generate_suggestions(self, result: DetectionResult) -> List[str]:
        """Generate suggestions based on detection result."""
        suggestions = []
        
        if result.has_classes:
            suggestions.append("Use torchvision.datasets.ImageFolder for PyTorch")
            suggestions.append("Use tf.keras.utils.image_dataset_from_directory for TensorFlow")
        
        if not result.has_splits:
            suggestions.append("Consider splitting data into train/val/test sets")
        
        if not result.is_balanced:
            suggestions.append("Consider using class weights or oversampling for imbalanced classes")
        
        if result.image_info and result.image_info.image_sizes:
            sizes = set(result.image_info.image_sizes)
            if len(sizes) > 1:
                suggestions.append("Images have varying sizes - consider resizing to uniform dimensions")
        
        return suggestions
    
    def _suggest_loader(self, result: DetectionResult) -> str:
        """Suggest appropriate data loader."""
        if result.has_classes:
            return "torchvision.datasets.ImageFolder"
        return "torch.utils.data.DataLoader with custom Dataset"

