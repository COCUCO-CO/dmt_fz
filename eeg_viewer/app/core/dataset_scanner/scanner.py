"""
Main DatasetScanner class.

Orchestrates all detectors and analyzers to produce complete DatasetInfo.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any

from .models import (
    DatasetType, SplitType, FileInfo, StructureInfo, SplitInfo, DatasetInfo
)
from .detectors import (
    ImageDetector, GraphDetector, TimeSeriesDetector, 
    TabularDetector, TextDetector, BinaryDatasetDetector
)
from .analyzers import StructureAnalyzer


class DatasetScanner:
    """
    Intelligent dataset scanner.
    
    Automatically detects dataset type, structure, and extracts metadata.
    
    Supported formats:
    - Images: PNG, JPG, TIFF, WEBP, BMP, GIF, DICOM, NIfTI
    - Graphs: PyTorch Geometric, GraphML, gpickle, phases-*.pkl
    - Time Series: NumPy, EEG (BDF, EDF, FIF, SET), Audio (WAV, MP3, FLAC)
    - Tabular: CSV, Excel, Parquet, HDF5
    - Text: TXT, JSON, JSONL
    - Binary: IDX/ubyte (MNIST-style)
    
    Usage:
        scanner = DatasetScanner()
        info = scanner.scan("/path/to/dataset")
        print(info.summary())
    """
    
    # Thresholds for MIXED detection
    MIXED_SCORE_RATIO = 0.5  # If secondary type scores > 50% of primary, consider mixed
    MIXED_MIN_FILES = 10     # Minimum files for secondary type to count as mixed
    
    def __init__(self):
        """Initialize scanner with all detectors."""
        self.image_detector = ImageDetector()
        self.graph_detector = GraphDetector()
        self.timeseries_detector = TimeSeriesDetector()
        self.tabular_detector = TabularDetector()
        self.text_detector = TextDetector()
        self.binary_detector = BinaryDatasetDetector()
        self.structure_analyzer = StructureAnalyzer()
        
        # Cache for repeated scans
        self._cache: Dict[str, DatasetInfo] = {}
    
    def scan(self, path: str | Path, 
             deep_scan: bool = False,
             sample_size: int = 10,
             detect_labels: bool = True,
             use_cache: bool = True) -> DatasetInfo:
        """
        Scan a directory to detect and analyze dataset.
        
        Args:
            path: Path to dataset directory
            deep_scan: Whether to analyze file contents in detail
            sample_size: Number of files to sample for analysis
            detect_labels: Whether to detect class labels
            use_cache: Whether to use cached results
            
        Returns:
            DatasetInfo with complete dataset information
        """
        path = Path(path)
        cache_key = f"{path}:{deep_scan}:{sample_size}"
        
        # Check cache
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Handle non-existent path
        if not path.exists():
            return self._create_error_info(path, f"Path does not exist: {path}")
        
        if not path.is_dir():
            return self._create_error_info(path, f"Path is not a directory: {path}")
        
        # Analyze structure first
        structure = self.structure_analyzer.analyze(path)
        
        # Run all detectors
        detection_results = {
            'image': self.image_detector.detect(path, deep_scan, sample_size),
            'graph': self.graph_detector.detect(path, deep_scan, sample_size),
            'timeseries': self.timeseries_detector.detect(path, deep_scan, sample_size),
            'tabular': self.tabular_detector.detect(path, deep_scan, sample_size),
            'text': self.text_detector.detect(path, deep_scan, sample_size),
            'binary': self.binary_detector.detect(path, deep_scan, sample_size),
        }
        
        # Determine primary type
        primary_type, best_result = self._determine_primary_type(detection_results)
        
        # Build file info
        files = self._build_file_info(path, detection_results, best_result)
        
        # Build split info if applicable
        splits = self._build_split_info(structure, best_result)
        
        # Update structure with detection results
        structure = self._merge_structure_info(structure, best_result)
        
        # Build type-specific info
        type_specific = self._build_type_specific(primary_type, best_result, detection_results)
        
        # Collect warnings and suggestions
        warnings = self._collect_warnings(detection_results, structure)
        suggestions = self._collect_suggestions(detection_results, primary_type, structure)
        
        # Create final info
        info = DatasetInfo(
            path=path,
            name=path.name,
            primary_type=primary_type,
            structure=structure,
            files=files,
            splits=splits,
            type_specific=type_specific,
            warnings=warnings,
            suggestions=suggestions,
        )
        
        # Cache result
        if use_cache:
            self._cache[cache_key] = info
        
        return info
    
    def _create_error_info(self, path: Path, error_msg: str) -> DatasetInfo:
        """Create DatasetInfo for error cases."""
        return DatasetInfo(
            path=path,
            name=path.name if path else "unknown",
            primary_type=DatasetType.UNKNOWN,
            structure=StructureInfo(
                split_type=SplitType.NONE,
                has_classes=False,
                classes=[],
                has_subjects=False,
                subjects=[],
                hierarchy_depth=0,
                root_subdirs=[],
            ),
            files=FileInfo(
                total_count=0,
                by_extension={},
                by_type={},
                sample_files=[],
                total_size_bytes=0,
            ),
            splits=None,
            type_specific={},
            warnings=[error_msg],
            suggestions=[],
        )
    
    def _determine_primary_type(self, results: dict) -> tuple:
        """
        Determine primary dataset type based on detection results.
        
        MIXED is returned when:
        - Multiple types are detected
        - Secondary type has significant file count (>= MIXED_MIN_FILES)
        - Secondary type score is >= MIXED_SCORE_RATIO of primary
        
        Returns:
            (DatasetType, best_detection_result, is_mixed, secondary_types)
        """
        type_map = {
            'image': DatasetType.IMAGE,
            'graph': DatasetType.GRAPH,
            'timeseries': DatasetType.TIMESERIES,
            'tabular': DatasetType.TABULAR,
            'text': DatasetType.TEXT,
            'binary': DatasetType.IMAGE,  # Binary (IDX) datasets are image-like
        }
        
        # Score each detection with file count weight
        scores = {}
        file_counts = {}
        for name, result in results.items():
            if result.is_detected:
                # Score based on confidence and file count
                score = result.confidence * (1 + min(result.file_count / 100, 1))
                scores[name] = score
                file_counts[name] = result.file_count
        
        if not scores:
            return DatasetType.UNKNOWN, None
        
        # Sort types by score
        sorted_types = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        best_type = sorted_types[0]
        best_score = scores[best_type]
        best_result = results[best_type]
        
        # Check for MIXED only if there are multiple detected types
        if len(sorted_types) > 1:
            secondary_significant = []
            
            for t in sorted_types[1:]:
                score_ratio = scores[t] / best_score
                file_count = file_counts[t]
                
                # A type is significant if:
                # 1. Score ratio is high enough (>= MIXED_SCORE_RATIO)
                # 2. File count is significant (>= MIXED_MIN_FILES)
                # 3. It's not a subset (e.g., text files that could be README)
                is_significant = (
                    score_ratio >= self.MIXED_SCORE_RATIO and
                    file_count >= self.MIXED_MIN_FILES and
                    not self._is_type_subset(t, best_type, results)
                )
                
                if is_significant:
                    secondary_significant.append(t)
            
            if secondary_significant:
                # This is a mixed dataset
                return DatasetType.MIXED, best_result
        
        # Check if binary should be promoted to IMAGE type with 'binary' format info
        if best_type == 'binary':
            return DatasetType.IMAGE, best_result
        
        return type_map[best_type], best_result
    
    def _is_type_subset(self, secondary: str, primary: str, results: dict) -> bool:
        """
        Check if secondary type is likely just a subset/artifact of primary.
        
        For example:
        - text files in an image dataset are likely README/metadata
        - tabular files in a graph dataset are likely edge lists
        """
        # Common artifact patterns
        artifacts = {
            'image': {'text'},      # README, metadata
            'graph': {'text', 'tabular'},  # Edge lists, node lists
            'timeseries': {'tabular'},  # Annotations, events
        }
        
        if primary in artifacts and secondary in artifacts[primary]:
            # Check if secondary has very few files compared to primary
            primary_count = results[primary].file_count
            secondary_count = results[secondary].file_count
            
            if secondary_count < primary_count * 0.1:  # Less than 10% of primary
                return True
        
        return False
    
    def _build_file_info(self, path: Path, results: dict, best_result) -> FileInfo:
        """Build FileInfo from detection results."""
        # Aggregate all extensions and counts
        all_extensions = {}
        total_count = 0
        total_size = 0
        sample_files = []
        by_type = {}
        
        for name, result in results.items():
            if result.is_detected:
                for ext in result.extensions_found:
                    # Count files with this extension
                    count = len([f for f in result.sample_files if f.suffix.lower() == ext])
                    if count > 0:
                        all_extensions[ext] = all_extensions.get(ext, 0) + result.file_count // max(len(result.extensions_found), 1)
                
                by_type[name] = result.file_count
                total_size = max(total_size, result.total_size_bytes)
                sample_files.extend(result.sample_files)
        
        # Use best result for primary counts
        if best_result:
            total_count = best_result.file_count
            for ext in best_result.extensions_found:
                if ext not in all_extensions:
                    all_extensions[ext] = best_result.file_count // max(len(best_result.extensions_found), 1)
            total_size = best_result.total_size_bytes
            sample_files = best_result.sample_files
        
        # Remove duplicates from sample files
        seen = set()
        unique_samples = []
        for f in sample_files:
            if f not in seen:
                seen.add(f)
                unique_samples.append(f)
        
        return FileInfo(
            total_count=total_count,
            by_extension=all_extensions,
            by_type=by_type,
            sample_files=unique_samples[:20],
            total_size_bytes=total_size,
        )
    
    def _build_split_info(self, structure: StructureInfo, best_result) -> Optional[SplitInfo]:
        """Build SplitInfo if splits are detected."""
        if not structure.split_type in {SplitType.TRAIN_TEST, SplitType.TRAIN_VAL_TEST, SplitType.HIERARCHICAL}:
            # Check detection result for splits
            if best_result and best_result.has_splits:
                return SplitInfo(
                    train_count=best_result.split_counts.get('train', 0),
                    val_count=best_result.split_counts.get('val', 0),
                    test_count=best_result.split_counts.get('test', 0),
                )
            return None
        
        # Use structure class counts as split approximation
        if structure.class_counts:
            # This is hierarchical - we don't have direct split counts
            # Would need to actually count files in train/val/test
            pass
        
        if best_result and best_result.has_splits:
            return SplitInfo(
                train_count=best_result.split_counts.get('train', 0),
                val_count=best_result.split_counts.get('val', 0),
                test_count=best_result.split_counts.get('test', 0),
            )
        
        return None
    
    def _merge_structure_info(self, structure: StructureInfo, best_result) -> StructureInfo:
        """Merge detection results into structure info."""
        if not best_result:
            return structure
        
        # Update classes if detected
        if best_result.has_classes and not structure.has_classes:
            structure.has_classes = True
            structure.classes = best_result.classes_found
            structure.class_counts = best_result.class_counts
        
        # Update subjects
        if best_result.has_subjects and not structure.has_subjects:
            structure.has_subjects = True
            structure.subjects = best_result.subjects_found
            structure.subject_counts = best_result.subject_counts
        
        # Update from conditions (graph detector)
        if best_result.has_conditions:
            if not structure.has_classes:
                structure.has_classes = True
                structure.classes = best_result.conditions_found
                structure.class_counts = best_result.condition_counts
            structure.split_type = SplitType.BY_CONDITION
        
        return structure
    
    def _build_type_specific(self, primary_type: DatasetType, best_result, results: dict) -> Dict[str, Any]:
        """Build type-specific information dictionary."""
        info = {}
        
        if not best_result:
            return info
        
        if primary_type == DatasetType.IMAGE:
            # Check if it's from binary detector (IDX/ubyte format)
            if 'binary' in results and results['binary'].is_detected:
                binary_result = results['binary']
                info['format'] = 'idx-ubyte'
                info['is_mnist_format'] = True
                if binary_result.image_info:
                    img_info = binary_result.image_info
                    info['sizes'] = img_info.image_sizes
                    info['channels'] = img_info.channels or 1
                    info['color_mode'] = img_info.color_mode or 'grayscale'
                    info['formats'] = img_info.formats
            elif best_result.image_info:
                img_info = best_result.image_info
                info['sizes'] = img_info.image_sizes
                info['channels'] = img_info.channels
                info['color_mode'] = img_info.color_mode
                info['formats'] = img_info.formats
        
        elif primary_type == DatasetType.GRAPH:
            if best_result.is_phases_format and best_result.phases_info:
                phases = best_result.phases_info
                info['format'] = 'phases'
                info['bands'] = phases.bands
                info['has_eeg'] = phases.has_eeg
                info['has_stc'] = phases.has_stc
                info['n_channels_eeg'] = phases.n_channels_eeg
                info['n_parcels'] = phases.n_parcels
                info['n_epochs_per_file'] = phases.n_epochs_per_file
                info['available_keys'] = phases.available_keys
            elif best_result.graph_info:
                graph = best_result.graph_info
                info['format'] = 'pyg'
                info['num_nodes_range'] = graph.num_nodes_range
                info['num_edges_range'] = graph.num_edges_range
                info['node_features_dim'] = graph.node_features_dim
                info['edge_features_dim'] = graph.edge_features_dim
                info['has_labels'] = graph.has_labels
                info['num_classes'] = graph.num_classes
        
        elif primary_type == DatasetType.TIMESERIES and best_result.timeseries_info:
            ts = best_result.timeseries_info
            info['sampling_rate'] = ts.sampling_rate
            info['num_channels'] = ts.num_channels
            info['num_channels_range'] = ts.num_channels_range
            info['signal_length'] = ts.signal_length
            info['signal_length_range'] = ts.signal_length_range
            info['dtype'] = ts.dtype
            info['is_eeg_format'] = best_result.is_eeg_format
            info['is_audio_format'] = best_result.is_audio_format
        
        elif primary_type == DatasetType.TABULAR and best_result.tabular_info:
            tab = best_result.tabular_info
            info['num_rows'] = tab.num_rows
            info['total_rows'] = tab.total_rows
            info['num_columns'] = tab.num_columns
            info['columns'] = tab.column_names
            info['numeric_columns'] = tab.numeric_columns
            info['categorical_columns'] = tab.categorical_columns
            info['has_missing'] = tab.has_missing_values
            info['suggested_target'] = tab.suggested_target
        
        elif primary_type == DatasetType.TEXT and best_result.text_info:
            txt = best_result.text_info
            info['document_count'] = txt.document_count
            info['avg_length'] = txt.avg_document_length
            info['total_words'] = txt.total_words
            info['vocabulary_size'] = txt.estimated_vocabulary_size
            info['is_json'] = txt.is_json
            info['is_jsonl'] = txt.is_jsonl
            info['json_keys'] = txt.json_keys
            info['is_conversational'] = txt.is_conversational
        
        return info
    
    def _collect_warnings(self, results: dict, structure: StructureInfo) -> List[str]:
        """Collect all warnings from detection results."""
        warnings = []
        
        for name, result in results.items():
            if result.is_detected:
                warnings.extend(result.warnings)
        
        # Count detected types
        detected_types = [name for name, r in results.items() if r.is_detected]
        
        # Warning for mixed dataset
        if len(detected_types) > 1:
            significant_types = []
            for name, result in results.items():
                if result.is_detected and result.file_count >= self.MIXED_MIN_FILES:
                    significant_types.append(f"{name} ({result.file_count} files)")
            
            if len(significant_types) > 1:
                warnings.append(f"Multiple data types detected: {', '.join(significant_types)}")
        
        # Add structure-based warnings
        if structure.has_classes and structure.class_counts:
            counts = list(structure.class_counts.values())
            if counts and max(counts) > min(counts) * 2:
                if "imbalance" not in ' '.join(warnings).lower():
                    warnings.append("Class imbalance detected in dataset")
        
        # Warning for small datasets
        total_files = sum(r.file_count for r in results.values() if r.is_detected)
        if total_files > 0 and total_files < 50:
            warnings.append(f"Small dataset ({total_files} files) - may not be sufficient for training")
        
        # Warning for no splits
        has_splits = any(r.has_splits for r in results.values() if r.is_detected)
        if not has_splits and structure.split_type == SplitType.NONE and total_files > 100:
            warnings.append("No train/val/test splits detected - consider splitting data")
        
        # Warning for corrupt files
        total_corrupt = sum(r.corrupt_files_count for r in results.values() if r.is_detected)
        if total_corrupt > 0 and "corrupt" not in ' '.join(warnings).lower():
            warnings.append(f"{total_corrupt} corrupt or unreadable files found")
        
        return list(set(warnings))  # Remove duplicates
    
    def _collect_suggestions(self, results: dict, primary_type: DatasetType, 
                            structure: StructureInfo) -> List[str]:
        """Collect all suggestions."""
        suggestions = []
        
        # Get suggestions from best detector
        for name, result in results.items():
            if result.is_detected:
                suggestions.extend(result.suggestions)
        
        # Add general suggestions
        if structure.split_type == SplitType.NONE and structure.total_files > 10:
            suggestions.append("Consider splitting data into train/val/test sets")
        
        if primary_type == DatasetType.GRAPH and structure.split_type == SplitType.BY_CONDITION:
            suggestions.append("Group splits by subject to avoid data leakage")
        
        return list(dict.fromkeys(suggestions))  # Remove duplicates, preserve order
    
    def clear_cache(self):
        """Clear the scan cache."""
        self._cache.clear()

