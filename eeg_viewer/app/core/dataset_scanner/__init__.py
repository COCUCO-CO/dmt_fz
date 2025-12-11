"""
Dataset Scanner Module.

Intelligent scanner for detecting and analyzing ML datasets.
Supports images, graphs, time series, tabular data, and text.
"""

from .models import (
    DatasetType,
    SplitType,
    FileInfo,
    StructureInfo,
    SplitInfo,
    ImageSpecificInfo,
    GraphSpecificInfo,
    TimeSeriesSpecificInfo,
    TabularSpecificInfo,
    TextSpecificInfo,
    DatasetInfo,
)

from .scanner import DatasetScanner

from .detectors import (
    ImageDetector,
    GraphDetector,
    TimeSeriesDetector,
    TabularDetector,
    TextDetector,
)

from .analyzers import StructureAnalyzer

__all__ = [
    # Main class
    'DatasetScanner',
    
    # Models
    'DatasetType',
    'SplitType',
    'FileInfo',
    'StructureInfo',
    'SplitInfo',
    'ImageSpecificInfo',
    'GraphSpecificInfo',
    'TimeSeriesSpecificInfo',
    'TabularSpecificInfo',
    'TextSpecificInfo',
    'DatasetInfo',
    
    # Detectors
    'ImageDetector',
    'GraphDetector',
    'TimeSeriesDetector',
    'TabularDetector',
    'TextDetector',
    
    # Analyzers
    'StructureAnalyzer',
]

