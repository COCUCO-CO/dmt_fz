"""Dataset type detectors."""

from .base import BaseDetector, DetectionResult
from .image import ImageDetector
from .graph import GraphDetector
from .timeseries import TimeSeriesDetector
from .tabular import TabularDetector
from .text import TextDetector

__all__ = [
    'BaseDetector',
    'DetectionResult',
    'ImageDetector',
    'GraphDetector',
    'TimeSeriesDetector',
    'TabularDetector',
    'TextDetector',
]

