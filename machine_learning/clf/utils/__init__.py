"""
Utility functions for training and evaluation.
"""

from .logger import setup_logging, log_metrics
from .visualization import plot_training_curves, plot_confusion_matrix, save_attention_heatmap
from .attention_logger import (
    extract_attention_matrices,
    plot_attention_heatmap,
    plot_attention_distributions,
    plot_average_attention,
    log_attention_to_tensorboard
)

__all__ = [
    'setup_logging',
    'log_metrics',
    'plot_training_curves',
    'plot_confusion_matrix',
    'save_attention_heatmap',
    'extract_attention_matrices',
    'plot_attention_heatmap',
    'plot_attention_distributions',
    'plot_average_attention',
    'log_attention_to_tensorboard',
]

