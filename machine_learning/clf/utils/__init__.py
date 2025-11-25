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
    log_attention_to_tensorboard,
    extract_embeddings_for_analysis,
    log_embeddings_to_tensorboard,
    save_embeddings_to_file,
    compute_attention_matrix_per_class,
    load_electrode_names,
    format_electrode_labels,
    plot_attention_per_class,
    extract_attention_per_class,
    plot_attention_distributions_by_class,
    generate_full_attention_analysis,
    compute_mst_from_attention,
    plot_attention_mst_graph
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
    'extract_embeddings_for_analysis',
    'log_embeddings_to_tensorboard',
    'save_embeddings_to_file',
    'compute_attention_matrix_per_class',
    'load_electrode_names',
    'format_electrode_labels',
    'plot_attention_per_class',
    'extract_attention_per_class',
    'plot_attention_distributions_by_class',
    'generate_full_attention_analysis',
    'compute_mst_from_attention',
    'plot_attention_mst_graph',
]

