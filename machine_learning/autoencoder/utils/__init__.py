"""
Utility functions for VAE training and evaluation.
"""

from .logger import setup_logging, log_metrics
from .visualization import (
    plot_training_curves,
    plot_latent_space,
    plot_reconstructions,
    plot_kl_per_dimension,
    log_latent_to_tensorboard,
    save_latent_embeddings
)

__all__ = [
    'setup_logging',
    'log_metrics',
    'plot_training_curves',
    'plot_latent_space',
    'plot_reconstructions',
    'plot_kl_per_dimension',
    'log_latent_to_tensorboard',
    'save_latent_embeddings'
]
















