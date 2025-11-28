"""
Models module for VAE with Graph Attention Networks.
"""

from .vae_model import BrainStateVAE, create_vae_from_config

__all__ = [
    'BrainStateVAE',
    'create_vae_from_config',
]

