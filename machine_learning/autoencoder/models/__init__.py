"""
Models module for VAE with Graph Attention Networks.
"""

from .vae_model import BrainStateVAE, create_vae_from_config
from .enhanced_vae import EnhancedBrainStateVAE, create_enhanced_vae

__all__ = [
    'BrainStateVAE',
    'create_vae_from_config',
    'EnhancedBrainStateVAE',
    'create_enhanced_vae',
]



