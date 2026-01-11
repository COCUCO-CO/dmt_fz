"""
Models module for VAE and SimCLR with Graph Attention Networks.
"""

from .vae_model import BrainStateVAE, create_vae_from_config
from .enhanced_vae import EnhancedBrainStateVAE, create_enhanced_vae
from .simclr_model import BrainStateSimCLR, create_simclr_from_config

__all__ = [
    'BrainStateVAE',
    'create_vae_from_config',
    'EnhancedBrainStateVAE',
    'create_enhanced_vae',
    'BrainStateSimCLR',
    'create_simclr_from_config',
]



