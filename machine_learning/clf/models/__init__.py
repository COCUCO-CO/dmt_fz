"""
Graph Neural Network models for EEG state classification.
"""

from .gat_model import BrainStateGAT, create_model_from_config

__all__ = ['BrainStateGAT', 'create_model_from_config']





