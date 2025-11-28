"""
Data module for VAE.

Reuses the dataset builder from the classifier module.
"""

import sys
import importlib.util
from pathlib import Path

# Load clf's dataset_builder directly to avoid naming conflicts
CLF_DATA_PATH = Path(__file__).parent.parent.parent / "clf" / "data" / "dataset_builder.py"

# Load module from file path
spec = importlib.util.spec_from_file_location("clf_dataset_builder", CLF_DATA_PATH)
clf_dataset_builder = importlib.util.module_from_spec(spec)
sys.modules["clf_dataset_builder"] = clf_dataset_builder
spec.loader.exec_module(clf_dataset_builder)

# Import from loaded module
EEGGraphDataset = clf_dataset_builder.EEGGraphDataset
create_dataset_from_config = clf_dataset_builder.create_dataset_from_config
load_phases_file = clf_dataset_builder.load_phases_file
build_graph_from_epoch = clf_dataset_builder.build_graph_from_epoch
build_graph_dict = clf_dataset_builder.build_graph_dict
process_single_file = clf_dataset_builder.process_single_file
compute_temporal_complexity = clf_dataset_builder.compute_temporal_complexity
compute_graph_topology = clf_dataset_builder.compute_graph_topology

__all__ = [
    'EEGGraphDataset',
    'create_dataset_from_config',
    'load_phases_file',
    'build_graph_from_epoch',
    'build_graph_dict',
    'process_single_file',
    'compute_temporal_complexity',
    'compute_graph_topology'
]

