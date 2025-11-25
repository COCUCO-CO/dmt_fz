#!/usr/bin/env python3
"""
Dataset validation tests.

Run these tests BEFORE training to ensure data integrity:
    python -m pytest tests/test_dataset.py -v

Or run directly:
    python tests/test_dataset.py
"""

import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class DatasetValidator:
    """
    Validates dataset properties for ML training.
    """
    
    def __init__(self, train_graphs: List, val_graphs: List, test_graphs: List,
                 class_names: List[str] = None):
        self.train_graphs = train_graphs
        self.val_graphs = val_graphs
        self.test_graphs = test_graphs
        self.class_names = class_names or ['DMT', 'EC', 'EO']
        self.all_graphs = train_graphs + val_graphs + test_graphs
        
        self.errors = []
        self.warnings = []
        self.info = []
    
    def _log_error(self, msg: str):
        self.errors.append(f"❌ ERROR: {msg}")
    
    def _log_warning(self, msg: str):
        self.warnings.append(f"⚠️  WARNING: {msg}")
    
    def _log_info(self, msg: str):
        self.info.append(f"✓ {msg}")
    
    def validate_all(self) -> bool:
        """Run all validations. Returns True if all critical checks pass."""
        print("\n" + "=" * 70)
        print("DATASET VALIDATION")
        print("=" * 70)
        
        self.check_labels_valid()
        self.check_class_distribution()
        self.check_data_leakage()
        self.check_feature_integrity()
        self.check_graph_structure()
        
        # Print results
        print("\n--- INFO ---")
        for msg in self.info:
            print(msg)
        
        if self.warnings:
            print("\n--- WARNINGS ---")
            for msg in self.warnings:
                print(msg)
        
        if self.errors:
            print("\n--- ERRORS ---")
            for msg in self.errors:
                print(msg)
        
        print("\n" + "=" * 70)
        if self.errors:
            print(f"VALIDATION FAILED: {len(self.errors)} errors found")
            print("=" * 70)
            return False
        else:
            print("VALIDATION PASSED ✓")
            print("=" * 70)
            return True
    
    def check_labels_valid(self):
        """Check that all labels are within valid range."""
        num_classes = len(self.class_names)
        
        for split_name, graphs in [('train', self.train_graphs), 
                                    ('val', self.val_graphs), 
                                    ('test', self.test_graphs)]:
            labels = [g.y.item() for g in graphs]
            invalid = [l for l in labels if l < 0 or l >= num_classes]
            
            if invalid:
                self._log_error(f"{split_name}: Found {len(invalid)} invalid labels: {set(invalid)}. "
                               f"Expected range [0, {num_classes-1}]")
            else:
                self._log_info(f"{split_name}: All {len(labels)} labels valid (range [0, {num_classes-1}])")
    
    def check_class_distribution(self, max_ratio_diff: float = 0.3):
        """
        Check that class distribution is similar across splits.
        
        Args:
            max_ratio_diff: Maximum allowed difference in class ratios between splits.
                           0.3 means train ratio can differ from val/test by at most 30%.
        """
        distributions = {}
        
        for split_name, graphs in [('train', self.train_graphs), 
                                    ('val', self.val_graphs), 
                                    ('test', self.test_graphs)]:
            if not graphs:
                self._log_error(f"{split_name}: Empty split!")
                continue
                
            labels = [g.y.item() for g in graphs]
            counts = Counter(labels)
            total = len(labels)
            
            # Calculate ratios
            ratios = {cls: counts.get(cls, 0) / total for cls in range(len(self.class_names))}
            distributions[split_name] = ratios
            
            # Log distribution
            dist_str = ", ".join([f"{self.class_names[cls]}: {counts.get(cls, 0)} ({ratios[cls]*100:.1f}%)" 
                                  for cls in range(len(self.class_names))])
            self._log_info(f"{split_name}: {dist_str}")
        
        # Compare distributions
        if 'train' in distributions and 'val' in distributions:
            for cls in range(len(self.class_names)):
                train_ratio = distributions['train'].get(cls, 0)
                val_ratio = distributions['val'].get(cls, 0)
                test_ratio = distributions.get('test', {}).get(cls, 0)
                
                # Check train vs val
                if train_ratio > 0:
                    diff_val = abs(train_ratio - val_ratio) / train_ratio
                    diff_test = abs(train_ratio - test_ratio) / train_ratio if test_ratio else 0
                    
                    if diff_val > max_ratio_diff:
                        self._log_warning(f"Class {self.class_names[cls]}: train/val ratio differs by {diff_val*100:.1f}% "
                                         f"(train={train_ratio*100:.1f}%, val={val_ratio*100:.1f}%)")
                    
                    if diff_test > max_ratio_diff:
                        self._log_warning(f"Class {self.class_names[cls]}: train/test ratio differs by {diff_test*100:.1f}% "
                                         f"(train={train_ratio*100:.1f}%, test={test_ratio*100:.1f}%)")
    
    def check_data_leakage(self):
        """Check for subject overlap between splits (data leakage)."""
        def get_subjects(graphs):
            subjects = set()
            for g in graphs:
                if hasattr(g, 'subject_id'):
                    subjects.add(g.subject_id)
            return subjects
        
        train_subjects = get_subjects(self.train_graphs)
        val_subjects = get_subjects(self.val_graphs)
        test_subjects = get_subjects(self.test_graphs)
        
        if not train_subjects and not val_subjects and not test_subjects:
            self._log_warning("No 'subject_id' attribute found. Cannot check for data leakage.")
            return
        
        # Check overlaps
        train_val_overlap = train_subjects & val_subjects
        train_test_overlap = train_subjects & test_subjects
        val_test_overlap = val_subjects & test_subjects
        
        if train_val_overlap:
            self._log_error(f"DATA LEAKAGE: {len(train_val_overlap)} subjects in BOTH train and val: {train_val_overlap}")
        else:
            self._log_info(f"No subject overlap between train and val ({len(train_subjects)} train, {len(val_subjects)} val)")
        
        if train_test_overlap:
            self._log_error(f"DATA LEAKAGE: {len(train_test_overlap)} subjects in BOTH train and test: {train_test_overlap}")
        else:
            self._log_info(f"No subject overlap between train and test ({len(train_subjects)} train, {len(test_subjects)} test)")
        
        if val_test_overlap:
            self._log_error(f"DATA LEAKAGE: {len(val_test_overlap)} subjects in BOTH val and test: {val_test_overlap}")
        else:
            self._log_info(f"No subject overlap between val and test ({len(val_subjects)} val, {len(test_subjects)} test)")
    
    def check_feature_integrity(self):
        """Check for NaN, Inf, and proper tensor shapes."""
        for split_name, graphs in [('train', self.train_graphs), 
                                    ('val', self.val_graphs), 
                                    ('test', self.test_graphs)]:
            nan_count = 0
            inf_count = 0
            shape_issues = []
            
            for i, g in enumerate(graphs):
                # Check node features
                if hasattr(g, 'x') and g.x is not None:
                    if torch.isnan(g.x).any():
                        nan_count += 1
                    if torch.isinf(g.x).any():
                        inf_count += 1
                
                # Check edge features
                if hasattr(g, 'edge_attr') and g.edge_attr is not None:
                    if torch.isnan(g.edge_attr).any():
                        nan_count += 1
                    if torch.isinf(g.edge_attr).any():
                        inf_count += 1
                
                # Check graph features
                if hasattr(g, 'graph_attr') and g.graph_attr is not None:
                    if torch.isnan(g.graph_attr).any():
                        nan_count += 1
                    if torch.isinf(g.graph_attr).any():
                        inf_count += 1
                
                # Check edge_index shape
                if hasattr(g, 'edge_index') and g.edge_index is not None:
                    if g.edge_index.shape[0] != 2:
                        shape_issues.append(f"Graph {i}: edge_index shape {g.edge_index.shape}")
            
            if nan_count > 0:
                self._log_error(f"{split_name}: Found NaN values in {nan_count} graphs")
            else:
                self._log_info(f"{split_name}: No NaN values")
            
            if inf_count > 0:
                self._log_error(f"{split_name}: Found Inf values in {inf_count} graphs")
            
            if shape_issues:
                self._log_error(f"{split_name}: Shape issues: {shape_issues[:5]}")
    
    def check_graph_structure(self):
        """Check graph structural properties."""
        for split_name, graphs in [('train', self.train_graphs), 
                                    ('val', self.val_graphs), 
                                    ('test', self.test_graphs)]:
            if not graphs:
                continue
            
            node_counts = [g.num_nodes for g in graphs]
            edge_counts = [g.edge_index.shape[1] for g in graphs]
            
            # Check consistency
            unique_nodes = set(node_counts)
            if len(unique_nodes) == 1:
                self._log_info(f"{split_name}: All graphs have {list(unique_nodes)[0]} nodes")
            else:
                self._log_info(f"{split_name}: Node counts vary: min={min(node_counts)}, max={max(node_counts)}, "
                              f"mean={np.mean(node_counts):.1f}")
            
            # Check for isolated nodes (no edges)
            isolated_graphs = sum(1 for ec in edge_counts if ec == 0)
            if isolated_graphs > 0:
                self._log_warning(f"{split_name}: {isolated_graphs} graphs have no edges")


def validate_dataset_from_config(config_path: str) -> bool:
    """
    Load dataset from config and validate.
    
    Args:
        config_path: Path to config.yaml
        
    Returns:
        True if validation passes, False otherwise
    """
    import yaml
    from data import create_dataset_from_config
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loading dataset from config: {config_path}")
    train_graphs, val_graphs, test_graphs = create_dataset_from_config(config)
    
    class_names = config['data'].get('conditions', ['DMT', 'EC', 'EO'])
    
    validator = DatasetValidator(train_graphs, val_graphs, test_graphs, class_names)
    return validator.validate_all()


def validate_dataset_from_cache(cache_path: str, class_names: List[str] = None) -> bool:
    """
    Load dataset from cache file and validate.
    
    Args:
        cache_path: Path to cached .pkl file
        class_names: List of class names
        
    Returns:
        True if validation passes, False otherwise
    """
    import pickle
    
    print(f"Loading dataset from cache: {cache_path}")
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    
    train_graphs = data.get('train', [])
    val_graphs = data.get('val', [])
    test_graphs = data.get('test', [])
    
    if class_names is None:
        # Infer from labels
        all_labels = set([g.y.item() for g in train_graphs + val_graphs + test_graphs])
        class_names = [f'Class_{i}' for i in range(max(all_labels) + 1)]
    
    validator = DatasetValidator(train_graphs, val_graphs, test_graphs, class_names)
    return validator.validate_all()


# Pytest-compatible test functions
def test_labels_valid():
    """Test that all labels are within valid range."""
    import yaml
    from data import create_dataset_from_config
    
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    train_graphs, val_graphs, test_graphs = create_dataset_from_config(config)
    num_classes = len(config['data']['conditions'])
    
    for graphs in [train_graphs, val_graphs, test_graphs]:
        for g in graphs:
            label = g.y.item()
            assert 0 <= label < num_classes, f"Invalid label {label}, expected [0, {num_classes-1}]"


def test_no_data_leakage():
    """Test that no subject appears in multiple splits."""
    import yaml
    from data import create_dataset_from_config
    
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    train_graphs, val_graphs, test_graphs = create_dataset_from_config(config)
    
    def get_subjects(graphs):
        return {g.subject_id for g in graphs if hasattr(g, 'subject_id')}
    
    train_subj = get_subjects(train_graphs)
    val_subj = get_subjects(val_graphs)
    test_subj = get_subjects(test_graphs)
    
    assert len(train_subj & val_subj) == 0, f"Data leakage: train/val overlap: {train_subj & val_subj}"
    assert len(train_subj & test_subj) == 0, f"Data leakage: train/test overlap: {train_subj & test_subj}"
    assert len(val_subj & test_subj) == 0, f"Data leakage: val/test overlap: {val_subj & test_subj}"


def test_class_distribution_balanced():
    """Test that class distribution is roughly similar across splits."""
    import yaml
    from data import create_dataset_from_config
    
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    train_graphs, val_graphs, test_graphs = create_dataset_from_config(config)
    
    def get_ratio(graphs, cls):
        labels = [g.y.item() for g in graphs]
        return labels.count(cls) / len(labels) if labels else 0
    
    num_classes = len(config['data']['conditions'])
    max_diff = 0.35  # Allow 35% difference
    
    for cls in range(num_classes):
        train_ratio = get_ratio(train_graphs, cls)
        val_ratio = get_ratio(val_graphs, cls)
        
        if train_ratio > 0:
            diff = abs(train_ratio - val_ratio) / train_ratio
            assert diff <= max_diff, f"Class {cls}: train/val ratio differs by {diff*100:.1f}% (max {max_diff*100}%)"


def test_no_nan_values():
    """Test that no NaN values exist in features."""
    import yaml
    from data import create_dataset_from_config
    
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    train_graphs, val_graphs, test_graphs = create_dataset_from_config(config)
    
    for split_name, graphs in [('train', train_graphs), ('val', val_graphs), ('test', test_graphs)]:
        for i, g in enumerate(graphs):
            if hasattr(g, 'x') and g.x is not None:
                assert not torch.isnan(g.x).any(), f"{split_name} graph {i}: NaN in node features"
            if hasattr(g, 'edge_attr') and g.edge_attr is not None:
                assert not torch.isnan(g.edge_attr).any(), f"{split_name} graph {i}: NaN in edge features"


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate dataset before training')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--cache', type=str, default=None,
                       help='Path to cache file (alternative to config)')
    parser.add_argument('--classes', type=str, nargs='+', default=None,
                       help='Class names (e.g., DMT EC)')
    
    args = parser.parse_args()
    
    if args.cache:
        success = validate_dataset_from_cache(args.cache, args.classes)
    else:
        success = validate_dataset_from_config(args.config)
    
    sys.exit(0 if success else 1)

