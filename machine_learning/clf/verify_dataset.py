#!/usr/bin/env python3
"""
Verify dataset integrity and label distribution.

Run this script to check:
1. Label distribution across train/val/test
2. Feature statistics per class
3. Potential data leakage
4. Class separability
"""

import argparse
import yaml
import numpy as np
import pickle
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import sys
sys.path.append(str(Path(__file__).parent))

from data import create_dataset_from_config


def verify_labels(train_graphs, val_graphs, test_graphs, class_names):
    """Verify label distribution."""
    print("\n" + "=" * 80)
    print("LABEL DISTRIBUTION")
    print("=" * 80)
    
    for name, graphs in [("Train", train_graphs), ("Val", val_graphs), ("Test", test_graphs)]:
        labels = [g.y.item() for g in graphs]
        counts = Counter(labels)
        total = len(labels)
        
        print(f"\n{name} ({total} samples):")
        for label_idx in sorted(counts.keys()):
            class_name = class_names[label_idx] if label_idx < len(class_names) else f"Class_{label_idx}"
            count = counts[label_idx]
            pct = 100 * count / total
            print(f"  {class_name} (label={label_idx}): {count:5d} ({pct:5.1f}%)")
    
    # Check for label consistency
    all_labels = set()
    for graphs in [train_graphs, val_graphs, test_graphs]:
        all_labels.update([g.y.item() for g in graphs])
    
    print(f"\nUnique labels found: {sorted(all_labels)}")
    if len(all_labels) != len(class_names):
        print(f"⚠️  WARNING: Found {len(all_labels)} unique labels but expected {len(class_names)}")


def verify_features(train_graphs, class_names):
    """Verify feature statistics per class."""
    print("\n" + "=" * 80)
    print("FEATURE STATISTICS BY CLASS")
    print("=" * 80)
    
    # Group by class
    class_features = {i: {'node': [], 'edge': [], 'graph': []} for i in range(len(class_names))}
    
    for g in train_graphs:
        label = g.y.item()
        if label < len(class_names):
            class_features[label]['node'].append(g.x.numpy())
            if hasattr(g, 'edge_attr') and g.edge_attr is not None:
                class_features[label]['edge'].append(g.edge_attr.numpy())
            if hasattr(g, 'graph_attr') and g.graph_attr is not None:
                class_features[label]['graph'].append(g.graph_attr.numpy())
    
    print("\nNode features (mean ± std):")
    for label, name in enumerate(class_names):
        if class_features[label]['node']:
            all_nodes = np.vstack(class_features[label]['node'])
            means = all_nodes.mean(axis=0)
            stds = all_nodes.std(axis=0)
            print(f"  {name}: mean={means.mean():.4f}±{means.std():.4f}, std={stds.mean():.4f}")
    
    print("\nEdge features (mean ± std):")
    for label, name in enumerate(class_names):
        if class_features[label]['edge']:
            all_edges = np.vstack(class_features[label]['edge'])
            print(f"  {name}: mean={all_edges.mean():.4f}±{all_edges.std():.4f}")
    
    print("\nGraph features (mean ± std):")
    for label, name in enumerate(class_names):
        if class_features[label]['graph']:
            all_graphs = np.vstack(class_features[label]['graph'])
            means = all_graphs.mean(axis=0)
            stds = all_graphs.std(axis=0)
            print(f"  {name}: {means.round(4).tolist()}")


def check_data_leakage(train_graphs, val_graphs, test_graphs):
    """Check for potential data leakage between splits."""
    print("\n" + "=" * 80)
    print("DATA LEAKAGE CHECK")
    print("=" * 80)
    
    # Check subject overlap
    train_subjects = set(g.subject_id for g in train_graphs if hasattr(g, 'subject_id'))
    val_subjects = set(g.subject_id for g in val_graphs if hasattr(g, 'subject_id'))
    test_subjects = set(g.subject_id for g in test_graphs if hasattr(g, 'subject_id'))
    
    print(f"\nTrain subjects: {len(train_subjects)}")
    print(f"Val subjects: {len(val_subjects)}")
    print(f"Test subjects: {len(test_subjects)}")
    
    train_val_overlap = train_subjects & val_subjects
    train_test_overlap = train_subjects & test_subjects
    val_test_overlap = val_subjects & test_subjects
    
    if train_val_overlap:
        print(f"⚠️  WARNING: {len(train_val_overlap)} subjects in both train and val: {train_val_overlap}")
    else:
        print("✓ No train/val subject overlap")
    
    if train_test_overlap:
        print(f"⚠️  WARNING: {len(train_test_overlap)} subjects in both train and test: {train_test_overlap}")
    else:
        print("✓ No train/test subject overlap")
    
    if val_test_overlap:
        print(f"⚠️  WARNING: {len(val_test_overlap)} subjects in both val and test: {val_test_overlap}")
    else:
        print("✓ No val/test subject overlap")


def compute_class_separability(train_graphs, class_names, save_dir):
    """Compute and visualize class separability using graph features."""
    print("\n" + "=" * 80)
    print("CLASS SEPARABILITY ANALYSIS")
    print("=" * 80)
    
    # Extract graph-level features
    features = []
    labels = []
    
    for g in train_graphs:
        if hasattr(g, 'graph_attr') and g.graph_attr is not None:
            features.append(g.graph_attr.numpy())
            labels.append(g.y.item())
    
    if not features:
        print("No graph features found")
        return
    
    features = np.array(features)
    labels = np.array(labels)
    
    # Compute silhouette score
    if len(np.unique(labels)) > 1:
        sil_score = silhouette_score(features, labels)
        print(f"\nSilhouette score (graph features): {sil_score:.4f}")
        print("  > 0.5: Good separation")
        print("  0.25-0.5: Moderate separation")
        print("  < 0.25: Poor separation")
    
    # PCA visualization
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    for label_idx, name in enumerate(class_names):
        mask = labels == label_idx
        ax.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                  label=name, alpha=0.5, s=10)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('Graph Features - PCA')
    ax.legend()
    
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / 'class_separability_pca.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved PCA plot to {save_dir / 'class_separability_pca.png'}")


def verify_single_sample(graphs, class_names, sample_idx=0):
    """Print detailed info about a single sample."""
    print("\n" + "=" * 80)
    print(f"SAMPLE INSPECTION (index={sample_idx})")
    print("=" * 80)
    
    g = graphs[sample_idx]
    
    print(f"\nLabel: {g.y.item()} ({class_names[g.y.item()] if g.y.item() < len(class_names) else 'unknown'})")
    print(f"Num nodes: {g.num_nodes}")
    print(f"Num edges: {g.edge_index.shape[1]}")
    
    print(f"\nNode features shape: {g.x.shape}")
    print(f"  Mean: {g.x.mean().item():.4f}")
    print(f"  Std: {g.x.std().item():.4f}")
    print(f"  Min: {g.x.min().item():.4f}")
    print(f"  Max: {g.x.max().item():.4f}")
    
    if hasattr(g, 'edge_attr') and g.edge_attr is not None:
        print(f"\nEdge features shape: {g.edge_attr.shape}")
        print(f"  Mean: {g.edge_attr.mean().item():.4f}")
        print(f"  Std: {g.edge_attr.std().item():.4f}")
    
    if hasattr(g, 'graph_attr') and g.graph_attr is not None:
        print(f"\nGraph features: {g.graph_attr.numpy().round(4)}")
    
    if hasattr(g, 'subject_id'):
        print(f"\nSubject ID: {g.subject_id}")
    if hasattr(g, 'condition'):
        print(f"Condition: {g.condition}")
    if hasattr(g, 'band'):
        print(f"Band: {g.band}")


def main(config_path: str):
    """Main verification function."""
    print("=" * 80)
    print("DATASET VERIFICATION")
    print("=" * 80)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    class_names = config['data']['conditions']
    output_dir = Path(config['paths']['output_dir']) / 'verification'
    
    print(f"\nLoading dataset...")
    print(f"Expected classes: {class_names}")
    
    # Load dataset
    num_workers = config.get('dataset_workers', None)
    train_graphs, val_graphs, test_graphs = create_dataset_from_config(
        config, num_workers=num_workers
    )
    
    print(f"\nLoaded: {len(train_graphs)} train, {len(val_graphs)} val, {len(test_graphs)} test")
    
    # Run all verification checks
    verify_labels(train_graphs, val_graphs, test_graphs, class_names)
    verify_features(train_graphs, class_names)
    check_data_leakage(train_graphs, val_graphs, test_graphs)
    compute_class_separability(train_graphs, class_names, output_dir)
    
    # Inspect samples from each class
    for class_idx, class_name in enumerate(class_names):
        samples = [i for i, g in enumerate(train_graphs) if g.y.item() == class_idx]
        if samples:
            print(f"\n--- Sample from class {class_name} ---")
            verify_single_sample(train_graphs, class_names, samples[0])
    
    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify dataset integrity')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    main(args.config)


