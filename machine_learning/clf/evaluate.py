#!/usr/bin/env python3
"""
Evaluation script for trained GAT model.

Loads a trained model and evaluates it on test set with detailed metrics.
"""

import argparse
import sys
import yaml
from pathlib import Path
import logging

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support
)

sys.path.append(str(Path(__file__).parent))

from data import create_dataset_from_config
from models import create_model_from_config
from utils import plot_confusion_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_model(model: nn.Module,
                   loader: DataLoader,
                   device: torch.device,
                   class_names: list) -> dict:
    """
    Comprehensive evaluation of model.
    
    Returns:
        Dictionary with all metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    for data in loader:
        data = data.to(device)
        out = model(data)
        
        probs = torch.exp(out)  # Convert log-probs to probs
        preds = out.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Overall metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None
    )
    
    # Macro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro'
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Per-class accuracy
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    results = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'per_class': {
            class_names[i]: {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': int(support[i]),
                'accuracy': per_class_accuracy[i]
            }
            for i in range(len(class_names))
        },
        'confusion_matrix': cm.tolist(),
        'predictions': all_preds.tolist(),
        'labels': all_labels.tolist(),
        'probabilities': all_probs.tolist()
    }
    
    return results


def main(checkpoint_path: str, config_path: str = None):
    """Main evaluation function."""
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get config (from checkpoint or separate file)
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = checkpoint.get('config')
        if config is None:
            raise ValueError("No config found in checkpoint and no config file provided")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading test dataset...")
    num_workers = config.get('dataset_workers', None)
    _, _, test_graphs = create_dataset_from_config(config, num_workers=num_workers)
    
    test_loader = DataLoader(
        test_graphs,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    
    # Create model
    sample_graph = test_graphs[0]
    num_node_features = sample_graph.x.shape[1]
    num_edge_features = sample_graph.edge_attr.shape[1] if hasattr(sample_graph, 'edge_attr') else 0
    num_graph_features = sample_graph.graph_attr.shape[0] if hasattr(sample_graph, 'graph_attr') else 0
    num_classes = len(config['data']['conditions'])
    
    model = create_model_from_config(
        config,
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        num_graph_features=num_graph_features,
        num_classes=num_classes
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    logger.info(f"Model loaded from epoch {checkpoint['epoch']}")
    logger.info(f"Best validation accuracy: {checkpoint['val_acc']:.4f}")
    
    # Evaluate
    logger.info("\nEvaluating on test set...")
    results = evaluate_model(
        model, test_loader, device,
        class_names=config['data']['conditions']
    )
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"\nOverall Accuracy: {results['accuracy']:.4f}")
    print(f"Macro Precision:  {results['precision_macro']:.4f}")
    print(f"Macro Recall:     {results['recall_macro']:.4f}")
    print(f"Macro F1:         {results['f1_macro']:.4f}")
    
    print("\nPer-Class Results:")
    print("-" * 80)
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Accuracy':<12} {'Support':<10}")
    print("-" * 80)
    
    for class_name, metrics in results['per_class'].items():
        print(f"{class_name:<10} "
              f"{metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} "
              f"{metrics['f1']:<12.4f} "
              f"{metrics['accuracy']:<12.4f} "
              f"{metrics['support']:<10}")
    
    print("\nConfusion Matrix:")
    print("-" * 80)
    cm = np.array(results['confusion_matrix'])
    class_names = config['data']['conditions']
    
    # Print header
    print(f"{'True/Pred':<12}", end='')
    for name in class_names:
        print(f"{name:<10}", end='')
    print()
    print("-" * 80)
    
    # Print rows
    for i, name in enumerate(class_names):
        print(f"{name:<12}", end='')
        for j in range(len(class_names)):
            print(f"{cm[i,j]:<10}", end='')
        print()
    
    print("="*80)
    
    # Save detailed results
    output_dir = Path(config['paths']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    results_clean = {k: v for k, v in results.items() 
                     if k not in ['predictions', 'labels', 'probabilities']}
    
    with open(output_dir / 'detailed_evaluation.json', 'w') as f:
        json.dump(results_clean, f, indent=2)
    
    logger.info(f"\nDetailed results saved to: {output_dir / 'detailed_evaluation.json'}")
    
    # Plot confusion matrix
    plot_confusion_matrix(
        np.array(results['labels']),
        np.array(results['predictions']),
        class_names=class_names,
        save_path=output_dir / 'confusion_matrix_detailed.png',
        normalize=True
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained GAT model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (if not in checkpoint)')
    
    args = parser.parse_args()
    main(args.checkpoint, args.config)

