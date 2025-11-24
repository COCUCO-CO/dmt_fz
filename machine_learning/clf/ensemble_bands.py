#!/usr/bin/env python3
"""
Ensemble predictions from multiple band-specific models.

Combines predictions from models trained on different frequency bands
using various fusion strategies (voting, averaging, weighted).
"""

import argparse
import sys
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any
import logging

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent))

from data import create_dataset_from_config
from models import create_model_from_config

logger = logging.getLogger(__name__)


def load_band_model(checkpoint_path: Path, device: torch.device):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint['config']
    model = create_model_from_config(
        config,
        num_node_features=checkpoint['num_node_features'],
        num_edge_features=checkpoint['num_edge_features'],
        num_graph_features=checkpoint['num_graph_features'],
        num_classes=checkpoint['num_classes']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config


def predict_with_model(model, dataloader, device):
    """Get predictions from a single model."""
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            output = model(batch)
            probs = torch.exp(output)  # Convert log probs to probs
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())
    
    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)
    
    return all_probs, all_labels


def ensemble_predict(band_predictions: Dict[str, np.ndarray], 
                     method: str = 'average',
                     weights: Dict[str, float] = None) -> np.ndarray:
    """
    Combine predictions from multiple bands.
    
    Args:
        band_predictions: Dict of {band_name: probability_matrix}
        method: 'average', 'weighted', 'voting', 'max'
        weights: Optional weights for each band (for weighted method)
        
    Returns:
        Final predictions (class indices)
    """
    bands = list(band_predictions.keys())
    probs = [band_predictions[band] for band in bands]
    
    if method == 'average':
        # Simple average of probabilities
        ensemble_probs = np.mean(probs, axis=0)
        
    elif method == 'weighted':
        # Weighted average
        if weights is None:
            raise ValueError("Weights must be provided for weighted method")
        
        weight_array = np.array([weights[band] for band in bands])
        weight_array = weight_array / weight_array.sum()  # Normalize
        
        ensemble_probs = np.average(probs, axis=0, weights=weight_array)
        
    elif method == 'voting':
        # Hard voting: each model votes for one class
        votes = np.array([np.argmax(p, axis=1) for p in probs])
        ensemble_preds = []
        
        for i in range(votes.shape[1]):
            vote_counts = np.bincount(votes[:, i])
            ensemble_preds.append(np.argmax(vote_counts))
        
        return np.array(ensemble_preds)
        
    elif method == 'max':
        # Take maximum probability across bands
        ensemble_probs = np.max(probs, axis=0)
    
    else:
        raise ValueError(f"Unknown ensemble method: {method}")
    
    # Convert probabilities to predictions
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    
    return ensemble_preds


def main():
    parser = argparse.ArgumentParser(description='Ensemble predictions from band-specific models')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--method', type=str, default='average',
                       choices=['average', 'weighted', 'voting', 'max'],
                       help='Ensemble method')
    parser.add_argument('--bands', type=str, nargs='+',
                       default=['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'],
                       help='Bands to ensemble')
    parser.add_argument('--output', type=str, default='output/ensemble_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*80}")
    print("Ensemble Prediction from Band-Specific Models")
    print(f"{'='*80}\n")
    print(f"Bands: {', '.join(args.bands)}")
    print(f"Method: {args.method}")
    print(f"Device: {device}")
    
    # Load models
    print("\nLoading models...")
    models = {}
    
    for band in args.bands:
        checkpoint_path = Path(f"checkpoints_{band.lower()}/best_model.pt")
        
        if not checkpoint_path.exists():
            print(f"  ✗ {band}: checkpoint not found at {checkpoint_path}")
            continue
        
        try:
            model, band_config = load_band_model(checkpoint_path, device)
            models[band] = model
            print(f"  ✓ {band}: loaded")
        except Exception as e:
            print(f"  ✗ {band}: error loading - {e}")
    
    if len(models) == 0:
        print("\nNo models loaded. Exiting.")
        return
    
    print(f"\nSuccessfully loaded {len(models)} models")
    
    # Load test data for each band
    print("\nGetting predictions from each band...")
    band_predictions = {}
    true_labels = None
    
    for band in models.keys():
        # Load dataset for this band
        band_config = config.copy()
        band_config['data']['bands'] = [band]
        
        _, _, test_data = create_dataset_from_config(band_config, force_rebuild=False)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
        
        # Get predictions
        probs, labels = predict_with_model(models[band], test_loader, device)
        band_predictions[band] = probs
        
        if true_labels is None:
            true_labels = labels
        
        # Calculate individual band accuracy
        preds = np.argmax(probs, axis=1)
        acc = accuracy_score(labels, preds)
        
        print(f"  {band}: {acc*100:.2f}% accuracy")
    
    # Ensemble predictions
    print(f"\nEnsembling predictions using {args.method} method...")
    ensemble_preds = ensemble_predict(band_predictions, method=args.method)
    
    # Calculate ensemble metrics
    ensemble_acc = accuracy_score(true_labels, ensemble_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, ensemble_preds, average='macro'
    )
    
    print(f"\n{'='*80}")
    print("Ensemble Results")
    print(f"{'='*80}")
    print(f"Accuracy:  {ensemble_acc*100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, ensemble_preds)
    
    # Save results
    results = {
        'method': args.method,
        'bands': list(models.keys()),
        'ensemble_accuracy': float(ensemble_acc),
        'ensemble_precision': float(precision),
        'ensemble_recall': float(recall),
        'ensemble_f1': float(f1),
        'confusion_matrix': cm.tolist(),
        'individual_accuracies': {}
    }
    
    # Add individual band accuracies
    for band, probs in band_predictions.items():
        preds = np.argmax(probs, axis=1)
        acc = accuracy_score(true_labels, preds)
        results['individual_accuracies'][band] = float(acc)
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=config['data']['conditions'],
                yticklabels=config['data']['conditions'])
    plt.title(f'Ensemble Confusion Matrix ({args.method})', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    cm_path = output_path.parent / 'ensemble_confusion_matrix.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confusion matrix saved to: {cm_path}")
    
    print(f"\n{'='*80}")
    print("Comparison: Ensemble vs Individual Bands")
    print(f"{'='*80}")
    
    print(f"\n{'Model':<15} {'Accuracy':<12}")
    print("-" * 27)
    
    for band in sorted(models.keys()):
        acc = results['individual_accuracies'][band]
        print(f"{band:<15} {acc*100:>6.2f}%")
    
    print("-" * 27)
    print(f"{'ENSEMBLE':<15} {ensemble_acc*100:>6.2f}%")
    
    # Check if ensemble is better
    best_single = max(results['individual_accuracies'].values())
    improvement = ensemble_acc - best_single
    
    if improvement > 0:
        print(f"\n✓ Ensemble improves by {improvement*100:.2f}% over best single band")
    else:
        print(f"\n⚠ Ensemble is {abs(improvement)*100:.2f}% worse than best single band")


if __name__ == '__main__':
    main()

