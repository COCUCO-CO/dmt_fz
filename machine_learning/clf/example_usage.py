#!/usr/bin/env python3
"""
Example usage of trained GAT model for predictions.

This script demonstrates how to:
1. Load a trained model
2. Make predictions on new data
3. Extract attention weights
4. Interpret results
"""

import torch
import yaml
from pathlib import Path

from data import create_dataset_from_config
from models import create_model_from_config
from torch_geometric.loader import DataLoader


def load_trained_model(checkpoint_path, config_path=None):
    """
    Load a trained GAT model.
    
    Args:
        checkpoint_path: Path to saved checkpoint
        config_path: Optional path to config (if not in checkpoint)
        
    Returns:
        Tuple of (model, config)
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get config
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = checkpoint['config']
    
    # Determine feature dimensions from config or checkpoint
    # You need at least one sample graph to know dimensions
    num_workers = config.get('dataset_workers', None)
    _, _, test_graphs = create_dataset_from_config(config, num_workers=num_workers)
    sample = test_graphs[0]
    
    num_node_features = sample.x.shape[1]
    num_edge_features = sample.edge_attr.shape[1] if hasattr(sample, 'edge_attr') else 0
    num_graph_features = sample.graph_attr.shape[0] if hasattr(sample, 'graph_attr') else 0
    num_classes = len(config['data']['conditions'])
    
    # Create model
    model = create_model_from_config(
        config,
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        num_graph_features=num_graph_features,
        num_classes=num_classes
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded from epoch {checkpoint['epoch']}")
    print(f"✓ Best validation accuracy: {checkpoint['val_acc']:.4f}")
    
    return model, config


def predict_single_graph(model, graph, device='cpu'):
    """
    Predict class for a single graph.
    
    Args:
        model: Trained GAT model
        graph: PyTorch Geometric Data object
        device: torch device
        
    Returns:
        Dictionary with prediction, probabilities, and attention
    """
    model = model.to(device)
    graph = graph.to(device)
    
    with torch.no_grad():
        # Forward pass
        log_probs = model(graph)
        probs = torch.exp(log_probs)
        pred_class = log_probs.argmax(dim=1).item()
        
        # Get attention weights
        edge_index_att, alpha = model.get_attention_weights(graph)
    
    return {
        'predicted_class': pred_class,
        'probabilities': probs.cpu().numpy()[0],
        'attention_weights': alpha.cpu().numpy(),
        'edge_index': edge_index_att.cpu().numpy()
    }


def batch_predict(model, data_loader, device='cpu'):
    """
    Predict classes for a batch of graphs.
    
    Args:
        model: Trained GAT model
        data_loader: DataLoader with graphs
        device: torch device
        
    Returns:
        List of predictions
    """
    model = model.to(device)
    predictions = []
    
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            log_probs = model(data)
            probs = torch.exp(log_probs)
            pred_classes = log_probs.argmax(dim=1)
            
            for i in range(data.num_graphs):
                predictions.append({
                    'predicted_class': pred_classes[i].item(),
                    'probabilities': probs[i].cpu().numpy(),
                    'true_class': data.y[i].item() if hasattr(data, 'y') else None
                })
    
    return predictions


def main():
    """Example usage."""
    
    # Paths
    checkpoint_path = 'checkpoints/best_model.pt'
    config_path = 'config/config.yaml'  # Optional
    
    print("="*80)
    print("Graph Neural Network - Example Usage")
    print("="*80)
    print()
    
    # 1. Load model
    print("1. Loading trained model...")
    model, config = load_trained_model(checkpoint_path, config_path)
    print()
    
    # 2. Load test data
    print("2. Loading test data...")
    num_workers = config.get('dataset_workers', None)
    _, _, test_graphs = create_dataset_from_config(config, num_workers=num_workers)
    print(f"   Loaded {len(test_graphs)} test graphs")
    print()
    
    # 3. Single prediction
    print("3. Single graph prediction:")
    sample_graph = test_graphs[0]
    result = predict_single_graph(model, sample_graph)
    
    class_names = config['data']['conditions']
    pred_class_name = class_names[result['predicted_class']]
    true_class_name = class_names[sample_graph.y.item()]
    
    print(f"   Predicted: {pred_class_name}")
    print(f"   True label: {true_class_name}")
    print(f"   Probabilities:")
    for i, prob in enumerate(result['probabilities']):
        print(f"     {class_names[i]}: {prob:.4f}")
    print(f"   Attention weights shape: {result['attention_weights'].shape}")
    print()
    
    # 4. Batch prediction
    print("4. Batch prediction (first 10 graphs):")
    test_loader = DataLoader(test_graphs[:10], batch_size=5)
    predictions = batch_predict(model, test_loader)
    
    correct = sum(1 for p in predictions if p['predicted_class'] == p['true_class'])
    print(f"   Accuracy: {correct}/{len(predictions)} = {correct/len(predictions):.2%}")
    print()
    
    # 5. Per-class performance
    print("5. Per-class predictions:")
    from collections import Counter
    pred_counter = Counter(p['predicted_class'] for p in predictions)
    true_counter = Counter(p['true_class'] for p in predictions)
    
    print("   Predicted distribution:")
    for class_idx, count in sorted(pred_counter.items()):
        print(f"     {class_names[class_idx]}: {count}")
    
    print("   True distribution:")
    for class_idx, count in sorted(true_counter.items()):
        print(f"     {class_names[class_idx]}: {count}")
    print()
    
    # 6. Confidence analysis
    print("6. Prediction confidence:")
    confidences = [max(p['probabilities']) for p in predictions]
    avg_confidence = sum(confidences) / len(confidences)
    print(f"   Average confidence: {avg_confidence:.4f}")
    print(f"   Min confidence: {min(confidences):.4f}")
    print(f"   Max confidence: {max(confidences):.4f}")
    print()
    
    print("="*80)
    print("Example complete!")
    print("="*80)


if __name__ == '__main__':
    main()

