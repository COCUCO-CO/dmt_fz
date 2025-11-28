#!/usr/bin/env python3
"""
Extract GAT layer activations from trained VAE for clustering analysis.

This script processes all EEG data through a trained VAE model and extracts
intermediate activations from GAT layers. The output is formatted to be
compatible with the clustering.py script in the pipeline.

Usage:
    python extract_activations.py --checkpoint best_model.pt
    python extract_activations.py --checkpoint best_model.pt --layers 0 1 2
    python extract_activations.py --checkpoint best_model.pt --per-subject

Output format (compatible with clustering.py):
    - activations_layer_{i}.pkl: Dict with structure similar to eigen_all.pkl
      {condition: {band: [list of activations per epoch]}}
    - attention_weights_layer_{i}.pkl: Attention weight matrices
"""

import argparse
import sys
import yaml
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict
import logging

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from data import create_dataset_from_config
from models import create_vae_from_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def load_model_and_data(checkpoint_path: Path, config: Dict[str, Any], device: torch.device):
    """
    Load trained VAE model from checkpoint and return datasets.
    
    This function loads the dataset once and reuses it, avoiding double loading.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration dictionary
        device: Torch device
        
    Returns:
        Tuple of (model, train_graphs, val_graphs, test_graphs)
    """
    # Load datasets (only once)
    dataset_workers = config.get('dataset_workers', 4)
    logger.info("Loading datasets...")
    train_graphs, val_graphs, test_graphs = create_dataset_from_config(
        config, num_workers=dataset_workers
    )
    
    # Infer dimensions from first sample
    sample = train_graphs[0]
    num_node_features = sample.x.shape[1]
    num_edge_features = sample.edge_attr.shape[1] if hasattr(sample, 'edge_attr') else 0
    num_graph_features = sample.graph_attr.shape[0] if hasattr(sample, 'graph_attr') else 0
    num_nodes = sample.num_nodes
    
    logger.info(f"  Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")
    logger.info(f"  Node features: {num_node_features}, Edge features: {num_edge_features}")
    logger.info(f"  Nodes per graph: {num_nodes}")
    
    # Create model
    model = create_vae_from_config(
        config,
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        num_graph_features=num_graph_features,
        num_nodes=num_nodes
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {checkpoint_path}")
    logger.info(f"  Best epoch: {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"  Val loss: {checkpoint.get('val_loss', 'unknown')}")
    
    return model, train_graphs, val_graphs, test_graphs


def extract_activations_for_dataset(model,
                                     graphs: List,
                                     device: torch.device,
                                     batch_size: int = 32,
                                     num_workers: int = 4) -> Dict[str, Any]:
    """
    Extract activations for all graphs in a dataset.
    
    Args:
        model: Trained VAE model
        graphs: List of graph data objects
        device: Torch device
        batch_size: Batch size for processing
        num_workers: DataLoader workers
        
    Returns:
        Dict with activations organized by layer and metadata
    """
    loader = DataLoader(
        graphs,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    # Storage for results
    all_activations = defaultdict(list)
    all_attention = defaultdict(list)
    metadata = []
    
    model.eval()
    with torch.no_grad():
        for data in tqdm(loader, desc="Extracting activations"):
            data = data.to(device)
            
            # Forward pass with activation storage
            activations = model.get_all_activations(data)
            
            # Store layer activations
            for key, value in activations.items():
                if 'encoder.layer' in key:
                    all_activations[key].append(value)
                elif 'attention.layer' in key:
                    all_attention[key].append(value)
                elif key in ['pooled', 'mu', 'z']:
                    all_activations[key].append(value)
            
            # Store metadata for each graph in batch
            batch_size_actual = data.num_graphs
            
            # Get number of nodes per graph (assuming all graphs have same num_nodes)
            # For batched data, we can get it from the first graph's ptr or from data
            if hasattr(data, 'ptr') and data.ptr is not None:
                # ptr contains cumulative node counts [0, n1, n1+n2, ...]
                nodes_per_graph = (data.ptr[1] - data.ptr[0]).item()
            else:
                # Fallback: assume equal distribution
                nodes_per_graph = data.x.shape[0] // batch_size_actual
            
            for i in range(batch_size_actual):
                # Safely extract metadata attributes (handles both list and single value cases)
                def safe_get_attr(attr_name, default='unknown'):
                    if not hasattr(data, attr_name):
                        return default
                    attr = getattr(data, attr_name)
                    if attr is None:
                        return default
                    # Handle list-like attributes (batched)
                    if isinstance(attr, (list, tuple)):
                        return attr[i] if i < len(attr) else default
                    # Handle single value (batch size 1)
                    if batch_size_actual == 1:
                        return attr
                    return default
                
                meta = {
                    'subject_id': safe_get_attr('subject_id'),
                    'condition': safe_get_attr('condition'),
                    'band': safe_get_attr('band'),
                    'epoch_idx': safe_get_attr('epoch_idx', i),
                    'label': data.y[i].item(),
                    'num_nodes': nodes_per_graph  # Added for proper node activation handling
                }
                metadata.append(meta)
    
    return {
        'activations': dict(all_activations),
        'attention': dict(all_attention),
        'metadata': metadata
    }


def organize_encoder_activations(extracted: Dict[str, Any],
                                  conditions: List[str],
                                  bands: List[str]) -> Dict[str, Dict]:
    """
    Organize encoder layer activations by aggregating node-level to graph-level.
    
    For node-level activations from encoder layers, we apply mean pooling
    to get one vector per graph, making it compatible with clustering.py.
    
    Args:
        extracted: Output from extract_activations_for_dataset
        conditions: List of condition names
        bands: List of band names
        
    Returns:
        Dict organized by layer, then condition, then band
    """
    activations = extracted['activations']
    metadata = extracted['metadata']
    
    organized = {}
    
    for layer_key in activations.keys():
        # Skip graph-level keys (handled by organize_pooled_activations)
        if layer_key in ['pooled', 'mu', 'z']:
            continue
            
        # Only process encoder layer activations
        if 'encoder.layer' not in layer_key:
            continue
        
        # Initialize structure for this layer
        layer_data = {cond: {band: [] for band in bands} for cond in conditions}
        
        # Get all activation tensors for this layer
        layer_acts = activations[layer_key]
        
        # Process each batch and aggregate nodes to graph level
        graph_idx = 0
        
        for batch_acts in layer_acts:
            if isinstance(batch_acts, torch.Tensor):
                batch_acts = batch_acts.numpy()
            
            # Determine how many graphs are in this batch
            # We need to split the node activations by graph
            batch_start = graph_idx
            
            # Find the nodes per graph from metadata
            current_pos = 0
            while graph_idx < len(metadata) and current_pos < len(batch_acts):
                num_nodes = metadata[graph_idx].get('num_nodes', 24)
                
                if current_pos + num_nodes > len(batch_acts):
                    break
                
                # Extract this graph's node activations and mean pool
                graph_nodes = batch_acts[current_pos:current_pos + num_nodes]
                graph_embedding = np.mean(graph_nodes, axis=0)  # Mean pooling
                
                # Get metadata for this graph
                meta = metadata[graph_idx]
                cond = meta['condition']
                band = meta['band']
                
                if cond in layer_data and band in layer_data[cond]:
                    layer_data[cond][band].append(graph_embedding)
                
                current_pos += num_nodes
                graph_idx += 1
        
        organized[layer_key] = layer_data
        logger.info(f"  Organized {layer_key}: {sum(len(layer_data[c][b]) for c in conditions for b in bands)} graphs")
    
    return organized


def organize_pooled_activations(extracted: Dict[str, Any],
                                 conditions: List[str],
                                 bands: List[str]) -> Dict[str, Any]:
    """
    Organize pooled (graph-level) activations for clustering.
    
    This is the most useful format for clustering.py since it provides
    one vector per epoch (like eigenvalues do).
    
    Args:
        extracted: Output from extract_activations_for_dataset
        conditions: List of condition names
        bands: List of band names
        
    Returns:
        Dict with structure compatible with clustering.py
    """
    activations = extracted['activations']
    metadata = extracted['metadata']
    
    result = {}
    
    # Process each type of graph-level representation
    for layer_key in ['pooled', 'mu', 'z']:
        if layer_key not in activations:
            continue
        
        # Initialize structure
        layer_data = {cond: {band: [] for band in bands} for cond in conditions}
        
        # Concatenate all batches
        layer_acts = activations[layer_key]
        all_acts = torch.cat(layer_acts, dim=0).numpy()
        
        # Organize by condition and band
        for idx, meta in enumerate(metadata):
            cond = meta['condition']
            band = meta['band']
            
            if cond in layer_data and band in layer_data[cond]:
                if idx < len(all_acts):
                    layer_data[cond][band].append(all_acts[idx])
        
        result[layer_key] = layer_data
    
    return result


def save_for_clustering(organized: Dict[str, Any],
                        output_dir: Path,
                        prefix: str = "vae_activations"):
    """
    Save organized activations in clustering.py compatible format.
    
    Args:
        organized: Organized activations dict
        output_dir: Output directory
        prefix: Filename prefix
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for layer_key, layer_data in organized.items():
        # Clean key for filename
        clean_key = layer_key.replace('.', '_').replace('/', '_')
        filepath = output_dir / f"{prefix}_{clean_key}.pkl"
        
        with open(filepath, 'wb') as f:
            pickle.dump(layer_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Log statistics
        total = sum(len(layer_data[c][b]) for c in layer_data for b in layer_data[c])
        logger.info(f"Saved {layer_key}: {total} epochs -> {filepath}")


def save_per_subject(extracted: Dict[str, Any],
                     output_dir: Path):
    """
    Save activations grouped by subject for detailed analysis.
    
    Args:
        extracted: Output from extract_activations_for_dataset
        output_dir: Output directory
    """
    metadata = extracted['metadata']
    
    # Group by subject
    subject_data = defaultdict(lambda: {
        'activations': defaultdict(list),
        'metadata': []
    })
    
    # Get graph-level activations
    for layer_key in ['pooled', 'mu', 'z']:
        if layer_key not in extracted['activations']:
            continue
        
        layer_acts = torch.cat(extracted['activations'][layer_key], dim=0).numpy()
        
        for idx, meta in enumerate(metadata):
            subject = meta['subject_id']
            if idx < len(layer_acts):
                subject_data[subject]['activations'][layer_key].append(layer_acts[idx])
    
    # Add metadata
    for idx, meta in enumerate(metadata):
        subject = meta['subject_id']
        subject_data[subject]['metadata'].append(meta)
    
    # Save per subject
    subjects_dir = output_dir / 'per_subject'
    subjects_dir.mkdir(parents=True, exist_ok=True)
    
    for subject, data in subject_data.items():
        # Convert lists to arrays
        for key in data['activations']:
            data['activations'][key] = np.array(data['activations'][key])
        
        filepath = subjects_dir / f'{subject}_activations.pkl'
        with open(filepath, 'wb') as f:
            pickle.dump(dict(data), f)
    
    logger.info(f"Saved {len(subject_data)} subject files to {subjects_dir}")


def save_attention_weights(extracted: Dict[str, Any],
                           output_dir: Path):
    """
    Save attention weights for visualization.
    
    Args:
        extracted: Output from extract_activations_for_dataset
        output_dir: Output directory
    """
    attention = extracted['attention']
    metadata = extracted['metadata']
    
    attention_dir = output_dir / 'attention_weights'
    attention_dir.mkdir(parents=True, exist_ok=True)
    
    for layer_key, layer_att in attention.items():
        # Save all attention data for this layer
        filepath = attention_dir / f'{layer_key.replace(".", "_")}.pkl'
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'attention': layer_att,
                'metadata': metadata
            }, f)
        
        logger.info(f"Saved attention weights: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract GAT activations from trained VAE for clustering'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config (default: from checkpoint dir)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (default: activations_dir from config)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for processing')
    parser.add_argument('--layers', type=int, nargs='+', default=None,
                       help='Specific layer indices to extract (default: all)')
    parser.add_argument('--per-subject', action='store_true',
                       help='Also save per-subject activations')
    parser.add_argument('--save-attention', action='store_true',
                       help='Save attention weight matrices')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Find config
    checkpoint_path = Path(args.checkpoint)
    if args.config:
        config_path = Path(args.config)
    else:
        # Try to find config in same directory structure
        possible_paths = [
            checkpoint_path.parent.parent / 'config.yaml',
            checkpoint_path.parent / 'config.yaml',
            Path('config/config.yaml')
        ]
        config_path = None
        for p in possible_paths:
            if p.exists():
                config_path = p
                break
        
        if config_path is None:
            logger.error("Could not find config file. Please specify with --config")
            sys.exit(1)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Using config: {config_path}")
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model and datasets (single load, no duplication)
    model, train_graphs, val_graphs, test_graphs = load_model_and_data(
        checkpoint_path, config, device
    )
    
    # Combine all graphs
    all_graphs = train_graphs + val_graphs + test_graphs
    logger.info(f"Total graphs: {len(all_graphs)}")
    
    # Extract activations
    logger.info("Extracting activations...")
    extracted = extract_activations_for_dataset(
        model, all_graphs, device,
        batch_size=args.batch_size,
        num_workers=config.get('num_workers', 4)
    )
    
    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(config['paths']['activations_dir'])
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Get conditions and bands from config
    conditions = config['data']['conditions']
    bands = config['data']['bands']
    
    # Organize and save for clustering
    logger.info("Organizing graph-level activations for clustering...")
    organized_pooled = organize_pooled_activations(extracted, conditions, bands)
    save_for_clustering(organized_pooled, output_dir)
    
    # Also organize encoder layer activations (node-level -> mean pooled to graph-level)
    logger.info("Organizing encoder layer activations (mean pooled)...")
    organized_encoder = organize_encoder_activations(extracted, conditions, bands)
    save_for_clustering(organized_encoder, output_dir, prefix="vae_encoder")
    
    # Optional: save per-subject
    if args.per_subject:
        logger.info("Saving per-subject activations...")
        save_per_subject(extracted, output_dir)
    
    # Optional: save attention weights
    if args.save_attention:
        logger.info("Saving attention weights...")
        save_attention_weights(extracted, output_dir)
    
    # Save metadata
    meta_path = output_dir / 'extraction_metadata.pkl'
    with open(meta_path, 'wb') as f:
        pickle.dump({
            'checkpoint': str(checkpoint_path),
            'config': str(config_path),
            'num_graphs': len(all_graphs),
            'conditions': conditions,
            'bands': bands,
            'layers_extracted': list(extracted['activations'].keys())
        }, f)
    
    logger.info("=" * 60)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Activations saved to: {output_dir}")
    logger.info(f"Files created:")
    for f in sorted(output_dir.glob("*.pkl")):
        logger.info(f"  - {f.name}")
    logger.info("")
    logger.info("To run clustering on these activations:")
    logger.info(f"  python pipeline/clustering.py --input {output_dir}/vae_activations_mu.pkl")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

