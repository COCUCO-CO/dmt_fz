#!/usr/bin/env python3
"""
Hyperparameter Search for DMT-Only VAE

25 experiments with:
- ChebConv and GATv2 convolutions
- Graph convolutional decoder (2-3 layers)
- Subject-wise splits (no data leakage)
- Extensive logging to disk and tensorboard

Evaluation metrics:
1. Reconstruction loss (primary)
2. KL divergence
3. Latent space quality (silhouette score by subject)
4. Generalization gap (train vs val loss)
"""

import os
import sys
import yaml
import json
import argparse
import subprocess
import random
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
import numpy as np
from copy import deepcopy

# Search space for DMT-only VAE
SEARCH_SPACE = {
    # Encoder architecture
    'encoder.conv_type': ['gatv2', 'cheby'],
    'encoder.hidden_dim': [32, 64, 128],
    'encoder.num_gat_layers': [2, 3],  # 2-3 layers only
    'encoder.num_attention_heads': [2, 4, 8],
    'encoder.cheby_k': [2, 3, 4],
    'encoder.dropout': [0.1, 0.2, 0.3],
    
    # Latent space
    'latent.dim': [16, 32, 64],
    
    # Decoder architecture
    'decoder.hidden_dims_config': ['small', 'medium', 'large'],  # Will be expanded
    'decoder.use_graph_conv': [True, False],  # Use graph convolutions in decoder
    'decoder.reconstruct_edges': [True, False],
    
    # Loss weights
    'loss.reconstruction.node_weight': [0.3, 0.5, 1.0],
    'loss.reconstruction.edge_weight': [0.5, 1.0, 2.0],
    'loss.kl.weight': [0.001, 0.01, 0.05],
    
    # KL Annealing
    'loss.kl.annealing.epochs': [20, 40],
    'loss.kl.annealing.type': ['linear', 'cosine'],
    
    # Free bits
    'loss.free_bits': [0.0, 0.1, 0.25],
    
    # Training
    'training.batch_size': [64, 128, 256],
    'training.learning_rate': [0.0005, 0.001, 0.002],
    
    # Pooling
    'pooling.method': ['mean', 'max', 'mean+max'],
}


def set_nested(d: dict, keys: str, value):
    """Set a nested dict value from dot-separated key path."""
    parts = keys.split('.')
    for key in parts[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]
    d[parts[-1]] = value


def get_nested(d: dict, keys: str, default=None):
    """Get a nested dict value from dot-separated key path."""
    parts = keys.split('.')
    for key in parts:
        if isinstance(d, dict) and key in d:
            d = d[key]
        else:
            return default
    return d


def sample_config(base_config: Dict, search_space: Dict, seed: int = None) -> Tuple[Dict, Dict]:
    """Sample a random configuration from search space."""
    if seed is not None:
        random.seed(seed)
    
    config = yaml.safe_load(yaml.dump(base_config))  # Deep copy
    sampled_params = {}
    
    for param_path, values in search_space.items():
        value = random.choice(values)
        sampled_params[param_path] = value
        set_nested(config, f"model.{param_path}" if not param_path.startswith(('loss', 'training', 'pooling')) 
                   else param_path.replace('pooling', 'model.pooling'), value)
    
    # Ensure consistency
    conv_type = sampled_params.get('encoder.conv_type', 'gatv2')
    config['model']['encoder']['conv_type'] = conv_type
    
    # Decoder uses same conv type as encoder when using graph conv
    use_graph_conv = sampled_params.get('decoder.use_graph_conv', True)
    config['model']['decoder']['use_graph_conv'] = use_graph_conv
    config['model']['decoder']['conv_type'] = conv_type  # Match encoder
    
    # If not using edges, set edge weight to 0
    if not sampled_params.get('decoder.reconstruct_edges', True):
        config['loss']['reconstruction']['edge_weight'] = 0.0
    
    # Expand decoder hidden_dims based on config
    dec_config = sampled_params.get('decoder.hidden_dims_config', 'medium')
    enc_hidden = sampled_params.get('encoder.hidden_dim', 64)
    
    if dec_config == 'small':
        config['model']['decoder']['hidden_dims'] = [enc_hidden, enc_hidden // 2]
    elif dec_config == 'medium':
        config['model']['decoder']['hidden_dims'] = [enc_hidden * 2, enc_hidden]
    else:  # large
        config['model']['decoder']['hidden_dims'] = [enc_hidden * 2, enc_hidden * 2, enc_hidden]
    
    return config, sampled_params


def create_experiment_id(sampled_params: Dict, exp_num: int) -> str:
    """Create descriptive experiment ID."""
    conv = sampled_params.get('encoder.conv_type', 'gatv2')[:4]
    hidden = sampled_params.get('encoder.hidden_dim', 64)
    layers = sampled_params.get('encoder.num_gat_layers', 2)
    latent = sampled_params.get('latent.dim', 32)
    lr = sampled_params.get('training.learning_rate', 0.001)
    
    return f"exp_{exp_num:03d}_{conv}_h{hidden}_l{layers}_z{latent}_lr{lr:.0e}"


def run_training(config: Dict, exp_dir: Path, timeout: int = 3600) -> Dict:
    """Run a single training experiment."""
    # Update paths
    config['paths']['output_dir'] = str(exp_dir / 'output')
    config['paths']['checkpoints'] = str(exp_dir / 'checkpoints')
    config['paths']['tensorboard'] = str(exp_dir / 'runs')
    
    # Save config
    config_path = exp_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Run training script
    train_script = Path(__file__).parent / 'train_dmt.py'
    
    cmd = [
        sys.executable, str(train_script),
        '--config', str(config_path)
    ]
    
    log_file = exp_dir / 'train.log'
    
    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                cwd=str(Path(__file__).parent)
            )
        
        # Parse results
        results_file = exp_dir / 'output' / 'results.json'
        if results_file.exists():
            with open(results_file, 'r') as f:
                return json.load(f)
        else:
            return {'status': 'no_results'}
            
    except subprocess.TimeoutExpired:
        return {'status': 'timeout'}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def compute_ranking_score(results: Dict) -> float:
    """
    Compute composite ranking score for experiment.
    
    Lower is better. Combines:
    - Validation loss (primary)
    - Generalization gap (train-val difference)
    - Latent quality (silhouette score, higher is better)
    """
    if results.get('status') != 'success':
        return float('inf')
    
    val_loss = results.get('val_loss', float('inf'))
    train_loss = results.get('train_loss', float('inf'))
    silhouette = results.get('latent_silhouette', 0.0)
    
    # Generalization gap penalty
    gap = abs(train_loss - val_loss) / max(train_loss, 1e-6)
    
    # Composite score (lower is better)
    score = val_loss + 0.1 * gap - 0.5 * silhouette
    
    return score


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Search for DMT VAE')
    parser.add_argument('--n-experiments', type=int, default=25)
    parser.add_argument('--base-config', type=str, default='config/config_dmt_only.yaml',
                        help='Base config file (use config_dmt_delta.yaml for Delta band)')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--timeout', type=int, default=1800)  # 30 min per experiment
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    print("=" * 70)
    print("DMT-Only VAE Hyperparameter Search")
    print("=" * 70)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Load base config
    base_config_path = Path(__file__).parent / args.base_config
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent / 'hyperparam_search_dmt' / f'search_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    print(f"Running {args.n_experiments} experiments")
    print(f"Timeout per experiment: {args.timeout}s")
    
    # Save search space
    with open(output_dir / 'search_space.json', 'w') as f:
        json.dump({k: [str(v) if not isinstance(v, (int, float, str, bool)) else v 
                      for v in vals] for k, vals in SEARCH_SPACE.items()}, f, indent=2)
    
    # Run experiments
    all_results = []
    
    for i in range(args.n_experiments):
        print(f"\n{'='*70}")
        print(f"Experiment {i+1}/{args.n_experiments}")
        print("=" * 70)
        
        # Sample config
        config, sampled_params = sample_config(base_config, SEARCH_SPACE, seed=args.seed + i)
        exp_id = create_experiment_id(sampled_params, i)
        exp_dir = output_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"ID: {exp_id}")
        print("Sampled parameters:")
        for k, v in sorted(sampled_params.items()):
            print(f"  {k}: {v}")
        
        # Save sampled params
        with open(exp_dir / 'sampled_params.json', 'w') as f:
            json.dump(sampled_params, f, indent=2)
        
        # Run training
        print("\nTraining...")
        results = run_training(config, exp_dir, timeout=args.timeout)
        results['experiment_id'] = exp_id
        results['sampled_params'] = sampled_params
        results['ranking_score'] = compute_ranking_score(results)
        
        # Save results
        with open(exp_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        all_results.append(results)
        
        # Print summary
        if results.get('status') == 'success':
            print(f"\n✓ Success!")
            print(f"  Val Loss: {results.get('val_loss', 'N/A'):.4f}")
            print(f"  Train Loss: {results.get('train_loss', 'N/A'):.4f}")
            print(f"  KL Loss: {results.get('kl_loss', 'N/A'):.4f}")
            print(f"  Silhouette: {results.get('latent_silhouette', 'N/A'):.4f}")
            print(f"  Ranking Score: {results['ranking_score']:.4f}")
        else:
            print(f"\n✗ Failed: {results.get('status')}")
    
    # Save all results
    with open(output_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Rank experiments
    successful = [r for r in all_results if r.get('status') == 'success']
    successful.sort(key=lambda x: x['ranking_score'])
    
    print("\n" + "=" * 70)
    print("FINAL RANKING (Top 10)")
    print("=" * 70)
    
    for i, r in enumerate(successful[:10]):
        print(f"\n{i+1}. {r['experiment_id']}")
        print(f"   Score: {r['ranking_score']:.4f}")
        print(f"   Val Loss: {r.get('val_loss', 'N/A'):.4f}")
        print(f"   Silhouette: {r.get('latent_silhouette', 'N/A'):.4f}")
        print(f"   Key params: conv={r['sampled_params'].get('encoder.conv_type')}, "
              f"latent={r['sampled_params'].get('latent.dim')}, "
              f"layers={r['sampled_params'].get('encoder.num_gat_layers')}")
    
    # Save best config
    if successful:
        best = successful[0]
        best_config_path = output_dir / best['experiment_id'] / 'config.yaml'
        if best_config_path.exists():
            import shutil
            shutil.copy(best_config_path, output_dir / 'best_config.yaml')
        
        with open(output_dir / 'best_experiment.json', 'w') as f:
            json.dump(best, f, indent=2)
        
        print(f"\n✓ Best config saved to: {output_dir / 'best_config.yaml'}")
    
    print(f"\n{'='*70}")
    print(f"Search complete! Results in: {output_dir}")
    print(f"Successful: {len(successful)}/{args.n_experiments}")
    print("=" * 70)


if __name__ == '__main__':
    main()

