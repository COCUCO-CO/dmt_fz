#!/usr/bin/env python3
"""
Hyperparameter Search for BrainStateVAE

Based on dataset analysis:
- 65,305 graphs (24 nodes, 552 edges each)
- 8 node features, 1 edge feature, 8 graph features
- 3 classes: DMT (39.6%), EC (32.4%), EO (28.0%)
- Node features have outliers (feat_6 especially)
- Classes are not easily separable in raw feature space
"""

import os
import sys
import yaml
import json
import argparse
import subprocess
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import numpy as np

# Search space based on dataset characteristics
SEARCH_SPACE = {
    # Encoder architecture
    'encoder.hidden_dim': [32, 64, 128],
    'encoder.num_gat_layers': [2, 3, 4],
    'encoder.num_attention_heads': [2, 4, 8],
    'encoder.dropout': [0.1, 0.2, 0.3],
    'encoder.attention_dropout': [0.0, 0.1, 0.2],
    
    # Latent space
    'latent.dim': [16, 32, 64, 128],
    
    # Decoder
    'decoder.reconstruct_edges': [True, False],
    
    # Loss weights - KEY for good latent space
    'loss.reconstruction.node_weight': [0.3, 0.5, 1.0],
    'loss.reconstruction.edge_weight': [0.5, 1.0, 2.0],
    'loss.kl.weight': [0.001, 0.01, 0.1],  # Final beta
    
    # KL Annealing - IMPORTANT
    'loss.kl.annealing.epochs': [20, 50, 100],
    'loss.kl.annealing.type': ['linear', 'cosine', 'cyclical'],
    
    # Free bits to prevent posterior collapse
    'loss.free_bits': [0.0, 0.1, 0.5],
    
    # Training
    'training.batch_size': [128, 256, 512],
    'training.learning_rate': [0.0001, 0.0005, 0.001],
    
    # Pooling
    'pooling.method': ['mean', 'max', 'mean+max'],
}


def sample_config(base_config: Dict[str, Any], search_space: Dict[str, List]) -> Dict[str, Any]:
    """Sample a random configuration from the search space."""
    config = yaml.safe_load(yaml.dump(base_config))  # Deep copy
    
    sampled_params = {}
    for param_path, values in search_space.items():
        value = random.choice(values)
        sampled_params[param_path] = value
        
        # Navigate to nested key and set value
        keys = param_path.split('.')
        d = config
        for key in keys[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]
        d[keys[-1]] = value
    
    # Ensure consistency
    # If reconstruct_edges is False, edge_weight doesn't matter
    if not config['decoder'].get('reconstruct_edges', False):
        config['loss']['reconstruction']['edge_weight'] = 0.0
    
    # Adjust hidden_dims in decoder based on encoder
    hidden_dim = config['encoder']['hidden_dim']
    num_heads = config['encoder']['num_attention_heads']
    encoder_out = hidden_dim * num_heads  # concat heads
    config['decoder']['hidden_dims'] = [encoder_out, hidden_dim]
    
    return config, sampled_params


def run_experiment(config: Dict[str, Any], 
                   experiment_id: str,
                   output_dir: Path,
                   subsample: float = 0.3,
                   max_epochs: int = 100) -> Dict[str, Any]:
    """Run a single training experiment."""
    
    # Create experiment directory
    exp_dir = output_dir / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Modify config for this experiment
    config['paths']['output_dir'] = str(exp_dir / 'output')
    config['paths']['checkpoints'] = str(exp_dir / 'checkpoints')
    config['paths']['tensorboard'] = str(exp_dir / 'runs')
    config['training']['num_epochs'] = max_epochs
    config['training']['early_stopping']['patience'] = 30
    config['visualization']['latent_method'] = 'umap'  # Use UMAP
    
    # Save config
    config_path = exp_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Run training
    cmd = [
        sys.executable, 'train.py',
        '--config', str(config_path),
        '--subsample', str(subsample)
    ]
    
    log_file = exp_dir / 'train.log'
    
    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=1800,  # 30 min timeout
                cwd=str(Path(__file__).parent)
            )
        
        # Parse results
        results_file = Path(config['paths']['output_dir']) / 'test_results.json'
        if results_file.exists():
            with open(results_file, 'r') as f:
                test_results = json.load(f)
            return {
                'status': 'success',
                'test_loss': test_results.get('test_loss', float('inf')),
                'test_recon': test_results.get('test_recon_loss', float('inf')),
                'test_kl': test_results.get('test_kl_loss', float('inf')),
                'best_epoch': test_results.get('best_epoch', -1)
            }
        else:
            return {'status': 'no_results', 'test_loss': float('inf')}
            
    except subprocess.TimeoutExpired:
        return {'status': 'timeout', 'test_loss': float('inf')}
    except Exception as e:
        return {'status': 'error', 'error': str(e), 'test_loss': float('inf')}


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Search for VAE')
    parser.add_argument('--n_experiments', type=int, default=20,
                       help='Number of experiments to run')
    parser.add_argument('--subsample', type=float, default=0.3,
                       help='Fraction of dataset to use (for speed)')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='Max epochs per experiment')
    parser.add_argument('--output_dir', type=str, default='hyperparam_search',
                       help='Output directory for experiments')
    parser.add_argument('--base_config', type=str, default='config/config.yaml',
                       help='Base configuration file')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"search_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("HYPERPARAMETER SEARCH FOR BrainStateVAE")
    print("=" * 70)
    print(f"Experiments: {args.n_experiments}")
    print(f"Subsample: {args.subsample * 100:.0f}%")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Output: {output_dir}")
    print("=" * 70)
    
    # Load base config
    with open(args.base_config, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Run experiments
    all_results = []
    
    for i in range(args.n_experiments):
        exp_id = f"exp_{i:03d}"
        print(f"\n[{i+1}/{args.n_experiments}] Running {exp_id}...")
        
        # Sample config
        config, sampled_params = sample_config(base_config, SEARCH_SPACE)
        
        # Print key params
        print(f"  hidden_dim={sampled_params.get('encoder.hidden_dim')}, "
              f"layers={sampled_params.get('encoder.num_gat_layers')}, "
              f"heads={sampled_params.get('encoder.num_attention_heads')}, "
              f"latent={sampled_params.get('latent.dim')}")
        print(f"  kl_weight={sampled_params.get('loss.kl.weight')}, "
              f"anneal_epochs={sampled_params.get('loss.kl.annealing.epochs')}, "
              f"type={sampled_params.get('loss.kl.annealing.type')}")
        
        # Run experiment
        results = run_experiment(
            config, exp_id, output_dir,
            subsample=args.subsample,
            max_epochs=args.max_epochs
        )
        
        results['experiment_id'] = exp_id
        results['params'] = sampled_params
        all_results.append(results)
        
        # Print result
        if results['status'] == 'success':
            print(f"  ✓ Loss: {results['test_loss']:.4f} "
                  f"(Recon: {results['test_recon']:.4f}, KL: {results['test_kl']:.4f})")
        else:
            print(f"  ✗ {results['status']}")
        
        # Save intermediate results
        with open(output_dir / 'results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
    
    # Find best experiments
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    successful = [r for r in all_results if r['status'] == 'success']
    
    if successful:
        # Sort by test loss
        successful.sort(key=lambda x: x['test_loss'])
        
        print("\nTop 5 experiments:")
        for i, r in enumerate(successful[:5]):
            print(f"\n{i+1}. {r['experiment_id']}")
            print(f"   Test Loss: {r['test_loss']:.4f} (Recon: {r['test_recon']:.4f}, KL: {r['test_kl']:.4f})")
            print(f"   Params: hidden={r['params'].get('encoder.hidden_dim')}, "
                  f"layers={r['params'].get('encoder.num_gat_layers')}, "
                  f"heads={r['params'].get('encoder.num_attention_heads')}, "
                  f"latent={r['params'].get('latent.dim')}")
            print(f"   KL: weight={r['params'].get('loss.kl.weight')}, "
                  f"anneal={r['params'].get('loss.kl.annealing.epochs')} epochs ({r['params'].get('loss.kl.annealing.type')})")
        
        # Save best config
        best = successful[0]
        best_config_src = output_dir / best['experiment_id'] / 'config.yaml'
        best_config_dst = output_dir / 'best_config.yaml'
        if best_config_src.exists():
            import shutil
            shutil.copy(best_config_src, best_config_dst)
            print(f"\nBest config saved to: {best_config_dst}")
    else:
        print("No successful experiments!")
    
    print(f"\nAll results saved to: {output_dir / 'results.json'}")
    print("=" * 70)


if __name__ == '__main__':
    main()
