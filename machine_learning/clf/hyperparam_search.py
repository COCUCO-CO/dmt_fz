#!/usr/bin/env python3
"""
Hyperparameter Random Search for BrainStateGAT.

This script runs multiple experiments with different hyperparameter configurations,
logging each experiment separately to TensorBoard for easy comparison.

Usage:
    python hyperparam_search.py --n_experiments 20 --bands Alpha
    python hyperparam_search.py --n_experiments 10 --bands Alpha Beta --max_epochs 100
"""

import argparse
import json
import yaml
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
import subprocess
import copy
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# HYPERPARAMETER SEARCH SPACE
# =============================================================================

# =============================================================================
# SEARCH SPACE CONFIGURATIONS
# Choose one: SEARCH_SPACE_FULL (exhaustive) or SEARCH_SPACE_FOCUSED (faster)
# =============================================================================

# FULL SEARCH SPACE - Explores many parameters (recommended: 50+ experiments)
SEARCH_SPACE_FULL = {
    # =========================================================================
    # MODEL ARCHITECTURE
    # =========================================================================
    'model.architecture.hidden_dim': {
        'type': 'choice',
        'values': [64, 128, 256, 512]
    },
    'model.architecture.num_gat_layers': {
        'type': 'choice',
        'values': [2, 3]  # Extended range
    },
    'model.architecture.num_attention_heads': {
        'type': 'choice',
        'values': [4, 8, 12]  # More options
    },
    'model.architecture.dropout': {
        'type': 'uniform',
        'low': 0.1,
        'high': 0.5
    },
    'model.architecture.attention_dropout': {
        'type': 'uniform',
        'low': 0.0,
        'high': 0.3
    },
    'model.architecture.concat_heads': {
        'type': 'choice',
        'values': [True, False]
    },
    'model.architecture.use_skip_connections': {
        'type': 'choice',
        'values': [True, False]
    },
    'model.architecture.negative_slope': {
        'type': 'choice',
        'values': [0.2]  # LeakyReLU slope
    },
    
    # =========================================================================
    # POOLING
    # =========================================================================
    'model.pooling.method': {
        'type': 'choice',
        'values': ['mean', 'max', 'add', 'attention', 'mean+max']
    },
    
    # =========================================================================
    # MLP CLASSIFIER
    # =========================================================================
    'model.mlp.hidden_dims': {
        'type': 'choice',
        'values': [
            [64],           # Shallow
            [128],          
            [256],          
            [128, 64],      # 2 layers
            [256, 128]
        ]
    },
    'model.mlp.dropout': {
        'type': 'uniform',
        'low': 0.2,
        'high': 0.6
    },
    'model.mlp.activation': {
        'type': 'choice',
        'values': ['relu']
    },
    
    # =========================================================================
    # OPTIMIZER
    # =========================================================================
    'training.optimizer': {
        'type': 'choice',
        'values': ['adam', 'adamw']
    },
    'training.learning_rate': {
        'type': 'loguniform',
        'low': 1e-5,
        'high': 1e-2
    },
    'training.weight_decay': {
        'type': 'loguniform',
        'low': 1e-5,
        'high': 1e-3
    },
    'training.batch_size': {
        'type': 'choice',
        'values': [64, 128]
    },
    
    # =========================================================================
    # SCHEDULER
    # =========================================================================
    'training.scheduler.type': {
        'type': 'choice',
        'values': ['reduce_on_plateau', 'cosine', 'step', 'none']
    },
    'training.scheduler.patience': {
        'type': 'choice',
        'values': [5, 10, 15, 20, 30]
    },
    'training.scheduler.factor': {
        'type': 'choice',
        'values': [0.1, 0.3, 0.5, 0.7]
    },
    
    # =========================================================================
    # REGULARIZATION
    # =========================================================================
    'training.label_smoothing': {
        'type': 'choice',
        'values': [0.0, 0.05, 0.1, 0.15]
    },
    'training.gradient_clip': {
        'type': 'choice',
        'values': [0.0, 0.5, 1.0, 2.0]  # 0 = no clipping
    },
    
    # =========================================================================
    # GRAPH CONSTRUCTION
    # =========================================================================
    'data.graph.fully_connected': {
        'type': 'choice',
        'values': [True]
    },
    'data.graph.edge_threshold': {
        'type': 'uniform',
        'low': 0.05,
        'high': 0.5
    },
}

# FOCUSED SEARCH SPACE - Most impactful parameters (recommended: 20-30 experiments)
SEARCH_SPACE_FOCUSED = {
    # Architecture (most impactful)
    'model.architecture.hidden_dim': {
        'type': 'choice',
        'values': [128, 256, 512]
    },
    'model.architecture.num_gat_layers': {
        'type': 'choice',
        'values': [1, 2, 3]
    },
    'model.architecture.num_attention_heads': {
        'type': 'choice',
        'values': [4, 8]
    },
    'model.architecture.dropout': {
        'type': 'uniform',
        'low': 0.2,
        'high': 0.5
    },
    'model.architecture.use_skip_connections': {
        'type': 'choice',
        'values': [True, False]
    },
    
    # Pooling
    'model.pooling.method': {
        'type': 'choice',
        'values': ['mean', 'attention', 'mean+max']
    },
    
    # MLP
    'model.mlp.hidden_dims': {
        'type': 'choice',
        'values': [[256, 128], [128, 64], [512, 256, 128]]
    },
    'model.mlp.dropout': {
        'type': 'uniform',
        'low': 0.3,
        'high': 0.6
    },
    
    # Optimizer
    'training.optimizer': {
        'type': 'choice',
        'values': ['adam', 'adamw']
    },
    'training.learning_rate': {
        'type': 'loguniform',
        'low': 5e-5,
        'high': 1e-2
    },
    'training.weight_decay': {
        'type': 'loguniform',
        'low': 1e-5,
        'high': 1e-3
    },
    'training.batch_size': {
        'type': 'choice',
        'values': [64, 128]
    },
    
    # Scheduler
    'training.scheduler.type': {
        'type': 'choice',
        'values': ['reduce_on_plateau', 'cosine']
    },
    
    # Regularization
    'training.label_smoothing': {
        'type': 'choice',
        'values': [0.0, 0.1]
    },
}

# SELECT WHICH SEARCH SPACE TO USE
# Change this to switch between full and focused search
SEARCH_SPACE = SEARCH_SPACE_FOCUSED  # or SEARCH_SPACE_FULL


def sample_hyperparameter(param_config: Dict[str, Any]) -> Any:
    """Sample a single hyperparameter value."""
    param_type = param_config['type']
    
    if param_type == 'choice':
        return random.choice(param_config['values'])
    elif param_type == 'uniform':
        return float(random.uniform(param_config['low'], param_config['high']))
    elif param_type == 'loguniform':
        log_low = float(np.log(param_config['low']))
        log_high = float(np.log(param_config['high']))
        return float(np.exp(random.uniform(log_low, log_high)))
    elif param_type == 'int_uniform':
        return int(random.randint(param_config['low'], param_config['high']))
    else:
        raise ValueError(f"Unknown parameter type: {param_type}")


def convert_numpy_to_native(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for YAML serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_native(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def set_nested_value(config: Dict, key_path: str, value: Any):
    """Set a nested value in a config dict using dot notation."""
    keys = key_path.split('.')
    d = config
    for key in keys[:-1]:
        d = d[key]
    d[keys[-1]] = value


def get_nested_value(config: Dict, key_path: str) -> Any:
    """Get a nested value from a config dict using dot notation."""
    keys = key_path.split('.')
    d = config
    for key in keys:
        d = d[key]
    return d


def sample_config(base_config: Dict[str, Any], search_space: Dict = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Sample a random configuration.
    
    Returns:
        Tuple of (full_config, sampled_params_only)
    """
    if search_space is None:
        search_space = SEARCH_SPACE
    
    config = copy.deepcopy(base_config)
    sampled_params = {}
    
    for param_key, param_config in search_space.items():
        value = sample_hyperparameter(param_config)
        
        # Round floats for cleaner logging
        if isinstance(value, float):
            value = round(value, 6)
        
        set_nested_value(config, param_key, value)
        sampled_params[param_key] = value
    
    return config, sampled_params


def create_experiment_name(params: Dict[str, Any], exp_idx: int) -> str:
    """Create a descriptive experiment name."""
    # Get key params for the name
    hidden = params.get('model.architecture.hidden_dim', '?')
    layers = params.get('model.architecture.num_gat_layers', '?')
    heads = params.get('model.architecture.num_attention_heads', '?')
    lr = params.get('training.learning_rate', 0)
    
    lr_str = f"{lr:.0e}" if lr else "?"
    
    return f"exp{exp_idx:03d}_h{hidden}_l{layers}_heads{heads}_lr{lr_str}"


def run_single_experiment(config: Dict[str, Any], 
                          exp_name: str,
                          exp_dir: Path,
                          max_epochs: int = None) -> Dict[str, Any]:
    """
    Run a single training experiment.
    
    Returns:
        Dict with results or None if failed
    """
    # Override paths for this experiment
    config['paths']['output_dir'] = str(exp_dir / 'output')
    config['paths']['checkpoints'] = str(exp_dir / 'checkpoints')
    config['paths']['tensorboard'] = str(exp_dir / 'tensorboard')
    
    if max_epochs:
        config['training']['num_epochs'] = max_epochs
    
    # Save config (convert numpy to native Python types first)
    config_path = exp_dir / 'config.yaml'
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    config_native = convert_numpy_to_native(config)
    with open(config_path, 'w') as f:
        yaml.dump(config_native, f, default_flow_style=False)
    
    # Run training (output shown in real-time)
    logger.info(f"Starting experiment: {exp_name}")
    logger.info("-" * 40)
    
    try:
        # Use Popen to stream output in real-time
        process = subprocess.Popen(
            ['python', '-u', 'train.py', '--config', str(config_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream output line by line
        stderr_lines = []
        for line in process.stdout:
            print(line, end='', flush=True)
            stderr_lines.append(line)
        
        process.wait(timeout=7200)  # 2 hour timeout
        
        logger.info("-" * 40)
        
        if process.returncode != 0:
            logger.error(f"Experiment {exp_name} failed (exit code: {process.returncode})")
            error_text = ''.join(stderr_lines[-20:])
            return {'status': 'failed', 'error': error_text[-500:]}
        
        # Load results
        results_path = exp_dir / 'output' / 'test_results.json'
        if results_path.exists():
            with open(results_path, 'r') as f:
                results = json.load(f)
            results['status'] = 'success'
            return results
        else:
            return {'status': 'no_results'}
            
    except (subprocess.TimeoutExpired, TimeoutError):
        logger.error(f"Experiment {exp_name} timed out")
        try:
            process.kill()
        except:
            pass
        return {'status': 'timeout'}
    except Exception as e:
        logger.error(f"Experiment {exp_name} error: {e}")
        return {'status': 'error', 'error': str(e)}


def run_search(base_config_path: str,
               n_experiments: int,
               output_dir: str,
               bands: List[str] = None,
               conditions: List[str] = None,
               max_epochs: int = None,
               seed: int = 42):
    """
    Run hyperparameter random search.
    
    Args:
        base_config_path: Path to base config file
        n_experiments: Number of experiments to run
        output_dir: Directory for all experiment outputs
        bands: List of frequency bands to train
        conditions: List of conditions to classify (e.g., ['DMT', 'EC'] for 2-class)
        max_epochs: Override max epochs (for faster search)
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Load base config
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Filter bands if specified
    if bands:
        base_config['data']['bands'] = bands
    
    # Filter conditions if specified (e.g., DMT vs EC only)
    if conditions:
        base_config['data']['conditions'] = conditions
        logger.info(f"Using conditions: {conditions} ({len(conditions)}-class classification)")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    search_name = f"search_{timestamp}"
    search_dir = output_path / search_name
    search_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info(f"HYPERPARAMETER RANDOM SEARCH")
    logger.info("=" * 80)
    logger.info(f"Number of experiments: {n_experiments}")
    logger.info(f"Bands: {bands or base_config['data']['bands']}")
    logger.info(f"Conditions: {conditions or base_config['data']['conditions']}")
    logger.info(f"Max epochs: {max_epochs or base_config['training']['num_epochs']}")
    logger.info(f"Output directory: {search_dir}")
    logger.info("=" * 80)
    
    # Track all experiments
    all_experiments = []
    
    for exp_idx in range(n_experiments):
        logger.info(f"\n{'='*40}")
        logger.info(f"EXPERIMENT {exp_idx + 1}/{n_experiments}")
        logger.info(f"{'='*40}")
        
        # Sample configuration
        config, params = sample_config(base_config)
        exp_name = create_experiment_name(params, exp_idx)
        exp_dir = search_dir / exp_name
        
        # Log sampled parameters
        logger.info("Sampled parameters:")
        for key, value in params.items():
            logger.info(f"  {key}: {value}")
        
        # Run experiment
        results = run_single_experiment(config, exp_name, exp_dir, max_epochs)
        
        # Store results (convert to native types for JSON serialization)
        experiment_record = {
            'name': exp_name,
            'index': exp_idx,
            'params': convert_numpy_to_native(params),
            'results': convert_numpy_to_native(results)
        }
        all_experiments.append(experiment_record)
        
        # Log results
        if results['status'] == 'success':
            logger.info(f"✓ Success! Test accuracy: {results.get('test_accuracy', 0)*100:.2f}%")
        else:
            logger.info(f"✗ Failed: {results['status']}")
        
        # Save intermediate results
        with open(search_dir / 'all_experiments.json', 'w') as f:
            json.dump(all_experiments, f, indent=2)
    
    # Generate summary
    generate_summary(all_experiments, search_dir)
    
    logger.info("\n" + "=" * 80)
    logger.info("SEARCH COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {search_dir}")
    logger.info(f"TensorBoard: tensorboard --logdir={search_dir}")


def generate_summary(experiments: List[Dict], output_dir: Path):
    """Generate summary of all experiments."""
    
    # Filter successful experiments
    successful = [e for e in experiments if e['results'].get('status') == 'success']
    
    if not successful:
        logger.warning("No successful experiments!")
        return
    
    # Sort by accuracy
    successful.sort(key=lambda e: e['results'].get('test_accuracy', 0), reverse=True)
    
    # Generate summary
    summary = {
        'total_experiments': len(experiments),
        'successful_experiments': len(successful),
        'failed_experiments': len(experiments) - len(successful),
        'best_experiment': successful[0] if successful else None,
        'top_5': successful[:5],
        'all_accuracies': [
            {
                'name': e['name'],
                'accuracy': e['results'].get('test_accuracy', 0),
                'f1': e['results'].get('test_f1', 0)
            }
            for e in successful
        ]
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total experiments: {len(experiments)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(experiments) - len(successful)}")
    
    if successful:
        logger.info("\nTop 5 experiments:")
        for i, exp in enumerate(successful[:5]):
            acc = exp['results'].get('test_accuracy', 0) * 100
            f1 = exp['results'].get('test_f1', 0)
            logger.info(f"  {i+1}. {exp['name']}: {acc:.2f}% (F1={f1:.4f})")
        
        logger.info("\nBest hyperparameters:")
        best = successful[0]
        for key, value in best['params'].items():
            logger.info(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Random Search')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Base configuration file')
    parser.add_argument('--n_experiments', type=int, default=20,
                       help='Number of experiments to run')
    parser.add_argument('--output_dir', type=str, default='hyperparam_search',
                       help='Output directory for search results')
    parser.add_argument('--bands', type=str, nargs='+', default=None,
                       help='Frequency bands to train (default: all)')
    parser.add_argument('--conditions', type=str, nargs='+', default=None,
                       help='Conditions to classify (e.g., --conditions DMT EC for 2-class)')
    parser.add_argument('--max_epochs', type=int, default=None,
                       help='Override max epochs (for faster search)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    run_search(
        base_config_path=args.config,
        n_experiments=args.n_experiments,
        output_dir=args.output_dir,
        bands=args.bands,
        conditions=args.conditions,
        max_epochs=args.max_epochs,
        seed=args.seed
    )


if __name__ == '__main__':
    main()

