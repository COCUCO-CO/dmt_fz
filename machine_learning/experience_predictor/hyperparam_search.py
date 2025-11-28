#!/usr/bin/env python3
"""
Hyperparameter search using Random Search for experience prediction.

Searches over the most important hyperparameters and logs all results.
"""

import argparse
import subprocess
import json
import yaml
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# HYPERPARAMETER SEARCH SPACE
# =============================================================================

SEARCH_SPACE = {
    # Input type
    'input_type': ['spectral', 'graphs'],
    
    # Model type for spectral data (tabular)
    # - MLP: Neural network
    # - XGBoost, LightGBM: State-of-the-art for tabular, often best for small N
    'spectral_model_type': ['mlp', 'xgboost'],
    
    # GNN conv type - GCN and SAGE are simpler, better for small N
    'gnn_conv_type': ['gatv2', 'cheby'],
    
    # Architecture - SMALLER models for N=24-29
    'hidden_dims': [
        [8],           # Ultra tiny
        [16],          # Very small
        [32],          # Small
        [16, 8],       # Tiny 2-layer
        [32, 16],      # Small 2-layer
        [64, 32],      # Medium (max reasonable for N=24)
    ],
    'gnn_hidden_dim': [8, 16, 32],  # Smaller for small N
    'gnn_num_layers': [1, 2],
    'gnn_num_heads': [1, 2],  # Removed 4, too much for N=24
    'gnn_pooling': ['mean', 'max', 'mean+max'],
    
    # Regularization - STRONGER for small N
    'dropout': [0.4, 0.5, 0.6, 0.7],  # Higher dropout
    'weight_decay': [1e-3, 1e-2, 0.05, 0.1, 0.2],  # Stronger regularization
    
    # Training (for neural networks)
    'learning_rate': [5e-5, 1e-4, 5e-4, 1e-3],  # Added lower LR
    'batch_size': [4, 8, 12],  # Smaller batches for small N
    'loss': ['mse', 'huber', 'l1'],
    
    # Scheduler - more patience since small val set is noisy
    'scheduler_patience': [20, 30, 50],
    'early_stopping_patience': [60, 100, 150],  # More patience
    
    # Activation
    'activation': ['relu', 'elu', 'gelu'],
    
    # ==== Boosting hyperparameters (XGBoost, LightGBM) ====
    'boosting_n_estimators': [50, 100, 200, 300],
    'boosting_max_depth': [2, 3, 4, 5],  # Shallow trees for small N
    'boosting_learning_rate': [0.01, 0.05, 0.1, 0.2],
    'boosting_subsample': [0.6, 0.7, 0.8, 0.9],
    'boosting_colsample': [0.6, 0.7, 0.8, 0.9],
    'boosting_reg_alpha': [0, 0.1, 0.5, 1.0],  # L1 regularization
    'boosting_reg_lambda': [0.5, 1.0, 2.0, 5.0],  # L2 regularization
}


def sample_hyperparams(search_space: Dict, input_type: str = None, model_type: str = None) -> Dict[str, Any]:
    """Sample random hyperparameters from search space."""
    params = {}
    
    # Model type first (affects input_type choice)
    if model_type:
        params['model_type'] = model_type
        # Boosting models require spectral (tabular) data
        if model_type in ['xgboost', 'lightgbm']:
            params['input_type'] = 'spectral'
        elif model_type == 'gnn':
            params['input_type'] = 'graphs'
        elif model_type == 'mlp':
            params['input_type'] = input_type if input_type else 'spectral'
        else:
            params['input_type'] = input_type if input_type else random.choice(search_space['input_type'])
    else:
        # Input type
        if input_type:
            params['input_type'] = input_type
        else:
            params['input_type'] = random.choice(search_space['input_type'])
        
        # Model type - LOGICAL pairing:
        # - Spectral data (tabular) → MLP, XGBoost, or LightGBM
        # - Graph data (structured) → GNN
        if params['input_type'] == 'spectral':
            params['model_type'] = random.choice(search_space['spectral_model_type'])
        else:
            params['model_type'] = 'gnn'  # Graphs = structured → GNN
    
    # Architecture for MLP
    params['hidden_dims'] = random.choice(search_space['hidden_dims'])
    params['dropout'] = random.choice(search_space['dropout'])
    params['activation'] = random.choice(search_space['activation'])
    
    # GNN specific (only for graphs input)
    if params['model_type'] == 'gnn':
        params['gnn_conv_type'] = random.choice(search_space['gnn_conv_type'])
        params['gnn_hidden_dim'] = random.choice(search_space['gnn_hidden_dim'])
        params['gnn_num_layers'] = random.choice(search_space['gnn_num_layers'])
        params['gnn_num_heads'] = random.choice(search_space['gnn_num_heads'])
        params['gnn_pooling'] = random.choice(search_space['gnn_pooling'])
    
    # Boosting specific (XGBoost, LightGBM)
    if params['model_type'] in ['xgboost', 'lightgbm']:
        params['boosting_n_estimators'] = random.choice(search_space['boosting_n_estimators'])
        params['boosting_max_depth'] = random.choice(search_space['boosting_max_depth'])
        params['boosting_learning_rate'] = random.choice(search_space['boosting_learning_rate'])
        params['boosting_subsample'] = random.choice(search_space['boosting_subsample'])
        params['boosting_colsample'] = random.choice(search_space['boosting_colsample'])
        params['boosting_reg_alpha'] = random.choice(search_space['boosting_reg_alpha'])
        params['boosting_reg_lambda'] = random.choice(search_space['boosting_reg_lambda'])
    
    # Regularization (for neural networks)
    params['weight_decay'] = random.choice(search_space['weight_decay'])
    
    # Training (for neural networks)
    params['learning_rate'] = random.choice(search_space['learning_rate'])
    params['batch_size'] = random.choice(search_space['batch_size'])
    params['loss'] = random.choice(search_space['loss'])
    
    # Scheduler
    params['scheduler_patience'] = random.choice(search_space['scheduler_patience'])
    params['early_stopping_patience'] = random.choice(search_space['early_stopping_patience'])
    
    return params


def create_config_from_params(base_config: Dict, params: Dict, output_dir: Path) -> Dict:
    """Create full config from sampled parameters."""
    import copy
    config = copy.deepcopy(base_config)
    
    # Input type
    config['input_type'] = params['input_type']
    
    # Paths
    config['paths']['output_dir'] = str(output_dir / 'output')
    config['paths']['checkpoints'] = str(output_dir / 'checkpoints')
    config['paths']['tensorboard'] = str(output_dir / 'runs')
    
    # Model
    config['model']['type'] = params['model_type']
    config['model']['mlp']['hidden_dims'] = params['hidden_dims']
    config['model']['mlp']['dropout'] = params['dropout']
    config['model']['mlp']['activation'] = params['activation']
    
    if params['model_type'] == 'gnn':
        config['model']['gnn']['conv_type'] = params['gnn_conv_type']
        config['model']['gnn']['hidden_dim'] = params['gnn_hidden_dim']
        config['model']['gnn']['num_layers'] = params['gnn_num_layers']
        config['model']['gnn']['num_heads'] = params['gnn_num_heads']
        config['model']['gnn']['pooling'] = params['gnn_pooling']
        config['model']['gnn']['dropout'] = params['dropout']
    
    # Boosting config (XGBoost, LightGBM)
    if params['model_type'] in ['xgboost', 'lightgbm']:
        if 'boosting' not in config:
            config['boosting'] = {}
        config['boosting']['type'] = params['model_type']
        config['boosting']['n_estimators'] = params['boosting_n_estimators']
        config['boosting']['max_depth'] = params['boosting_max_depth']
        config['boosting']['learning_rate'] = params['boosting_learning_rate']
        config['boosting']['subsample'] = params['boosting_subsample']
        config['boosting']['colsample_bytree'] = params['boosting_colsample']
        config['boosting']['reg_alpha'] = params['boosting_reg_alpha']
        config['boosting']['reg_lambda'] = params['boosting_reg_lambda']
    
    # Training (for neural networks)
    config['training']['learning_rate'] = params['learning_rate']
    config['training']['weight_decay'] = params['weight_decay']
    config['training']['batch_size'] = params['batch_size']
    config['training']['loss'] = params['loss']
    config['training']['scheduler']['patience'] = params['scheduler_patience']
    config['training']['early_stopping']['patience'] = params['early_stopping_patience']
    
    return config


def run_single_trial(config: Dict, trial_dir: Path, trial_num: int) -> Dict:
    """Run a single hyperparameter trial."""
    
    # Create directories
    for key in ['output_dir', 'checkpoints', 'tensorboard']:
        Path(config['paths'][key]).mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = trial_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Run training
    logger.info(f"Running trial {trial_num}...")
    
    log_file = trial_dir / 'train.log'
    
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            ['python', '-u', 'train.py', '--config', str(config_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=Path(__file__).parent
        )
        
        for line in process.stdout:
            print(line, end='', flush=True)
            f.write(line)
        
        process.wait()
    
    # Load results
    results_file = Path(config['paths']['output_dir']) / 'cv_results.json'
    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Search')
    parser.add_argument('--base-config', type=str, default='config/config_spectral_simple.yaml',
                       help='Base configuration file')
    parser.add_argument('--n-trials', type=int, default=30,
                       help='Number of random trials')
    parser.add_argument('--input-type', type=str, default=None,
                       choices=['spectral', 'graphs'],
                       help='Fix input type (default: search both)')
    parser.add_argument('--model-type', type=str, default=None,
                       choices=['mlp', 'xgboost', 'lightgbm', 'gnn'],
                       help='Fix model type (default: search all compatible)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output-dir', type=str, default='hyperparam_search',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Validate combinations
    if args.model_type in ['xgboost', 'lightgbm'] and args.input_type == 'graphs':
        logger.error("ERROR: Boosting models (xgboost, lightgbm) require spectral (tabular) data!")
        logger.error("       Use --input-type spectral or remove --input-type flag")
        return
    if args.model_type == 'gnn' and args.input_type == 'spectral':
        logger.warning("WARNING: GNN with spectral data will build graphs from AAL regions")
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = Path(__file__).parent / args.output_dir / f'search_{timestamp}'
    base_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info(f"HYPERPARAMETER SEARCH - {args.n_trials} TRIALS")
    logger.info(f"Output: {base_dir}")
    logger.info("="*70)
    
    # Load base config
    with open(args.base_config, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Run trials
    all_results = []
    
    for trial_num in range(args.n_trials):
        # Sample hyperparameters
        params = sample_hyperparams(SEARCH_SPACE, args.input_type, args.model_type)
        
        # Create descriptive trial name
        input_short = 'spec' if params['input_type'] == 'spectral' else 'graph'
        model_short = params['model_type'].upper()
        if params['model_type'] == 'gnn':
            model_short = params.get('gnn_conv_type', 'gnn').upper()
        
        trial_name = f"trial_{trial_num:03d}_{input_short}_{model_short}"
        
        logger.info(f"\n{'#'*70}")
        logger.info(f"### TRIAL {trial_num + 1}/{args.n_trials}: {input_short.upper()} + {model_short}")
        logger.info(f"{'#'*70}")
        
        logger.info("\nSampled hyperparameters:")
        for k, v in params.items():
            logger.info(f"  {k}: {v}")
        
        # Create trial directory with descriptive name
        trial_dir = base_dir / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)
        
        # Save params
        with open(trial_dir / 'params.json', 'w') as f:
            json.dump(params, f, indent=2)
        
        # Create config
        config = create_config_from_params(base_config, params, trial_dir)
        config['seed'] = args.seed + trial_num
        config['data']['random_state'] = args.seed + trial_num
        
        # Run trial
        result = run_single_trial(config, trial_dir, trial_num + 1)
        
        if result:
            trial_result = {
                'trial': trial_num,
                'params': params,
                'mean_pearson': result['cv_metrics']['mean_pearson'],
                'std_pearson': result['cv_std']['mean_pearson'],
                'mse': result['cv_metrics']['mse'],
                'r2': result['cv_metrics']['r2'],
            }
            all_results.append(trial_result)
            
            logger.info(f"\n✓ Trial {trial_num + 1} completed:")
            logger.info(f"  Mean Pearson: {trial_result['mean_pearson']:.4f} ± {trial_result['std_pearson']:.4f}")
            logger.info(f"  MSE: {trial_result['mse']:.4f}")
            logger.info(f"  R²: {trial_result['r2']:.4f}")
        else:
            logger.warning(f"\n✗ Trial {trial_num + 1} failed")
            all_results.append({'trial': trial_num, 'params': params, 'mean_pearson': -999})
    
    # Sort by mean_pearson
    all_results.sort(key=lambda x: x.get('mean_pearson', -999), reverse=True)
    
    # Save all results
    with open(base_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("HYPERPARAMETER SEARCH RESULTS")
    logger.info("="*70)
    
    logger.info("\nTop 10 Configurations:")
    logger.info("-"*70)
    
    for i, result in enumerate(all_results[:10]):
        if result.get('mean_pearson', -999) > -900:
            logger.info(f"\n#{i+1}: Mean Pearson = {result['mean_pearson']:.4f} ± {result.get('std_pearson', 0):.4f}")
            logger.info(f"     MSE = {result.get('mse', 0):.2f}, R² = {result.get('r2', 0):.4f}")
            logger.info(f"     Params:")
            for k, v in result['params'].items():
                logger.info(f"       {k}: {v}")
    
    # Save best config
    if all_results and all_results[0].get('mean_pearson', -999) > -900:
        best_params = all_results[0]['params']
        best_config = create_config_from_params(base_config, best_params, base_dir / 'best')
        
        with open(base_dir / 'best_config.yaml', 'w') as f:
            yaml.dump(best_config, f, default_flow_style=False)
        
        with open(base_dir / 'best_params.json', 'w') as f:
            json.dump(best_params, f, indent=2)
        
        logger.info(f"\n\nBest configuration saved to: {base_dir / 'best_config.yaml'}")
    
    # Summary stats
    valid_results = [r for r in all_results if r.get('mean_pearson', -999) > -900]
    if valid_results:
        pearsons = [r['mean_pearson'] for r in valid_results]
        logger.info(f"\n\nSearch Statistics:")
        logger.info(f"  Completed trials: {len(valid_results)}/{args.n_trials}")
        logger.info(f"  Best Mean Pearson: {max(pearsons):.4f}")
        logger.info(f"  Worst Mean Pearson: {min(pearsons):.4f}")
        logger.info(f"  Median Mean Pearson: {np.median(pearsons):.4f}")
    
    logger.info(f"\n\nAll results saved to: {base_dir}")
    logger.info("="*70)


if __name__ == '__main__':
    main()

