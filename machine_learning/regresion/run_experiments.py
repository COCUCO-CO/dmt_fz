#!/usr/bin/env python3
"""
Run multiple experiments with different random seeds.

Launches N experiments using graph-based input and logs comprehensive statistics.
"""

import argparse
import subprocess
import json
import yaml
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_single_experiment(config_path: str, seed: int, experiment_dir: Path, num_workers: int = 8) -> dict:
    """Run a single experiment with given seed."""
    
    # Load and modify config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update seed and paths
    config['seed'] = seed
    config['data']['random_state'] = seed
    config['paths']['output_dir'] = str(experiment_dir / 'output')
    config['paths']['checkpoints'] = str(experiment_dir / 'checkpoints')
    config['paths']['tensorboard'] = str(experiment_dir / 'runs')
    config['num_workers'] = num_workers
    
    # Create directories
    for key in ['output_dir', 'checkpoints', 'tensorboard']:
        Path(config['paths'][key]).mkdir(parents=True, exist_ok=True)
    
    # Save modified config
    temp_config = experiment_dir / 'config.yaml'
    with open(temp_config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Run training with real-time output
    logger.info(f"Running experiment with seed {seed} (workers={num_workers})...")
    
    log_file = experiment_dir / 'train.log'
    
    # Run with real-time output to console AND save to file
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            ['python', '-u', 'train.py', '--config', str(temp_config)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=Path(__file__).parent
        )
        
        # Stream output in real-time
        for line in process.stdout:
            print(line, end='', flush=True)
            f.write(line)
        
        process.wait()
    
    # Load results
    results_file = Path(config['paths']['output_dir']) / 'cv_results.json'
    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)
    else:
        logger.warning(f"No results file found for seed {seed}")
        return None


def aggregate_results(all_results: list, target_names: list) -> dict:
    """Aggregate results across all experiments."""
    
    # Filter valid results
    valid_results = [r for r in all_results if r is not None]
    
    if not valid_results:
        return {}
    
    # Aggregate CV metrics
    metrics_keys = ['mse', 'mae', 'r2', 'mean_pearson']
    aggregated = {
        'n_experiments': len(valid_results),
        'overall': {}
    }
    
    for key in metrics_keys:
        values = [r['cv_metrics'][key] for r in valid_results if key in r.get('cv_metrics', {})]
        if values:
            aggregated['overall'][key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'all_values': [float(v) for v in values]
            }
    
    # Aggregate per-target results
    aggregated['per_target'] = {}
    
    for target in target_names:
        pearson_values = []
        for r in valid_results:
            if 'per_target' in r and target in r['per_target']:
                pearson_values.append(r['per_target'][target]['mean_r'])
        
        if pearson_values:
            aggregated['per_target'][target] = {
                'mean_r': float(np.mean(pearson_values)),
                'std_r': float(np.std(pearson_values)),
                'min_r': float(np.min(pearson_values)),
                'max_r': float(np.max(pearson_values))
            }
    
    return aggregated


def print_summary(aggregated: dict, target_names: list):
    """Print comprehensive summary of experiments."""
    
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*80)
    
    logger.info(f"\nTotal experiments: {aggregated.get('n_experiments', 0)}")
    
    # Overall metrics
    logger.info("\n" + "-"*40)
    logger.info("OVERALL METRICS (across all experiments)")
    logger.info("-"*40)
    
    for metric, stats in aggregated.get('overall', {}).items():
        logger.info(f"{metric.upper():15s}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                   f"[{stats['min']:.4f}, {stats['max']:.4f}]")
    
    # Per-target results
    logger.info("\n" + "-"*40)
    logger.info("PER-TARGET PEARSON CORRELATION")
    logger.info("-"*40)
    
    # Group by category
    categories = {
        'ASC': target_names[:11],
        'NDE': target_names[11:15],
        'MEQ': target_names[15:20],
        'Post': target_names[20:23]
    }
    
    for cat_name, cat_targets in categories.items():
        logger.info(f"\n{cat_name}:")
        cat_values = []
        for target in cat_targets:
            if target in aggregated.get('per_target', {}):
                stats = aggregated['per_target'][target]
                logger.info(f"  {target:25s}: r = {stats['mean_r']:.3f} ± {stats['std_r']:.3f}")
                cat_values.append(stats['mean_r'])
        
        if cat_values:
            logger.info(f"  {'Category Mean':25s}: r = {np.mean(cat_values):.3f}")
    
    # Best and worst targets
    logger.info("\n" + "-"*40)
    logger.info("BEST AND WORST PREDICTED TARGETS")
    logger.info("-"*40)
    
    target_scores = [(t, s['mean_r']) for t, s in aggregated.get('per_target', {}).items()]
    target_scores.sort(key=lambda x: x[1], reverse=True)
    
    logger.info("\nTop 5 best predicted:")
    for target, score in target_scores[:5]:
        logger.info(f"  {target:25s}: r = {score:.3f}")
    
    logger.info("\nTop 5 worst predicted:")
    for target, score in target_scores[-5:]:
        logger.info(f"  {target:25s}: r = {score:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Run multiple experiments')
    parser.add_argument('--config', type=str, default='config/config_graphs.yaml',
                       help='Base configuration file')
    parser.add_argument('--n-experiments', type=int, default=20,
                       help='Number of experiments to run')
    parser.add_argument('--start-seed', type=int, default=0,
                       help='Starting seed')
    parser.add_argument('--output-dir', type=str, 
                       default='experiments_graphs',
                       help='Output directory for all experiments')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of workers for data loading')
    
    args = parser.parse_args()
    
    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = Path(__file__).parent / args.output_dir / f'run_{timestamp}'
    base_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info(f"RUNNING {args.n_experiments} EXPERIMENTS")
    logger.info(f"Output directory: {base_dir}")
    logger.info("="*80)
    
    # Load target names
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get target names from spectral_sources
    import pandas as pd
    target_labels_path = Path(config['paths']['spectral_dir']) / 'target_labels.txt'
    if not target_labels_path.exists():
        logger.error(f"Target labels file not found: {target_labels_path}")
        return
    target_names = pd.read_csv(target_labels_path, header=None)[0].tolist()
    if not target_names:
        logger.error("No target names found in target_labels.txt")
        return
    
    # Run experiments
    all_results = []
    
    for i in range(args.n_experiments):
        seed = args.start_seed + i
        experiment_dir = base_dir / f'exp_{i:03d}_seed_{seed}'
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\n{'#'*70}")
        logger.info(f"### EXPERIMENT {i+1}/{args.n_experiments} (seed={seed}) ###")
        logger.info(f"{'#'*70}\n")
        
        result = run_single_experiment(args.config, seed, experiment_dir, num_workers=args.workers)
        all_results.append(result)
        
        if result:
            logger.info(f"\n{'='*50}")
            logger.info(f"✓ EXPERIMENT {i+1} COMPLETED")
            logger.info(f"  MSE:     {result['cv_metrics']['mse']:.4f}")
            logger.info(f"  MAE:     {result['cv_metrics']['mae']:.4f}")
            logger.info(f"  R²:      {result['cv_metrics']['r2']:.4f}")
            logger.info(f"  Pearson: {result['cv_metrics']['mean_pearson']:.4f}")
            logger.info(f"{'='*50}")
        else:
            logger.warning(f"\n✗ EXPERIMENT {i+1} FAILED (no results)")
    
    # Aggregate and save results
    aggregated = aggregate_results(all_results, target_names)
    
    with open(base_dir / 'aggregated_results.json', 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    # Print summary
    print_summary(aggregated, target_names)
    
    # Save summary to file
    summary_file = base_dir / 'summary.txt'
    with open(summary_file, 'w') as f:
        f.write(f"Experiments: {args.n_experiments}\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"Seeds: {args.start_seed} to {args.start_seed + args.n_experiments - 1}\n\n")
        
        f.write("Overall Metrics:\n")
        for metric, stats in aggregated.get('overall', {}).items():
            f.write(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
        
        f.write("\nPer-Target Pearson r:\n")
        for target, stats in aggregated.get('per_target', {}).items():
            f.write(f"  {target}: {stats['mean_r']:.3f} ± {stats['std_r']:.3f}\n")
    
    logger.info(f"\n\nAll results saved to: {base_dir}")
    logger.info("="*80)


if __name__ == '__main__':
    main()

