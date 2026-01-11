"""
Logging utilities for TensorBoard and file logging.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import json

logger = logging.getLogger(__name__)

# TensorBoard import with fallback
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logger.warning("TensorBoard not available")


def setup_tensorboard(log_dir: Path) -> Optional[Any]:
    """
    Setup TensorBoard writer.
    
    Args:
        log_dir: Directory for TensorBoard logs
        
    Returns:
        SummaryWriter or None if not available
    """
    if not TENSORBOARD_AVAILABLE:
        return None
    
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(log_dir)
    logger.info(f"TensorBoard logging to: {log_dir}")
    
    return writer


def log_epoch_metrics(writer: Optional[Any],
                     epoch: int,
                     train_loss: float,
                     val_loss: float,
                     val_metrics: Dict[str, float],
                     lr: float,
                     fold: int = 0) -> None:
    """
    Log metrics for a single epoch to TensorBoard.
    """
    if writer is None:
        return
    
    prefix = f"fold_{fold}/" if fold > 0 else ""
    
    # Loss
    writer.add_scalar(f'{prefix}Loss/train', train_loss, epoch)
    writer.add_scalar(f'{prefix}Loss/val', val_loss, epoch)
    
    # Metrics
    for name, value in val_metrics.items():
        if isinstance(value, (int, float)) and not np.isnan(value):
            writer.add_scalar(f'{prefix}Metrics/{name}', value, epoch)
    
    # Learning rate
    writer.add_scalar(f'{prefix}LearningRate', lr, epoch)


def log_per_target_metrics(writer: Optional[Any],
                          epoch: int,
                          per_target: Dict[str, Dict[str, float]],
                          fold: int = 0) -> None:
    """
    Log per-target metrics to TensorBoard.
    """
    if writer is None:
        return
    
    prefix = f"fold_{fold}/" if fold > 0 else ""
    
    for target_name, metrics in per_target.items():
        # Clean target name for TensorBoard (remove special chars)
        clean_name = target_name.replace(' ', '_').replace('/', '_')
        
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                writer.add_scalar(f'{prefix}PerTarget/{clean_name}/{metric_name}', value, epoch)


def log_final_results(writer: Optional[Any],
                     results: Dict[str, Any],
                     fold: int = 0) -> None:
    """
    Log final results summary to TensorBoard.
    """
    if writer is None:
        return
    
    prefix = f"fold_{fold}/" if fold > 0 else ""
    
    # Overall metrics
    for name, value in results.items():
        if isinstance(value, (int, float)) and not np.isnan(value):
            writer.add_scalar(f'{prefix}Final/{name}', value, 0)
    
    # Per-target correlations as histogram
    if 'per_target' in results:
        correlations = [v.get('pearson_r', 0) for v in results['per_target'].values()]
        if correlations:
            writer.add_histogram(f'{prefix}Final/correlations_distribution', 
                               np.array(correlations), 0)


def log_model_graph(writer: Optional[Any], 
                   model: Any, 
                   sample_input: Any) -> None:
    """
    Log model graph to TensorBoard.
    """
    if writer is None:
        return
    
    try:
        writer.add_graph(model, sample_input)
    except Exception as e:
        logger.warning(f"Could not log model graph: {e}")


def save_results_json(results: Dict[str, Any],
                     output_path: Path) -> None:
    """
    Save results to JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    results_clean = convert(results)
    
    with open(output_path, 'w') as f:
        json.dump(results_clean, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")


def create_experiment_summary(all_fold_results: List[Dict],
                             target_names: List[str],
                             output_dir: Path) -> Dict[str, Any]:
    """
    Create comprehensive experiment summary from all folds.
    """
    summary = {
        'n_folds': len(all_fold_results),
        'overall': {},
        'per_fold': [],
        'per_target': {},
        'best_fold': None,
        'best_targets': [],
        'worst_targets': []
    }
    
    # Aggregate metrics
    metrics_keys = ['mse', 'mae', 'r2', 'mean_pearson']
    for key in metrics_keys:
        values = [r['metrics'].get(key, 0) for r in all_fold_results if r]
        if values:
            summary['overall'][key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
    
    # Per-fold summary
    for i, r in enumerate(all_fold_results):
        if r:
            summary['per_fold'].append({
                'fold': i,
                'mse': r['metrics'].get('mse', 0),
                'mean_pearson': r['metrics'].get('mean_pearson', 0)
            })
    
    # Find best fold (highest mean_pearson)
    if summary['per_fold']:
        best = max(summary['per_fold'], key=lambda x: x['mean_pearson'])
        summary['best_fold'] = best['fold']
    
    # Per-target aggregation
    for target in target_names:
        values = []
        for r in all_fold_results:
            if r and 'per_target' in r['metrics'] and target in r['metrics']['per_target']:
                values.append(r['metrics']['per_target'][target].get('pearson_r', 0))
        
        if values:
            summary['per_target'][target] = {
                'mean_r': float(np.mean(values)),
                'std_r': float(np.std(values))
            }
    
    # Best and worst targets
    if summary['per_target']:
        sorted_targets = sorted(summary['per_target'].items(), 
                               key=lambda x: x[1]['mean_r'], reverse=True)
        summary['best_targets'] = [t[0] for t in sorted_targets[:5]]
        summary['worst_targets'] = [t[0] for t in sorted_targets[-5:]]
    
    # Save summary
    save_results_json(summary, output_dir / 'experiment_summary.json')
    
    return summary



