"""
Visualization utilities for experience prediction.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
from scipy.stats import pearsonr
import logging

logger = logging.getLogger(__name__)

# Set style with fallback for older matplotlib versions
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        pass  # Use default style
sns.set_palette("husl")


def plot_training_curves(history: Dict[str, List[float]], 
                         save_path: Path,
                         title: str = "Training Curves") -> None:
    """
    Plot training and validation loss/metrics curves.
    """
    if not history or 'train_loss' not in history or len(history['train_loss']) == 0:
        logger.warning("Empty history, skipping training curves plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax = axes[0]
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('Loss Curves', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Pearson correlation curves
    ax = axes[1]
    if 'val_pearson' in history:
        ax.plot(epochs, history['val_pearson'], 'g-', label='Val Pearson r', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Mean Pearson r', fontsize=12)
    ax.set_title('Correlation Curves', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 1.0)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved training curves to {save_path}")


def plot_predictions_scatter(y_true: np.ndarray, 
                            y_pred: np.ndarray,
                            target_names: List[str],
                            save_path: Path,
                            n_cols: int = 6) -> None:
    """
    Plot scatter plots of predictions vs true values for each target.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        logger.warning("Empty predictions, skipping scatter plot")
        return
    
    n_targets = len(target_names)
    n_rows = (n_targets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = axes.flatten() if n_targets > 1 else [axes]
    
    for i, (ax, name) in enumerate(zip(axes[:n_targets], target_names)):
        true_i = y_true[:, i]
        pred_i = y_pred[:, i]
        
        # Scatter plot
        ax.scatter(true_i, pred_i, alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
        
        # Fit line
        if len(true_i) > 2:
            z = np.polyfit(true_i, pred_i, 1)
            p = np.poly1d(z)
            x_line = np.linspace(true_i.min(), true_i.max(), 100)
            ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.8)
        
        # Perfect prediction line
        lims = [min(true_i.min(), pred_i.min()), max(true_i.max(), pred_i.max())]
        ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=1)
        
        # Correlation
        r, p = pearsonr(true_i, pred_i) if len(true_i) > 2 else (0, 1)
        
        ax.set_xlabel('True', fontsize=9)
        ax.set_ylabel('Predicted', fontsize=9)
        ax.set_title(f'{name}\nr={r:.3f}', fontsize=10)
        ax.tick_params(labelsize=8)
    
    # Hide empty subplots
    for ax in axes[n_targets:]:
        ax.axis('off')
    
    plt.suptitle('Predictions vs True Values', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved scatter plots to {save_path}")


def plot_per_target_correlations(y_true: np.ndarray,
                                 y_pred: np.ndarray, 
                                 target_names: List[str],
                                 save_path: Path) -> Dict[str, float]:
    """
    Plot bar chart of per-target correlations.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        logger.warning("Empty data, skipping correlation plot")
        return {}
    
    correlations = {}
    n_targets = min(len(target_names), y_true.shape[1] if y_true.ndim > 1 else 1)
    for i, name in enumerate(target_names[:n_targets]):
        try:
            r, _ = pearsonr(y_true[:, i], y_pred[:, i]) if len(y_true) > 2 else (0, 1)
            correlations[name] = r if not np.isnan(r) else 0
        except Exception:
            correlations[name] = 0
    
    if not correlations:
        logger.warning("No correlations computed, skipping plot")
        return {}
    
    # Sort by correlation
    sorted_items = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in sorted_items]
    values = [x[1] for x in sorted_items]
    
    # Color by value
    colors = ['green' if v > 0.3 else 'orange' if v > 0 else 'red' for v in values]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(len(names)), values, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('Pearson Correlation (r)', fontsize=12)
    ax.set_title('Per-Target Prediction Correlations', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=1)
    ax.axvline(x=0.3, color='green', linewidth=1, linestyle='--', alpha=0.5)
    ax.set_xlim(-0.5, 1.0)
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{val:.2f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved correlation bar chart to {save_path}")
    
    return correlations


def plot_residuals_heatmap(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          target_names: List[str],
                          save_path: Path) -> None:
    """
    Plot heatmap of prediction residuals (signed errors) per target and sample.
    Positive = overestimation, Negative = underestimation.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        logger.warning("Empty data, skipping residuals heatmap")
        return
    
    residuals = y_pred - y_true  # Positive = overestimate, Negative = underestimate
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Truncate target names for display
    short_names = [n[:20] + '...' if len(n) > 20 else n for n in target_names]
    
    # Use diverging colormap centered at 0
    max_abs = np.max(np.abs(residuals))
    
    sns.heatmap(residuals.T, ax=ax, cmap='RdBu_r', center=0,
                vmin=-max_abs, vmax=max_abs,
                xticklabels=[f'S{i+1}' for i in range(len(y_true))],
                yticklabels=short_names,
                cbar_kws={'label': 'Residual (Pred - True)'})
    
    ax.set_xlabel('Subject', fontsize=12)
    ax.set_ylabel('Target', fontsize=12)
    ax.set_title('Prediction Residuals (Red=Overestimate, Blue=Underestimate)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved residuals heatmap to {save_path}")


def plot_normalized_error_heatmap(y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  target_names: List[str],
                                  save_path: Path) -> None:
    """
    Plot heatmap of NORMALIZED prediction errors per target and sample.
    Each target is normalized by its range, making errors comparable across scales.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        logger.warning("Empty data, skipping normalized error heatmap")
        return
    
    # Normalize errors by each target's range
    errors = y_true - y_pred
    target_ranges = np.ptp(y_true, axis=0)  # max - min per target
    target_ranges[target_ranges == 0] = 1  # Avoid division by zero
    normalized_errors = errors / target_ranges * 100  # As percentage of range
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    short_names = [n[:20] + '...' if len(n) > 20 else n for n in target_names]
    
    # Use diverging colormap
    max_abs = min(np.percentile(np.abs(normalized_errors), 95), 100)
    
    sns.heatmap(normalized_errors.T, ax=ax, cmap='RdBu_r', center=0,
                vmin=-max_abs, vmax=max_abs,
                xticklabels=[f'S{i+1}' for i in range(len(y_true))],
                yticklabels=short_names,
                cbar_kws={'label': 'Normalized Error (% of range)'})
    
    ax.set_xlabel('Subject', fontsize=12)
    ax.set_ylabel('Target', fontsize=12)
    ax.set_title('Normalized Prediction Errors (% of each target range)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved normalized error heatmap to {save_path}")


def plot_target_heatmap(y_true: np.ndarray,
                       y_pred: np.ndarray,
                       target_names: List[str],
                       save_path: Path) -> None:
    """
    Plot heatmap of prediction errors (absolute) per target and sample.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        logger.warning("Empty data, skipping heatmap")
        return
    
    errors = np.abs(y_true - y_pred)
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Truncate target names for display
    short_names = [n[:20] + '...' if len(n) > 20 else n for n in target_names]
    
    sns.heatmap(errors.T, ax=ax, cmap='YlOrRd', 
                xticklabels=[f'S{i+1}' for i in range(len(y_true))],
                yticklabels=short_names,
                cbar_kws={'label': 'Absolute Error'})
    
    ax.set_xlabel('Subject', fontsize=12)
    ax.set_ylabel('Target', fontsize=12)
    ax.set_title('Prediction Errors Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved error heatmap to {save_path}")


def plot_category_summary(correlations: Dict[str, float],
                         target_names: List[str],
                         save_path: Path) -> None:
    """
    Plot summary of correlations by target category (ASC, NDE, MEQ, Post).
    """
    if not correlations:
        logger.warning("No correlations, skipping category summary")
        return
    
    categories = {
        'ASC': target_names[:11],
        'NDE': target_names[11:15] if len(target_names) > 11 else [],
        'MEQ': target_names[15:20] if len(target_names) > 15 else [],
        'Post': target_names[20:23] if len(target_names) > 20 else []
    }
    
    cat_means = {}
    cat_stds = {}
    
    for cat, targets in categories.items():
        values = [correlations.get(t, 0) for t in targets if t in correlations]
        if values:
            cat_means[cat] = np.mean(values)
            cat_stds[cat] = np.std(values)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    cats = list(cat_means.keys())
    means = [cat_means[c] for c in cats]
    stds = [cat_stds[c] for c in cats]
    
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    bars = ax.bar(cats, means, yerr=stds, capsize=5, 
                  color=colors[:len(cats)], edgecolor='black', linewidth=1)
    
    ax.set_ylabel('Mean Pearson r', fontsize=12)
    ax.set_title('Prediction Performance by Category', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=1)
    ax.axhline(y=0.3, color='green', linewidth=1, linestyle='--', alpha=0.5)
    ax.set_ylim(-0.3, 0.8)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                f'{mean:.2f}', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved category summary to {save_path}")


def save_all_visualizations(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           target_names: List[str],
                           history: Dict[str, List[float]],
                           output_dir: Path,
                           prefix: str = "") -> Dict[str, float]:
    """
    Generate and save all visualizations.
    
    Returns:
        Per-target correlations dict
    """
    # Validate inputs
    if y_true is None or y_pred is None or len(y_true) == 0 or len(y_pred) == 0:
        logger.warning("Empty predictions, skipping visualizations")
        return {}
    
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    output_dir = Path(output_dir)
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    prefix = f"{prefix}_" if prefix else ""
    
    correlations = {}
    
    try:
        # Training curves
        if history:
            plot_training_curves(history, viz_dir / f'{prefix}training_curves.png')
        
        # Predictions scatter
        plot_predictions_scatter(y_true, y_pred, target_names, 
                                viz_dir / f'{prefix}predictions_scatter.png')
        
        # Per-target correlations
        correlations = plot_per_target_correlations(y_true, y_pred, target_names,
                                                    viz_dir / f'{prefix}correlations_bar.png')
        
        # Error heatmap (absolute)
        plot_target_heatmap(y_true, y_pred, target_names,
                           viz_dir / f'{prefix}error_heatmap.png')
        
        # Residuals heatmap (signed - shows over/underestimation)
        plot_residuals_heatmap(y_true, y_pred, target_names,
                              viz_dir / f'{prefix}residuals_heatmap.png')
        
        # Normalized error heatmap (comparable across different scales)
        plot_normalized_error_heatmap(y_true, y_pred, target_names,
                                     viz_dir / f'{prefix}normalized_error_heatmap.png')
        
        # Category summary
        plot_category_summary(correlations, target_names,
                             viz_dir / f'{prefix}category_summary.png')
        
        logger.info(f"All visualizations saved to {viz_dir}")
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
    
    return correlations

