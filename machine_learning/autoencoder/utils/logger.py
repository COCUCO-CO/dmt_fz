"""
Logging utilities for VAE experiment tracking.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


def setup_logging(config: Dict[str, Any], log_dir: Path) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        config: Configuration dictionary
        log_dir: Directory to save log files
        
    Returns:
        Configured logger
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_vae_{timestamp}.log"
    
    # Get log level from config
    level_name = config['logging']['level']
    level = getattr(logging, level_name)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # Override any existing config
    )
    
    logger = logging.getLogger('BrainStateVAE')
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger


def log_metrics(logger: logging.Logger,
                epoch: int,
                phase: str,
                metrics: Dict[str, float],
                prefix: str = ""):
    """
    Log metrics in a formatted way.
    
    Args:
        logger: Logger instance
        epoch: Current epoch number
        phase: Training phase ('train', 'val', 'test')
        metrics: Dictionary of metric names and values
        prefix: Optional prefix for log message
    """
    metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    logger.info(f"{prefix}Epoch {epoch:03d} [{phase}] - {metrics_str}")


def log_vae_metrics(logger: logging.Logger,
                    epoch: int,
                    phase: str,
                    loss: float,
                    recon_loss: float,
                    kl_loss: float,
                    beta: float):
    """
    Log VAE-specific metrics.
    
    Args:
        logger: Logger instance
        epoch: Current epoch number
        phase: Training phase
        loss: Total loss
        recon_loss: Reconstruction loss
        kl_loss: KL divergence loss
        beta: Current beta value
    """
    logger.info(
        f"Epoch {epoch:03d} [{phase}] - "
        f"Loss: {loss:.4f} | Recon: {recon_loss:.4f} | "
        f"KL: {kl_loss:.4f} | β: {beta:.4f}"
    )









