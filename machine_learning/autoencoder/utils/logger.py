"""
Logging utilities for VAE experiment tracking.
"""

import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from io import TextIOWrapper


class UnbufferedStream:
    """Wrapper that flushes after every write."""
    def __init__(self, stream):
        self.stream = stream
    
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    
    def flush(self):
        self.stream.flush()
    
    def __getattr__(self, attr):
        return getattr(self.stream, attr)


class FlushingStreamHandler(logging.StreamHandler):
    """StreamHandler that flushes after every emit for real-time output."""
    
    def __init__(self, stream=None):
        # Wrap stdout in unbuffered wrapper
        if stream is None:
            stream = sys.stdout
        super().__init__(UnbufferedStream(stream))
    
    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_logging(config: Dict[str, Any], log_dir: Path) -> logging.Logger:
    """
    Setup logging configuration with unbuffered output.
    
    Args:
        config: Configuration dictionary
        log_dir: Directory to save log files
        
    Returns:
        Configured logger
    """
    # Force unbuffered stdout/stderr (only wrap if not already wrapped)
    if not isinstance(sys.stdout, UnbufferedStream):
        sys.stdout = UnbufferedStream(sys.stdout)
    if not isinstance(sys.stderr, UnbufferedStream):
        sys.stderr = UnbufferedStream(sys.stderr)
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_vae_{timestamp}.log"
    
    # Get log level from config
    level_name = config['logging']['level']
    level = getattr(logging, level_name)
    
    # Create handlers - use FlushingStreamHandler for real-time output
    file_handler = logging.FileHandler(log_file)
    stream_handler = FlushingStreamHandler()
    
    # Set format
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    # Clear existing handlers
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
    
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









