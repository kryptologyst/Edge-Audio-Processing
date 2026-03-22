"""Core utilities for edge audio processing."""

import random
import numpy as np
import torch
import tensorflow as tf
from typing import Dict, Any, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_deterministic_seed(seed: int = 42) -> None:
    """Set deterministic seeds for all random number generators.
    
    Args:
        seed: Random seed value for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tf.random.set_seed(seed)
    
    # Additional PyTorch settings for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Deterministic seed set to {seed}")


def get_device() -> str:
    """Get the best available device for computation.
    
    Returns:
        Device string ('cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using MPS device (Apple Silicon)")
    else:
        device = "cpu"
        logger.info("Using CPU device")
    
    return device


def format_model_size(model: torch.nn.Module) -> str:
    """Calculate and format model size in MB.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Formatted model size string.
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size
    size_mb = total_size / (1024 * 1024)
    return f"{size_mb:.2f} MB"


def format_latency_ms(latency_seconds: float) -> str:
    """Format latency in milliseconds.
    
    Args:
        latency_seconds: Latency in seconds.
        
    Returns:
        Formatted latency string.
    """
    latency_ms = latency_seconds * 1000
    if latency_ms < 1:
        return f"{latency_ms:.2f} ms"
    else:
        return f"{latency_ms:.1f} ms"


class EdgeMetrics:
    """Container for edge performance metrics."""
    
    def __init__(self):
        self.latencies: list[float] = []
        self.memory_usage: list[float] = []
        self.energy_consumption: list[float] = []
        self.accuracy: Optional[float] = None
        self.model_size_mb: Optional[float] = None
        
    def add_latency(self, latency_seconds: float) -> None:
        """Add latency measurement."""
        self.latencies.append(latency_seconds)
        
    def add_memory_usage(self, memory_mb: float) -> None:
        """Add memory usage measurement."""
        self.memory_usage.append(memory_mb)
        
    def add_energy_consumption(self, energy_joules: float) -> None:
        """Add energy consumption measurement."""
        self.energy_consumption.append(energy_joules)
        
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        summary = {}
        
        if self.latencies:
            summary["latency_p50_ms"] = np.percentile(self.latencies, 50) * 1000
            summary["latency_p95_ms"] = np.percentile(self.latencies, 95) * 1000
            summary["latency_mean_ms"] = np.mean(self.latencies) * 1000
            
        if self.memory_usage:
            summary["memory_mean_mb"] = np.mean(self.memory_usage)
            summary["memory_peak_mb"] = np.max(self.memory_usage)
            
        if self.energy_consumption:
            summary["energy_mean_j"] = np.mean(self.energy_consumption)
            
        if self.accuracy is not None:
            summary["accuracy"] = self.accuracy
            
        if self.model_size_mb is not None:
            summary["model_size_mb"] = self.model_size_mb
            
        return summary
