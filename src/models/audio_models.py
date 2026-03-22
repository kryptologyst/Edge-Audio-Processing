"""Neural network models for edge audio processing."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AudioEventCNN(nn.Module):
    """1D CNN model for audio event detection optimized for edge deployment."""
    
    def __init__(
        self,
        input_length: int = 100,
        n_mfcc: int = 13,
        n_classes: int = 3,
        dropout_rate: float = 0.2
    ):
        """Initialize audio event CNN.
        
        Args:
            input_length: Length of input sequence.
            n_mfcc: Number of MFCC coefficients.
            n_classes: Number of output classes.
            dropout_rate: Dropout rate for regularization.
        """
        super().__init__()
        
        self.input_length = input_length
        self.n_mfcc = n_mfcc
        self.n_classes = n_classes
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(n_mfcc, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        
        # Calculate flattened size
        self.flattened_size = self._get_flattened_size()
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_classes)
        
    def _get_flattened_size(self) -> int:
        """Calculate the size after convolutional layers."""
        # Simulate forward pass to get output size
        x = torch.zeros(1, self.n_mfcc, self.input_length)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        return x.numel()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor with shape (batch_size, n_mfcc, input_length).
            
        Returns:
            Output logits with shape (batch_size, n_classes).
        """
        # Transpose to (batch_size, n_mfcc, input_length) if needed
        if x.dim() == 3 and x.shape[1] != self.n_mfcc:
            x = x.transpose(1, 2)
            
        # Convolutional layers
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


class TinyAudioNet(nn.Module):
    """Ultra-lightweight model for microcontrollers."""
    
    def __init__(
        self,
        input_length: int = 100,
        n_mfcc: int = 13,
        n_classes: int = 3
    ):
        """Initialize tiny audio network.
        
        Args:
            input_length: Length of input sequence.
            n_mfcc: Number of MFCC coefficients.
            n_classes: Number of output classes.
        """
        super().__init__()
        
        self.input_length = input_length
        self.n_mfcc = n_mfcc
        self.n_classes = n_classes
        
        # Minimal convolutional layers
        self.conv1 = nn.Conv1d(n_mfcc, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=4)
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=4)
        
        # Calculate flattened size
        self.flattened_size = self._get_flattened_size()
        
        # Single fully connected layer
        self.fc = nn.Linear(self.flattened_size, n_classes)
        
    def _get_flattened_size(self) -> int:
        """Calculate the size after convolutional layers."""
        x = torch.zeros(1, self.n_mfcc, self.input_length)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        return x.numel()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor with shape (batch_size, n_mfcc, input_length).
            
        Returns:
            Output logits with shape (batch_size, n_classes).
        """
        # Transpose if needed
        if x.dim() == 3 and x.shape[1] != self.n_mfcc:
            x = x.transpose(1, 2)
            
        # Convolutional layers
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        
        # Flatten and fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class AudioEventTransformer(nn.Module):
    """Transformer-based model for audio event detection."""
    
    def __init__(
        self,
        input_length: int = 100,
        n_mfcc: int = 13,
        n_classes: int = 3,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout_rate: float = 0.1
    ):
        """Initialize audio event transformer.
        
        Args:
            input_length: Length of input sequence.
            n_mfcc: Number of MFCC coefficients.
            n_classes: Number of output classes.
            d_model: Model dimension.
            n_heads: Number of attention heads.
            n_layers: Number of transformer layers.
            dropout_rate: Dropout rate.
        """
        super().__init__()
        
        self.input_length = input_length
        self.n_mfcc = n_mfcc
        self.n_classes = n_classes
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(n_mfcc, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(input_length, d_model) * 0.1
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, n_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor with shape (batch_size, input_length, n_mfcc).
            
        Returns:
            Output logits with shape (batch_size, n_classes).
        """
        # Transpose if needed
        if x.dim() == 3 and x.shape[2] != self.n_mfcc:
            x = x.transpose(1, 2)
            
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.pos_encoding.unsqueeze(0)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classification
        x = self.classifier(x)
        
        return x


def create_model(
    model_type: str = "cnn",
    input_length: int = 100,
    n_mfcc: int = 13,
    n_classes: int = 3,
    **kwargs
) -> nn.Module:
    """Create audio event detection model.
    
    Args:
        model_type: Type of model ('cnn', 'tiny', 'transformer').
        input_length: Length of input sequence.
        n_mfcc: Number of MFCC coefficients.
        n_classes: Number of output classes.
        **kwargs: Additional model parameters.
        
    Returns:
        PyTorch model instance.
    """
    if model_type == "cnn":
        return AudioEventCNN(
            input_length=input_length,
            n_mfcc=n_mfcc,
            n_classes=n_classes,
            **kwargs
        )
    elif model_type == "tiny":
        return TinyAudioNet(
            input_length=input_length,
            n_mfcc=n_mfcc,
            n_classes=n_classes
        )
    elif model_type == "transformer":
        return AudioEventTransformer(
            input_length=input_length,
            n_mfcc=n_mfcc,
            n_classes=n_classes,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
