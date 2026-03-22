"""Model compression and optimization for edge deployment."""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.quantization import quantize_dynamic, quantize_static
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ModelCompressor:
    """Model compression utilities for edge deployment."""
    
    def __init__(self, model: nn.Module):
        """Initialize model compressor.
        
        Args:
            model: PyTorch model to compress.
        """
        self.model = model
        self.original_model = None
        
    def save_original_model(self) -> None:
        """Save a copy of the original model."""
        self.original_model = self.model.state_dict().copy()
        
    def restore_original_model(self) -> None:
        """Restore the original model."""
        if self.original_model is not None:
            self.model.load_state_dict(self.original_model)
            
    def prune_model(
        self,
        pruning_ratio: float = 0.2,
        pruning_type: str = "magnitude"
    ) -> nn.Module:
        """Prune the model to reduce size.
        
        Args:
            pruning_ratio: Fraction of parameters to prune.
            pruning_type: Type of pruning ('magnitude', 'structured').
            
        Returns:
            Pruned model.
        """
        logger.info(f"Pruning model with ratio {pruning_ratio} using {pruning_type} pruning")
        
        if pruning_type == "magnitude":
            return self._magnitude_pruning(pruning_ratio)
        elif pruning_type == "structured":
            return self._structured_pruning(pruning_ratio)
        else:
            raise ValueError(f"Unknown pruning type: {pruning_type}")
            
    def _magnitude_pruning(self, pruning_ratio: float) -> nn.Module:
        """Apply magnitude-based pruning."""
        parameters_to_prune = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                parameters_to_prune.append((module, 'weight'))
                
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_ratio,
        )
        
        # Remove pruning reparameterization
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
            
        return self.model
        
    def _structured_pruning(self, pruning_ratio: float) -> nn.Module:
        """Apply structured pruning."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv1d):
                prune.ln_structured(
                    module, name='weight', amount=pruning_ratio, n=2, dim=0
                )
                prune.remove(module, 'weight')
            elif isinstance(module, nn.Linear):
                prune.ln_structured(
                    module, name='weight', amount=pruning_ratio, n=2, dim=0
                )
                prune.remove(module, 'weight')
                
        return self.model
        
    def quantize_model(
        self,
        quantization_type: str = "dynamic",
        calibration_data: Optional[torch.Tensor] = None
    ) -> nn.Module:
        """Quantize the model for edge deployment.
        
        Args:
            quantization_type: Type of quantization ('dynamic', 'static').
            calibration_data: Data for static quantization calibration.
            
        Returns:
            Quantized model.
        """
        logger.info(f"Quantizing model using {quantization_type} quantization")
        
        if quantization_type == "dynamic":
            return self._dynamic_quantization()
        elif quantization_type == "static":
            return self._static_quantization(calibration_data)
        else:
            raise ValueError(f"Unknown quantization type: {quantization_type}")
            
    def _dynamic_quantization(self) -> nn.Module:
        """Apply dynamic quantization."""
        self.model.eval()
        quantized_model = quantize_dynamic(
            self.model, {nn.Linear, nn.Conv1d}, dtype=torch.qint8
        )
        return quantized_model
        
    def _static_quantization(self, calibration_data: torch.Tensor) -> nn.Module:
        """Apply static quantization."""
        if calibration_data is None:
            raise ValueError("Calibration data required for static quantization")
            
        self.model.eval()
        
        # Set quantization configuration
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Prepare model for quantization
        prepared_model = torch.quantization.prepare(self.model)
        
        # Calibrate with sample data
        with torch.no_grad():
            for i in range(min(100, len(calibration_data))):
                prepared_model(calibration_data[i:i+1])
                
        # Convert to quantized model
        quantized_model = torch.quantization.convert(prepared_model)
        
        return quantized_model


class KnowledgeDistillation:
    """Knowledge distillation for creating smaller student models."""
    
    def __init__(self, teacher_model: nn.Module, student_model: nn.Module):
        """Initialize knowledge distillation.
        
        Args:
            teacher_model: Large teacher model.
            student_model: Small student model.
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.teacher_model.eval()
        
    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 3.0,
        alpha: float = 0.7
    ) -> torch.Tensor:
        """Compute distillation loss.
        
        Args:
            student_logits: Student model logits.
            teacher_logits: Teacher model logits.
            labels: Ground truth labels.
            temperature: Temperature for softmax.
            alpha: Weight between distillation and classification loss.
            
        Returns:
            Combined distillation loss.
        """
        # Softmax with temperature
        student_soft = F.softmax(student_logits / temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
        
        # Distillation loss (KL divergence)
        distillation_loss = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=1),
            teacher_soft,
            reduction='batchmean'
        ) * (temperature ** 2)
        
        # Classification loss
        classification_loss = F.cross_entropy(student_logits, labels)
        
        # Combined loss
        total_loss = alpha * distillation_loss + (1 - alpha) * classification_loss
        
        return total_loss
        
    def train_student(
        self,
        train_loader: torch.utils.data.DataLoader,
        epochs: int = 10,
        learning_rate: float = 0.001,
        temperature: float = 3.0,
        alpha: float = 0.7
    ) -> Dict[str, float]:
        """Train student model using knowledge distillation.
        
        Args:
            train_loader: Training data loader.
            epochs: Number of training epochs.
            learning_rate: Learning rate for optimizer.
            temperature: Temperature for distillation.
            alpha: Weight between distillation and classification loss.
            
        Returns:
            Training metrics.
        """
        optimizer = torch.optim.Adam(self.student_model.parameters(), lr=learning_rate)
        self.student_model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_idx, (data, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                
                # Forward pass
                student_logits = self.student_model(data)
                
                with torch.no_grad():
                    teacher_logits = self.teacher_model(data)
                
                # Compute loss
                loss = self.distillation_loss(
                    student_logits, teacher_logits, labels, temperature, alpha
                )
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                epoch_loss += loss.item()
                _, predicted = torch.max(student_logits.data, 1)
                epoch_total += labels.size(0)
                epoch_correct += (predicted == labels).sum().item()
                
            # Epoch statistics
            avg_loss = epoch_loss / len(train_loader)
            accuracy = 100.0 * epoch_correct / epoch_total
            
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
            
            total_loss += avg_loss
            correct += epoch_correct
            total += epoch_total
            
        return {
            "avg_loss": total_loss / epochs,
            "accuracy": 100.0 * correct / total
        }


def compress_model_for_edge(
    model: nn.Module,
    compression_config: Dict[str, Any]
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Compress model for edge deployment.
    
    Args:
        model: PyTorch model to compress.
        compression_config: Compression configuration.
        
    Returns:
        Tuple of (compressed_model, compression_stats).
    """
    compressor = ModelCompressor(model)
    compressor.save_original_model()
    
    stats = {}
    
    # Get original model size
    original_params = sum(p.numel() for p in model.parameters())
    stats["original_params"] = original_params
    
    # Apply pruning if specified
    if compression_config.get("pruning_ratio", 0) > 0:
        model = compressor.prune_model(
            pruning_ratio=compression_config["pruning_ratio"],
            pruning_type=compression_config.get("pruning_type", "magnitude")
        )
        
        pruned_params = sum(p.numel() for p in model.parameters())
        stats["pruned_params"] = pruned_params
        stats["pruning_ratio"] = 1 - (pruned_params / original_params)
        
    # Apply quantization if specified
    if compression_config.get("quantization_type"):
        model = compressor.quantize_model(
            quantization_type=compression_config["quantization_type"],
            calibration_data=compression_config.get("calibration_data")
        )
        
        # Quantized models have different parameter counting
        stats["quantized"] = True
        
    return model, stats
