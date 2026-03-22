"""Training and evaluation pipeline for edge audio processing."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Dict, Any, Tuple, Optional, List
import logging
import time
from tqdm import tqdm

from ..utils.core import EdgeMetrics, get_device
from ..utils.audio import AudioProcessor, AudioDataset
from .audio_models import create_model
from .compression import ModelCompressor, KnowledgeDistillation

logger = logging.getLogger(__name__)


class AudioTrainer:
    """Trainer for audio event detection models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4
    ):
        """Initialize trainer.
        
        Args:
            model: PyTorch model to train.
            device: Device to use for training.
            learning_rate: Learning rate for optimizer.
            weight_decay: Weight decay for regularization.
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        self.metrics = EdgeMetrics()
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader.
            epoch: Current epoch number.
            
        Returns:
            Training metrics for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (data, labels) in enumerate(pbar):
            data, labels = data.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.0 * correct / total:.2f}%'
            })
            
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': 100.0 * correct / total
        }
        
    def evaluate(
        self,
        test_loader: DataLoader,
        return_predictions: bool = False
    ) -> Dict[str, Any]:
        """Evaluate the model.
        
        Args:
            test_loader: Test data loader.
            return_predictions: Whether to return predictions.
            
        Returns:
            Evaluation metrics and optionally predictions.
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        inference_times = []
        
        with torch.no_grad():
            for data, labels in tqdm(test_loader, desc="Evaluating"):
                data, labels = data.to(self.device), labels.to(self.device)
                
                # Measure inference time
                start_time = time.time()
                outputs = self.model(data)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Calculate loss
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                # Get predictions
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        avg_inference_time = np.mean(inference_times)
        
        metrics = {
            'loss': total_loss / len(test_loader),
            'accuracy': accuracy,
            'avg_inference_time_ms': avg_inference_time * 1000,
            'inference_times': inference_times
        }
        
        if return_predictions:
            metrics['predictions'] = all_predictions
            metrics['labels'] = all_labels
            
        return metrics
        
    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int = 10,
        save_best: bool = True
    ) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            train_loader: Training data loader.
            test_loader: Test data loader.
            epochs: Number of training epochs.
            save_best: Whether to save the best model.
            
        Returns:
            Training history.
        """
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': [],
            'test_accuracy': []
        }
        
        best_accuracy = 0.0
        best_model_state = None
        
        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch + 1)
            history['train_loss'].append(train_metrics['loss'])
            history['train_accuracy'].append(train_metrics['accuracy'])
            
            # Evaluate
            test_metrics = self.evaluate(test_loader)
            history['test_loss'].append(test_metrics['loss'])
            history['test_accuracy'].append(test_metrics['accuracy'])
            
            logger.info(
                f"Epoch {epoch+1}/{epochs}: "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                f"Test Loss: {test_metrics['loss']:.4f}, "
                f"Test Acc: {test_metrics['accuracy']:.2f}%"
            )
            
            # Save best model
            if save_best and test_metrics['accuracy'] > best_accuracy:
                best_accuracy = test_metrics['accuracy']
                best_model_state = self.model.state_dict().copy()
                
        if save_best and best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info(f"Best model restored with accuracy: {best_accuracy:.2f}%")
            
        return history


class ModelEvaluator:
    """Comprehensive model evaluation for edge deployment."""
    
    def __init__(self, model: nn.Module, device: str = "cpu"):
        """Initialize evaluator.
        
        Args:
            model: PyTorch model to evaluate.
            device: Device to use for evaluation.
        """
        self.model = model.to(device)
        self.device = device
        self.metrics = EdgeMetrics()
        
    def evaluate_accuracy(
        self,
        test_loader: DataLoader,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Evaluate model accuracy.
        
        Args:
            test_loader: Test data loader.
            class_names: Names of classes for reporting.
            
        Returns:
            Accuracy metrics.
        """
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in tqdm(test_loader, desc="Evaluating accuracy"):
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        
        metrics = {
            'accuracy': accuracy,
            'predictions': all_predictions,
            'labels': all_labels
        }
        
        # Classification report
        if class_names:
            metrics['classification_report'] = classification_report(
                all_labels, all_predictions, target_names=class_names
            )
            
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(all_labels, all_predictions)
        
        return metrics
        
    def evaluate_performance(
        self,
        test_loader: DataLoader,
        n_warmup: int = 10,
        n_iterations: int = 100
    ) -> Dict[str, Any]:
        """Evaluate model performance metrics.
        
        Args:
            test_loader: Test data loader.
            n_warmup: Number of warmup iterations.
            n_iterations: Number of timing iterations.
            
        Returns:
            Performance metrics.
        """
        self.model.eval()
        
        # Warmup
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                if i >= n_warmup:
                    break
                data = data.to(self.device)
                _ = self.model(data)
                
        # Timing
        times = []
        memory_usage = []
        
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                if i >= n_iterations:
                    break
                    
                data = data.to(self.device)
                
                # Measure memory before
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    
                # Time inference
                start_time = time.time()
                _ = self.model(data)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                    memory_usage.append(memory_mb)
                    
                end_time = time.time()
                times.append(end_time - start_time)
                
        # Calculate statistics
        latency_p50 = np.percentile(times, 50)
        latency_p95 = np.percentile(times, 95)
        latency_mean = np.mean(times)
        
        metrics = {
            'latency_p50_ms': latency_p50 * 1000,
            'latency_p95_ms': latency_p95 * 1000,
            'latency_mean_ms': latency_mean * 1000,
            'throughput_fps': 1.0 / latency_mean,
            'latency_times': times
        }
        
        if memory_usage:
            metrics['memory_mean_mb'] = np.mean(memory_usage)
            metrics['memory_peak_mb'] = np.max(memory_usage)
            
        return metrics
        
    def evaluate_model_size(self) -> Dict[str, Any]:
        """Evaluate model size metrics.
        
        Returns:
            Model size metrics.
        """
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Estimate model size
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        total_size = param_size + buffer_size
        
        metrics = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_size / (1024 * 1024),
            'model_size_bytes': total_size
        }
        
        return metrics
        
    def comprehensive_evaluation(
        self,
        test_loader: DataLoader,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive model evaluation.
        
        Args:
            test_loader: Test data loader.
            class_names: Names of classes for reporting.
            
        Returns:
            Comprehensive evaluation metrics.
        """
        logger.info("Starting comprehensive model evaluation...")
        
        # Accuracy evaluation
        accuracy_metrics = self.evaluate_accuracy(test_loader, class_names)
        
        # Performance evaluation
        performance_metrics = self.evaluate_performance(test_loader)
        
        # Model size evaluation
        size_metrics = self.evaluate_model_size()
        
        # Combine all metrics
        all_metrics = {
            **accuracy_metrics,
            **performance_metrics,
            **size_metrics
        }
        
        logger.info("Comprehensive evaluation completed")
        return all_metrics


def create_data_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """Create PyTorch data loaders.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        batch_size: Batch size for data loaders.
        shuffle: Whether to shuffle training data.
        
    Returns:
        Tuple of (train_loader, test_loader).
    """
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, test_loader
