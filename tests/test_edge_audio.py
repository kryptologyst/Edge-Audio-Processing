"""Tests for edge audio processing."""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils.core import set_deterministic_seed, get_device, EdgeMetrics
from src.utils.audio import AudioProcessor, AudioDataset
from src.models.audio_models import create_model, AudioEventCNN, TinyAudioNet
from src.models.compression import ModelCompressor
from src.pipelines.training import create_data_loaders


class TestCoreUtils:
    """Test core utility functions."""
    
    def test_deterministic_seed(self):
        """Test deterministic seeding."""
        set_deterministic_seed(42)
        
        # Test numpy
        np.random.seed(42)
        val1 = np.random.random()
        np.random.seed(42)
        val2 = np.random.random()
        assert val1 == val2
        
        # Test torch
        torch.manual_seed(42)
        val1 = torch.rand(1).item()
        torch.manual_seed(42)
        val2 = torch.rand(1).item()
        assert val1 == val2
        
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert device in ["cpu", "cuda", "mps"]
        
    def test_edge_metrics(self):
        """Test edge metrics collection."""
        metrics = EdgeMetrics()
        
        # Add some data
        metrics.add_latency(0.001)
        metrics.add_latency(0.002)
        metrics.add_memory_usage(100.0)
        metrics.accuracy = 0.95
        metrics.model_size_mb = 1.5
        
        # Get summary
        summary = metrics.get_summary()
        
        assert "latency_p50_ms" in summary
        assert "latency_p95_ms" in summary
        assert "memory_mean_mb" in summary
        assert "accuracy" in summary
        assert "model_size_mb" in summary


class TestAudioProcessor:
    """Test audio processing utilities."""
    
    def test_audio_processor_init(self):
        """Test audio processor initialization."""
        processor = AudioProcessor()
        
        assert processor.sample_rate == 16000
        assert processor.n_mfcc == 13
        assert processor.max_length == 100
        
    def test_synthesize_audio_event(self):
        """Test audio event synthesis."""
        processor = AudioProcessor()
        
        # Test different event types
        for event_type in ["clap", "glass_break", "noise"]:
            audio = processor.synthesize_audio_event(event_type, duration=1.0)
            
            assert len(audio) == processor.sample_rate
            assert np.all(np.abs(audio) <= 1.0)  # Normalized
            
    def test_extract_mfcc(self):
        """Test MFCC extraction."""
        processor = AudioProcessor()
        
        # Generate test audio
        audio = processor.synthesize_audio_event("clap", duration=1.0)
        
        # Extract MFCC
        mfcc = processor.extract_mfcc(audio)
        
        assert mfcc.shape[1] == processor.n_mfcc
        assert mfcc.shape[0] > 0
        
    def test_preprocess_audio(self):
        """Test complete audio preprocessing."""
        processor = AudioProcessor()
        
        # Generate test audio
        audio = processor.synthesize_audio_event("clap", duration=1.0)
        
        # Preprocess
        features = processor.preprocess_audio(audio)
        
        assert features.shape == (processor.max_length, processor.n_mfcc)
        
    def test_audio_dataset(self):
        """Test audio dataset."""
        processor = AudioProcessor()
        dataset = AudioDataset(processor)
        
        # Generate synthetic data
        X, y = dataset.generate_synthetic_data(n_samples_per_class=10)
        
        assert len(X) == len(y)
        assert len(np.unique(y)) == 3  # 3 classes
        assert X.shape[1:] == (processor.max_length, processor.n_mfcc)


class TestModels:
    """Test neural network models."""
    
    def test_create_model(self):
        """Test model creation."""
        # Test different model types
        for model_type in ["cnn", "tiny", "transformer"]:
            model = create_model(
                model_type=model_type,
                input_length=100,
                n_mfcc=13,
                n_classes=3
            )
            
            assert model is not None
            assert isinstance(model, torch.nn.Module)
            
    def test_cnn_model(self):
        """Test CNN model."""
        model = AudioEventCNN(input_length=100, n_mfcc=13, n_classes=3)
        
        # Test forward pass
        x = torch.randn(2, 13, 100)  # batch_size=2
        output = model(x)
        
        assert output.shape == (2, 3)
        assert torch.allclose(torch.sum(output, dim=1), torch.ones(2), atol=1e-6)
        
    def test_tiny_model(self):
        """Test tiny model."""
        model = TinyAudioNet(input_length=100, n_mfcc=13, n_classes=3)
        
        # Test forward pass
        x = torch.randn(2, 13, 100)
        output = model(x)
        
        assert output.shape == (2, 3)
        
    def test_model_parameters(self):
        """Test model parameter counts."""
        cnn_model = AudioEventCNN(input_length=100, n_mfcc=13, n_classes=3)
        tiny_model = TinyAudioNet(input_length=100, n_mfcc=13, n_classes=3)
        
        cnn_params = sum(p.numel() for p in cnn_model.parameters())
        tiny_params = sum(p.numel() for p in tiny_model.parameters())
        
        # Tiny model should have fewer parameters
        assert tiny_params < cnn_params


class TestCompression:
    """Test model compression."""
    
    def test_model_compressor(self):
        """Test model compressor."""
        model = AudioEventCNN(input_length=100, n_mfcc=13, n_classes=3)
        compressor = ModelCompressor(model)
        
        # Test saving and restoring
        compressor.save_original_model()
        original_params = sum(p.numel() for p in model.parameters())
        
        # Apply pruning
        pruned_model = compressor.prune_model(pruning_ratio=0.1)
        pruned_params = sum(p.numel() for p in pruned_model.parameters())
        
        # Parameters should be the same (pruning removes connections, not parameters)
        assert pruned_params == original_params
        
        # Restore original
        compressor.restore_original_model()
        restored_params = sum(p.numel() for p in model.parameters())
        assert restored_params == original_params


class TestTraining:
    """Test training pipeline."""
    
    def test_create_data_loaders(self):
        """Test data loader creation."""
        # Create dummy data
        X_train = np.random.randn(100, 100, 13)
        y_train = np.random.randint(0, 3, 100)
        X_test = np.random.randn(20, 100, 13)
        y_test = np.random.randint(0, 3, 20)
        
        train_loader, test_loader = create_data_loaders(
            X_train, y_train, X_test, y_test, batch_size=16
        )
        
        # Test train loader
        for batch_x, batch_y in train_loader:
            assert batch_x.shape[0] <= 16
            assert batch_x.shape[1:] == (100, 13)
            assert batch_y.shape[0] <= 16
            break
            
        # Test test loader
        for batch_x, batch_y in test_loader:
            assert batch_x.shape[0] <= 16
            assert batch_x.shape[1:] == (100, 13)
            assert batch_y.shape[0] <= 16
            break


if __name__ == "__main__":
    pytest.main([__file__])
