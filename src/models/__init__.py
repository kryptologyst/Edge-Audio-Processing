"""Neural network models for edge audio processing."""

from .audio_models import create_model, AudioEventCNN, TinyAudioNet, AudioEventTransformer
from .compression import ModelCompressor, KnowledgeDistillation

__all__ = [
    "create_model",
    "AudioEventCNN",
    "TinyAudioNet",
    "AudioEventTransformer",
    "ModelCompressor",
    "KnowledgeDistillation"
]
