"""Edge Audio Processing Package."""

__version__ = "1.0.0"
__author__ = "Edge AI Team"
__email__ = "team@example.com"

from .utils.core import set_deterministic_seed, get_device, EdgeMetrics
from .utils.audio import AudioProcessor, AudioDataset
from .models.audio_models import create_model, AudioEventCNN, TinyAudioNet, AudioEventTransformer
from .models.compression import ModelCompressor, KnowledgeDistillation
from .pipelines.training import AudioTrainer, ModelEvaluator, create_data_loaders
from .export.deployment import ModelExporter, EdgeRuntime, DeploymentManager

__all__ = [
    "set_deterministic_seed",
    "get_device", 
    "EdgeMetrics",
    "AudioProcessor",
    "AudioDataset",
    "create_model",
    "AudioEventCNN",
    "TinyAudioNet", 
    "AudioEventTransformer",
    "ModelCompressor",
    "KnowledgeDistillation",
    "AudioTrainer",
    "ModelEvaluator",
    "create_data_loaders",
    "ModelExporter",
    "EdgeRuntime",
    "DeploymentManager"
]
