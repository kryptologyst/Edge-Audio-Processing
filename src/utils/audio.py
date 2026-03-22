"""Audio processing utilities for edge devices."""

import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Audio processing utilities optimized for edge devices."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        max_length: int = 100
    ):
        """Initialize audio processor.
        
        Args:
            sample_rate: Target sample rate for audio.
            n_mfcc: Number of MFCC coefficients.
            n_fft: FFT window size.
            hop_length: Number of samples between successive frames.
            n_mels: Number of mel bands.
            max_length: Maximum sequence length for fixed-size inputs.
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.max_length = max_length
        
    def load_audio(self, file_path: str) -> np.ndarray:
        """Load audio file and resample to target sample rate.
        
        Args:
            file_path: Path to audio file.
            
        Returns:
            Audio signal as numpy array.
        """
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            return audio
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {e}")
            raise
            
    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC features from audio signal.
        
        Args:
            audio: Audio signal.
            
        Returns:
            MFCC features with shape (n_frames, n_mfcc).
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        return mfcc.T
        
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Extract mel spectrogram from audio signal.
        
        Args:
            audio: Audio signal.
            
        Returns:
            Mel spectrogram with shape (n_mels, n_frames).
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        return librosa.power_to_db(mel_spec, ref=np.max)
        
    def pad_or_truncate(self, features: np.ndarray) -> np.ndarray:
        """Pad or truncate features to fixed length.
        
        Args:
            features: Feature array with shape (n_frames, n_features).
            
        Returns:
            Fixed-size feature array with shape (max_length, n_features).
        """
        if features.shape[0] > self.max_length:
            # Truncate
            return features[:self.max_length]
        elif features.shape[0] < self.max_length:
            # Pad with zeros
            padding = np.zeros((self.max_length - features.shape[0], features.shape[1]))
            return np.vstack([features, padding])
        else:
            return features
            
    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Complete audio preprocessing pipeline.
        
        Args:
            audio: Raw audio signal.
            
        Returns:
            Preprocessed MFCC features ready for model input.
        """
        # Extract MFCC features
        mfcc = self.extract_mfcc(audio)
        
        # Pad or truncate to fixed length
        mfcc_fixed = self.pad_or_truncate(mfcc)
        
        return mfcc_fixed
        
    def synthesize_audio_event(
        self,
        event_type: str,
        duration: float = 1.0,
        noise_level: float = 0.1
    ) -> np.ndarray:
        """Synthesize audio event for testing.
        
        Args:
            event_type: Type of audio event ('clap', 'glass_break', 'noise').
            duration: Duration in seconds.
            noise_level: Background noise level.
            
        Returns:
            Synthesized audio signal.
        """
        n_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, n_samples)
        
        if event_type == "clap":
            # Sharp transient with exponential decay
            signal = np.exp(-t * 10) * np.sin(2 * np.pi * 500 * t)
        elif event_type == "glass_break":
            # High-frequency burst with multiple harmonics
            signal = np.sin(2 * np.pi * 1500 * t) + 0.5 * np.sin(2 * np.pi * 3000 * t)
            signal *= np.exp(-t * 5)
        elif event_type == "noise":
            # Random noise
            signal = np.random.normal(0, 0.1, n_samples)
        else:
            raise ValueError(f"Unknown event type: {event_type}")
            
        # Add background noise
        noise = np.random.normal(0, noise_level, n_samples)
        signal += noise
        
        # Normalize
        signal = signal / np.max(np.abs(signal))
        
        return signal


class AudioDataset:
    """Dataset class for audio event detection."""
    
    def __init__(self, audio_processor: AudioProcessor):
        """Initialize dataset.
        
        Args:
            audio_processor: Audio processor instance.
        """
        self.audio_processor = audio_processor
        self.classes = ["clap", "glass_break", "noise"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
    def generate_synthetic_data(
        self,
        n_samples_per_class: int = 100,
        duration: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic audio event data.
        
        Args:
            n_samples_per_class: Number of samples per class.
            duration: Duration of each sample in seconds.
            
        Returns:
            Tuple of (features, labels) arrays.
        """
        features = []
        labels = []
        
        for class_name in self.classes:
            for _ in range(n_samples_per_class):
                # Generate synthetic audio
                audio = self.audio_processor.synthesize_audio_event(
                    class_name, duration=duration
                )
                
                # Extract features
                mfcc = self.audio_processor.preprocess_audio(audio)
                features.append(mfcc)
                labels.append(self.class_to_idx[class_name])
                
        return np.array(features), np.array(labels)
        
    def load_real_data(self, data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load real audio data from directory.
        
        Args:
            data_dir: Directory containing audio files organized by class.
            
        Returns:
            Tuple of (features, labels) arrays.
        """
        # This would be implemented to load real audio files
        # For now, return synthetic data
        logger.warning("Real data loading not implemented, using synthetic data")
        return self.generate_synthetic_data()
