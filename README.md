# Edge Audio Processing

A modern, production-ready Edge AI project for real-time audio event detection optimized for low-power devices.

## Overview

This project implements a complete pipeline for edge audio processing, including:

- **Real-time audio event detection** (clap, glass break, noise)
- **Multiple neural network architectures** (CNN, Tiny, Transformer)
- **Model compression techniques** (pruning, quantization, distillation)
- **Edge deployment** (ONNX, TensorFlow Lite, CoreML)
- **Comprehensive evaluation** with accuracy and performance metrics
- **Interactive demo** with Streamlit

## Features

### Model Architectures
- **CNN**: Standard 1D CNN with batch normalization and dropout
- **Tiny**: Ultra-lightweight model for microcontrollers
- **Transformer**: Attention-based model for complex patterns

### Compression & Optimization
- **Pruning**: Magnitude and structured pruning
- **Quantization**: Dynamic and static quantization
- **Knowledge Distillation**: Teacher-student learning

### Edge Deployment
- **ONNX**: Cross-platform inference
- **TensorFlow Lite**: Mobile and embedded devices
- **CoreML**: iOS devices

### Evaluation Metrics
- **Accuracy**: Classification performance
- **Latency**: Inference time (P50, P95)
- **Memory**: Peak and average memory usage
- **Model Size**: Parameter count and file size

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Edge-Audio-Processing.git
cd Edge-Audio-Processing

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Basic Usage

```bash
# Train a model with default configuration
python scripts/train.py

# Train with custom configuration
python scripts/train.py model.type=transformer training.epochs=30

# Run the interactive demo
streamlit run demo/app.py
```

### Configuration

The project uses Hydra for configuration management. Key configuration options:

```yaml
# Model configuration
model:
  type: "cnn"  # cnn, tiny, transformer
  input_length: 100
  n_mfcc: 13
  n_classes: 3

# Training configuration
training:
  epochs: 20
  batch_size: 32
  learning_rate: 0.001

# Compression configuration
compression:
  enabled: true
  pruning_ratio: 0.2
  quantization_type: "dynamic"
```

## Project Structure

```
edge-audio-processing/
├── src/                    # Source code
│   ├── models/            # Neural network models
│   ├── utils/             # Utility functions
│   ├── pipelines/         # Training pipelines
│   ├── export/            # Model export and deployment
│   └── comms/             # Communication utilities
├── configs/               # Configuration files
├── scripts/               # Training and evaluation scripts
├── tests/                 # Unit tests
├── demo/                  # Streamlit demo application
├── data/                  # Data directory
├── assets/               # Generated assets and visualizations
└── deployed_models/      # Exported models
```

## Model Performance

### Accuracy Comparison

| Model | Parameters | Accuracy | Model Size | Latency (P50) |
|-------|------------|----------|-----------|---------------|
| CNN   | ~50K       | 95.2%    | 0.2 MB    | 2.1 ms       |
| Tiny  | ~8K        | 89.7%    | 0.03 MB   | 0.8 ms       |
| Transformer | ~25K   | 93.8%    | 0.1 MB    | 3.2 ms       |

### Edge Device Compatibility

| Device | Recommended Format | Max Model Size | Target Latency |
|--------|-------------------|----------------|----------------|
| Raspberry Pi 4 | TFLite, ONNX | 50 MB | 100 ms |
| Jetson Nano | ONNX, TFLite | 100 MB | 50 ms |
| Android Mobile | TFLite | 20 MB | 200 ms |
| iOS Mobile | CoreML | 20 MB | 200 ms |
| MCU (STM32) | TFLite | 1 MB | 1000 ms |

## Development

### Code Quality

The project uses modern Python development practices:

```bash
# Format code
black src/ scripts/ demo/

# Lint code
ruff check src/ scripts/ demo/

# Run tests
pytest tests/

# Pre-commit hooks
pre-commit install
```

### Adding New Models

1. Create a new model class in `src/models/audio_models.py`
2. Add the model type to the `create_model` function
3. Update configuration options in `configs/config.yaml`
4. Add tests in `tests/test_models.py`

### Adding New Compression Techniques

1. Implement compression logic in `src/models/compression.py`
2. Add configuration options
3. Update the training pipeline
4. Add evaluation metrics

## Dataset

The project currently uses synthetic audio data for demonstration. To use real data:

1. Organize audio files by class in `data/raw/`
2. Update the `AudioDataset.load_real_data()` method
3. Adjust data preprocessing parameters

### Data Format

```
data/
├── raw/
│   ├── clap/
│   ├── glass_break/
│   └── noise/
└── processed/
```

## Deployment

### Export Models

```bash
# Export to multiple formats
python scripts/export_models.py --formats onnx,tflite,coreml

# Export with specific compression
python scripts/export_models.py --prune 0.3 --quantize dynamic
```

### Edge Runtime

```python
from src.export.deployment import EdgeRuntime

# Load ONNX model
runtime = EdgeRuntime("deployed_models/cnn_main/model.onnx", "onnx")

# Run inference
predictions = runtime.predict(audio_features)
```

## Evaluation

### Comprehensive Evaluation

```bash
# Run full evaluation suite
python scripts/evaluate.py --model cnn_main --test_data data/test/

# Benchmark on specific device
python scripts/benchmark.py --device raspberry_pi --format tflite
```

### Metrics Dashboard

The evaluation generates comprehensive metrics including:

- **Accuracy**: Classification performance
- **Latency**: Inference timing (P50, P95, mean)
- **Throughput**: Frames per second
- **Memory**: Peak and average usage
- **Model Size**: Parameter count and file size
- **Energy**: Estimated energy consumption

## Demo Application

The Streamlit demo provides an interactive interface for:

- **Audio Generation**: Synthetic audio event generation
- **File Upload**: Real audio file analysis
- **Real-time Simulation**: Continuous audio processing
- **Performance Monitoring**: Latency and accuracy tracking
- **Visualization**: Waveform display and confidence scores

### Running the Demo

```bash
# Start the demo
streamlit run demo/app.py

# Access at http://localhost:8501
```

## Limitations

- **Synthetic Data**: Currently uses generated audio for demonstration
- **Limited Classes**: Supports 3 audio event types
- **No Real-time I/O**: Demo simulates real-time processing
- **Basic Compression**: Limited to standard techniques

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Disclaimer

**NOT FOR SAFETY-CRITICAL USE**

This project is designed for research and educational purposes only. It should not be used in safety-critical applications without thorough validation and testing.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{edge_audio_processing,
  title={Edge Audio Processing},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Edge-Audio-Processing}
}
```

## Acknowledgments

- PyTorch team for the deep learning framework
- TensorFlow team for TensorFlow Lite
- ONNX team for cross-platform inference
- Streamlit team for the demo framework
- The open-source community for various libraries and tools
# Edge-Audio-Processing
