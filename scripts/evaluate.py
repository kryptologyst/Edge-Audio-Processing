"""Evaluation script for edge audio processing models."""

import argparse
import json
import logging
from pathlib import Path
import torch
import numpy as np
from sklearn.model_selection import train_test_split

from src.utils.core import set_deterministic_seed, get_device
from src.utils.audio import AudioProcessor, AudioDataset
from src.models.audio_models import create_model
from src.pipelines.training import ModelEvaluator, create_data_loaders
from src.export.deployment import EdgeRuntime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model_performance(model_path: str, model_format: str = "onnx"):
    """Evaluate model performance on edge runtime."""
    
    # Load edge runtime
    runtime = EdgeRuntime(model_path, model_format)
    
    # Generate test data
    audio_processor = AudioProcessor()
    dataset = AudioDataset(audio_processor)
    X_test, y_test = dataset.generate_synthetic_data(n_samples_per_class=50)
    
    # Preprocess test data
    X_test_processed = np.array([audio_processor.preprocess_audio(audio) for audio in X_test])
    
    # Run inference
    predictions = []
    latencies = []
    
    for i, features in enumerate(X_test_processed):
        import time
        start_time = time.time()
        pred = runtime.predict(features.reshape(1, -1))
        latency = time.time() - start_time
        
        predictions.append(np.argmax(pred))
        latencies.append(latency)
        
        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1}/{len(X_test_processed)} samples")
    
    # Calculate metrics
    accuracy = np.mean(np.array(predictions) == y_test)
    avg_latency = np.mean(latencies)
    latency_p95 = np.percentile(latencies, 95)
    
    metrics = {
        "accuracy": accuracy,
        "avg_latency_ms": avg_latency * 1000,
        "latency_p95_ms": latency_p95 * 1000,
        "throughput_fps": 1.0 / avg_latency,
        "total_samples": len(X_test)
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate edge audio processing models")
    parser.add_argument("--model_path", type=str, required=True, help="Path to exported model")
    parser.add_argument("--format", type=str, default="onnx", choices=["onnx", "tflite", "coreml"], help="Model format")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output file for results")
    
    args = parser.parse_args()
    
    logger.info(f"Evaluating model: {args.model_path}")
    logger.info(f"Model format: {args.format}")
    
    try:
        metrics = evaluate_model_performance(args.model_path, args.format)
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        logger.info(f"Evaluation completed. Results saved to: {args.output}")
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Average Latency: {metrics['avg_latency_ms']:.2f} ms")
        print(f"Latency (P95): {metrics['latency_p95_ms']:.2f} ms")
        print(f"Throughput: {metrics['throughput_fps']:.1f} FPS")
        print(f"Total Samples: {metrics['total_samples']}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
