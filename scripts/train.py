"""Main training script for edge audio processing."""

import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import torch
import numpy as np
from pathlib import Path
import json
import time

from src.utils.core import set_deterministic_seed, get_device
from src.utils.audio import AudioProcessor, AudioDataset
from src.models.audio_models import create_model
from src.models.compression import ModelCompressor, KnowledgeDistillation
from src.pipelines.training import AudioTrainer, ModelEvaluator, create_data_loaders
from src.export.deployment import DeploymentManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training pipeline."""
    
    # Set random seed for reproducibility
    set_deterministic_seed(cfg.seed)
    
    # Get device
    if cfg.training.device == "auto":
        device = get_device()
    else:
        device = cfg.training.device
        
    logger.info(f"Using device: {device}")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Create output directories
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize audio processor and dataset
    audio_processor = AudioProcessor(
        sample_rate=cfg.data.sample_rate,
        n_mfcc=cfg.model.n_mfcc,
        max_length=cfg.model.input_length
    )
    
    dataset = AudioDataset(audio_processor)
    
    # Generate synthetic data
    logger.info("Generating synthetic audio data...")
    X, y = dataset.generate_synthetic_data(
        n_samples_per_class=cfg.data.n_samples_per_class,
        duration=cfg.data.duration
    )
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.data.test_split, random_state=cfg.seed, stratify=y
    )
    
    logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        X_train, y_train, X_test, y_test,
        batch_size=cfg.training.batch_size
    )
    
    # Create models
    logger.info(f"Creating {cfg.model.type} model...")
    model = create_model(
        model_type=cfg.model.type,
        input_length=cfg.model.input_length,
        n_mfcc=cfg.model.n_mfcc,
        n_classes=cfg.model.n_classes,
        dropout_rate=cfg.model.dropout_rate
    )
    
    # Create tiny model for comparison
    tiny_model = create_model(
        model_type="tiny",
        input_length=cfg.model.input_length,
        n_mfcc=cfg.model.n_mfcc,
        n_classes=cfg.model.n_classes
    )
    
    logger.info(f"Model created. Parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Tiny model created. Parameters: {sum(p.numel() for p in tiny_model.parameters())}")
    
    # Train main model
    logger.info("Training main model...")
    trainer = AudioTrainer(
        model=model,
        device=device,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay
    )
    
    start_time = time.time()
    history = trainer.train(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=cfg.training.epochs
    )
    training_time = time.time() - start_time
    
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Train tiny model
    logger.info("Training tiny model...")
    tiny_trainer = AudioTrainer(
        model=tiny_model,
        device=device,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay
    )
    
    tiny_history = tiny_trainer.train(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=cfg.training.epochs
    )
    
    # Evaluate models
    logger.info("Evaluating models...")
    evaluator = ModelEvaluator(model, device)
    tiny_evaluator = ModelEvaluator(tiny_model, device)
    
    # Comprehensive evaluation
    main_metrics = evaluator.comprehensive_evaluation(test_loader, dataset.classes)
    tiny_metrics = tiny_evaluator.comprehensive_evaluation(test_loader, dataset.classes)
    
    # Create results summary
    results = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "training_time_seconds": training_time,
        "main_model": {
            "type": cfg.model.type,
            "parameters": sum(p.numel() for p in model.parameters()),
            "metrics": main_metrics
        },
        "tiny_model": {
            "type": "tiny",
            "parameters": sum(p.numel() for p in tiny_model.parameters()),
            "metrics": tiny_metrics
        },
        "class_names": dataset.classes
    }
    
    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"Results saved to: {results_path}")
    
    # Model compression
    if cfg.compression.enabled:
        logger.info("Applying model compression...")
        
        compressor = ModelCompressor(model)
        compressed_model = compressor.prune_model(
            pruning_ratio=cfg.compression.pruning_ratio,
            pruning_type=cfg.compression.pruning_type
        )
        
        # Evaluate compressed model
        compressed_evaluator = ModelEvaluator(compressed_model, device)
        compressed_metrics = compressed_evaluator.comprehensive_evaluation(test_loader, dataset.classes)
        
        results["compressed_model"] = {
            "type": f"{cfg.model.type}_compressed",
            "parameters": sum(p.numel() for p in compressed_model.parameters()),
            "metrics": compressed_metrics
        }
        
        # Update results file
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info("Model compression completed")
    
    # Model export and deployment
    if cfg.export.enabled:
        logger.info("Exporting models for edge deployment...")
        
        deployment_manager = DeploymentManager(cfg.export.output_dir)
        
        # Deploy main model
        main_deployment = deployment_manager.deploy_model(
            model=model,
            model_name=f"{cfg.model.type}_main",
            input_shape=(1, cfg.model.input_length, cfg.model.n_mfcc),
            target_formats=cfg.export.formats
        )
        
        # Deploy tiny model
        tiny_deployment = deployment_manager.deploy_model(
            model=tiny_model,
            model_name="tiny_model",
            input_shape=(1, cfg.model.input_length, cfg.model.n_mfcc),
            target_formats=cfg.export.formats
        )
        
        results["deployments"] = {
            "main_model": main_deployment,
            "tiny_model": tiny_deployment
        }
        
        # Update results file
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info("Model deployment completed")
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Main Model ({cfg.model.type}):")
    print(f"  - Parameters: {results['main_model']['parameters']:,}")
    print(f"  - Accuracy: {results['main_model']['metrics']['accuracy']:.4f}")
    print(f"  - Model Size: {results['main_model']['metrics']['model_size_mb']:.2f} MB")
    print(f"  - Latency (P50): {results['main_model']['metrics']['latency_p50_ms']:.2f} ms")
    
    print(f"\nTiny Model:")
    print(f"  - Parameters: {results['tiny_model']['parameters']:,}")
    print(f"  - Accuracy: {results['tiny_model']['metrics']['accuracy']:.4f}")
    print(f"  - Model Size: {results['tiny_model']['metrics']['model_size_mb']:.2f} MB")
    print(f"  - Latency (P50): {results['tiny_model']['metrics']['latency_p50_ms']:.2f} ms")
    
    if cfg.compression.enabled and "compressed_model" in results:
        print(f"\nCompressed Model:")
        print(f"  - Parameters: {results['compressed_model']['parameters']:,}")
        print(f"  - Accuracy: {results['compressed_model']['metrics']['accuracy']:.4f}")
        print(f"  - Model Size: {results['compressed_model']['metrics']['model_size_mb']:.2f} MB")
        print(f"  - Latency (P50): {results['compressed_model']['metrics']['latency_p50_ms']:.2f} ms")
    
    print(f"\nTraining Time: {training_time:.2f} seconds")
    print(f"Results saved to: {results_path}")
    print("="*60)


if __name__ == "__main__":
    main()
