"""Model export and deployment pipeline for edge devices."""

import torch
import torch.onnx
import tensorflow as tf
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
import os
import json
from pathlib import Path

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("ONNX not available. Install onnx and onnxruntime for ONNX export.")

try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False
    logging.warning("CoreML not available. Install coremltools for CoreML export.")

logger = logging.getLogger(__name__)


class ModelExporter:
    """Export PyTorch models to various edge deployment formats."""
    
    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        """Initialize model exporter.
        
        Args:
            model: PyTorch model to export.
            device: Device to use for export.
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
    def export_to_onnx(
        self,
        output_path: str,
        input_shape: Tuple[int, ...],
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        opset_version: int = 11
    ) -> Dict[str, Any]:
        """Export model to ONNX format.
        
        Args:
            output_path: Path to save ONNX model.
            input_shape: Input tensor shape (batch_size, ...).
            input_names: Names of input tensors.
            output_names: Names of output tensors.
            opset_version: ONNX opset version.
            
        Returns:
            Export metadata.
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX not available. Install onnx and onnxruntime.")
            
        logger.info(f"Exporting model to ONNX: {output_path}")
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Default names
        if input_names is None:
            input_names = ["input"]
        if output_names is None:
            output_names = ["output"]
            
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                input_names[0]: {0: "batch_size"},
                output_names[0]: {0: "batch_size"}
            }
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        # Get model size
        model_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        
        metadata = {
            "format": "ONNX",
            "path": output_path,
            "size_mb": model_size,
            "opset_version": opset_version,
            "input_shape": input_shape,
            "input_names": input_names,
            "output_names": output_names
        }
        
        logger.info(f"ONNX export completed. Model size: {model_size:.2f} MB")
        return metadata
        
    def export_to_tflite(
        self,
        output_path: str,
        input_shape: Tuple[int, ...],
        quantize: bool = True
    ) -> Dict[str, Any]:
        """Export model to TensorFlow Lite format.
        
        Args:
            output_path: Path to save TFLite model.
            input_shape: Input tensor shape.
            quantize: Whether to apply quantization.
            
        Returns:
            Export metadata.
        """
        logger.info(f"Exporting model to TFLite: {output_path}")
        
        # Convert PyTorch model to TensorFlow
        tf_model = self._pytorch_to_tensorflow(input_shape)
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
        
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            
        tflite_model = converter.convert()
        
        # Save model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
            
        # Get model size
        model_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        
        metadata = {
            "format": "TFLite",
            "path": output_path,
            "size_mb": model_size,
            "quantized": quantize,
            "input_shape": input_shape
        }
        
        logger.info(f"TFLite export completed. Model size: {model_size:.2f} MB")
        return metadata
        
    def export_to_coreml(
        self,
        output_path: str,
        input_shape: Tuple[int, ...],
        input_name: str = "audio_input",
        output_name: str = "class_output"
    ) -> Dict[str, Any]:
        """Export model to CoreML format.
        
        Args:
            output_path: Path to save CoreML model.
            input_shape: Input tensor shape.
            input_name: Name of input feature.
            output_name: Name of output feature.
            
        Returns:
            Export metadata.
        """
        if not COREML_AVAILABLE:
            raise ImportError("CoreML not available. Install coremltools.")
            
        logger.info(f"Exporting model to CoreML: {output_path}")
        
        # Convert PyTorch model to TensorFlow
        tf_model = self._pytorch_to_tensorflow(input_shape)
        
        # Convert to CoreML
        coreml_model = ct.convert(
            tf_model,
            inputs=[ct.TensorType(shape=input_shape, name=input_name)]
        )
        
        # Add metadata
        coreml_model.short_description = "Audio Event Detection Model"
        coreml_model.author = "Edge AI Team"
        coreml_model.license = "MIT"
        
        # Save model
        coreml_model.save(output_path)
        
        # Get model size
        model_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        
        metadata = {
            "format": "CoreML",
            "path": output_path,
            "size_mb": model_size,
            "input_shape": input_shape,
            "input_name": input_name,
            "output_name": output_name
        }
        
        logger.info(f"CoreML export completed. Model size: {model_size:.2f} MB")
        return metadata
        
    def _pytorch_to_tensorflow(self, input_shape: Tuple[int, ...]) -> tf.keras.Model:
        """Convert PyTorch model to TensorFlow Keras model.
        
        Args:
            input_shape: Input tensor shape.
            
        Returns:
            TensorFlow Keras model.
        """
        # This is a simplified conversion - in practice, you'd need more sophisticated conversion
        # For now, we'll create a simple TensorFlow equivalent
        
        inputs = tf.keras.Input(shape=input_shape[1:])  # Remove batch dimension
        
        # Simple CNN equivalent
        x = tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same')(inputs)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.GlobalMaxPooling1D()(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs)
        return model


class EdgeRuntime:
    """Runtime for running exported models on edge devices."""
    
    def __init__(self, model_path: str, model_format: str):
        """Initialize edge runtime.
        
        Args:
            model_path: Path to exported model.
            model_format: Format of the model ('onnx', 'tflite', 'coreml').
        """
        self.model_path = model_path
        self.model_format = model_format.lower()
        self.model = None
        self._load_model()
        
    def _load_model(self) -> None:
        """Load the exported model."""
        if self.model_format == "onnx":
            self._load_onnx_model()
        elif self.model_format == "tflite":
            self._load_tflite_model()
        elif self.model_format == "coreml":
            self._load_coreml_model()
        else:
            raise ValueError(f"Unsupported model format: {self.model_format}")
            
    def _load_onnx_model(self) -> None:
        """Load ONNX model."""
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX not available.")
            
        self.model = ort.InferenceSession(self.model_path)
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name
        
    def _load_tflite_model(self) -> None:
        """Load TFLite model."""
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
    def _load_coreml_model(self) -> None:
        """Load CoreML model."""
        if not COREML_AVAILABLE:
            raise ImportError("CoreML not available.")
            
        self.model = ct.models.MLModel(self.model_path)
        
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference on input data.
        
        Args:
            input_data: Input data array.
            
        Returns:
            Prediction results.
        """
        if self.model_format == "onnx":
            return self._predict_onnx(input_data)
        elif self.model_format == "tflite":
            return self._predict_tflite(input_data)
        elif self.model_format == "coreml":
            return self._predict_coreml(input_data)
        else:
            raise ValueError(f"Unsupported model format: {self.model_format}")
            
    def _predict_onnx(self, input_data: np.ndarray) -> np.ndarray:
        """Run ONNX inference."""
        input_data = input_data.astype(np.float32)
        outputs = self.model.run([self.output_name], {self.input_name: input_data})
        return outputs[0]
        
    def _predict_tflite(self, input_data: np.ndarray) -> np.ndarray:
        """Run TFLite inference."""
        input_data = input_data.astype(np.float32)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data
        
    def _predict_coreml(self, input_data: np.ndarray) -> np.ndarray:
        """Run CoreML inference."""
        # Convert input to CoreML format
        input_dict = {"audio_input": input_data}
        
        # Run prediction
        prediction = self.model.predict(input_dict)
        
        # Extract output
        output = prediction["class_output"]
        return output


class DeploymentManager:
    """Manage model deployment to various edge targets."""
    
    def __init__(self, output_dir: str = "deployed_models"):
        """Initialize deployment manager.
        
        Args:
            output_dir: Directory to save deployed models.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def deploy_model(
        self,
        model: torch.nn.Module,
        model_name: str,
        input_shape: Tuple[int, ...],
        target_formats: List[str] = ["onnx", "tflite"],
        device_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Deploy model to multiple edge formats.
        
        Args:
            model: PyTorch model to deploy.
            model_name: Name for the deployed model.
            input_shape: Input tensor shape.
            target_formats: List of target formats to export.
            device_config: Device-specific configuration.
            
        Returns:
            Deployment metadata.
        """
        logger.info(f"Deploying model '{model_name}' to formats: {target_formats}")
        
        exporter = ModelExporter(model)
        deployment_info = {
            "model_name": model_name,
            "input_shape": input_shape,
            "target_formats": target_formats,
            "exports": {},
            "device_config": device_config or {}
        }
        
        # Create model directory
        model_dir = self.output_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        # Export to each format
        for format_name in target_formats:
            try:
                output_path = model_dir / f"{model_name}.{format_name}"
                
                if format_name == "onnx":
                    metadata = exporter.export_to_onnx(str(output_path), input_shape)
                elif format_name == "tflite":
                    metadata = exporter.export_to_tflite(str(output_path), input_shape)
                elif format_name == "coreml":
                    metadata = exporter.export_to_coreml(str(output_path), input_shape)
                else:
                    logger.warning(f"Unsupported format: {format_name}")
                    continue
                    
                deployment_info["exports"][format_name] = metadata
                logger.info(f"Successfully exported to {format_name}")
                
            except Exception as e:
                logger.error(f"Failed to export to {format_name}: {e}")
                deployment_info["exports"][format_name] = {"error": str(e)}
                
        # Save deployment info
        info_path = model_dir / "deployment_info.json"
        with open(info_path, 'w') as f:
            json.dump(deployment_info, f, indent=2)
            
        logger.info(f"Deployment completed. Models saved to: {model_dir}")
        return deployment_info
        
    def create_device_configs(self) -> Dict[str, Dict[str, Any]]:
        """Create device-specific configurations.
        
        Returns:
            Dictionary of device configurations.
        """
        configs = {
            "raspberry_pi_4": {
                "cpu": "ARM Cortex-A72",
                "memory": "4GB",
                "recommended_formats": ["tflite", "onnx"],
                "max_model_size_mb": 50,
                "target_latency_ms": 100
            },
            "jetson_nano": {
                "cpu": "ARM Cortex-A57",
                "gpu": "Maxwell 128-core",
                "memory": "4GB",
                "recommended_formats": ["onnx", "tflite"],
                "max_model_size_mb": 100,
                "target_latency_ms": 50
            },
            "android_mobile": {
                "cpu": "ARM64",
                "recommended_formats": ["tflite", "coreml"],
                "max_model_size_mb": 20,
                "target_latency_ms": 200
            },
            "ios_mobile": {
                "cpu": "ARM64",
                "recommended_formats": ["coreml"],
                "max_model_size_mb": 20,
                "target_latency_ms": 200
            },
            "mcu_stm32": {
                "cpu": "ARM Cortex-M4",
                "memory": "1MB",
                "recommended_formats": ["tflite"],
                "max_model_size_mb": 1,
                "target_latency_ms": 1000
            }
        }
        
        return configs
