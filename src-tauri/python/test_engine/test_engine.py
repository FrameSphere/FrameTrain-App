"""
FrameTrain v2 - Universal Testing Engine (Prototype)
====================================================
Modular, extensible testing engine supporting multiple modalities:
- Text (NLP, LLMs)
- Vision (Image Classification, Detection, Segmentation)
- Audio (Speech Recognition, TTS, Audio Classification)
- Multi-Modal (CLIP, Stable Diffusion)

Architecture:
- Plugin-based system for extensibility
- Registry pattern for test loaders
- Abstract base classes for different modalities
- Task-specific metrics

Communication: JSON messages via stdout to Rust backend
"""

import os
import sys
import json
import time
import argparse
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import importlib.util


# Add current directory to path for plugin_dependency_manager
try:
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    from plugin_dependency_manager import DependencyManager, PluginManifest
    DEPENDENCY_MANAGER_AVAILABLE = True
except ImportError:
    DEPENDENCY_MANAGER_AVAILABLE = False


# ============================================================================
# COMMUNICATION PROTOCOL
# ============================================================================

class MessageType(Enum):
    """Types of messages sent to backend"""
    PROGRESS = "progress"
    STATUS = "status"
    ERROR = "error"
    WARNING = "warning"
    COMPLETE = "complete"
    DEBUG = "debug"


class MessageProtocol:
    """Handles all communication with Rust backend via stdout"""
    
    @staticmethod
    def send(msg_type: MessageType, data: Dict[str, Any]):
        """Send JSON message to backend"""
        message = {
            "type": msg_type.value,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        print(json.dumps(message), flush=True)
    
    @staticmethod
    def progress(current: int, total: int, samples_per_second: float = 0.0):
        """Send progress update"""
        remaining = total - current
        eta = remaining / samples_per_second if samples_per_second > 0 else None
        
        MessageProtocol.send(MessageType.PROGRESS, {
            "current_sample": current,
            "total_samples": total,
            "progress_percent": (current / total) * 100 if total > 0 else 0,
            "samples_per_second": samples_per_second,
            "estimated_time_remaining": eta
        })
    
    @staticmethod
    def status(status: str, message: str = ""):
        """Send status update"""
        MessageProtocol.send(MessageType.STATUS, {
            "status": status,
            "message": message
        })
    
    @staticmethod
    def error(error: str, details: str = ""):
        """Send error message"""
        MessageProtocol.send(MessageType.ERROR, {
            "error": error,
            "details": details
        })
    
    @staticmethod
    def warning(message: str):
        """Send warning message"""
        MessageProtocol.send(MessageType.WARNING, {
            "message": message
        })
    
    @staticmethod
    def complete(results: Dict[str, Any]):
        """Send test complete notification"""
        MessageProtocol.send(MessageType.COMPLETE, results)
    
    @staticmethod
    def debug(message: str, data: Any = None):
        """Send debug message (only in debug mode)"""
        if os.getenv("FRAMETRAIN_DEBUG"):
            MessageProtocol.send(MessageType.DEBUG, {
                "message": message,
                "data": data
            })


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class TestConfig:
    """Test configuration"""
    model_path: str = ""
    dataset_path: str = ""
    output_path: str = ""
    batch_size: int = 8
    max_samples: Optional[int] = None
    model_type: str = "auto"  # auto, transformers, vision, audio, etc.
    task_type: str = "auto"   # auto, classification, detection, etc.
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TestConfig':
        """Create config from dictionary"""
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in known_fields}
        return cls(**filtered_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


# ============================================================================
# MODALITY DETECTION
# ============================================================================

class Modality(Enum):
    """Supported data modalities"""
    TEXT = "text"
    VISION = "vision"
    AUDIO = "audio"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    MULTIMODAL = "multimodal"
    TABULAR = "tabular"
    TIMESERIES = "timeseries"
    GRAPH = "graph"
    RL = "rl"
    UNKNOWN = "unknown"


class ModalityDetector:
    """Automatically detects data modality from model or dataset"""
    
    @staticmethod
    def detect_from_model(model_path: str) -> Tuple[Modality, Dict[str, Any]]:
        """Detect modality from model path/structure"""
        model_path = Path(model_path)
        
        metadata = {"source": "model_path"}
        
        # Check for config files that indicate model type
        config_file = model_path / "config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # Check for vision models
                if "image_size" in config or "vision" in str(config).lower():
                    metadata["detected_from"] = "config.json"
                    return Modality.VISION, metadata
                
                # Check for detection models
                if "num_classes" in config and "detection" in str(config).lower():
                    metadata["detected_from"] = "config.json"
                    return Modality.DETECTION, metadata
                
            except:
                pass
        
        # Check model path name
        path_lower = str(model_path).lower()
        if any(x in path_lower for x in ['resnet', 'vit', 'efficientnet', 'mobilenet']):
            metadata["detected_from"] = "path_name"
            return Modality.VISION, metadata
        
        if any(x in path_lower for x in ['yolo', 'rcnn', 'detection', 'detector']):
            metadata["detected_from"] = "path_name"
            return Modality.DETECTION, metadata
        
        if any(x in path_lower for x in ['whisper', 'wav2vec', 'audio']):
            metadata["detected_from"] = "path_name"
            return Modality.AUDIO, metadata
        
        # Default to text
        metadata["detected_from"] = "default"
        return Modality.TEXT, metadata
    
    @staticmethod
    def detect_from_dataset(dataset_path: str) -> Tuple[Modality, Dict[str, Any]]:
        """Detect modality from dataset structure"""
        dataset_path = Path(dataset_path)
        
        # Check test directory
        test_path = dataset_path / "test"
        if not test_path.exists():
            test_path = dataset_path / "val"
        if not test_path.exists():
            return Modality.UNKNOWN, {"error": "No test/val directory found"}
        
        # Get file extensions
        files = list(test_path.glob("*"))
        extensions = {f.suffix.lower() for f in files if f.is_file()}
        
        metadata = {
            "total_files": len(files),
            "extensions": list(extensions),
            "sample_files": [f.name for f in files[:5]]
        }
        
        # Detection logic
        if extensions & {'.txt', '.json', '.jsonl', '.csv'}:
            return Modality.TEXT, metadata
        
        elif extensions & {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.webp'}:
            # Check for detection annotations
            if (test_path / "labels").exists() or any(f.suffix == '.txt' for f in files):
                return Modality.DETECTION, metadata
            return Modality.VISION, metadata
        
        elif extensions & {'.wav', '.mp3', '.flac', '.ogg'}:
            return Modality.AUDIO, metadata
        
        return Modality.UNKNOWN, metadata


# ============================================================================
# ABSTRACT BASE CLASSES
# ============================================================================

class BaseTestLoader(ABC):
    """Abstract base class for test loaders"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.model = None
        self.device = None
        self.test_data = None
    
    @abstractmethod
    def load_model(self):
        """Load the model for testing"""
        pass
    
    @abstractmethod
    def load_test_data(self) -> List[Any]:
        """Load test dataset"""
        pass
    
    @abstractmethod
    def test_sample(self, sample: Any) -> Dict[str, Any]:
        """Test a single sample and return results"""
        pass
    
    @abstractmethod
    def compute_metrics(self, all_results: List[Dict]) -> Dict[str, Any]:
        """Compute final metrics from all test results"""
        pass
    
    def get_device(self):
        """Get the best available device"""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        except:
            return "cpu"


# ============================================================================
# REGISTRY SYSTEM
# ============================================================================

class TestRegistry:
    """Registry for test loaders"""
    
    def __init__(self):
        self._test_loaders = {}
    
    def register_test_loader(self, modality: Modality, loader_class: type):
        """Register a test loader for a modality"""
        self._test_loaders[modality] = loader_class
    
    def get_test_loader(self, modality: Modality) -> Optional[type]:
        """Get test loader class for modality"""
        return self._test_loaders.get(modality)
    
    def list_supported(self) -> List[str]:
        """List all supported modalities"""
        return [m.value for m in self._test_loaders.keys()]


# Global registry instance
TEST_REGISTRY = TestRegistry()


# ============================================================================
# TEXT TEST LOADER (Built-in)
# ============================================================================

class TextTestLoader(BaseTestLoader):
    """Test loader for text/NLP models"""
    
    def __init__(self, config: TestConfig):
        super().__init__(config)
        self.tokenizer = None
        self.model_class = None
    
    def load_model(self):
        """Load transformers model"""
        try:
            import torch
            from transformers import (
                AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
                AutoModelForSequenceClassification, AutoTokenizer
            )
            
            MessageProtocol.status("loading", f"Loading model from {self.config.model_path}...")
            
            self.device = self.get_device()
            MessageProtocol.status("device", f"Using device: {self.device}")
            
            model_path = Path(self.config.model_path)
            
            # Check if tokenizer files exist
            tokenizer_files = ['tokenizer_config.json', 'tokenizer.json', 'vocab.json', 'special_tokens_map.json']
            has_tokenizer = any((model_path / f).exists() for f in tokenizer_files)
            
            # If no tokenizer, try to find original model info
            tokenizer_path = self.config.model_path
            if not has_tokenizer:
                # Check for tokenizer_info.txt (created by train engine)
                tokenizer_info_file = model_path / "tokenizer_info.txt"
                if tokenizer_info_file.exists():
                    MessageProtocol.status("loading", "Found tokenizer_info.txt, reading original model path...")
                    with open(tokenizer_info_file, 'r') as f:
                        for line in f:
                            if line.startswith("Original model:"):
                                original_model = line.split(":", 1)[1].strip()
                                # Check if it's a HuggingFace model ID (not a local path)
                                if not original_model.startswith('/') and not original_model.startswith('.'):
                                    MessageProtocol.status("loading", f"Using tokenizer from original model: {original_model}")
                                    tokenizer_path = original_model
                                    break
                
                # If still no luck, check for original_model_id in config
                if tokenizer_path == self.config.model_path:
                    config_file = model_path / "config.json"
                    if config_file.exists():
                        with open(config_file, 'r') as f:
                            config_data = json.load(f)
                            original_model = config_data.get('_name_or_path') or config_data.get('original_model_id')
                            if original_model and not original_model.startswith('/'):
                                MessageProtocol.status("loading", f"Using tokenizer from config: {original_model}")
                                tokenizer_path = original_model
                            else:
                                MessageProtocol.warning(f"No tokenizer files found and no original model ID. Will try to load anyway.")
            
            # Load config
            model_config = AutoConfig.from_pretrained(self.config.model_path)
            architectures = getattr(model_config, 'architectures', [])
            
            # Determine model class
            if architectures:
                arch_name = architectures[0]
                if any(x in arch_name for x in ['T5', 'MT5', 'Bart', 'Pegasus']):
                    self.model_class = 'seq2seq'
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path)
                elif 'SequenceClassification' in arch_name:
                    self.model_class = 'classification'
                    self.model = AutoModelForSequenceClassification.from_pretrained(self.config.model_path)
                else:
                    self.model_class = 'causal_lm'
                    self.model = AutoModelForCausalLM.from_pretrained(self.config.model_path)
            else:
                self.model_class = 'causal_lm'
                self.model = AutoModelForCausalLM.from_pretrained(self.config.model_path)
            
            self.model.to(self.device)
            self.model.eval()
            
            # Load tokenizer from appropriate path
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            except Exception as e:
                MessageProtocol.error(f"Failed to load tokenizer from {tokenizer_path}", str(e))
                # Last resort: try model name from architectures
                if architectures:
                    try:
                        base_model = architectures[0].replace('ForSeq2SeqLM', '').replace('ForCausalLM', '').replace('ForSequenceClassification', '')
                        MessageProtocol.status("loading", f"Trying base model tokenizer: {base_model}")
                        self.tokenizer = AutoTokenizer.from_pretrained(base_model.lower())
                    except:
                        raise e
                else:
                    raise e
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            MessageProtocol.status("loaded", f"Model loaded ({self.model_class})")
            
        except Exception as e:
            MessageProtocol.error("Failed to load model", str(e))
            raise
    
    def load_test_data(self) -> List[Dict[str, str]]:
        """Load text test data"""
        MessageProtocol.status("loading", "Loading test data...")
        
        test_path = Path(self.config.dataset_path) / "test"
        if not test_path.exists():
            test_path = Path(self.config.dataset_path) / "val"
        if not test_path.exists():
            raise ValueError(f"Test data not found: {self.config.dataset_path}")
        
        test_data = []
        
        # Load different file formats
        for file_path in test_path.glob("*"):
            if file_path.suffix == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            test_data.append({"text": line, "expected": None})
            
            elif file_path.suffix == '.csv':
                import csv
                with open(file_path, 'r', encoding='utf-8') as f:
                    csv_reader = csv.reader(f)
                    for row in csv_reader:
                        if len(row) >= 2:
                            test_data.append({"text": row[0].strip(), "expected": row[1].strip()})
                        elif len(row) == 1:
                            test_data.append({"text": row[0].strip(), "expected": None})
            
            elif file_path.suffix == '.jsonl':
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            text = data.get('input') or data.get('text') or data.get('prompt') or ''
                            expected = data.get('target') or data.get('output') or data.get('label')
                            test_data.append({"text": text, "expected": expected})
                        except:
                            pass
            
            elif file_path.suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                text = item.get('input') or item.get('text') or item.get('prompt') or ''
                                expected = item.get('target') or item.get('output') or item.get('label')
                                test_data.append({"text": text, "expected": expected})
                    except:
                        pass
        
        if not test_data:
            raise ValueError("No test data found")
        
        # Limit samples
        if self.config.max_samples and len(test_data) > self.config.max_samples:
            test_data = test_data[:self.config.max_samples]
        
        MessageProtocol.status("loaded", f"Loaded {len(test_data)} test samples")
        return test_data
    
    def test_sample(self, sample: Dict[str, str]) -> Dict[str, Any]:
        """Test a single text sample"""
        import torch
        
        text = sample["text"]
        expected = sample.get("expected")
        
        start_time = time.time()
        
        with torch.no_grad():
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Generate prediction
            if self.model_class == 'seq2seq':
                outputs = self.model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True
                )
                predicted = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Calculate loss if expected available
                loss = None
                if expected:
                    with self.tokenizer.as_target_tokenizer():
                        labels = self.tokenizer(expected, return_tensors="pt", truncation=True, max_length=512).input_ids.to(self.device)
                    outputs_with_loss = self.model(**inputs, labels=labels)
                    loss = outputs_with_loss.loss.item()
            
            elif self.model_class == 'classification':
                outputs = self.model(**inputs)
                predicted_label = torch.argmax(outputs.logits, dim=-1).item()
                predicted = str(predicted_label)
                
                # Calculate loss
                loss = None
                if expected is not None:
                    try:
                        labels = torch.tensor([int(expected)]).to(self.device)
                        outputs_with_loss = self.model(**inputs, labels=labels)
                        loss = outputs_with_loss.loss.item()
                    except:
                        pass
            
            else:  # causal_lm
                outputs = self.model.generate(
                    **inputs,
                    max_length=len(inputs.input_ids[0]) + 100,
                    num_beams=1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                predicted = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Remove input from output
                if predicted.startswith(text):
                    predicted = predicted[len(text):].strip()
                
                loss = None
        
        inference_time = time.time() - start_time
        
        # Check correctness
        is_correct = False
        if expected:
            pred_norm = predicted.strip().lower()
            exp_norm = expected.strip().lower()
            is_correct = pred_norm == exp_norm
        
        # CRITICAL FIX: Return with correct field names for Rust deserialization
        return {
            "input_text": text,  # ADD: Input text for Rust
            "predicted_output": predicted,  # FIX: Rename to match Rust struct
            "expected_output": expected,  # FIX: Rename to match Rust struct
            "is_correct": is_correct,
            "loss": loss,
            "confidence": None,  # ADD: Optional confidence field
            "inference_time": inference_time,
            "error_type": None  # ADD: Optional error_type field
        }
    
    def compute_metrics(self, all_results: List[Dict]) -> Dict[str, Any]:
        """Compute text metrics"""
        total = len(all_results)
        correct = sum(1 for r in all_results if r.get("is_correct"))
        
        losses = [r["loss"] for r in all_results if r.get("loss") is not None]
        avg_loss = sum(losses) / len(losses) if losses else None
        
        inference_times = [r["inference_time"] for r in all_results]
        avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
        
        return {
            "accuracy": (correct / total * 100) if total > 0 else 0,
            "total_samples": total,
            "correct_predictions": correct,
            "incorrect_predictions": total - correct,
            "average_loss": avg_loss,
            "average_inference_time": avg_inference_time
        }


# Register text test loader
TEST_REGISTRY.register_test_loader(Modality.TEXT, TextTestLoader)


# ============================================================================
# PLUGIN LOADER
# ============================================================================

class PluginLoader:
    """Load test plugins with dependency management"""
    
    def __init__(self, plugin_dir: str):
        self.plugin_dir = Path(plugin_dir)
        self.dep_manager = DependencyManager() if DEPENDENCY_MANAGER_AVAILABLE else None
        self.loaded_plugins = {}
    
    def load_plugin(self, plugin_file: str, auto_install: bool = None) -> Optional[Any]:
        """Load a single plugin with dependency management"""
        plugin_path = self.plugin_dir / plugin_file
        
        if not plugin_path.exists():
            MessageProtocol.warning(f"Plugin not found: {plugin_path}")
            return None
        
        # Dependency management
        if DEPENDENCY_MANAGER_AVAILABLE and self.dep_manager:
            try:
                manifest = PluginManifest.from_docstring(plugin_path)
                if manifest:
                    missing_req, _ = self.dep_manager.check_dependencies(manifest)
                    
                    if missing_req:
                        MessageProtocol.status("missing_deps", f"Plugin requires: {', '.join(missing_req)}")
                        
                        if auto_install is None:
                            auto_install = os.getenv("FRAMETRAIN_AUTO_INSTALL", "false").lower() == "true"
                        
                        success = self.dep_manager.install_dependencies(manifest, auto_install=auto_install)
                        if not success:
                            MessageProtocol.error(f"Failed to install dependencies for {manifest.plugin_name}", "")
                            return None
            except Exception as e:
                MessageProtocol.debug(f"Could not check dependencies: {e}")
        
        # Load plugin
        try:
            spec = importlib.util.spec_from_file_location(plugin_path.stem, plugin_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            self.loaded_plugins[plugin_path.name] = module
            MessageProtocol.debug(f"✓ Loaded plugin: {plugin_path.name}")
            return module
        except Exception as e:
            MessageProtocol.error(f"Failed to load plugin {plugin_path.name}", str(e))
            return None
    
    def load_all_plugins(self, auto_install: bool = None):
        """Load all plugins from directory"""
        if not self.plugin_dir.exists():
            return
        
        for plugin_file in self.plugin_dir.glob("plugin_*.py"):
            self.load_plugin(plugin_file.name, auto_install=auto_install)


# ============================================================================
# TEST ENGINE
# ============================================================================

class TestEngine:
    """Main test engine that orchestrates testing"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.modality = None
        self.test_loader = None
        self.plugin_loader = None
    
    def _ensure_plugin_loaded(self, modality: Modality):
        """Ensure appropriate plugin is loaded"""
        plugin_map = {
            Modality.TEXT: None,  # Built-in
            Modality.VISION: "plugin_vision_test.py",
            Modality.AUDIO: "plugin_audio_test.py",
            Modality.DETECTION: "plugin_detection_test.py",
            Modality.SEGMENTATION: "plugin_segmentation_test.py",
        }
        
        plugin_file = plugin_map.get(modality)
        
        if plugin_file is None:
            MessageProtocol.debug(f"Modality {modality.value} uses built-in support")
            return
        
        if TEST_REGISTRY.get_test_loader(modality) is not None:
            MessageProtocol.debug(f"Plugin for {modality.value} already loaded")
            return
        
        # Load plugin
        MessageProtocol.status("loading_plugin", f"Loading test plugin for {modality.value}...")
        
        if self.plugin_loader is None:
            plugin_dir = Path(__file__).parent / "plugins"
            self.plugin_loader = PluginLoader(str(plugin_dir))
        
        auto_install = os.getenv("FRAMETRAIN_AUTO_INSTALL", "false").lower() == "true"
        module = self.plugin_loader.load_plugin(plugin_file, auto_install=auto_install)
        
        if module is None:
            raise RuntimeError(
                f"Failed to load required plugin: {plugin_file}\n"
                f"This plugin is needed for {modality.value} testing."
            )
        
        if TEST_REGISTRY.get_test_loader(modality) is None:
            raise RuntimeError(f"Plugin {plugin_file} loaded but did not register")
    
    def detect_and_setup(self):
        """Detect modality and setup test loader"""
        MessageProtocol.status("detecting", "Detecting modality...")
        
        # Try to detect from model first
        self.modality, metadata = ModalityDetector.detect_from_model(self.config.model_path)
        
        # If unknown, try dataset
        if self.modality == Modality.UNKNOWN:
            self.modality, metadata = ModalityDetector.detect_from_dataset(self.config.dataset_path)
        
        MessageProtocol.status("detected", f"Detected modality: {self.modality.value}")
        MessageProtocol.debug("Detection metadata", metadata)
        
        # Ensure plugin loaded
        self._ensure_plugin_loaded(self.modality)
        
        # Get test loader
        loader_class = TEST_REGISTRY.get_test_loader(self.modality)
        if not loader_class:
            raise ValueError(f"No test loader registered for modality: {self.modality.value}")
        
        self.test_loader = loader_class(self.config)
    
    def run_test(self):
        """Execute the test"""
        start_time = time.time()
        
        try:
            MessageProtocol.status("starting", "Test starting...")
            
            # Setup
            self.detect_and_setup()
            
            # Load model and data
            self.test_loader.load_model()
            test_data = self.test_loader.load_test_data()
            
            MessageProtocol.status("testing", "Testing in progress...")
            
            # Test all samples
            all_results = []
            predictions = []
            
            for i, sample in enumerate(test_data):
                result = self.test_loader.test_sample(sample)
                all_results.append(result)
                
                # Create prediction entry
                prediction = {
                    "sample_id": i,
                    **result
                }
                predictions.append(prediction)
                
                # Progress
                elapsed = time.time() - start_time
                samples_per_sec = (i + 1) / elapsed if elapsed > 0 else 0
                MessageProtocol.progress(i + 1, len(test_data), samples_per_sec)
            
            # Compute final metrics
            metrics = self.test_loader.compute_metrics(all_results)
            
            # Save results FIRST
            output_path = Path(self.config.output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            results_file = output_path / "test_results.json"
            
            # Prepare results - CRITICAL: Don't send predictions in event!
            results_summary = {
                **metrics,
                "total_time": time.time() - start_time,
                "samples_per_second": len(test_data) / (time.time() - start_time),
                "results_file": str(results_file)  # Send file path instead!
            }
            
            # Full results with predictions for file only
            results_full = {
                **metrics,
                "predictions": predictions,
                "total_time": time.time() - start_time,
                "samples_per_second": len(test_data) / (time.time() - start_time)
            }
            
            # Save full results to file
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results_full, f, indent=2)
            
            MessageProtocol.status("saved", f"Results saved to {results_file}")
            MessageProtocol.status("complete", f"Test complete: {metrics.get('accuracy', 0):.2f}%")
            
            # CRITICAL: Send ONLY summary, NOT the full predictions!
            MessageProtocol.complete(results_summary)
            
        except Exception as e:
            MessageProtocol.error("Test failed", traceback.format_exc())
            raise


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="FrameTrain v2 Universal Test Engine")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    parser.add_argument("--list-supported", action="store_true", help="List supported modalities")
    
    args = parser.parse_args()
    
    # List supported
    if args.list_supported:
        supported = TEST_REGISTRY.list_supported()
        print(json.dumps({"type": "supported", "data": supported}))
        return
    
    # Load config
    with open(args.config, 'r') as f:
        config_dict = json.load(f)
    
    config = TestConfig.from_dict(config_dict)
    
    # Check PyTorch
    try:
        import torch
    except ImportError:
        MessageProtocol.error("PyTorch not installed", "Install with: pip install torch")
        sys.exit(1)
    
    # Run test
    engine = TestEngine(config)
    engine.run_test()


if __name__ == "__main__":
    main()
