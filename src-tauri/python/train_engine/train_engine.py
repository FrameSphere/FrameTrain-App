"""
FrameTrain v2 - Universal Training Engine (Prototype)
======================================================
Modular, extensible training engine supporting multiple modalities:
- Text (NLP, LLMs)
- Vision (Image Classification, Detection, Segmentation)
- Audio (Speech Recognition, TTS, Audio Classification)
- Graphs (GNN, Node/Edge Classification)
- Tabular (TabNet, Neural Networks for structured data)
- Time Series (Forecasting, Anomaly Detection)
- Reinforcement Learning (PPO, DQN, SAC)
- Multi-Modal (CLIP, Flamingo, etc.)

Architecture:
- Plugin-based system for extensibility
- Registry pattern for model types
- Factory pattern for data loaders
- Strategy pattern for training modes
- Observer pattern for progress updates

Communication: JSON messages via stdout to Rust backend
"""

import os
import sys
import json
import time
import signal
import argparse
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import importlib.util

# Import dependency manager for automatic plugin dependency installation
try:
    # Add plugins directory to path
    plugins_dir = Path(__file__).parent / "plugins"
    if str(plugins_dir) not in sys.path:
        sys.path.insert(0, str(plugins_dir))
    
    from plugin_dependency_manager import (
        DependencyManager,
        PluginManifest
    )
    DEPENDENCY_MANAGER_AVAILABLE = True
except ImportError:
    DEPENDENCY_MANAGER_AVAILABLE = False
    # MessageProtocol not yet defined at this point, so skip warning


# ============================================================================
# COMMUNICATION PROTOCOL
# ============================================================================

class MessageType(Enum):
    """Types of messages sent to backend"""
    PROGRESS = "progress"
    STATUS = "status"
    ERROR = "error"
    WARNING = "warning"
    CHECKPOINT = "checkpoint"
    COMPLETE = "complete"
    METRIC = "metric"
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
    def progress(epoch: int, total_epochs: int, step: int, total_steps: int,
                 train_loss: float, val_loss: Optional[float] = None,
                 learning_rate: float = 0.0, metrics: Dict[str, float] = None):
        """Send progress update"""
        MessageProtocol.send(MessageType.PROGRESS, {
            "epoch": epoch,
            "total_epochs": total_epochs,
            "step": step,
            "total_steps": total_steps,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": learning_rate,
            "metrics": metrics or {},
            "progress_percent": ((epoch - 1) * total_steps + step) / (total_epochs * total_steps) * 100
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
    def checkpoint(path: str, epoch: int, metrics: Dict[str, float]):
        """Send checkpoint saved notification"""
        MessageProtocol.send(MessageType.CHECKPOINT, {
            "path": path,
            "epoch": epoch,
            "metrics": metrics
        })
    
    @staticmethod
    def complete(model_path: str, metrics: Dict[str, Any]):
        """Send training complete notification"""
        MessageProtocol.send(MessageType.COMPLETE, {
            "model_path": model_path,
            "output_path": model_path,
            "final_metrics": metrics
        })
    
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
class TrainingConfig:
    """Complete training configuration"""
    
    # Paths (filled by backend)
    model_path: str = ""
    dataset_path: str = ""
    output_path: str = ""
    checkpoint_dir: str = ""
    
    # Model Type & Task
    model_type: str = "auto"  # auto, transformers, vision, audio, graph, tabular, rl
    task_type: str = "auto"  # auto, classification, detection, segmentation, generation, etc.
    
    # Training Basics
    epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_steps: int = -1  # -1 = use epochs
    
    # Learning Rate
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    
    # Optimizer
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    sgd_momentum: float = 0.9
    
    # Scheduler
    scheduler: str = "linear"
    scheduler_step_size: int = 1
    scheduler_gamma: float = 0.1
    cosine_min_lr: float = 0.0
    
    # Regularization
    dropout: float = 0.1
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.0
    
    # Mixed Precision
    fp16: bool = False
    bf16: bool = False
    
    # LoRA / PEFT
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    # Quantization
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    
    # Data Processing
    max_seq_length: int = 512
    image_size: Tuple[int, int] = (224, 224)
    audio_sample_rate: int = 16000
    num_workers: int = 4
    pin_memory: bool = True
    
    # Evaluation
    eval_steps: int = 500
    eval_strategy: str = "steps"  # steps, epoch
    save_steps: int = 500
    save_strategy: str = "steps"  # steps, epoch
    save_total_limit: int = 3
    
    # Logging
    logging_steps: int = 10
    
    # Advanced
    seed: int = 42
    dataloader_drop_last: bool = False
    group_by_length: bool = False
    
    # Multi-GPU / Distributed
    distributed: bool = False
    local_rank: int = -1
    world_size: int = 1
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary"""
        # Filter only known fields
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
    GRAPH = "graph"
    TABULAR = "tabular"
    TIME_SERIES = "time_series"
    MULTIMODAL = "multimodal"
    REINFORCEMENT = "reinforcement"
    UNKNOWN = "unknown"


class ModalityDetector:
    """Automatically detects data modality from dataset"""
    
    @staticmethod
    def detect(dataset_path: str) -> Tuple[Modality, Dict[str, Any]]:
        """
        Detect modality from dataset structure and files
        Returns: (modality, metadata)
        """
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            return Modality.UNKNOWN, {"error": "Dataset path does not exist"}
        
        # Check train directory
        train_path = dataset_path / "train"
        if not train_path.exists():
            return Modality.UNKNOWN, {"error": "No train directory found"}
        
        # Get file extensions
        files = list(train_path.glob("*"))
        extensions = {f.suffix.lower() for f in files if f.is_file()}
        
        metadata = {
            "total_files": len(files),
            "extensions": list(extensions),
            "sample_files": [f.name for f in files[:5]]
        }
        
        # Detection logic
        if extensions & {'.txt', '.json', '.jsonl', '.csv'}:
            # Check if it's tabular
            if '.csv' in extensions or '.parquet' in extensions:
                # Could be tabular or text
                # TODO: Better detection by inspecting CSV structure
                metadata["note"] = "CSV detected - could be text or tabular"
                return Modality.TEXT, metadata  # For now, treat as text
            return Modality.TEXT, metadata
        
        elif extensions & {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}:
            # Check for annotations (YOLO, COCO, etc.)
            if (train_path / "labels").exists() or any(f.suffix == '.xml' for f in files):
                metadata["annotation_format"] = "detection"
                return Modality.VISION, metadata
            return Modality.VISION, metadata
        
        elif extensions & {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}:
            return Modality.AUDIO, metadata
        
        elif extensions & {'.graph', '.graphml', '.gexf', '.gml', '.edgelist'}:
            return Modality.GRAPH, metadata
        
        elif extensions & {'.npy', '.npz', '.h5', '.hdf5'}:
            # Could be embeddings, time series, or preprocessed data
            metadata["note"] = "Binary format - need more inspection"
            return Modality.UNKNOWN, metadata
        
        else:
            return Modality.UNKNOWN, metadata


# ============================================================================
# ABSTRACT BASE CLASSES
# ============================================================================

class BaseDataLoader(ABC):
    """Abstract base class for data loaders"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
    
    @abstractmethod
    def load(self) -> Tuple[Any, Optional[Any], Optional[Any]]:
        """
        Load dataset and create data loaders
        Returns: (train_loader, val_loader, test_loader)
        """
        pass
    
    @abstractmethod
    def get_sample_info(self) -> Dict[str, Any]:
        """Get information about a sample from the dataset"""
        pass


class BaseModelHandler(ABC):
    """Abstract base class for model handlers"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.device = None
    
    @abstractmethod
    def load_model(self):
        """Load the model"""
        pass
    
    @abstractmethod
    def forward(self, batch: Any) -> Any:
        """Forward pass"""
        pass
    
    @abstractmethod
    def compute_loss(self, outputs: Any, batch: Any) -> Any:
        """Compute loss"""
        pass
    
    @abstractmethod
    def save_model(self, path: str):
        """Save the model"""
        pass


class BaseTrainer(ABC):
    """Abstract base class for trainers"""
    
    def __init__(self, config: TrainingConfig, model_handler: BaseModelHandler, 
                 data_loader: BaseDataLoader):
        self.config = config
        self.model_handler = model_handler
        self.data_loader = data_loader
        self.optimizer = None
        self.scheduler = None
        self.is_stopped = False
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        self.training_logs = []  # Track all training steps for analysis
        self.start_time = None
        
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle stop signals"""
        MessageProtocol.status("stopping", "Training is being stopped...")
        self.is_stopped = True
    
    @abstractmethod
    def train_epoch(self, epoch: int) -> float:
        """Train one epoch"""
        pass
    
    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set"""
        pass
    
    @abstractmethod
    def train(self):
        """Main training loop"""
        pass


# ============================================================================
# REGISTRY SYSTEM
# ============================================================================

class Registry:
    """Registry for model handlers and data loaders"""
    
    def __init__(self):
        self._data_loaders = {}
        self._model_handlers = {}
        self._trainers = {}
    
    def register_data_loader(self, modality: Modality, loader_class: type):
        """Register a data loader for a modality"""
        self._data_loaders[modality] = loader_class
    
    def register_model_handler(self, model_type: str, handler_class: type):
        """Register a model handler"""
        self._model_handlers[model_type] = handler_class
    
    def register_trainer(self, trainer_type: str, trainer_class: type):
        """Register a trainer"""
        self._trainers[trainer_type] = trainer_class
    
    def get_data_loader(self, modality: Modality) -> Optional[type]:
        """Get data loader class for modality"""
        return self._data_loaders.get(modality)
    
    def get_model_handler(self, model_type: str) -> Optional[type]:
        """Get model handler class"""
        return self._model_handlers.get(model_type)
    
    def get_trainer(self, trainer_type: str) -> Optional[type]:
        """Get trainer class"""
        return self._trainers.get(trainer_type)
    
    def list_supported(self) -> Dict[str, List[str]]:
        """List all supported types"""
        return {
            "data_loaders": [m.value for m in self._data_loaders.keys()],
            "model_handlers": list(self._model_handlers.keys()),
            "trainers": list(self._trainers.keys())
        }


# Global registry instance
REGISTRY = Registry()


# ============================================================================
# TEXT DATA LOADER
# ============================================================================

class TextDataLoader(BaseDataLoader):
    """Data loader for text/NLP tasks"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.tokenizer = None
    
    def load(self) -> Tuple[Any, Optional[Any], Optional[Any]]:
        """Load text data"""
        try:
            import torch
            from torch.utils.data import DataLoader, Dataset
            
            # Check if transformers is available
            try:
                from transformers import AutoTokenizer
                from datasets import Dataset as HFDataset
                
                MessageProtocol.status("loading", "Loading tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                MessageProtocol.status("loading", "Loading text datasets...")
                
                # Load train data
                train_path = Path(self.config.dataset_path) / "train"
                train_data = self._load_text_files(train_path)
                
                if not train_data:
                    raise ValueError(f"No valid text data found in {train_path}")
                
                MessageProtocol.status("loading", f"Loaded {len(train_data)} training samples")
                
                # Create HF dataset and tokenize
                train_dataset = HFDataset.from_list(train_data)
                train_dataset = train_dataset.map(
                    self._tokenize_function,
                    batched=True,
                    remove_columns=[col for col in train_dataset.column_names 
                                  if col not in ['input_ids', 'attention_mask', 'labels']]
                )
                train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
                
                self.train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=True,
                    num_workers=0,  # Safer for compatibility
                    pin_memory=False
                )
                
                # Load validation data if exists
                val_path = Path(self.config.dataset_path) / "val"
                if val_path.exists():
                    val_data = self._load_text_files(val_path)
                    if val_data:
                        val_dataset = HFDataset.from_list(val_data)
                        val_dataset = val_dataset.map(self._tokenize_function, batched=True)
                        val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
                        
                        self.val_loader = DataLoader(
                            val_dataset,
                            batch_size=self.config.batch_size,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=False
                        )
                
                return self.train_loader, self.val_loader, None
                
            except ImportError:
                MessageProtocol.error("Transformers not installed", 
                                    "Install with: pip install transformers datasets")
                raise
                
        except Exception as e:
            MessageProtocol.error("Failed to load text data", str(e))
            raise
    
    def _load_text_files(self, path: Path) -> List[Dict[str, str]]:
        """Load text files from directory"""
        data = []
        
        # Load .txt files
        for txt_file in path.glob("*.txt"):
            with open(txt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append({"text": line})
        
        # Load .json files
        for json_file in path.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                try:
                    json_data = json.load(f)
                    if isinstance(json_data, list):
                        data.extend(json_data)
                    else:
                        data.append(json_data)
                except:
                    pass
        
        # Load .jsonl files
        for jsonl_file in path.glob("*.jsonl"):
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data.append(json.loads(line))
                    except:
                        pass
        
        # Load .csv files
        for csv_file in path.glob("*.csv"):
            import csv
            with open(csv_file, 'r', encoding='utf-8') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    if len(row) >= 2:
                        data.append({"input": row[0].strip(), "target": row[1].strip()})
                    elif len(row) == 1:
                        data.append({"text": row[0].strip()})
        
        return data
    
    def _tokenize_function(self, examples):
        """Tokenize examples"""
        # Determine text field
        text_field = None
        if 'text' in examples:
            text_field = 'text'
        elif 'input' in examples:
            text_field = 'input'
        elif 'content' in examples:
            text_field = 'content'
        
        if not text_field:
            raise ValueError(f"No text field found in: {list(examples.keys())}")
        
        texts = examples[text_field]
        
        # Tokenize
        model_inputs = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.config.max_seq_length,
            padding="max_length"
        )
        
        # Create labels (for causal LM, labels = input_ids)
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        
        return model_inputs
    
    def get_sample_info(self) -> Dict[str, Any]:
        """Get sample information"""
        return {
            "tokenizer": str(type(self.tokenizer).__name__) if self.tokenizer else None,
            "vocab_size": self.tokenizer.vocab_size if self.tokenizer else None,
            "max_length": self.config.max_seq_length
        }


# Register text data loader
REGISTRY.register_data_loader(Modality.TEXT, TextDataLoader)


# ============================================================================
# TRANSFORMERS MODEL HANDLER
# ============================================================================

class TransformersModelHandler(BaseModelHandler):
    """Handler for HuggingFace Transformers models"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.model_class = None
        self.tokenizer = None
    
    def load_model(self):
        """Load transformers model"""
        try:
            import torch
            from transformers import (
                AutoConfig, AutoModel, AutoModelForCausalLM,
                AutoModelForSeq2SeqLM, AutoTokenizer
            )
            
            MessageProtocol.status("loading", "Loading model...")
            
            # Determine device
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                MessageProtocol.status("device", f"Using GPU: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
                MessageProtocol.status("device", "Using Apple Silicon GPU (MPS)")
            else:
                self.device = torch.device("cpu")
                MessageProtocol.status("device", "Using CPU")
            
            # Load config to detect model type
            model_config = AutoConfig.from_pretrained(self.config.model_path)
            model_type = model_config.model_type
            architectures = getattr(model_config, 'architectures', [])
            
            MessageProtocol.status("loading", f"Detected model type: {model_type}")
            
            # Determine appropriate model class
            if architectures:
                arch_name = architectures[0]
                if any(x in arch_name for x in ['T5', 'Bart', 'Pegasus']):
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path)
                    self.model_class = 'seq2seq'
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(self.config.model_path)
                    self.model_class = 'causal_lm'
            else:
                # Default to causal LM
                self.model = AutoModelForCausalLM.from_pretrained(self.config.model_path)
                self.model_class = 'causal_lm'
            
            # Apply LoRA if configured
            if self.config.use_lora:
                self._apply_lora()
            
            # Move to device
            self.model = self.model.to(self.device)
            
            # Enable gradient checkpointing for memory efficiency on MPS
            if self.device.type == 'mps' and hasattr(self.model, 'gradient_checkpointing_enable'):
                try:
                    self.model.gradient_checkpointing_enable()
                    MessageProtocol.status("loaded", "Gradient checkpointing enabled")
                except:
                    pass
            
            MessageProtocol.status("loaded", f"Model loaded: {self.config.model_path}")
            
        except Exception as e:
            MessageProtocol.error("Model loading failed", str(e))
            raise
    
    def _apply_lora(self):
        """Apply LoRA to model"""
        try:
            from peft import get_peft_model, LoraConfig, TaskType
            
            MessageProtocol.status("loading", "Applying LoRA...")
            
            # Determine task type
            task_type = TaskType.CAUSAL_LM if self.model_class == 'causal_lm' else TaskType.SEQ_2_SEQ_LM
            
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                bias="none",
                task_type=task_type
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            
        except ImportError:
            MessageProtocol.error("PEFT not installed", "Install with: pip install peft")
            raise
    
    def forward(self, batch: Any) -> Any:
        """Forward pass"""
        if not isinstance(batch, dict):
            raise ValueError(f"Expected dict batch, got {type(batch)}")
        
        # Move batch to device
        batch = {k: v.to(self.device) if hasattr(v, 'to') else v 
                for k, v in batch.items()}
        
        return self.model(**batch)
    
    def compute_loss(self, outputs: Any, batch: Any) -> Any:
        """Compute loss"""
        return outputs.loss
    
    def save_model(self, path: str):
        """Save model and tokenizer"""
        output_path = Path(path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Merge LoRA weights if applicable
        if self.config.use_lora:
            try:
                self.model = self.model.merge_and_unload()
                # Make tensors contiguous
                for param in self.model.parameters():
                    if param.data is not None and not param.is_contiguous():
                        param.data = param.data.contiguous()
            except:
                pass
        
        # Save model
        self.model.save_pretrained(output_path)
        
        # CRITICAL: Save tokenizer as well!
        try:
            from transformers import AutoTokenizer
            # Try to load and save tokenizer from original model
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            tokenizer.save_pretrained(output_path)
            MessageProtocol.status("saved", f"Model and tokenizer saved to {output_path}")
        except Exception as e:
            MessageProtocol.warning(f"Could not save tokenizer: {e}. Test engine will need to load tokenizer from original model.")
            # Save a note file about the original model for the test engine
            with open(output_path / "tokenizer_info.txt", 'w') as f:
                f.write(f"Original model: {self.config.model_path}\n")
                f.write(f"Tokenizer should be loaded from: {self.config.model_path}\n")
            MessageProtocol.status("saved", f"Model saved to {output_path}")


# Register transformers handler
REGISTRY.register_model_handler("transformers", TransformersModelHandler)


# ============================================================================
# UNIVERSAL TRAINER
# ============================================================================

class UniversalTrainer(BaseTrainer):
    """Universal trainer that works with any model handler and data loader"""
    
    def __init__(self, config: TrainingConfig, model_handler: BaseModelHandler,
                 data_loader: BaseDataLoader):
        super().__init__(config, model_handler, data_loader)
    
    def setup(self):
        """Setup optimizer, scheduler, etc."""
        import torch.optim as optim
        
        MessageProtocol.status("setup", "Setting up training...")
        
        # Create optimizer
        params = [p for p in self.model_handler.model.parameters() if p.requires_grad]
        
        if self.config.optimizer == "adamw":
            self.optimizer = optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "adam":
            self.optimizer = optim.Adam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "sgd":
            self.optimizer = optim.SGD(
                params,
                lr=self.config.learning_rate,
                momentum=self.config.sgd_momentum,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        # Create scheduler (simplified for prototype)
        if self.config.scheduler == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            num_steps = len(self.data_loader.train_loader) * self.config.epochs
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_steps)
        
        MessageProtocol.status("ready", "Training setup complete")
    
    def train_epoch(self, epoch: int) -> float:
        """Train one epoch"""
        import torch
        
        self.model_handler.model.train()
        total_loss = 0.0
        num_batches = len(self.data_loader.train_loader)
        
        for step, batch in enumerate(self.data_loader.train_loader):
            if self.is_stopped:
                break
            
            # Clear cache for MPS
            if self.model_handler.device.type == 'mps':
                torch.mps.empty_cache()
            
            # Forward pass
            outputs = self.model_handler.forward(batch)
            loss = self.model_handler.compute_loss(outputs, batch)
            
            # Backward pass
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()
            
            # Optimizer step
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model_handler.model.parameters(),
                    self.config.max_grad_norm
                )
                self.optimizer.step()
                
                if self.scheduler:
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Clear cache after optimizer step
                if self.model_handler.device.type == 'mps':
                    torch.mps.empty_cache()
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            
            # Progress update
            if step % self.config.logging_steps == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Collect log entry for analysis
                log_entry = {
                    "epoch": epoch,
                    "step": step + 1,
                    "train_loss": total_loss / (step + 1),
                    "val_loss": None,  # Will be filled during evaluation
                    "learning_rate": current_lr,
                    "timestamp": datetime.now().isoformat()
                }
                self.training_logs.append(log_entry)
                
                MessageProtocol.progress(
                    epoch=epoch,
                    total_epochs=self.config.epochs,
                    step=step + 1,
                    total_steps=num_batches,
                    train_loss=total_loss / (step + 1),
                    learning_rate=current_lr
                )
        
        return total_loss / num_batches
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set"""
        import torch
        
        if self.data_loader.val_loader is None:
            return {}
        
        self.model_handler.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in self.data_loader.val_loader:
                outputs = self.model_handler.forward(batch)
                loss = self.model_handler.compute_loss(outputs, batch)
                total_loss += loss.item()
        
        return {"val_loss": total_loss / len(self.data_loader.val_loader)}
    
    def train(self):
        """Main training loop"""
        self.start_time = time.time()
        
        try:
            MessageProtocol.status("starting", "Starting training...")
            
            # Load model and data
            self.model_handler.load_model()
            self.data_loader.load()
            self.setup()
            
            MessageProtocol.status("training", "Training in progress...")
            
            for epoch in range(1, self.config.epochs + 1):
                if self.is_stopped:
                    break
                
                self.current_epoch = epoch
                MessageProtocol.status("epoch", f"Epoch {epoch}/{self.config.epochs}")
                
                # Train
                train_loss = self.train_epoch(epoch)
                
                # Evaluate
                eval_metrics = self.evaluate()
                val_loss = eval_metrics.get("val_loss")
                
                # Final progress for epoch
                MessageProtocol.progress(
                    epoch=epoch,
                    total_epochs=self.config.epochs,
                    step=len(self.data_loader.train_loader),
                    total_steps=len(self.data_loader.train_loader),
                    train_loss=train_loss,
                    val_loss=val_loss,
                    learning_rate=self.optimizer.param_groups[0]['lr']
                )
            
            # Training complete
            if not self.is_stopped:
                self.model_handler.save_model(self.config.output_path)
                
                training_duration = int(time.time() - self.start_time)
                final_metrics = {
                    "final_train_loss": train_loss,
                    "final_val_loss": val_loss if val_loss else None,
                    "best_val_loss": self.best_metric if self.best_metric != float('inf') else None,
                    "training_duration_seconds": training_duration,
                    "total_epochs": self.config.epochs,
                    "total_steps": self.global_step
                }
                
                # Save metrics to file for version system and analysis
                metrics_file = Path(self.config.output_path) / "metrics.json"
                metrics_file.parent.mkdir(parents=True, exist_ok=True)
                with open(metrics_file, 'w', encoding='utf-8') as f:
                    json.dump(final_metrics, f, indent=2)
                MessageProtocol.status("metrics_saved", f"Metrics saved: {metrics_file}")
                
                # Save training logs
                logs_dir = Path(self.config.output_path) / "logs"
                logs_dir.mkdir(parents=True, exist_ok=True)
                logs_file = logs_dir / "training_logs.json"
                with open(logs_file, 'w', encoding='utf-8') as f:
                    json.dump(self.training_logs, f, indent=2)
                MessageProtocol.status("logs_saved", f"Training logs saved: {logs_file}")
                
                MessageProtocol.complete(self.config.output_path, final_metrics)
            else:
                MessageProtocol.status("stopped", "Training was stopped")
                
        except Exception as e:
            MessageProtocol.error("Training failed", traceback.format_exc())
            raise


# Register universal trainer
REGISTRY.register_trainer("universal", UniversalTrainer)


# ============================================================================
# TRAINING ENGINE
# ============================================================================

class TrainingEngine:
    """Main training engine that orchestrates everything"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.modality = None
        self.data_loader = None
        self.model_handler = None
        self.trainer = None
        self.plugin_loader = None
    
    def _ensure_plugin_loaded(self, modality: Modality):
        """
        Ensure the appropriate plugin is loaded for the detected modality.
        This includes automatic dependency installation.
        """
        # Map modalities to plugin files
        plugin_map = {
            Modality.TEXT: None,  # Built-in, no plugin needed
            Modality.VISION: "plugin_vision.py",
            Modality.AUDIO: "plugin_audio.py",
            Modality.GRAPH: "plugin_graph.py",
            Modality.TABULAR: "plugin_tabular.py",
            Modality.TIME_SERIES: "plugin_timeseries.py",
            Modality.MULTIMODAL: "plugin_multimodal.py",
            Modality.REINFORCEMENT: "plugin_rl.py",
        }
        
        plugin_file = plugin_map.get(modality)
        
        if plugin_file is None:
            # No plugin needed (e.g., TEXT is built-in)
            MessageProtocol.debug(f"Modality {modality.value} uses built-in support")
            return
        
        # Check if plugin is already loaded in REGISTRY
        if REGISTRY.get_data_loader(modality) is not None:
            MessageProtocol.debug(f"Plugin for {modality.value} already loaded")
            return
        
        # Need to load the plugin
        MessageProtocol.status(
            "loading_plugin",
            f"Loading plugin for {modality.value}..."
        )
        
        # Create plugin loader if not exists
        if self.plugin_loader is None:
            plugin_dir = Path(__file__).parent
            self.plugin_loader = PluginLoader(str(plugin_dir))
        
        # Determine auto-install setting
        auto_install = os.getenv("FRAMETRAIN_AUTO_INSTALL", "false").lower() == "true"
        
        # Load the specific plugin
        module = self.plugin_loader.load_plugin(plugin_file, auto_install=auto_install)
        
        if module is None:
            raise RuntimeError(
                f"Failed to load required plugin: {plugin_file}\n"
                f"This plugin is needed for {modality.value} tasks.\n"
                f"Try installing dependencies manually or set FRAMETRAIN_AUTO_INSTALL=true"
            )
        
        # Verify plugin registered correctly
        if REGISTRY.get_data_loader(modality) is None:
            raise RuntimeError(
                f"Plugin {plugin_file} loaded but did not register for {modality.value}\n"
                f"This is a bug in the plugin."
            )
    
    def detect_and_setup(self):
        """Detect modality and setup appropriate components"""
        MessageProtocol.status("detecting", "Detecting data modality...")
        
        # Detect modality
        self.modality, metadata = ModalityDetector.detect(self.config.dataset_path)
        
        MessageProtocol.status("detected", f"Detected modality: {self.modality.value}")
        MessageProtocol.debug("Modality metadata", metadata)
        
        # Check if plugin needs to be loaded for this modality
        self._ensure_plugin_loaded(self.modality)
        
        # Get appropriate data loader
        loader_class = REGISTRY.get_data_loader(self.modality)
        if not loader_class:
            raise ValueError(f"No data loader registered for modality: {self.modality.value}")
        
        self.data_loader = loader_class(self.config)
        
        # Get appropriate model handler
        model_type = self.config.model_type
        if model_type == "auto":
            # Auto-detect based on modality
            if self.modality == Modality.TEXT:
                model_type = "transformers"
            else:
                raise ValueError(f"No auto model type for modality: {self.modality.value}")
        
        handler_class = REGISTRY.get_model_handler(model_type)
        if not handler_class:
            raise ValueError(f"No model handler registered for type: {model_type}")
        
        self.model_handler = handler_class(self.config)
        
        # Get trainer
        trainer_class = REGISTRY.get_trainer("universal")
        self.trainer = trainer_class(self.config, self.model_handler, self.data_loader)
    
    def train(self):
        """Execute training"""
        try:
            self.detect_and_setup()
            self.trainer.train()
        except Exception as e:
            MessageProtocol.error("Training engine failed", str(e))
            raise


# ============================================================================
# PLUGIN LOADER WITH DEPENDENCY MANAGEMENT
# ============================================================================

class PluginLoader:
    """Load custom plugins with automatic dependency management"""
    
    def __init__(self, plugin_dir: str):
        self.plugin_dir = Path(plugin_dir)
        self.dep_manager = DependencyManager() if DEPENDENCY_MANAGER_AVAILABLE else None
        self.loaded_plugins = {}
    
    def load_plugin(self, plugin_file: str, auto_install: bool = None) -> Optional[Any]:
        """
        Load a single plugin with dependency management
        
        Args:
            plugin_file: Plugin filename (e.g., 'plugin_vision.py')
            auto_install: Auto-install dependencies (None = use env var)
        """
        plugin_path = self.plugin_dir / plugin_file
        
        if not plugin_path.exists():
            MessageProtocol.warning(f"Plugin not found: {plugin_path}")
            return None
        
        # Check if dependency manager is available
        if not DEPENDENCY_MANAGER_AVAILABLE:
            MessageProtocol.warning(
                f"Loading {plugin_file} without dependency management"
            )
            return self._load_plugin_without_deps(plugin_path)
        
        # Parse manifest
        try:
            manifest = PluginManifest.from_docstring(plugin_path)
        except Exception as e:
            MessageProtocol.warning(
                f"Could not parse manifest from {plugin_file}: {e}\n"
                f"Loading plugin without dependency check..."
            )
            return self._load_plugin_without_deps(plugin_path)
        
        if manifest is None:
            MessageProtocol.debug(
                f"No manifest found in {plugin_file}, loading without dep check"
            )
            return self._load_plugin_without_deps(plugin_path)
        
        # Check dependencies
        MessageProtocol.status(
            "checking_deps",
            f"Checking dependencies for {manifest.plugin_name}..."
        )
        
        missing_req, missing_opt = self.dep_manager.check_dependencies(manifest)
        
        if not missing_req:
            MessageProtocol.status(
                "deps_ok",
                f"All dependencies for {manifest.plugin_name} are installed"
            )
            return self._load_plugin_without_deps(plugin_path)
        
        # Dependencies missing - need to install
        MessageProtocol.status(
            "missing_deps",
            f"Plugin {manifest.plugin_name} requires: {', '.join(missing_req)}"
        )
        
        # Determine if we should auto-install
        if auto_install is None:
            auto_install = os.getenv("FRAMETRAIN_AUTO_INSTALL", "false").lower() == "true"
        
        # Install dependencies
        success = self.dep_manager.install_dependencies(
            manifest,
            auto_install=auto_install,
            install_optional=False  # Don't install optional by default
        )
        
        if not success:
            MessageProtocol.error(
                f"Failed to install dependencies for {manifest.plugin_name}",
                "Training cannot continue without required dependencies"
            )
            return None
        
        # Dependencies installed, now load the plugin
        MessageProtocol.status("loading_plugin", f"Loading {plugin_file}...")
        return self._load_plugin_without_deps(plugin_path)
    
    def _load_plugin_without_deps(self, plugin_path: Path) -> Optional[Any]:
        """Load plugin module without dependency check"""
        try:
            spec = importlib.util.spec_from_file_location(plugin_path.stem, plugin_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            self.loaded_plugins[plugin_path.name] = module
            MessageProtocol.debug(f"✓ Loaded plugin: {plugin_path.name}")
            return module
            
        except Exception as e:
            MessageProtocol.error(
                f"Failed to load plugin {plugin_path.name}",
                str(e)
            )
            return None
    
    def load_plugins(self, auto_install: bool = None):
        """Load all plugins from directory"""
        if not self.plugin_dir.exists():
            MessageProtocol.warning(f"Plugin directory not found: {self.plugin_dir}")
            return
        
        plugin_files = list(self.plugin_dir.glob("plugin_*.py"))
        MessageProtocol.status(
            "loading_plugins",
            f"Found {len(plugin_files)} plugins in {self.plugin_dir}"
        )
        
        for plugin_file in plugin_files:
            self.load_plugin(plugin_file.name, auto_install=auto_install)
    
    @staticmethod
    def load_plugins_static(plugin_dir: str, auto_install: bool = None):
        """Static method for backward compatibility"""
        loader = PluginLoader(plugin_dir)
        loader.load_plugins(auto_install=auto_install)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="FrameTrain v2 Universal Training Engine")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    parser.add_argument("--list-supported", action="store_true", help="List supported types")
    parser.add_argument("--plugins", type=str, help="Path to plugins directory")
    
    args = parser.parse_args()
    
    # Load plugins if specified
    if args.plugins:
        auto_install = os.getenv("FRAMETRAIN_AUTO_INSTALL", "false").lower() == "true"
        PluginLoader.load_plugins_static(args.plugins, auto_install=auto_install)
    
    # List supported types
    if args.list_supported:
        supported = REGISTRY.list_supported()
        print(json.dumps({"type": "supported", "data": supported}))
        return
    
    # Load config
    with open(args.config, 'r') as f:
        config_dict = json.load(f)
    
    config = TrainingConfig.from_dict(config_dict)
    
    # Check PyTorch
    try:
        import torch
    except ImportError:
        MessageProtocol.error("PyTorch not installed", "Install with: pip install torch")
        sys.exit(1)
    
    # Run training
    engine = TrainingEngine(config)
    engine.train()


if __name__ == "__main__":
    main()
