"""
Multi-Modal Plugin for FrameTrain v2
====================================
This plugin adds support for multi-modal models:
- CLIP (Contrastive Language-Image Pre-training)
- Stable Diffusion (Text-to-Image Generation)
- Flamingo (Vision-Language Models)
- BLIP (Bootstrapping Language-Image Pre-training)

MANIFEST:
{
    "name": "Multi-Modal Plugin",
    "description": "Multi-modal models (CLIP, Stable Diffusion, BLIP)",
    "modality": "multimodal",
    "required": [
        "torch",
        "transformers",
        "pillow"
    ],
    "optional": [
        "diffusers",
        "accelerate",
        "open_clip_torch"
    ],
    "python": "3.8"
}

Installation:
    pip install transformers diffusers accelerate
    pip install open_clip_torch  # Alternative CLIP implementation

Usage:
    python train_engine.py --config config.json --plugins ./plugins/
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import json
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from torchvision import transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

try:
    from transformers import (
        CLIPProcessor, CLIPModel,
        BlipProcessor, BlipForConditionalGeneration,
        AutoProcessor, AutoModel
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from diffusers import StableDiffusionPipeline, DDPMScheduler
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False


# Import from main engine
try:
    from train_engine import (
        BaseDataLoader, BaseModelHandler, Modality, 
        REGISTRY, MessageProtocol, TrainingConfig
    )
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent))
    from train_engine import (
        BaseDataLoader, BaseModelHandler, Modality, 
        REGISTRY, MessageProtocol, TrainingConfig
    )


# ============================================================================
# MULTI-MODAL DATASET
# ============================================================================

class MultiModalDataset(Dataset):
    """Dataset for image-text pairs"""
    
    def __init__(self, data_dir: Path, transform=None, max_text_length: int = 77):
        """
        Args:
            data_dir: Directory with data
            transform: Image transforms
            max_text_length: Maximum text length
        """
        self.data_dir = data_dir
        self.transform = transform
        self.max_text_length = max_text_length
        
        # Load image-text pairs
        self.pairs = self._load_pairs()
    
    def _load_pairs(self) -> List[Dict[str, str]]:
        """Load image-text pairs from directory"""
        pairs = []
        
        # Method 1: annotations.json
        annotations_file = self.data_dir / "annotations.json"
        if annotations_file.exists():
            with open(annotations_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for item in data:
                    if 'image' in item and 'text' in item:
                        pairs.append({
                            'image': str(self.data_dir / item['image']),
                            'text': item['text']
                        })
            
            return pairs
        
        # Method 2: Separate text files (image.jpg -> image.txt)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        for ext in image_extensions:
            for img_file in self.data_dir.glob(f"*{ext}"):
                txt_file = img_file.with_suffix('.txt')
                if txt_file.exists():
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    
                    pairs.append({
                        'image': str(img_file),
                        'text': text
                    })
        
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Load image
        image = Image.open(pair['image']).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        text = pair['text']
        
        return {
            'image': image,
            'text': text
        }


# ============================================================================
# MULTI-MODAL DATA LOADER
# ============================================================================

class MultiModalDataLoader(BaseDataLoader):
    """Data loader for multi-modal tasks"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.processor = None
        
        # Setup transforms (will be overridden by processor if available)
        self.transform = transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                               [0.26862954, 0.26130258, 0.27577711])
        ])
    
    def load(self) -> Tuple[Any, Optional[Any], Optional[Any]]:
        """Load multi-modal datasets"""
        try:
            MessageProtocol.status("loading", "Loading multi-modal datasets...")
            
            dataset_path = Path(self.config.dataset_path)
            train_path = dataset_path / "train"
            
            if not train_path.exists():
                raise ValueError(f"Training path not found: {train_path}")
            
            # Create dataset
            train_dataset = MultiModalDataset(
                train_path,
                transform=self.transform
            )
            
            if len(train_dataset) == 0:
                raise ValueError("No image-text pairs found. Ensure you have:\n"
                               "1. annotations.json with image-text pairs, OR\n"
                               "2. .txt files with same name as images")
            
            MessageProtocol.status("loading", 
                f"Loaded {len(train_dataset)} image-text pairs")
            
            # Custom collate function (will process with model's processor)
            def collate_fn(batch):
                images = [item['image'] for item in batch]
                texts = [item['text'] for item in batch]
                
                return {
                    'images': images,
                    'texts': texts
                }
            
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=0,  # Set to 0 to avoid pickling issues with processor
                pin_memory=self.config.pin_memory,
                collate_fn=collate_fn
            )
            
            # Load validation data if exists
            val_path = dataset_path / "val"
            if val_path.exists():
                val_dataset = MultiModalDataset(
                    val_path,
                    transform=self.transform
                )
                
                if len(val_dataset) > 0:
                    self.val_loader = DataLoader(
                        val_dataset,
                        batch_size=self.config.batch_size,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=self.config.pin_memory,
                        collate_fn=collate_fn
                    )
                    
                    MessageProtocol.status("loading", 
                        f"Loaded {len(val_dataset)} validation pairs")
            
            return self.train_loader, self.val_loader, None
            
        except Exception as e:
            MessageProtocol.error("Failed to load multi-modal data", str(e))
            raise
    
    def get_sample_info(self) -> Dict[str, Any]:
        """Get sample information"""
        return {
            "image_size": self.config.image_size,
            "num_train_pairs": len(self.train_loader.dataset) if self.train_loader else 0,
            "num_val_pairs": len(self.val_loader.dataset) if self.val_loader else 0
        }


# ============================================================================
# MULTI-MODAL MODEL HANDLER (CLIP, BLIP, etc.)
# ============================================================================

class MultiModalModelHandler(BaseModelHandler):
    """Handler for multi-modal models"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.processor = None
        self.model_type = None
    
    def load_model(self):
        """Load multi-modal model"""
        try:
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch not available")
            
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("Transformers not available. Install with: pip install transformers")
            
            MessageProtocol.status("loading", "Loading multi-modal model...")
            
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
            
            model_path = self.config.model_path
            
            # Detect model type
            if "clip" in model_path.lower():
                self.model_type = "clip"
                MessageProtocol.status("loading", f"Loading CLIP model: {model_path}")
                
                self.processor = CLIPProcessor.from_pretrained(model_path)
                self.model = CLIPModel.from_pretrained(model_path)
            
            elif "blip" in model_path.lower():
                self.model_type = "blip"
                MessageProtocol.status("loading", f"Loading BLIP model: {model_path}")
                
                self.processor = BlipProcessor.from_pretrained(model_path)
                self.model = BlipForConditionalGeneration.from_pretrained(model_path)
            
            elif "stable-diffusion" in model_path.lower() or "sd-" in model_path.lower():
                if not DIFFUSERS_AVAILABLE:
                    raise ImportError("Diffusers not installed. Install with: pip install diffusers")
                
                self.model_type = "stable_diffusion"
                MessageProtocol.status("loading", f"Loading Stable Diffusion: {model_path}")
                
                # This is simplified - full SD training is complex
                self.model = StableDiffusionPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
                )
                self.processor = self.model.tokenizer
            
            else:
                # Try generic auto model
                MessageProtocol.status("loading", f"Loading generic multi-modal model: {model_path}")
                
                self.processor = AutoProcessor.from_pretrained(model_path)
                self.model = AutoModel.from_pretrained(model_path)
                self.model_type = "generic"
            
            # Move to device
            if self.model_type != "stable_diffusion":  # SD has its own device management
                self.model = self.model.to(self.device)
            
            MessageProtocol.status("loaded", f"Multi-modal model loaded: {self.model_type}")
            
        except Exception as e:
            MessageProtocol.error("Multi-modal model loading failed", str(e))
            raise
    
    def forward(self, batch: Any) -> Any:
        """Forward pass"""
        images = batch['images']
        texts = batch['texts']
        
        if self.model_type == "clip":
            # CLIP contrastive learning
            inputs = self.processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            
            # Compute contrastive loss
            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text
            
            # Ground truth: diagonal is positive pairs
            batch_size = len(texts)
            labels = torch.arange(batch_size, device=self.device)
            
            loss_i = nn.CrossEntropyLoss()(logits_per_image, labels)
            loss_t = nn.CrossEntropyLoss()(logits_per_text, labels)
            loss = (loss_i + loss_t) / 2
            
            return {"loss": loss, "outputs": outputs}
        
        elif self.model_type == "blip":
            # BLIP image captioning
            inputs = self.processor(
                images=images,
                text=texts,
                return_tensors="pt",
                padding=True
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            
            return {"loss": outputs.loss, "outputs": outputs}
        
        elif self.model_type == "stable_diffusion":
            # Stable Diffusion training (very simplified)
            # Full SD training requires diffusion process
            MessageProtocol.warning("Stable Diffusion training is complex and requires specialized setup")
            return {"loss": torch.tensor(0.0, device=self.device)}
        
        else:
            # Generic
            inputs = self.processor(
                images=images,
                text=texts,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            return outputs
    
    def compute_loss(self, outputs: Any, batch: Any) -> Any:
        """Compute loss"""
        if isinstance(outputs, dict) and "loss" in outputs:
            return outputs["loss"]
        else:
            # If no loss in outputs, return 0
            return torch.tensor(0.0, device=self.device)
    
    def save_model(self, path: str):
        """Save model"""
        output_path = Path(path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model and processor
        if self.model_type == "stable_diffusion":
            self.model.save_pretrained(output_path)
        else:
            self.model.save_pretrained(output_path)
            self.processor.save_pretrained(output_path)
        
        # Save config
        config_dict = {
            "model_type": self.model_type,
            "image_size": self.config.image_size
        }
        
        with open(output_path / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        MessageProtocol.status("saved", f"Multi-modal model saved to {output_path}")


# ============================================================================
# REGISTER PLUGIN
# ============================================================================

# Register multi-modal components
REGISTRY.register_data_loader(Modality.MULTIMODAL, MultiModalDataLoader)
REGISTRY.register_model_handler("multimodal", MultiModalModelHandler)

MessageProtocol.debug("Multi-modal plugin loaded", {
    "transformers_available": TRANSFORMERS_AVAILABLE,
    "diffusers_available": DIFFUSERS_AVAILABLE
})
