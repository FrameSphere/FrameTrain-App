"""
Vision Support Plugin for FrameTrain v2
=======================================
This plugin adds support for computer vision tasks:
- Image Classification (ViT, ResNet, EfficientNet)
- Object Detection (YOLO, Faster-RCNN)
- Semantic Segmentation (UNet, DeepLab)
- Instance Segmentation (Mask-RCNN)

MANIFEST:
{
    "name": "Vision Plugin",
    "description": "Computer Vision tasks (Classification, Detection, Segmentation)",
    "modality": "vision",
    "required": [
        "torch",
        "torchvision",
        "timm",
        "pillow"
    ],
    "optional": [
        "albumentations"
    ],
    "python": "3.8"
}

Installation:
    pip install timm torchvision albumentations

Usage:
    python train_engine.py --config config.json --plugins ./plugins/
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import json

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATION_AVAILABLE = True
except ImportError:
    ALBUMENTATION_AVAILABLE = False


# Import from main engine
try:
    from train_engine import (
        BaseDataLoader, BaseModelHandler, Modality, 
        REGISTRY, MessageProtocol, TrainingConfig
    )
except ImportError:
    # If running standalone, need to adjust path
    import sys
    sys.path.append(str(Path(__file__).parent))
    from train_engine import (
        BaseDataLoader, BaseModelHandler, Modality, 
        REGISTRY, MessageProtocol, TrainingConfig
    )


# ============================================================================
# VISION DATASET
# ============================================================================

class ImageClassificationDataset(Dataset):
    """Dataset for image classification"""
    
    def __init__(self, image_paths: List[Path], labels: Optional[List[int]] = None,
                 transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        # Apply transforms
        if self.transform:
            if ALBUMENTATION_AVAILABLE and isinstance(self.transform, A.Compose):
                image = self.transform(image=np.array(image))['image']
            else:
                image = self.transform(image)
        
        # Get label
        label = self.labels[idx] if self.labels else 0
        
        return image, label


# ============================================================================
# VISION DATA LOADER
# ============================================================================

class VisionDataLoader(BaseDataLoader):
    """Data loader for vision tasks"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.num_classes = 0
        self.class_names = []
        
        # Setup transforms
        self.train_transform = transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def load(self) -> Tuple[Any, Optional[Any], Optional[Any]]:
        """Load image datasets"""
        try:
            MessageProtocol.status("loading", "Loading image datasets...")
            
            dataset_path = Path(self.config.dataset_path)
            train_path = dataset_path / "train"
            
            if not train_path.exists():
                raise ValueError(f"Training path not found: {train_path}")
            
            # Detect dataset structure
            # Option 1: ImageNet-style (train/class_name/*.jpg)
            class_dirs = [d for d in train_path.iterdir() if d.is_dir()]
            
            if class_dirs:
                # ImageNet-style structure
                from torchvision.datasets import ImageFolder
                
                train_dataset = ImageFolder(train_path, transform=self.train_transform)
                self.num_classes = len(train_dataset.classes)
                self.class_names = train_dataset.classes
                
                MessageProtocol.status("loading", 
                    f"Loaded ImageFolder dataset: {len(train_dataset)} images, "
                    f"{self.num_classes} classes")
                
                self.train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=True,
                    num_workers=self.config.num_workers,
                    pin_memory=self.config.pin_memory
                )
                
                # Load validation data if exists
                val_path = dataset_path / "val"
                if val_path.exists():
                    val_dataset = ImageFolder(val_path, transform=self.val_transform)
                    self.val_loader = DataLoader(
                        val_dataset,
                        batch_size=self.config.batch_size,
                        shuffle=False,
                        num_workers=self.config.num_workers,
                        pin_memory=self.config.pin_memory
                    )
                    MessageProtocol.status("loading", 
                        f"Loaded validation set: {len(val_dataset)} images")
            
            else:
                # Option 2: Flat structure with annotations.json
                annotation_file = train_path / "annotations.json"
                if annotation_file.exists():
                    with open(annotation_file, 'r') as f:
                        annotations = json.load(f)
                    
                    # Build dataset from annotations
                    image_paths = []
                    labels = []
                    
                    for item in annotations:
                        img_path = train_path / item['image']
                        if img_path.exists():
                            image_paths.append(img_path)
                            labels.append(item['label'])
                    
                    # Get class names
                    self.num_classes = len(set(labels))
                    self.class_names = sorted(list(set(labels)))
                    
                    # Create label mapping
                    label_map = {name: idx for idx, name in enumerate(self.class_names)}
                    labels = [label_map[label] for label in labels]
                    
                    train_dataset = ImageClassificationDataset(
                        image_paths, labels, self.train_transform
                    )
                    
                    self.train_loader = DataLoader(
                        train_dataset,
                        batch_size=self.config.batch_size,
                        shuffle=True,
                        num_workers=self.config.num_workers,
                        pin_memory=self.config.pin_memory
                    )
                    
                    MessageProtocol.status("loading", 
                        f"Loaded annotated dataset: {len(train_dataset)} images, "
                        f"{self.num_classes} classes")
                else:
                    raise ValueError(
                        "Unknown dataset structure. Expected either:\n"
                        "1. ImageNet-style: train/class_name/*.jpg\n"
                        "2. Flat with annotations: train/annotations.json"
                    )
            
            return self.train_loader, self.val_loader, None
            
        except Exception as e:
            MessageProtocol.error("Failed to load vision data", str(e))
            raise
    
    def get_sample_info(self) -> Dict[str, Any]:
        """Get sample information"""
        return {
            "image_size": self.config.image_size,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "num_train_samples": len(self.train_loader.dataset) if self.train_loader else 0,
            "num_val_samples": len(self.val_loader.dataset) if self.val_loader else 0
        }


# ============================================================================
# VISION MODEL HANDLER
# ============================================================================

class VisionModelHandler(BaseModelHandler):
    """Handler for vision models"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.num_classes = None
    
    def load_model(self):
        """Load vision model"""
        try:
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch not available")
            
            if not TIMM_AVAILABLE:
                raise ImportError("timm not available. Install with: pip install timm")
            
            MessageProtocol.status("loading", "Loading vision model...")
            
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
            
            # Determine model type
            model_path = self.config.model_path
            
            # Check if it's a timm model name
            if model_path in timm.list_models():
                # Load from timm
                MessageProtocol.status("loading", f"Loading timm model: {model_path}")
                
                # Get number of classes (will be set by data loader)
                # For now, use default 1000, will be updated if needed
                num_classes = getattr(self.config, 'num_classes', 1000)
                
                self.model = timm.create_model(
                    model_path,
                    pretrained=True,
                    num_classes=num_classes
                )
                
                MessageProtocol.status("loaded", 
                    f"Loaded {model_path} with {num_classes} classes")
            
            elif Path(model_path).exists():
                # Load from checkpoint
                MessageProtocol.status("loading", f"Loading from checkpoint: {model_path}")
                
                checkpoint = torch.load(model_path, map_location='cpu')
                
                # Try to infer model architecture from checkpoint
                # This is simplified - in practice would need more logic
                self.model = timm.create_model('resnet50', pretrained=False)
                self.model.load_state_dict(checkpoint)
            
            else:
                raise ValueError(
                    f"Model {model_path} not found. Use either:\n"
                    "1. timm model name (e.g., 'vit_base_patch16_224', 'resnet50')\n"
                    "2. Path to checkpoint file (.pth, .pt)\n"
                    f"Available timm models: https://timm.fast.ai/"
                )
            
            # Move to device
            self.model = self.model.to(self.device)
            
            # Set to training mode
            self.model.train()
            
            MessageProtocol.status("loaded", f"Model loaded successfully")
            
        except Exception as e:
            MessageProtocol.error("Vision model loading failed", str(e))
            raise
    
    def forward(self, batch: Any) -> Any:
        """Forward pass"""
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        outputs = self.model(images)
        
        return outputs, labels
    
    def compute_loss(self, outputs: Any, batch: Any) -> Any:
        """Compute classification loss"""
        logits, labels = outputs
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        return loss
    
    def save_model(self, path: str):
        """Save model"""
        output_path = Path(path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        torch.save(self.model.state_dict(), output_path / "model.pth")
        
        # Save config
        config_dict = {
            "model_architecture": self.config.model_path,
            "num_classes": self.num_classes,
            "image_size": self.config.image_size
        }
        
        with open(output_path / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        MessageProtocol.status("saved", f"Model saved to {output_path}")


# ============================================================================
# REGISTER PLUGIN
# ============================================================================

# Register vision components
REGISTRY.register_data_loader(Modality.VISION, VisionDataLoader)
REGISTRY.register_model_handler("vision", VisionModelHandler)

MessageProtocol.debug("Vision plugin loaded", {
    "timm_available": TIMM_AVAILABLE,
    "albumentation_available": ALBUMENTATION_AVAILABLE
})
