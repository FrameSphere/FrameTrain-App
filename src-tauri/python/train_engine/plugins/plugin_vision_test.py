"""
Vision Plugin for FrameTrain - Image Classification and Detection
==================================================================

MANIFEST:
{
    "name": "Vision Plugin",
    "description": "Computer Vision tasks - Image Classification, Object Detection, Segmentation",
    "modality": "vision",
    "required": [
        "torch",
        "torchvision",
        "pillow"
    ],
    "optional": [
        "opencv-python",
        "albumentations"
    ],
    "python": "3.8"
}
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import json

# These imports will only work if dependencies are installed
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms, models
    from PIL import Image
    VISION_AVAILABLE = True
except ImportError as e:
    VISION_AVAILABLE = False
    print(f"[Vision Plugin] Warning: Missing dependencies: {e}")


# ============================================================================
# VISION DATA LOADER
# ============================================================================

class VisionDataLoader:
    """Data loader for vision tasks"""
    
    def __init__(self, config):
        self.config = config
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        if not VISION_AVAILABLE:
            raise ImportError("Vision dependencies not installed. Install: torch, torchvision, pillow")
    
    def load(self) -> Tuple[Any, Optional[Any], Optional[Any]]:
        """Load vision dataset"""
        print("[Vision] Loading image dataset...")
        
        # Image transforms
        transform = transforms.Compose([
            transforms.Resize(self.config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load train data
        train_path = Path(self.config.dataset_path) / "train"
        train_dataset = ImageFolderDataset(train_path, transform=transform)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        # Load validation data if exists
        val_path = Path(self.config.dataset_path) / "val"
        if val_path.exists():
            val_dataset = ImageFolderDataset(val_path, transform=transform)
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory
            )
        
        print(f"[Vision] Loaded {len(train_dataset)} training images")
        
        return self.train_loader, self.val_loader, None
    
    def get_sample_info(self) -> Dict[str, Any]:
        """Get sample information"""
        return {
            "image_size": self.config.image_size,
            "num_classes": len(self.train_loader.dataset.classes) if hasattr(self.train_loader.dataset, 'classes') else None
        }


class ImageFolderDataset(Dataset):
    """Simple image folder dataset"""
    
    def __init__(self, root_dir: Path, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.classes = []
        
        # Load images from class folders
        if root_dir.exists():
            class_dirs = [d for d in root_dir.iterdir() if d.is_dir()]
            self.classes = [d.name for d in class_dirs]
            
            for class_idx, class_dir in enumerate(class_dirs):
                for img_file in class_dir.glob("*"):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        self.images.append(img_file)
                        self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'pixel_values': image,
            'labels': torch.tensor(label, dtype=torch.long)
        }


# ============================================================================
# VISION MODEL HANDLER
# ============================================================================

class VisionModelHandler:
    """Handler for vision models"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.device = None
        
        if not VISION_AVAILABLE:
            raise ImportError("Vision dependencies not installed")
    
    def load_model(self):
        """Load vision model"""
        print("[Vision] Loading model...")
        
        # Determine device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"[Vision] Using GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("[Vision] Using Apple Silicon GPU (MPS)")
        else:
            self.device = torch.device("cpu")
            print("[Vision] Using CPU")
        
        # Load pretrained model (example: ResNet)
        if 'resnet' in self.config.model_path.lower():
            self.model = models.resnet50(pretrained=True)
            # Modify final layer for custom number of classes
            # num_classes = ... (would need to be passed in config)
        else:
            # Load custom model from path
            self.model = torch.load(self.config.model_path, map_location=self.device)
        
        self.model = self.model.to(self.device)
        self.model.train()
        
        print(f"[Vision] Model loaded: {type(self.model).__name__}")
    
    def forward(self, batch: Any) -> Any:
        """Forward pass"""
        images = batch['pixel_values'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        outputs = self.model(images)
        return outputs
    
    def compute_loss(self, outputs: Any, batch: Any) -> Any:
        """Compute loss"""
        labels = batch['labels'].to(self.device)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        return loss
    
    def save_model(self, path: str):
        """Save model"""
        output_path = Path(path)
        output_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), output_path / "model.pth")
        print(f"[Vision] Model saved to {output_path}")


# ============================================================================
# REGISTRATION
# ============================================================================

def register_plugin(registry):
    """Register vision plugin components"""
    if not VISION_AVAILABLE:
        print("[Vision Plugin] Skipping registration - dependencies not installed")
        return
    
    from train_engine import Modality
    
    registry.register_data_loader(Modality.VISION, VisionDataLoader)
    registry.register_model_handler("vision", VisionModelHandler)
    
    print("[Vision Plugin] ✓ Registered successfully")


# Auto-register if imported
try:
    from train_engine import REGISTRY
    register_plugin(REGISTRY)
except ImportError:
    print("[Vision Plugin] Registry not available, skipping auto-registration")
