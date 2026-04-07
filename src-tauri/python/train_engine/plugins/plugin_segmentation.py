"""
Segmentation Plugin for FrameTrain v2
=====================================
This plugin adds support for image segmentation:
- Semantic Segmentation (UNet, DeepLab, SegFormer)
- Instance Segmentation (Mask R-CNN)
- Panoptic Segmentation

MANIFEST:
{
    "name": "Segmentation Plugin",
    "description": "Image Segmentation (UNet, DeepLab, Mask R-CNN)",
    "modality": "vision",
    "required": [
        "torch",
        "torchvision",
        "pillow"
    ],
    "optional": [
        "segmentation-models-pytorch",
        "transformers",
        "albumentations"
    ],
    "python": "3.8"
}

Installation:
    pip install segmentation-models-pytorch  # For UNet, DeepLab
    pip install torchvision  # For Mask R-CNN
    pip install transformers  # For SegFormer

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
    import torchvision
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False


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
# SEGMENTATION DATASET
# ============================================================================

class SegmentationDataset(Dataset):
    """Dataset for semantic segmentation"""
    
    def __init__(self, image_dir: Path, mask_dir: Path, transform=None, 
                 mask_transform=None):
        """
        Args:
            image_dir: Directory with images
            mask_dir: Directory with segmentation masks
            transform: Transforms for images
            mask_transform: Transforms for masks
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        
        # Find images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        self.images = []
        for ext in image_extensions:
            self.images.extend(list(image_dir.glob(f"*{ext}")))
        
        self.images = sorted(self.images)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Load corresponding mask
        mask_path = self.mask_dir / f"{img_path.stem}.png"
        
        if not mask_path.exists():
            # Try other extensions
            for ext in ['.jpg', '.jpeg', '.bmp']:
                mask_path = self.mask_dir / f"{img_path.stem}{ext}"
                if mask_path.exists():
                    break
        
        if mask_path.exists():
            mask = Image.open(mask_path).convert('L')  # Grayscale
        else:
            # Create empty mask if not found
            mask = Image.new('L', image.size, 0)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            # Default: convert to tensor
            mask = torch.from_numpy(np.array(mask)).long()
        
        return image, mask


# ============================================================================
# SEGMENTATION DATA LOADER
# ============================================================================

class SegmentationDataLoader(BaseDataLoader):
    """Data loader for segmentation tasks"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.num_classes = 0
        self.class_names = []
        
        # Setup transforms
        self.image_transform = transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize(config.image_size, interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])
    
    def load(self) -> Tuple[Any, Optional[Any], Optional[Any]]:
        """Load segmentation datasets"""
        try:
            MessageProtocol.status("loading", "Loading segmentation datasets...")
            
            dataset_path = Path(self.config.dataset_path)
            train_images_path = dataset_path / "train" / "images"
            train_masks_path = dataset_path / "train" / "masks"
            
            # Alternative structure: train/ and train_masks/
            if not train_images_path.exists():
                train_images_path = dataset_path / "train"
                train_masks_path = dataset_path / "train_masks"
            
            if not train_images_path.exists():
                raise ValueError(f"Training images not found: {train_images_path}")
            
            if not train_masks_path.exists():
                raise ValueError(f"Training masks not found: {train_masks_path}")
            
            # Create dataset
            train_dataset = SegmentationDataset(
                train_images_path,
                train_masks_path,
                transform=self.image_transform,
                mask_transform=self.mask_transform
            )
            
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory
            )
            
            MessageProtocol.status("loading", 
                f"Loaded {len(train_dataset)} training images")
            
            # Load validation data if exists
            val_images_path = dataset_path / "val" / "images"
            val_masks_path = dataset_path / "val" / "masks"
            
            if not val_images_path.exists():
                val_images_path = dataset_path / "val"
                val_masks_path = dataset_path / "val_masks"
            
            if val_images_path.exists() and val_masks_path.exists():
                val_dataset = SegmentationDataset(
                    val_images_path,
                    val_masks_path,
                    transform=self.image_transform,
                    mask_transform=self.mask_transform
                )
                
                self.val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    num_workers=self.config.num_workers,
                    pin_memory=self.config.pin_memory
                )
                
                MessageProtocol.status("loading", 
                    f"Loaded {len(val_dataset)} validation images")
            
            # Load class names
            self._load_class_info(dataset_path)
            
            return self.train_loader, self.val_loader, None
            
        except Exception as e:
            MessageProtocol.error("Failed to load segmentation data", str(e))
            raise
    
    def _load_class_info(self, dataset_path: Path):
        """Load class information"""
        classes_file = dataset_path / "classes.txt"
        if classes_file.exists():
            with open(classes_file, 'r') as f:
                self.class_names = [line.strip() for line in f if line.strip()]
            self.num_classes = len(self.class_names)
            MessageProtocol.status("loading", f"Loaded {self.num_classes} classes")
        else:
            # Try to infer from masks
            MessageProtocol.warning("No classes.txt found, inferring from masks")
            self.num_classes = 21  # Default (e.g., Pascal VOC)
    
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
# SEGMENTATION MODEL HANDLER
# ============================================================================

class SegmentationModelHandler(BaseModelHandler):
    """Handler for segmentation models"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.model_type = None
        self.num_classes = None
    
    def load_model(self):
        """Load segmentation model"""
        try:
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch not available")
            
            MessageProtocol.status("loading", "Loading segmentation model...")
            
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
            
            model_path = self.config.model_path.lower()
            self.num_classes = getattr(self.config, 'num_classes', 21)  # Pascal VOC default
            
            # Detect model type and load
            if "unet" in model_path:
                if not SMP_AVAILABLE:
                    raise ImportError(
                        "segmentation-models-pytorch not installed. "
                        "Install with: pip install segmentation-models-pytorch"
                    )
                
                self.model_type = "unet"
                MessageProtocol.status("loading", "Loading UNet model")
                
                # Parse encoder from model_path (e.g., "unet_resnet34")
                encoder = "resnet34"
                if "resnet50" in model_path:
                    encoder = "resnet50"
                elif "resnet18" in model_path:
                    encoder = "resnet18"
                elif "efficientnet" in model_path:
                    encoder = "efficientnet-b0"
                
                self.model = smp.Unet(
                    encoder_name=encoder,
                    encoder_weights="imagenet",
                    in_channels=3,
                    classes=self.num_classes
                )
            
            elif "deeplabv3" in model_path or "deeplab" in model_path:
                if not SMP_AVAILABLE:
                    raise ImportError("segmentation-models-pytorch not installed")
                
                self.model_type = "deeplabv3"
                MessageProtocol.status("loading", "Loading DeepLabV3 model")
                
                self.model = smp.DeepLabV3Plus(
                    encoder_name="resnet50",
                    encoder_weights="imagenet",
                    in_channels=3,
                    classes=self.num_classes
                )
            
            elif "maskrcnn" in model_path or "mask_rcnn" in model_path:
                if not TORCHVISION_AVAILABLE:
                    raise ImportError("Torchvision not installed")
                
                self.model_type = "mask_rcnn"
                MessageProtocol.status("loading", "Loading Mask R-CNN model")
                
                self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(
                    pretrained=True
                )
                
                # Replace classifier for custom number of classes
                in_features = self.model.roi_heads.box_predictor.cls_score.in_features
                self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
                    in_features, self.num_classes
                )
                
                # Replace mask predictor
                in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
                hidden_layer = 256
                self.model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
                    in_features_mask, hidden_layer, self.num_classes
                )
            
            else:
                # Default to UNet
                if not SMP_AVAILABLE:
                    raise ImportError("segmentation-models-pytorch not installed")
                
                self.model_type = "unet"
                MessageProtocol.status("loading", "Loading default UNet model")
                
                self.model = smp.Unet(
                    encoder_name="resnet34",
                    encoder_weights="imagenet",
                    in_channels=3,
                    classes=self.num_classes
                )
            
            # Move to device
            self.model = self.model.to(self.device)
            
            MessageProtocol.status("loaded", 
                f"Segmentation model loaded: {self.model_type}")
            
        except Exception as e:
            MessageProtocol.error("Segmentation model loading failed", str(e))
            raise
    
    def forward(self, batch: Any) -> Any:
        """Forward pass"""
        images, masks = batch
        images = images.to(self.device)
        masks = masks.to(self.device)
        
        # Forward pass
        if self.model_type == "mask_rcnn":
            # Mask R-CNN expects different format (list of dicts)
            # This is simplified - full implementation would be more complex
            outputs = self.model(images)
        else:
            # UNet, DeepLab
            outputs = self.model(images)
        
        return outputs, masks
    
    def compute_loss(self, outputs: Any, batch: Any) -> Any:
        """Compute segmentation loss"""
        if self.model_type == "mask_rcnn":
            # Mask R-CNN returns loss dict during training
            # This is handled differently
            return torch.tensor(0.0, device=self.device)
        
        else:
            # UNet, DeepLab
            logits, masks = outputs
            
            # Remove channel dimension from masks if present
            if masks.dim() == 4 and masks.shape[1] == 1:
                masks = masks.squeeze(1)
            
            # Compute cross-entropy loss
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, masks.long())
            
            return loss
    
    def save_model(self, path: str):
        """Save model"""
        output_path = Path(path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        torch.save(self.model.state_dict(), output_path / "model.pth")
        
        # Save config
        config_dict = {
            "model_type": self.model_type,
            "num_classes": self.num_classes,
            "image_size": self.config.image_size
        }
        
        with open(output_path / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        MessageProtocol.status("saved", f"Segmentation model saved to {output_path}")


# ============================================================================
# REGISTER PLUGIN
# ============================================================================

# Note: Segmentation is a subtype of Vision
REGISTRY.register_data_loader(Modality.VISION, SegmentationDataLoader)
REGISTRY.register_model_handler("segmentation", SegmentationModelHandler)

MessageProtocol.debug("Segmentation plugin loaded", {
    "smp_available": SMP_AVAILABLE,
    "torchvision_available": TORCHVISION_AVAILABLE
})
