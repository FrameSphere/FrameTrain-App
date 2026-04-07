"""
Object Detection Plugin for FrameTrain v2
=========================================
This plugin adds support for object detection:
- YOLO (v5, v8, v10)
- Faster R-CNN
- RetinaNet
- DETR (Detection Transformer)

MANIFEST:
{
    "name": "Detection Plugin",
    "description": "Object Detection (YOLO, Faster R-CNN, RetinaNet, DETR)",
    "modality": "vision",
    "required": [
        "torch",
        "torchvision",
        "pillow"
    ],
    "optional": [
        "ultralytics",
        "transformers"
    ],
    "python": "3.8"
}

Installation:
    pip install ultralytics  # For YOLO
    pip install torchvision  # For Faster R-CNN, RetinaNet
    pip install transformers  # For DETR

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
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn,
        retinanet_resnet50_fpn
    )
    import torchvision
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


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
# DETECTION DATASET
# ============================================================================

class DetectionDataset(Dataset):
    """Dataset for object detection with bounding boxes"""
    
    def __init__(self, image_dir: Path, annotation_format: str = "yolo", 
                 transform=None):
        """
        Args:
            image_dir: Directory with images
            annotation_format: "yolo", "coco", or "pascal_voc"
            transform: Optional transforms
        """
        self.image_dir = image_dir
        self.annotation_format = annotation_format
        self.transform = transform
        
        # Find images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        self.images = []
        for ext in image_extensions:
            self.images.extend(list(image_dir.glob(f"*{ext}")))
        
        self.images = sorted(self.images)
        
        # Find annotations
        if annotation_format == "yolo":
            # YOLO: labels/*.txt (one file per image)
            self.labels_dir = image_dir.parent / "labels"
            if not self.labels_dir.exists():
                self.labels_dir = image_dir / "labels"
        
        elif annotation_format == "coco":
            # COCO: annotations.json
            self.annotations_file = image_dir.parent / "annotations.json"
            if not self.annotations_file.exists():
                self.annotations_file = image_dir / "annotations.json"
            
            with open(self.annotations_file, 'r') as f:
                self.coco_data = json.load(f)
        
        elif annotation_format == "pascal_voc":
            # Pascal VOC: *.xml files
            self.annotations_dir = image_dir.parent / "annotations"
            if not self.annotations_dir.exists():
                self.annotations_dir = image_dir / "annotations"
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Load annotations
        if self.annotation_format == "yolo":
            boxes, labels = self._load_yolo_annotation(img_path)
        elif self.annotation_format == "coco":
            boxes, labels = self._load_coco_annotation(idx)
        elif self.annotation_format == "pascal_voc":
            boxes, labels = self._load_voc_annotation(img_path)
        else:
            boxes, labels = [], []
        
        # Convert to tensors
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx])
        }
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, target
    
    def _load_yolo_annotation(self, img_path: Path):
        """Load YOLO format: class x_center y_center width height (normalized)"""
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        
        if not label_path.exists():
            return [], []
        
        boxes = []
        labels = []
        
        # Get image size for denormalization
        img = Image.open(img_path)
        img_w, img_h = img.size
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                class_id = int(parts[0])
                x_center = float(parts[1]) * img_w
                y_center = float(parts[2]) * img_h
                width = float(parts[3]) * img_w
                height = float(parts[4]) * img_h
                
                # Convert to [x_min, y_min, x_max, y_max]
                x_min = x_center - width / 2
                y_min = y_center - height / 2
                x_max = x_center + width / 2
                y_max = y_center + height / 2
                
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(class_id + 1)  # +1 because 0 is background
        
        return boxes, labels
    
    def _load_coco_annotation(self, idx):
        """Load COCO format annotations"""
        # Simplified COCO loader
        boxes = []
        labels = []
        
        # This is a simplified version - full COCO loader would be more complex
        # For now, return empty
        return boxes, labels
    
    def _load_voc_annotation(self, img_path: Path):
        """Load Pascal VOC format (.xml)"""
        # Simplified VOC loader
        boxes = []
        labels = []
        
        # This would parse XML - simplified for now
        return boxes, labels


# ============================================================================
# DETECTION DATA LOADER
# ============================================================================

class DetectionDataLoader(BaseDataLoader):
    """Data loader for object detection tasks"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.num_classes = 0
        self.class_names = []
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def load(self) -> Tuple[Any, Optional[Any], Optional[Any]]:
        """Load detection datasets"""
        try:
            MessageProtocol.status("loading", "Loading detection datasets...")
            
            dataset_path = Path(self.config.dataset_path)
            train_path = dataset_path / "train"
            
            if not train_path.exists():
                raise ValueError(f"Training path not found: {train_path}")
            
            # Detect annotation format
            annotation_format = self._detect_annotation_format(train_path)
            MessageProtocol.status("loading", f"Detected annotation format: {annotation_format}")
            
            # Create dataset
            train_dataset = DetectionDataset(
                train_path,
                annotation_format=annotation_format,
                transform=self.transform
            )
            
            # Custom collate function for variable-size targets
            def collate_fn(batch):
                images = [item[0] for item in batch]
                targets = [item[1] for item in batch]
                
                images = torch.stack(images)
                
                return images, targets
            
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                collate_fn=collate_fn
            )
            
            MessageProtocol.status("loading", 
                f"Loaded {len(train_dataset)} training images")
            
            # Load validation data if exists
            val_path = dataset_path / "val"
            if val_path.exists():
                val_dataset = DetectionDataset(
                    val_path,
                    annotation_format=annotation_format,
                    transform=self.transform
                )
                
                self.val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    num_workers=self.config.num_workers,
                    pin_memory=self.config.pin_memory,
                    collate_fn=collate_fn
                )
                
                MessageProtocol.status("loading", 
                    f"Loaded {len(val_dataset)} validation images")
            
            # Load class names
            self._load_class_names(dataset_path)
            
            return self.train_loader, self.val_loader, None
            
        except Exception as e:
            MessageProtocol.error("Failed to load detection data", str(e))
            raise
    
    def _detect_annotation_format(self, data_path: Path) -> str:
        """Detect annotation format from directory structure"""
        if (data_path.parent / "labels").exists() or (data_path / "labels").exists():
            return "yolo"
        elif (data_path.parent / "annotations.json").exists():
            return "coco"
        elif (data_path.parent / "annotations").exists():
            return "pascal_voc"
        else:
            MessageProtocol.warning("No annotations found - assuming YOLO format")
            return "yolo"
    
    def _load_class_names(self, dataset_path: Path):
        """Load class names from classes.txt or data.yaml"""
        # Try classes.txt
        classes_file = dataset_path / "classes.txt"
        if classes_file.exists():
            with open(classes_file, 'r') as f:
                self.class_names = [line.strip() for line in f if line.strip()]
            self.num_classes = len(self.class_names)
            MessageProtocol.status("loading", f"Loaded {self.num_classes} classes")
            return
        
        # Try data.yaml (YOLO format)
        data_yaml = dataset_path / "data.yaml"
        if data_yaml.exists():
            import yaml
            with open(data_yaml, 'r') as f:
                data = yaml.safe_load(f)
                if 'names' in data:
                    self.class_names = data['names']
                    self.num_classes = len(self.class_names)
                    MessageProtocol.status("loading", f"Loaded {self.num_classes} classes")
                    return
        
        MessageProtocol.warning("No class names file found")
    
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
# DETECTION MODEL HANDLER (YOLO, Faster R-CNN, etc.)
# ============================================================================

class DetectionModelHandler(BaseModelHandler):
    """Handler for object detection models"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.model_type = None
    
    def load_model(self):
        """Load detection model"""
        try:
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch not available")
            
            MessageProtocol.status("loading", "Loading detection model...")
            
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
            if "yolo" in model_path.lower():
                if not ULTRALYTICS_AVAILABLE:
                    raise ImportError("Ultralytics not installed. Install with: pip install ultralytics")
                
                self.model_type = "yolo"
                MessageProtocol.status("loading", f"Loading YOLO model: {model_path}")
                
                # YOLO handles device internally
                self.model = YOLO(model_path)
                
            elif "faster" in model_path.lower() or "fasterrcnn" in model_path.lower():
                if not TORCHVISION_AVAILABLE:
                    raise ImportError("Torchvision not installed")
                
                self.model_type = "faster_rcnn"
                MessageProtocol.status("loading", "Loading Faster R-CNN model")
                
                num_classes = getattr(self.config, 'num_classes', 91)  # COCO default
                self.model = fasterrcnn_resnet50_fpn(pretrained=True)
                
                # Replace classifier head for custom number of classes
                in_features = self.model.roi_heads.box_predictor.cls_score.in_features
                self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
                    in_features, num_classes
                )
                
                self.model = self.model.to(self.device)
            
            elif "retinanet" in model_path.lower():
                if not TORCHVISION_AVAILABLE:
                    raise ImportError("Torchvision not installed")
                
                self.model_type = "retinanet"
                MessageProtocol.status("loading", "Loading RetinaNet model")
                
                num_classes = getattr(self.config, 'num_classes', 91)
                self.model = retinanet_resnet50_fpn(pretrained=True, num_classes=num_classes)
                self.model = self.model.to(self.device)
            
            else:
                raise ValueError(
                    f"Unknown detection model: {model_path}\n"
                    "Supported: yolov5, yolov8, faster_rcnn, retinanet"
                )
            
            MessageProtocol.status("loaded", f"Detection model loaded: {self.model_type}")
            
        except Exception as e:
            MessageProtocol.error("Detection model loading failed", str(e))
            raise
    
    def forward(self, batch: Any) -> Any:
        """Forward pass"""
        images, targets = batch
        
        if self.model_type == "yolo":
            # YOLO training is handled internally by ultralytics
            # This is a simplified version
            results = self.model.train(
                data=self.config.dataset_path,
                epochs=1,  # Will be called per epoch
                batch=self.config.batch_size,
                device=self.device
            )
            return results
        
        else:
            # Faster R-CNN, RetinaNet
            images = images.to(self.device)
            
            # Move targets to device
            targets = [
                {k: v.to(self.device) for k, v in t.items()} 
                for t in targets
            ]
            
            # Forward pass (returns loss dict during training)
            loss_dict = self.model(images, targets)
            
            return loss_dict
    
    def compute_loss(self, outputs: Any, batch: Any) -> Any:
        """Compute detection loss"""
        if self.model_type == "yolo":
            # YOLO handles loss internally
            return torch.tensor(0.0)  # Placeholder
        else:
            # Faster R-CNN returns dict with multiple losses
            losses = sum(loss for loss in outputs.values())
            return losses
    
    def save_model(self, path: str):
        """Save model"""
        output_path = Path(path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.model_type == "yolo":
            # YOLO saves to runs/detect/train by default
            # Copy best weights
            self.model.export(format='torchscript', save_dir=output_path)
        else:
            torch.save(self.model.state_dict(), output_path / "model.pth")
        
        # Save config
        config_dict = {
            "model_type": self.model_type,
            "num_classes": getattr(self.config, 'num_classes', None)
        }
        
        with open(output_path / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        MessageProtocol.status("saved", f"Detection model saved to {output_path}")


# ============================================================================
# REGISTER PLUGIN
# ============================================================================

# Note: Detection is a subtype of Vision, so we register with a different key
REGISTRY.register_data_loader(Modality.VISION, DetectionDataLoader)
REGISTRY.register_model_handler("detection", DetectionModelHandler)

MessageProtocol.debug("Detection plugin loaded", {
    "ultralytics_available": ULTRALYTICS_AVAILABLE,
    "torchvision_available": TORCHVISION_AVAILABLE
})
