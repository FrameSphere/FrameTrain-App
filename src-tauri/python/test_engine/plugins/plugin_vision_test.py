"""
Vision Test Plugin for FrameTrain v2
====================================
This plugin adds testing support for computer vision tasks:
- Image Classification (ViT, ResNet, EfficientNet)
- Supports timm models and torchvision models

MANIFEST:
{
    "name": "Vision Test Plugin",
    "description": "Testing for Image Classification models",
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

Usage:
    python test_engine.py --config test_config.json
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from PIL import Image
    import timm
    TORCH_AVAILABLE = True
    TIMM_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    TIMM_AVAILABLE = False
    print(f"Warning: {e}")

from test_engine import (
    BaseTestLoader, Modality, TEST_REGISTRY,
    MessageProtocol, TestConfig
)


class VisionTestLoader(BaseTestLoader):
    """Test loader for vision models"""
    
    def __init__(self, config: TestConfig):
        super().__init__(config)
        self.transform = None
        self.class_names = []
        self.num_classes = 0
    
    def load_model(self):
        """Load vision model"""
        if not TORCH_AVAILABLE or not TIMM_AVAILABLE:
            raise ImportError("PyTorch and timm are required. Install with: pip install torch timm")
        
        MessageProtocol.status("loading", f"Loading vision model from {self.config.model_path}...")
        
        self.device = self.get_device()
        MessageProtocol.status("device", f"Using device: {self.device}")
        
        model_path = Path(self.config.model_path)
        
        # Check if it's a timm model or checkpoint
        if model_path.is_dir():
            # Load from checkpoint
            checkpoint_file = model_path / "model.pth"
            config_file = model_path / "config.json"
            
            if config_file.exists():
                import json
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    model_arch = config.get("model_architecture", "resnet50")
                    self.num_classes = config.get("num_classes", 1000)
            else:
                model_arch = "resnet50"
                self.num_classes = 1000
            
            self.model = timm.create_model(model_arch, pretrained=False, num_classes=self.num_classes)
            
            if checkpoint_file.exists():
                state_dict = torch.load(checkpoint_file, map_location='cpu')
                self.model.load_state_dict(state_dict)
        else:
            # Assume it's a timm model name
            self.model = timm.create_model(str(model_path), pretrained=True)
            self.num_classes = self.model.num_classes
        
        self.model.to(self.device)
        self.model.eval()
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        MessageProtocol.status("loaded", f"Vision model loaded with {self.num_classes} classes")
    
    def load_test_data(self) -> List[Dict[str, Any]]:
        """Load image test data"""
        MessageProtocol.status("loading", "Loading test images...")
        
        test_path = Path(self.config.dataset_path) / "test"
        if not test_path.exists():
            test_path = Path(self.config.dataset_path) / "val"
        if not test_path.exists():
            raise ValueError(f"Test data not found: {self.config.dataset_path}")
        
        test_data = []
        
        # Check if it's ImageFolder structure (class_name/image.jpg)
        class_dirs = [d for d in test_path.iterdir() if d.is_dir()]
        
        if class_dirs:
            # ImageFolder structure
            self.class_names = sorted([d.name for d in class_dirs])
            
            for class_dir in class_dirs:
                class_idx = self.class_names.index(class_dir.name)
                
                for img_file in class_dir.glob("*"):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                        test_data.append({
                            "image_path": str(img_file),
                            "label": class_idx,
                            "label_name": class_dir.name
                        })
        else:
            # Flat structure - no labels
            for img_file in test_path.glob("*"):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                    test_data.append({
                        "image_path": str(img_file),
                        "label": None,
                        "label_name": None
                    })
        
        if not test_data:
            raise ValueError("No test images found")
        
        # Limit samples
        if self.config.max_samples and len(test_data) > self.config.max_samples:
            test_data = test_data[:self.config.max_samples]
        
        MessageProtocol.status("loaded", f"Loaded {len(test_data)} test images")
        return test_data
    
    def test_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Test a single image"""
        image_path = sample["image_path"]
        true_label = sample.get("label")
        
        start_time = time.time()
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_label = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_label].item()
                
                # Get top-5 predictions
                top5_probs, top5_indices = torch.topk(probabilities[0], min(5, self.num_classes))
                top5_predictions = [
                    {"class": int(idx), "confidence": float(prob)}
                    for idx, prob in zip(top5_indices, top5_probs)
                ]
            
            inference_time = time.time() - start_time
            
            # Calculate loss if label available
            loss = None
            is_correct = False
            if true_label is not None:
                criterion = nn.CrossEntropyLoss()
                labels = torch.tensor([true_label]).to(self.device)
                loss = criterion(outputs, labels).item()
                is_correct = (predicted_label == true_label)
            
            # CRITICAL FIX: Return with Rust-compatible field names
            predicted_class_name = self.class_names[predicted_label] if self.class_names else str(predicted_label)
            true_class_name = sample.get("label_name") or (self.class_names[true_label] if true_label is not None and self.class_names else None)
            
            return {
                # Rust-compatible fields (required for database)
                "input_text": image_path,  # Use image_path as input_text
                "predicted_output": predicted_class_name,  # Use class name as predicted_output
                "expected_output": true_class_name,  # Use true class name as expected_output
                "is_correct": is_correct,
                "loss": loss,
                "confidence": confidence,
                "inference_time": inference_time,
                "error_type": None,
                # Vision-specific fields (for detailed analysis)
                "image_path": image_path,
                "predicted_label": predicted_label,
                "predicted_class": predicted_class_name,
                "true_label": true_label,
                "true_class": true_class_name,
                "top5_predictions": top5_predictions
            }
            
        except Exception as e:
            MessageProtocol.warning(f"Failed to process {image_path}: {e}")
            return {
                # Rust-compatible fields
                "input_text": image_path,
                "predicted_output": "ERROR",
                "expected_output": None,
                "is_correct": False,
                "loss": None,
                "confidence": None,
                "inference_time": time.time() - start_time,
                "error_type": str(e),
                # Vision-specific fields
                "image_path": image_path,
                "predicted_label": None,
                "error": str(e)
            }
    
    def compute_metrics(self, all_results: List[Dict]) -> Dict[str, Any]:
        """Compute vision metrics"""
        # Filter out errors
        valid_results = [r for r in all_results if r.get("predicted_label") is not None]
        
        if not valid_results:
            return {"error": "No valid predictions"}
        
        total = len(valid_results)
        
        # Accuracy
        results_with_labels = [r for r in valid_results if r.get("true_label") is not None]
        if results_with_labels:
            correct = sum(1 for r in results_with_labels if r.get("is_correct"))
            accuracy = (correct / len(results_with_labels)) * 100
            
            # Top-5 accuracy
            top5_correct = 0
            for r in results_with_labels:
                true_label = r["true_label"]
                top5_preds = [p["class"] for p in r.get("top5_predictions", [])]
                if true_label in top5_preds:
                    top5_correct += 1
            top5_accuracy = (top5_correct / len(results_with_labels)) * 100
        else:
            accuracy = None
            top5_accuracy = None
            correct = 0
        
        # Average confidence
        confidences = [r["confidence"] for r in valid_results if "confidence" in r]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Average loss
        losses = [r["loss"] for r in valid_results if r.get("loss") is not None]
        avg_loss = sum(losses) / len(losses) if losses else None
        
        # Average inference time
        inference_times = [r["inference_time"] for r in valid_results]
        avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
        
        # Per-class accuracy
        per_class_metrics = {}
        if results_with_labels and self.class_names:
            for class_idx, class_name in enumerate(self.class_names):
                class_results = [r for r in results_with_labels if r["true_label"] == class_idx]
                if class_results:
                    class_correct = sum(1 for r in class_results if r["is_correct"])
                    per_class_metrics[class_name] = {
                        "accuracy": (class_correct / len(class_results)) * 100,
                        "total": len(class_results),
                        "correct": class_correct
                    }
        
        return {
            "accuracy": accuracy,
            "top5_accuracy": top5_accuracy,
            "total_samples": total,
            "correct_predictions": correct,
            "incorrect_predictions": len(results_with_labels) - correct if results_with_labels else 0,
            "average_confidence": avg_confidence,
            "average_loss": avg_loss,
            "average_inference_time": avg_inference_time,
            "per_class_metrics": per_class_metrics,
            "num_classes": self.num_classes,
            "class_names": self.class_names
        }


# Register vision test loader
TEST_REGISTRY.register_test_loader(Modality.VISION, VisionTestLoader)

MessageProtocol.debug("Vision test plugin loaded", {
    "torch_available": TORCH_AVAILABLE,
    "timm_available": TIMM_AVAILABLE
})
