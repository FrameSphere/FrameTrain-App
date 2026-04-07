"""
Segmentation Test Plugin for FrameTrain v2
==========================================
Testing support for image segmentation models:
- Semantic Segmentation
- Instance Segmentation

MANIFEST:
{
    "name": "Segmentation Test Plugin",
    "description": "Testing for Image Segmentation models",
    "modality": "segmentation",
    "required": ["torch", "torchvision", "pillow", "numpy"],
    "optional": [],
    "python": "3.8"
}
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from PIL import Image
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from test_engine import BaseTestLoader, Modality, TEST_REGISTRY, MessageProtocol, TestConfig


class SegmentationTestLoader(BaseTestLoader):
    """Test loader for segmentation models"""
    
    def __init__(self, config: TestConfig):
        super().__init__(config)
        self.num_classes = 21  # VOC default
        
    def load_model(self):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch and torchvision required")
        
        self.device = self.get_device()
        MessageProtocol.status("loading", "Loading segmentation model...")
        
        # Simple UNet-like model
        class SimpleSegNet(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU()
                )
                self.decoder = nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(128, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, num_classes, 1)
                )
            
            def forward(self, x):
                x = self.encoder(x)
                x = self.decoder(x)
                return x
        
        model_path = Path(self.config.model_path)
        checkpoint = model_path / "model.pth"
        
        self.model = SimpleSegNet(self.num_classes)
        if checkpoint.exists():
            self.model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
        
        self.model.to(self.device).eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        MessageProtocol.status("loaded", f"Segmentation model loaded ({self.num_classes} classes)")
    
    def load_test_data(self) -> List[Dict[str, Any]]:
        test_path = Path(self.config.dataset_path) / "test"
        images_path = test_path / "images"
        masks_path = test_path / "masks"
        
        if not images_path.exists():
            images_path = test_path
        
        test_data = []
        for img_file in images_path.glob("*.jpg") + list(images_path.glob("*.png")):
            mask_file = None
            if masks_path.exists():
                mask_file = masks_path / f"{img_file.stem}.png"
                if not mask_file.exists():
                    mask_file = None
            
            test_data.append({
                "image_path": str(img_file),
                "mask_path": str(mask_file) if mask_file else None
            })
        
        if self.config.max_samples:
            test_data = test_data[:self.config.max_samples]
        
        MessageProtocol.status("loaded", f"Loaded {len(test_data)} images")
        return test_data
    
    def test_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            image = Image.open(sample["image_path"]).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(image_tensor)
                pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            
            result = {
                "image_path": sample["image_path"],
                "predicted_mask_shape": pred_mask.shape,
                "inference_time": time.time() - start_time
            }
            
            # Compute IoU if ground truth available
            if sample["mask_path"]:
                gt_mask = np.array(Image.open(sample["mask_path"]).resize((256, 256)))
                
                iou_scores = []
                for class_id in range(self.num_classes):
                    pred_class = (pred_mask == class_id)
                    gt_class = (gt_mask == class_id)
                    
                    intersection = np.logical_and(pred_class, gt_class).sum()
                    union = np.logical_or(pred_class, gt_class).sum()
                    
                    if union > 0:
                        iou_scores.append(intersection / union)
                
                result["mean_iou"] = np.mean(iou_scores) if iou_scores else 0.0
                result["has_ground_truth"] = True
            else:
                result["has_ground_truth"] = False
            
            return result
            
        except Exception as e:
            return {"image_path": sample["image_path"], "error": str(e)}
    
    def compute_metrics(self, all_results: List[Dict]) -> Dict[str, Any]:
        valid = [r for r in all_results if "error" not in r]
        
        if not valid:
            return {"error": "No valid predictions"}
        
        results_with_gt = [r for r in valid if r.get("has_ground_truth")]
        
        metrics = {
            "total_samples": len(valid),
            "average_inference_time": sum(r["inference_time"] for r in valid) / len(valid)
        }
        
        if results_with_gt:
            mean_iou = sum(r["mean_iou"] for r in results_with_gt) / len(results_with_gt)
            metrics["mean_iou"] = mean_iou
            metrics["samples_with_ground_truth"] = len(results_with_gt)
        
        return metrics


TEST_REGISTRY.register_test_loader(Modality.SEGMENTATION, SegmentationTestLoader)
MessageProtocol.debug("Segmentation test plugin loaded", {"torch": TORCH_AVAILABLE})
