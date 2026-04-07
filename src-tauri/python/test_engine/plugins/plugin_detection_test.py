"""
Detection Test Plugin for FrameTrain v2
=======================================
This plugin adds testing support for object detection:
- YOLO (v5, v8, v10)
- Computes mAP, Precision, Recall, IoU

MANIFEST:
{
    "name": "Detection Test Plugin",
    "description": "Testing for Object Detection models (YOLO, etc.)",
    "modality": "detection",
    "required": [
        "torch",
        "torchvision",
        "pillow",
        "numpy"
    ],
    "optional": [
        "ultralytics"
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

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import numpy as np
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

from test_engine import (
    BaseTestLoader, Modality, TEST_REGISTRY,
    MessageProtocol, TestConfig
)


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def calculate_map(predictions, ground_truths, iou_threshold=0.5):
    """Calculate mAP (mean Average Precision)"""
    if not predictions or not ground_truths:
        return 0.0
    
    # Match predictions to ground truths
    tp = 0  # True positives
    fp = 0  # False positives
    fn = len(ground_truths)  # False negatives (all ground truths initially)
    
    matched_gts = set()
    
    # Sort predictions by confidence
    predictions = sorted(predictions, key=lambda x: x.get('confidence', 0), reverse=True)
    
    for pred in predictions:
        pred_box = pred['box']
        best_iou = 0
        best_gt_idx = -1
        
        for i, gt in enumerate(ground_truths):
            if i in matched_gts:
                continue
            
            gt_box = gt['box']
            iou = calculate_iou(pred_box, gt_box)
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        
        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp += 1
            fn -= 1
            matched_gts.add(best_gt_idx)
        else:
            fp += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


class DetectionTestLoader(BaseTestLoader):
    """Test loader for detection models"""
    
    def __init__(self, config: TestConfig):
        super().__init__(config)
        self.model_type = None
    
    def load_model(self):
        """Load detection model"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        MessageProtocol.status("loading", f"Loading detection model from {self.config.model_path}...")
        
        self.device = self.get_device()
        MessageProtocol.status("device", f"Using device: {self.device}")
        
        model_path = str(self.config.model_path)
        
        # Detect model type
        if "yolo" in model_path.lower() or ULTRALYTICS_AVAILABLE:
            if not ULTRALYTICS_AVAILABLE:
                raise ImportError("Ultralytics YOLO not installed. Install with: pip install ultralytics")
            
            self.model_type = "yolo"
            self.model = YOLO(model_path)
            MessageProtocol.status("loaded", "YOLO model loaded")
        else:
            raise ValueError(f"Unknown detection model type: {model_path}")
    
    def load_test_data(self) -> List[Dict[str, Any]]:
        """Load detection test data"""
        MessageProtocol.status("loading", "Loading test images and annotations...")
        
        test_path = Path(self.config.dataset_path) / "test"
        if not test_path.exists():
            test_path = Path(self.config.dataset_path) / "val"
        if not test_path.exists():
            raise ValueError(f"Test data not found")
        
        test_data = []
        
        # Look for images
        images_path = test_path / "images" if (test_path / "images").exists() else test_path
        labels_path = test_path / "labels" if (test_path / "labels").exists() else test_path
        
        for img_file in images_path.glob("*"):
            if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue
            
            # Look for corresponding label file
            label_file = labels_path / f"{img_file.stem}.txt"
            
            annotations = []
            if label_file.exists():
                # Parse YOLO format: class x_center y_center width height (normalized)
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            annotations.append({
                                'class': int(parts[0]),
                                'x_center': float(parts[1]),
                                'y_center': float(parts[2]),
                                'width': float(parts[3]),
                                'height': float(parts[4])
                            })
            
            test_data.append({
                "image_path": str(img_file),
                "annotations": annotations
            })
        
        if not test_data:
            raise ValueError("No test images found")
        
        # Limit samples
        if self.config.max_samples and len(test_data) > self.config.max_samples:
            test_data = test_data[:self.config.max_samples]
        
        MessageProtocol.status("loaded", f"Loaded {len(test_data)} test images")
        return test_data
    
    def test_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Test a single image for detection"""
        image_path = sample["image_path"]
        ground_truth = sample["annotations"]
        
        start_time = time.time()
        
        try:
            # Load image to get dimensions
            image = Image.open(image_path)
            img_width, img_height = image.size
            
            # Run detection
            if self.model_type == "yolo":
                results = self.model(image_path, verbose=False)
                result = results[0]
                
                # Extract predictions
                predictions = []
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        # Get box coordinates (xyxy format)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())
                        
                        predictions.append({
                            'box': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': conf,
                            'class': cls
                        })
            else:
                predictions = []
            
            # Convert ground truth to absolute coordinates
            gt_boxes = []
            for ann in ground_truth:
                x_center = ann['x_center'] * img_width
                y_center = ann['y_center'] * img_height
                width = ann['width'] * img_width
                height = ann['height'] * img_height
                
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                
                gt_boxes.append({
                    'box': [x1, y1, x2, y2],
                    'class': ann['class']
                })
            
            inference_time = time.time() - start_time
            
            # Calculate metrics for this image
            metrics_50 = calculate_map(predictions, gt_boxes, iou_threshold=0.5)
            metrics_75 = calculate_map(predictions, gt_boxes, iou_threshold=0.75)
            
            return {
                "image_path": image_path,
                "predictions": predictions,
                "ground_truth": gt_boxes,
                "num_predictions": len(predictions),
                "num_ground_truth": len(gt_boxes),
                "metrics_iou50": metrics_50,
                "metrics_iou75": metrics_75,
                "inference_time": inference_time
            }
            
        except Exception as e:
            MessageProtocol.warning(f"Failed to process {image_path}: {e}")
            return {
                "image_path": image_path,
                "error": str(e),
                "inference_time": time.time() - start_time
            }
    
    def compute_metrics(self, all_results: List[Dict]) -> Dict[str, Any]:
        """Compute detection metrics"""
        valid_results = [r for r in all_results if "error" not in r]
        
        if not valid_results:
            return {"error": "No valid predictions"}
        
        # Aggregate metrics
        total_tp_50 = sum(r['metrics_iou50']['tp'] for r in valid_results)
        total_fp_50 = sum(r['metrics_iou50']['fp'] for r in valid_results)
        total_fn_50 = sum(r['metrics_iou50']['fn'] for r in valid_results)
        
        total_tp_75 = sum(r['metrics_iou75']['tp'] for r in valid_results)
        total_fp_75 = sum(r['metrics_iou75']['fp'] for r in valid_results)
        total_fn_75 = sum(r['metrics_iou75']['fn'] for r in valid_results)
        
        # Calculate overall precision/recall
        precision_50 = total_tp_50 / (total_tp_50 + total_fp_50) if (total_tp_50 + total_fp_50) > 0 else 0
        recall_50 = total_tp_50 / (total_tp_50 + total_fn_50) if (total_tp_50 + total_fn_50) > 0 else 0
        f1_50 = 2 * (precision_50 * recall_50) / (precision_50 + recall_50) if (precision_50 + recall_50) > 0 else 0
        
        precision_75 = total_tp_75 / (total_tp_75 + total_fp_75) if (total_tp_75 + total_fp_75) > 0 else 0
        recall_75 = total_tp_75 / (total_tp_75 + total_fn_75) if (total_tp_75 + total_fn_75) > 0 else 0
        f1_75 = 2 * (precision_75 * recall_75) / (precision_75 + recall_75) if (precision_75 + recall_75) > 0 else 0
        
        # mAP (average of mAP@50 and mAP@75)
        map_50_75 = (f1_50 + f1_75) / 2
        
        # Average inference time
        inference_times = [r['inference_time'] for r in valid_results]
        avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
        
        return {
            "mAP@50": f1_50 * 100,
            "mAP@75": f1_75 * 100,
            "mAP@50:75": map_50_75 * 100,
            "precision@50": precision_50 * 100,
            "recall@50": recall_50 * 100,
            "precision@75": precision_75 * 100,
            "recall@75": recall_75 * 100,
            "total_samples": len(valid_results),
            "total_predictions": sum(r['num_predictions'] for r in valid_results),
            "total_ground_truth": sum(r['num_ground_truth'] for r in valid_results),
            "average_inference_time": avg_inference_time,
            "true_positives@50": total_tp_50,
            "false_positives@50": total_fp_50,
            "false_negatives@50": total_fn_50
        }


# Register detection test loader
TEST_REGISTRY.register_test_loader(Modality.DETECTION, DetectionTestLoader)

MessageProtocol.debug("Detection test plugin loaded", {
    "torch_available": TORCH_AVAILABLE,
    "ultralytics_available": ULTRALYTICS_AVAILABLE
})
