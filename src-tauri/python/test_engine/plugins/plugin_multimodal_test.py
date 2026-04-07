"""
Multimodal Test Plugin for FrameTrain v2
========================================
Testing support for multimodal models:
- CLIP (Image-Text)
- Vision-Language Models

MANIFEST:
{
    "name": "Multimodal Test Plugin",
    "description": "Testing for Multimodal models (CLIP, VLMs)",
    "modality": "multimodal",
    "required": ["torch", "torchvision", "pillow"],
    "optional": ["transformers"],
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
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from test_engine import BaseTestLoader, Modality, TEST_REGISTRY, MessageProtocol, TestConfig


class MultimodalTestLoader(BaseTestLoader):
    """Test loader for multimodal models"""
    
    def __init__(self, config: TestConfig):
        super().__init__(config)
        
    def load_model(self):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        self.device = self.get_device()
        MessageProtocol.status("loading", "Loading multimodal model...")
        
        # Placeholder model
        class DualEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.image_encoder = nn.Sequential(
                    nn.Conv2d(3, 64, 3), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                    nn.Linear(64, 512)
                )
                self.text_encoder = nn.Embedding(10000, 512)
            
            def forward(self, image=None, text=None):
                if image is not None and text is not None:
                    img_emb = self.image_encoder(image)
                    txt_emb = self.text_encoder(text).mean(dim=1)
                    similarity = torch.cosine_similarity(img_emb, txt_emb)
                    return similarity
                return None
        
        self.model = DualEncoder().to(self.device).eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        MessageProtocol.status("loaded", "Multimodal model loaded")
    
    def load_test_data(self) -> List[Dict[str, Any]]:
        test_path = Path(self.config.dataset_path) / "test"
        
        # Find image-text pairs
        test_data = []
        for img_file in test_path.glob("*.jpg") + list(test_path.glob("*.png")):
            txt_file = img_file.with_suffix('.txt')
            if txt_file.exists():
                with open(txt_file) as f:
                    text = f.read().strip()
                test_data.append({
                    "image_path": str(img_file),
                    "text": text
                })
        
        if self.config.max_samples:
            test_data = test_data[:self.config.max_samples]
        
        MessageProtocol.status("loaded", f"Loaded {len(test_data)} image-text pairs")
        return test_data
    
    def test_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            image = Image.open(sample["image_path"]).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Simple tokenization (placeholder)
            text_tokens = torch.randint(0, 10000, (1, 77)).to(self.device)
            
            with torch.no_grad():
                similarity = self.model(image_tensor, text_tokens).item()
            
            return {
                "image_path": sample["image_path"],
                "text": sample["text"],
                "similarity_score": similarity,
                "inference_time": time.time() - start_time
            }
        except Exception as e:
            return {"image_path": sample["image_path"], "error": str(e)}
    
    def compute_metrics(self, all_results: List[Dict]) -> Dict[str, Any]:
        valid = [r for r in all_results if "error" not in r]
        
        if not valid:
            return {"error": "No valid predictions"}
        
        similarities = [r["similarity_score"] for r in valid]
        
        return {
            "total_samples": len(valid),
            "mean_similarity": sum(similarities) / len(similarities),
            "average_inference_time": sum(r["inference_time"] for r in valid) / len(valid)
        }


TEST_REGISTRY.register_test_loader(Modality.MULTIMODAL, MultimodalTestLoader)
MessageProtocol.debug("Multimodal test plugin loaded", {"torch": TORCH_AVAILABLE})
