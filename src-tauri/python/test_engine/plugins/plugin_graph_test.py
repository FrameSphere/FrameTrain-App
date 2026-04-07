"""
Graph Test Plugin for FrameTrain v2
===================================
Testing support for graph neural networks:
- Node Classification
- Graph Classification
- Link Prediction

MANIFEST:
{
    "name": "Graph Test Plugin",
    "description": "Testing for Graph Neural Networks",
    "modality": "graph",
    "required": ["torch", "numpy"],
    "optional": ["torch_geometric", "networkx"],
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
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from test_engine import BaseTestLoader, Modality, TEST_REGISTRY, MessageProtocol, TestConfig


class GraphTestLoader(BaseTestLoader):
    """Test loader for graph models"""
    
    def __init__(self, config: TestConfig):
        super().__init__(config)
        self.num_features = 16
        self.num_classes = 7
        
    def load_model(self):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        self.device = self.get_device()
        MessageProtocol.status("loading", "Loading graph model...")
        
        # Simple GNN
        class SimpleGNN(nn.Module):
            def __init__(self, in_features, hidden_dim, out_features):
                super().__init__()
                self.fc1 = nn.Linear(in_features, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, out_features)
            
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        
        self.model = SimpleGNN(self.num_features, 64, self.num_classes).to(self.device).eval()
        
        MessageProtocol.status("loaded", "Graph model loaded")
    
    def load_test_data(self) -> List[Dict[str, Any]]:
        # Placeholder: Load graph data
        MessageProtocol.status("loading", "Loading graph test data...")
        
        # Generate dummy node features
        test_data = []
        for i in range(100):
            test_data.append({
                "node_id": i,
                "features": np.random.randn(self.num_features),
                "label": np.random.randint(0, self.num_classes)
            })
        
        if self.config.max_samples:
            test_data = test_data[:self.config.max_samples]
        
        MessageProtocol.status("loaded", f"Loaded {len(test_data)} graph nodes")
        return test_data
    
    def test_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            features = torch.tensor(sample["features"], dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(features)
                predicted_class = torch.argmax(output, dim=1).item()
            
            is_correct = (predicted_class == sample["label"])
            
            return {
                "node_id": sample["node_id"],
                "predicted_class": predicted_class,
                "true_class": sample["label"],
                "is_correct": is_correct,
                "inference_time": time.time() - start_time
            }
        except Exception as e:
            return {"node_id": sample["node_id"], "error": str(e)}
    
    def compute_metrics(self, all_results: List[Dict]) -> Dict[str, Any]:
        valid = [r for r in all_results if "error" not in r]
        
        if not valid:
            return {"error": "No valid predictions"}
        
        correct = sum(1 for r in valid if r["is_correct"])
        accuracy = (correct / len(valid)) * 100
        
        return {
            "total_samples": len(valid),
            "accuracy": accuracy,
            "correct_predictions": correct,
            "incorrect_predictions": len(valid) - correct,
            "average_inference_time": sum(r["inference_time"] for r in valid) / len(valid)
        }


TEST_REGISTRY.register_test_loader(Modality.GRAPH, GraphTestLoader)
MessageProtocol.debug("Graph test plugin loaded", {"torch": TORCH_AVAILABLE})
