"""
Tabular Test Plugin for FrameTrain v2
=====================================
This plugin adds testing support for tabular/structured data:
- Neural Networks for Tabular Data
- Classification and Regression

MANIFEST:
{
    "name": "Tabular Test Plugin",
    "description": "Testing for Tabular/Structured data models",
    "modality": "tabular",
    "required": [
        "torch",
        "pandas",
        "numpy"
    ],
    "optional": [
        "scikit-learn"
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
    import numpy as np
    import pandas as pd
    TORCH_AVAILABLE = True
    PANDAS_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    PANDAS_AVAILABLE = False
    print(f"Warning: {e}")

try:
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from test_engine import (
    BaseTestLoader, Modality, TEST_REGISTRY,
    MessageProtocol, TestConfig
)


class TabularTestLoader(BaseTestLoader):
    """Test loader for tabular models"""
    
    def __init__(self, config: TestConfig):
        super().__init__(config)
        self.feature_names = []
        self.target_name = None
        self.num_features = 0
        self.num_classes = 0
        self.task_type = "classification"  # or "regression"
        self.scaler = None
        self.label_encoder = None
    
    def load_model(self):
        """Load tabular model"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")
        
        MessageProtocol.status("loading", f"Loading tabular model from {self.config.model_path}...")
        
        self.device = self.get_device()
        MessageProtocol.status("device", f"Using device: {self.device}")
        
        model_path = Path(self.config.model_path)
        
        # Load config
        config_file = model_path / "config.json"
        if config_file.exists():
            import json
            with open(config_file, 'r') as f:
                config = json.load(f)
                self.num_features = config.get("num_features", 10)
                self.num_classes = config.get("num_classes", 2)
                self.task_type = "classification" if self.num_classes > 1 else "regression"
        else:
            raise ValueError(f"No config.json found in {model_path}")
        
        # Create model architecture
        self.model = self._create_mlp(self.num_features, self.num_classes)
        
        # Load weights
        checkpoint_file = model_path / "model.pth"
        if checkpoint_file.exists():
            state_dict = torch.load(checkpoint_file, map_location='cpu')
            self.model.load_state_dict(state_dict)
        else:
            raise ValueError(f"No model.pth found in {model_path}")
        
        self.model.to(self.device)
        self.model.eval()
        
        MessageProtocol.status("loaded", f"Tabular model loaded ({self.task_type}, {self.num_features} features)")
    
    def _create_mlp(self, in_features, out_features):
        """Create MLP architecture matching training"""
        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                hidden_dim = 256
                
                self.layers = nn.Sequential(
                    nn.Linear(in_features, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim // 2, out_features)
                )
            
            def forward(self, x):
                return self.layers(x)
        
        return MLP()
    
    def load_test_data(self) -> List[Dict[str, Any]]:
        """Load tabular test data"""
        if not PANDAS_AVAILABLE:
            raise ImportError("Pandas is required. Install with: pip install pandas")
        
        MessageProtocol.status("loading", "Loading test data...")
        
        test_path = Path(self.config.dataset_path) / "test"
        if not test_path.exists():
            test_path = Path(self.config.dataset_path) / "val"
        if not test_path.exists():
            raise ValueError(f"Test data not found: {self.config.dataset_path}")
        
        # Find data file
        csv_files = list(test_path.glob("*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in {test_path}")
        
        df = pd.read_csv(csv_files[0])
        
        # Identify columns
        self._identify_columns(df)
        
        # Prepare features
        X = df[self.feature_names].values
        X = np.nan_to_num(X, nan=0.0)
        
        # Scale features if scaler available
        if SKLEARN_AVAILABLE:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        
        # Get targets
        y = df[self.target_name].values
        
        # Encode labels if classification
        if self.task_type == "classification":
            if SKLEARN_AVAILABLE and (y.dtype == object or len(np.unique(y)) < 100):
                self.label_encoder = LabelEncoder()
                y = self.label_encoder.fit_transform(y)
        
        # Create test samples
        test_data = []
        for i in range(len(X)):
            test_data.append({
                "index": i,
                "features": X[i],
                "target": y[i]
            })
        
        # Limit samples
        if self.config.max_samples and len(test_data) > self.config.max_samples:
            test_data = test_data[:self.config.max_samples]
        
        MessageProtocol.status("loaded", f"Loaded {len(test_data)} test samples")
        return test_data
    
    def _identify_columns(self, df: pd.DataFrame):
        """Identify feature and target columns"""
        target_candidates = ['target', 'label', 'class', 'y']
        
        for candidate in target_candidates:
            if candidate in df.columns:
                self.target_name = candidate
                break
        
        if not self.target_name:
            self.target_name = df.columns[-1]
        
        self.feature_names = [col for col in df.columns if col != self.target_name]
        self.num_features = len(self.feature_names)
    
    def test_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Test a single sample"""
        features = sample["features"]
        true_target = sample["target"]
        
        start_time = time.time()
        
        try:
            # Convert to tensor
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(features_tensor)
                
                if self.task_type == "classification":
                    probabilities = torch.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][predicted_class].item()
                    
                    is_correct = (predicted_class == true_target)
                    
                    # Loss
                    criterion = nn.CrossEntropyLoss()
                    target_tensor = torch.tensor([true_target], dtype=torch.long).to(self.device)
                    loss = criterion(outputs, target_tensor).item()
                    
                    result = {
                        "index": sample["index"],
                        "predicted_class": predicted_class,
                        "true_class": int(true_target),
                        "confidence": confidence,
                        "is_correct": is_correct,
                        "loss": loss,
                        "inference_time": time.time() - start_time
                    }
                else:
                    # Regression
                    predicted_value = outputs.squeeze().item()
                    error = abs(predicted_value - true_target)
                    
                    # Loss
                    criterion = nn.MSELoss()
                    target_tensor = torch.tensor([true_target], dtype=torch.float32).to(self.device)
                    loss = criterion(outputs.squeeze(), target_tensor).item()
                    
                    result = {
                        "index": sample["index"],
                        "predicted_value": predicted_value,
                        "true_value": float(true_target),
                        "error": error,
                        "loss": loss,
                        "inference_time": time.time() - start_time
                    }
                
                return result
                
        except Exception as e:
            MessageProtocol.warning(f"Failed to process sample {sample['index']}: {e}")
            return {
                "index": sample["index"],
                "error": str(e),
                "inference_time": time.time() - start_time
            }
    
    def compute_metrics(self, all_results: List[Dict]) -> Dict[str, Any]:
        """Compute tabular metrics"""
        valid_results = [r for r in all_results if "error" not in r]
        
        if not valid_results:
            return {"error": "No valid predictions"}
        
        total = len(valid_results)
        
        # Average inference time
        inference_times = [r["inference_time"] for r in valid_results]
        avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
        
        # Average loss
        losses = [r["loss"] for r in valid_results]
        avg_loss = sum(losses) / len(losses) if losses else None
        
        metrics = {
            "total_samples": total,
            "average_loss": avg_loss,
            "average_inference_time": avg_inference_time,
            "task_type": self.task_type
        }
        
        if self.task_type == "classification":
            # Classification metrics
            correct = sum(1 for r in valid_results if r.get("is_correct"))
            accuracy = (correct / total) * 100
            
            # Per-class metrics
            per_class = {}
            for class_idx in range(self.num_classes):
                class_results = [r for r in valid_results if r["true_class"] == class_idx]
                if class_results:
                    class_correct = sum(1 for r in class_results if r["is_correct"])
                    per_class[f"class_{class_idx}"] = {
                        "accuracy": (class_correct / len(class_results)) * 100,
                        "total": len(class_results),
                        "correct": class_correct
                    }
            
            metrics.update({
                "accuracy": accuracy,
                "correct_predictions": correct,
                "incorrect_predictions": total - correct,
                "per_class_metrics": per_class
            })
        else:
            # Regression metrics
            predictions = [r["predicted_value"] for r in valid_results]
            targets = [r["true_value"] for r in valid_results]
            
            mae = sum(r["error"] for r in valid_results) / len(valid_results)
            
            if SKLEARN_AVAILABLE:
                mse = mean_squared_error(targets, predictions)
                rmse = np.sqrt(mse)
                r2 = r2_score(targets, predictions)
            else:
                mse = sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(predictions)
                rmse = np.sqrt(mse)
                r2 = None
            
            metrics.update({
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "r2_score": r2
            })
        
        return metrics


# Register tabular test loader
TEST_REGISTRY.register_test_loader(Modality.TABULAR, TabularTestLoader)

MessageProtocol.debug("Tabular test plugin loaded", {
    "torch_available": TORCH_AVAILABLE,
    "pandas_available": PANDAS_AVAILABLE,
    "sklearn_available": SKLEARN_AVAILABLE
})
