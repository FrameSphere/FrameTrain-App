"""
Time Series Test Plugin for FrameTrain v2
==========================================
Testing support for time series models:
- Forecasting
- Anomaly Detection
- Classification

MANIFEST:
{
    "name": "TimeSeries Test Plugin",
    "description": "Testing for Time Series models",
    "modality": "timeseries",
    "required": ["torch", "numpy"],
    "optional": ["pandas"],
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


class TimeSeriesTestLoader(BaseTestLoader):
    """Test loader for time series models"""
    
    def __init__(self, config: TestConfig):
        super().__init__(config)
        self.seq_length = 128
        self.output_size = 1
        
    def load_model(self):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        self.device = self.get_device()
        MessageProtocol.status("loading", f"Loading time series model...")
        
        # Simple LSTM model
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
                self.fc = nn.Linear(hidden_size, output_size)
            
            def forward(self, x):
                out, _ = self.lstm(x)
                out = self.fc(out[:, -1, :])
                return out
        
        model_path = Path(self.config.model_path)
        checkpoint = model_path / "model.pth"
        
        if checkpoint.exists():
            state = torch.load(checkpoint, map_location='cpu')
            self.model = LSTMModel(1, 64, self.output_size)
            self.model.load_state_dict(state)
        else:
            self.model = LSTMModel(1, 64, self.output_size)
        
        self.model.to(self.device).eval()
        MessageProtocol.status("loaded", "Time series model loaded")
    
    def load_test_data(self) -> List[Dict[str, Any]]:
        test_path = Path(self.config.dataset_path) / "test"
        
        # Load numpy file or CSV
        data_file = list(test_path.glob("*.npy")) + list(test_path.glob("*.csv"))
        if not data_file:
            raise ValueError("No test data found")
        
        if data_file[0].suffix == '.npy':
            data = np.load(data_file[0])
        else:
            import pandas as pd
            df = pd.read_csv(data_file[0])
            data = df.values
        
        # Create sequences
        test_data = []
        for i in range(len(data) - self.seq_length):
            test_data.append({
                "index": i,
                "sequence": data[i:i+self.seq_length],
                "target": data[i+self.seq_length]
            })
        
        if self.config.max_samples:
            test_data = test_data[:self.config.max_samples]
        
        MessageProtocol.status("loaded", f"Loaded {len(test_data)} sequences")
        return test_data
    
    def test_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            sequence = torch.tensor(sample["sequence"], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)
            
            with torch.no_grad():
                prediction = self.model(sequence).squeeze().item()
            
            target = float(sample["target"])
            error = abs(prediction - target)
            
            return {
                "index": sample["index"],
                "predicted": prediction,
                "target": target,
                "error": error,
                "inference_time": time.time() - start_time
            }
        except Exception as e:
            return {"index": sample["index"], "error": str(e)}
    
    def compute_metrics(self, all_results: List[Dict]) -> Dict[str, Any]:
        valid = [r for r in all_results if "error" not in r or "predicted" in r]
        
        if not valid:
            return {"error": "No valid predictions"}
        
        errors = [r["error"] for r in valid]
        mae = sum(errors) / len(errors)
        mse = sum(e**2 for e in errors) / len(errors)
        rmse = np.sqrt(mse)
        
        return {
            "total_samples": len(valid),
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "average_inference_time": sum(r["inference_time"] for r in valid) / len(valid)
        }


TEST_REGISTRY.register_test_loader(Modality.TIMESERIES, TimeSeriesTestLoader)
MessageProtocol.debug("TimeSeries test plugin loaded", {"torch": TORCH_AVAILABLE})
