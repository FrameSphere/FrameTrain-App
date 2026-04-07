"""
Time Series Plugin for FrameTrain v2
====================================
This plugin adds support for time series tasks:
- Forecasting (Time Series Transformer, Prophet, LSTM)
- Anomaly Detection
- Classification

MANIFEST:
{
    "name": "Time Series Plugin",
    "description": "Time Series forecasting and analysis (LSTM, Transformer, Prophet)",
    "modality": "time_series",
    "required": [
        "torch",
        "pandas",
        "numpy"
    ],
    "optional": [
        "prophet",
        "transformers"
    ],
    "python": "3.8"
}

Installation:
    pip install prophet  # For Prophet
    pip install transformers  # For Time Series Transformer

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
    import numpy as np
    import pandas as pd
    TORCH_AVAILABLE = True
    PANDAS_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    PANDAS_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


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
# TIME SERIES DATASET
# ============================================================================

class TimeSeriesDataset(Dataset):
    """Dataset for time series with sliding windows"""
    
    def __init__(self, data: np.ndarray, window_size: int = 24, 
                 horizon: int = 1):
        """
        Args:
            data: Time series data [time_steps, features]
            window_size: Input sequence length
            horizon: Prediction horizon
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.window_size = window_size
        self.horizon = horizon
    
    def __len__(self):
        return len(self.data) - self.window_size - self.horizon + 1
    
    def __getitem__(self, idx):
        # Input: [idx:idx+window_size]
        x = self.data[idx:idx + self.window_size]
        
        # Target: [idx+window_size:idx+window_size+horizon]
        y = self.data[idx + self.window_size:idx + self.window_size + self.horizon]
        
        return x, y


# ============================================================================
# TIME SERIES DATA LOADER
# ============================================================================

class TimeSeriesDataLoader(BaseDataLoader):
    """Data loader for time series tasks"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.num_features = 0
        self.window_size = getattr(config, 'window_size', 24)
        self.horizon = getattr(config, 'horizon', 1)
    
    def load(self) -> Tuple[Any, Optional[Any], Optional[Any]]:
        """Load time series datasets"""
        try:
            if not PANDAS_AVAILABLE:
                raise ImportError("Pandas not installed")
            
            MessageProtocol.status("loading", "Loading time series datasets...")
            
            dataset_path = Path(self.config.dataset_path)
            train_file = self._find_data_file(dataset_path / "train")
            
            if not train_file:
                raise ValueError(f"No data file found in {dataset_path / 'train'}")
            
            # Load data
            train_df = pd.read_csv(train_file)
            MessageProtocol.status("loading", f"Loaded time series with {len(train_df)} timesteps")
            
            # Prepare data
            train_data = self._prepare_data(train_df)
            
            # Create dataset
            train_dataset = TimeSeriesDataset(
                train_data,
                window_size=self.window_size,
                horizon=self.horizon
            )
            
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=self.config.pin_memory
            )
            
            MessageProtocol.status("loading", 
                f"Created {len(train_dataset)} training windows")
            
            # Load validation data
            val_file = self._find_data_file(dataset_path / "val")
            if val_file:
                val_df = pd.read_csv(val_file)
                val_data = self._prepare_data(val_df)
                
                val_dataset = TimeSeriesDataset(
                    val_data,
                    window_size=self.window_size,
                    horizon=self.horizon
                )
                
                self.val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=self.config.pin_memory
                )
                
                MessageProtocol.status("loading", 
                    f"Created {len(val_dataset)} validation windows")
            
            return self.train_loader, self.val_loader, None
            
        except Exception as e:
            MessageProtocol.error("Failed to load time series data", str(e))
            raise
    
    def _find_data_file(self, data_path: Path) -> Optional[Path]:
        """Find CSV file"""
        if not data_path.exists():
            return None
        
        csv_files = list(data_path.glob("*.csv"))
        return csv_files[0] if csv_files else None
    
    def _prepare_data(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare time series data"""
        # Remove timestamp column if present
        timestamp_cols = ['timestamp', 'date', 'datetime', 'time']
        for col in timestamp_cols:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # Convert to numpy
        data = df.values.astype(np.float32)
        
        # Handle missing values
        data = np.nan_to_num(data, nan=0.0)
        
        # Normalize (simple min-max)
        data_min = data.min(axis=0, keepdims=True)
        data_max = data.max(axis=0, keepdims=True)
        data = (data - data_min) / (data_max - data_min + 1e-8)
        
        self.num_features = data.shape[1]
        
        return data
    
    def get_sample_info(self) -> Dict[str, Any]:
        """Get sample information"""
        return {
            "num_features": self.num_features,
            "window_size": self.window_size,
            "horizon": self.horizon,
            "num_train_windows": len(self.train_loader.dataset) if self.train_loader else 0,
            "num_val_windows": len(self.val_loader.dataset) if self.val_loader else 0
        }


# ============================================================================
# TIME SERIES MODEL HANDLER
# ============================================================================

class TimeSeriesModelHandler(BaseModelHandler):
    """Handler for time series models"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.model_type = None
    
    def load_model(self):
        """Load time series model"""
        try:
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch not available")
            
            MessageProtocol.status("loading", "Loading time series model...")
            
            # Determine device
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                MessageProtocol.status("device", f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device("cpu")
                MessageProtocol.status("device", "Using CPU")
            
            model_path = self.config.model_path.lower()
            
            # Get model parameters
            num_features = getattr(self.config, 'num_features', 1)
            window_size = getattr(self.config, 'window_size', 24)
            horizon = getattr(self.config, 'horizon', 1)
            
            # Create model based on type
            if "prophet" in model_path:
                if not PROPHET_AVAILABLE:
                    raise ImportError("Prophet not installed. Install with: pip install prophet")
                
                self.model_type = "prophet"
                MessageProtocol.status("loading", "Creating Prophet model")
                
                # Prophet has special training
                # For now, fallback to LSTM
                MessageProtocol.warning("Using LSTM instead of Prophet for prototype")
                self.model_type = "lstm"
                self.model = self._create_lstm(num_features, window_size, horizon)
            
            elif "lstm" in model_path:
                self.model_type = "lstm"
                self.model = self._create_lstm(num_features, window_size, horizon)
            
            elif "transformer" in model_path:
                self.model_type = "transformer"
                self.model = self._create_transformer(num_features, window_size, horizon)
            
            else:
                # Default: LSTM
                self.model_type = "lstm"
                self.model = self._create_lstm(num_features, window_size, horizon)
            
            self.model = self.model.to(self.device)
            
            MessageProtocol.status("loaded", f"Time series model loaded: {self.model_type}")
            
        except Exception as e:
            MessageProtocol.error("Time series model loading failed", str(e))
            raise
    
    def _create_lstm(self, num_features, window_size, horizon):
        """Create LSTM model"""
        class LSTM(nn.Module):
            def __init__(self):
                super().__init__()
                hidden_dim = 128
                num_layers = 2
                
                self.lstm = nn.LSTM(
                    num_features,
                    hidden_dim,
                    num_layers,
                    batch_first=True,
                    dropout=0.2
                )
                
                self.fc = nn.Linear(hidden_dim, num_features * horizon)
                self.horizon = horizon
                self.num_features = num_features
            
            def forward(self, x):
                # x: [batch, window_size, num_features]
                lstm_out, _ = self.lstm(x)
                
                # Take last output
                last_out = lstm_out[:, -1, :]
                
                # Predict horizon
                out = self.fc(last_out)
                out = out.view(-1, self.horizon, self.num_features)
                
                return out
        
        return LSTM()
    
    def _create_transformer(self, num_features, window_size, horizon):
        """Create Transformer model"""
        class TransformerModel(nn.Module):
            def __init__(self):
                super().__init__()
                d_model = 128
                nhead = 4
                num_layers = 2
                
                self.input_proj = nn.Linear(num_features, d_model)
                
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=512,
                    dropout=0.1,
                    batch_first=True
                )
                
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
                
                self.fc = nn.Linear(d_model, num_features * horizon)
                self.horizon = horizon
                self.num_features = num_features
            
            def forward(self, x):
                # x: [batch, window_size, num_features]
                x = self.input_proj(x)
                
                x = self.transformer(x)
                
                # Take last output
                last_out = x[:, -1, :]
                
                out = self.fc(last_out)
                out = out.view(-1, self.horizon, self.num_features)
                
                return out
        
        return TransformerModel()
    
    def forward(self, batch: Any) -> Any:
        """Forward pass"""
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        
        pred = self.model(x)
        
        return pred, y
    
    def compute_loss(self, outputs: Any, batch: Any) -> Any:
        """Compute loss"""
        pred, target = outputs
        
        criterion = nn.MSELoss()
        loss = criterion(pred, target)
        
        return loss
    
    def save_model(self, path: str):
        """Save model"""
        output_path = Path(path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.model.state_dict(), output_path / "model.pth")
        
        config_dict = {
            "model_type": self.model_type
        }
        
        with open(output_path / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        MessageProtocol.status("saved", f"Time series model saved to {output_path}")


# ============================================================================
# REGISTER PLUGIN
# ============================================================================

REGISTRY.register_data_loader(Modality.TIME_SERIES, TimeSeriesDataLoader)
REGISTRY.register_model_handler("timeseries", TimeSeriesModelHandler)

MessageProtocol.debug("Time series plugin loaded", {
    "pandas_available": PANDAS_AVAILABLE,
    "prophet_available": PROPHET_AVAILABLE
})
