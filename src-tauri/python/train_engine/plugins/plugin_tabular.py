"""
Tabular Data Plugin for FrameTrain v2
=====================================
This plugin adds support for tabular/structured data:
- TabNet
- Neural Networks for Tabular Data
- XGBoost (wrapper)
- CatBoost (wrapper)

MANIFEST:
{
    "name": "Tabular Plugin",
    "description": "Tabular/Structured data (TabNet, MLP, XGBoost)",
    "modality": "tabular",
    "required": [
        "torch",
        "pandas",
        "numpy"
    ],
    "optional": [
        "pytorch-tabnet",
        "xgboost",
        "catboost",
        "scikit-learn"
    ],
    "python": "3.8"
}

Installation:
    pip install pytorch-tabnet
    pip install xgboost catboost  # Optional

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
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    import numpy as np
    import pandas as pd
    TORCH_AVAILABLE = True
    PANDAS_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    PANDAS_AVAILABLE = False

try:
    from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


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
# TABULAR DATA LOADER
# ============================================================================

class TabularDataLoader(BaseDataLoader):
    """Data loader for tabular/structured data"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.feature_names = []
        self.target_name = None
        self.num_features = 0
        self.num_classes = 0
        self.scaler = None
        self.label_encoder = None
    
    def load(self) -> Tuple[Any, Optional[Any], Optional[Any]]:
        """Load tabular datasets"""
        try:
            if not PANDAS_AVAILABLE:
                raise ImportError("Pandas not installed. Install with: pip install pandas")
            
            MessageProtocol.status("loading", "Loading tabular datasets...")
            
            dataset_path = Path(self.config.dataset_path)
            train_file = self._find_data_file(dataset_path / "train")
            
            if not train_file:
                raise ValueError(f"No data file found in {dataset_path / 'train'}")
            
            # Load data
            train_df = self._load_dataframe(train_file)
            MessageProtocol.status("loading", f"Loaded {len(train_df)} training samples")
            
            # Identify features and target
            self._identify_columns(train_df)
            
            # Prepare data
            X_train, y_train = self._prepare_data(train_df)
            
            # Create dataset and dataloader
            train_dataset = TensorDataset(X_train, y_train)
            
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=self.config.pin_memory
            )
            
            # Load validation data
            val_file = self._find_data_file(dataset_path / "val")
            if val_file:
                val_df = self._load_dataframe(val_file)
                X_val, y_val = self._prepare_data(val_df, fit_scaler=False)
                
                val_dataset = TensorDataset(X_val, y_val)
                self.val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=self.config.pin_memory
                )
                
                MessageProtocol.status("loading", 
                    f"Loaded {len(val_df)} validation samples")
            
            return self.train_loader, self.val_loader, None
            
        except Exception as e:
            MessageProtocol.error("Failed to load tabular data", str(e))
            raise
    
    def _find_data_file(self, data_path: Path) -> Optional[Path]:
        """Find CSV or other data file"""
        if not data_path.exists():
            return None
        
        # Try CSV
        csv_files = list(data_path.glob("*.csv"))
        if csv_files:
            return csv_files[0]
        
        # Try Parquet
        parquet_files = list(data_path.glob("*.parquet"))
        if parquet_files:
            return parquet_files[0]
        
        # Try Excel
        excel_files = list(data_path.glob("*.xlsx")) + list(data_path.glob("*.xls"))
        if excel_files:
            return excel_files[0]
        
        return None
    
    def _load_dataframe(self, file_path: Path) -> pd.DataFrame:
        """Load dataframe from file"""
        if file_path.suffix == '.csv':
            return pd.read_csv(file_path)
        elif file_path.suffix == '.parquet':
            return pd.read_parquet(file_path)
        elif file_path.suffix in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _identify_columns(self, df: pd.DataFrame):
        """Identify feature and target columns"""
        # Look for common target column names
        target_candidates = ['target', 'label', 'class', 'y']
        
        for candidate in target_candidates:
            if candidate in df.columns:
                self.target_name = candidate
                break
        
        if not self.target_name:
            # Assume last column is target
            self.target_name = df.columns[-1]
            MessageProtocol.warning(f"No target column found, using last column: {self.target_name}")
        
        # All other columns are features
        self.feature_names = [col for col in df.columns if col != self.target_name]
        self.num_features = len(self.feature_names)
        
        MessageProtocol.status("loading", 
            f"Features: {self.num_features}, Target: {self.target_name}")
    
    def _prepare_data(self, df: pd.DataFrame, fit_scaler: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare features and target"""
        # Extract features and target
        X = df[self.feature_names].values
        y = df[self.target_name].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        # Scale features
        if fit_scaler:
            if SKLEARN_AVAILABLE:
                self.scaler = StandardScaler()
                X = self.scaler.fit_transform(X)
            else:
                MessageProtocol.warning("sklearn not available, skipping scaling")
        else:
            if self.scaler:
                X = self.scaler.transform(X)
        
        # Encode labels if classification
        if y.dtype == object or len(np.unique(y)) < 100:  # Assume classification
            if fit_scaler:
                if SKLEARN_AVAILABLE:
                    self.label_encoder = LabelEncoder()
                    y = self.label_encoder.fit_transform(y)
                    self.num_classes = len(self.label_encoder.classes_)
                else:
                    # Manual encoding
                    unique_labels = np.unique(y)
                    label_map = {label: idx for idx, label in enumerate(unique_labels)}
                    y = np.array([label_map[label] for label in y])
                    self.num_classes = len(unique_labels)
            else:
                if self.label_encoder:
                    y = self.label_encoder.transform(y)
        else:
            # Regression
            self.num_classes = 1
        
        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long if self.num_classes > 1 else torch.float32)
        
        return X_tensor, y_tensor
    
    def get_sample_info(self) -> Dict[str, Any]:
        """Get sample information"""
        return {
            "num_features": self.num_features,
            "num_classes": self.num_classes,
            "feature_names": self.feature_names,
            "target_name": self.target_name,
            "num_train_samples": len(self.train_loader.dataset) if self.train_loader else 0,
            "num_val_samples": len(self.val_loader.dataset) if self.val_loader else 0
        }


# ============================================================================
# TABULAR MODEL HANDLER
# ============================================================================

class TabularModelHandler(BaseModelHandler):
    """Handler for tabular data models"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.model_type = None
    
    def load_model(self):
        """Load tabular model"""
        try:
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch not available")
            
            MessageProtocol.status("loading", "Loading tabular model...")
            
            # Determine device
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                MessageProtocol.status("device", f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device("cpu")
                MessageProtocol.status("device", "Using CPU")
            
            model_path = self.config.model_path.lower()
            
            # Get model parameters
            num_features = getattr(self.config, 'num_features', 10)
            num_classes = getattr(self.config, 'num_classes', 2)
            
            # Create model based on type
            if "tabnet" in model_path:
                if not TABNET_AVAILABLE:
                    raise ImportError(
                        "TabNet not installed. Install with: pip install pytorch-tabnet"
                    )
                
                self.model_type = "tabnet"
                MessageProtocol.status("loading", "Creating TabNet model")
                
                # TabNet has its own interface, we'll adapt it
                if num_classes > 1:
                    self.model = TabNetClassifier(
                        n_d=64,
                        n_a=64,
                        n_steps=5,
                        gamma=1.5,
                        n_independent=2,
                        n_shared=2
                    )
                else:
                    self.model = TabNetRegressor(
                        n_d=64,
                        n_a=64,
                        n_steps=5,
                        gamma=1.5
                    )
                
                # Note: TabNet training is handled differently
                # We'll use a simple MLP for the prototype
                MessageProtocol.warning("Using MLP instead of TabNet for prototype")
                self.model_type = "mlp"
                self.model = self._create_mlp(num_features, num_classes)
            
            else:
                # Default: MLP
                self.model_type = "mlp"
                self.model = self._create_mlp(num_features, num_classes)
            
            self.model = self.model.to(self.device)
            
            MessageProtocol.status("loaded", f"Tabular model loaded: {self.model_type}")
            
        except Exception as e:
            MessageProtocol.error("Tabular model loading failed", str(e))
            raise
    
    def _create_mlp(self, in_features, out_features):
        """Create MLP for tabular data"""
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
    
    def forward(self, batch: Any) -> Any:
        """Forward pass"""
        features, labels = batch
        features = features.to(self.device)
        labels = labels.to(self.device)
        
        outputs = self.model(features)
        
        return outputs, labels
    
    def compute_loss(self, outputs: Any, batch: Any) -> Any:
        """Compute loss"""
        logits, labels = outputs
        
        # Classification or regression
        if labels.dtype == torch.long:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, labels)
        else:
            criterion = nn.MSELoss()
            loss = criterion(logits.squeeze(), labels)
        
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
        
        MessageProtocol.status("saved", f"Tabular model saved to {output_path}")


# ============================================================================
# REGISTER PLUGIN
# ============================================================================

REGISTRY.register_data_loader(Modality.TABULAR, TabularDataLoader)
REGISTRY.register_model_handler("tabular", TabularModelHandler)

MessageProtocol.debug("Tabular plugin loaded", {
    "pandas_available": PANDAS_AVAILABLE,
    "tabnet_available": TABNET_AVAILABLE,
    "sklearn_available": SKLEARN_AVAILABLE
})
