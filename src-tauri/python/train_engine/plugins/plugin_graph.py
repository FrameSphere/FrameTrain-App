"""
Graph Neural Network Plugin for FrameTrain v2
=============================================
This plugin adds support for graph-based learning:
- Node Classification (GCN, GAT, GraphSAGE)
- Edge Prediction
- Graph Classification
- Link Prediction

MANIFEST:
{
    "name": "Graph Plugin",
    "description": "Graph Neural Networks (GCN, GAT, GraphSAGE)",
    "modality": "graph",
    "required": [
        "torch",
        "torch-geometric"
    ],
    "optional": [
        "torch-scatter",
        "torch-sparse",
        "torch-cluster",
        "networkx"
    ],
    "python": "3.8"
}

Installation:
    pip install torch-geometric
    pip install torch-scatter torch-sparse torch-cluster

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
    from torch.utils.data import DataLoader
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import torch_geometric as pyg
    from torch_geometric.data import Data, Dataset
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv
    from torch_geometric.loader import DataLoader as PyGDataLoader
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


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
# GRAPH DATA LOADER
# ============================================================================

class GraphDataLoader(BaseDataLoader):
    """Data loader for graph neural networks"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.num_node_features = 0
        self.num_classes = 0
    
    def load(self) -> Tuple[Any, Optional[Any], Optional[Any]]:
        """Load graph datasets"""
        try:
            if not TORCH_GEOMETRIC_AVAILABLE:
                raise ImportError(
                    "PyTorch Geometric not installed. Install with:\n"
                    "pip install torch-geometric\n"
                    "pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html"
                )
            
            MessageProtocol.status("loading", "Loading graph datasets...")
            
            dataset_path = Path(self.config.dataset_path)
            train_path = dataset_path / "train"
            
            if not train_path.exists():
                raise ValueError(f"Training path not found: {train_path}")
            
            # Load graphs
            graphs = self._load_graphs(train_path)
            
            if not graphs:
                raise ValueError("No graphs found")
            
            MessageProtocol.status("loading", f"Loaded {len(graphs)} graphs")
            
            # Get graph info
            if len(graphs) > 0:
                self.num_node_features = graphs[0].x.shape[1] if graphs[0].x is not None else 0
                self.num_classes = len(set([g.y.item() if g.y.dim() == 0 else g.y[0].item() for g in graphs if g.y is not None]))
            
            MessageProtocol.status("loading", 
                f"Graph features: {self.num_node_features}, Classes: {self.num_classes}")
            
            # Create DataLoader
            self.train_loader = PyGDataLoader(
                graphs,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=0  # PyG can have issues with multiprocessing
            )
            
            # Load validation graphs
            val_path = dataset_path / "val"
            if val_path.exists():
                val_graphs = self._load_graphs(val_path)
                
                if val_graphs:
                    self.val_loader = PyGDataLoader(
                        val_graphs,
                        batch_size=self.config.batch_size,
                        shuffle=False,
                        num_workers=0
                    )
                    
                    MessageProtocol.status("loading", 
                        f"Loaded {len(val_graphs)} validation graphs")
            
            return self.train_loader, self.val_loader, None
            
        except Exception as e:
            MessageProtocol.error("Failed to load graph data", str(e))
            raise
    
    def _load_graphs(self, data_path: Path) -> List[Data]:
        """Load graphs from directory"""
        graphs = []
        
        # Method 1: PyG format (.pt files)
        for pt_file in data_path.glob("*.pt"):
            try:
                graph = torch.load(pt_file)
                if isinstance(graph, Data):
                    graphs.append(graph)
            except Exception as e:
                MessageProtocol.warning(f"Failed to load {pt_file}: {e}")
        
        if graphs:
            return graphs
        
        # Method 2: NetworkX format (.graphml, .gml, .edgelist)
        if NETWORKX_AVAILABLE:
            for graph_file in list(data_path.glob("*.graphml")) + \
                             list(data_path.glob("*.gml")) + \
                             list(data_path.glob("*.edgelist")):
                try:
                    if graph_file.suffix == ".graphml":
                        G = nx.read_graphml(graph_file)
                    elif graph_file.suffix == ".gml":
                        G = nx.read_gml(graph_file)
                    elif graph_file.suffix == ".edgelist":
                        G = nx.read_edgelist(graph_file)
                    else:
                        continue
                    
                    # Convert to PyG Data
                    graph = self._networkx_to_pyg(G)
                    if graph:
                        graphs.append(graph)
                
                except Exception as e:
                    MessageProtocol.warning(f"Failed to load {graph_file}: {e}")
        
        # Method 3: JSON format
        for json_file in data_path.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    graph_dict = json.load(f)
                
                graph = self._dict_to_pyg(graph_dict)
                if graph:
                    graphs.append(graph)
            
            except Exception as e:
                MessageProtocol.warning(f"Failed to load {json_file}: {e}")
        
        return graphs
    
    def _networkx_to_pyg(self, G: 'nx.Graph') -> Optional[Data]:
        """Convert NetworkX graph to PyG Data"""
        try:
            # Get edges
            edge_index = torch.tensor(list(G.edges())).t().contiguous()
            
            # Get node features (if available)
            if G.nodes() and 'features' in G.nodes[0]:
                x = torch.tensor([G.nodes[i]['features'] for i in G.nodes()])
            else:
                # Use node degree as feature
                x = torch.tensor([[G.degree(i)] for i in G.nodes()], dtype=torch.float)
            
            # Get labels (if available)
            y = None
            if 'label' in G.graph:
                y = torch.tensor([G.graph['label']])
            
            return Data(x=x, edge_index=edge_index, y=y)
        
        except Exception as e:
            MessageProtocol.warning(f"Failed to convert NetworkX graph: {e}")
            return None
    
    def _dict_to_pyg(self, graph_dict: Dict) -> Optional[Data]:
        """Convert dictionary to PyG Data"""
        try:
            # Expected format:
            # {
            #   "nodes": [[features], ...],
            #   "edges": [[src, dst], ...],
            #   "label": class_id
            # }
            
            x = torch.tensor(graph_dict['nodes'], dtype=torch.float)
            edge_index = torch.tensor(graph_dict['edges']).t().contiguous()
            y = torch.tensor([graph_dict.get('label', 0)])
            
            return Data(x=x, edge_index=edge_index, y=y)
        
        except Exception as e:
            MessageProtocol.warning(f"Failed to convert dict to graph: {e}")
            return None
    
    def get_sample_info(self) -> Dict[str, Any]:
        """Get sample information"""
        return {
            "num_node_features": self.num_node_features,
            "num_classes": self.num_classes,
            "num_train_graphs": len(self.train_loader.dataset) if self.train_loader else 0,
            "num_val_graphs": len(self.val_loader.dataset) if self.val_loader else 0
        }


# ============================================================================
# GRAPH MODEL HANDLER
# ============================================================================

class GNNModelHandler(BaseModelHandler):
    """Handler for Graph Neural Network models"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.model_type = None
    
    def load_model(self):
        """Load GNN model"""
        try:
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch not available")
            
            if not TORCH_GEOMETRIC_AVAILABLE:
                raise ImportError("PyTorch Geometric not installed")
            
            MessageProtocol.status("loading", "Loading GNN model...")
            
            # Determine device
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                MessageProtocol.status("device", f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device("cpu")
                MessageProtocol.status("device", "Using CPU")
            
            model_path = self.config.model_path.lower()
            
            # Get model parameters
            num_features = getattr(self.config, 'num_node_features', 16)
            num_classes = getattr(self.config, 'num_classes', 2)
            hidden_channels = getattr(self.config, 'hidden_channels', 64)
            
            # Create model based on type
            if "gcn" in model_path:
                self.model_type = "gcn"
                self.model = self._create_gcn(num_features, hidden_channels, num_classes)
            
            elif "gat" in model_path:
                self.model_type = "gat"
                self.model = self._create_gat(num_features, hidden_channels, num_classes)
            
            elif "sage" in model_path or "graphsage" in model_path:
                self.model_type = "graphsage"
                self.model = self._create_sage(num_features, hidden_channels, num_classes)
            
            else:
                # Default to GCN
                self.model_type = "gcn"
                self.model = self._create_gcn(num_features, hidden_channels, num_classes)
                MessageProtocol.warning(f"Unknown model type, using GCN")
            
            self.model = self.model.to(self.device)
            
            MessageProtocol.status("loaded", f"GNN model loaded: {self.model_type}")
            
        except Exception as e:
            MessageProtocol.error("GNN model loading failed", str(e))
            raise
    
    def _create_gcn(self, in_channels, hidden_channels, out_channels):
        """Create GCN model"""
        class GCN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = GCNConv(in_channels, hidden_channels)
                self.conv2 = GCNConv(hidden_channels, hidden_channels)
                self.conv3 = GCNConv(hidden_channels, out_channels)
            
            def forward(self, x, edge_index, batch):
                x = self.conv1(x, edge_index).relu()
                x = self.conv2(x, edge_index).relu()
                x = self.conv3(x, edge_index)
                
                # Global pooling
                x = pyg.nn.global_mean_pool(x, batch)
                return x
        
        return GCN()
    
    def _create_gat(self, in_channels, hidden_channels, out_channels):
        """Create GAT model"""
        class GAT(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = GATConv(in_channels, hidden_channels, heads=4)
                self.conv2 = GATConv(hidden_channels * 4, hidden_channels, heads=4)
                self.conv3 = GATConv(hidden_channels * 4, out_channels, heads=1)
            
            def forward(self, x, edge_index, batch):
                x = self.conv1(x, edge_index).relu()
                x = self.conv2(x, edge_index).relu()
                x = self.conv3(x, edge_index)
                
                x = pyg.nn.global_mean_pool(x, batch)
                return x
        
        return GAT()
    
    def _create_sage(self, in_channels, hidden_channels, out_channels):
        """Create GraphSAGE model"""
        class SAGE(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = SAGEConv(in_channels, hidden_channels)
                self.conv2 = SAGEConv(hidden_channels, hidden_channels)
                self.conv3 = SAGEConv(hidden_channels, out_channels)
            
            def forward(self, x, edge_index, batch):
                x = self.conv1(x, edge_index).relu()
                x = self.conv2(x, edge_index).relu()
                x = self.conv3(x, edge_index)
                
                x = pyg.nn.global_mean_pool(x, batch)
                return x
        
        return SAGE()
    
    def forward(self, batch: Any) -> Any:
        """Forward pass"""
        batch = batch.to(self.device)
        
        # Forward through model
        out = self.model(batch.x, batch.edge_index, batch.batch)
        
        return out, batch.y
    
    def compute_loss(self, outputs: Any, batch: Any) -> Any:
        """Compute loss"""
        out, labels = outputs
        labels = labels.to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        loss = criterion(out, labels)
        
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
        
        MessageProtocol.status("saved", f"GNN model saved to {output_path}")


# ============================================================================
# REGISTER PLUGIN
# ============================================================================

REGISTRY.register_data_loader(Modality.GRAPH, GraphDataLoader)
REGISTRY.register_model_handler("graph", GNNModelHandler)

MessageProtocol.debug("Graph plugin loaded", {
    "torch_geometric_available": TORCH_GEOMETRIC_AVAILABLE,
    "networkx_available": NETWORKX_AVAILABLE
})
