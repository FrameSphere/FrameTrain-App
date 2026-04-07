"""
Reinforcement Learning Test Plugin for FrameTrain v2
====================================================
Testing support for RL agents:
- Policy Evaluation
- Environment Testing
- Performance Metrics

MANIFEST:
{
    "name": "RL Test Plugin",
    "description": "Testing for Reinforcement Learning agents",
    "modality": "rl",
    "required": ["torch", "numpy"],
    "optional": ["gym", "stable-baselines3"],
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


class RLTestLoader(BaseTestLoader):
    """Test loader for RL agents"""
    
    def __init__(self, config: TestConfig):
        super().__init__(config)
        self.state_dim = 4
        self.action_dim = 2
        
    def load_model(self):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        self.device = self.get_device()
        MessageProtocol.status("loading", "Loading RL agent...")
        
        # Simple policy network
        class PolicyNetwork(nn.Module):
            def __init__(self, state_dim, action_dim):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(state_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Linear(64, action_dim)
                )
            
            def forward(self, state):
                return self.network(state)
        
        self.model = PolicyNetwork(self.state_dim, self.action_dim).to(self.device).eval()
        
        model_path = Path(self.config.model_path)
        checkpoint = model_path / "model.pth"
        if checkpoint.exists():
            self.model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
        
        MessageProtocol.status("loaded", "RL agent loaded")
    
    def load_test_data(self) -> List[Dict[str, Any]]:
        # Generate test episodes
        MessageProtocol.status("loading", "Generating test episodes...")
        
        num_episodes = 10 if not self.config.max_samples else self.config.max_samples
        test_data = []
        
        for ep in range(num_episodes):
            # Generate episode trajectory
            states = [np.random.randn(self.state_dim) for _ in range(100)]
            rewards = [np.random.rand() for _ in range(100)]
            
            test_data.append({
                "episode_id": ep,
                "states": states,
                "rewards": rewards
            })
        
        MessageProtocol.status("loaded", f"Loaded {len(test_data)} test episodes")
        return test_data
    
    def test_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            total_reward = 0
            actions_taken = []
            
            for state in sample["states"]:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    action_logits = self.model(state_tensor)
                    action = torch.argmax(action_logits, dim=1).item()
                
                actions_taken.append(action)
            
            total_reward = sum(sample["rewards"])
            
            return {
                "episode_id": sample["episode_id"],
                "total_reward": total_reward,
                "episode_length": len(sample["states"]),
                "actions_distribution": {
                    "action_0": actions_taken.count(0),
                    "action_1": actions_taken.count(1) if self.action_dim > 1 else 0
                },
                "inference_time": time.time() - start_time
            }
        except Exception as e:
            return {"episode_id": sample["episode_id"], "error": str(e)}
    
    def compute_metrics(self, all_results: List[Dict]) -> Dict[str, Any]:
        valid = [r for r in all_results if "error" not in r]
        
        if not valid:
            return {"error": "No valid episodes"}
        
        rewards = [r["total_reward"] for r in valid]
        episode_lengths = [r["episode_length"] for r in valid]
        
        return {
            "total_episodes": len(valid),
            "mean_reward": sum(rewards) / len(rewards),
            "std_reward": np.std(rewards),
            "max_reward": max(rewards),
            "min_reward": min(rewards),
            "mean_episode_length": sum(episode_lengths) / len(episode_lengths),
            "average_inference_time": sum(r["inference_time"] for r in valid) / len(valid)
        }


TEST_REGISTRY.register_test_loader(Modality.RL, RLTestLoader)
MessageProtocol.debug("RL test plugin loaded", {"torch": TORCH_AVAILABLE})
