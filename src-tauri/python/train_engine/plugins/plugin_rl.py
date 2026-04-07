"""
Reinforcement Learning Plugin for FrameTrain v2
===============================================
This plugin adds support for RL algorithms:
- PPO (Proximal Policy Optimization)
- DQN (Deep Q-Network)
- SAC (Soft Actor-Critic)
- A2C (Advantage Actor-Critic)

MANIFEST:
{
    "name": "Reinforcement Learning Plugin",
    "description": "RL algorithms (PPO, DQN, SAC, A2C)",
    "modality": "reinforcement",
    "required": [
        "torch",
        "gymnasium",
        "numpy"
    ],
    "optional": [
        "stable-baselines3"
    ],
    "python": "3.8"
}

Installation:
    pip install stable-baselines3[extra]
    pip install gymnasium  # Modern replacement for gym

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
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import gymnasium as gym
    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False

try:
    from stable_baselines3 import PPO, DQN, SAC, A2C
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


# Import from main engine
try:
    from train_engine import (
        BaseDataLoader, BaseModelHandler, BaseTrainer, Modality, 
        REGISTRY, MessageProtocol, TrainingConfig
    )
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent))
    from train_engine import (
        BaseDataLoader, BaseModelHandler, BaseTrainer, Modality, 
        REGISTRY, MessageProtocol, TrainingConfig
    )


# ============================================================================
# RL DATA LOADER (Special case - uses environment)
# ============================================================================

class RLDataLoader(BaseDataLoader):
    """Data loader for RL - manages environment"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.env = None
        self.env_name = None
    
    def load(self) -> Tuple[Any, Optional[Any], Optional[Any]]:
        """Load RL environment"""
        try:
            if not GYMNASIUM_AVAILABLE:
                raise ImportError(
                    "Gymnasium not installed. Install with: pip install gymnasium"
                )
            
            MessageProtocol.status("loading", "Loading RL environment...")
            
            # Get environment name from config
            self.env_name = getattr(self.config, 'env_name', 'CartPole-v1')
            
            # Create environment
            try:
                self.env = gym.make(self.env_name)
                MessageProtocol.status("loading", f"Created environment: {self.env_name}")
            except Exception as e:
                MessageProtocol.error(f"Failed to create environment: {e}", 
                    "Available environments: CartPole-v1, MountainCar-v0, LunarLander-v2, etc.")
                raise
            
            # Get environment info
            obs_space = self.env.observation_space
            action_space = self.env.action_space
            
            MessageProtocol.status("loading", 
                f"Observation space: {obs_space}, Action space: {action_space}")
            
            # For RL, we don't use traditional dataloaders
            # The environment IS the data source
            return self.env, None, None
            
        except Exception as e:
            MessageProtocol.error("Failed to load RL environment", str(e))
            raise
    
    def get_sample_info(self) -> Dict[str, Any]:
        """Get environment information"""
        if self.env:
            return {
                "env_name": self.env_name,
                "observation_space": str(self.env.observation_space),
                "action_space": str(self.env.action_space)
            }
        return {}


# ============================================================================
# RL MODEL HANDLER
# ============================================================================

class RLModelHandler(BaseModelHandler):
    """Handler for RL algorithms"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.model_type = None
        self.env = None
    
    def load_model(self):
        """Load RL model/algorithm"""
        try:
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch not available")
            
            if not SB3_AVAILABLE:
                raise ImportError(
                    "Stable-Baselines3 not installed. Install with:\n"
                    "pip install stable-baselines3[extra]"
                )
            
            MessageProtocol.status("loading", "Loading RL algorithm...")
            
            # Determine device
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                device_str = "cuda"
                MessageProtocol.status("device", f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device("cpu")
                device_str = "cpu"
                MessageProtocol.status("device", "Using CPU")
            
            model_path = self.config.model_path.lower()
            
            # Get environment name
            env_name = getattr(self.config, 'env_name', 'CartPole-v1')
            
            # Create vectorized environment
            self.env = DummyVecEnv([lambda: gym.make(env_name)])
            
            # Determine algorithm
            if "ppo" in model_path:
                self.model_type = "ppo"
                MessageProtocol.status("loading", "Creating PPO agent")
                
                self.model = PPO(
                    "MlpPolicy",
                    self.env,
                    learning_rate=self.config.learning_rate,
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    verbose=0,
                    device=device_str
                )
            
            elif "dqn" in model_path:
                self.model_type = "dqn"
                MessageProtocol.status("loading", "Creating DQN agent")
                
                self.model = DQN(
                    "MlpPolicy",
                    self.env,
                    learning_rate=self.config.learning_rate,
                    buffer_size=100000,
                    learning_starts=1000,
                    batch_size=32,
                    tau=1.0,
                    gamma=0.99,
                    train_freq=4,
                    gradient_steps=1,
                    target_update_interval=1000,
                    verbose=0,
                    device=device_str
                )
            
            elif "sac" in model_path:
                self.model_type = "sac"
                MessageProtocol.status("loading", "Creating SAC agent")
                
                self.model = SAC(
                    "MlpPolicy",
                    self.env,
                    learning_rate=self.config.learning_rate,
                    buffer_size=100000,
                    learning_starts=100,
                    batch_size=256,
                    tau=0.005,
                    gamma=0.99,
                    train_freq=1,
                    gradient_steps=1,
                    verbose=0,
                    device=device_str
                )
            
            elif "a2c" in model_path:
                self.model_type = "a2c"
                MessageProtocol.status("loading", "Creating A2C agent")
                
                self.model = A2C(
                    "MlpPolicy",
                    self.env,
                    learning_rate=self.config.learning_rate,
                    n_steps=5,
                    gamma=0.99,
                    gae_lambda=1.0,
                    ent_coef=0.0,
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    verbose=0,
                    device=device_str
                )
            
            else:
                # Default to PPO
                self.model_type = "ppo"
                MessageProtocol.status("loading", "Creating default PPO agent")
                
                self.model = PPO(
                    "MlpPolicy",
                    self.env,
                    learning_rate=self.config.learning_rate,
                    verbose=0,
                    device=device_str
                )
            
            MessageProtocol.status("loaded", f"RL algorithm loaded: {self.model_type}")
            
        except Exception as e:
            MessageProtocol.error("RL model loading failed", str(e))
            raise
    
    def forward(self, batch: Any) -> Any:
        """Forward pass (not used in RL)"""
        # RL uses model.learn() instead
        return None
    
    def compute_loss(self, outputs: Any, batch: Any) -> Any:
        """Compute loss (handled internally by SB3)"""
        return torch.tensor(0.0)
    
    def save_model(self, path: str):
        """Save RL model"""
        output_path = Path(path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save(output_path / "model")
        
        # Save config
        config_dict = {
            "model_type": self.model_type,
            "env_name": getattr(self.config, 'env_name', 'CartPole-v1')
        }
        
        with open(output_path / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        MessageProtocol.status("saved", f"RL model saved to {output_path}")


# ============================================================================
# RL TRAINER (Custom trainer for RL)
# ============================================================================

class RLTrainer(BaseTrainer):
    """Custom trainer for RL algorithms"""
    
    def __init__(self, config: TrainingConfig, model_handler: RLModelHandler,
                 data_loader: RLDataLoader):
        super().__init__(config, model_handler, data_loader)
        self.callback = None
    
    def setup(self):
        """Setup RL training"""
        MessageProtocol.status("setup", "Setting up RL training...")
        
        # Create callback for progress updates
        class ProgressCallback(BaseCallback):
            def __init__(self, trainer):
                super().__init__()
                self.trainer = trainer
                self.episode_rewards = []
                self.episode_lengths = []
            
            def _on_step(self) -> bool:
                # Check if training should stop
                if self.trainer.is_stopped:
                    return False
                
                # Log progress every N steps
                if self.n_calls % 1000 == 0:
                    mean_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
                    
                    MessageProtocol.progress(
                        epoch=1,
                        total_epochs=1,
                        step=self.n_calls,
                        total_steps=self.trainer.config.max_steps,
                        train_loss=float(-mean_reward),  # Use negative reward as "loss"
                        learning_rate=self.trainer.config.learning_rate
                    )
                
                return True
            
            def _on_rollout_end(self) -> None:
                # Collect episode statistics
                if hasattr(self.locals.get('infos', [{}])[0], 'episode'):
                    episode_info = self.locals['infos'][0]['episode']
                    self.episode_rewards.append(episode_info['r'])
                    self.episode_lengths.append(episode_info['l'])
        
        self.callback = ProgressCallback(self)
        
        MessageProtocol.status("ready", "RL training ready")
    
    def train_epoch(self, epoch: int) -> float:
        """Train one epoch (not used in RL)"""
        return 0.0
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate RL agent"""
        # Simple evaluation: run a few episodes
        total_reward = 0
        num_episodes = 10
        
        for _ in range(num_episodes):
            obs = self.data_loader.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = self.model_handler.model.predict(obs, deterministic=True)
                obs, reward, done, _ = self.data_loader.env.step(action)
                episode_reward += reward
            
            total_reward += episode_reward
        
        mean_reward = total_reward / num_episodes
        return {"mean_reward": mean_reward}
    
    def train(self):
        """Main RL training loop"""
        import time
        self.start_time = time.time()
        
        try:
            MessageProtocol.status("starting", "Starting RL training...")
            
            # Load model and environment
            self.model_handler.load_model()
            self.data_loader.load()
            self.setup()
            
            MessageProtocol.status("training", "RL training in progress...")
            
            # Determine total timesteps
            total_timesteps = self.config.max_steps if self.config.max_steps > 0 else 100000
            
            # Train using SB3's learn method
            self.model_handler.model.learn(
                total_timesteps=total_timesteps,
                callback=self.callback
            )
            
            # Training complete
            if not self.is_stopped:
                self.model_handler.save_model(self.config.output_path)
                
                # Evaluate final performance
                eval_results = self.evaluate()
                
                training_duration = int(time.time() - self.start_time)
                final_metrics = {
                    "mean_reward": eval_results.get("mean_reward", 0),
                    "training_duration_seconds": training_duration,
                    "total_timesteps": total_timesteps
                }
                
                MessageProtocol.complete(self.config.output_path, final_metrics)
            else:
                MessageProtocol.status("stopped", "RL training was stopped")
                
        except Exception as e:
            import traceback
            MessageProtocol.error("RL training failed", traceback.format_exc())
            raise


# ============================================================================
# REGISTER PLUGIN
# ============================================================================

REGISTRY.register_data_loader(Modality.REINFORCEMENT, RLDataLoader)
REGISTRY.register_model_handler("rl", RLModelHandler)
REGISTRY.register_trainer("rl", RLTrainer)

MessageProtocol.debug("RL plugin loaded", {
    "gymnasium_available": GYMNASIUM_AVAILABLE,
    "sb3_available": SB3_AVAILABLE
})
