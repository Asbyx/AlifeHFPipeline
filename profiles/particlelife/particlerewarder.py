from rlhfalife.utils import *
from rlhfalife.torchutils.torchrewarder import TorchRewarder
from .utils import get_color_tables, render_frame
from .clipvip import clipvip32
import torch
import torch.nn as nn
from typing import Optional, List, Any
import os

class ParticleRewarder(TorchRewarder):
    """
    A rewarder that uses a frozen CLIP-ViP model with a mini-head to score particle life simulations
    """
    def __init__(self, clipvip_path, config=None, model_path=None, device="cuda" if torch.cuda.is_available() else "cpu", simulator: Optional[Simulator] = None):
        """
        Initialize the ParticleRewarder
        
        Args:
            config: Dictionary with configuration parameters:
                - num_frames: Number of frames to process (default: 12)
                - lr: Learning rate for training (default: 0.0001)
                - batch_size: Batch size for training (default: 8)
                - epochs: Number of training epochs (default: 50)
                - head_type: Type of head to use (default: 'linear')
            model_path: Path to save/load the model
            device: Device to run the model on
        """
        # Initialize with default config values
        config = config or {}
        config.setdefault('num_frames', 12)
        config.setdefault('lr', 0.0001)
        config.setdefault('batch_size', 8)
        config.setdefault('epochs', 50)
        wandb_params = {
            "project": "RLHF particle life",
            "name": model_path.split("\\")[-1],
            "config": config
        }

        super().__init__(config=config, model_path=model_path, device=device, simulator=simulator, wandb_params=wandb_params)
        
        self.num_frames = config['num_frames']
        self.image_size = 224
        # Initialize CLIP-ViP model
        self.clipvip = clipvip32(weights_path=clipvip_path).to(self.device)
        self.clipvip.eval()
        self.clipvip.requires_grad_(False)

        # Get output dimensions by doing a forward pass 
        with torch.no_grad():
            dummy_input = torch.randn(1, self.num_frames, 3, self.image_size, self.image_size, device=self.device)
            outputs = self.clipvip(dummy_input)
            _, embed_dim = outputs.last_hidden_state.shape[-2:]
            
        # Initialize minihead
        # Dictionary of head architectures
        head_architectures = {
            'linear': lambda dim: nn.Linear(dim, 1),
            'mlp': lambda dim: nn.Sequential(
                nn.Linear(dim, dim//2),
                nn.ReLU(),
                nn.Linear(dim//2, 1)
            ),
            # xgboost ? 
        }
        if self.config['head_type'] not in head_architectures:
            raise ValueError(f"Invalid head type: {self.config['head_type']}")
            
        # Create the head and move to device
        self.head = head_architectures[self.config['head_type']](embed_dim).to(self.device)
    
        # Move model to the appropriate device
        self.to(device)
        
    def parameters(self, recurse: bool = True):
        return self.head.parameters(recurse)

    def preprocess(self, data: List[Any]) -> torch.Tensor:
        """
        Preprocess the simulation output data to create video frames for CLIPVIP.
        Converts particle positions and types into a batch of downscaled video tensors.

        Args:
            data (List[Dict]): Batch of simulation outputs.
                Each dict contains:
                    "positions": torch.Tensor (total_steps, num_particles, 2) on CPU.
                    "types": torch.Tensor (num_particles,) on CPU.
            
        Returns:
            torch.Tensor: Preprocessed video data as (B, T, C, H, W), float, normalized to [0,1].
                          T is self.num_frames, H and W are self.image_size. C is RGB.
        """
        batch_size = len(data)

        target_h, target_w = self.image_size, self.image_size
        output_frames_batch = torch.zeros((batch_size, self.num_frames, 3, target_h, target_w), device=self.device, dtype=torch.uint8)

        for b, sample in enumerate(data):
            positions_history = sample.get("positions")
            indices = torch.linspace(0, positions_history.shape[0] - 1, self.num_frames).int()
            positions_history = positions_history[indices]

            types = sample.get("types")

            color_table_size = types.max().item() + 1
            color_table = get_color_tables(color_table_size).to(positions_history.device)

            for t, positions in enumerate(positions_history):
                frame = render_frame(
                    positions, 
                    types, 
                    target_w, 
                    target_h, 
                    color_table, 
                    particle_radius=1 
                )
                
                # Convert NumPy frame (H, W, C) BGR to PyTorch tensor (C, H, W) RGB
                frame = frame.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
                frame = frame[[2,1,0], :, :] # Swap B and R: (B,G,R) -> (R,G,B)
                output_frames_batch[b, t] = frame # Already on self.device

        # todo: Normalize to [0,1] float for the model
        return output_frames_batch.float()
    
    def forward(self, x):
        """
        Forward pass through the model
        
        Args:
            x: Input tensor of shape (B, T, C, H, W)
            
        Returns:
            Output scores of shape (B,)
        """        
        features = self.clipvip(x) # (B, T, C, H, W) -> (B, ? token number, embed_dim)
        outputs = self.head(features.pooler_output) # We take the mean of the last hidden state
        return outputs.squeeze(-1)
    
    def save(self):
        """
        Save the model to disk using the model_path attribute.
        """
        torch.save({
            'model_state_dict': self.head.state_dict(),
            'config': self.config
        }, self.model_path)
        print(f"Model saved to {self.model_path}")
    
    def load(self):
        """
        Load the model from disk using the model_path attribute.
        
        Returns:
            The loaded rewarder
        """
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Update config
            self.config = checkpoint.get('config', self.config)
            
            # Load model state
            self.head.load_state_dict(checkpoint['model_state_dict'])
            
            print(f"Model loaded from {self.model_path}")
        else:
            print(f"No model found at {self.model_path}, using initialized model.")
            
        return self
