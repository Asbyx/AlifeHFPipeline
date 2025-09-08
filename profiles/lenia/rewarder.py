import rlhfalife.utils as rlu
from rlhfalife.torchutils.torchrewarder import TorchRewarder
from .clipvip  import CLIPVIPReward
from typing import List, Any
import torch, torch.nn.functional as F, torch.nn as nn
from pathlib import Path
from torch.optim import AdamW
curpath = Path(__file__).parent.resolve() # UGLY, change it to load from module correctly


class LeniaRewarder(TorchRewarder):
    """
    A rewarder that uses a torch model. 

    The only method that needs to be implemented is the forward method.

    Note: using this class assume that the outputs are torch tensors (dtype=torch.float32) saved as pt files, with torch.save.
    """
    def __init__(self, config: dict, model_path: str, device: str = "cpu", wandb_params: dict = None, simulator=None):
        """
        Initialize the TorchRewarder.
    
        Args:
            config: Dictionary containing configuration parameters. All are optional.
                lr (default 0.001): Learning rate
                epochs (default 100): Number of epochs to train
                batch_size (default 16): Batch size
                val_split (default 0.2): Validation split
                early_stopping_patience (default 10): Early stopping patience
                loss (default "cross_entropy"): Loss function to use. Can be "margin" or "cross_entropy".
                clipvip_size (default 16): Size of the CLIPVIP model. Can be 16 or 32.
                minihead (default True): Whether to use the minihead architecture or not.
                num_frames (default 12): Number of frames to process.
            model_path: Path to save or load the model
            device: Device to run the model on. Defaults to "cuda" if available, otherwise "cpu".
            wandb_params: Dictionary containing wandb parameters. Defaults to None.
        """
        super().__init__(config = config, model_path=model_path, device=device,simulator=simulator, wandb_params=wandb_params)
        clipvip_weights = curpath.parent / 'particlelife'/ 'rewarder' / f'clipvip_{config.get('clipvip_size',32)}.pt'

            
        self.model = CLIPVIPReward(patch_size=config.get('clipvip_size',32), clipvip_weights=clipvip_weights, minihead=config.get('minihead',True),num_frames=config.get('num_frames',12),
                                    device=device)
        self.model.freeze_body()

    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor, assumed to be torch tensors already on the correct device. Should be a batch of data.
            
        Returns:
            Output tensor, shape (batch_size)
        """
        
        return self.model(x) # (B, )
        
    def _setup_optimizer(self):
        """Set up the optimizer using the config parameters"""
        lr = self.config.get('lr', 0.001)
        self.optimizer = AdamW(self.model.head_params(), lr=lr)

    def preprocess(self, data: List[Any]) -> torch.Tensor:
        """
        Preprocess the data.

        Args:
            data: List of (T, 3, H, W) tensors representing the videos. Each tensor should be a video with T frames.
            
        Returns:
            Preprocessed data ready for forward pass (B, T', 3, H', W').
        """
        data = torch.stack(data, dim=0) # (B,T,3,H,W)
        data = self._process_video(data) # (B,T',3,H',W')
        
        return data.to(self.device) # (B,T',3,H',W')
    
    def _process_video(self, tensvid):
        """
            Given a video tensor, returns the processed tensor.

            Args:
            tensvid : (B,T,3,H,W) representing the videos

            Returns:
            (B,T',3,H',W') tensor, processed tensvid in model's format
        """
        tar_T, _, tar_H, tar_W = self.model.input_shape

        B,T,C,H,W = tensvid.shape
        assert tar_T <= T, f'tensvid {T} frames, need at least {tar_T} frames'

        # Take tar_T equally spaced frames
        tensvid = tensvid[:,torch.linspace(0,T-1,tar_T).long()]

        # tensvid = torch.einsum('btchw->bcthw', tensvid) # interpolate expects channels first
        tensvid = tensvid.reshape(B*tar_T,C,H,W) # (B*T,3,H,W)
        # Resize the frames
        tensvid = F.interpolate(tensvid, size=(tar_H,tar_W), mode='bilinear')
        tensvid = tensvid.reshape(B,tar_T,C,tar_H,tar_W)

        assert tensvid.shape[1:] == self.model.input_shape, f'tensvid shape {tensvid.shape[0]} not equal to {self.model.input_shape}'

        return tensvid