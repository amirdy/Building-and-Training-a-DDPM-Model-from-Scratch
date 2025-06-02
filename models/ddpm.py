import torch.nn as nn 
from models.unet import UNet

class DDPM(nn.Module):
    def __init__(self, config):
        """
        Denoising Diffusion Probabilistic Model (DDPM) wrapper around a UNet.

        Args:
            config: Configuration object with attributes:
                - in_channels (int): Number of input channels (e.g., 3 for RGB)
                - base_channels (int): Base number of channels for UNet
                - multipliers (tuple): Channel multipliers for UNet down/up sampling
                - num_heads (int): Number of attention heads for LinearAttention module
        """
        super().__init__()
        self.unet = UNet(
                    config.in_channels,
                    config.base_channels,
                    config.multipliers,
                    4 * config.base_channels,  # Positional embedding dimension
                    config.num_heads
                )
    def forward(self, x, t ):
        """
        Forward pass predicting noise added to input at timestep t.

        Args:
            x (torch.Tensor): Noisy input image tensor of shape (B, C, H, W)
            t (torch.Tensor): Tensor of timesteps of shape (B, 1)

        Returns:
            torch.Tensor: Predicted noise tensor of same shape as input x
        """
        predicted_noise = self.unet(x, t)

        return predicted_noise
    
