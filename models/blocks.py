import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class PositionalEmbedding(nn.Module):
    def __init__(self, pos_emb_dim):
        """
        Initialize sinusoidal positional embeddings as in the Transformer paper.

        Args:
            pos_emb_dim (int): Dimensionality of the positional embedding (must be even).
        """
        super().__init__()
        assert pos_emb_dim%2 == 0, 'pos_emb_dim must be even'

        self.half_dim = pos_emb_dim//2
        i = torch.arange(self.half_dim).float()  # [0, 1, ..., half_dim-1]  
        denominator = torch.pow(10000, 2 * i / self.half_dim) 
        
        self.register_buffer("denominator", denominator)  # Not a parameter, but persistent on device

    def forward(self, positions):
        '''
        Compute sinusoidal positional embedding for given positions.

        Args:
            positions (Tensor): Shape (batch_size, 1), -> timesteps

        Returns:
            Tensor: Positional embeddings of shape (batch_size, pos_emb_dim)
        '''
        angles = positions / self.denominator[None, :]       # (batch_size, half_dim)
        embedding = torch.zeros((positions.shape[0], self.half_dim * 2), device=positions.device)  # (batch_size, dim)
        embedding[:, 0::2] = torch.sin(angles)
        embedding[:, 1::2] = torch.cos(angles)

        return embedding
    



class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        A basic convolutional block consisting of:
        - 2D convolution with kernel size 3 and padding 1 (preserves spatial size)
        - Group Normalization  
        - SiLU activation  
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), # no change in the spatial 
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True)
        ) 

    def forward(self, x):
        """
        Forward pass through the block.
        
        Args:
            x (Tensor): Input tensor of shape (B, C_in, H, W)
        
        Returns:
            Tensor: Output tensor of shape (B, C_out, H, W)
        """
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pos_emb_dim):
        """
        A residual block that incorporates time (or positional) embedding information.

        Structure:
        - Block1: Conv2D + GroupNorm + SiLU
        - Time embedding MLP: projects time embedding to match feature map channels
        - Block2: Conv2D + GroupNorm + SiLU
        - Residual connection: uses identity if shapes match, otherwise 1x1 Conv

        Args:
            in_channels (int): Number of input channels in the feature map.
            out_channels (int): Number of output channels in the feature map.
            pos_emb_dim (int): Dimensionality of the time (or positional) embedding vector.
        """
        super().__init__()
        self.block1 = Block(in_channels, out_channels)
        self.t_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(pos_emb_dim, out_channels)
        )
        self.block2 = Block(out_channels, out_channels)
        self.residual_conv = nn.Identity()
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # To evade shape mismatch

    def forward(self, x, t):
        """
        Forward pass of the residual block.

        Args:
            x (Tensor): Input feature map of shape (B, C_in, H, W)
            t (Tensor): Time/positional embedding of shape (B, pos_emb_dim)

        Returns:
            Tensor: Output feature map of shape (B, C_out, H, W)
        """
        h = self.block1(x)                 # (B, out_channels, H, W)
        emb = self.t_mlp(t)                # (B, out_channels)
        emb = emb[:, :, None, None]        # (B, out_channels, 1, 1)
        h = h + emb                        # Broadcast add: (B, out_channels, H, W)
        h = self.block2(h)                 # (B, out_channels, H, W)
        return h + self.residual_conv(x)   # (B, out_channels, H, W)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Downsamples the spatial resolution by a factor of 2 using a patch-unfolding trick,
        and projects the increased channel dimension with a 1x1 convolution.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels after downsampling.
        """
        super().__init__()
        self.downsample = nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2), # Instead of pooling! (taking non-overlapping 2×2 patches from each feature map, and turning each 2×2 patch into additional channels. It’s spatial downsampling with channel expansion.)
        nn.Conv2d(in_channels * 4, out_channels, kernel_size=1) # to have a learnable effect | out_channels = 2 * in_channels
        )
    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (B, in_channels, H, W)
        Returns:
            Tensor: Downsampled tensor of shape (B, out_channels, H/2, W/2)
        """
        return  self.downsample(x)
 


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Upsamples the input spatially by a factor of 2 and applies a convolution
        to make the upsampling learnable.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels after upsampling.
        """
        super().__init__()
        self.upsample = nn.Sequential(

                # Nearest neighbor upsampling (doubles H and W)
                # Input: (B, C, H, W) -> Output: (B, C, 2H, 2W)
                # No trainable parameter (each pixel is duplicated)
                nn.Upsample(scale_factor=2, mode="nearest"), 
                
                # 3x3 convolution with padding to keep H and W unchanged
                # Output: (B, out_channels, 2H, 2W)
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding = 1) # to have a learnable effect | k=3 and padding = 1 leads to unchanging the w and h
        )
    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (B, in_channels, H, W)
        Returns:
            Tensor: Upsampled tensor of shape (B, out_channels, 2H, 2W)
        """
        return  self.upsample(x)


class Normalization(nn.Module):
    def __init__(self, dim):
        """
        Applies GroupNorm.

        Args:
            dim (int): Number of feature channels (C).
        """
        super().__init__()
        self.norm = nn.GroupNorm(8, dim)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, H, W)
        Returns:
            Tensor: Normalized tensor of same shape
        """
        return self.norm(x)
    