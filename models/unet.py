import torch
import torch.nn as nn
from models.blocks import *
from models.linear_attention import LinearAttention

class UNet(nn.Module):
    def __init__(self, in_channels=3,
                 base_channels=64,              # Base channel width (doubled at each downsampling)
                 multipliers=(1, 2, 4, 8, 16),  # Scales for downsampling and upsampling
                 pos_emb_dim=128,               # Dimensionality of positional (time) embedding
                 num_heads=2                    # Attention heads for LinearAttention
                 ):
        super().__init__()

        # Initial and final convolutions (keep spatial dims same via kernel=3 and padding=1)
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1) # changed to 1 and 0 from 7,3
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)  
        )
        
        # Positional (time-step) embedding
        self.t_emb = PositionalEmbedding(pos_emb_dim)

        # Compute feature channel sizes at each resolution
        feature_channels = list(map(lambda i: i * base_channels, multipliers))  # e.g. [64, 128, 256, 512, 1024]
        in_out = list(zip(feature_channels[:-1], feature_channels[1:]))      # e.g. [(64,128), (128,256), (256,512), (512,1024)]
        
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for ch_in, ch_out in in_out: # [(64, 128), (128, 256), (256, 512), (512, 1024)]
            self.downs.append(
                nn.ModuleList([
                    ResidualBlock(ch_in, ch_in, pos_emb_dim),     # ResBlock 1 (preserve ch_in)
                    Normalization(ch_in),
                    LinearAttention(ch_in, num_heads),
                    ResidualBlock(ch_in, ch_in, pos_emb_dim),     # ResBlock 2
                    DownSample(ch_in, ch_out)                     # Reduce spatial size, double channels
                ])                
                )

        for ch_out, ch_in in reversed(in_out): # [(512, 1024), (256, 512), (128, 256), (64, 128)]
            self.ups.append(
                nn.ModuleList([
                    UpSample(ch_in, ch_out),                        # Upsample spatial size, halve channels
                    ResidualBlock(2 * ch_out, ch_out, pos_emb_dim),  
                    Normalization(ch_out),
                    LinearAttention(ch_out, num_heads),
                    ResidualBlock(ch_out, ch_out, pos_emb_dim),
                ])                
                )

        self.middle = ResidualBlock(feature_channels[-1], feature_channels[-1], pos_emb_dim)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
            t: Timestep tensor of shape (B, 1)

        Returns:
            Tensor of shape (B, C, H, W)
        """

        x = self.init_conv(x) # (B, base_channels, H, W)
        t = self.t_emb(t)     # (B, pos_emb_dim)

        skips =[]   # For skip connections

        # Downsampling path
        for residual_block, norm, attention, residual_block_2, down in self.downs:
            x = residual_block(x, t)        # (B, ch_in, H, W)
            x = norm(x)
            x = attention(x)                # Linear attention (preserves shape)
            x = residual_block_2(x, t)
            skips.append(x)                 # Save for skip connection
            x = down(x)                     # (B, ch_out, H/2, W/2)

        # Middle bottleneck
        x = self.middle(x, t)

        # Upsampling path
        for up, residual_block, norm, attention, residual_block_2 in self.ups:
            skip = skips.pop()               # Get corresponding skip connection
            x = up(x)                        # (B, ch_out, H, W)
            x = torch.cat([x, skip], dim=1)  # (B, 2 * ch_out, H, W)
            x = residual_block(x, t)
            x = norm(x)
            x = attention(x)
            x = residual_block_2(x, t)
        
        # Final output projection to original input channels
        x = self.final_conv(x)               # (B, in_channels, H, W)

        return x

