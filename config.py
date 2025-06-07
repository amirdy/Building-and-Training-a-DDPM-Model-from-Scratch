from dataclasses import dataclass

@dataclass
class Config:

    # Training parameters
    epochs = 2000         # Number of training epochs
    lr = 1e-4             # Learning rate for the optimizer
    batch_size = 64       # Number of samples per training batch
    num_workers = 0       # Number of workers for data loading

    # Model architecture parameters
    in_channels = 3             # Number of input channels (RGB images)
    H = 32                      # Height/Width of the input images
    base_channels = 64          # Base number of channels in the UNet
    multipliers = (1,2,4,8,16)  # Channel multipliers for each UNet level
    num_heads = 8               # Number of attention heads in transformer blocks

    # Diffusion process parameters
    num_timesteps = 1000 # Number of diffusion timesteps
    beta_start = 1e-4    # Starting noise schedule value
    beta_end = 0.02      # Ending noise schedule value
