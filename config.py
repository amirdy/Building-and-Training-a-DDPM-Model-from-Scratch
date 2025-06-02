from dataclasses import dataclass

@dataclass
class Config:
    epochs = 2000
    in_channels = 3
    H = 32
    num_timesteps = 1000
    beta_start = 1e-4
    beta_end = 0.02
    lr = 1e-4
    batch_size = 64
    num_workers = 0
    base_channels = 64
    multipliers = (1,2,4,4,8)
    # pos_emb_dim = 128
    num_heads = 4



    # # Training
    # epochs = 2000
    # lr = 2e-4
    # batch_size = 64
    # num_workers = 4  # Increase if on multi-core CPU
    # use_ema = True   # Exponential Moving Average
    # grad_clip = 1.0  # Gradient clipping

    # # Diffusion
    # num_timesteps = 1000
    # beta_start = 1e-4
    # beta_end = 0.02
    # schedule_type = "cosine"  # "linear" or "cosine"

    # # Architecture
    # in_channels = 3
    # H = 32
    # base_channels = 64
    # multipliers = (1, 2, 4, 8)  # For 32x32, reduce complexity
    # pos_emb_dim = 128
    # num_heads = 4               # Increased for better attention