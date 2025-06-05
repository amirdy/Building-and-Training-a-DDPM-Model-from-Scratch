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
    multipliers = (1,2,4, 8,16)
    # pos_emb_dim = 128
    num_heads = 8
