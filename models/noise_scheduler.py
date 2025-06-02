import torch

class NoiseScheduler:
    def __init__(self, num_timesteps, beta_start, beta_end, device):
        self.num_timesteps = num_timesteps            
        self.beta_start = beta_start            
        self.beta_end = beta_end  

        # Linear schedule for beta values: controls the noise strength at each timestep          
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device) # (T,)

        # Precompute alpha values (1 - beta) and their cumulative products
        self.alphas = 1.0 - self.betas                                           # (T,)
        self.sqrt_alphas = torch.sqrt(self.alphas)                               # (T,)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)                      # (T,)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)                       # (T,)
        self.one_minus_alpha_bars = 1 - self.alpha_bars                          # (T,) 
        self.sqrt_one_minus_alpha_bars = torch.sqrt(self.one_minus_alpha_bars)   # (T,)

        
                
    def add_noise(self, x0, t, eps):
        """
        Add noise to x0 at timestep t using the DDPM forward process:
        x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps

        Args:
            x0 (Tensor): Original images, shape (B, C, H, W), range from -1 to 1
            t (Tensor): Timesteps for each image in batch, shape (B,)
            eps (Tensor): Random noise sampled from N(0, 1), shape (B, C, H, W)

        Returns:
            x_t (Tensor): Noisy image at timestep t
        """
        
        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)
        x_t = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * eps

        return x_t
    



