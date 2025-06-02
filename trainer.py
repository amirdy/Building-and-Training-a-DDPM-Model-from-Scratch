import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path  
from PIL  import Image
import torch.nn.functional as F
from models.noise_scheduler import NoiseScheduler
 

class Trainer:
    def __init__(self, config, train_dataloader, val_dataloader, ddpm, device):
        self.noise_schedular = NoiseScheduler(config.num_timesteps, config.beta_start, config.beta_end, device)
        self.config = config
        self.epochs = self.config.epochs
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = ddpm
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.config.lr)
        self.best_val_loss = float('inf')
        self.checkpoint_dir = Path("ckpt")
        self._prepare_checkpoint_dir()
 
    def _prepare_checkpoint_dir(self):
        """ Prepare the checkpoint directory. """
        try:
            self.checkpoint_dir.mkdir(parents=True)
            print(f"Checkpoint directory is ready at: {self.checkpoint_dir}...")
        except FileExistsError:
            print(f"The checkpoint directory ({self.checkpoint_dir}) already exists...")   

    def _train_step(self, x0, t):
        '''
        Args: 
            x0: a tensor of shape (B, C, H, W)
            t: a tensor of shape (B, 1)

        Return:
            single value returing loss
        '''
        self.model.train()
        eps = torch.randn_like(x0)
        xt = self.noise_schedular.add_noise(x0, t, eps)
        # with torch.autocast(device_type=str(self.device), dtype=torch.bfloat16):
        #     x0 = x0.to(self.device, dtype=torch.bfloat16)
        predicted_noise = self.model(xt, t)
        loss = self.criterion(predicted_noise, eps)

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()

        return loss.item()

    def _val_step(self, x0, t):
        
        self.model.eval()
        with torch.no_grad():
            # with torch.autocast(device_type=str(self.device), dtype=torch.bfloat16):
            #     x0 = x0.to(self.device, dtype=torch.bfloat16)
            eps = torch.randn_like(x0)
            xt = self.noise_schedular.add_noise(x0, t, eps)
            predicted_noise = self.model(xt, t)
            loss = self.criterion(predicted_noise, eps)
        
        return loss.item()
    
    def _save_checkpoint(self):
        model_filename = self.checkpoint_dir / f'best_model.pth'
        torch.save(self.model.state_dict(), model_filename)

    def sample(self):
        self.model.eval()
        with torch.no_grad():
            config = self.config
            sched = self.noise_schedular
            images = []
            x = torch.randn((1, config.in_channels, config.H, config.H)).to(self.device)
            assets_dir = Path("assets")
            assets_dir.mkdir(exist_ok=True)

            for t in reversed(range(config.num_timesteps)):
                t_tensor = torch.tensor([t]).unsqueeze(0).to(x.device)
                
                # Predict noise
                predicted_noise = self.model(x, t_tensor)
                
                alpha_t = sched.alphas[t]
                alpha_bar_t = sched.alpha_bars[t]
                alpha_bar_prev = sched.alpha_bars[t-1] if t > 0 else torch.tensor(1.0).to(x.device)
                
                # Calculate coefficients
                beta_t = sched.betas[t]
                if t > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                
                # Calculate mean
                mean = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise)
                
                # Calculate variance
                if t > 0:
                    variance = ((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * beta_t
                else:
                    variance = 0

                variance = torch.as_tensor(variance, device=x.device, dtype=x.dtype)
                # Update x
                x = mean + torch.sqrt(variance) * noise
                
                # # Clamp values to prevent extreme values
                # x = torch.clamp(x, -1, 1)
                if t % 100 == 0:
                    temp = ((x + 1) / 2)
                    temp = temp.clamp(0, 1)
                    temp = temp * 255.
                    temp = temp[0, :, :, :].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    images.append(temp)

            image_recovered = ((x + 1) / 2)
            image_recovered = image_recovered.clamp(0, 1)
            image_recovered = image_recovered * 255.
            image_recovered_np = image_recovered[0, :, :, :].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            pil_img = Image.fromarray(image_recovered_np)
            pil_img.save(assets_dir / "sample.png")

    def plot_image_grid(self, images, rows=3, cols=3):
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))

        for i, ax in enumerate(axes.flat):
            if i < len(images):
                ax.imshow(images[i])
                ax.axis('off')
            else:
                ax.remove()
    
        plt.tight_layout()
        plt.savefig("sample.png", dpi=300)
        plt.show()

    def train(self):
        for epoch in range(self.epochs):
            train_loss = []
            val_loss = []

            for x0, t in self.train_dataloader:
                x0, t = x0.to(self.device), t.to(self.device)      
                loss = self._train_step(x0, t)
                train_loss.append(loss)

            for x0, t in self.val_dataloader:
                x0, t = x0.to(self.device), t.to(self.device)      
                loss = self._val_step(x0, t)
                val_loss.append(loss)

            train_loss = np.mean(train_loss)
            val_loss = np.mean(val_loss)

            if  val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint()
            
            print(f'Epoch {epoch + 1}, train loss: {np.mean(train_loss):.4f}, val loss: {np.mean(val_loss):.4f}')
      
            if epoch%20 == 0 or epoch== self.epochs-1:
                self.sample()






