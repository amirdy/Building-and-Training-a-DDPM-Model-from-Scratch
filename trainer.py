import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path  
from PIL  import Image
import torch.nn.functional as F
from models.noise_scheduler import NoiseScheduler
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

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
        # print(f"x min/max: 0", x.min().item(), x.max().item(),x.std().item())

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
                if t%100 == 0:
                    temp = ((x + 1) / 2)
                    temp = temp.clamp(0, 1)
                    temp = temp * 255.
                    temp = temp[0,:,:,:].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    images.append(temp)

        # print("Sample min/max:", x.min().item(), x.max().item())
            
            # Convert to image
        # self.plot_image_grid(images[-9:], rows=3, cols=3)

        image_recovered = ((x + 1) / 2)
        image_recovered = image_recovered.clamp(0, 1)
        image_recovered = image_recovered * 255.
        image_recovered_np = image_recovered[0,:,:,:].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        pil_img = Image.fromarray(image_recovered_np)
        pil_img.save("sample.png")
        self._create_animation(images, 'diffusion_process.gif')



    def _create_animation(self, images, output_path="diffusion_process.gif", fps=3):
        """Create a GIF animation from a list of images."""
        fig = plt.figure(figsize=(2, 2))
        plt.axis('off')
        
        # Create animation
        def update(frame):
            plt.clf()
            plt.axis('off')
            f = 900 - frame*100
            plt.imshow(images[frame])
            plt.title(f"Step {f}")
        
        anim = FuncAnimation(
            fig, 
            update, 
            frames=len(images), 
            interval=1000/fps
        )
        
        # Save as GIF
        anim.save(output_path, writer='pillow', fps=fps, dpi=100)
        print(f"✅ Saved animation to {output_path}")


    def _create_animation_(self, images, timesteps, output_path, fps=15):
        """Create a GIF animation showing the denoising process."""
        fig = plt.figure(figsize=(8, 8))
        plt.axis('off')
        
        # Create initial frame
        img = plt.imshow(images[0])
        title = plt.title(f"Timestep: {timesteps[0]} (Noisy)")
        
        def update(frame):
            img.set_array(images[frame])
            if timesteps[frame] == 0:
                title.set_text("Final Image (Timestep: 0)")
            else:
                title.set_text(f"Denoising... Timestep: {timesteps[frame]}")
            return img, title
        
        # Calculate total frames and duration
        total_frames = len(images)
        duration_ms = 1000 / fps  # Duration per frame in ms
        
        # Create animation with slower speed
        anim = FuncAnimation(
            fig, 
            update, 
            frames=total_frames,
            interval=duration_ms,
            blit=True
        )
        
        # Save with slower speed and higher quality
        writer = animation.PillowWriter(
            fps=fps,
            bitrate=1800,
            extra_args=['-quality', '90']
        )
        
        anim.save(
            output_path,
            writer=writer,
            dpi=100,
            progress_callback=lambda i, n: print(f'Saving frame {i+1}/{n}')
        )
        print(f"✅ Saved denoising animation to {output_path}")
    def ddim_sample(self, num_steps=50,save_path="ddim_sample.png"):
        """
        DDIM sampling: deterministic fast sampling from the trained DDPM model.

        Args:
            num_steps (int): Number of DDIM steps (must be <= config.num_timesteps)
            eta (float): 0 = deterministic (DDIM), >0 = stochasticity like DDPM
        """
        self.model.eval()
        config = self.config
        sched = self.noise_schedular

        # Use linearly spaced indices to skip steps
        total_steps = config.num_timesteps
        ddim_steps = np.linspace(0, total_steps - 1, num_steps, dtype=int)

        with torch.no_grad():
            x = torch.randn((1, config.in_channels, config.H, config.H), device=self.device)

            for i in range(num_steps - 1, -1, -1):
                t = ddim_steps[i]
                t_tensor = torch.full((1, 1), t, device=self.device, dtype=torch.long)

                # Predict noise and x0
                eps_theta = self.model(x, t_tensor)
                alpha_bar_t = sched.alpha_bars[t]
                x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps_theta) / torch.sqrt(alpha_bar_t)
                sigma_t = 0
                noise = 0
                if i > 0:
                    t_prev = ddim_steps[i - 1]
                    alpha_bar_prev = sched.alpha_bars[t_prev]

                    x = torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(1 - alpha_bar_prev - sigma_t ** 2) * eps_theta + sigma_t * noise
                else:
                    x = x0_pred

            # Convert to image
            x_img = ((x + 1) / 2).clamp(0, 1) * 255
            img_np = x_img[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            Image.fromarray(img_np).save(save_path)
            print(f"[DDIM] Saved image to {save_path}")

    def generate_and_plot_samples(self, num_samples=90, img_size=32):
        """
        Generates `num_samples` images using the diffusion model and plots them in a grid.
        Each image will be of shape (img_size x img_size).
        """
        self.model.eval()
        samples = []

        with torch.no_grad():
            for _ in range(num_samples):
                x = torch.randn((1, self.config.in_channels, img_size, img_size)).to(self.device)
                
                for t in reversed(range(self.config.num_timesteps)):
                    t_tensor = torch.tensor([t]).unsqueeze(0).to(self.device)
                    predicted_noise = self.model(x, t_tensor)
                    
                    alpha_t = self.noise_schedular.alphas[t]
                    alpha_bar_t = self.noise_schedular.alpha_bars[t]
                    alpha_bar_prev = self.noise_schedular.alpha_bars[t - 1] if t > 0 else torch.tensor(1.0).to(x.device)
                    beta_t = self.noise_schedular.betas[t]
                    
                    noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
                    mean = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise)
                    variance = ((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * beta_t if t > 0 else 0
                    variance = torch.as_tensor(variance, device=x.device, dtype=x.dtype)
                    x = mean + torch.sqrt(variance) * noise
                
                # Post-process and collect the sample
                x_image = ((x + 1) / 2).clamp(0, 1) * 255.
                img_np = x_image[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                samples.append(img_np)

        # Plot all samples in a 9x10 grid
        rows, cols = 6, 15
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
        for i, ax in enumerate(axes.flat):
            if i < len(samples):
                ax.imshow(samples[i])
                ax.axis('off')
        plt.tight_layout()
        plt.savefig("grid_samples.png", dpi=300)
        plt.show()


    def generate_and_plot_samples_2(self, num_samples=40, img_size=32):
        """
        Generate `num_samples` images in a batch and plot them in a 5x6 grid.
        """
        self.model.eval()
        config = self.config
        sched = self.noise_schedular

        with torch.no_grad():
            x = torch.randn((num_samples, config.in_channels, img_size, img_size), device=self.device)

            for t in reversed(range(config.num_timesteps)):
                t_tensor = torch.full((num_samples, 1), t, device=self.device, dtype=torch.long)
                predicted_noise = self.model(x, t_tensor)

                alpha_t = sched.alphas[t]
                alpha_bar_t = sched.alpha_bars[t]
                alpha_bar_prev = sched.alpha_bars[t - 1] if t > 0 else torch.tensor(1.0, device=self.device)
                beta_t = sched.betas[t]

                noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
                mean = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise)
                variance = ((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * beta_t if t > 0 else 0.0
                variance = torch.as_tensor(variance, device=x.device, dtype=x.dtype)
                x = mean + torch.sqrt(variance) * noise

            x = ((x + 1) / 2).clamp(0, 1) * 255
            x = x.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

        # Plot in 5x6 grid
        rows, cols = 4, 10
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        for i, ax in enumerate(axes.flat):
            ax.imshow(x[i])
            ax.axis("off")

        plt.tight_layout()
        plt.savefig("grid_samples.png", dpi=300)
        plt.show()

    def generate_and_plot_samples_3(self, num_samples=36, img_size=32, ddim_steps=50):
        """
        Generate `num_samples` images in a batch using DDIM sampling and plot them in a grid.
        """
        self.model.eval()
        config = self.config
        sched = self.noise_schedular
        total_steps = config.num_timesteps
        ddim_indices = np.linspace(0, total_steps - 1, ddim_steps, dtype=int)

        with torch.no_grad():
            x = torch.randn((num_samples, config.in_channels, img_size, img_size), device=self.device)

            for i in range(ddim_steps - 1, -1, -1):
                t = ddim_indices[i]
                t_tensor = torch.full((num_samples, 1), t, device=self.device, dtype=torch.long)

                eps_theta = self.model(x, t_tensor)

                alpha_bar_t = sched.alpha_bars[t]
                x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps_theta) / torch.sqrt(alpha_bar_t)

                if i > 0:
                    t_prev = ddim_indices[i - 1]
                    alpha_bar_prev = sched.alpha_bars[t_prev]
                    x = torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(1 - alpha_bar_prev) * eps_theta
                else:
                    x = x0_pred

            # Post-process images
            x = ((x + 1) / 2).clamp(0, 1) * 255
            x = x.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

        # Plot in 3x12 grid
        rows, cols = 3, 12
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        for i, ax in enumerate(axes.flat):
            ax.imshow(x[i])
            ax.axis("off")

        plt.tight_layout()
        plt.savefig("grid_samples_ddim.png", dpi=300)
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
            torch.save(self.model.state_dict(), 'last_model.pth')
            if epoch%20 == 0 or epoch== self.epochs-1:
                self.sample()






    