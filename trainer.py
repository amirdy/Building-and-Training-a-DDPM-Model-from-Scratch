import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path  
from PIL  import Image
from models.noise_scheduler import NoiseScheduler
from matplotlib.animation import FuncAnimation


class Trainer:
    def __init__(self, config, train_dataloader, val_dataloader, ddpm, device):
        '''
        Initializes the DDPM trainer with all necessary components.
        
        Args:
            config: Configuration object containing all hyperparameters
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            ddpm: The DDPM model to train
            device: Device to train on (cuda/cpu)
            
        Components initialized:
        - Noise scheduler for the diffusion process
        - Model configuration
        - Data loaders
        - Loss function (MSE)
        - Optimizer (Adam)
        - Checkpoint directory for saving best model
        '''
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
 
    def _prepare_checkpoint_dir(self):
        """ Prepare the checkpoint directory. """
        try:
            self.checkpoint_dir.mkdir(parents=True)
            print(f"Checkpoint directory is ready at: {self.checkpoint_dir}...")
        except FileExistsError:
            print(f"The checkpoint directory ({self.checkpoint_dir}) already exists...")   

    def _train_step(self, x0, t):
        '''
        Performs a single training step for the DDPM model.
        
        The training process follows these steps:
        1. Generate random noise
        2. Add noise to the input image based on the timestep
        3. Predict the noise using the model
        4. Calculate loss between predicted and actual noise
        5. Update model parameters using backpropagation
        
        Args: 
            x0: Original clean images tensor of shape (B, C, H, W)
            t: Timestep tensor of shape (B, 1)

        Return:
            float: The loss value for this training step
        '''
        self.model.train()
        eps = torch.randn_like(x0)
        xt = self.noise_schedular.add_noise(x0, t, eps)
        predicted_noise = self.model(xt, t)
        loss = self.criterion(predicted_noise, eps)

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()

        return loss.item()

    def _val_step(self, x0, t):
        '''
        Performs a single validation step for the DDPM model.
        
        Similar to training step but:
        1. Model is in eval mode
        2. No gradient computation
        3. No parameter updates
        
        This is used to evaluate model performance on validation data
        without affecting the model's parameters.
        
        Args:
            x0: Original clean images tensor of shape (B, C, H, W)
            t: Timestep tensor of shape (B, 1)
            
        Return:
            float: The loss value for this validation step
        '''
        self.model.eval()
        with torch.no_grad():
            eps = torch.randn_like(x0)
            xt = self.noise_schedular.add_noise(x0, t, eps)
            predicted_noise = self.model(xt, t)
            loss = self.criterion(predicted_noise, eps)
        
        return loss.item()
    
    def _save_checkpoint(self):
        model_filename = self.checkpoint_dir / f'best_model.pth'
        torch.save(self.model.state_dict(), model_filename)

    def sample(self):
        '''
        Generates a new image using the trained DDPM model through the full diffusion process.
        
        The sampling process:
        1. Starts with pure random noise
        2. Iteratively denoises the image by predicting and removing noise
        3. Uses the full number of timesteps (1000 by default)
        4. Saves both the final image and a GIF of the denoising process
        
        The method saves:
        - sample.png: The final generated image
        - diffusion_process.gif: Animation of the denoising process
        '''
        self.model.eval()
        with torch.no_grad():
            config = self.config
            sched = self.noise_schedular
            images = []
            x = torch.randn((1, config.in_channels, config.H, config.H)).to(self.device)

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
                
                # Store intermediate denoising steps every 100 timesteps for the diffusion process GIF
                # This creates frames for the animation that shows how the model gradually denoises
                # the image from pure noise to the final generated image
                # This block is only for the GIF generation
                if t%100 == 0:
                    temp = ((x + 1) / 2)
                    temp = temp.clamp(0, 1)
                    temp = temp * 255.
                    temp = temp[0,:,:,:].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    images.append(temp)

        
        x_img = ((x + 1) / 2).clamp(0, 1) * 255
        img_np = x_img[0,:,:,:].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        pil_img.save("sample.png")
        print(f"[DDPM] Saved image to sample.png")
        self._create_animation(images, 'diffusion_process.gif')

    def _create_animation(self, images, output_path="diffusion_process.gif", fps=3):
        '''
        Creates an animated GIF showing the denoising process.
        
        Args:
            images (list): List of numpy arrays containing the intermediate denoised images
            output_path (str): Path where the GIF will be saved
            fps (int): Frames per second for the animation
            
        The animation shows:
        - The gradual denoising process from noise to final image
        - Timestep information for each frame
        - A smooth transition between denoising steps
        '''
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
        print(f"Saved animation to {output_path}")

    def ddim_sample(self, num_steps=50):
        """
        Performs DDIM (Denoising Diffusion Implicit Models) sampling.
        
        DDIM is a faster sampling method that:
        1. Uses fewer steps than the full diffusion process
        2. Makes the sampling process deterministic
        3. Can generate images in fewer steps while maintaining quality
        
        The process:
        1. Starts with random noise
        2. Uses a subset of timesteps (linearly spaced)
        3. Predicts and removes noise in fewer, larger steps
        4. Saves the final generated image
        
        Args:
            num_steps (int): Number of DDIM steps (must be <= config.num_timesteps)
                            Default is 50 steps, much faster than full 1000 steps
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
            img_np = x_img[0,:,:,:].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            pil_img.save('ddim_sample.png')
            print(f"[DDIM] Saved image to ddim_sample.png")

    def train(self):
        '''
        Main training loop for the DDPM model.
        
        The training process:
        1. Iterates through specified number of epochs
        2. For each epoch:
            - Trains on training data
            - Validates on validation data
            - Saves best model based on validation loss
            - Saves last model checkpoint
            - Generates sample images periodically (every 20 epochs)
        
        The method:
        - Tracks and prints training/validation losses
        - Saves model checkpoints
        - Monitors training progress through sample generation
        - Uses early stopping by saving best model based on validation loss
        '''
        self._prepare_checkpoint_dir()
        for epoch in range(self.epochs):
            train_loss = []
            val_loss = []

            # Training phase
            for x0, t in self.train_dataloader:
                x0, t = x0.to(self.device), t.to(self.device)      
                loss = self._train_step(x0, t)
                train_loss.append(loss)

            # Validation phase
            for x0, t in self.val_dataloader:
                x0, t = x0.to(self.device), t.to(self.device)      
                loss = self._val_step(x0, t)
                val_loss.append(loss)

            # Calculate average losses
            train_loss = np.mean(train_loss)
            val_loss = np.mean(val_loss)

            # Save best model if validation loss improved
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint()
            
            # Print progress and save checkpoint
            print(f'Epoch {epoch + 1}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}')
            torch.save(self.model.state_dict(), './ckpt/last_model.pth')
            
            # Generate samples periodically to monitor progress
            if epoch%20 == 0 or epoch== self.epochs-1:
                self.sample()






    
