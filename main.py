import torch
from config import Config
from dataset.data_module import DataModule
from trainer import Trainer
from models.ddpm import DDPM
import time
import os
import random

def main():

    # Load and prepare the CIFAR-10 (cars) dataset
    images_dir = './cifar10_cars'
    all_images = os.listdir(images_dir)
    all_images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png'))]

    # Set random seed for reproducibility
    random.seed(42)

    # Split dataset into training (85%) and validation (15%) sets
    random.shuffle(all_images)
    split_idx = int(0.85 * len(all_images))
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    print(f"Training images: {len(train_images)}, Validation images: {len(val_images)}")

    # Configure device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on {device}...')

    # Initialize model configuration
    config = Config()

    # Set up data loaders for training and validation
    dm = DataModule(config, train_images, val_images, images_dir)
    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()

    # Initialize the DDPM model
    model = DDPM(config)
    model.to(device)  # Move model to appropriate device

    # Initialize the trainer with model and data loaders
    trainer = Trainer(
        config = config,
        train_dataloader = train_dataloader,
        val_dataloader = val_dataloader,
        ddpm = model,
        device = device
    )

    # Start training process
    print('Start training')
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    training_time_minutes = (end_time - start_time)/60
    print(f'\n Training completed in {training_time_minutes:.2f} minutes.')

if __name__ == "__main__":
    main()


