
import torch
from config import Config
from dataset.data_module import DataModule
from models.unet import UNet
from trainer import Trainer
from models.ddpm import DDPM
import time
import os
import random

def main():
    """ Main function to set up and train the GPT model. """

    # Load dataset
    images_dir = './cifar10_cars_64x64'
    all_images = os.listdir(images_dir)
    all_images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    random.seed(42)  # For reproducibility

    # Shuffle and split
    random.shuffle(all_images)
    split_idx = int(0.85 * len(all_images))  # 80% train, 20% val

    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    print(len(train_images), len(val_images))
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on {device}...')

    # Initialize configurations
    config = Config()
    # print(type(config.H))


    # Initialize Data Module and Data Loaders
    dm = DataModule(config, train_images, val_images, images_dir)
    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()

    # Create the GPT model
    model = DDPM(config)
    model.to(device) # Move the model to the device


    # Create the trainer
    trainer = Trainer(
        config = config,
        train_dataloader = train_dataloader,
        val_dataloader = val_dataloader,
        ddpm = model,
        device = device
    )
    print('Start training')
    # Start training
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    training_time_minutes = (end_time - start_time)/60
    print(f'\n Training completed in {training_time_minutes:.2f} minutes.')

if __name__ == "__main__":
    main()


