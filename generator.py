import torch
import argparse  # Add this import
from config import Config
from dataset.data_module import DataModule
from models.unet import UNet
from trainer import Trainer
from models.ddpm import DDPM


def main():
    """ Main function to set up and train the GPT model. """
    
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('model_type', choices=['ddpm', 'ddim'], 
                       help='Type of model to use (ddpm or ddim)')
    args = parser.parse_args()

    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize configurations
    config = Config()

    # Create the model  
    model = DDPM(config)
    model_path = "/content/last_model_sob.pth"
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Create the trainer
    trainer = Trainer(
        config=config,
        train_dataloader=None,
        val_dataloader=None,
        ddpm=model,  # Note: You might want to rename this parameter
        device=device
    )
    
    # Select appropriate sampling
    if args.model_type == 'ddpm':
        trainer.sample()
    else:  # ddim
        trainer.ddim_sample()  

    trainer.sample()

if __name__ == "__main__":
    main()