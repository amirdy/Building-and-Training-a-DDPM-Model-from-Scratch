import torch
import argparse  
from config import Config
from trainer import Trainer
from models.ddpm import DDPM


def main():
    
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
    model_path = "/ckpt/last_model.pth"
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # Create the trainer
    trainer = Trainer(
        config=config,
        train_dataloader=None,
        val_dataloader=None,
        ddpm=model, 
        device=device
    )
    
    # Select appropriate sampling
    if args.model_type == 'ddpm':
        trainer.sample()
    else:  # ddim
        trainer.ddim_sample()  


if __name__ == "__main__":
    main()