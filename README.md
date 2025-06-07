# Building and Training a DDPM Model from Scratch

A PyTorch implementation of Denoising Diffusion Probabilistic Models (DDPM) from scratch, focusing on image generation. This project provides a clean, modular, and well-documented implementation of DDPM with both DDPM and DDIM sampling methods.

## Features

- 🎨 Custom implementation of DDPM with a UNet backbone
- 🔄 Support for both DDPM and DDIM sampling methods
- 📊 Comprehensive training and validation pipelines
- 💾 Automatic checkpointing and model saving
- 🎯 Configurable noise scheduling and diffusion process
- 🚀 Efficient implementation with PyTorch
- 📈 Training progress visualization
- 🎮 Interactive sampling and generation

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- requirements.txt file (contains all necessary Python package dependencies)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/DDPM-from-Scratch.git
   cd DDPM-from-Scratch
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download and prepare the dataset:
   ```bash
   # Download CIFAR-10 dataset
   python -c "from torchvision.datasets import CIFAR10; CIFAR10(root='./data', download=True)"
   
   # Extract car images
   python extract_cifar10_cars.py
   ```

## Project Structure

```
DDPM-from-Scratch/
├── models/
│   ├── unet.py           # UNet architecture implementation
│   ├── noise_scheduler.py # Noise scheduling logic
│   ├── blocks.py         # Reusable model components
│   ├── linear_attention.py # Linear attention implementation
│   └── ddpm.py          # DDPM model wrapper
├── dataset/
│   ├── dataset.py       # Dataset class implementation
│   └── data_module.py   # Data loading and preprocessing
├── assets/              # Generated samples and visualizations
├── ckpt/               # Model checkpoints
│   ├── best_model.pth  # Best model based on validation loss
│   └── last_model.pth  # Latest model checkpoint
├── config.py           # Configuration parameters
├── main.py            # Training script
├── generator.py       # Generation script
└── trainer.py         # Training logic
```

## Model Architecture

The implementation uses a UNet-based architecture with:
- Residual blocks
- Linear attention mechanisms (to reduce both time and space complexity)
- Configurable channel multipliers
- Time embedding (sinusoidal positional embedding)
- Skip connections

## Dataset

The model is trained on the CIFAR-10 car class dataset:
- Training set: 4,250 images
- Validation set: 750 images
- Image size: 32x32 pixels
- Channels: 3 (RGB)

## Usage

### Training

To train the DDPM model with default configurations:
```bash
python main.py
```

The training configuration can be modified in `config.py`:
- Number of epochs
- Learning rate
- Batch size
- Model architecture parameters
- Noise scheduling parameters

### Generation

#### DDPM Sampling
To generate samples using the standard DDPM sampling method:
```bash
python generate.py ddpm
```
Generated images will be saved in the `assets/` folder.

#### DDIM Sampling
The DDIM sampling method offers faster generation with fewer steps:
- Uses 50 steps instead of 1000
- Approximately 20x faster than DDPM
- Maintains comparable quality

To generate samples using DDIM:
```bash
python generate.py ddim
```

### DDIM Generated Samples
The following are samples generated using the DDIM method, which produces similar quality images in fewer steps:

<div align="center">
<img src="assets/ddim_1.png" alt="DDIM Sample" >
<img src="assets/ddim_2.png" alt="DDIM Sample" >
<img src="assets/ddim_3.png" alt="DDIM Sample" >
<img src="assets/ddim_4.png" alt="DDIM Sample" >
</div>

> **Note**: Before generating samples, make sure you have a trained model checkpoint. The default path is set to "ckpt/last_model.pth" in the generator.py script. You can modify this path in the script if your checkpoint is stored elsewhere.

## Training Results

The model was trained for 2000 epochs on an A40 GPU (approximately 8.5 hours):
- Final training loss: ~0.04
- Final validation loss: ~0.04

Training loss plot:
<img src="assets/loss_plot.png" alt="Training Loss Plot" width=500>

## Generated Samples

<div align="center">

<img src="assets/ddpm_1.png" alt="Generated Sample"  >
<img src="assets/ddpm_2.png" alt="Generated Sample"  >
<img src="assets/ddpm_3.png" alt="Generated Sample" >
<img src="assets/ddpm_4.png" alt="Generated Sample"  >
<img src="assets/ddpm_5.png" alt="Generated Sample"  >
<img src="assets/ddpm_6.png" alt="Generated Sample" >
<img src="assets/ddpm_7.png" alt="Generated Sample"  >
<img src="assets/ddpm_8.png" alt="Generated Sample"  >
<img src="assets/ddpm_9.png" alt="Generated Sample"  >
<img src="assets/ddpm_10.png" alt="Generated Sample"  >
<img src="assets/ddpm_11.png" alt="Generated Sample"  >
<img src="assets/ddpm_12.png" alt="Generated Sample"  >
<img src="assets/ddpm_14.png" alt="Generated Sample"  >

</div>

### Diffusion Process Visualization
The following GIFs show the gradual denoising process from random noise to the final generated image:

<div align="center">

<img src="assets/sample1.gif" alt="Generated Sample" width=70>
<img src="assets/sample2.gif" alt="Generated Sample" width=70>
<img src="assets/sample3.gif" alt="Generated Sample" width=70>
<img src="assets/sample4.gif" alt="Generated Sample" width=70>
<img src="assets/sample5.gif" alt="Generated Sample" width=70>
<img src="assets/sample6.gif" alt="Generated Sample" width=70>
<img src="assets/sample7.gif" alt="Generated Sample" width=70>
<img src="assets/sample8.gif" alt="Generated Sample" width=70>
<img src="assets/sample9.gif" alt="Generated Sample" width=70>
<img src="assets/sample10.gif" alt="Generated Sample" width=70>
<img src="assets/sample11.gif" alt="Generated Sample" width=70>
<img src="assets/sample12.gif" alt="Generated Sample" width=70>

</div>

## Image Quality Variations

The generated images show varying levels of quality and noise. This is normal and expected in diffusion models.

**Quality Improvement Tips**:
- Train for more epochs
- Increase model capacity
- Use larger training dataset
- Try different noise schedules
- Experiment with DDIM sampling parameters

## Technical Details

### Configuration Parameters
- Input channels: 3 (RGB)
- Image size: 32x32
- Number of timesteps: 1000
- Beta schedule: Linear from 1e-4 to 0.02
- Learning rate: 1e-4
- Batch size: 64
- Base channels: 64
- Channel multipliers: (1, 2, 4, 8, 16)
- Number of attention heads: 8

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original DDPM paper: [Ho et al., 2020](https://arxiv.org/abs/2006.11239)
- DDIM paper: [Song et al., 2020](https://arxiv.org/abs/2010.02502)
