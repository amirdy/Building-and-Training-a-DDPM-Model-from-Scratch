import os
from torchvision.datasets import CIFAR10

# CIFAR-10 class 'automobile' has label = 1
CAR_LABEL = 1

# Load dataset
dataset = CIFAR10(root='./data', train=True, download=True)
output_dir = 'cifar10'
os.makedirs(output_dir, exist_ok=True)

# Save only car images
count = 0
for idx, (img, label) in enumerate(dataset):
    if label == CAR_LABEL:
        img.save(os.path.join(output_dir, f"{count:05d}_{label}.png"))
        count += 1

