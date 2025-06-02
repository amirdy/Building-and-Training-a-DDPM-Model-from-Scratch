import torch
import os 
import numpy as np
from PIL import Image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, images_list, augmentation, T, images_dir):
        self.images_list = images_list
        self.augmentation = augmentation
        self.T = T
        self.images_dir = images_dir

    def __getitem__(self, index):
        image_name = self.images_list[index]
        image_path = os.path.join(self.images_dir, image_name)
        image = Image.open(image_path).convert("RGB") 
        image = np.array(image)
        image = self.augmentation(image=image)['image']
        image = image.float() / 255.0 
        normalized_image = image * 2 - 1
        t = torch.randint(0, self.T, (1,))  

        return normalized_image, t
    
    def __len__(self):
        return len(self.images_list)