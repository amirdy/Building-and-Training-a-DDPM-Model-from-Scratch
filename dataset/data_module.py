from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset.dataset import Dataset

class DataModule:

    def __init__(self, config, train_images, val_images, images_dir):
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        
        self.train_augmentation_ = DataModule.train_augmentation(config.H)
        self.inference_augmentation_ = DataModule.inference_augmentation(config.H)

        self.train_dataset = Dataset(train_images, self.train_augmentation_, config.num_timesteps, images_dir)
        self.val_dataset = Dataset(val_images, self.inference_augmentation_, config.num_timesteps, images_dir)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.num_workers)
    
    @staticmethod
    def train_augmentation(H):
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.Sharpen(alpha=(0.5, 0.75), lightness=(0.5, 1.0), p=0.2),
                A.Resize(
                    height=H,
                    width=H
                ),
                # A.InvertImg(p=0.2),
                ToTensorV2()
            ],
            p=1,
        )

    @staticmethod
    def inference_augmentation(H):
        return A.Compose(
            [
                A.Resize(height=H, width=H),
                ToTensorV2()
            ],
            p=1,
        )