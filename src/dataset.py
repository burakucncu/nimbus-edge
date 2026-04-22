import os
import rasterio
import numpy as np
import torch
from torch.utils.data import Dataset

class NimbusCloudDataset(Dataset):
    """
    Custom PyTorch Dataset for loading multi-band satellite imagery and cloud masks.
    """
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # Retrieve all TIFF files in the directory
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.tif')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        # Read multi-band image and normalize to [0.0, 1.0]
        with rasterio.open(img_path) as src:
            image = src.read().astype(np.float32) / 65535.0

        # Read binary mask (1: Cloud, 0: Clear)
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.float32)

        # Apply transformations (data augmentation) if any
        if self.transform:
            # Albumentations requires HWC format, Rasterio gives CHW
            image = np.transpose(image, (1, 2, 0))
            augmented = self.transform(image=image, mask=mask)
            image = np.transpose(augmented['image'], (2, 0, 1))
            mask = augmented['mask']

        return torch.tensor(image), torch.tensor(mask).unsqueeze(0)