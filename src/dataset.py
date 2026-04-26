import os
import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np

class NimbusCloudDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        # Sadece tif dosyalarını okumayı garanti altına alalım
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.tif')])
        self.masks = sorted([f for f in os.listdir(mask_dir) if f.endswith('.tif')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        with rasterio.open(img_path) as src:
            img = src.read([1, 2, 3, 4]).astype(np.float32)
            
            # KRİTİK DÜZELTME: Eğitim verisini Test verisiyle AYNI DİLE (0-1) çeviriyoruz!
            if src.dtypes[0] == 'uint8':
                img = img / 255.0
            else:
                img = img / 10000.0
                img = np.clip(img, 0, 1)

        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.float32)
            # Maskenin kesinlikle 0 veya 1 olduğundan emin oluyoruz
            mask = (mask > 0).astype(np.float32) 

        return torch.tensor(img), torch.tensor(mask).unsqueeze(0)