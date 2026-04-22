import torch
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os
from model import LightweightUNet

def scale_percentile(img):
    # 16-bit veriyi %2-%98 aralığında ölçekle
    low, high = np.percentile(img, (2, 98))
    return np.clip((img - low) / (high - low), 0, 1)

def show_truth_gallery():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = LightweightUNet(in_channels=4, out_channels=1)
    model.load_state_dict(torch.load("models/nimbus_model_v1.pt", map_location=device))
    model.to(device).eval()

    # Test edilecek yama numaraları
    patch_indices = ["0100", "0250", "0400", "0550", "0700", "0800"]
    
    # 6 satır, 3 sütun: [Orijinal] [Gerçek Maske] [Model Tahmini]
    fig, axes = plt.subplots(len(patch_indices), 3, figsize=(15, 20))

    for i, idx in enumerate(patch_indices):
        img_path = f"data/images/patch_{idx}.tif"
        mask_path = f"data/masks/patch_{idx}.tif"
        if not os.path.exists(img_path): continue

        # Görüntüyü oku
        with rasterio.open(img_path) as src:
            image = src.read().astype(np.float32) / 65535.0
        
        # Gerçek maskeyi oku (Ground Truth)
        with rasterio.open(mask_path) as src:
            ground_truth = src.read(1)

        # Model tahmini
        input_tensor = torch.tensor(image).unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = torch.sigmoid(model(input_tensor)).squeeze().cpu().numpy()

        # 1. Sütun: Orijinal
        rgb = np.transpose(image[:3, :, :], (1, 2, 0))
        axes[i, 0].imshow(scale_percentile(rgb))
        axes[i, 0].set_title(f"Original {idx}")
        axes[i, 0].axis('off')

        # 2. Sütun: Gerçek Maske (Eğitimde kullanılan)
        axes[i, 1].imshow(ground_truth, cmap='gray')
        axes[i, 1].set_title("Ground Truth (Label)")
        axes[i, 1].axis('off')

        # 3. Sütun: Modelin Tahmini
        axes[i, 2].imshow(prediction, cmap='magma', vmin=0, vmax=1)
        axes[i, 2].set_title(f"Prediction (Mean: {prediction.mean():.2f})")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    show_truth_gallery()