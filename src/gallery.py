import torch
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os
from model import LightweightUNet

def show_gallery():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = LightweightUNet(in_channels=4, out_channels=1)
    model.load_state_dict(torch.load("models/nimbus_model_v1.pt", map_location=device))
    model.to(device).eval()

    # Test edilecek 6 farklı yama numarası seçelim
    patch_indices = ["0100", "0250", "0400", "0550", "0700", "0800"]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, idx in enumerate(patch_indices):
        img_path = f"data/images/patch_{idx}.tif"
        if not os.path.exists(img_path): continue

        with rasterio.open(img_path) as src:
            image = src.read().astype(np.float32) / 65535.0
        
        input_tensor = torch.tensor(image).unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = torch.sigmoid(model(input_tensor)).squeeze().cpu().numpy()

        # Görselleştirme: RGB oluştur (Scaling ile)
        rgb = np.transpose(image[:3, :, :], (1, 2, 0))
        rgb = np.clip(rgb * 10, 0, 1) # Biraz daha parlak yapalım

        # Tahmini (Maskeyi) orijinalin üzerine şeffaf mavi olarak ekle
        mask = (prediction > 0.5).astype(np.float32)
        
        axes[i].imshow(rgb)
        axes[i].imshow(mask, cmap='Blues', alpha=0.4) # Bulutları mavi katman olarak göster
        axes[i].set_title(f"Patch {idx}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    show_gallery()