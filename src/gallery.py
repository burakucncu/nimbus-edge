import torch
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os
from model import LightweightUNet

def scale_percentile(img):
    """16-bit veriyi görselleştirmek için %2-%98 ölçekleme yapar."""
    low, high = np.percentile(img, (2, 98))
    return np.clip((img - low) / (high - low), 0, 1)

def show_debug_gallery():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = LightweightUNet(in_channels=4, out_channels=1)
    model.load_state_dict(torch.load("models/nimbus_model_v1.pt", map_location=device))
    model.to(device).eval()

    # Test edilecek yamalar
    patch_indices = ["0100", "0250", "0400", "0550", "0700", "0800"]
    
    # 6 satır (yamalar), 2 sütun (Orijinal vs Tahmin)
    fig, axes = plt.subplots(len(patch_indices), 2, figsize=(10, 20))

    for i, idx in enumerate(patch_indices):
        img_path = f"data/images/patch_{idx}.tif"
        if not os.path.exists(img_path): continue

        with rasterio.open(img_path) as src:
            image = src.read().astype(np.float32) / 65535.0
        
        input_tensor = torch.tensor(image).unsqueeze(0).to(device)
        with torch.no_grad():
            # Model logits döndürdüğü için sigmoid uyguluyoruz
            prediction = torch.sigmoid(model(input_tensor)).squeeze().cpu().numpy()

        # --- SOL SÜTUN: ORIJINAL ---
        rgb = np.transpose(image[:3, :, :], (1, 2, 0))
        rgb_scaled = scale_percentile(rgb)
        axes[i, 0].imshow(rgb_scaled)
        axes[i, 0].set_title(f"Original Patch {idx}")
        axes[i, 0].axis('off')

        # --- SAĞ SÜTUN: TAHMİN (MASK) ---
        axes[i, 1].imshow(prediction, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f"Prediction (Mean: {prediction.mean():.2f})")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    show_debug_gallery()