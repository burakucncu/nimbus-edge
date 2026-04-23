import torch
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os
from model import LightweightUNet

def scale_percentile(img):
    """16-bit uydu verisini görselleştirmek için %2-%98 ölçekleme yapar."""
    # NaN veya sonsuz değerleri temizle
    img = np.nan_to_num(img)
    low, high = np.percentile(img, (2, 98))
    if high <= low: return img # Değerler sabitse olduğu gibi döndür
    return np.clip((img - low) / (high - low), 0, 1)

def show_truth_gallery():
    # 1. Cihaz Ayarı (MacBook Air GPU - MPS)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 2. Modeli Yükle
    model = LightweightUNet(in_channels=4, out_channels=1)
    model_path = "models/nimbus_model_v1.pt"
    
    if not os.path.exists(model_path):
        print(f"Hata: {model_path} bulunamadı! Önce eğitimi tamamla.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    # 3. İçinde Bulut Olan Yamaları Otomatik Bul
    mask_dir = "data/masks"
    cloudy_indices = []
    
    print("Gerçek bulut kütleleri aranıyor (Hassasiyet artırıldı)...")
    for f in sorted(os.listdir(mask_dir)):
        if f.endswith('.tif'):
            with rasterio.open(os.path.join(mask_dir, f)) as m:
                mask_data = m.read(1)
                # Sadece içinde en az 500 piksel bulut olanları al (Gürültüyü ele)
                if np.sum(mask_data > 0) > 500: 
                    idx = f.split('_')[1].split('.')[0]
                    cloudy_indices.append(idx)
        
        if len(cloudy_indices) >= 6:
            break

    if not cloudy_indices:
        print("Uyarı: Hiç bulutlu yama bulunamadı. Eşik değerini kontrol et!")
        return

    # 4. Görselleştirme Ayarları
    fig, axes = plt.subplots(len(cloudy_indices), 3, figsize=(15, 20))
    plt.suptitle("Nimbus-Edge: Cloud Detection Performance (Truth vs Prediction)", fontsize=16)

    for i, idx in enumerate(cloudy_indices):
        img_path = f"data/images/patch_{idx}.tif"
        mask_path = f"data/masks/patch_{idx}.tif"

        # Görüntü ve Maskeyi Oku
        with rasterio.open(img_path) as src:
            image = src.read().astype(np.float32) / 65535.0
        with rasterio.open(mask_path) as src:
            ground_truth = src.read(1)

        # Model Tahmini (Inference)
        input_tensor = torch.tensor(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_tensor)
            prediction = torch.sigmoid(logits).squeeze().cpu().numpy()

        # --- SÜTUN 1: ORIJINAL (RGB) ---
        rgb = np.transpose(image[:3, :, :], (1, 2, 0))
        axes[i, 0].imshow(scale_percentile(rgb))
        axes[i, 0].set_title(f"Original Patch {idx}")
        axes[i, 0].axis('off')

        # --- SÜTUN 2: GROUND TRUTH (LABEL) ---
        axes[i, 1].imshow(ground_truth, cmap='gray')
        axes[i, 1].set_title("Ground Truth (Labels)")
        axes[i, 1].axis('off')

        # --- SÜTUN 3: PREDICTION (MODEL) ---
        # vmin/vmax kullanarak 0-1 arasını netleştiriyoruz
        im = axes[i, 2].imshow(prediction, cmap='magma', vmin=0, vmax=1)
        axes[i, 2].set_title(f"Prediction (Mean: {prediction.mean():.2f})")
        axes[i, 2].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    show_truth_gallery()