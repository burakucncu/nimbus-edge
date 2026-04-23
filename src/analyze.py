import torch
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os
from model import LightweightUNet

def scale_percentile(img):
    """16-bit görüntüyü insan gözüne uygun hale getirir."""
    img = np.nan_to_num(img)
    low, high = np.percentile(img, (2, 98))
    if high <= low: return img
    return np.clip((img - low) / (high - low), 0, 1)

def analyze_cloud_cover(image_path, model_path="models/nimbus_model_v1.pt"):
    """Tek bir uydu görüntüsünü alır, maske çıkarır ve bulut oranını hesaplar."""
    
    if not os.path.exists(image_path):
        print(f"Hata: {image_path} bulunamadı!")
        return

    # 1. Cihaz ve Model Hazırlığı
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = LightweightUNet(in_channels=4, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    # 2. Görüntüyü Oku ve Normalize Et
    with rasterio.open(image_path) as src:
        image = src.read().astype(np.float32) / 65535.0
        profile = src.profile # Kaydetmek için meta veriyi sakla

    # 3. Model Tahmini (Inference)
    input_tensor = torch.tensor(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        # Sigmoid ile 0-1 arası olasılıklara çevir
        probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()

    # 4. Karar Aşaması (Eşikleme)
    # Model %50'den eminse oraya bulut (1.0) diyoruz
    cloud_mask = (probabilities > 0.5).astype(np.uint8)

    # 5. İSTATİSTİK: Bulut Yüzdesini Hesapla
    total_pixels = cloud_mask.size
    cloudy_pixels = np.sum(cloud_mask)
    cloud_percentage = (cloudy_pixels / total_pixels) * 100

    print("\n" + "="*40)
    print("☁️ NİMBUS-EDGE ANALİZ RAPORU ☁️")
    print("="*40)
    print(f"İncelenen Dosya  : {os.path.basename(image_path)}")
    print(f"Toplam Piksel    : {total_pixels:,}")
    print(f"Bulutlu Piksel   : {cloudy_pixels:,}")
    print(f"Bulutluluk Oranı : %{cloud_percentage:.2f}")
    print("="*40)

    # 6. Sonuçları Kaydet (Opsiyonel - Uyduda diske yazmak istersen)
    os.makedirs("output", exist_ok=True)
    out_mask_path = f"output/mask_{os.path.basename(image_path)}"
    
    profile.update(count=1, dtype=rasterio.uint8)
    with rasterio.open(out_mask_path, 'w', **profile) as dest:
        dest.write(cloud_mask, 1)
    print(f"Maske kaydedildi: {out_mask_path}")

    # 7. Görselleştirme (Ekranda Göster)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Orijinal Görüntü
    rgb = np.transpose(image[:3, :, :], (1, 2, 0))
    axes[0].imshow(scale_percentile(rgb))
    axes[0].set_title("Orijinal Görüntü")
    axes[0].axis("off")
    
    # Modelin Olasılık Haritası (Isı Haritası)
    axes[1].imshow(probabilities, cmap="magma", vmin=0, vmax=1)
    axes[1].set_title("Yapay Zeka Olasılık Haritası")
    axes[1].axis("off")
    
    # Nihai Maske (Siyah/Beyaz) ve Oran
    axes[2].imshow(cloud_mask, cmap="gray")
    axes[2].set_title(f"Nihai Maske (%{cloud_percentage:.1f} Bulutlu)")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test etmek istediğin herhangi bir dosyanın yolunu buraya yaz
    test_image = "data/images/patch_0309.tif"
    analyze_cloud_cover(test_image)