import torch
import rasterio
from rasterio.windows import Window
import numpy as np
import os
import time
from PIL import Image
from model import LightweightUNet

def create_hann_window(patch_size):
    """
    Yamanın merkezine 1.0 (tam güven), kenarlarına 0.0 (sıfır güven) veren 
    2 boyutlu bir ağırlık matrisi (Hanning Window) oluşturur.
    """
    window_1d = np.hanning(patch_size)
    window_2d = np.outer(window_1d, window_1d).astype(np.float32)
    return window_2d

def process_large_scene(image_path, model_path="models/nimbus_model_v4.pt"):
    if not os.path.exists(image_path):
        print(f"Hata: {image_path} bulunamadı!")
        return

    # --- 4 BANT GÜVENLİK KİLİDİ ---
    with rasterio.open(image_path) as temp_src:
        if temp_src.count < 4:
            print(f"❌ HATA: Model 4 bant (RGB+NIR) bekliyor, ancak '{image_path}' sadece {temp_src.count} banda sahip.")
            print("Lütfen gerçek çok bantlı (Multispectral) bir uydu görüntüsü kullanın.")
            return

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Modelimiz saf 4 bant bekliyor (RGB + NIR)
    model = LightweightUNet(in_channels=4, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    print(f"[{device.type.upper()}] Model yüklendi. Saf 4-Bant Analizi (Ağırlıklı Harmanlama ile) başlıyor...")
    start_time = time.time()

    patch_size = 256
    stride = 128  # %50 Örtüşme
    inference_threshold = 0.25 

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.join("output", base_name)
    os.makedirs(out_dir, exist_ok=True)

    weight_patch = create_hann_window(patch_size)

    with rasterio.open(image_path) as src:
        width = src.width
        height = src.height
        profile = src.profile

        # Otomatik Bit Derinliği
        if src.dtypes[0] == 'uint8':
            norm_factor = 255.0
        else:
            norm_factor = 65535.0

        print("Geçerli alan (NoData) tespiti yapılıyor...")
        band1 = src.read(1)
        valid_pixels_mask = (band1 > 0)
        total_valid_pixels = np.sum(valid_pixels_mask)
        print(f"Toplam Çerçeve: {width*height:,} piksel | Gerçek Görüntü Alanı: {total_valid_pixels:,} piksel")

        full_probs_sum = np.zeros((height, width), dtype=np.float32)
        weight_sum = np.zeros((height, width), dtype=np.float32)
        
        y_steps = list(range(0, height - patch_size + 1, stride))
        if y_steps[-1] != height - patch_size: y_steps.append(height - patch_size)
        
        x_steps = list(range(0, width - patch_size + 1, stride))
        if x_steps[-1] != width - patch_size: x_steps.append(width - patch_size)

        total_patches = len(y_steps) * len(x_steps)
        processed = 0

        with torch.no_grad():
            for y in y_steps:
                for x in x_steps:
                    window = Window(x, y, patch_size, patch_size)
                    
                    # Sadece ilk 4 bandı okuyoruz (RGB + NIR)
                    patch = src.read([1, 2, 3, 4], window=window, boundless=True, fill_value=0).astype(np.float32) / norm_factor

                    input_tensor = torch.tensor(patch).unsqueeze(0).to(device)
                    logits = model(input_tensor)
                    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
                    
                    h_valid = min(patch_size, height - y)
                    w_valid = min(patch_size, width - x)
                    
                    # AĞIRLIKLI BİRLEŞTİRME (Weighted Blending)
                    full_probs_sum[y:y+h_valid, x:x+w_valid] += probs[:h_valid, :w_valid] * weight_patch[:h_valid, :w_valid]
                    weight_sum[y:y+h_valid, x:x+w_valid] += weight_patch[:h_valid, :w_valid]
                    
                    processed += 1
                    if processed % 500 == 0:
                        print(f"İşlenen yama: {processed}/{total_patches}")

    weight_sum[weight_sum == 0] = 1.0
    final_probs = full_probs_sum / weight_sum

    full_mask = (final_probs > inference_threshold).astype(np.uint8)
    full_mask[~valid_pixels_mask] = 0

    cloudy_pixels = np.sum(full_mask)
    cloud_percentage = (cloudy_pixels / total_valid_pixels) * 100

    end_time = time.time()

    print("\n" + "="*40)
    print("🌍 KUSURSUZ SAHNE ANALİZ RAPORU 🌍")
    print("="*40)
    print(f"Toplam Süre        : {end_time - start_time:.2f} saniye")
    print(f"GERÇEK BULUT ORANI : %{cloud_percentage:.2f}")
    print("="*40)

    # Orijinal profil özelliklerini kopyala ama maske için 1 bant yap
    out_profile = profile.copy()
    out_profile.update(count=1, dtype=rasterio.uint8, compress='lzw')
    
    out_tif_path = os.path.join(out_dir, f"{base_name}_mask.tif")
    with rasterio.open(out_tif_path, 'w', **out_profile) as dest:
        dest.write(full_mask, 1)
        
    out_jpg_path = os.path.join(out_dir, f"{base_name}_mask.jpg")
    jpeg_mask = (full_mask * 255).astype(np.uint8)
    Image.fromarray(jpeg_mask).save(out_jpg_path, quality=90)
    
    print(f"Çıktılar '{out_dir}/' klasörüne başarıyla kaydedildi!")

if __name__ == "__main__":
    # Test için orijinal, saf 4 bantlı uydu görüntümüze geri döndük
    large_scene = "data/raw/test_1.tif" 
    process_large_scene(large_scene)