import os
import rasterio
from rasterio.windows import Window
import numpy as np

def prepare_golden_dataset():
    # Dosya yolları
    raw_image_path = "data/raw/kennedy_2.tif"
    golden_mask_path = "data/raw/kennedy_2_golden_mask.tif"
    img_dir = "data/images"
    mask_dir = "data/masks"
    patch_size = 256

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    print("Altın Veri (Golden Data) İşleme Hattı Başlatılıyor...")

    if not os.path.exists(raw_image_path) or not os.path.exists(golden_mask_path):
        print("HATA: Raw görüntü veya Altın Maske bulunamadı! Lütfen dosya yollarını kontrol et.")
        return

    with rasterio.open(raw_image_path) as src_img, rasterio.open(golden_mask_path) as src_mask:
        width = src_img.width
        height = src_img.height
        
        profile_img = src_img.profile.copy()
        profile_img.update(width=patch_size, height=patch_size, count=4)
        
        profile_mask = src_mask.profile.copy()
        profile_mask.update(width=patch_size, height=patch_size, count=1, dtype=rasterio.uint8)

        band1 = src_img.read(1)
        valid_patch_mask = (band1 > 0)
        scene_patches = 0

        # Görüntüyü 256x256 piksellik yamalara böl
        for y in range(0, height - patch_size + 1, patch_size):
            for x in range(0, width - patch_size + 1, patch_size):
                
                # NoData (Siyah boşluk) kontrolü
                if np.sum(valid_patch_mask[y:y+patch_size, x:x+patch_size]) < (patch_size * patch_size * 0.8):
                    continue

                window = Window(x, y, patch_size, patch_size)
                
                # İŞTE BÜYÜ BURADA: Artık kural yok, doğrudan senin maskeni okuyoruz!
                img_patch = src_img.read([1, 2, 3, 4], window=window)
                mask_patch = src_mask.read(1, window=window) 

                # Yamaları kaydet
                img_path = os.path.join(img_dir, f"golden_patch_{scene_patches:04d}.tif")
                mask_path = os.path.join(mask_dir, f"golden_patch_{scene_patches:04d}.tif")

                with rasterio.open(img_path, 'w', **profile_img) as dest:
                    dest.write(img_patch)
                    
                with rasterio.open(mask_path, 'w', **profile_mask) as dest:
                    dest.write(mask_patch, 1)

                scene_patches += 1
                
    print(f"🎉 Başarılı! Kendi ellerinle çizdiğin veriden {scene_patches} adet kusursuz yama üretildi.")

if __name__ == "__main__":
    prepare_golden_dataset()