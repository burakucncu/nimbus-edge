import os
import glob
import rasterio
from rasterio.windows import Window
import numpy as np

def create_dataset(raw_dir="data/raw", img_dir="data/images", mask_dir="data/masks", patch_size=256):
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # raw klasöründeki tüm tif dosyalarını bul
    tif_files = glob.glob(os.path.join(raw_dir, "*.tif"))
    
    if not tif_files:
        print(f"Hata: '{raw_dir}' klasöründe görüntü bulunamadı!")
        return

    print(f"Toplam {len(tif_files)} adet uydu görüntüsü bulundu. Veri madenciliği başlıyor...\n")
    total_patches = 0

    for filepath in tif_files:
        # Dosya adını uzantısız alır (Örn: "orman", "col", "scene_1")
        scene_name = os.path.splitext(os.path.basename(filepath))[0]
        print(f"[{scene_name}] İşleniyor...")

        with rasterio.open(filepath) as src:
            # 4 Bant Güvenlik Kilidi (Daha önce eklediğimiz kurala sadık kalıyoruz)
            if src.count < 4:
                print(f"  -> ATLANDI: 4 bantlı değil ({src.count} bant). Sadece Multispectral veriler alınır.")
                continue

            width = src.width
            height = src.height
            
            profile_img = src.profile.copy()
            profile_img.update(width=patch_size, height=patch_size, count=4)
            
            profile_mask = src.profile.copy()
            profile_mask.update(width=patch_size, height=patch_size, count=1, dtype=rasterio.uint8)

            band1 = src.read(1) # NoData kontrolü için
            scene_patches = 0

            # Görüntüyü 256x256 piksellik yamalara böl
            for y in range(0, height - patch_size + 1, patch_size):
                for x in range(0, width - patch_size + 1, patch_size):
                    window = Window(x, y, patch_size, patch_size)
                    
                    # NoData (Siyah) alan kontrolü: Yamanın %80'inden fazlası siyahsa/boşsa atla!
                    valid_patch = band1[y:y+patch_size, x:x+patch_size]
                    if np.sum(valid_patch > 0) < (patch_size * patch_size * 0.8):
                        continue

                    patch = src.read([1, 2, 3, 4], window=window)
                    
                    # --- OTOMATİK ETİKETLEME (Pseudo-Labeling) ---
                    nir_band = patch[3]
                    max_val = np.max(nir_band)
                    if max_val == 0: continue
                    # Dinamik Eşikleme ile maske üretimi
                    mask = (nir_band > (max_val * 0.45)).astype(np.uint8)
                    # ---------------------------------------------

                    # Çakışmayı önlemek için dosya adına sahne adını ekliyoruz
                    img_path = os.path.join(img_dir, f"{scene_name}_patch_{scene_patches:04d}.tif")
                    mask_path = os.path.join(mask_dir, f"{scene_name}_patch_{scene_patches:04d}.tif")

                    with rasterio.open(img_path, 'w', **profile_img) as dest:
                        dest.write(patch)
                        
                    with rasterio.open(mask_path, 'w', **profile_mask) as dest:
                        dest.write(mask, 1)

                    scene_patches += 1
                    total_patches += 1
                    
        print(f"  -> Başarılı: {scene_patches} adet yama çıkarıldı.\n")

    print("="*40)
    print(f"🎉 GLOBAL VERİ SETİ HAZIR! Toplam {total_patches} adet yama üretildi.")
    print("="*40)

if __name__ == "__main__":
    create_dataset()