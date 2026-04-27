import rasterio
import numpy as np
import os

def run_diagnostics():
    print("="*40)
    print("🕵🏻‍♂️ NİMBUS-EDGE DEDEKTİF RAPORU 🕵🏻‍♂️")
    print("="*40)
    
    img_dir = "data/images"
    mask_dir = "data/masks"
    
    if not os.path.exists(img_dir) or len(os.listdir(img_dir)) == 0:
        print("❌ HATA: Veri klasörleri boş!")
        return

    # 1. Karanlık Oda Testi (Maksimum Piksel Değeri)
    sample_img = "data/raw/gazze.tif"
    with rasterio.open(sample_img) as src:
        img_data = src.read()
        max_val = img_data.max()
        print(f"Görüntü Veri Tipi: {src.dtypes[0]}")
        print(f"Görüntüdeki EN PARLAK piksel: {max_val}")
        if max_val < 15000 and src.dtypes[0] == 'uint16':
            print("⚠️ TEŞHİS: Görüntü zifiri karanlık (Dark Image)! 65535'e bölmek modeli kör ediyor.")

    # 2. 0 Bulut Testi (Maskelerde hiç beyaz piksel var mı?)
    total_cloud_pixels = 0
    for f in os.listdir(mask_dir):
        if f.endswith('.tif'):
            with rasterio.open(os.path.join(mask_dir, f)) as m:
                total_cloud_pixels += np.sum(m.read(1))
                
    print("-" * 40)
    print(f"Eğitimdeki Toplam Bulut Pikseli: {total_cloud_pixels:,}")
    if total_cloud_pixels == 0:
        print("🚨 KRİTİK TEŞHİS: Maskelerin hepsi boş! Model sadece okyanusla eğitilmiş.")
    else:
        print("✅ Maskelerde bulut var. Çizimler koda ulaşmış.")
    print("="*40)

if __name__ == "__main__":
    run_diagnostics()