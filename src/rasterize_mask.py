import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import numpy as np
import os

def create_golden_mask():
    # Dosya yolları (Kendi klasör yapına göre kontrol et)
    raster_path = "data/raw/kennedy_2.tif"
    vector_path = "data/polygons/kennedy_2_cloud.gpkg" 
    output_mask = "data/raw/kennedy_2_golden_mask.tif"

    print("Geometri ve Referans Uydu verileri yükleniyor...")
    
    if not os.path.exists(vector_path):
        print(f"Hata: {vector_path} bulunamadı! Dosya yolunu kontrol et.")
        return

    # 1. Vektör veriyi (Senin çizdiğin poligonları) oku
    gdf = gpd.read_file(vector_path)

    # 2. Referans uydu görüntüsünü oku (Birebir aynı koordinat ve piksel boyutunu almak için)
    with rasterio.open(raster_path) as src:
        meta = src.meta.copy()
        transform = src.transform
        shape = (src.height, src.width)

        # Maske formatını ayarla (Tek bantlı ve uint8 formatında 0/1 değerleri olacak)
        meta.update(count=1, dtype=rasterio.uint8)

        print("Poligonlar piksellere dönüştürülüyor (Rasterization)...")
        # Geometrileri ve senin girdiğin 'class' (1) değerlerini eşleştir
        shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf['class']))

        # 3. Rasterize işlemi: Çizilmeyen her yer arka plan yani 0 (Siyah) olacak!
        mask = rasterize(
            shapes=shapes,
            out_shape=shape,
            transform=transform,
            fill=0, 
            all_touched=False,
            dtype=np.uint8
        )

        # 4. Kusursuz maskeyi kaydet
        with rasterio.open(output_mask, 'w', **meta) as dest:
            dest.write(mask, 1)

    print(f"🎉 Mükemmel! Altın maske başarıyla oluşturuldu: {output_mask}")

if __name__ == "__main__":
    create_golden_mask()