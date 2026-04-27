import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import numpy as np
import os

def create_golden_mask():
    raster_path = "data/raw/gazze.tif"
    vector_path = "data/polygons/gazze_cloud.gpkg" 
    output_mask = "data/raw/gazze_golden_mask.tif"

    print("Geometri ve Referans Uydu verileri yükleniyor...")
    
    if not os.path.exists(vector_path):
        print(f"❌ HATA: {vector_path} bulunamadı! Dosyanın doğru klasörde olduğundan emin ol.")
        return

    # Vektör veriyi oku
    gdf = gpd.read_file(vector_path)

    # Referans uydu görüntüsünü oku
    with rasterio.open(raster_path) as src:
        
        # 🚨 İŞTE HAYAT KURTARAN O SATIR: Koordinat Sistemlerini (CRS) havada eşitliyoruz! 🚨
        if gdf.crs != src.crs:
            print(f"⚠️ CRS Uyuşmazlığı Tespit Edildi!")
            print(f"🔄 Vektör ({gdf.crs}) -> Raster ({src.crs}) formatına dönüştürülüyor...")
            gdf = gdf.to_crs(src.crs)

        meta = src.meta.copy()
        transform = src.transform
        shape = (src.height, src.width)

        # Maske formatını ayarla
        meta.update(count=1, dtype=rasterio.uint8)

        print("Poligonlar piksellere dönüştürülüyor (Rasterization)...")
        
        shapes = ((geom, 1) for geom in gdf.geometry)

        mask = rasterize(
            shapes=shapes,
            out_shape=shape,
            transform=transform,
            fill=0, 
            all_touched=False,
            dtype=np.uint8
        )

        # Maskeyi kaydet
        with rasterio.open(output_mask, 'w', **meta) as dest:
            dest.write(mask, 1)

    print(f"🎉 Mükemmel! Gazze altın maskesi başarıyla oluşturuldu: {output_mask}")

if __name__ == "__main__":
    create_golden_mask()