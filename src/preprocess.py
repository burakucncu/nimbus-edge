import os
import numpy as np
import rasterio
from rasterio.windows import Window

def create_patches(input_path, output_img_dir, output_mask_dir, patch_size=256):
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    print(f"Processing scene: {input_path}")
    
    with rasterio.open(input_path) as src:
        width, height = src.width, src.height
        patch_count = 0
        
        for y in range(0, height - patch_size + 1, patch_size):
            for x in range(0, width - patch_size + 1, patch_size):
                window = Window(x, y, patch_size, patch_size)
                img_patch = src.read(window=window)
                
                if np.sum(img_patch == 0) > (patch_size * patch_size * 0.5):
                    continue
                
                # --- DYNAMIC DEBUGGING & NORMALIZATION ---
                # Normalize 16-bit
                normalized_patch = img_patch.astype(np.float32) / 65535.0
                max_val = np.max(normalized_patch)
                
                # Eğer yama çok karanlıksa (neredeyse tamamen siyahsa) işlem yapma
                if max_val < 0.02: 
                    mask_patch = np.zeros((patch_size, patch_size), dtype=np.float32)
                else:
                    threshold = max_val * 0.45
                    mean_spatial = np.mean(normalized_patch, axis=0)
                    mask_patch = np.where(mean_spatial > threshold, 1.0, 0.0).astype(np.float32)
                
                # --- SAVE ---
                img_out_path = os.path.join(output_img_dir, f"patch_{patch_count:04d}.tif")
                mask_out_path = os.path.join(output_mask_dir, f"patch_{patch_count:04d}.tif")

                profile = src.profile.copy()
                profile.update({"height": patch_size, "width": patch_size, "transform": src.window_transform(window)})
                
                with rasterio.open(img_out_path, 'w', **profile) as dest:
                    dest.write(img_patch)
                
                mask_profile = profile.copy()
                mask_profile.update({"count": 1})
                with rasterio.open(mask_out_path, 'w', **mask_profile) as dest:
                    dest.write(mask_patch, 1)

                patch_count += 1

    print(f"Extraction complete! Created {patch_count} patches.")

if __name__ == "__main__":
    create_patches("data/raw/scene_1.tif", "data/images", "data/masks")