import os
import numpy as np
import rasterio
from rasterio.windows import Window
import cv2

def create_patches(input_path, output_img_dir, output_mask_dir, patch_size=256):
    """
    Reads a large satellite image, generates a basic cloud mask using thresholding,
    and slices both into smaller patches for U-Net training.
    """
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    print(f"Processing scene: {input_path}")
    
    with rasterio.open(input_path) as src:
        width = src.width
        height = src.height
        
        patch_count = 0
        
        # Loop through the image in grid steps
        for y in range(0, height - patch_size + 1, patch_size):
            for x in range(0, width - patch_size + 1, patch_size):
                window = Window(x, y, patch_size, patch_size)
                
                # Read the patch
                img_patch = src.read(window=window)
                
                # Handle NoData (skip patches that are mostly empty borders)
                if np.sum(img_patch == 0) > (patch_size * patch_size * 0.5):
                    continue
                
                # --- AUTO-MASK GENERATION (Baseline Thresholding) ---
                # Calculate mean brightness across bands (assuming CHW format)
                mean_brightness = np.mean(img_patch, axis=0)
                
                # Simple rule: If pixel brightness is high, it's a cloud (1), else background (0)
                # Adjust threshold (e.g., 180 for 8-bit, or 0.6 for normalized) based on your image
                threshold = np.max(img_patch) * 0.6 
                mask_patch = np.where(mean_brightness > 0.4, 1.0, 0.0).astype(np.float32)
                
                # ----------------------------------------------------

                # Save Image Patch
                img_out_path = os.path.join(output_img_dir, f"patch_{patch_count:04d}.tif")
                # Save Mask Patch
                mask_out_path = os.path.join(output_mask_dir, f"patch_{patch_count:04d}.tif")

                # Write Image
                profile = src.profile.copy()
                profile.update({
                    "height": patch_size,
                    "width": patch_size,
                    "transform": src.window_transform(window)
                })
                
                with rasterio.open(img_out_path, 'w', **profile) as dest:
                    dest.write(img_patch)
                
                # Write Mask (Single band)
                mask_profile = profile.copy()
                mask_profile.update({"count": 1})
                with rasterio.open(mask_out_path, 'w', **mask_profile) as dest:
                    dest.write(mask_patch, 1)

                patch_count += 1

        print(f"Extraction complete! Created {patch_count} patches.")

if __name__ == "__main__":
    RAW_IMAGE = "data/raw/scene_1.tif" 
    IMG_DIR = "data/images"
    MASK_DIR = "data/masks"
    
    create_patches(RAW_IMAGE, IMG_DIR, MASK_DIR)
