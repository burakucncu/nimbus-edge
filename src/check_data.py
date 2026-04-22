import os
import rasterio
import numpy as np

def check_mask_quality():
    mask_dir = "data/masks"
    if not os.path.exists(mask_dir):
        print("Mask directory not found!")
        return

    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.tif')]
    total_files = len(mask_files)
    cloudy_masks = []

    print(f"Scanning {total_files} masks...")

    for f in mask_files:
        path = os.path.join(mask_dir, f)
        with rasterio.open(path) as src:
            mask = src.read(1)
            # Eğer maske içinde en az bir tane 1.0 (beyaz) piksel varsa
            if np.any(mask > 0):
                cloudy_masks.append(f)

    print("-" * 30)
    print(f"Total Masks Analyzed: {total_files}")
    print(f"Masks with Clouds: {len(cloudy_masks)}")
    print(f"Empty (Black) Masks: {total_files - len(cloudy_masks)}")
    
    if cloudy_masks:
        print(f"\nExample cloudy mask found: {cloudy_masks[0]}")
        print("Success: You have positive samples to train on!")
    else:
        print("\nWarning: ALL masks are empty (black).")
        print("Action: Lower the threshold in preprocess.py and run it again.")
    print("-" * 30)

if __name__ == "__main__":
    check_mask_quality()