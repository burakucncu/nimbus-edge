import torch
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from model import LightweightUNet

def predict():
    # 1. Setup Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 2. Load Model
    model = LightweightUNet(in_channels=4, out_channels=1)
    model.load_state_dict(torch.load("models/nimbus_model_v1.pt"))
    model.to(device)
    model.eval()

    # 3. Load a Sample Patch (Let's pick patch 0)
    test_patch_path = "data/images/patch_0400.tif"
    with rasterio.open(test_patch_path) as src:
        image = src.read().astype(np.float32) / 65535.0
        
    # Prepare tensor (C, H, W) -> (1, C, H, W)
    input_tensor = torch.tensor(image).unsqueeze(0).to(device)

    # 4. Inference
    with torch.no_grad():
        logits = model(input_tensor)
        # Tahminde olasılığa çevirmek için sigmoid'i buraya ekliyoruz
        prediction = torch.sigmoid(logits) 
        prediction = prediction.squeeze().cpu().numpy()

    # 5. Visualize Results
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Otomatik parlaklık ayarı (Percentile Scaling)
    def scale_img(img):
        low, high = np.percentile(img, (2, 98))
        return np.clip((img - low) / (high - low), 0, 1)

    # RGB bandlarını (0,1,2) al ve ölçekle
    rgb_image = np.transpose(image[:3, :, :], (1, 2, 0))
    rgb_image = scale_img(rgb_image)
    
    ax[0].imshow(rgb_image)
    ax[0].set_title("Original Satellite Patch (Auto-Scaled)")
    ax[0].axis('off')
    
    # Olasılık haritasını göster (Threshold uygulayarak: > 0.5 olanlar bulut)
    cloud_mask = (prediction > 0.5).astype(np.float32)
    ax[1].imshow(cloud_mask, cmap='Blues')
    ax[1].set_title("Detected Cloud Mask (Threshold: 0.5)")
    ax[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Show RGB (Assuming first 3 bands are R, G, B)
    rgb_image = np.transpose(image[:3, :, :], (1, 2, 0))
    # Clip for visualization
    rgb_image = np.clip(rgb_image * 5, 0, 1) 
    
    ax[0].imshow(rgb_image)
    ax[0].set_title("Original Satellite Patch")
    
    ax[1].imshow(prediction, cmap='gray')
    ax[1].set_title("Cloud Probability Map")
    
    plt.show()
    print("Inference complete. Check the plot to see results!")

if __name__ == "__main__":
    predict()