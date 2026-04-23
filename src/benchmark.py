import torch
import time
import numpy as np
from model import LightweightUNet

def run_benchmark():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = LightweightUNet(in_channels=4, out_channels=1).to(device).eval()
    
    # 1. Isınma Turu (Cihazı hazırlayalım)
    dummy_input = torch.randn(1, 4, 256, 256).to(device)
    for _ in range(10): _ = model(dummy_input)
    
    # 2. Gerçek Test (100 kare işleyelim)
    iterations = 100
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
            if device.type == "mps": torch.mps.synchronize() # Metal senkronizasyonu
            
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = (total_time / iterations) * 1000 # milisaniye
    fps = 1 / (total_time / iterations)
    
    print("-" * 30)
    print(f"DEVICE: {device}")
    print(f"Total Time for {iterations} patches: {total_time:.4f} sec")
    print(f"Average Inference Time: {avg_time:.2f} ms per patch")
    print(f"Processing Speed: {fps:.2f} FPS")
    print("-" * 30)

if __name__ == "__main__":
    run_benchmark()