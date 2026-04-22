import torch
from model import LightweightUNet

def run_sanity_check():
    print("Starting sanity check for Lightweight U-Net...")
    
    # Uydu görüntüsü parametreleri (Batch Size, Channels, Height, Width)
    # in_channels=4 (örneğin: Kırmızı, Yeşil, Mavi, NIR)
    batch_size = 2
    in_channels = 4 
    height = 256
    width = 256

    # Rastgele sayılardan oluşan, uydu görüntüsünü taklit eden sahte bir tensör oluşturuyoruz
    dummy_input = torch.randn(batch_size, in_channels, height, width)
    print(f"-> Input shape provided to model: {dummy_input.shape}")

    # Modeli başlat
    model = LightweightUNet(in_channels=in_channels, out_channels=1)

    try:
        # İleri besleme (Forward pass) testi
        output = model(dummy_input)
        print(f"-> Output shape from model: {output.shape}")
        
        # Beklenen çıktı boyutu ile modelin ürettiği boyutu karşılaştır
        expected_shape = (batch_size, 1, height, width)
        assert output.shape == expected_shape, f"Mismatch! Expected {expected_shape}, got {output.shape}"
        
        print("\n[SUCCESS] The model architecture is stable! Input and output shapes align perfectly.")
        
    except Exception as e:
        print(f"\n[FAILED] Architecture has a bug. Error details:\n{e}")

if __name__ == "__main__":
    run_sanity_check()