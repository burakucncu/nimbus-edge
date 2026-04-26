import torch
import os
import segmentation_models_pytorch as smp

def download_and_save_weights():
    # 1. Klasör kontrolü
    os.makedirs("models", exist_ok=True)
    save_path = "models/resnet34_imagenet.pth"

    print("🌐 PyTorch Sunucularına Bağlanılıyor...")
    print("🧠 ResNet34 (ImageNet) 'Gerçek' Beyni İndiriliyor (Yaklaşık 80-100MB)...")

    try:
        # SMP kütüphanesi üzerinden orijinal mimariyi çağırıyoruz
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet", # Bu komut internetten indirmeyi tetikler
            in_channels=4,
            classes=1
        )

        # 2. İndirilen bu muazzam zekayı fiziksel dosya olarak kaydediyoruz
        torch.save(model.state_dict(), save_path)
        
        print("-" * 40)
        print(f"✅ BAŞARILI! Gerçek ağırlık dosyası oluşturuldu: {save_path}")
        print(f"Dosya Boyutu: {os.path.getsize(save_path) / (1024*1024):.2f} MB")
        print("Artık bu dosya senin projenin kalıcı bir parçasıdır.")
        print("-" * 40)

    except Exception as e:
        print(f"❌ HATA: İndirme sırasında bir sorun oluştu: {e}")

if __name__ == "__main__":
    download_and_save_weights()