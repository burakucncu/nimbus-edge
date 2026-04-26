import torch
import segmentation_models_pytorch as smp
import os

def get_pretrained_unet(device):
    """
    Endüstri standardı olan ResNet34 omurgalı U-Net mimarisini kurar
    ve yerel diskteki (.pth) uzman beyni modele enjekte eder.
    """
    # Yerel ağırlık dosyamızın yolu (fetch_real_brain.py ile mühürlediğimiz dosya)
    local_weights = "models/resnet34_imagenet.pth"
    
    # 1. Ham Mimariyi Kur (İnternete bağlanmadan)
    model = smp.Unet(
        encoder_name="resnet34",  # 34 katmanlı uzman omurga
        encoder_weights=None,     # None yapıyoruz çünkü internetten değil yerelden okuyacağız
        in_channels=4,            # Göktürk/Planet vb. 4 bantlı (RGB + NIR) görüntümüz
        classes=1                 # Çıktı: Sadece Bulut Maskesi (Binary)
    )

    # 2. Yerel Beyni (Ağırlıkları) Modele Enjekte Et
    if os.path.exists(local_weights):
        print(f"🧠 Yerel Uzman Beyin Yükleniyor: {local_weights}")
        # map_location ile ağırlıkları doğrudan senin Mac M serisi (mps) işlemcine yönlendiriyoruz
        model.load_state_dict(torch.load(local_weights, map_location=device))
    else:
        print("⚠️ KRİTİK UYARI: Yerel ağırlık bulunamadı! Lütfen önce 'python3 src/fetch_real_brain.py' çalıştırın.")

    return model