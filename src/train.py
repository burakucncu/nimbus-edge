import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import NimbusCloudDataset

# 1. DEĞİŞİKLİK: Eski modeli değil, gerçek mimariyi çağıran fonksiyonumuzu ekliyoruz
from model import get_pretrained_unet

def train_model():
    # 1. Hardware Configuration (Optimized for Mac M1/M2/M3)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"🚀 Using device: {device}")

    # 2. Hyperparameters
    # 2. DEĞİŞİKLİK: Öğrenme Oranını (LR) 1e-4'ten 1e-5'e düşürdük! (Hafızayı korumak için)
    LEARNING_RATE = 1e-5
    BATCH_SIZE = 8
    NUM_EPOCHS = 20
    IMAGE_DIR = "data/images"
    MASK_DIR = "data/masks"

    # 3. Data Loading
    dataset = NimbusCloudDataset(image_dir=IMAGE_DIR, mask_dir=MASK_DIR)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 4. Model, Loss, and Optimizer
    # 3. DEĞİŞİKLİK: Modeli yeni fonksiyondan alıp cihaza (MPS) yüklüyoruz
    model = get_pretrained_unet(device).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Training Loop
    print("🧠 Transfer Learning (İnce Ayar) başlatılıyor...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device)
            targets = targets.to(device)

            # Forward pass
            predictions = model(data)
            loss = criterion(predictions, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Loss: {epoch_loss/len(train_loader):.4f}")

    # 6. Save Model
    # 4. DEĞİŞİKLİK: Kayıt adını yeni ResNet34 mimarisine uygun hale getirdik
    model_save_path = "models/nimbus_resnet34_finetuned.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"✅ Eğitim tamamlandı. Kennedy Uzmanı Model şuraya kaydedildi: {model_save_path}")

if __name__ == "__main__":
    train_model()