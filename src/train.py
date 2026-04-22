import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import NimbusCloudDataset
from model import LightweightUNet

def train_model():
    # 1. Hardware Configuration (Optimized for Mac M1/M2/M3)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")

    # 2. Hyperparameters
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 8
    NUM_EPOCHS = 20
    IMAGE_DIR = "data/images"
    MASK_DIR = "data/masks"

    # 3. Data Loading
    dataset = NimbusCloudDataset(image_dir=IMAGE_DIR, mask_dir=MASK_DIR)
    # Note: Ensure you have data in the folders before running
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 4. Model, Loss, and Optimizer
    model = LightweightUNet(in_channels=4, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Training Loop
    print("Starting training...")
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
    torch.save(model.state_dict(), "models/nimbus_model_v1.pt")
    print("Training complete. Model saved to models/nimbus_model_v1.pt")

if __name__ == "__main__":
    train_model()