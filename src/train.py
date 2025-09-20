import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from dataset import build_dataloaders
from model import AlzheimerCNN
from evaluate import evaluate_model
from visualization import plot_training
from transforms import train_transform, val_transform

def train_model(
    data_root="data/processed",
    epochs=20,
    batch_size=32,
    lr=0.001,
    device=None,
    save_dir="outputs/models"
):
    # Cihaz seçimi
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # DataLoader’ları hazırla
    train_loader, val_loader, test_loader, meta = build_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        train_transform=train_transform,
        val_transform=val_transform
    )
    classes = meta["classes"]
    print(f"Classes: {classes}")

    # Model & optimizer
    model = AlzheimerCNN(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # TensorBoard için log
    writer = SummaryWriter(log_dir="outputs/logs")

    # History dict
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # Eğitim döngüsü
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total

        # Validation
        val_acc, val_loss = evaluate_model(model, val_loader, criterion, device)

        # Log kayıt
        writer.add_scalars("Loss", {"Train": train_loss, "Val": val_loss}, epoch+1)
        writer.add_scalars("Accuracy", {"Train": train_acc, "Val": val_acc}, epoch+1)

        # History
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    # Model kaydet
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "alzheimer_cnn.pt")
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model kaydedildi: {model_path}")

    # Eğitim grafikleri
    os.makedirs("outputs/figures", exist_ok=True)
    plot_training(history, save_path="outputs/figures/training_curves.png")

    return model, history, classes


if __name__ == "__main__":
    train_model()
