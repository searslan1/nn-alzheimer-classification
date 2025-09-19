import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import AlzheimerDataset
from model import AlzheimerCNN
from evaluate import evaluate_model
from transforms import train_transform, val_transform
import os

def train_model(data_dir="data/processed", epochs=20, batch_size=32, lr=0.001, device="cuda"):
    # Dataset
    train_dataset = AlzheimerDataset(os.path.join(data_dir, "train"), transform=train_transform)
    val_dataset   = AlzheimerDataset(os.path.join(data_dir, "val"), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = AlzheimerCNN(num_classes=4).to(device)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training Loop
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
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total
        val_acc, val_loss = evaluate_model(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {running_loss/len(train_loader):.4f} "
              f"Train Acc: {train_acc:.2f}% "
              f"Val Loss: {val_loss:.4f} "
              f"Val Acc: {val_acc:.2f}%")

    # Model kaydet
    os.makedirs("outputs/models", exist_ok=True)
    torch.save(model.state_dict(), "outputs/models/alzheimer_cnn.pt")

    return model
