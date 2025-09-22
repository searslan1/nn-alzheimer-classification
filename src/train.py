import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import build_dataloaders
from evaluate import evaluate_model, save_confusion_matrix, save_classification_report
from visualization import plot_training, generate_gradcam, plot_gradcam_on_image
from transforms import base_train_transform, default_transform, class_transforms
from torchvision import transforms
from model import get_model
from losses import FocalLoss   # 🔑 eklendi


def train_model(
    data_root="data/processed",
    epochs=50,
    batch_size=32,
    lr=0.001,
    device=None,
    save_dir="outputs/models",
    model_name="resnet",
    use_sampler=True,
    use_focal=False,                # 🔑 focal loss entegrasyonu
    gamma=2.0,                     # 🔑 focal loss parametresi
    early_stopping_patience=10
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # --------------------------
    # DataLoader
    # --------------------------
    train_loader, val_loader, test_loader, meta = build_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        train_transform=base_train_transform,   # 🔑 her zaman tensor + normalize
        val_transform=default_transform,
        class_transforms=class_transforms,
        use_sampler=use_sampler
    )
    classes = meta["classes"]
    print(f"Classes: {classes}")
    print(f"Class distribution: {meta['class_distribution']}")

    # --------------------------
    # Model
    # --------------------------
    model = get_model(
        model_name=model_name,
        num_classes=len(classes),
        pretrained=True,
        freeze_backbone=True
    ).to(device)

    # Grad-CAM için layer seçimi
    if model_name == "cnn":
        gradcam_layer = model.conv3
    elif model_name == "resnet":
        gradcam_layer = model.backbone.layer4[-1]

    # --------------------------
    # Loss, optimizer, scheduler
    # --------------------------
    class_weights = meta["class_weights"].to(device)

    if use_focal:
        criterion = FocalLoss(alpha=class_weights, gamma=gamma)
        print("🔑 Using Focal Loss")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("🔑 Using CrossEntropyLoss")

    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5, verbose=True
    )

    # --------------------------
    # TensorBoard
    # --------------------------
    run_id = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=f"outputs/logs/{model_name}_{run_id}")

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    best_val_loss = float("inf")
    patience_counter = 0

    # --------------------------
    # Training loop
    # --------------------------
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
        val_acc, val_loss, _, _ = evaluate_model(model, val_loader, criterion, device)

        # Scheduler step
        scheduler.step(val_loss)

        # Logging
        writer.add_scalars("Loss", {"Train": train_loss, "Val": val_loss}, epoch+1)
        writer.add_scalars("Accuracy", {"Train": train_acc, "Val": val_acc}, epoch+1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% "
              f"(LR: {optimizer.param_groups[0]['lr']:.6f})")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("⏹ Early stopping triggered.")
                break

    # --------------------------
    # Save model & results
    # --------------------------
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"alzheimer_{model_name}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model kaydedildi: {model_path}")

    os.makedirs("outputs/figures", exist_ok=True)
    plot_training(history, save_path="outputs/figures/training_curves.png")

    print("Evaluating on validation set...")
    val_acc, val_loss, y_true, y_pred = evaluate_model(model, val_loader, criterion, device)

    os.makedirs("outputs/reports", exist_ok=True)
    save_confusion_matrix(y_true, y_pred, classes, save_path="outputs/figures/confusion_matrix.png")
    save_classification_report(y_true, y_pred, classes, save_path="outputs/reports/classification_report.txt")

    print("✅ Evaluation results saved.")

    # --------------------------
    # Grad-CAM örneği
    # --------------------------
    print("Generating Grad-CAM example...")
    sample_img, sample_label = next(iter(val_loader))
    sample_img = sample_img[0].unsqueeze(0).to(device)

    heatmap, pred_class = generate_gradcam(
        model,
        sample_img,
        conv_layer=gradcam_layer,
        device=device
    )

    inv_transform = transforms.Compose([
        transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        ),
        transforms.ToPILImage()
    ])
    original_img = inv_transform(sample_img.squeeze().cpu())

    plot_gradcam_on_image(
        original_img,
        heatmap,
        save_path="outputs/figures/gradcam_example.png"
    )
    print("✅ Grad-CAM example saved.")

    return model, history, classes


if __name__ == "__main__":
    train_model(use_focal=False) 
