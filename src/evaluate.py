import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

import torch


def evaluate_model(model, data_loader, criterion, device="cpu"):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    acc = 100. * correct / total
    return acc, avg_loss, y_true, y_pred


def save_confusion_matrix(y_true, y_pred, classes, save_path="outputs/figures/confusion_matrix.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Confusion matrix saved at {save_path}")


def save_classification_report(y_true, y_pred, classes, save_path="outputs/reports/classification_report.txt"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    report = classification_report(y_true, y_pred, target_names=classes)

    with open(save_path, "w") as f:
        f.write(report)

    print(f"✅ Classification report saved at {save_path}")
