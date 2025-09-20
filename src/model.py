import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# -----------------------------
# 1) Basit CNN Modeli
# -----------------------------
class AlzheimerCNN(nn.Module):
    def __init__(self, num_classes: int = 4):
        super(AlzheimerCNN, self).__init__()

        # 1. Convolutional block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 2. Convolutional block
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 3. Convolutional block
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1   = nn.Linear(128 * 28 * 28, 512)  # Input boyutu image size'a göre değişebilir!
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2   = nn.Linear(512, num_classes)

    def forward(self, x):
        # Block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # Block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        # Block 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        # Flatten
        x = x.view(x.size(0), -1)
        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = self.fc2(x)
        return x


# -----------------------------
# 2) Transfer Learning: ResNet50
# -----------------------------
class AlzheimerResNet(nn.Module):
    def __init__(self, num_classes: int = 4, pretrained: bool = True, freeze_backbone: bool = True):
        super(AlzheimerResNet, self).__init__()

        # Torchvision’dan ResNet50 yükle
        self.backbone = models.resnet50(pretrained=pretrained)

        # İsteğe bağlı: backbone dondur
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Son katmanı Alzheimer sınıfları için değiştir
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


# -----------------------------
# 3) Model seçici yardımcı fonksiyon
# -----------------------------
def get_model(model_name: str, num_classes: int = 4, pretrained: bool = True, freeze_backbone: bool = True):
    """
    model_name: 'cnn' veya 'resnet'
    """
    if model_name.lower() == "cnn":
        return AlzheimerCNN(num_classes=num_classes)
    elif model_name.lower() == "resnet":
        return AlzheimerResNet(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
