from torchvision import transforms

# Normalization değerleri (ImageNet pretrain standardı)
# MRI dataset özelinde normalize edilebilir ama genelde bunlar güvenli seçimdir.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Training transform: augmentations + normalize
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),          # tüm resimler 224x224 olacak
    transforms.RandomHorizontalFlip(p=0.5), # rastgele yatay çevirme
    transforms.RandomRotation(15),          # ±15 derece döndürme
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # renk/kontrast varyasyonu
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Validation & Test transform: sadece resize + normalize
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
