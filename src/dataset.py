import os
from typing import List, Tuple, Optional, Dict
from PIL import Image
from transforms import train_transform, val_transform

import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit

# -----------------------------
# 0) Yardımcılar / sabitler
# -----------------------------
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

def _is_image_file(filename: str) -> bool:
    return os.path.splitext(filename)[-1].lower() in IMAGE_EXTS

def _list_classes(root_dir: str) -> List[str]:
    """root_dir altında klasör bazlı sınıf isimlerini (alfabetik) döndürür."""
    classes = [d for d in os.listdir(root_dir)
               if os.path.isdir(os.path.join(root_dir, d))]
    classes = sorted(classes)
    if len(classes) == 0:
        raise RuntimeError(f"No class folders found in: {root_dir}")
    return classes

def _scan_image_paths(root_dir: str, classes: List[str]) -> Tuple[List[str], List[int]]:
    """Sınıf klasörlerini gezerek (path, label_id) listeleri çıkarır."""
    image_paths, labels = [], []
    for cls_idx, cls_name in enumerate(classes):
        cls_dir = os.path.join(root_dir, cls_name)
        for fname in os.listdir(cls_dir):
            if _is_image_file(fname):
                image_paths.append(os.path.join(cls_dir, fname))
                labels.append(cls_idx)
    if len(image_paths) == 0:
        raise RuntimeError(f"No images found under: {root_dir}")
    return image_paths, labels


# -----------------------------
# 1) Custom Dataset
# -----------------------------
class AlzheimerDataset(Dataset):
    """
    Klasör yapısı:
      root_dir/
        NonDemented/
          img1.jpg ...
        VeryMildDemented/
          ...
        MildDemented/
        ModerateDemented/
    """
    def __init__(self,
                 root_dir: str,
                 transform=None,
                 return_path: bool = False):
        self.root_dir = root_dir
        self.transform = transform
        self.return_path = return_path

        self.classes = _list_classes(root_dir)
        self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(self.classes)}
        self.image_paths, self.labels = _scan_image_paths(root_dir, self.classes)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image: {path}\n{e}")

        if self.transform is not None:
            img = self.transform(img)

        if self.return_path:
            return img, label, path
        return img, label


# -----------------------------
# 2) Class distribution & weights
# -----------------------------
def compute_class_distribution(labels: List[int], num_classes: int) -> List[int]:
    counts = [0] * num_classes
    for y in labels:
        counts[y] += 1
    return counts

def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    """
    weight_c = total_samples / (num_classes * count_c)
    """
    counts = compute_class_distribution(labels, num_classes)
    total = len(labels)
    weights = []
    for c in range(num_classes):
        if counts[c] == 0:
            # sınıf yoksa 0 bölmeye karşı koruma
            weights.append(0.0)
        else:
            weights.append(total / (num_classes * counts[c]))
    return torch.tensor(weights, dtype=torch.float32)


# -----------------------------
# 3) Stratified split yardımcıları
# -----------------------------
def stratified_split_indices(labels: List[int],
                             val_ratio: float = 0.2,
                             seed: int = 42):
    """
    labels üzerinden stratified split indeksleri döndürür.
    """
    indices = list(range(len(labels)))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(sss.split(indices, labels))
    return train_idx.tolist(), val_idx.tolist()


# -----------------------------
# 4) DataLoader kurucu
# -----------------------------
def build_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 2,
    pin_memory: bool = True,
    train_transform=None,
    val_transform=None,
    train_dir: str = "train",
    val_dir: str = "val",
    test_dir: str = "test"
):
    train_path = os.path.join(data_root, train_dir)
    val_path = os.path.join(data_root, val_dir)
    test_path = os.path.join(data_root, test_dir)

    train_set = AlzheimerDataset(train_path, transform=train_transform)
    val_set   = AlzheimerDataset(val_path, transform=val_transform)
    test_set  = AlzheimerDataset(test_path, transform=val_transform) if os.path.isdir(test_path) else None

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = None
    if test_set is not None:
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=pin_memory)

    meta = {
        "classes": train_set.classes,
        "num_classes": len(train_set.classes)
    }
    return train_loader, val_loader, test_loader, meta

