import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

# --------------------------
# 1) Accuracy / Loss Grafikleri
# --------------------------
def plot_training(history, save_path="outputs/figures/training_curves.png"):
    """
    history dict (train_loss, val_loss, train_acc, val_acc) üzerinden epoch bazlı grafik üretir.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training & Validation Loss")

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Training & Validation Accuracy")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Training curves saved at {save_path}")


# --------------------------
# 2) Grad-CAM
# --------------------------
def generate_gradcam(model, image_tensor, target_class, conv_layer, device="cpu"):
    """
    Tek bir görüntü için Grad-CAM heatmap üretir.
    - image_tensor: (1, C, H, W) boyutunda Tensor
    - target_class: hedef sınıf id’si
    - conv_layer: modeldeki conv layer referansı (örn. model.conv3)
    """
    model.eval()
    image_tensor = image_tensor.to(device)

    # Hook mekanizması
    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out.detach())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    # Hook’ları kaydet
    fwd_hook = conv_layer.register_forward_hook(forward_hook)
    bwd_hook = conv_layer.register_backward_hook(backward_hook)

    # Forward + backward
    output = model(image_tensor)
    pred_class = output.argmax(dim=1).item()
    loss = output[0, target_class]
    model.zero_grad()
    loss.backward()

    # Aktivasyon ve gradientleri al
    act = activations[0]      # (batch, channels, H, W)
    grad = gradients[0]       # (batch, channels, H, W)

    weights = grad.mean(dim=(2, 3), keepdim=True)  # (batch, channels, 1, 1)
    cam = (weights * act).sum(dim=1, keepdim=True)  # (batch, 1, H, W)
    cam = F.relu(cam)
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # normalize [0,1]

    # Hook’ları temizle
    fwd_hook.remove()
    bwd_hook.remove()

    return cam, pred_class


def plot_gradcam_on_image(image_pil, heatmap, save_path="outputs/figures/gradcam.png", alpha=0.5):
    """
    Orijinal PIL image ile Grad-CAM heatmap’i üst üste bindirip kaydeder.
    """
    import cv2
    import numpy as np

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Görseli numpy array’e çevir
    img = np.array(image_pil.convert("RGB"))
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
    cv2.imwrite(save_path, overlay[:, :, ::-1])  # RGB → BGR dönüşümü
    print(f"✅ Grad-CAM saved at {save_path}")
