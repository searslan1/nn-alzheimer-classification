import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# ==============================
# 1. Eğitim Grafikleri
# ==============================
def plot_training(history, save_path="outputs/figures/training_curves.png"):
    plt.figure(figsize=(12,5))

    # Accuracy
    plt.subplot(1,2,1)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curve")

    # Loss
    plt.subplot(1,2,2)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")

    plt.savefig(save_path)
    plt.close()


# ==============================
# 2. Grad-CAM
# ==============================
def grad_cam(model, image_tensor, target_class, layer_name="conv2"):
    """
    Grad-CAM: Görüntü üzerinde modelin dikkat ettiği bölgeleri gösterir.
    """
    model.eval()

    # Forward hook için feature map
    activations = {}
    def forward_hook(module, input, output):
        activations["value"] = output

    # Backward hook için gradyan
    gradients = {}
    def backward_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0]

    # Hook kaydet
    layer = dict([*model.named_modules()])[layer_name]
    layer.register_forward_hook(forward_hook)
    layer.register_backward_hook(backward_hook)

    # Forward pass
    output = model(image_tensor.unsqueeze(0))
    loss = output[0, target_class]
    loss.backward()

    # Grad-CAM hesapla
    grad = gradients["value"].cpu().data.numpy()[0]
    fmap = activations["value"].cpu().data.numpy()[0]

    weights = np.mean(grad, axis=(1,2))
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * fmap[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cam / cam.max()  # normalize

    return cam
