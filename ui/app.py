import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import random
import sys
from src.utils import get_peak_activation_region, explain_prediction_with_chatgpt

# Proje kök dizinini Python yoluna ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import get_model
from src.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.visualization import generate_gradcam
from src.dataset import _list_classes

# --- 1. Model ve Sınıfları Yükle ---
@st.cache_resource
def load_model(model_path, num_classes):
    # train.py'deki son başarılı konfigürasyonu kullan
    model = get_model(
        model_name="resnet", 
        num_classes=num_classes, 
        pretrained=False, 
        freeze_backbone=False
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Model ve ağırlıkların yolu
MODEL_PATH = 'outputs/models/alzheimer_resnet.pt'
NUM_CLASSES = 4
MODEL = load_model(MODEL_PATH, NUM_CLASSES)
CLASS_NAMES = _list_classes(os.path.join(os.path.dirname(__file__), '../data/processed/train'))

# --- 2. Dönüşüm Fonksiyonu ---
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# --- 3. Tahmin ve Grad-CAM Fonksiyonu ---
def predict_and_grad_cam(image, model, transform, class_names):
    img_tensor = transform(image).unsqueeze(0)
    
    target_layer = model.backbone.layer4[-1] 
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        predicted_class = class_names[predicted_idx]
        confidence = probabilities[0][predicted_idx].item()
    
    grad_cam_input_img = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])(image).unsqueeze(0)
    
    heatmap, _ = generate_gradcam(
        model, 
        grad_cam_input_img.to(torch.device('cpu')), 
        conv_layer=target_layer, 
        target_class=predicted_idx,
        device=torch.device('cpu')
    )
    
    return predicted_class, confidence, heatmap, grad_cam_input_img.squeeze()

peak_region = get_peak_activation_region(heatmap)

with st.spinner("ChatGPT yorumu hazırlanıyor..."):
    explanation = explain_prediction_with_chatgpt(predicted_class, confidence, peak_region, CLASS_NAMES)

st.subheader("🧾 ChatGPT Yorumu")
st.write(explanation)

# --- 4. Streamlit UI ---
st.set_page_config(layout="wide", page_title="Alzheimer Sınıflandırma Sistemi")

st.title("🧠 Alzheimer MRI Görüntü Sınıflandırması")
st.markdown("Bu uygulama, MRI beyin görüntülerini kullanarak Alzheimer hastalığının farklı evrelerini sınıflandırmaktadır.")

upload_option = st.radio("Bir görsel yükleyin veya örnek bir görsel seçin:", ("Dosya Yükle", "Örnek Görsel Seç"))

uploaded_file = None
selected_sample_image = None
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/processed/test')

if upload_option == "Dosya Yükle":
    uploaded_file = st.file_uploader("Bir MRI görüntüsü yükleyin...", type=["jpg", "jpeg", "png"])
elif upload_option == "Örnek Görsel Seç":
    all_test_images = []
    if os.path.exists(TEST_DATA_DIR):
        for cls in os.listdir(TEST_DATA_DIR):
            cls_path = os.path.join(TEST_DATA_DIR, cls)
            if os.path.isdir(cls_path):
                for img_name in os.listdir(cls_path):
                    if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                        all_test_images.append(os.path.join(cls_path, img_name))
    
    if all_test_images:
        random_image_path = random.choice(all_test_images)
        selected_sample_image = Image.open(random_image_path).convert('RGB')
        st.write(f"Seçilen örnek görsel: **{os.path.basename(random_image_path)}**")
        st.image(selected_sample_image, caption="Seçilen Örnek Görsel", use_column_width=True)
    else:
        st.warning(f"Örnek görsel bulunamadı. Lütfen '{TEST_DATA_DIR}' klasörünüzün doğru yapılandırıldığından emin olun.")

image_to_process = None
if uploaded_file is not None:
    image_to_process = Image.open(uploaded_file).convert('RGB')
    st.image(image_to_process, caption="Yüklenen Görsel", use_column_width=True)
elif selected_sample_image is not None:
    image_to_process = selected_sample_image

if image_to_process is not None:
    if st.button("Sınıflandır"):
        with st.spinner("Görüntü sınıflandırılıyor ve Grad-CAM oluşturuluyor..."):
            predicted_class, confidence, heatmap, original_tensor = predict_and_grad_cam(
                image_to_process, MODEL, inference_transform, CLASS_NAMES
            )
            
            st.success("Sınıflandırma Tamamlandı!")
            st.write(f"**Tahmin Edilen Sınıf:** **{predicted_class}**")
            st.write(f"**Güven Seviyesi:** **%{(confidence * 100):.2f}**")
            
            st.subheader("Modelin Odaklandığı Alanlar (Grad-CAM)")

            # 🔥 Eğitimdeki gibi overlay yap
            import cv2
            import numpy as np

            img = np.array(image_to_process.convert("RGB"))
            heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img, 0.5, heatmap_colored, 0.5, 0)

            st.image(overlay, caption="Grad-CAM Overlay", use_column_width=True)
            st.info("Kırmızı bölgeler modelin karar verirken en çok dikkat ettiği yerleri gösterir.")
            peak_region = get_peak_activation_region(heatmap)
        with st.spinner("ChatGPT yorumu hazırlanıyor..."):
            explanation = explain_prediction_with_chatgpt(predicted_class, confidence, peak_region, CLASS_NAMES)
            st.subheader("🧾 ChatGPT Yorumu")
            st.write(explanation)
    else:
            st.info("Lütfen bir MRI görüntüsü yükleyin veya yukarıdan bir örnek seçin.")
