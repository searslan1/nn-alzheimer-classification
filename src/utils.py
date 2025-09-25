import numpy as np
import os

# 🔑 Kaggle Secrets veya ortam değişkenlerinden API key'i al
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    OPENAI_KEY = user_secrets.get_secret("OPENAI_API_KEY")
    GEMINI_KEY = user_secrets.get_secret("GEMINI_API_KEY")
except Exception:
    OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
    GEMINI_KEY = os.environ.get("GEMINI_API_KEY")

# --- Grad-CAM Analiz Fonksiyonu ---
def get_peak_activation_region(heatmap):
    max_val = heatmap.max()
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return {"x": int(x), "y": int(y), "value": float(max_val)}

# --- Açıklama Fonksiyonu ---
def explain_prediction_with_ai(predicted_class, confidence, peak_region, class_names):
    notes = f"Grad-CAM en yüksek aktivasyon noktası (x={peak_region['x']}, y={peak_region['y']}), yoğunluk={peak_region['value']:.2f}."

    prompt = f"""
    Bir Alzheimer sınıflandırma modeli MRI görüntüsünü analiz etti.
    Tahmin edilen sınıf: {predicted_class}
    Güven seviyesi: %{confidence*100:.2f}

    {notes}

    Lütfen modelin bu sınıfa nasıl karar vermiş olabileceğini, beynin potansiyel bölgeleri açısından
    (ör. hipokampus, korteks, temporal lob) yorumla. Açıklaman tıbbi bir içgörü gibi olsun, ama kullanıcı dostu kal.
    """

    # --- Direkt Gemini kullan ---
    import google.generativeai as genai
    if not GEMINI_KEY:
        raise ValueError("GEMINI_API_KEY bulunamadı! Kaggle Secrets veya .env üzerinden tanımlayın.")
    
    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text
