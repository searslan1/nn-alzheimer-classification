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

    # --- Öncelik OpenAI ---
    if OPENAI_KEY:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Daha ekonomik
            messages=[
                {"role": "system", "content": "Sen bir nöroloji uzmanı gibi açıklama yap."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    # --- Fallback: Gemini ---
    elif GEMINI_KEY:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text

    else:
        raise ValueError("Ne OPENAI_API_KEY ne de GEMINI_API_KEY bulundu! Kaggle Secrets veya .env üzerinden tanımlayın.")
