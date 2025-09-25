import numpy as np
import os
from openai import OpenAI

def get_peak_activation_region(heatmap):
    """
    Grad-CAM ısı haritasındaki en yüksek aktivasyon noktasını bulur.
    heatmap: H×W numpy array
    """
    max_val = heatmap.max()
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return {"x": int(x), "y": int(y), "value": float(max_val)}

def explain_prediction_with_chatgpt(predicted_class, confidence, peak_region, class_names):
    """
    Modelin tahminini ve Grad-CAM bilgilerini OpenAI ChatGPT API üzerinden yorumlatır.
    Kaggle Secrets içinde saklanan OPENAI_API_KEY kullanılır.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY bulunamadı! Kaggle Secrets bölümüne eklediğinizden emin olun.")

    client = OpenAI(api_key=api_key)

    notes = f"Grad-CAM en yüksek aktivasyon noktası (x={peak_region['x']}, y={peak_region['y']}), yoğunluk={peak_region['value']:.2f}."

    prompt = f"""
    Bir Alzheimer sınıflandırma modeli MRI görüntüsünü analiz etti.
    Tahmin edilen sınıf: {predicted_class}
    Güven seviyesi: %{confidence*100:.2f}

    {notes}

    Lütfen modelin bu sınıfa nasıl karar vermiş olabileceğini, beynin potansiyel bölgeleri açısından
    (ör. hipokampus, korteks, temporal lob) yorumla. Açıklaman tıbbi bir içgörü gibi olsun, ama kullanıcı dostu kal.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Sen bir nöroloji uzmanı gibi açıklama yap."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content
