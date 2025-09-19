🧠 Alzheimer MRI Classification with CNN
📌 Proje Özeti

Bu proje, Convolutional Neural Networks (CNN) kullanarak MRI beyin görüntülerinden Alzheimer hastalığının evrelerini sınıflandırmayı amaçlamaktadır.
Erken teşhis için yapay zekâ tabanlı görüntü analizi, hastaların yaşam kalitesini artırma ve doktorlara yardımcı olma potansiyeline sahiptir.

Sınıflar (labels):

Non Demented (Sağlıklı)

Very Mild Demented (Çok Hafif Evre)

Mild Demented (Hafif Evre)

Moderate Demented (Orta Evre)

📂 Veri Seti

Kaynak: Kaggle Alzheimer Dataset

İçerik: MRI görüntüleri, train/test klasörleri

Önişleme:

Görseller 128x128 px boyutuna getirildi

Normalizasyon (0–1 aralığı)

Data augmentation (rotation, zoom, flip)

⚙️ Model Mimarisi

Conv2D + ReLU (özellik çıkarma)

MaxPooling2D (boyut küçültme)

Dropout (overfitting önleme)

Dense Layer (tam bağlı katmanlar)

Softmax Output (olasılık dağılımı → sınıf tahmini)

📌 Ayrıca transfer learning (ResNet, VGG16) ile ek deneyler yapılmıştır.

🏋️‍♂️ Eğitim

Framework: TensorFlow/Keras (Python)

Loss: Categorical Crossentropy

Optimizer: Adam

Epochs: 20–30 (early stopping ile kontrol)

Batch Size: 32

📊 Sonuçlar

Accuracy/Loss grafikleri (epoch bazlı)

Confusion Matrix → hangi evrelerin karıştığı

Classification Report (precision, recall, F1)

Grad-CAM görselleştirme → MRI’da hangi bölgelerin karar için kullanıldığı

(Ekran görüntüleri buraya eklenecek)

🔬 Analiz

Overfitting/underfitting durumları incelendi.

Dropout ve augmentation ile genelleme artırıldı.

Hiperparametre optimizasyonu (learning rate, batch size, dropout) test edildi.

📎 Kullanım
git clone https://github.com/<username>/alzheimer-cnn.git
cd alzheimer-cnn
pip install -r requirements.txt
python train.py

📑 Kaynaklar

Kaggle Alzheimer Dataset

Research Paper

📌 Bağlantılar

📓 Kaggle Notebook

💻 GitHub Repo