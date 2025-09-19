# 🧠 Alzheimer CNN Classification Project

## 📌 Project Overview

This project applies **Convolutional Neural Networks (CNNs)** using **PyTorch** to classify MRI brain images into different stages of **Alzheimer’s disease progression**. The aim is to develop a model that can support early detection and staging of Alzheimer’s using deep learning techniques.

---

## 🎯 Objectives

* Train a CNN model on MRI images from the Alzheimer dataset.
* Perform multi-class classification into 4 categories:

  * **Non Demented**
  * **Very Mild Demented**
  * **Mild Demented**
  * **Moderate Demented**
* Implement data augmentation, dropout, and evaluation metrics to ensure robustness.
* Visualize model performance with accuracy/loss curves, confusion matrix, and Grad-CAM heatmaps.

---

## 📂 Project Structure

```
Alzheimer-CNN-Project/
│
├── data/                        # Dataset (raw & processed)
├── notebooks/                   # Kaggle notebooks (EDA, training, transfer learning)
├── src/                         # Source code modules
│   ├── dataset.py               # Custom dataset & DataLoader
│   ├── transforms.py            # Data augmentation & normalization
│   ├── model.py                 # CNN architecture
│   ├── train.py                 # Training loop
│   ├── evaluate.py              # Evaluation functions
│   ├── visualization.py         # Accuracy/Loss plots & Grad-CAM
│   └── utils.py                 # Helper functions
├── outputs/                     # Models, logs, figures
├── docker/                      # Docker setup (Dockerfile, compose)
├── requirements.txt             # Dependencies
└── README.md                    # Project documentation
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/<username>/Alzheimer-CNN-Project.git
cd Alzheimer-CNN-Project
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Training

```bash
python src/train.py
```

### 4. Run Evaluation

```bash
python src/evaluate.py
```

### 5. Run with Docker (optional)

```bash
docker build -t alzheimer-cnn .
docker run -it -p 8888:8888 alzheimer-cnn
```

---

## 📊 Results & Evaluation

* **Metrics:** Accuracy, Loss, Precision, Recall, F1-score.
* **Visualizations:**

  * Accuracy & Loss curves (epoch-wise)
  * Confusion matrix
  * Classification report
  * Grad-CAM heatmaps highlighting important brain regions

---

## 🚀 Extensions & Bonus

* **Transfer Learning:** Experiments with pre-trained models (ResNet, EfficientNet).
* **HPO:** Hyperparameter optimization (learning rate, batch size, dropout).
* **Explainability:** Grad-CAM for medical interpretability.

---

## 📎 Dataset Reference

* Kaggle: [Alzheimer MRI Dataset](https://www.kaggle.com/yasserhessein/dataset-alzheimer)
* Related Paper: [Deep learning based prediction of Alzheimer’s disease from MRI](https://www.researchgate.net/publication/348486602_Deep_learning_based_prediction_of_Alzheimer's_disease_from_magnetic_resonance_images)

---

## 👤 Author

* **Your Name**
  Bootcamp Project – CNN Deep Learning (Alzheimer MRI Classification)

---

## 📜 License

This project is licensed under the MIT License.
