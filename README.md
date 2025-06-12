# Pneumonia_detection
# Pneumonia Detection with Chest X-ray

This is a simple web-based application built with **Streamlit** that uses a pre-trained deep learning model to detect **Pneumonia** from chest X-ray images.

---

## 🧠 Model Summary

- Model Type: Convolutional Neural Network (CNN)
- Input Image Size: 224x224 (RGB)
- Framework: TensorFlow/Keras
- Output: Binary classification — Pneumonia (1) or Normal (0)

---

## 🖥️ App Features

- Upload a chest X-ray image (JPEG/PNG)
- Predict whether the patient is likely suffering from pneumonia
- Display confidence score for prediction
- Visual interface using Streamlit

---

## Getting started
### 1. Clone the repository
```bash
git clone https://github.com/your-username/Pneumonia-detection-app.git
cd pneumonia-detection-app
```

### 2. Create virtual env (if running locally)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


### 3. Installing dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
I used google colab for running the app using ngrok, so double check the dependcies if needed

## App Outputs
