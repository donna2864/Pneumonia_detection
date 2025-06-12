import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# App
st.title("Pneumonia Detection with Chest X-ray")
st.text("Upload a chest X-ray image (JPG/PNG)")

uploaded_file = st.file_uploader("Choose file to upload", type=["jpg", "jpeg", "png"])

@st.cache_resource
def load_pneumonia_model():
    model = load_model("pneumonia_detection.h5", compile=False)
    return model

model = load_pneumonia_model()

# Prediction 
def predict(img):
    img = img.resize((224, 224))  
    img_array = image.img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = img_array / 255.0  
    prediction = model.predict(img_array)[0][0]
    return prediction

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded X-ray", use_column_width=True)

    if st.button("Predict"):
        pred = predict(img)
        if pred > 0.5:
            st.error(f"Prediction: Pneumonia detected (Confidence: {pred:.2f})")
        else:
            st.success(f"Prediction: Normal (Confidence: {1 - pred:.2f})")
