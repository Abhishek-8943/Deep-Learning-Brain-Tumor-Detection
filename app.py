"""
Brain Tumor Detection Web Application
Deployed on Streamlit Cloud
"""

import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import json
import os
import pandas as pd
import plotly.express as px

# Page Configuration
st.set_page_config(
    page_title="Brain Tumor Detection AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
MODEL_PATH = 'brain_tumor_model.keras'
CLASS_INDICES_PATH = 'class_indices.json'
IMG_SIZE = (224, 224)

# Load Model
@st.cache_resource
def load_model_and_indices():
    if os.path.exists(CLASS_INDICES_PATH):
        with open(CLASS_INDICES_PATH, 'r') as f:
            class_indices = json.load(f)
        index_to_class = {v: k for k, v in class_indices.items()}
    else:
        index_to_class = {0: 'Brain Tumor', 1: 'Healthy'}
    
    model = keras.models.load_model(MODEL_PATH)
    return model, index_to_class

try:
    model, index_to_class = load_model_and_indices()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_loaded = False

# Prediction Function
def predict_brain_scan(image, model, index_to_class):
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize(IMG_SIZE)
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array, verbose=0)
        
        results = {}
        for idx, prob in enumerate(predictions[0]):
            class_name = index_to_class.get(idx, f"Class {idx}")
            results[class_name] = float(prob)
        
        return results
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# Header
st.markdown("""
<div class="main-header">
    <h1>🧠 Brain Tumor Detection AI</h1>
    <p>AI-Powered Medical Image Analysis System</p>
</div>
""", unsafe_allow_html=True)

# Main Content
if not model_loaded:
    st.error("Model failed to load")
    st.stop()

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Upload Brain Scan")
    uploaded_file = st.file_uploader("Choose a brain scan", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Scan", use_column_width=True)
        
        if st.button("Analyze Scan", type="primary"):
            with st.spinner("Analyzing..."):
                results = predict_brain_scan(image, model, index_to_class)
                if results:
                    st.session_state['results'] = results
                    st.session_state['analyzed'] = True

with col2:
    st.markdown("### Analysis Results")
    
    if 'analyzed' in st.session_state and st.session_state['analyzed']:
        results = st.session_state['results']
        max_class = max(results, key=results.get)
        max_confidence = results[max_class]
        
        st.success(f"Prediction: {max_class}")
        st.metric("Confidence", f"{max_confidence*100:.1f}%")
        
        for class_name, confidence in results.items():
            st.write(f"{class_name}: {confidence*100:.2f}%")
    else:
        st.info("Upload an image and click Analyze")

st.warning("⚠️ Educational purposes only. Not for clinical diagnosis.")
