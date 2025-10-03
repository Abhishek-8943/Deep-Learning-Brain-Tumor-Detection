"""
Brain Tumor Detection Web Application - FIXED VERSION
Interactive webpage for testing your trained model
Run this in Google Colab after training your model
"""

# ============================================================================
# STEP 1: Install Gradio
# ============================================================================
!pip install gradio -q

# ============================================================================
# STEP 2: Import Libraries
# ============================================================================
import gradio as gr
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import json
import os
import pandas as pd

print("Libraries imported successfully!")

# ============================================================================
# STEP 3: Configuration
# ============================================================================
MODEL_PATH = '/content/drive/MyDrive/brain_tumor_model.keras'
CLASS_INDICES_PATH = '/content/drive/MyDrive/class_indices.json'
IMG_SIZE = (224, 224)

# ============================================================================
# STEP 4: Load Model and Class Indices
# ============================================================================
print("\n" + "="*70)
print("LOADING MODEL...")
print("="*70)

# Check if files exist
if not os.path.exists(MODEL_PATH):
    print(f"❌ ERROR: Model not found at {MODEL_PATH}")
    print("Please train the model first!")
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not os.path.exists(CLASS_INDICES_PATH):
    print(f"⚠ Warning: Class indices not found at {CLASS_INDICES_PATH}")
    print("Using default class mapping...")
    index_to_class = {0: 'Brain Tumor', 1: 'Healthy'}
else:
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
    index_to_class = {v: k for k, v in class_indices.items()}
    print(f"✓ Class indices loaded: {class_indices}")

# Load model
model = keras.models.load_model(MODEL_PATH)
print("✓ Model loaded successfully!")
print(f"✓ Model input shape: {model.input_shape}")
print(f"✓ Model output shape: {model.output_shape}")

# ============================================================================
# STEP 5: Prediction Function
# ============================================================================
def predict_brain_scan(image):
    """
    Predict whether the brain scan shows a tumor or is healthy
    """
    if image is None:
        return None, None, "⚠️ Please upload an image first."

    try:
        print("\n📸 Processing image...")

        # Convert to PIL Image if numpy array
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        print(f"✓ Original image size: {image.size}")

        # Resize image
        image = image.resize(IMG_SIZE)
        print(f"✓ Resized to: {IMG_SIZE}")

        # Convert to array
        img_array = np.array(image, dtype=np.float32)

        # Normalize to [0, 1]
        img_array = img_array / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        print(f"✓ Input array shape: {img_array.shape}")
        print("🔮 Making prediction...")

        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        print(f"✓ Raw predictions: {predictions[0]}")

        # Get probabilities for each class
        results = {}
        for idx, prob in enumerate(predictions[0]):
            class_name = index_to_class.get(idx, f"Class {idx}")
            results[class_name] = float(prob)

        print(f"✓ Formatted results: {results}")

        # Prepare bar plot data as pandas DataFrame
        import pandas as pd
        plot_data = pd.DataFrame({
            'Class': list(results.keys()),
            'Confidence': list(results.values())
        })

        # Get prediction
        max_class = max(results, key=results.get)
        max_confidence = results[max_class]

        # Generate interpretation
        if max_class == "Brain Tumor":
            emoji = "⚠️"
            if max_confidence > 0.9:
                level = "HIGH CONFIDENCE"
                message = f"The model detects a potential brain tumor with {max_confidence*100:.1f}% confidence. Immediate medical consultation is strongly recommended."
            elif max_confidence > 0.7:
                level = "MODERATE CONFIDENCE"
                message = f"The model suggests a possible brain tumor with {max_confidence*100:.1f}% confidence. Please consult a healthcare professional for proper diagnosis."
            else:
                level = "LOW CONFIDENCE"
                message = f"The model indicates a potential tumor with {max_confidence*100:.1f}% confidence. Further medical evaluation recommended for confirmation."
        else:
            emoji = "✅"
            if max_confidence > 0.9:
                level = "HIGH CONFIDENCE"
                message = f"The scan appears healthy with {max_confidence*100:.1f}% confidence. However, regular check-ups are always recommended."
            elif max_confidence > 0.7:
                level = "MODERATE CONFIDENCE"
                message = f"The scan appears healthy with {max_confidence*100:.1f}% confidence. Consider professional evaluation if symptoms persist."
            else:
                level = "LOW CONFIDENCE"
                message = f"The model is uncertain ({max_confidence*100:.1f}% confidence). Medical evaluation is recommended for accurate diagnosis."

        interpretation = f"{emoji} {level}\n\n{message}"

        print("✓ Prediction complete!\n")

        return results, plot_data, interpretation

    except Exception as e:
        error_msg = f"❌ Error during prediction: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return {"Error": 1.0}, None, error_msg

# ============================================================================
# STEP 6: Create Gradio Interface
# ============================================================================

# Create the interface
demo = gr.Blocks(title="Brain Tumor Detection", theme=gr.themes.Soft())

with demo:

    # Header
    gr.HTML("""
        <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 15px; margin-bottom: 30px;">
            <h1 style="margin: 0; font-size: 2.5em;">🧠 Brain Tumor Detection System</h1>
            <p style="margin: 10px 0 0 0; font-size: 1.2em;">Upload a brain scan image to detect potential tumors using AI</p>
        </div>
    """)

    with gr.Row():
        # Left Column - Input
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload Brain Scan")

            image_input = gr.Image(
                type="pil",
                label="Brain Scan Image",
                sources=["upload", "clipboard"]
            )

            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear", size="sm")
                predict_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")

            gr.Markdown("""
            ### ℹ️ Instructions:
            1. **Upload** a brain MRI/CT scan image
            2. Click **"Analyze"** to get prediction
            3. View results with confidence scores

            **Supported formats:** JPG, PNG, BMP, JPEG
            """)

        # Right Column - Output
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Analysis Results")

            # Prediction label
            prediction_output = gr.Label(
                label="Diagnosis Prediction",
                num_top_classes=2
            )

            # Interpretation text
            result_text = gr.Textbox(
                label="Detailed Interpretation",
                lines=6,
                interactive=False
            )

            # Confidence bar chart
            gr.Markdown("### 📈 Confidence Visualization")
            confidence_chart = gr.BarPlot(
                x="Class",
                y="Confidence",
                title="Prediction Confidence",
                y_lim=[0, 1],
                height=250,
                show_label=True
            )

    # Footer with disclaimer
    gr.HTML("""
        <div style="text-align: center; margin-top: 40px; padding: 20px; background-color: #f5f5f5; border-radius: 10px;">
            <p style="margin: 5px 0; color: #d32f2f; font-weight: bold;">⚠️ MEDICAL DISCLAIMER</p>
            <p style="margin: 5px 0; color: #666;">This is an AI model for educational and research purposes only.</p>
            <p style="margin: 5px 0; color: #666;">Always consult qualified medical professionals for accurate diagnosis and treatment.</p>
            <p style="margin: 15px 0 5px 0; color: #999; font-size: 0.9em;">Model: MobileNetV2 Transfer Learning | Framework: TensorFlow/Keras</p>
        </div>
    """)

    # ========================================================================
    # Event Handlers
    # ========================================================================

    def process_and_predict(image):
        """Wrapper function for prediction"""
        if image is None:
            return None, None, "⚠️ Please upload an image first."

        results, plot_data, interpretation = predict_brain_scan(image)
        return results, interpretation, plot_data

    def clear_all():
        """Clear all inputs and outputs"""
        return None, None, "", None

    # Connect button clicks
    predict_btn.click(
        fn=process_and_predict,
        inputs=[image_input],
        outputs=[prediction_output, result_text, confidence_chart]
    )

    clear_btn.click(
        fn=clear_all,
        outputs=[image_input, prediction_output, result_text, confidence_chart]
    )

# ============================================================================
# STEP 7: Launch the Application
# ============================================================================
print("\n" + "="*70)
print("LAUNCHING BRAIN TUMOR DETECTION WEB APP")
print("="*70)
print("\n✅ Model is ready!")
print("✅ Starting web interface...\n")

# Launch with share=True to get public URL
demo.launch(
    share=True,
    debug=True,
    server_name="0.0.0.0",
    server_port=7860
)

print("\n✅ Application is running!")
print("📱 Share the public URL with others to test the model")
print("⏹️ Press Ctrl+C in the terminal to stop")