import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import os
from datetime import datetime

MODEL_PATH = "models/cnn_model.h5"
LOG_FILE = "logs/predictions.csv"

os.makedirs("logs", exist_ok=True)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Dummy labels (replace later if dataset expands)
scanner_labels = ["Scanner_A", "Scanner_B", "Scanner_C"]

st.set_page_config(page_title="TraceFinder Scanner Detection", layout="centered")
st.title("📄 TraceFinder – Scanner Identification System")

uploaded_file = st.file_uploader("Upload a scanned document image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_resized = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img_resized / 255.0, axis=0)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100

    predicted_scanner = scanner_labels[predicted_index]

    st.success(f"🖨️ Predicted Scanner: **{predicted_scanner}**")
    st.info(f"📊 Confidence Score: **{confidence:.2f}%**")

    # Log prediction
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scanner": predicted_scanner,
        "confidence": round(confidence, 2)
    }

    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        df = pd.concat([df, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([log_entry])

    df.to_csv(LOG_FILE, index=False)

    st.download_button(
        "⬇️ Download Prediction Log",
        df.to_csv(index=False),
        file_name="predictions.csv",
        mime="text/csv"
    )
