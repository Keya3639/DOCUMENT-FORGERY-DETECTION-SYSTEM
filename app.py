import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from io import BytesIO
import pandas as pd
import os

os.environ["PATH"] += os.pathsep + r"C:\Users\HP\Downloads\Release-25.12.0-0\poppler-25.12.0\Library\bin"

from pdf2image import convert_from_bytes
from docx import Document

# ----------------------------
# Constants
# ----------------------------
IMG_SIZE = 256
MODEL_PATH = "forgery_detector.h5"
THRESHOLD = 0.6   # <-- new threshold
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png"]
SUPPORTED_PDF = ["pdf"]
SUPPORTED_DOC = ["docx", "txt"]

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_forgery_model():
    return load_model(MODEL_PATH)

model = load_forgery_model()

# ----------------------------
# Session State for History
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["Filename", "Prediction", "Confidence"])

# ----------------------------
# Helper Functions
# ----------------------------
def preprocess_image(img):
    """Resize, normalize contrast, normalize and reshape grayscale image"""
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # 🔥 Improve contrast — helps with scanned pages
    try:
        img_resized = cv2.equalizeHist(img_resized)
    except:
        pass

    img_resized = img_resized / 255.0
    img_resized = img_resized.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    return img_resized


def predict_forgery(img):
    """Predict whether image is Original or Forged"""
    batch = preprocess_image(img)

    pred_value = model.predict(batch)[0][0]

    # model predicts probability of FORGED
    if pred_value < THRESHOLD:
        label = "ORIGINAL"
        confidence = 1 - pred_value
    else:
        label = "FORGED"
        confidence = pred_value

    return label, confidence, batch, pred_value


def create_downloadable_result(filename, label, confidence):
    result_str = f"File: {filename}\nPrediction: {label}\nConfidence: {confidence:.2f}"
    return BytesIO(result_str.encode())


def process_pdf(file_bytes):
    pages = convert_from_bytes(file_bytes)
    images = []
    for page in pages:
        img = np.array(page.convert('L'))  # grayscale
        images.append(img)
    return images


def process_docx(file_bytes):
    with BytesIO(file_bytes) as f:
        doc = Document(f)
        text = "\n".join([p.text for p in doc.paragraphs])
    return text


def process_txt(file_bytes):
    return file_bytes.decode('utf-8')


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Advanced Document Forgery Detection", layout="wide")
st.title("📄 Advanced Document Forgery Detection System")

st.markdown(
    "Upload an image, PDF, or document. For images/PDF pages, the system predicts whether the document is ORIGINAL or FORGED."
)

uploaded_file = st.file_uploader(
    "Choose a file...",
    type=SUPPORTED_IMAGE_TYPES + SUPPORTED_PDF + SUPPORTED_DOC
)

if uploaded_file is not None:
    file_ext = uploaded_file.name.split('.')[-1].lower()
    file_bytes = uploaded_file.read()

    # ---------------- IMAGE ----------------
    if file_ext in SUPPORTED_IMAGE_TYPES:
        img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)

        if img is not None:
            with st.spinner("Processing image..."):
                label, confidence, resized, raw = predict_forgery(img)

            processed = (resized[0].reshape(IMG_SIZE, IMG_SIZE) * 255).astype(np.uint8)

            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption="Uploaded Image", use_column_width=True, channels="GRAY")
            with col2:
                st.image(processed, caption="Preprocessed Image", use_column_width=True, channels="GRAY")

            st.subheader("Prediction Result")
            st.write(f"**Raw model value (probability of FORGED):** `{raw:.3f}`")

            if label == "ORIGINAL":
                st.success(f"✅ Document is {label} ({confidence:.2f} confidence)")
            else:
                st.error(f"❌ Document is {label} ({confidence:.2f} confidence)")

            result_file = create_downloadable_result(uploaded_file.name, label, confidence)

            st.download_button(
                label="💾 Download Result",
                data=result_file,
                file_name=f"{uploaded_file.name}_result.txt",
                mime="text/plain"
            )

            st.session_state.history = pd.concat(
                [
                    st.session_state.history,
                    pd.DataFrame([[uploaded_file.name, label, confidence]],
                                 columns=["Filename", "Prediction", "Confidence"])
                ],
                ignore_index=True
            )
        else:
            st.error("⚠️ Could not read the image.")

    # ---------------- PDF ----------------
    elif file_ext in SUPPORTED_PDF:
        with st.spinner("Processing PDF..."):
            images = process_pdf(file_bytes)

            for i, img in enumerate(images):
                label, confidence, resized, raw = predict_forgery(img)

                st.subheader(f"Page {i+1}")
                processed = (resized[0].reshape(IMG_SIZE, IMG_SIZE) * 255).astype(np.uint8)

                col1, col2 = st.columns(2)
                with col1:
                    st.image(img, caption="PDF Page Image", use_column_width=True, channels="GRAY")
                with col2:
                    st.image(processed, caption="Preprocessed Page Image", use_column_width=True, channels="GRAY")

                st.write(f"Raw probability (FORGED): `{raw:.3f}`")
                st.write(f"Prediction: **{label}** ({confidence:.2f} confidence)")

                st.session_state.history = pd.concat(
                    [
                        st.session_state.history,
                        pd.DataFrame([[f"{uploaded_file.name} - Page {i+1}", label, confidence]],
                                     columns=["Filename", "Prediction", "Confidence"])
                    ],
                    ignore_index=True
                )

    # ---------------- TEXT DOCS ----------------
    elif file_ext in ["docx", "txt"]:
        text = process_docx(file_bytes) if file_ext == "docx" else process_txt(file_bytes)
        st.subheader("Document Text Preview")
        st.text_area("Text Content", text, height=300)

# ----------------------------
# History & Confidence Plot
# ----------------------------
if not st.session_state.history.empty:
    st.markdown("---")
    st.subheader("History of Uploaded Documents")
    st.dataframe(st.session_state.history)

    st.subheader("Confidence over Time")
    fig, ax = plt.subplots()
    ax.plot(st.session_state.history.index + 1, st.session_state.history["Confidence"], marker="o")
    ax.set_xlabel("Upload Index")
    ax.set_ylabel("Confidence")
    ax.set_title("Prediction Confidence Trend")
    st.pyplot(fig)

    if st.button("🗑️ Clear History"):
        st.session_state.history = pd.DataFrame(columns=["Filename", "Prediction", "Confidence"])
        st.experimental_rerun()

st.markdown("---")
st.markdown("Developed by **Keya Das** | Advanced Document Forgery Detection System")
