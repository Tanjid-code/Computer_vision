import streamlit as st
from ultralytics import YOLO
from PIL import Image
import requests
from io import BytesIO
import numpy as np

# =========================================
# LOAD MODEL
# =========================================
model = YOLO("best.pt")

st.set_page_config(page_title="Cat vs Dog Detector", layout="centered")

st.title("🐱🐶 Cat vs Dog Detector (YOLO)")

st.write("Upload an image or paste an image URL.")

# =========================================
# INPUT OPTION
# =========================================
option = st.radio("Choose input type:", ["Upload Image", "Image URL"])

img = None

# -----------------------------------------
# Upload image
# -----------------------------------------
if option == "Upload Image":
    file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

    if file:
        img = Image.open(file).convert("RGB")

# -----------------------------------------
# Image URL
# -----------------------------------------
elif option == "Image URL":
    url = st.text_input("Paste image URL")

    if url:
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content)).convert("RGB")
        except:
            st.error("Invalid URL or image could not be loaded")

# =========================================
# PREDICTION
# =========================================
if img is not None:

    st.image(img, caption="Input Image", use_container_width=True)

    results = model.predict(img, conf=0.25)
    r = results[0]

    # image with bounding boxes
    plotted_img = r.plot()

    st.image(plotted_img, caption="Detection Result", use_container_width=True)

    # =====================================
    # DISPLAY RESULTS
    # =====================================
    st.subheader("Detections:")

    if r.boxes is not None and len(r.boxes) > 0:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            label = model.names[cls_id]

            st.write(f"👉 {label} | Confidence: {conf:.2f}")
    else:
        st.write("No objects detected")