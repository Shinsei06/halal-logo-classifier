import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title="Halal Logo Classifier")

# Load the YOLO model
model = YOLO("best (4).pt")  # Replace with your actual YOLOv8 model file

# App title and description
st.title("🕌 Halal Logo Classifier")
st.write("Check if a halal logo is **Authentic** or **Fake** by uploading an image or using your device camera.")

# Input method selection
input_method = st.radio("Choose input method:", ["📁 Upload Image", "📷 Camera Snapshot"])

image = None

# Handle image input
if input_method == "📁 Upload Image":
    uploaded_file = st.file_uploader("Upload Halal Logo Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

elif input_method == "📷 Camera Snapshot":
    camera_image = st.camera_input("Take a photo of the halal logo")
    if camera_image:
        image = Image.open(camera_image).convert("RGB")

# Process and predict
if image:
    with st.spinner("🔍 Classifying..."):
        # Run YOLO prediction
        results = model.predict(image)

        # Convert image to OpenCV format for annotation
        img_np = np.array(image)
        img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        detected_labels = []

        for box in results[0].boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[class_id]
            detected_labels.append(label)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = (0, 255, 0) if "authentic" in label.lower() else (0, 0, 255)
            cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_cv2, f"{label} {conf:.2%}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Convert back to RGB for display
        result_image = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        st.image(result_image, caption="🔎 Detected Results", use_container_width=True)

        # Show result summary
        if detected_labels:
            if any("authentic" in lbl.lower() for lbl in detected_labels):
                st.success("✅ **Authentic** halal logo detected.")
            else:
                st.error("❌ No **Authentic** logos found. Possibly **Fake**.")
        else:
            st.warning("⚠️ No halal logo detected. Please try another image.")
