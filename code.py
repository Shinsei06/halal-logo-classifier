import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Load the model
model = YOLO("best (1).pt")  # Make sure 'best.pt' is in the same folder

# Page config
st.set_page_config(page_title="Halal Logo Classifier", layout="centered")
st.title("🟢 Halal Logo Classifier")
st.write("Upload an image to check if the halal logo is authentic or fake.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict
    with st.spinner("Classifying..."):
        results = model.predict(image)

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls]

                if label.lower() == "authentic":
                    st.success(f"✅ Authentic Logo (Confidence: {conf:.2%})")
                else:
                    st.error(f"❌ Fake Logo (Confidence: {conf:.2%})")
