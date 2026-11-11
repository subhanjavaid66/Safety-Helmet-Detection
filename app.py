
from pathlib import Path

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd


# Page configuration
st.set_page_config(
    page_title="Safety Helmet Detection",
    page_icon="🦺",
    layout="wide"
)


# Load trained model
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "models" / "best.pt"

    if not model_path.exists():
        st.error(f"Model file not found: {model_path}")
        st.stop()

    return YOLO(str(model_path))


model = load_model()


# Application heading
st.title("Safety Helmet Detection System")

st.write(
    "Upload an image to detect people, helmets, "
    "and people without helmets."
)

st.divider()


# Image uploader
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


# Confidence threshold
confidence = st.slider(
    "Detection confidence",
    min_value=0.10,
    max_value=1.00,
    value=0.25,
    step=0.05
)


# Run detection
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(image, use_container_width=True)

    if st.button("Detect Objects", type="primary"):

        with st.spinner("Detecting objects..."):

            results = model.predict(
                source=image,
                conf=confidence,
                imgsz=640,
                verbose=False
            )

            result = results[0]

            # Convert annotated image from BGR to RGB
            annotated_image = result.plot()
            annotated_image = annotated_image[:, :, ::-1]

            st.success("Detection completed!")

            # Display original and detected images
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)

            with col2:
                st.subheader("Detection Results")
                st.image(
                    annotated_image,
                    use_container_width=True
                )

            # Extract detection details
            detections = []

            for box in result.boxes:

                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                conf = float(box.conf[0])

                detections.append({
                    "Detected Object": class_name,
                    "Confidence (%)": round(conf * 100, 2)
                })

            # Detection summary
            st.divider()
            st.subheader("Detection Summary")

            if detections:

                df = pd.DataFrame(detections)

                col1, col2 = st.columns(2)

                col1.metric(
                    "Total Objects",
                    len(detections)
                )

                col2.metric(
                    "Unique Classes",
                    df["Detected Object"].nunique()
                )

                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )

            else:
                st.warning("No objects detected in this image.")
