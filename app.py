import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import pandas as pd

# -----------------------
# CONFIG
# -----------------------
st.set_page_config(page_title="Traffic Sign AI", layout="centered")

st.title("Traffic Sign Recognition AI")
st.write("Upload an image and get prediction + confidence")

# -----------------------
# LOAD MODEL
# -----------------------
model = tf.keras.models.load_model("traffic_sign_model.keras")

# -----------------------
# CLASS LABELS
# -----------------------
class_labels = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles > 3.5 tons",
    11: "Right-of-way at intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles > 3.5 tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Ice/snow",
    31: "Wild animals crossing",
    32: "End of all limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Straight or right",
    37: "Straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing vehicles > 3.5 tons"
}

# -----------------------
# PREPROCESS
# -----------------------
def preprocess(img):
    img = cv2.resize(img, (32, 32))
    img = img.astype(np.float32) / 255.0
    return img

# -----------------------
# UI
# -----------------------
uploaded_file = st.file_uploader("Upload Traffic Sign Image", type=["jpg", "png", "jpeg"])

if uploaded_file:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        st.image(img, channels="BGR", use_container_width=True)

    # preprocess
    img_resized = cv2.resize(img, (32, 32))
    img_norm = img_resized.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_norm, axis=0)

    # prediction
    preds = model.predict(img_input)[0]
    pred_class = np.argmax(preds)
    confidence = np.max(preds)

    with col2:
        st.subheader("Prediction")
        st.success(class_labels[pred_class])
        st.info(f"Confidence: {confidence*100:.2f}%")

    # -----------------------
    # TOP 5
    # -----------------------
    st.subheader("Top 5 Predictions")

    top5_idx = np.argsort(preds)[-5:][::-1]

    df = pd.DataFrame({
        "Class": [class_labels[i] for i in top5_idx],
        "Confidence": preds[top5_idx]
    })

    st.bar_chart(df.set_index("Class"))

else:
    st.warning("Upload an image to start prediction")