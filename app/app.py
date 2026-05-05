import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from model.predict import predict_image
from model.gradcam import get_gradcam
from PIL import Image
import time

# Page config
st.set_page_config(page_title="Plant AI", layout="centered")

# Custom CSS
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
.title {
    text-align: center;
    font-size: 45px;
    font-weight: bold;
    color: #22c55e;
}
.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 25px;
}
.result-box {
    background: #111827;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #1f2937;
    margin-top: 20px;
}
.good {
    color: #22c55e;
    font-weight: bold;
}
.bad {
    color: #ef4444;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title">🌿 Plant Disease Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered system for detecting plant diseases and suggesting treatments</div>', unsafe_allow_html=True)

# Upload
uploaded_file = st.file_uploader("📤 Upload a plant leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.convert("RGB")

    st.image(image, use_container_width=True)

    # Save temp file
    image.save("temp.jpg")

    # Loading
    with st.spinner("🔍 Analyzing plant health..."):
        time.sleep(1)
        disease, confidence, info = predict_image("temp.jpg")
        gradcam_img = get_gradcam("temp.jpg")

    # Color logic
    status_class = "good" if disease == "healthy" else "bad"

    # Grad-CAM output
    st.markdown("### 🔍 Model Focus Area")
    st.image(
        gradcam_img.astype("uint8"),
        caption="Highlighted regions show model attention",
        use_container_width=True
    )

    # Result box
    st.markdown(f"""
    <div class="result-box">
        <h3>🦠 Disease: <span class="{status_class}">{disease.replace('_',' ').upper()}</span></h3>
        <h4>📊 Confidence: {round(confidence*100,2)}%</h4>
    </div>
    """, unsafe_allow_html=True)

    # 🔥 Confidence interpretation (NEW)
    if confidence > 0.8:
        st.success("High confidence prediction")
    elif confidence > 0.5:
        st.warning("Moderate confidence prediction")
    else:
        st.error("Low confidence — try a clearer image")

    # Remedies
    st.markdown("### 💊 Treatment")
    for r in info["remedy"]:
        st.success(r)

    # Prevention
    st.markdown("### 🛡️ Prevention")
    for p in info["prevention"]:
        st.info(p)

    st.success("✅ Analysis Complete")

    # Notes (professional touch)
    st.markdown("### ⚠️ Notes")
    st.write("""
    - Works best on clear images of a single leaf  
    - Performance may vary on real-world images  
    - Background noise can affect predictions  
    """)