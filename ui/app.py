import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from src.utils import load_models, predict_image_array
from src.data_loader import load_uploaded_image, bgr_to_rgb

st.set_page_config(
    page_title="AI Hazard Risk Detection Dashboard",
    page_icon="🦺",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
.title-text {
    font-size: 2rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 0.4rem;
    line-height: 1.25;
    white-space: normal;
    word-break: break-word;
    overflow-wrap: break-word;
}
.subtitle-text {
    font-size: 1.05rem;
    color: #d1d5db;
    margin-bottom: 1.5rem;
}
.section-header {
    font-size: 1.25rem;
    font-weight: 700;
    color: #ffffff;
    margin-top: 1rem;
    margin-bottom: 0.8rem;
}
.info-card {
    background-color: #111827;
    border: 1px solid #374151;
    padding: 1rem 1.2rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    color: #f9fafb;
}
.risk-low {
    background-color: #ecfdf5;
    border: 1px solid #10b981;
    padding: 1rem 1.2rem;
    border-radius: 12px;
    font-size: 1.1rem;
    font-weight: 700;
    color: #065f46;
}
.risk-medium {
    background-color: #fff7ed;
    border: 1px solid #f59e0b;
    padding: 1rem 1.2rem;
    border-radius: 12px;
    font-size: 1.1rem;
    font-weight: 700;
    color: #9a3412;
}
.risk-high {
    background-color: #fef2f2;
    border: 1px solid #ef4444;
    padding: 1rem 1.2rem;
    border-radius: 12px;
    font-size: 1.1rem;
    font-weight: 700;
    color: #991b1b;
}
.explanation-box {
    background-color: #0f172a;
    border-left: 5px solid #3b82f6;
    padding: 0.9rem 1rem;
    border-radius: 8px;
    color: #f8fafc;
}
.small-note {
    color: #d1d5db;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_models():
    return load_models()


try:
    detector, classifier = get_models()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


st.markdown(
    '<div class="title-text">🦺 AI Hazard Risk Detection Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle-text">'
    'Upload a workplace image to analyze PPE usage, estimate risk level, '
    'and visualize the model output.'
    '</div>',
    unsafe_allow_html=True
)


with st.sidebar:
    st.markdown("### System Overview")
    st.write("This interface analyzes workplace images using:")
    st.write("- YOLO-based PPE detection")
    st.write("- Structured feature extraction")
    st.write("- Learned risk classification")
    st.write("- Visual grounding")

    st.markdown("### Visual Classes")
    st.write("- Helmet")
    st.write("- Vest")
    st.write("- Gloves")
    st.write("- Goggles")

    st.markdown("### Risk Levels")
    st.write("🟢 Low")
    st.write("🟠 Medium")
    st.write("🔴 High")


st.markdown('<div class="section-header">Image Upload</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a workplace image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.markdown(
        '<div class="info-card">'
        '<b>How to use:</b><br>'
        '1. Upload a workplace or construction image.<br>'
        '2. Click <b>Analyze Risk</b>.<br>'
        '3. Review the risk label, explanation, and annotated output.'
        '</div>',
        unsafe_allow_html=True
    )
    st.stop()


try:
    image_bgr = load_uploaded_image(uploaded_file)
    image_rgb = bgr_to_rgb(image_bgr)
except Exception as e:
    st.error(f"Could not read uploaded image: {e}")
    st.stop()


analyze = st.button("Analyze Risk")

if not analyze:
    st.image(image_rgb, caption="Uploaded Image", width=450)
    st.stop()


with st.spinner("Analyzing image and generating risk assessment..."):
    try:
        output = predict_image_array(
            image_bgr=image_bgr,
            detector_model=detector,
            classifier_model=classifier
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()


risk_name = output["risk_name"]
explanation = output["explanation"]
features = output["features"]
grounded_rgb = bgr_to_rgb(output["grounded_bgr"])


st.markdown('<div class="section-header">Risk Analysis Results</div>', unsafe_allow_html=True)

if risk_name == "Low":
    st.markdown(
        f'<div class="risk-low">🟢 Final Risk Assessment: {risk_name.upper()}</div>',
        unsafe_allow_html=True
    )
elif risk_name == "Medium":
    st.markdown(
        f'<div class="risk-medium">🟠 Final Risk Assessment: {risk_name.upper()}</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        f'<div class="risk-high">🔴 Final Risk Assessment: {risk_name.upper()}</div>',
        unsafe_allow_html=True
    )

st.markdown(
    f'<div class="explanation-box"><b>Explanation:</b> {explanation}</div>',
    unsafe_allow_html=True
)


st.markdown('<div class="section-header">Image Comparison</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.image(image_rgb, caption="Original Image", width=400)

with col2:
    st.image(grounded_rgb, caption="Annotated Output", width=400)


st.markdown('<div class="section-header">Detected PPE Summary</div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Helmets", int(features["helmet_count"]))
m2.metric("Vests", int(features["vest_count"]))
m3.metric("Gloves", int(features["gloves_count"]))
m4.metric("Goggles", int(features["goggles_count"]))


with st.expander("Show Detailed Feature Values"):
    hidden_features = {
        "person_count",
        "helmet_ratio",
        "vest_ratio",
        "gloves_ratio",
        "goggles_ratio",
        "person_conf_mean"
    }

    filtered_feature_keys = [
        k for k in features.keys()
        if k not in hidden_features
    ]

    feature_df = {
        "Feature": filtered_feature_keys,
        "Value": [features[k] for k in filtered_feature_keys]
    }

    st.dataframe(feature_df, use_container_width=True)


st.markdown("---")
st.markdown(
    '<div class="small-note">'
    'This system is intended as an assistive safety screening tool. '
    'Predictions should support, not replace, human judgment in safety-critical environments.'
    '</div>',
    unsafe_allow_html=True
)