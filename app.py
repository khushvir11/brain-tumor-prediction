import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
from streamlit_lottie import st_lottie

# --- LOTTIE ANIMATION LOADER ---
def load_lottieurl(url: str):
    """Fetches a Lottie JSON from the given URL."""
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide", # Use wide layout for a more advanced look
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR ADVANCED UI ---
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stApp {
        background-color: #f0f2f6;
    }
    .st-emotion-cache-1y4p8pa {
        max-width: 95%;
    }
    .block-container {
        padding-top: 2rem;
    }
    .st-emotion-cache-16txtl3 {
        padding: 1.5rem;
    }
    div[data-testid="stVerticalBlock"] {
        gap: 0.5rem;
    }
    div.stButton > button {
        width: 100%;
        border-radius: 20px;
    }
    div[data-testid="stFileUploader"] {
        border: 2px dashed #4B8BF5;
        background-color: #f7faff;
        border-radius: 10px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    """
    Loads the machine learning model from a file.
    """
    try:
        model = tf.keras.models.load_model('Brain_tumor_trainedmodel.h5')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please ensure a 'model.h5' file is in the correct directory.")
        return None
    return model

# Load the model
model = load_model()

# --- PREDICTION FUNCTION ---
def predict(image: Image.Image):
    """
    Takes a PIL image, preprocesses it, and returns the model's prediction.
    """
    if model is None:
        return "Model not loaded", 0.0

    target_size = (224, 224)
    img = image.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    confidence = predictions[0][0]
    
    if confidence > 0.5:
        return "Tumor Detected", float(confidence)
    else:
        return "No Tumor Detected", 1 - float(confidence)

# --- SIDEBAR ---
with st.sidebar:
    st.title("About")
    st.info(
        "This application uses a deep learning model to predict the presence of a brain tumor "
        "from MRI scans. It is an educational tool and not a substitute for professional "
        "medical advice."
    )
    lottie_brain_url = "https://lottie.host/81b2559a-1171-424a-9310-f16616a90899/E9dI0lfan8.json"
    lottie_brain = load_lottieurl(lottie_brain_url)
    if lottie_brain:
        st_lottie(lottie_brain, speed=1, height=200, key="sidebar_brain")
    st.header("How It Works")
    st.markdown("""
    1.  **Upload MRI:** Select a brain MRI scan image.
    2.  **AI Analysis:** The model processes the image to identify patterns.
    3.  **Get Result:** View the prediction and the model's confidence score.
    """)

# --- MAIN UI LAYOUT ---
st.title("Advanced Brain Tumor Detection AI 🧠")
st.markdown("Upload a brain MRI scan, and the AI will analyze it for potential tumors.")

col1, col2 = st.columns([1.5, 1], gap="large")

with col1:
    with st.container(border=True):
        st.subheader("Upload & Analyze")
        uploaded_file = st.file_uploader(
            "Choose an MRI image...", 
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded MRI Scan', use_container_width=True)

        analyze_button = st.button('Analyze Image', type="primary", use_container_width=True)

with col2:
    with st.container(border=True):
        st.subheader("🤖 Analysis Result")
        if analyze_button:
            if uploaded_file is not None:
                with st.spinner('Analyzing the image...'):
                    prediction, confidence = predict(image)
                
                if prediction == "Tumor Detected":
                    st.error(f"**Result:** {prediction}")
                elif prediction == "No Tumor Detected":
                    st.success(f"**Result:** {prediction}")
                
                if confidence > 0:
                    st.info(f"**Confidence:** {confidence:.2%}")
                
                st.warning("Disclaimer: This is not a medical diagnosis.", icon="⚠️")
            else:
                st.warning("Please upload an image first.", icon="📁")
        else:
            st.info("The analysis results will appear here after you upload an image and click 'Analyze'.")

st.markdown("---")
st.markdown("<div style='text-align: center;'>App powered by Streamlit & TensorFlow</div>", unsafe_allow_html=True)

