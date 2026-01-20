import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Language Detection System",
    page_icon="🔴",
    layout="centered"
)

# ================== GLOBAL DARK RED THEME CSS ==================
st.markdown("""
<style>

/* ===== GLOBAL BACKGROUND ===== */
.stApp {
    background: radial-gradient(circle at top, #2b0a0a 0%, #050505 60%);
}

/* Center main container */
.main > div {
    max-width: 780px;
}

/* ===== CARD CONTAINER ===== */
.card {
    background: linear-gradient(135deg, #0f0f0f, #1a0a0a);
    padding: 40px;
    border-radius: 20px;
    box-shadow: 0px 10px 40px rgba(255, 0, 0, 0.25);
    margin-top: 50px;
    border: 1px solid rgba(255, 0, 0, 0.15);
}

/* ===== TITLE ===== */
.app-title {
    text-align: center;
    font-size: 2.6rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: 1px;
    margin-bottom: 8px;
}

/* Red dot */
.red-dot {
    color: #ff3b3b;
}

/* ===== SUBTITLE ===== */
.subtitle {
    text-align: center;
    color: #c7c7c7;
    margin-bottom: 35px;
    font-size: 1rem;
}

/* ===== ENTER TEXT LABEL (BIGGER + RED) ===== */
.input-label {
    color: #ff3b3b;
    font-size: 1.15rem;
    font-weight: 700;
    margin-bottom: 6px;
}

/* ===== TEXT AREA ===== */
textarea {
    background-color: #050505 !important;
    color: #f5f5f5 !important;
    border-radius: 14px !important;
    border: 1px solid #3a0f0f !important;
    font-size: 16px !important;
}

/* Placeholder */
textarea::placeholder {
    color: #9a9a9a !important;
}

/* ===== BUTTON ===== */
div.stButton > button {
    background: linear-gradient(135deg, #ff2d2d, #c00000);
    color: white;
    border-radius: 12px;
    height: 3.2em;
    width: 100%;
    font-size: 18px;
    font-weight: 700;
    border: none;
    transition: all 0.3s ease;
}

div.stButton > button:hover {
    transform: scale(1.04);
    box-shadow: 0px 0px 25px rgba(255, 45, 45, 0.7);
}

/* ===== RESULT BOX ===== */
.result-box {
    background: linear-gradient(135deg, #0a0a0a, #140606);
    padding: 25px;
    border-radius: 14px;
    margin-top: 30px;
    border: 1px solid rgba(255, 0, 0, 0.25);
}

/* Predicted language text in RED */
.predicted-text {
    color: #ff3b3b;
    font-size: 1.6rem;
    font-weight: 900;
}

/* Confidence text */
.conf-text {
    color: #dddddd;
    margin-top: 8px;
}

/* Progress bar color */
.stProgress > div > div > div {
    background-color: #ff2d2d;
}

</style>
""", unsafe_allow_html=True)

# ================== LOAD MODEL & TOKENIZER ==================
@st.cache_resource
def load_artifacts():
    model = load_model("saved_model/simple_rnn_model.h5")
    with open("saved_model/tokenizer.pkl", "rb") as f:
        tokenizer, label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = load_artifacts()

# ================== PREDICTION FUNCTION ==================
def predict_language(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=80)
    probs = model.predict(padded, verbose=0)[0]
    class_index = np.argmax(probs)
    language = label_encoder.inverse_transform([class_index])[0]
    confidence = probs[class_index]
    return language, confidence

# ================== UI LAYOUT ==================

st.markdown('<div class="card">', unsafe_allow_html=True)

# Title with red dot after Language
st.markdown(
    '<div class="app-title">Language<span class="red-dot">.</span> Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Detect the language of any sentence using Deep Learning</div>",
    unsafe_allow_html=True
)

# Bigger red label
st.markdown('<div class="input-label"> Enter your text below</div>', unsafe_allow_html=True)

user_text = st.text_area(
    "",
    height=150,
    placeholder="Example: यह एक अच्छा दिन है | This is a beautiful day | C'est une belle journée"
)

if st.button("🔍 Detect Language"):
    if user_text.strip() == "":
        st.warning("⚠️ Please enter some text to detect the language.")
    else:
        with st.spinner("Analyzing language..."):
            language, confidence = predict_language(user_text)

        st.markdown('<div class="result-box">', unsafe_allow_html=True)

        # Predicted Language in RED
        st.markdown(
            f"<div class='predicted-text'>Predicted Language: {language}</div>",
            unsafe_allow_html=True
        )

        st.markdown(
    f"<div class='conf-text'>Confidence: {confidence*100:.1f}%</div>",
    unsafe_allow_html=True
    )


        st.progress(float(confidence))

        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ================== SIDEBAR ==================
st.sidebar.header("ℹ️ About This App")
st.sidebar.write("""
- Built using **TensorFlow & RNN**
- Supports multiple languages  
- Dark Red Tech Theme  
- Deployed on **Streamlit Cloud**  
""")

st.sidebar.markdown("---")
st.sidebar.write("👨‍💻 Developer: **Kumar Ketan**")

