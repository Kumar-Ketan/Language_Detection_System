import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Language Detection App",
    page_icon="🌍",
    layout="centered"
)

# ================== GLOBAL CSS ==================
st.markdown("""
<style>
/* Background */
.stApp {
    background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
}

/* Center main container */
.main > div {
    max-width: 700px;
}

/* Card container */
.card {
    background-color: white;
    padding: 35px;
    border-radius: 18px;
    box-shadow: 0px 6px 25px rgba(0,0,0,0.12);
    margin-top: 40px;
}

/* Title */
.app-title {
    text-align: center;
    font-size: 2.2rem;
    font-weight: 700;
    color: #2c3e50;
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #6c757d;
    margin-bottom: 25px;
}

/* Button styling */
div.stButton > button {
    background-color: #4b6cb7;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: 600;
}

/* Text area */
textarea {
    border-radius: 10px !important;
}

/* Result box */
.result-box {
    background-color: #f8f9fa;
    padding: 20px;
    border-radius: 12px;
    margin-top: 25px;
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
    return label_encoder.inverse_transform([class_index])[0], probs[class_index]

# ================== UI LAYOUT ==================

# Card Start
st.markdown('<div class="card">', unsafe_allow_html=True)

st.markdown('<div class="app-title">🌍 Language Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect the language of any sentence using Machine Learning</div>', unsafe_allow_html=True)

st.write("")

user_text = st.text_area(
    "✍️ Enter your text below",
    height=130,
    placeholder="Example: यह एक अच्छा दिन है | This is a beautiful day | C'est une belle journée"
)

if st.button("🔍 Detect Language"):
    if user_text.strip() == "":
        st.warning("⚠️ Please enter some text to detect the language.")
    else:
        with st.spinner("Detecting language..."):
            language, confidence = predict_language(user_text)

        # Result Box
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.success(f"✅ Predicted Language: **{language}**")
        st.write(f"📈 Confidence: **{confidence:.3f}**")

        # Confidence progress bar (nice UX touch)
        st.progress(float(confidence))
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # Card End

# ================== SIDEBAR (UX BOOST) ==================
st.sidebar.header("ℹ️ About This App")
st.sidebar.write("""
- Built using **TensorFlow & RNN**
- Supports multiple languages  
- Deployed on **Streamlit Cloud**  
- Designed for portfolio showcase  
""")

st.sidebar.markdown("---")
st.sidebar.write("👨‍💻 Developer: Kumar Ketan")

