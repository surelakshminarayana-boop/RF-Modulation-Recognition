import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="RF Modulation Recognition System",
    layout="wide"
)

# ----------------------------------
# CONSTANTS
# ----------------------------------
MODS = [
    "BPSK", "QPSK", "8PSK", "QAM16", "QAM64",
    "GFSK", "PAM4", "AM-DSB", "AM-SSB", "WBFM", "CPFSK"
]

MODEL_PATH = "models/cldnn_rml2016_best.h5"

# ----------------------------------
# LOAD MODEL (with error handling)
# ----------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found: {MODEL_PATH}")
        st.error("Ensure 'models/cldnn_rml2016_best.h5' is in your repository.")
        st.stop()
    
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Model load failed: {str(e)}")
        st.stop()

# Load model early to catch errors
try:
    model = load_model()
except:
    st.stop()

# ----------------------------------
# SIGNAL FUNCTIONS
# ----------------------------------
def normalize_iq(x):
    p = np.mean(x**2, axis=(1, 2), keepdims=True)
    return x / np.sqrt(p + 1e-8)

def add_awgn(iq, snr_db):
    sig_p = np.mean(iq**2)
    snr = 10**(snr_db / 10)
    noise_p = sig_p / snr
    noise = np.sqrt(noise_p) * np.random.randn(*iq.shape)
    return iq + noise

# ----------------------------------
# HEADER
# ----------------------------------
st.markdown(
    """
    <h1 style='text-align:center;'>🛰️ RF Modulation Recognition System</h1>
    <p style='text-align:center; color:gray;'>
    Deep Learning-based Automatic Modulation Classification (AMC)
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ==================================
# MAIN GRID (2 COLUMNS)
# ==================================
left, right = st.columns(2)

# ==================================
# LEFT CARD — SIGNAL GENERATOR
# ==================================
with left:
    st.subheader("📡 Signal Generator")

    mod_type = st.selectbox(
        "Modulation Type",
        MODS
    )

    snr = st.slider(
        "SNR (dB)",
        min_value=-20,
        max_value=20,
        value=10,
        step=1
    )

    sample_len = st.select_slider(
        "Sample Length",
        options=[128, 256, 512, 1024],
        value=128
    )

    generate = st.button("Generate Signal", type="primary")

    if generate:
        # Simple synthetic signal (demo purpose)
        t = np.arange(sample_len)
        
        # Generate different modulation patterns
        if mod_type == "BPSK":
            i = np.cos(2 * np.pi * 0.05 * t)
            q = np.zeros_like(i)
        elif mod_type == "QPSK":
            i = np.cos(2 * np.pi * 0.05 * t)
            q = np.sin(2 * np.pi * 0.05 * t)
        elif mod_type == "8PSK":
            i = np.cos(2 * np.pi * 0.05 * t + np.pi/8 * (t % 8))
            q = np.sin(2 * np.pi * 0.05 * t + np.pi/8 * (t % 8))
        else:  # Default QAM-like
            i = np.cos(2 * np.pi * 0.05 * t)
            q = np.sin(2 * np.pi * 0.05 * t)
            
        iq = np.stack([i, q], axis=1)  # shape: (T, 2)
        iq = normalize_iq(iq)  # Normalize first
        iq = add_awgn(iq, snr)  # Then add noise

        # store in session_state
        st.session_state["iq"] = iq
        st.session_state["snr"] = snr
        st.session_state["mod_type"] = mod_type

        # Time-domain plot (first 200 samples)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(iq[:200, 0], label="I", alpha=0.8)
        ax.plot(iq[:200, 1], label="Q", alpha=0.8)
        ax.set_title(f"Time-Domain IQ Signal ({mod_type} @ {snr} dB)")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# ==================================
# RIGHT CARD — CLASSIFIER
# ==================================
with right:
    st.subheader("🧠 Modulation Classifier")

    if "iq" in st.session_state:
        iq = st.session_state["iq"]
        snr_val = st.session_state.get("snr", 0)
        mod_type = st.session_state.get("mod_type", "Unknown")

        # Prepare for model (assuming model expects shape [N, T, 2])
        x = normalize_iq(iq[np.newaxis, ...])

        with st.spinner("Running inference..."):
            preds = model.predict(x, verbose=0)[0]
            pred_idx = np.argmax(preds)

        tabs = st.tabs(["🎯 Predictions", "📊 I/Q Constellation", "ℹ️ Model Info"])

        # -------- TAB 1: PREDICTIONS --------
        with tabs[0]:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.metric(
                    label="Primary Classification", 
                    value=f"**{MODS[pred_idx]}**",
                    delta=f"({preds[pred_idx]*100:.1f}%)"
                )
            with col2:
                st.info(f"True: {mod_type}")
            
            st.markdown("**Confidence Scores:**")
            progress_container = st.container()
            for i, (m, p) in enumerate(sorted(zip(MODS, preds), key=lambda x: x[1], reverse=True)[:5]):
                with progress_container:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        progress = st.progress(0)
                        progress.progress(float(p))
                    with col2:
                        st.caption(f"{m}: {p*100:.1f}%")

        # -------- TAB 2: CONSTELLATION --------
        with tabs[1]:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(iq[:, 0], iq[:, 1], s=15, alpha=0.7, c='blue')
            ax.set_xlabel("In-phase (I)")
            ax.set_ylabel("Quadrature (Q)")
            ax.set_title(f"I/Q Constellation Diagram\n({mod_type} @ {snr_val} dB)")
            ax.grid(True, alpha=0.3)
            ax.set_aspect("equal")
            plt.tight_layout()
            st.pyplot(fig)

        # -------- TAB 3: MODEL INFO --------
        with tabs[2]:
            st.markdown("""
            **Model Details:**
            - **Architecture**: CLDNN (Convolutional + LSTM + Dense Network)
            - **Dataset**: RML2016.10a
            - **Input Shape**: (128, 2) IQ samples
            - **Classes**: 11 modulation types
            - **Deployment**: Streamlit Cloud (CPU inference)
            
            **Status**: ✅ Ready for inference
            """)

    else:
        st.info("👈 Generate a signal on the left to see predictions and visualizations.")
