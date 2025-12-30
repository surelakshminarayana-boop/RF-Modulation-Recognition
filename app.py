import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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
# LOAD MODEL
# ----------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

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

    generate = st.button("Generate Signal")

    if generate:
        # Simple synthetic signal (demo purpose)
        t = np.arange(sample_len)
        i = np.cos(2 * np.pi * 0.05 * t)
        q = np.sin(2 * np.pi * 0.05 * t)
        iq = np.stack([i, q], axis=1)  # shape: (T, 2)

        iq = add_awgn(iq, snr)

        # store in session_state
        st.session_state["iq"] = iq

        # Time-domain plot
        fig, ax = plt.subplots()
        ax.plot(iq[:200, 0], label="I")
        ax.plot(iq[:200, 1], label="Q")
        ax.set_title("Time-Domain IQ Signal")
        ax.legend()
        st.pyplot(fig)

# ==================================
# RIGHT CARD — CLASSIFIER
# ==================================
with right:
    st.subheader("🧠 Modulation Classifier")

    tabs = st.tabs(["Predictions", "I/Q Constellation", "Model Info"])

    if "iq" in st.session_state:
        iq = st.session_state["iq"]

        # Prepare for model (assuming model expects shape [N, T, 2])
        x = normalize_iq(iq[np.newaxis, ...])

        preds = model.predict(x)[0]
        pred_idx = np.argmax(preds)

        # -------- TAB 1: PREDICTIONS --------
        with tabs[0]:
            st.success(f"Primary Classification: **{MODS[pred_idx]}**")

            for m, p in sorted(zip(MODS, preds), key=lambda x: x[1], reverse=True):
                st.progress(float(p), text=f"{m}: {p*100:.1f}%")

        # -------- TAB 2: CONSTELLATION --------
        with tabs[1]:
            fig, ax = plt.subplots()
            ax.scatter(iq[:, 0], iq[:, 1], s=8)
            ax.set_xlabel("I")
            ax.set_ylabel("Q")
            ax.set_title(f"Constellation @ {snr} dB")
            ax.grid(True)
            ax.set_aspect("equal")
            st.pyplot(fig)

        # -------- TAB 3: MODEL INFO --------
        with tabs[2]:
            st.markdown(
                """
                **Model Architecture:**
                CLDNN trained on RML2016.10a dataset for automatic modulation classification.
                """
            )
    else:
        with tabs[0]:
            st.info("Generate a signal on the left to see predictions.")
        with tabs[1]:
            st.info("Generate a signal on the left to see the constellation.")
        with tabs[2]:
            st.markdown("Model ready. Generate a signal to run inference.")
