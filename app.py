import streamlit as st
import sounddevice as sd
import numpy as np

from src.audio_features import extract_feature_vector_from_array
from src.model_utils import load_model, predict_probabilities

st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎤")
st.title("Speech Emotion Recognition")

MODEL_PATH = "models/emotion_model.pkl"
METADATA_PATH = "models/emotion_labels.json"
SAMPLE_RATE = 16000
DURATION_SECONDS = 2.0

if "model" not in st.session_state:
    try:
        st.session_state.model, st.session_state.index_to_label = load_model(MODEL_PATH, METADATA_PATH)
    except Exception as exc:
        st.session_state.model = None
        st.session_state.error = str(exc)

if st.session_state.get("model") is None:
    st.error("Train the model first with: python train.py --data-dir data")
    st.stop()

if st.button("Record emotion"):
    st.info("Recording for 2 seconds. Speak clearly into the microphone.")
    recording = sd.rec(int(DURATION_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    audio = recording[:, 0]
    features = extract_feature_vector_from_array(audio, SAMPLE_RATE)
    probabilities = predict_probabilities(st.session_state.model, features)

    label_scores = sorted(
        zip(st.session_state.index_to_label.values(), probabilities),
        key=lambda item: item[1],
        reverse=True,
    )

    st.subheader("Prediction")
    for label, score in label_scores:
        st.write(f"{label}: {score * 100:.1f}%")

    top_label, top_score = label_scores[0]
    st.success(f"Predicted emotion: {top_label} ({top_score * 100:.1f}%)")
