import streamlit as st
import tempfile
import os
import sys
from pathlib import Path
import hashlib


# --------------------------------------------------
# Project path
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------
# Project imports
# --------------------------------------------------

from src.predict import predict_audio
from src.preprocessing import extract_features


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="centered"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🎙️ Speech Emotion Recognition")

st.write(
    "Upload an audio file or record your voice "
    "to predict the speaker's emotion."
)


# --------------------------------------------------
# Input selection
# --------------------------------------------------

input_method = st.radio(
    "Choose an input method:",
    ["Upload Audio", "Record Audio"],
    horizontal=True
)


audio_file = None


# --------------------------------------------------
# Upload audio
# --------------------------------------------------

if input_method == "Upload Audio":

    audio_file = st.file_uploader(
        "Upload a WAV audio file",
        type=["wav"]
    )


# --------------------------------------------------
# Record audio
# --------------------------------------------------

else:

    st.write("Record your voice using your microphone.")

    audio_file = st.audio_input(
        "Record audio",
        sample_rate=16000
    )


# --------------------------------------------------
# Process audio
# --------------------------------------------------

if audio_file is not None:


    audio_bytes = audio_file.getvalue()

    st.write("Audio size:", len(audio_bytes), "bytes")

    audio_hash = hashlib.md5(audio_bytes).hexdigest()

    st.write("Audio hash:", audio_hash)

    st.audio(
        audio_file,
        format="audio/wav"
    )

    if st.button("Predict Emotion"):

        # ------------------------------------------
        # Save audio temporarily
        # ------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as temp_file:

            temp_file.write(
                audio_file.getvalue()
            )

            temp_audio_path = temp_file.name

        # Debug feature extraction
        debug_features = extract_features(temp_audio_path)

        st.write("Feature shape:", debug_features.shape)
        st.write("Feature mean:", debug_features.mean())
        st.write("Feature standard deviation:", debug_features.std())
        st.write("First 10 features:", debug_features[:10])


        try:

            # --------------------------------------
            # Run existing ML inference pipeline
            # --------------------------------------

            result = predict_audio(
                temp_audio_path
            )
            st.write("DEBUG RESULT:")
            st.write(result)


            # --------------------------------------
            # Prediction
            # --------------------------------------

            st.subheader("Prediction")

            st.success(
                f"Predicted emotion: "
                f"**{result['prediction'].upper()}**"
            )


            # --------------------------------------
            # Emotion probabilities
            # --------------------------------------

            st.subheader("Emotion Probabilities")

            for emotion, probability in result[
                "probabilities"
            ].items():

                st.write(
                    f"**{emotion.capitalize()}**: "
                    f"{probability * 100:.2f}%"
                )

                st.progress(
                    float(probability)
                )


            # --------------------------------------
            # Inference latency
            # --------------------------------------

            st.subheader("Inference")

            st.write(
                f"Model inference time: "
                f"**{result['inference_time_ms']:.2f} ms**"
            )


        except Exception as e:

            st.error(
                f"Unable to process the audio: {e}"
            )


        finally:

            # --------------------------------------
            # Delete temporary file
            # --------------------------------------

            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)