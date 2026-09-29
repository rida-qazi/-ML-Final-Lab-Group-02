import streamlit as st
import tempfile
import os

from src.predict import predict_audio


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
    "Upload an audio recording and the model will "
    "predict the speaker's emotion."
)


# --------------------------------------------------
# Audio upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an audio file",
    type=["wav"]
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if uploaded_file is not None:

    st.audio(
        uploaded_file,
        format="audio/wav"
    )

    if st.button("Predict Emotion"):

        # Create a temporary WAV file
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as temp_file:

            temp_file.write(
                uploaded_file.getbuffer()
            )

            temp_audio_path = temp_file.name

        try:

            # Use the existing ML inference pipeline
            result = predict_audio(
                temp_audio_path
            )

            # ------------------------------------------
            # Prediction
            # ------------------------------------------

            st.subheader("Prediction")

            st.success(
                f"Predicted emotion: "
                f"**{result['prediction'].upper()}**"
            )

            # ------------------------------------------
            # Probabilities
            # ------------------------------------------

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

            # ------------------------------------------
            # Inference latency
            # ------------------------------------------

            st.subheader("Inference")

            st.write(
                f"Model inference time: "
                f"**{result['inference_time_ms']:.2f} ms**"
            )

        finally:

            # Remove temporary audio file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)