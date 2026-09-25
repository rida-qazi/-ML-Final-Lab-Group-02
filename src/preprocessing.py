#ridas code for navin
import os
import numpy as np
import pandas as pd
import librosa


# ============================================================
# PATHS
# ============================================================

# Project root = one level above the src folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Raw RAVDESS dataset
DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "RAVDESS"
)

# Processed dataset output
OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "ravdess_features.csv"
)


# ============================================================
# RAVDESS EMOTION LABELS
# ============================================================

EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(file_path):
    """
    Extract MFCC, Chroma and Mel-spectrogram features
    from one RAVDESS audio file.
    """

    # Load audio as mono at 16 kHz
    audio, sample_rate = librosa.load(
        file_path,
        sr=16000,
        mono=True
    )

    # Short-Time Fourier Transform
    stft = librosa.stft(audio)

    # --------------------------------------------------------
    # MFCC
    # --------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    # Take mean across time
    mfcc_features = np.mean(
        mfcc,
        axis=1
    )

    # --------------------------------------------------------
    # Chroma
    # --------------------------------------------------------

    chroma = librosa.feature.chroma_stft(
        S=np.abs(stft),
        sr=sample_rate
    )

    # Take mean across time
    chroma_features = np.mean(
        chroma,
        axis=1
    )

    # --------------------------------------------------------
    # Mel-spectrogram
    # --------------------------------------------------------

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate
    )

    # Take mean across time
    mel_features = np.mean(
        mel,
        axis=1
    )

    # --------------------------------------------------------
    # Combine all features
    # --------------------------------------------------------

    features = np.concatenate([
        mfcc_features,
        chroma_features,
        mel_features
    ])

    return features


# ============================================================
# CREATE FEATURE DATASET
# ============================================================

def create_feature_dataset():

    rows = []

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"RAVDESS dataset not found at:\n{DATA_PATH}"
        )

    print("RAVDESS location:")
    print(DATA_PATH)
    print()

    # Go through Actor_01, Actor_02, ... Actor_24
    for actor_folder in sorted(os.listdir(DATA_PATH)):

        actor_path = os.path.join(
            DATA_PATH,
            actor_folder
        )

        # Ignore anything that isn't a folder
        if not os.path.isdir(actor_path):
            continue

        # Get actor number from folder name
        actor_id = actor_folder.replace(
            "Actor_",
            ""
        )

        # Go through audio files
        for filename in sorted(os.listdir(actor_path)):

            if not filename.lower().endswith(".wav"):
                continue

            file_path = os.path.join(
                actor_path,
                filename
            )

            try:

                # ------------------------------------------------
                # RAVDESS filename format:
                #
                # 03-01-05-01-02-01-12.wav
                #
                # Position 0 = Modality
                # Position 1 = Vocal channel
                # Position 2 = Emotion
                # Position 3 = Emotional intensity
                # Position 4 = Statement
                # Position 5 = Repetition
                # Position 6 = Actor
                # ------------------------------------------------

                parts = filename.replace(
                    ".wav",
                    ""
                ).split("-")

                emotion_code = parts[2]

                emotion = EMOTIONS.get(
                    emotion_code
                )

                if emotion is None:
                    print(
                        f"Unknown emotion code in {filename}"
                    )
                    continue

                # Extract audio features
                features = extract_features(
                    file_path
                )

                # Start the row with basic information
                row = {
                    "filename": filename,
                    "actor": int(actor_id),
                    "emotion": emotion
                }

                # Add feature columns
                for i, value in enumerate(features):

                    row[f"feature_{i + 1}"] = value

                rows.append(row)

            except Exception as e:

                print(
                    f"Error processing {filename}: {e}"
                )

    # Convert everything into a DataFrame
    df = pd.DataFrame(rows)

    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("Starting RAVDESS feature extraction...")
    print()

    df = create_feature_dataset()

    # Make sure data/processed exists
    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    # Save processed dataset
    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print()
    print("Feature extraction completed.")
    print()
    print(f"Number of samples: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")

    print()
    print("Emotion distribution:")
    print(df["emotion"].value_counts())

    print()
    print("Saved processed dataset to:")
    print(OUTPUT_PATH)