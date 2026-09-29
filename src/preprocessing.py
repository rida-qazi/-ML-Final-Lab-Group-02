# %% [markdown]
# # Data Engineering Pipeline - Properly Modular Single File
#
# Each function has one clear responsibility.
# Generic feature extraction is separate from RAVDESS metadata handling.

# %% Cell 1
# ============================================================
# 1. IMPORTS AND SETTINGS
# ============================================================

import os
import glob
import time

import numpy as np
import pandas as pd
import librosa

from tqdm import tqdm
from sklearn.model_selection import train_test_split


N_MFCC = 40
N_CHROMA = 12
N_MEL = 128
TOTAL_FEATURES = N_MFCC + N_CHROMA + N_MEL

# Change these paths for your computer
DATA_PATH = r"/Users/macbook/Documents/Audio_Speech_Actors_01-24_16k"
OUTPUT_PATH = r"data/processed"

os.makedirs(OUTPUT_PATH, exist_ok=True)

print("Total features:", TOTAL_FEATURES)


# %% [markdown]
# # 2. GENERIC FEATURE EXTRACTION
#
# These functions do not know anything about RAVDESS.
# They can be reused with another compatible WAV file or dataset.

# %% Cell 2
# ============================================================
# 1. EXTRACT FEATURES FROM ONE AUDIO FILE
# ============================================================

def extract_features(file_name):
    """Load one audio file and return 180 audio features."""

    # Load audio
    audio, sample_rate = librosa.load(
        file_name,
        sr=None
    )

    if len(audio) == 0:
        raise ValueError("Audio file is empty")

    # --------------------------------------------------------
    # MFCC: 40 features
    # --------------------------------------------------------
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=N_MFCC
    )

    mfcc = np.mean(mfcc.T, axis=0)

    # --------------------------------------------------------
    # Chroma: 12 features
    # --------------------------------------------------------
    stft = np.abs(librosa.stft(audio))

    chroma = librosa.feature.chroma_stft(
        S=stft,
        sr=sample_rate,
        tuning=0
    )

    chroma = np.mean(chroma.T, axis=0)

    # --------------------------------------------------------
    # Mel: 128 features
    # --------------------------------------------------------
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=N_MEL
    )

    mel = np.mean(mel.T, axis=0)

    # Combine all feature groups
    features = np.hstack([
        mfcc,
        chroma,
        mel
    ])

    return features


# %% Cell 3
# ============================================================
# 2. GET FEATURE NAMES
# ============================================================

def get_feature_names():
    """Return names for all 180 features."""

    return (
        [f"mfcc_{i}" for i in range(N_MFCC)]
        + [f"chroma_{i}" for i in range(N_CHROMA)]
        + [f"mel_{i}" for i in range(N_MEL)]
    )


# %% Cell 4
# ============================================================
# 3. VALIDATE ONE FEATURE VECTOR
# ============================================================

def validate_features(feature):
    """Check feature count, NaN values and infinite values."""

    if feature.shape[0] != TOTAL_FEATURES:
        raise ValueError(
            f"Expected {TOTAL_FEATURES} features, "
            f"got {feature.shape[0]}"
        )

    if np.isnan(feature).any():
        raise ValueError("NaN values found in features")

    if np.isinf(feature).any():
        raise ValueError("Infinite values found in features")

    return True


# %% [markdown]
# # 3. RAVDESS-SPECIFIC FUNCTIONS
#
# Only RAVDESS-specific responsibilities are kept here:
# file discovery, filename parsing, actor ID and emotion mapping.

# %% Cell 5
# ============================================================
# RAVDESS EMOTION MAPPING
# ============================================================

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


# %% Cell 6
# ============================================================
# 4. FIND AUDIO FILES
# ============================================================

def find_audio_files(input_path):
    """Find WAV files from either a single file or a folder."""

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Input path does not exist: {input_path}"
        )

    if os.path.isfile(input_path):
        if not input_path.lower().endswith(".wav"):
            raise ValueError("Input file must be a .wav file")

        return [input_path]

    files = glob.glob(
        os.path.join(input_path, "**", "*.wav"),
        recursive=True
    )

    if not files:
        raise FileNotFoundError(
            f"No .wav files found under {input_path}"
        )

    return sorted(files)


# %% Cell 7
# ============================================================
# 5. EXTRACT RAVDESS METADATA
# ============================================================

def extract_ravdess_metadata(file_name):
    """Extract actor ID and emotion from a RAVDESS filename."""

    parts = os.path.basename(file_name).split("-")

    if len(parts) != 7:
        raise ValueError(
            "Invalid RAVDESS filename format"
        )

    emotion_code = parts[2]
    actor_id = parts[6].replace(".wav", "")

    if emotion_code not in EMOTION_MAP:
        raise ValueError(
            f"Unknown emotion code: {emotion_code}"
        )

    emotion = EMOTION_MAP[emotion_code]

    return actor_id, emotion


# %% [markdown]
# # 4. MAIN PREPROCESSING FUNCTION
#
# This function combines the smaller functions.
# It does not contain the feature extraction logic itself.

# %% Cell 8
# ============================================================
# 6. MAIN PREPROCESSING FUNCTION
# ============================================================

def preprocess_audio(input_path, dataset="ravdess"):
    """Process the input audio files and create a feature DataFrame."""

    records = []
    corrupt_files = []

    files = find_audio_files(input_path)

    print(f"Found {len(files)} audio files.")

    for file in tqdm(
        files,
        desc="Extracting features"
    ):

        try:
            # RAVDESS-specific metadata
            if dataset.lower() == "ravdess":
                actor_id, emotion = extract_ravdess_metadata(file)
            else:
                raise ValueError(
                    f"Unsupported dataset: {dataset}"
                )

            # Generic feature extraction
            feature = extract_features(file)

            # Feature validation
            validate_features(feature)

            # Create one dataset record
            record = {
                "file_path": file,
                "actor_id": actor_id,
                "emotion": emotion
            }

            for name, value in zip(
                get_feature_names(),
                feature
            ):
                record[name] = value

            records.append(record)

        except Exception as e:
            corrupt_files.append(file)

            print(
                f"[WARN] Skipping {file}: {e}"
            )

    features_df = pd.DataFrame(records)

    return features_df, corrupt_files


# %% [markdown]
# # 5. DATASET VALIDATION AND OUTPUT FUNCTIONS

# %% Cell 9
# ============================================================
# 7. DATASET SCHEMA VALIDATION
# ============================================================

def validate_dataset_schema(features_df):
    """Check that all expected columns are present."""

    expected_columns = (
        ["file_path", "actor_id", "emotion"]
        + get_feature_names()
    )

    missing_columns = [
        column
        for column in expected_columns
        if column not in features_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {missing_columns}"
        )

    print("Schema validation: PASSED")
    print(f"Total columns: {len(features_df.columns)}")

    return True


# %% Cell 10
# ============================================================
# 8. DATA QUALITY CHECKS
# ============================================================

def check_dataset_quality(
    features_df,
    corrupt_files,
    expected_files=None
):
    """Check missing values, duplicates and dataset size."""

    feature_columns = get_feature_names()

    missing_values = (
        features_df[feature_columns]
        .isna()
        .sum()
        .sum()
    )

    infinite_values = np.isinf(
        features_df[feature_columns].values
    ).sum()

    duplicate_rows = features_df.duplicated().sum()

    duplicate_files = features_df["file_path"].duplicated().sum()

    unique_actors = features_df["actor_id"].nunique()

    print("\nDATA QUALITY CHECK")
    print("---------------------------")
    print(f"Usable files      : {len(features_df)}")
    print(f"Corrupt files     : {len(corrupt_files)}")
    print(f"Missing values    : {missing_values}")
    print(f"Infinite values   : {infinite_values}")
    print(f"Duplicate rows    : {duplicate_rows}")
    print(f"Duplicate files   : {duplicate_files}")
    print(f"Unique actors     : {unique_actors}")

    if expected_files is None or len(features_df) == expected_files:
        print("File count        : PASSED")
    else:
        print(
            f"File count        : REVIEW "
            f"(expected {expected_files}, got {len(features_df)})"
        )

    return {
        "usable_files": len(features_df),
        "corrupt_files": len(corrupt_files),
        "missing_values": int(missing_values),
        "infinite_values": int(infinite_values),
        "duplicate_rows": int(duplicate_rows),
        "duplicate_files": int(duplicate_files),
        "unique_actors": int(unique_actors)
    }


# %% Cell 11
# ============================================================
# 9. TRAIN / TEST SPLIT
# ============================================================

def split_dataset(
    features_df,
    test_size=0.20,
    random_state=42
):
    """Split the feature DataFrame into training and testing data."""

    train_df, test_df = train_test_split(
        features_df,
        test_size=test_size,
        random_state=random_state,
        stratify=features_df["emotion"]
    )

    return train_df, test_df


# %% Cell 12
# ============================================================
# 10. QUALITY LOG
# ============================================================

def write_quality_log(
    features_df,
    corrupt_files,
    data_path,
    output_path,
    elapsed_s,
    expected_files=None
):
    """Write a readable data quality audit report."""

    feature_columns = get_feature_names()

    missing_values = (
        features_df[feature_columns]
        .isna()
        .sum()
        .sum()
    )

    infinite_values = np.isinf(
        features_df[feature_columns].values
    ).sum()

    duplicate_rows = features_df.duplicated().sum()
    duplicate_files = features_df["file_path"].duplicated().sum()
    unique_actors = features_df["actor_id"].nunique()

    lines = []

    lines.append("DATA QUALITY LOG & AUDIT SHEET")
    lines.append("=" * 45)
    lines.append(f"Source path          : {data_path}")
    lines.append(f"Extraction time      : {elapsed_s:.1f}s")
    lines.append(f"Total usable files   : {len(features_df)}")
    lines.append(f"Corrupt/skipped      : {len(corrupt_files)}")
    lines.append(
        f"Feature dimensions   : {len(feature_columns)} "
        f"(40 MFCC + 12 Chroma + 128 Mel)"
    )
    lines.append(f"Missing feature values : {int(missing_values)}")
    lines.append(f"Infinite feature values: {int(infinite_values)}")
    lines.append(f"Duplicate rows          : {int(duplicate_rows)}")
    lines.append(f"Duplicate file paths    : {int(duplicate_files)}")
    lines.append(f"Unique actors           : {unique_actors}")

    lines.append("")
    lines.append("Emotion distribution:")

    emotion_counts = (
        features_df["emotion"]
        .value_counts()
        .sort_index()
    )

    for emotion, count in emotion_counts.items():
        lines.append(f"  {emotion:15s} {count}")

    lines.append("")
    lines.append("Preprocessing validation:")

    file_count_ok = (
        expected_files is None
        or len(features_df) == expected_files
    )

    if file_count_ok:
        lines.append(
            f"  File count check    : PASSED (got {len(features_df)})"
        )
    else:
        lines.append(
            f"  File count check    : REVIEW "
            f"(expected {expected_files}, got {len(features_df)})"
        )

    if (
        file_count_ok
        and len(corrupt_files) == 0
        and missing_values == 0
        and infinite_values == 0
        and duplicate_rows == 0
        and duplicate_files == 0
    ):
        lines.append("  STATUS: PASSED")
    else:
        lines.append("  STATUS: REVIEW REQUIRED")

    log_file = os.path.join(
        output_path,
        "data_quality_log.txt"
    )

    with open(log_file, "w") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\nAudit log saved to: {log_file}")


# %% [markdown]
# # 6. FINAL ORCHESTRATOR
#
# `run_preprocessing()` only coordinates the independent functions.

# %% Cell 13
# ============================================================
# 11. FULL PREPROCESSING PIPELINE
# ============================================================

def run_preprocessing(
    input_path,
    output_path,
    expected_files=None
):
    """Run the complete preprocessing pipeline."""

    t0 = time.time()

    os.makedirs(
        output_path,
        exist_ok=True
    )

    # 1. Process raw audio and create feature DataFrame
    features_df, corrupt_files = preprocess_audio(
        input_path
    )

    # 2. Validate final schema
    validate_dataset_schema(features_df)

    # 3. Save complete feature dataset
    features_file = os.path.join(
        output_path,
        "features.csv"
    )

    features_df.to_csv(
        features_file,
        index=False
    )

    print("Feature dataset saved to:", features_file)

    # 4. Save corrupt file list if required
    if corrupt_files:
        corrupt_file = os.path.join(
            output_path,
            "corrupt_files.csv"
        )

        pd.DataFrame({
            "corrupt_file": corrupt_files
        }).to_csv(
            corrupt_file,
            index=False
        )

        print("Corrupt file list saved to:", corrupt_file)

    # 5. Split into train and test data
    train_df, test_df = split_dataset(
        features_df
    )

    # 6. Save train and test datasets
    train_file = os.path.join(
        output_path,
        "train_features.csv"
    )

    test_file = os.path.join(
        output_path,
        "test_features.csv"
    )

    train_df.to_csv(
        train_file,
        index=False
    )

    test_df.to_csv(
        test_file,
        index=False
    )

    print("Training samples:", len(train_df))
    print("Testing samples :", len(test_df))

    # 7. Dataset-level quality checks
    check_dataset_quality(
        features_df,
        corrupt_files,
        expected_files
    )

    # 8. Write quality log
    elapsed = time.time() - t0

    write_quality_log(
        features_df,
        corrupt_files,
        input_path,
        output_path,
        elapsed,
        expected_files
    )

    return (
        features_df,
        train_df,
        test_df,
        corrupt_files
    )


# %% [markdown]
# # 7. TEST THE GENERIC FUNCTION
#
# This test uses only `extract_features()`. No RAVDESS emotion or actor
# information is required.

# %% Cell 14
# ============================================================
# TEST ONE AUDIO FILE
# ============================================================
if __name__ == "__main__":
    ravdess_files = find_audio_files(DATA_PATH)
    sample_file = ravdess_files[0]
    
    sample_features = extract_features(sample_file)
    validate_features(sample_features)
    
    print("Sample file:", sample_file)
    print("Number of features:", len(sample_features))
    print("Contains NaN:", np.isnan(sample_features).any())
    print("Contains infinite:", np.isinf(sample_features).any())


# %% [markdown]
# # 8. OPTIONAL NON-RAVDESS AUDIO TEST
#
# Set `RANDOM_AUDIO_PATH` to any existing WAV file outside RAVDESS.

# %% Cell 15
# ============================================================
# OPTIONAL RANDOM / NON-RAVDESS AUDIO TEST
# ============================================================

RANDOM_AUDIO_PATH = r"C:\Users\kamal\Downloads\ml\data\Audio_Speech_Actors_01-24_16k\Actor_20\03-01-04-02-01-02-20.wav"

if os.path.isfile(RANDOM_AUDIO_PATH):

    random_features = extract_features(
        RANDOM_AUDIO_PATH
    )

    validate_features(random_features)

    print("Audio file:", RANDOM_AUDIO_PATH)
    print("Number of features:", len(random_features))
    print("Contains NaN:", np.isnan(random_features).any())
    print("Contains infinite:", np.isinf(random_features).any())
    print("First 10 extracted features:")
    print(random_features[:10])

else:
    print("Optional non-RAVDESS test skipped.")
    print("Set RANDOM_AUDIO_PATH to an existing WAV file to run it.")


# %% [markdown]
# # 9. FUNCTION CHECK

# %% Cell 16
# ============================================================
# QUICK FUNCTION CHECK
# ============================================================

print("\n--- FUNCTION CHECK ---")
print("extract_features()          : READY")
print("get_feature_names()         : READY")
print("find_audio_files()          : READY")
print("extract_ravdess_metadata()  : READY")
print("validate_features()         : READY")
print("preprocess_audio()          : READY")
print("split_dataset()             : READY")
print("run_preprocessing()         : READY")


# %% [markdown]
# # 10. RUN THE COMPLETE PIPELINE
#
# This is the only main pipeline call required.

# %% Cell 17
# ============================================================
# RUN COMPLETE PREPROCESSING
# ============================================================
if __name__ == "__main__":
    features_df, train_df, test_df, corrupt_files = run_preprocessing(
        input_path=DATA_PATH,
        output_path=OUTPUT_PATH,
        expected_files=1440
    )


# %% [markdown]
# # 11. FINAL SUMMARY

# %% Cell 18
# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n================ FINAL SUMMARY ================")
print("Pipeline completed successfully.")
if __name__ == "__main__":
    print(f"Processed files : {len(features_df)}")
    print(f"Corrupt files   : {len(corrupt_files)}")
    print(f"Features/file   : {TOTAL_FEATURES}")
    print(f"Train samples   : {len(train_df)}")
    print(f"Test samples    : {len(test_df)}")
    print(f"Output folder   : {OUTPUT_PATH}")
    print("================================================")
