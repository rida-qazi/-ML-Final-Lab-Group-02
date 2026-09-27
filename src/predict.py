import time
import joblib
import numpy as np
import pandas as pd
import os
import csv
from preprocessing import extract_features, validate_features


MODEL_PATH = r"C:/Users/Hadel/OneDrive/Desktop/ML_Project/-ML-Final-Lab-Group-02/models/final_svm_model.pkl"


# Load the trained model once
model = joblib.load(MODEL_PATH)

#for single audio file
def predict_audio(audio_path):
    """
    Extract features from an audio file and generate
    an emotion prediction with SVM decision scores.
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(
        f"Audio file not found: {audio_path}"
        )

    # Extract features
    features = extract_features(audio_path)

    # Validate features
    validate_features(features)

    # Reshape to one sample with 180 features
    features = np.asarray(features).reshape(1, -1)

    # Add the feature names expected by the trained model
    features_df = pd.DataFrame(
        features,
        columns=model.feature_names_in_
    )

    # Measure model inference time
    start_time = time.perf_counter()

    prediction = model.predict(features_df)

    end_time = time.perf_counter()

    inference_time_ms = (end_time - start_time) * 1000

    # Get probability estimates
    probabilities = model.predict_proba(features_df)

    probability_dict = dict(
        zip(model.classes_, probabilities[0])
    )


    # Return structured result
    return {
        "prediction": prediction[0],
        "probabilities": probability_dict,
        "inference_time_ms": inference_time_ms
    }

#for test data
def predict_features(features_df):
    """
    Generate predictions and probability estimates from
    already-extracted audio features.
    """

    # Keep only the features expected by the model
    features = features_df[model.feature_names_in_]

    # Measure model inference time
    start_time = time.perf_counter()

    predictions = model.predict(features)

    # Probability estimates
    probabilities = model.predict_proba(features)

    end_time = time.perf_counter()

    inference_time_ms = (end_time - start_time) * 1000

    # Create output DataFrame
    results = pd.DataFrame()

    # Preserve useful metadata
    if "file_path" in features_df.columns:
        results["file_path"] = features_df["file_path"].values

    if "actor_id" in features_df.columns:
        results["actor_id"] = features_df["actor_id"].values

    if "emotion" in features_df.columns:
        results["actual_emotion"] = features_df["emotion"].values

    # Model prediction
    results["predicted_emotion"] = predictions

    #I dont know if you need this but i'll give you prediction confidence also
    results["prediction_confidence"] = probabilities.max(axis=1)

    # Probability estimates
    for i, emotion in enumerate(model.classes_):
        results[f"{emotion}_probability"] = probabilities[:, i]

    # Average inference time per sample
    results["inference_time_ms"] = (
        inference_time_ms / len(features)
    )

    return results






if __name__ == "__main__":

    #testing single audio file
    #AUDIO_PATH = r"C:/Users/Hadel/Downloads/RAVDESS/Audio_Speech_Actors_01-24_16k/Actor_08/03-01-03-01-01-01-08.wav"

    #result = predict_audio(AUDIO_PATH)

    #print("Predicted emotion:", result["prediction"])

    #print("\nEmotion probabilities:")
    #for emotion, probability in result["probabilities"].items():
    #    print(f"{emotion}: {probability * 100:.2f}%")
    #print(f"\nInference time: {result['inference_time_ms']:.2f} ms")

    #testing test data
    test_data = pd.read_csv(
        r"C:/Users/Hadel/OneDrive/Desktop/ML_Project/-ML-Final-Lab-Group-02/data/processed/test_features.csv"
    )

    results = predict_features(test_data)

    print(results.head())
    OUTPUT_PATH = r"C:/Users/Hadel/OneDrive/Desktop/ML_Project/-ML-Final-Lab-Group-02/outputs/ML_Engineering/prediction_results.csv"

    results.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"\nPrediction results saved to: {OUTPUT_PATH}")