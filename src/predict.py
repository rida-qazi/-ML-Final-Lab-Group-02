#Imports section~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import time
import joblib
import numpy as np
import pandas as pd
import os
import logging
from pathlib import Path
import argparse
from preprocessing import extract_features, validate_features

#Paths section~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "final_svm_model.pkl"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ML_Engineering"
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)
LOG_PATH = OUTPUT_DIR / "inference.log"

#Configuring logging~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

#Loading trained model~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
model = joblib.load(MODEL_PATH)

logger.info(
    "Model loaded successfully from %s",
    MODEL_PATH
)

#Function for single audio file prediction~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def predict_audio(audio_path):
    """
    Extract features from an audio file and generate
    an emotion prediction with emotion probability estimates.
    """

    if not os.path.isfile(audio_path):
        logger.error(
            "Audio file not found: %s",
            audio_path
        )

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

    # Get probability estimates
    probabilities = model.predict_proba(features_df)

    end_time = time.perf_counter()

    inference_time_ms = (end_time - start_time) * 1000
    logger.info(
    "Audio inference completed | file=%s | prediction=%s | latency_ms=%.2f",
    audio_path,
    prediction[0],
    inference_time_ms
    )

    probability_dict = dict(
        zip(model.classes_, probabilities[0])
    )

    # Return structured result
    return {
        "prediction": prediction[0],
        "probabilities": probability_dict,
        "inference_time_ms": inference_time_ms
    }

#Function for test data predictions for project report~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    logger.info(
        "Batch inference completed | samples=%d | total_latency_ms=%.2f | average_latency_ms=%.2f",
        len(features),
        inference_time_ms,
        inference_time_ms / len(features)
    )

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

#Run code for test data batch inference~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def run_batch_inference():
    """
    Run inference on the test feature dataset and save
    the prediction results for downstream analytics.
    """

    test_data_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "test_features.csv"
    )

    output_path = OUTPUT_DIR / "prediction_results.csv"

    print("\nRunning batch inference...")
    print("=" * 50)

    test_data = pd.read_csv(test_data_path)

    results = predict_features(test_data)

    results.to_csv(
        output_path,
        index=False
    )

    print(f"Processed {len(results)} samples.")
    print(f"Prediction results saved to: {output_path}")

    logger.info(
        "Batch prediction results saved to %s",
        output_path
    )


#Run code for single audio file inference~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def run_single_audio(audio_path):
    """
    Run inference on a single audio file and display
    the predicted emotion, probabilities, and latency.
    """

    result = predict_audio(audio_path)

    print("\nPrediction Result")
    print("=" * 50)

    print(
        f"Predicted emotion: {result['prediction']}"
    )

    print("\nEmotion probabilities:")

    for emotion, probability in result["probabilities"].items():
        print(
            f"{emotion}: {probability * 100:.2f}%"
        )

    print(
        f"\nInference time: "
        f"{result['inference_time_ms']:.2f} ms"
    )


#Sanity checks~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def run_sanity_checks():
    """
    Run basic checks to verify that the inference pipeline
    is functioning correctly.
    """

    print("\nRunning pipeline sanity checks...")
    print("=" * 50)

    # --------------------------------------------------
    # 1. Check model
    # --------------------------------------------------

    assert model is not None, "Model failed to load."

    print("✓ Model loaded successfully")

    # --------------------------------------------------
    # 2. Check model classes
    # --------------------------------------------------

    expected_emotions = {
        "angry",
        "calm",
        "disgust",
        "fearful",
        "happy",
        "neutral",
        "sad",
        "surprised"
    }

    actual_emotions = set(model.classes_)

    assert actual_emotions == expected_emotions, (
        f"Unexpected model classes: {actual_emotions}"
    )

    print("✓ Model contains all expected emotion classes")


    # --------------------------------------------------
    # 3. Test raw audio inference
    # --------------------------------------------------

    audio_path = (
        r"C:/Users/Hadel/Downloads/RAVDESS/"
        r"Audio_Speech_Actors_01-24_16k/Actor_08/"
        r"03-01-03-01-01-01-08.wav"
    )

    result = predict_audio(audio_path)

    assert result["prediction"] in expected_emotions

    print("✓ Audio inference produces a valid prediction")


    # --------------------------------------------------
    # 4. Check probability outputs
    # --------------------------------------------------

    probabilities = result["probabilities"]

    assert set(probabilities.keys()) == expected_emotions

    print("✓ Probability output contains all emotions")


    # --------------------------------------------------
    # 5. Check probability range
    # --------------------------------------------------

    assert all(
        0 <= probability <= 1
        for probability in probabilities.values()
    )

    print("✓ All probabilities are between 0 and 1")


    # --------------------------------------------------
    # 6. Check probabilities sum to 1
    # --------------------------------------------------

    probability_sum = sum(probabilities.values())

    assert abs(probability_sum - 1.0) < 1e-6

    print("✓ Probabilities sum to 1")


    # --------------------------------------------------
    # 7. Check inference latency
    # --------------------------------------------------

    assert result["inference_time_ms"] >= 0

    print(
        f"✓ Inference latency recorded: "
        f"{result['inference_time_ms']:.2f} ms"
    )


    # --------------------------------------------------
    # 8. Check invalid audio handling
    # --------------------------------------------------

    invalid_path = r"C:/this_file_does_not_exist.wav"

    try:
        predict_audio(invalid_path)

        # If we reach here, the function failed to
        # reject the invalid path.
        raise AssertionError(
            "Invalid audio path was not rejected."
        )

    except FileNotFoundError:
        print("✓ Invalid audio path handled correctly")


    print("=" * 50)
    print("ALL SANITY CHECKS PASSED")


#Running ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Emotion recognition inference pipeline"
    )

    mode = parser.add_mutually_exclusive_group(required=True)

    mode.add_argument(
        "--batch",
        action="store_true",
        help="Run inference on the test feature dataset"
    )

    mode.add_argument(
        "--audio",
        type=str,
        help="Run inference on a single audio file"
    )

    mode.add_argument(
        "--sanity-check",
        action="store_true",
        help="Run pipeline sanity checks"
    )

    args = parser.parse_args()

    if args.batch:
        run_batch_inference()

    elif args.audio:
        run_single_audio(args.audio)

    elif args.sanity_check:
        run_sanity_checks()

    else:
        parser.print_help()