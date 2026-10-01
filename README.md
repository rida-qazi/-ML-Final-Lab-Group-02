# Speech Emotion Recognition

## Overview

This project implements a machine learning pipeline for Speech Emotion Recognition (SER) using the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset.

The system processes speech recordings, extracts acoustic features, trains an SVM-based classification model, and predicts one of eight emotions:

- Angry
- Calm
- Disgust
- Fearful
- Happy
- Neutral
- Sad
- Surprised

The project includes data preprocessing, exploratory analysis, machine learning, model inference, analytics outputs, pipeline validation, and a Streamlit-based demonstration application.

---

## Project Objectives

The project aims to:

- Process and validate speech audio data.
- Extract meaningful acoustic features from audio recordings.
- Perform exploratory data analysis on the extracted features.
- Train a machine learning model for emotion classification.
- Generate emotion probability estimates during inference.
- Evaluate model predictions on a held-out test dataset.
- Build a reusable inference pipeline for individual audio files.
- Provide an interactive demonstration through Streamlit.
- Maintain a modular and reproducible project structure.

---

## Dataset

This project uses the RAVDESS Speech Audio dataset.

The raw dataset is not included in this repository because of its size.

After downloading the dataset, place the RAVDESS audio files under:

```text
data/
└── raw/
    └── RAVDESS/
        ├── Actor_01/
        ├── Actor_02/
        ├── ...
        └── Actor_24/
```

RAVDESS filenames contain metadata describing the emotion and actor. The preprocessing pipeline extracts this metadata automatically.

### Emotion Mapping

| RAVDESS Code | Emotion |
|---|---|
| 01 | Neutral |
| 02 | Calm |
| 03 | Happy |
| 04 | Sad |
| 05 | Angry |
| 06 | Fearful |
| 07 | Disgust |
| 08 | Surprised |

---

## Machine Learning Pipeline

The project follows the pipeline:

```text
RAVDESS Audio
      |
      v
Audio Loading
      |
      v
Feature Extraction
      |
      +-- 40 MFCC features
      +-- 12 Chroma features
      +-- 128 Mel-spectrogram features
      |
      v
180 Audio Features
      |
      v
Feature Validation
      |
      v
StandardScaler
      |
      v
Support Vector Classifier
      |
      v
Emotion Prediction
      |
      +-- Predicted Emotion
      +-- Emotion Probabilities
```

---

## Feature Extraction

The preprocessing pipeline extracts three groups of acoustic features:

### MFCC

40 Mel-Frequency Cepstral Coefficients are extracted to represent the spectral characteristics of speech.

### Chroma

12 chroma features are extracted to represent pitch-class information.

### Mel Features

128 Mel-spectrogram features are extracted to represent the distribution of spectral energy across the Mel frequency scale.

This results in:

```text
40 MFCC
+ 12 Chroma
+ 128 Mel
----------------
180 features
```

The extracted features are validated for:

- Correct feature count
- Missing values
- NaN values
- Infinite values

---

## Model

The trained model is stored as:

```text
models/final_svm_model.pkl
```

The saved model is an scikit-learn pipeline consisting of:

```text
StandardScaler
      |
      v
SVC
```

The SVC configuration is:

```text
C = 10
gamma = 0.01
probability = True
```

Enabling probability estimation allows the inference pipeline to return probability estimates for all eight emotion classes.

---

## Repository Structure

```text
.
├── app/
│   └── app.py
│
├── data/
│   └── processed/
│       ├── features.csv
│       ├── train_features.csv
│       ├── test_features.csv
│       └── data_quality_log.txt
│
├── models/
│   └── final_svm_model.pkl
│
├── notebooks/
│   ├── 02_data_analysis (1).ipynb
│   └── ML_Project.ipnyb
│
├── outputs/
│   ├── Analytics_Engineering/
│   ├── ML_Engineering/
│   └── eda/
│
├── src/
│   ├── __init__.py
│   ├── model.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── Analytics Engineering.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/rida-qazi/-ML-Final-Lab-Group-02.git
cd -ML-Final-Lab-Group-02
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

#### Windows

```bash
.venv\Scripts\activate
```

#### macOS/Linux

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Inference Pipeline

The inference pipeline is implemented in:

```text
src/predict.py
```

The script provides three main use cases:

- Batch inference
- Single-audio inference
- Pipeline sanity checks

### Batch inference

Run predictions on the processed test dataset.

The pipeline reads:

```text
data/processed/test_features.csv
```

and produces:

```text
outputs/ML_Engineering/prediction_results.csv
```

The output includes:

- File path
- Actor ID
- Actual emotion
- Predicted emotion
- Prediction confidence
- Probability for each emotion
- Average inference time per sample

### Single audio inference

The inference pipeline can classify an individual WAV file.

The pipeline:

1. Loads the audio file.
2. Extracts the 180 features.
3. Validates the feature vector.
4. Runs the trained model.
5. Generates probability estimates.
6. Reports inference latency.

### Pipeline Sanity Checks

The inference pipeline contains built-in sanity checks.

The checks verify:

- The trained model loads successfully.
- All expected emotion classes are present.
- Audio inference produces a valid prediction.
- Probability output contains all emotion classes.
- Probabilities are within the range `[0, 1]`.
- Probabilities sum to approximately `1`.
- Inference latency is recorded.
- Invalid audio paths are handled correctly.

A successful run ends with:

```text
ALL SANITY CHECKS PASSED
```

---

## Streamlit Application

An interactive demonstration application is provided in:

```text
app/app.py
```

Start the application with:

```bash
streamlit run app/app.py
```

The application currently supports WAV audio upload and displays:

- Predicted emotion
- Probability for each emotion
- Model inference latency

The application uses the same inference pipeline implemented in `src/predict.py`.

---

## Outputs

### ML Engineering Outputs

The ML inference pipeline produces outputs under:

```text
outputs/ML_Engineering/
```

These include inference logs and prediction results.

`prediction_results.csv` contains predictions and probability estimates generated from the held-out test feature dataset.

### Analytics Engineering Outputs

Analytics-related outputs are stored in:

```text
outputs/Analytics_Engineering/
```

These include downstream analysis of model predictions.

### Exploratory Data Analysis Outputs

EDA results are stored in:

```text
outputs/eda/
```

These include analysis of:

- Class distribution
- Actor distribution
- Feature distributions
- Correlation
- Feature interactions
- MFCCs
- Outliers
- Skewness
- Feature statistics
- Data quality

---

## Logging

The inference pipeline maintains an inference log at:

```text
outputs/ML_Engineering/inference.log
```

The log records information such as:

- Model loading
- Individual audio inference
- Batch inference
- Number of processed samples
- Inference latency
- Output locations

---

## Technologies Used

- Python
- NumPy
- Pandas
- Librosa
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib
- Jupyter Notebook

---

## Project Components

### Data / Preprocessing

Responsible for:

- Audio file discovery
- RAVDESS metadata extraction
- Feature extraction
- Feature validation
- Dataset validation
- Data quality checks
- Creation of processed feature datasets

### Machine Learning

Responsible for:

- Model training
- Model persistence
- Prediction
- Probability estimation
- Inference latency measurement

### Analytics Engineering

Responsible for downstream analysis of model predictions.

### Application

The Streamlit application provides an interactive interface for demonstrating the trained model.

---

## Current Limitations

The model is trained and evaluated using the RAVDESS dataset, which consists of controlled speech recordings.

As a result, performance on naturally recorded speech may differ from performance on RAVDESS recordings due to differences in:

- Speaker characteristics
- Recording environments
- Microphones
- Background noise
- Audio characteristics
- Speaking style

The current interactive application is intended primarily as a demonstration of the trained inference pipeline.

---

## Future Improvements

Potential future improvements include:

- Training with a more diverse collection of speech recordings.
- Improving robustness to different recording environments and microphones.
- Evaluating additional feature representations.
- Comparing multiple classification algorithms.
- Hyperparameter optimization.
- More extensive model calibration and evaluation.
- Improved real-world speech generalization.
- Containerized deployment.
- Cloud deployment of the demonstration application.

---

## License

This project was developed as part of an academic machine learning project.
