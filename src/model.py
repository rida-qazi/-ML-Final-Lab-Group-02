import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

results = []

warnings.filterwarnings("ignore")

# This is the processed feature file that already contains the extracted audio features.
FEATURES_PATH = r"/Users/macbook/Desktop/ml last/-ML-Final-Lab-Group-02/data/processed/features.csv"

# Load the feature dataset so we can prepare it for model training.
df = pd.read_csv(FEATURES_PATH)

print("Dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nEmotion distribution:")
print(df["emotion"].value_counts())

from sklearn.model_selection import train_test_split

# Keep only the actual audio features here. File path, actor ID, and emotion are metadata/target columns.
feature_columns = [
    col for col in df.columns
    if col not in ["file_path", "actor_id", "emotion"]
]

X = df[feature_columns]
y = df["emotion"]

# Split the data into training and testing sets while keeping the emotion classes balanced.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTest class distribution:")
print(y_test.value_counts())

# Start by training an SVM model as one of the baseline classifiers.

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Scaling is important for SVM because the features can have very different ranges.
svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True
        
    ))
])

# Use five folds to check how consistently the SVM performs on different parts of the training data.
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision_macro",
    "recall": "recall_macro",
    "f1": "f1_macro"
}

svm_cv = cross_validate(
    svm_pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring=scoring,
    return_train_score=True,
    n_jobs=-1
)

# Average the validation scores from all five folds so we get a more stable estimate of performance.
cv_train_acc = svm_cv["train_accuracy"].mean()
cv_accuracy = svm_cv["test_accuracy"].mean()
cv_precision = svm_cv["test_precision"].mean()
cv_recall = svm_cv["test_recall"].mean()
cv_f1 = svm_cv["test_f1"].mean()

# Run the grid search on the training data. the final SVM using all available training samples.
svm_pipeline.fit(X_train, y_train)

# Generate the final predictions on both training and test data.
y_train_pred = svm_pipeline.predict(X_train)
y_test_pred = svm_pipeline.predict(X_test)

# Calculate the main metrics we will use to evaluate the model.
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(
    y_test, y_test_pred, average="macro"
)
test_recall = recall_score(
    y_test, y_test_pred, average="macro"
)
test_f1 = f1_score(
    y_test, y_test_pred, average="macro"
)

print("SVM (RBF, C=10)")
print("=" * 50)

print(f"Training Accuracy:  {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"CV Accuracy:        {cv_accuracy:.4f} ({cv_accuracy*100:.2f}%)")
print(f"Test Accuracy:      {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

print(f"\nCV Macro F1:        {cv_f1:.4f}")
print(f"Test Macro F1:      {test_f1:.4f}")
print(f"Test Macro Precision: {test_precision:.4f}")
print(f"Test Macro Recall:    {test_recall:.4f}")

print(f"\nTrain-Test Gap:      {(train_accuracy-test_accuracy)*100:.2f}%")

# Save the SVM results so they can be compared with the other models later.
results.append({
    "Model": "SVM (RBF)",
    "Training Accuracy": train_accuracy,
    "CV Accuracy": cv_accuracy,
    "Test Accuracy": test_accuracy,
    "CV Macro F1": cv_f1,
    "Test Macro F1": test_f1,
    "Test Macro Precision": test_precision,
    "Test Macro Recall": test_recall,
    "Train-Test Gap": train_accuracy - test_accuracy
})
print("\nResults stored successfully.")

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Generate the final predictions on both training and test data. on the test set
y_test_pred = svm_pipeline.predict(X_test)

# The confusion matrix shows which emotions are being predicted correctly and which are getting mixed up.
cm = confusion_matrix(y_test, y_test_pred)

# Keep the emotion labels in a fixed order so the matrix is easy to read.
classes = sorted(y_test.unique())

# Display the confusion matrix as a heatmap.
fig, ax = plt.subplots(figsize=(9, 7))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)

disp.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    xticks_rotation=45
)

plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted Emotion")
plt.ylabel("Actual Emotion")
plt.tight_layout()
plt.show()

# Now train Logistic Regression and evaluate it using the same setup.

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Standardize the features first, then train Logistic Regression.
logistic_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=2000,
        random_state=42
    ))
])

# Use five folds to check how consistently the SVM performs on different parts of the training data.
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision_macro",
    "recall": "recall_macro",
    "f1": "f1_macro"
}

logistic_cv = cross_validate(
    logistic_pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring=scoring,
    return_train_score=True,
    n_jobs=-1
)

# Average the validation scores from all five folds so we get a more stable estimate of performance.
cv_train_acc = logistic_cv["train_accuracy"].mean()
cv_accuracy = logistic_cv["test_accuracy"].mean()
cv_precision = logistic_cv["test_precision"].mean()
cv_recall = logistic_cv["test_recall"].mean()
cv_f1 = logistic_cv["test_f1"].mean()

# Run the grid search on the training data. Logistic Regression on the complete training portion.
logistic_pipeline.fit(X_train, y_train)

# Generate the final predictions on both training and test data.
y_train_pred = logistic_pipeline.predict(X_train)
y_test_pred = logistic_pipeline.predict(X_test)

# Calculate the main metrics we will use to evaluate the model.
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

test_precision = precision_score(
    y_test,
    y_test_pred,
    average="macro"
)

test_recall = recall_score(
    y_test,
    y_test_pred,
    average="macro"
)

test_f1 = f1_score(
    y_test,
    y_test_pred,
    average="macro"
)

print("Logistic Regression")
print("=" * 50)

print(
    f"Training Accuracy:  {train_accuracy:.4f} "
    f"({train_accuracy*100:.2f}%)"
)

print(
    f"CV Accuracy:        {cv_accuracy:.4f} "
    f"({cv_accuracy*100:.2f}%)"
)

print(
    f"Test Accuracy:      {test_accuracy:.4f} "
    f"({test_accuracy*100:.2f}%)"
)

print(f"\nCV Macro F1:          {cv_f1:.4f}")
print(f"Test Macro F1:        {test_f1:.4f}")
print(f"Test Macro Precision: {test_precision:.4f}")
print(f"Test Macro Recall:    {test_recall:.4f}")

print(
    f"\nTrain-Test Gap:       "
    f"{(train_accuracy-test_accuracy)*100:.2f}%"
)

# Keep the Logistic Regression metrics for the final model comparison.
logistic_result = {
    "Model": "Logistic Regression",
    "Training Accuracy": train_accuracy,
    "CV Accuracy": cv_accuracy,
    "Test Accuracy": test_accuracy,
    "CV Macro F1": cv_f1,
    "Test Macro F1": test_f1,
    "Test Macro Precision": test_precision,
    "Test Macro Recall": test_recall,
    "Train-Test Gap": train_accuracy - test_accuracy
}

print("\nResults stored successfully.")

import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# Generate the final predictions on both training and test data. on the test set
y_test_pred = logistic_pipeline.predict(X_test)

# Keep the emotion labels in a fixed order so the matrix is easy to read.
classes = sorted(y_test.unique())

# The confusion matrix shows which emotions are being predicted correctly and which are getting mixed up.
cm = confusion_matrix(
    y_test,
    y_test_pred,
    labels=classes
)

# Display the confusion matrix as a heatmap.
fig, ax = plt.subplots(figsize=(9, 7))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)

disp.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    xticks_rotation=45
)

plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted Emotion")
plt.ylabel("Actual Emotion")
plt.tight_layout()
plt.show()


# Show precision, recall and F1-score separately for each emotion.
print("\nLogistic Regression Classification Report:")
print(
    classification_report(
        y_test,
        y_test_pred,
        labels=classes,
        target_names=classes
    )
)

# Run the grid search on the training data. a Random Forest model to compare a tree-based approach with the previous models.

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set up the Random Forest with the selected hyperparameters.
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

# Use five folds to check how consistently the SVM performs on different parts of the training data.
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision_macro",
    "recall": "recall_macro",
    "f1": "f1_macro"
}

rf_cv = cross_validate(
    rf,
    X_train,
    y_train,
    cv=cv,
    scoring=scoring,
    return_train_score=True,
    n_jobs=-1
)

# Average the validation scores from all five folds so we get a more stable estimate of performance.
cv_train_acc = rf_cv["train_accuracy"].mean()
cv_accuracy = rf_cv["test_accuracy"].mean()
cv_precision = rf_cv["test_precision"].mean()
cv_recall = rf_cv["test_recall"].mean()
cv_f1 = rf_cv["test_f1"].mean()

# Run the grid search on the training data. the final Random Forest using the full training data.
rf.fit(X_train, y_train)

# Generate the final predictions on both training and test data.
y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)

# Calculate the main metrics we will use to evaluate the model.
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

test_precision = precision_score(
    y_test,
    y_test_pred,
    average="macro"
)

test_recall = recall_score(
    y_test,
    y_test_pred,
    average="macro"
)

test_f1 = f1_score(
    y_test,
    y_test_pred,
    average="macro"
)

# Save the SVM results so they can be compared with the other models later.
results.append({
    "Model": "Random Forest",
    "Training Accuracy": train_accuracy,
    "CV Accuracy": cv_accuracy,
    "Test Accuracy": test_accuracy,
    "CV Macro F1": cv_f1,
    "Test Macro F1": test_f1,
    "Test Macro Precision": test_precision,
    "Test Macro Recall": test_recall,
    "Train-Test Gap": train_accuracy - test_accuracy
})

# Print the Random Forest performance so we can compare it with the other models.
print("Random Forest")
print("=" * 50)

print(
    f"Training Accuracy:    {train_accuracy:.4f} "
    f"({train_accuracy*100:.2f}%)"
)

print(
    f"CV Accuracy:          {cv_accuracy:.4f} "
    f"({cv_accuracy*100:.2f}%)"
)

print(
    f"Test Accuracy:        {test_accuracy:.4f} "
    f"({test_accuracy*100:.2f}%)"
)

print(f"\nCV Macro F1:          {cv_f1:.4f}")
print(f"Test Macro F1:        {test_f1:.4f}")
print(f"Test Macro Precision: {test_precision:.4f}")
print(f"Test Macro Recall:    {test_recall:.4f}")

print(
    f"\nTrain-Test Gap:       "
    f"{(train_accuracy-test_accuracy)*100:.2f}%"
)

print("\nResults stored successfully.")

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Generate the final predictions on both training and test data.
y_test_pred = rf.predict(X_test)

# Use the same emotion order when displaying the confusion matrix.
classes = sorted(y_test.unique())

# The confusion matrix shows which emotions are being predicted correctly and which are getting mixed up.
cm = confusion_matrix(
    y_test,
    y_test_pred,
    labels=classes
)

# Display the confusion matrix as a heatmap.
fig, ax = plt.subplots(figsize=(9, 7))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)

disp.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    xticks_rotation=45
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Emotion")
plt.ylabel("Actual Emotion")
plt.tight_layout()
plt.show()

# Run the grid search on the training data. XGBoost as another model for multiclass emotion classification.

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# XGBoost needs the emotion labels in numeric form, so encode them first.
encoder = LabelEncoder()

y_train_xgb = encoder.fit_transform(y_train)
y_test_xgb = encoder.transform(y_test)

print("Classes:")
for i, label in enumerate(encoder.classes_):
    print(i, "=", label)

# Set up the XGBoost classifier with the chosen model settings.
xgb_model = XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    num_class=len(encoder.classes_),
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

# Use five folds to check how consistently the SVM performs on different parts of the training data.
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision_macro",
    "recall": "recall_macro",
    "f1": "f1_macro"
}

xgb_cv = cross_validate(
    xgb_model,
    X_train,
    y_train_xgb,
    cv=cv,
    scoring=scoring,
    return_train_score=True,
    n_jobs=-1
)

# Average the validation scores from all five folds so we get a more stable estimate of performance.
cv_train_acc = xgb_cv["train_accuracy"].mean()
cv_accuracy = xgb_cv["test_accuracy"].mean()
cv_precision = xgb_cv["test_precision"].mean()
cv_recall = xgb_cv["test_recall"].mean()
cv_f1 = xgb_cv["test_f1"].mean()

# Run the grid search on the training data. XGBoost on all of the training samples.
xgb_model.fit(X_train, y_train_xgb)

# Generate the final predictions on both training and test data.
y_train_pred_xgb = xgb_model.predict(X_train)
y_test_pred_xgb = xgb_model.predict(X_test)

# Calculate the main metrics we will use to evaluate the model.
train_accuracy = accuracy_score(
    y_train_xgb,
    y_train_pred_xgb
)

test_accuracy = accuracy_score(
    y_test_xgb,
    y_test_pred_xgb
)

test_precision = precision_score(
    y_test_xgb,
    y_test_pred_xgb,
    average="macro"
)

test_recall = recall_score(
    y_test_xgb,
    y_test_pred_xgb,
    average="macro"
)

test_f1 = f1_score(
    y_test_xgb,
    y_test_pred_xgb,
    average="macro"
)

# Save the SVM results so they can be compared with the other models later.
results.append({
    "Model": "XGBoost",
    "Training Accuracy": train_accuracy,
    "CV Accuracy": cv_accuracy,
    "Test Accuracy": test_accuracy,
    "CV Macro F1": cv_f1,
    "Test Macro F1": test_f1,
    "Test Macro Precision": test_precision,
    "Test Macro Recall": test_recall,
    "Train-Test Gap": train_accuracy - test_accuracy
})
print("\nResults stored successfully.")

# Print the Random Forest performance so we can compare it with the other models.
print("\nXGBoost")
print("=" * 50)

print(
    f"Training Accuracy:    {train_accuracy:.4f} "
    f"({train_accuracy*100:.2f}%)"
)

print(
    f"CV Accuracy:          {cv_accuracy:.4f} "
    f"({cv_accuracy*100:.2f}%)"
)

print(
    f"Test Accuracy:        {test_accuracy:.4f} "
    f"({test_accuracy*100:.2f}%)"
)

print(f"\nCV Macro F1:          {cv_f1:.4f}")
print(f"Test Macro F1:        {test_f1:.4f}")
print(f"Test Macro Precision: {test_precision:.4f}")
print(f"Test Macro Recall:    {test_recall:.4f}")

print(
    f"\nTrain-Test Gap:       "
    f"{(train_accuracy-test_accuracy)*100:.2f}%"
)

import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# The predictions are numeric, so we use the encoded class labels when plotting the results.
y_test_pred_xgb = xgb_model.predict(X_test)

# Use the same emotion order when displaying the confusion matrix.
classes_xgb = encoder.classes_

# The confusion matrix shows which emotions are being predicted correctly and which are getting mixed up.
cm_xgb = confusion_matrix(
    y_test_xgb,
    y_test_pred_xgb,
    labels=range(len(classes_xgb))
)

# Display the confusion matrix as a heatmap.
fig, ax = plt.subplots(figsize=(9, 7))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_xgb,
    display_labels=classes_xgb
)

disp.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    xticks_rotation=45
)

plt.title("XGBoost Confusion Matrix")
plt.xlabel("Predicted Emotion")
plt.ylabel("Actual Emotion")
plt.tight_layout()
plt.show()

# Show precision, recall and F1-score separately for each emotion.
print("\nXGBoost Classification Report:")
print(
    classification_report(
        y_test_xgb,
        y_test_pred_xgb,
        labels=range(len(classes_xgb)),
        target_names=classes_xgb
    )
)

# Run the grid search on the training data. a small neural network model to see how it performs against the traditional classifiers.

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Scale the features before passing them into the neural network.
mlp_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", MLPClassifier(
        hidden_layer_sizes=(64, 32),
        alpha=1.0,
        batch_size=32,
        learning_rate="adaptive",
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=42
    ))
])

# Use five folds to check how consistently the SVM performs on different parts of the training data.
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision_macro",
    "recall": "recall_macro",
    "f1": "f1_macro"
}

mlp_cv = cross_validate(
    mlp_pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring=scoring,
    return_train_score=True,
    n_jobs=-1
)

# Average the validation scores from all five folds so we get a more stable estimate of performance.
cv_train_acc = mlp_cv["train_accuracy"].mean()
cv_accuracy = mlp_cv["test_accuracy"].mean()
cv_precision = mlp_cv["test_precision"].mean()
cv_recall = mlp_cv["test_recall"].mean()
cv_f1 = mlp_cv["test_f1"].mean()

# Run the grid search on the training data. the final MLP using the complete training set.
mlp_pipeline.fit(X_train, y_train)

# Generate the final predictions on both training and test data.
y_train_pred = mlp_pipeline.predict(X_train)
y_test_pred = mlp_pipeline.predict(X_test)

# Calculate the main metrics we will use to evaluate the model.
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

test_precision = precision_score(
    y_test,
    y_test_pred,
    average="macro"
)

test_recall = recall_score(
    y_test,
    y_test_pred,
    average="macro"
)

test_f1 = f1_score(
    y_test,
    y_test_pred,
    average="macro"
)

# Save the SVM results so they can be compared with the other models later.
results.append({
    "Model": "MLP",
    "Training Accuracy": train_accuracy,
    "CV Accuracy": cv_accuracy,
    "Test Accuracy": test_accuracy,
    "CV Macro F1": cv_f1,
    "Test Macro F1": test_f1,
    "Test Macro Precision": test_precision,
    "Test Macro Recall": test_recall,
    "Train-Test Gap": train_accuracy - test_accuracy
})

# Print the Random Forest performance so we can compare it with the other models.
print("MLP Neural Network")
print("=" * 50)

print(
    f"Training Accuracy:    {train_accuracy:.4f} "
    f"({train_accuracy*100:.2f}%)"
)

print(
    f"CV Accuracy:          {cv_accuracy:.4f} "
    f"({cv_accuracy*100:.2f}%)"
)

print(
    f"Test Accuracy:        {test_accuracy:.4f} "
    f"({test_accuracy*100:.2f}%)"
)

print(f"\nCV Macro F1:          {cv_f1:.4f}")
print(f"Test Macro F1:        {test_f1:.4f}")
print(f"Test Macro Precision: {test_precision:.4f}")
print(f"Test Macro Recall:    {test_recall:.4f}")

print(
    f"\nTrain-Test Gap:       "
    f"{(train_accuracy-test_accuracy)*100:.2f}%"
)

import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# Get the MLP predictions for the unseen test data.
y_test_pred_mlp = mlp_pipeline.predict(X_test)

# Use the same emotion order when displaying the confusion matrix.
classes_mlp = sorted(y_test.unique())

# The confusion matrix shows which emotions are being predicted correctly and which are getting mixed up.
cm_mlp = confusion_matrix(
    y_test,
    y_test_pred_mlp,
    labels=classes_mlp
)

# Display the confusion matrix as a heatmap.
fig, ax = plt.subplots(figsize=(9, 7))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_mlp,
    display_labels=classes_mlp
)

disp.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    xticks_rotation=45
)

plt.title("MLP Confusion Matrix")
plt.xlabel("Predicted Emotion")
plt.ylabel("Actual Emotion")
plt.tight_layout()
plt.show()

# Show precision, recall and F1-score separately for each emotion.
print("\nMLP Classification Report:")
print(
    classification_report(
        y_test,
        y_test_pred_mlp,
        labels=classes_mlp,
        target_names=classes_mlp
    )
)

# Bring all model results together so their performance can be compared in one table.

import pandas as pd

comparison_df = pd.DataFrame(results)

# Convert decimal scores into percentages to make the comparison easier to understand.
display_df = comparison_df.copy()

metric_columns = [
    "Training Accuracy",
    "CV Accuracy",
    "Test Accuracy",
    "CV Macro F1",
    "Test Macro F1",
    "Test Macro Precision",
    "Test Macro Recall",
    "Train-Test Gap"
]

for col in metric_columns:
    display_df[col] = display_df[col] * 100

display_df = display_df.round(2)

print("FINAL MODEL COMPARISON")
print("=" * 100)
print(display_df.to_string(index=False))

# After comparing the initial models, tune the strongest candidates to search for better parameters.

# Search through different SVM settings to find a better combination.

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Use the same scaling and SVM structure while testing different parameter combinations.
# --------------------------------------------------

svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(probability=True))
])

# Define the XGBoost values that will be tested during grid search.
# --------------------------------------------------

param_grid = {
    "model__C": [0.1, 1, 10, 100, 1000],
    "model__gamma": [1, 0.1, 0.01, 0.001, 0.0001],
    "model__kernel": ["rbf"]
}

# Use the same five-fold stratified validation approach for a fair comparison.
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Test the selected XGBoost combinations and choose the one with the best macro F1.
# --------------------------------------------------

grid = GridSearchCV(
    estimator=svm_pipeline,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv,
    refit=True,
    verbose=3,
    n_jobs=-1
)

# Run the grid search on the training data.
grid.fit(X_train, y_train)

# Show the SVM settings that gave the best cross-validation score.
# --------------------------------------------------

print("\nBEST PARAMETERS:")
print(grid.best_params_)

print("\nBEST CV MACRO F1:")
print(f"{grid.best_score_:.4f}")
print(f"{grid.best_score_ * 100:.2f}%")

# Evaluate the tuned SVM on both the training and test sets.

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Use the best SVM found by the grid search.
best_svm = grid.best_estimator_

# Generate the final predictions on both training and test data.
y_train_pred_tuned = best_svm.predict(X_train)
y_test_pred_tuned = best_svm.predict(X_test)

# Calculate the final metrics for the tuned XGBoost model.
train_accuracy_tuned = accuracy_score(
    y_train, y_train_pred_tuned
)

test_accuracy_tuned = accuracy_score(
    y_test, y_test_pred_tuned
)

test_precision_tuned = precision_score(
    y_test,
    y_test_pred_tuned,
    average="macro"
)

test_recall_tuned = recall_score(
    y_test,
    y_test_pred_tuned,
    average="macro"
)

test_f1_tuned = f1_score(
    y_test,
    y_test_pred_tuned,
    average="macro"
)

train_test_gap_tuned = (
    train_accuracy_tuned - test_accuracy_tuned
)

# Print the tuned XGBoost performance for comparison with the tuned SVM.
# --------------------------------------------------

print("TUNED SVM RESULTS")
print("=" * 55)

print(f"Best Parameters: {grid.best_params_}")

print(
    f"\nTraining Accuracy:     "
    f"{train_accuracy_tuned:.4f} "
    f"({train_accuracy_tuned*100:.2f}%)"
)

print(
    f"CV Macro F1:           "
    f"{grid.best_score_:.4f} "
    f"({grid.best_score_*100:.2f}%)"
)

print(
    f"Test Accuracy:         "
    f"{test_accuracy_tuned:.4f} "
    f"({test_accuracy_tuned*100:.2f}%)"
)

print(
    f"Test Macro F1:         "
    f"{test_f1_tuned:.4f} "
    f"({test_f1_tuned*100:.2f}%)"
)

print(
    f"Test Macro Precision:   "
    f"{test_precision_tuned:.4f} "
    f"({test_precision_tuned*100:.2f}%)"
)

print(
    f"Test Macro Recall:      "
    f"{test_recall_tuned:.4f} "
    f"({test_recall_tuned*100:.2f}%)"
)

print(
    f"Train-Test Gap:        "
    f"{train_test_gap_tuned*100:.2f}%"
)

# Check the detailed performance for each emotion class.
# --------------------------------------------------

print("\nCLASSIFICATION REPORT")
print("=" * 55)

print(
    classification_report(
        y_test,
        y_test_pred_tuned
    )
)

# Tune XGBoost parameters to see if its performance can be improved.

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Convert the emotion names into numbers because XGBoost works with encoded class labels.
# --------------------------------------------------

encoder = LabelEncoder()

y_train_xgb = encoder.fit_transform(y_train)
y_test_xgb = encoder.transform(y_test)

# Set up the base XGBoost model before testing different parameter combinations.
# --------------------------------------------------

xgb_model = XGBClassifier(
    objective="multi:softmax",
    num_class=len(encoder.classes_),
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

# Define the XGBoost values that will be tested during grid search.
# --------------------------------------------------

xgb_param_grid = {
    "n_estimators": [200, 400],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.8],
    "colsample_bytree": [0.8]
}

# Use the same five-fold stratified validation approach for a fair comparison.
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Test the selected XGBoost combinations and choose the one with the best macro F1.
# --------------------------------------------------

xgb_grid = GridSearchCV(
    estimator=xgb_model,
    param_grid=xgb_param_grid,
    scoring="f1_macro",
    cv=cv,
    refit=True,
    verbose=2,
    n_jobs=-1
)

print("Tuning XGBoost...")

xgb_grid.fit(
    X_train,
    y_train_xgb
)

# --------------------------------------------------
# BEST RESULTS
# --------------------------------------------------

print("\nBEST XGBOOST PARAMETERS:")
print(xgb_grid.best_params_)

print("\nBEST XGBOOST CV MACRO F1:")
print(
    f"{xgb_grid.best_score_:.4f} "
    f"({xgb_grid.best_score_ * 100:.2f}%)"
)

# # Tuned XGB Classifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Keep the best XGBoost model found during tuning.
best_xgb = xgb_grid.best_estimator_

# Generate the final predictions on both training and test data.
y_train_pred_xgb = best_xgb.predict(X_train)
y_test_pred_xgb = best_xgb.predict(X_test)

# Calculate the final metrics for the tuned XGBoost model.
train_accuracy_xgb = accuracy_score(
    y_train_xgb,
    y_train_pred_xgb
)

test_accuracy_xgb = accuracy_score(
    y_test_xgb,
    y_test_pred_xgb
)

test_precision_xgb = precision_score(
    y_test_xgb,
    y_test_pred_xgb,
    average="macro"
)

test_recall_xgb = recall_score(
    y_test_xgb,
    y_test_pred_xgb,
    average="macro"
)

test_f1_xgb = f1_score(
    y_test_xgb,
    y_test_pred_xgb,
    average="macro"
)

train_test_gap_xgb = (
    train_accuracy_xgb - test_accuracy_xgb
)

# Print the tuned XGBoost performance for comparison with the tuned SVM.
# --------------------------------------------------

print("TUNED XGBOOST RESULTS")
print("=" * 60)

print("Best Parameters:")
print(xgb_grid.best_params_)

print(
    f"\nTraining Accuracy:   "
    f"{train_accuracy_xgb*100:.2f}%"
)

print(
    f"CV Macro F1:         "
    f"{xgb_grid.best_score_*100:.2f}%"
)

print(
    f"Test Accuracy:       "
    f"{test_accuracy_xgb*100:.2f}%"
)

print(
    f"Test Macro F1:       "
    f"{test_f1_xgb*100:.2f}%"
)

print(
    f"Test Macro Precision:"
    f" {test_precision_xgb*100:.2f}%"
)

print(
    f"Test Macro Recall:   "
    f"{test_recall_xgb*100:.2f}%"
)

print(
    f"Train-Test Gap:      "
    f"{train_test_gap_xgb*100:.2f}%"
)

print("\nCLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test_xgb,
        y_test_pred_xgb,
        target_names=encoder.classes_
    )
)

# The tuned SVM is used as the final model for the project.

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# Use the best SVM returned by the grid search as the final model.
# ==========================================================

final_model = grid.best_estimator_

# Generate the final predictions on both training and test data.
y_train_final = final_model.predict(X_train)
y_test_final = final_model.predict(X_test)

# Calculate the final model's accuracy, precision, recall and F1-score.
# ----------------------------------------------------------

train_accuracy = accuracy_score(
    y_train,
    y_train_final
)

test_accuracy = accuracy_score(
    y_test,
    y_test_final
)

test_precision = precision_score(
    y_test,
    y_test_final,
    average="macro"
)

test_recall = recall_score(
    y_test,
    y_test_final,
    average="macro"
)

test_f1 = f1_score(
    y_test,
    y_test_final,
    average="macro"
)

train_test_gap = train_accuracy - test_accuracy

# Print the final model performance and the parameters selected during tuning.
# ----------------------------------------------------------

print("FINAL MODEL: TUNED SVM")
print("=" * 60)

print("Best Parameters:")
print(grid.best_params_)

print(f"\nTraining Accuracy:    {train_accuracy*100:.2f}%")
print(f"CV Macro F1:          {grid.best_score_*100:.2f}%")
print(f"Test Accuracy:        {test_accuracy*100:.2f}%")
print(f"Test Macro F1:        {test_f1*100:.2f}%")
print(f"Test Macro Precision: {test_precision*100:.2f}%")
print(f"Test Macro Recall:    {test_recall*100:.2f}%")
print(f"Train-Test Gap:       {train_test_gap*100:.2f}%")

# Show detailed performance for each emotion in the final model.
# ----------------------------------------------------------

print("\nCLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_test_final
    )
)

# Visualize which emotions the final model gets right and which ones it confuses.
# ----------------------------------------------------------

classes = sorted(y_test.unique())

cm = confusion_matrix(
    y_test,
    y_test_final,
    labels=classes
)

fig, ax = plt.subplots(figsize=(9, 7))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)

disp.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    xticks_rotation=45
)

plt.title("Final SVM Confusion Matrix")
plt.xlabel("Predicted Emotion")
plt.ylabel("Actual Emotion")
plt.tight_layout()
plt.show()

# Save the trained final model so it can be reused later without retraining.

# Save the final tuned SVM to the project's models folder.
# ----------------------------------------------------------

import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "final_svm_model.pkl")


joblib.dump(
    final_model,
    MODEL_PATH
)

print("Final SVM model saved successfully.")
print("Path:", MODEL_PATH)

# Reload the saved model and make sure it produces the same predictions. 

# Load the saved model and compare its predictions with the original final predictions.
# ----------------------------------------------------------

loaded_model = joblib.load(MODEL_PATH)

test_predictions = loaded_model.predict(X_test)

print("Model loaded successfully.")
print("Predictions match:",
      (test_predictions == y_test_final).all()) 


