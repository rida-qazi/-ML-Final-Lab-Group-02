#!/usr/bin/env python
# coding: utf-8

# 
# ## ANALYTICS ENGINEERING

# In[43]:


"""
Analytics Engineer (AE) Pipeline
================================

Input: outputs/ML_Engineering/prediction_results.csv

Outputs:
    outputs/Analytics_Engineering/
        ├── ae_predictions.csv
        ├── ae_summary.csv
        ├── business_kpis.csv
        ├── confusion_cost_matrix.csv
        └── decision_threshold_tuning.csv

Responsibilities:
    1. Validate MLE prediction output
    2. Map emotions to business groups
    3. Calculate business-level metrics
    4. Apply business confusion costs
    5. Calculate negative-group probability
    6. Calculate Type I / Type II errors
    7. Tune decision thresholds
    8. Produce BI-ready KPI data
"""


import os
import pandas as pd


# In[44]:


# ============================================================
# 1. EMOTIONS
# ============================================================

EMOTIONS = [
    "angry",
    "calm",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
]


# In[45]:


# 2. PROJECT PATHS


# Expected project structure:
#
# project_root/
# ├── src/
# │   └── Analytics_Engineering/
# │       └── analytics_engineer.py
# │
# └── outputs/
#     ├── ML_Engineering/
#     │   └── prediction_results.csv
#     │
#     └── Analytics_Engineering/
#
# Going from:
# src/Analytics_Engineering/
#
# to:
# project_root/
#
# requires going up two directories.

import os

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

# ------------------------------------------------------------
# MLE input
# ------------------------------------------------------------

MLE_OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "ML_Engineering",
    "prediction_results.csv",
)

# ------------------------------------------------------------
# AE output
# ------------------------------------------------------------

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "Analytics_Engineering",
)

# Create output directory if it does not exist
os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# In[47]:


# ============================================================
# 3. MLE PROBABILITY COLUMNS
# ============================================================

# The MLE CSV uses:
#
# angry_probability
# calm_probability
# disgust_probability
# fearful_probability
# happy_probability
# neutral_probability
# sad_probability
# surprised_probability

PROB_COLUMNS = [
    f"{emotion}_probability"
    for emotion in EMOTIONS
]


# In[48]:


# 4. BUSINESS GROUPING

"""
Business-level grouping:
NEGATIVE:angry disgust fearful sad
POSITIVE: happy surprised
NEUTRAL: calm neutral
"""

BUSINESS_GROUP = {  "angry": "negative", "disgust": "negative", "fearful": "negative", "sad": "negative",
    "happy": "positive", "surprised": "positive",
    "calm": "neutral", "neutral": "neutral", }

GROUP_ORDER = ["negative","neutral","positive", ]


# In[49]:


# 5. BUSINESS COST MODEL

"""
Assumed business costs.
These are project assumptions and should be documented
as assumptions in the final report.

Same business group: 1
Negative -> Positive: 10
Negative -> Neutral: 7
Positive -> Negative: 8
Neutral -> Negative: 6
Other cross-group: 2
"""

UNIT_COST_SAME_GROUP = 1

UNIT_COST_NEG_TO_POS = 10
UNIT_COST_NEG_TO_NEUTRAL = 7

UNIT_COST_POS_TO_NEG = 8
UNIT_COST_NEUTRAL_TO_NEG = 6

UNIT_COST_OTHER_CROSS_GROUP = 2


def unit_cost(actual_emotion, predicted_emotion):
    """
    Return the assumed business cost for one
    actual-vs-predicted emotion pair.
    """

    if actual_emotion not in BUSINESS_GROUP:
        raise ValueError(
            f"Unknown actual emotion: {actual_emotion}"
        )

    if predicted_emotion not in BUSINESS_GROUP:
        raise ValueError(
            f"Unknown predicted emotion: {predicted_emotion}"
        )

    actual_group = BUSINESS_GROUP[actual_emotion]
    predicted_group = BUSINESS_GROUP[predicted_emotion]

    # Correct prediction = no error
    if actual_emotion == predicted_emotion:
        return 0

    # Same business group
    if actual_group == predicted_group:
        return UNIT_COST_SAME_GROUP

    # Negative -> Positive
    if (
        actual_group == "negative"
        and predicted_group == "positive"
    ):
        return UNIT_COST_NEG_TO_POS

    # Negative -> Neutral
    if (
        actual_group == "negative"
        and predicted_group == "neutral"
    ):
        return UNIT_COST_NEG_TO_NEUTRAL

    # Positive -> Negative
    if (
        actual_group == "positive"
        and predicted_group == "negative"
    ):
        return UNIT_COST_POS_TO_NEG

    # Neutral -> Negative
    if (
        actual_group == "neutral"
        and predicted_group == "negative"
    ):
        return UNIT_COST_NEUTRAL_TO_NEG

    # Remaining cross-group cases
    return UNIT_COST_OTHER_CROSS_GROUP


# In[50]:


# 6. NEGATIVE BUSINESS GROUP

NEGATIVE_EMOTIONS = [
    emotion
    for emotion, group in BUSINESS_GROUP.items()
    if group == "negative"
]

NEGATIVE_PROBABILITY_COLUMNS = [
    f"{emotion}_probability"
    for emotion in NEGATIVE_EMOTIONS
]


# In[51]:


# 7. THRESHOLD CONFIGURATION

THRESHOLDS_TO_TEST = [0.40, 0.50, 0.60, 0.70, 0.80,]

DEFAULT_OPERATING_THRESHOLD = 0.50

# Costs for binary negative-flag decision

FLAG_COST_FALSE_NEGATIVE = 10
FLAG_COST_FALSE_POSITIVE = 3


# In[52]:


# 8. LOAD MLE PREDICTIONS

def load_mle_predictions():
    """
    Load prediction_results.csv generated by MLE.
    """

    print("=" * 70)
    print("ANALYTICS ENGINEER PIPELINE")
    print("=" * 70)

    print("\nLoading MLE prediction output...")
    print(
        f"Input file:\n{MLE_OUTPUT_PATH}"
    )

    # Check that file exists
    if not os.path.exists(MLE_OUTPUT_PATH):
        raise FileNotFoundError(
            "\nMLE output file was not found.\n\n"
            f"Expected location:\n"
            f"{MLE_OUTPUT_PATH}\n\n"
            "Make sure the MLE pipeline has generated "
            "prediction_results.csv."
        )

    # Read CSV
    df = pd.read_csv(
        MLE_OUTPUT_PATH
    )

    print(
        f"\nLoaded {len(df):,} prediction records."
    )

    return df


# In[53]:


# 9. VALIDATE MLE OUTPUT

def validate_input(df):
    """
    Validate the structure and values of the MLE output.
    """

    print("\n" + "=" * 70)
    print("INPUT VALIDATION")
    print("=" * 70)

    # Required columns
    required_columns = [
        "file_path",
        "actor_id",
        "actual_emotion",
        "predicted_emotion",
        "prediction_confidence",
        "inference_time_ms",
    ] + PROB_COLUMNS

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "\nMissing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_columns
            )
        )

    print("✓ Required columns are present")

    # Empty dataset
    if len(df) == 0:
        raise ValueError(
            "The MLE prediction file is empty."
        )

    print("✓ Dataset is not empty")

    # Validate actual emotions
    invalid_actual = (
        set(df["actual_emotion"].dropna())
        - set(EMOTIONS)
    )

    if invalid_actual:
        raise ValueError(
            "Invalid actual emotions found:\n"
            f"{invalid_actual}"
        )

    # Validate predicted emotions
    invalid_predicted = (
        set(df["predicted_emotion"].dropna())
        - set(EMOTIONS)
    )

    if invalid_predicted:
        raise ValueError(
            "Invalid predicted emotions found:\n"
            f"{invalid_predicted}"
        )

    print("✓ Emotion labels are valid")

    # Check missing values
    important_columns = [
        "actual_emotion",
        "predicted_emotion",
        "prediction_confidence",
    ] + PROB_COLUMNS

    missing_counts = (
        df[important_columns]
        .isnull()
        .sum()
    )

    missing_values = missing_counts[
        missing_counts > 0
    ]

    if len(missing_values) > 0:
        print("\nWARNING: Missing values detected:")

        for column, count in missing_values.items():
            print(f"  {column}: {count}")
    else:
        print("✓ No missing values in important columns")

    # Probability validation
    for column in PROB_COLUMNS:
        invalid_probability = (
            (df[column] < 0)
            | (df[column] > 1)
        ).sum()

        if invalid_probability > 0:
            raise ValueError(
                f"Invalid probability values found "
                f"in {column}: "
                f"{invalid_probability}"
            )

    print("✓ Probability values are between 0 and 1")

    # Confidence validation
    invalid_confidence = (
        (df["prediction_confidence"] < 0)
        | (df["prediction_confidence"] > 1)
    ).sum()

    if invalid_confidence > 0:
        raise ValueError(
            "Invalid prediction confidence values found."
        )

    print("✓ Prediction confidence values are valid")

    print("\nInput validation completed successfully.")


# In[54]:


# 10. ADD BUSINESS GROUPS

def add_business_groups(df):
    """
    Map the eight emotion classes to three
    business-level categories.
    """

    print("\n" + "=" * 70)
    print("BUSINESS GROUP TRANSFORMATION")
    print("=" * 70)

    # Actual group
    df["actual_group"] = (
        df["actual_emotion"]
        .map(BUSINESS_GROUP)
    )

    # Predicted group
    df["predicted_group"] = (
        df["predicted_emotion"]
        .map(BUSINESS_GROUP)
    )

    # Business-group correctness
    df["business_group_correct"] = (
        df["actual_group"] == df["predicted_group"]
    )

    # Type I / Type II error classification
    df["error_type"] = "Correct"

    # Type I = False Positive
    # Actually non-negative, predicted as negative
    df.loc[
        (df["actual_group"] != "negative") &
        (df["predicted_group"] == "negative"),
        "error_type"
    ] = "Type I(FP)"

    # Type II = False Negative
    # Actually negative, predicted as non-negative
    df.loc[
        (df["actual_group"] == "negative") &
        (df["predicted_group"] != "negative"),
        "error_type"
    ] = "Type II(FN)"

    # Display distribution
    print("\nActual business-group distribution:")

    distribution = (
        df["actual_group"]
        .value_counts()
        .reindex(GROUP_ORDER)
        .fillna(0)
        .astype(int)
    )

    print(distribution)

    return df

# In[55]:


# 11. CALCULATE BUSINESS COST

def calculate_business_cost(df):
    """
    Calculate the assumed business cost for
    every prediction.
    """

    print("\n" + "=" * 70)
    print("BUSINESS COST CALCULATION")
    print("=" * 70)

    # Cost for each prediction
    df["unit_cost"] = df.apply(
        lambda row: unit_cost(
            row["actual_emotion"],
            row["predicted_emotion"],
        ),
        axis=1,
    )

    # Total cost
    total_business_cost = df["unit_cost"].sum()

    # Average cost
    average_business_cost = df["unit_cost"].mean()

    # Business group accuracy
    business_group_accuracy = df["business_group_correct"].mean()

    print(
        f"Total business cost: "
        f"{total_business_cost:.2f}"
    )

    print(
        f"Average business cost: "
        f"{average_business_cost:.4f}"
    )

    print(
        f"Business-group accuracy: "
        f"{business_group_accuracy:.4f}"
    )

    return df


# In[56]:


# 12. CREATE CONFUSION COST MATRIX

def create_cost_matrix():
    """
    Create an 8 x 8 emotion-level business cost matrix.

    Rows: Actual emotion
    Columns: Predicted emotion
    """

    print("\n" + "=" * 70)
    print("CONFUSION COST MATRIX")
    print("=" * 70)

    cost_matrix = pd.DataFrame(
        index=EMOTIONS,
        columns=EMOTIONS,
        dtype=float,
    )

    for actual in EMOTIONS:
        for predicted in EMOTIONS:
            cost_matrix.loc[
                actual,
                predicted
            ] = unit_cost(
                actual,
                predicted,
            )

    cost_matrix.index.name = "Actual \\ Predicted"

    print("\nCost matrix:")
    print("Rows = Actual emotion | Columns = Predicted emotion")
    print(cost_matrix)

    # Save
    output_path = os.path.join(
        OUTPUT_DIR,
        "confusion_cost_matrix.csv",
    )

    cost_matrix.to_csv(
        output_path
    )

    print(
        f"\nSaved cost matrix to:\n"
        f"{output_path}"
    )

    return cost_matrix


# In[57]:


# 13. CALCULATE NEGATIVE PROBABILITY

def calculate_negative_probability(df):
    """
    Calculate combined probability for the negative business group.
    Negative emotions:angry, disgust, fearful, sad
    """
    print("\n" + "=" * 70)
    print("NEGATIVE PROBABILITY CALCULATION")
    print("=" * 70)

    df["negative_probability"] = (
        df[
            NEGATIVE_PROBABILITY_COLUMNS
        ]
        .sum(axis=1)
    )

    print(
        "\nNegative probability statistics:"
    )

    print(
        df["negative_probability"]
        .describe()
    )

    return df


# In[58]:


# 14. APPLY DEFAULT THRESHOLD

def apply_default_threshold(df):
    """
    Apply the default operating threshold.
    If negative_probability >= threshold:
        predicted_negative = 1
    Otherwise:
        predicted_negative = 0
    """

    print("\n" + "=" * 70)
    print("DEFAULT THRESHOLD ANALYSIS")
    print("=" * 70)

    threshold = DEFAULT_OPERATING_THRESHOLD

    # Actual negative status
    df["actual_negative"] = (
        df["actual_group"] == "negative"
    ).astype(int)

    # Predicted negative status
    df["predicted_negative"] = (
        df["negative_probability"] >= threshold
    ).astype(int)

    # True Positive
    df["true_positive"] = (
        (df["predicted_negative"] == 1)
        & (df["actual_negative"] == 1)
    ).astype(int)

    # True Negative
    df["true_negative"] = (
        (df["predicted_negative"] == 0)
        & (df["actual_negative"] == 0)
    ).astype(int)

    # False Positive
    df["false_positive"] = (
        (df["predicted_negative"] == 1)
        & (df["actual_negative"] == 0)
    ).astype(int)

    # False Negative
    df["false_negative"] = (
        (df["predicted_negative"] == 0)
        & (df["actual_negative"] == 1)
    ).astype(int)

    # Counts
    tp = df["true_positive"].sum()
    tn = df["true_negative"].sum()
    fp = df["false_positive"].sum()
    fn = df["false_negative"].sum()

    print(f"\nOperating threshold: {threshold}")
    print(f"True positives:  {tp:,}")
    print(f"True negatives:  {tn:,}")
    print(f"False positives: {fp:,}")
    print(f"False negatives: {fn:,}")

    # Error costs
    false_positive_cost = fp * FLAG_COST_FALSE_POSITIVE

    false_negative_cost = fn * FLAG_COST_FALSE_NEGATIVE

    total_threshold_cost = (
        false_positive_cost + false_negative_cost
    )

    print(
        f"\nFalse positive cost: "
        f"{false_positive_cost:.2f}"
    )

    print(
        f"False negative cost: "
        f"{false_negative_cost:.2f}"
    )

    print(
        f"Total threshold cost: "
        f"{total_threshold_cost:.2f}"
    )

    return df


# In[59]:


# 15. DECISION THRESHOLD TUNING=

def tune_thresholds(df):
    """
    Test multiple decision thresholds.

    Produces:
        decision_threshold_tuning.csv
    """

    print("\n" + "=" * 70)
    print("DECISION THRESHOLD TUNING")
    print("=" * 70)

    results = []

    actual_negative = df["actual_group"] == "negative"

    for threshold in THRESHOLDS_TO_TEST:
        # Predicted negative
        predicted_negative = (
            df["negative_probability"] >= threshold
        )

        # Confusion counts
        true_positive = (
            predicted_negative & actual_negative
        ).sum()

        true_negative = (
            ~predicted_negative & ~actual_negative
        ).sum()

        false_positive = (
            predicted_negative & ~actual_negative
        ).sum()

        false_negative = (
            ~predicted_negative & actual_negative
        ).sum()

        # Costs
        false_positive_cost = (
            false_positive * FLAG_COST_FALSE_POSITIVE
        )

        false_negative_cost = (
            false_negative * FLAG_COST_FALSE_NEGATIVE
        )

        total_cost = (
            false_positive_cost + false_negative_cost
        )

        # Precision
        precision_denominator = true_positive + false_positive

        if precision_denominator > 0:
            precision = (
                true_positive / precision_denominator
            )
        else:
            precision = 0

        # Recall
        recall_denominator = true_positive + false_negative

        if recall_denominator > 0:
            recall = (
                true_positive / recall_denominator
            )
        else:
            recall = 0

        # Accuracy
        accuracy = (
            true_positive + true_negative
        ) / len(df)

        # Save result
        results.append({
            "threshold": threshold,
            "true_positive_count": int(true_positive),
            "true_negative_count": int(true_negative),
            "false_positive_count": int(false_positive),
            "false_negative_count": int(false_negative),
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "false_positive_cost": int(false_positive_cost),
            "false_negative_cost": int(false_negative_cost),
            "total_cost": int(total_cost),
        })

    threshold_df = pd.DataFrame(results)

    print("\nThreshold analysis:")
    print(threshold_df.to_string(index=False))

    # Save
    output_path = os.path.join(
        OUTPUT_DIR,
        "decision_threshold_tuning.csv",
    )

    threshold_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved threshold analysis to:\n"
        f"{output_path}"
    )

    return threshold_df


# In[60]:


# 16. CREATE AE SUMMARY

def create_summary(df):
    """
    Create overall AE summary metrics.
    """

    print("\n" + "=" * 70)
    print("AE SUMMARY")
    print("=" * 70)

    # Basic metrics
    total_predictions = len(df)

    emotion_accuracy = (
        df["actual_emotion"] == df["predicted_emotion"]
    ).mean()

    business_group_accuracy = df["business_group_correct"].mean()

    average_confidence = df["prediction_confidence"].mean()

    average_inference_time = df["inference_time_ms"].mean()

    # Business cost
    total_business_cost = df["unit_cost"].sum()

    average_business_cost = df["unit_cost"].mean()

    # Default threshold metrics
    false_positive_count = df["false_positive"].sum()

    false_negative_count = df["false_negative"].sum()

    false_positive_cost = (
        false_positive_count * FLAG_COST_FALSE_POSITIVE
    )

    false_negative_cost = (
        false_negative_count * FLAG_COST_FALSE_NEGATIVE
    )

    total_threshold_cost = (
        false_positive_cost + false_negative_cost
    )

    # Summary dictionary
    summary = {
        "total_predictions": total_predictions,
        "emotion_accuracy": emotion_accuracy,
        "business_group_accuracy": business_group_accuracy,
        "average_prediction_confidence": average_confidence,
        "average_inference_time_ms": average_inference_time,
        "total_business_cost": total_business_cost,
        "average_business_cost": average_business_cost,
        "default_threshold": DEFAULT_OPERATING_THRESHOLD,
        "false_positive_count": int(false_positive_count),
        "false_negative_count": int(false_negative_count),
        "false_positive_cost": int(false_positive_cost),
        "false_negative_cost": int(false_negative_cost),
        "total_threshold_cost": int(total_threshold_cost),
    }

    summary_df = pd.DataFrame([summary])

    print(summary_df.to_string(index=False))

    # Save
    output_path = os.path.join(
        OUTPUT_DIR,
        "ae_summary.csv",
    )

    summary_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved AE summary to:\n"
        f"{output_path}"
    )

    return summary_df


# In[61]:


# 17. CREATE BI KPI FILE

def create_business_kpis(df):
    """
    Create a simple KPI table specifically for the BI / Power BI developer.
    The BI developer can import business_kpis.csv and use the values for executive KPI cards.
    """

    print("\n" + "=" * 70)
    print("BI KPI OUTPUT")
    print("=" * 70)

    # KPI calculations
    total_predictions = len(df)

    emotion_accuracy = (df["actual_emotion"] == df["predicted_emotion"]).mean()

    business_group_accuracy = df["business_group_correct"].mean()

    average_confidence = df["prediction_confidence"].mean()

    average_inference_time = df["inference_time_ms"].mean()

    total_business_cost = df["unit_cost"].sum()

    average_business_cost = df["unit_cost"].mean()

    false_positive_count = df["false_positive"].sum()

    false_negative_count = df["false_negative"].sum()

    false_positive_cost = (false_positive_count * FLAG_COST_FALSE_POSITIVE)

    false_negative_cost = (false_negative_count * FLAG_COST_FALSE_NEGATIVE)

    total_threshold_cost = (false_positive_cost + false_negative_cost)

    # KPI table
    kpis = [
        {
            "metric": "Total Predictions",
            "value": total_predictions,
            "unit": "records",
        },
        {
            "metric": "Emotion Accuracy",
            "value": emotion_accuracy,
            "unit": "ratio",
        },
        {
            "metric": "Business Group Accuracy",
            "value": business_group_accuracy,
            "unit": "ratio",
        },
        {
            "metric": "Average Prediction Confidence",
            "value": average_confidence,
            "unit": "ratio",
        },
        {
            "metric": "Average Inference Time (ms)",
            "value": average_inference_time,
            "unit": "ms",
        },
        {
            "metric": "Total Business Cost",
            "value": total_business_cost,
            "unit": "cost units",
        },
        {
            "metric": "Average Business Cost",
            "value": average_business_cost,
            "unit": "cost units/record",
        },
        {
            "metric": "False Positive Count",
            "value": false_positive_count,
            "unit": "records",
        },
        {
            "metric": "False Negative Count",
            "value": false_negative_count,
            "unit": "records",
        },
        {
            "metric": "False Positive Cost",
            "value": false_positive_cost,
            "unit": "cost units",
        },
        {
            "metric": "False Negative Cost",
            "value": false_negative_cost,
            "unit": "cost units",
        },
        {
            "metric": "Total Threshold Cost",
            "value": total_threshold_cost,
            "unit": "cost units",
        },
    ]

    kpi_df = pd.DataFrame(kpis)
    print(kpi_df.to_string(index=False))

    # Save
    output_path = os.path.join(
        OUTPUT_DIR,
        "business_kpis.csv",
    )

    kpi_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved BI KPI table to:\n"
        f"{output_path}"
    )

    return kpi_df


# In[62]:


# 18. SAVE AE PREDICTIONS

def save_ae_predictions(df):

    # Keep only the dataset-relative path.
    # This removes the user's local computer path.
    df["file_path"] = (
        df["file_path"]
        .str.split(
            "Audio_Speech_Actors_01-24_16k",
            n=1
        )
        .str[-1]
        .str.lstrip("\\/")
    )

    # Add the dataset folder name back
    df["file_path"] = (
        "Audio_Speech_Actors_01-24_16k\\"
        + df["file_path"]
    )

    # Save the original MLE predictions
    # plus all AE-derived fields.
    output_path = os.path.join(
        OUTPUT_DIR,
        "ae_predictions.csv",
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved AE prediction dataset to:\n"
        f"{output_path}"
    )


# In[63]:


#19. MAIN PIPELINE

def main():
    # Step 1 - Load MLE output
    df = load_mle_predictions()

    # Step 2 - Validate input
    validate_input(df)

    # Step 3 - Add business groups
    df = add_business_groups(df)

    # Step 4 - Calculate business costs
    df = calculate_business_cost(df)

    # Step 5 - Create cost matrix
    create_cost_matrix()

    # Step 6 - Calculate negative probability
    df = calculate_negative_probability(df)

    # Step 7 - Apply default threshold
    df = apply_default_threshold(df)

    # Step 8 - Tune thresholds
    threshold_df = tune_thresholds(df)

    # Step 9 - Create AE summary
    create_summary(df)

    # Step 10 - Create BI KPI table
    create_business_kpis(df)

    # Step 11 - Save AE predictions
    save_ae_predictions(df)

    # Final message
    print("\n" + "=" * 70)
    print("ANALYTICS ENGINEER PIPELINE COMPLETED")
    print("=" * 70)

    print("\nInput:")
    print(MLE_OUTPUT_PATH)

    print("\nGenerated outputs:")
    print(f"1. {os.path.join(OUTPUT_DIR, 'ae_predictions.csv')}")
    print(f"2. {os.path.join(OUTPUT_DIR, 'ae_summary.csv')}")
    print(f"3. {os.path.join(OUTPUT_DIR, 'business_kpis.csv')}")
    print(f"4. {os.path.join(OUTPUT_DIR, 'confusion_cost_matrix.csv')}")
    print(f"5. {os.path.join(OUTPUT_DIR, 'decision_threshold_tuning.csv')}")

    print("\nAE → BI handoff is ready.")


# ================20. RUN SCRIPT================

if __name__ == "__main__":
    main()


# In[ ]:




