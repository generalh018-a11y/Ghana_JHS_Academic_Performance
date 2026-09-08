"""
STEP 26 — THRESHOLD SENSITIVITY ANALYSIS

Purpose:
    Evaluate the robustness of the locked primary ensemble
    across alternative classification thresholds.

Thresholds:
    0.30, 0.40, 0.50, 0.60, 0.70

Important:
    - No model is retrained.
    - No ensemble weight is changed.
    - No threshold is selected using test performance.
    - This is a sensitivity analysis of the locked probabilities.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]


# ============================================================
# INPUT FILE
# ============================================================

PREDICTION_FILE = (
    PROCESSED_DIR /
    "ten_seed_ensemble_test_predictions.csv"
)


# ============================================================
# LOAD PREDICTIONS
# ============================================================

print("=" * 75)
print("STEP 26 — THRESHOLD SENSITIVITY ANALYSIS")
print("=" * 75)

print("\nLoading locked primary ensemble predictions...")

df = pd.read_csv(PREDICTION_FILE)

print(f"Rows loaded: {len(df)}")

print("\nAvailable columns:")
print(list(df.columns))


# ============================================================
# IDENTIFY REQUIRED COLUMNS
# ============================================================

possible_target_columns = [
    "actual",
    "y_true",
    "true_label",
    "at_risk",
    "target",
]

possible_probability_columns = [
    "ensemble_probability",
    "ensemble_prob",
    "probability",
    "predicted_probability",
]


target_col = None
prob_col = None


for col in possible_target_columns:
    if col in df.columns:
        target_col = col
        break


for col in possible_probability_columns:
    if col in df.columns:
        prob_col = col
        break


if target_col is None:
    raise ValueError(
        "Could not identify the true target column."
    )

if prob_col is None:
    raise ValueError(
        "Could not identify the ensemble probability column."
    )


print(f"\nTarget column       : {target_col}")
print(f"Probability column  : {prob_col}")


# ============================================================
# PREPARE DATA
# ============================================================

y_true = df[target_col].astype(int)
probabilities = df[prob_col].astype(float)


if len(y_true) != len(probabilities):
    raise ValueError(
        "Target and probability lengths do not match."
    )


if probabilities.isna().any():
    raise ValueError(
        "Missing ensemble probabilities detected."
    )


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

results = []

prediction_rows = []


for threshold in THRESHOLDS:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    predicted_at_risk = int(
        predictions.sum()
    )

    predicted_not_at_risk = int(
        len(predictions) - predictions.sum()
    )

    results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "macro_f1": macro_f1,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "predicted_at_risk": predicted_at_risk,
        "predicted_not_at_risk": predicted_not_at_risk,
    })

    temp = pd.DataFrame({
        "threshold": threshold,
        "actual_at_risk": y_true,
        "ensemble_probability": probabilities,
        "predicted_at_risk": predictions,
    })

    prediction_rows.append(temp)


results_df = pd.DataFrame(results)

threshold_predictions_df = pd.concat(
    prediction_rows,
    ignore_index=True
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = (
    PROCESSED_DIR /
    "threshold_sensitivity_results.csv"
)

predictions_file = (
    PROCESSED_DIR /
    "threshold_sensitivity_predictions.csv"
)

results_df.to_csv(
    results_file,
    index=False
)

threshold_predictions_df.to_csv(
    predictions_file,
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 75)
print("THRESHOLD SENSITIVITY RESULTS")
print("=" * 75)

print(
    results_df[
        [
            "threshold",
            "accuracy",
            "precision",
            "recall",
            "macro_f1",
            "TN",
            "FP",
            "FN",
            "TP",
            "predicted_at_risk",
        ]
    ].to_string(index=False)
)


# ============================================================
# COMPARE WITH BASELINE THRESHOLD
# ============================================================

baseline_row = results_df[
    results_df["threshold"] == 0.50
].iloc[0]

print("\n")
print("=" * 75)
print("REFERENCE — LOCKED 0.50 THRESHOLD")
print("=" * 75)

print(
    f"Accuracy : {baseline_row['accuracy']:.6f}"
)

print(
    f"Precision: {baseline_row['precision']:.6f}"
)

print(
    f"Recall   : {baseline_row['recall']:.6f}"
)

print(
    f"Macro-F1 : {baseline_row['macro_f1']:.6f}"
)

print(
    f"TN={int(baseline_row['TN'])}, "
    f"FP={int(baseline_row['FP'])}, "
    f"FN={int(baseline_row['FN'])}, "
    f"TP={int(baseline_row['TP'])}"
)


# ============================================================
# EARLY-WARNING INTERPRETATION
# ============================================================

print("\n")
print("=" * 75)
print("EARLY-WARNING TRADE-OFF")
print("=" * 75)

for _, row in results_df.iterrows():

    print(
        f"Threshold {row['threshold']:.2f}: "
        f"Recall={row['recall']:.4f}, "
        f"Precision={row['precision']:.4f}, "
        f"FN={int(row['FN'])}, "
        f"FP={int(row['FP'])}"
    )


# ============================================================
# COMPLETION
# ============================================================

print("\n")
print("=" * 75)
print("STEP 26 COMPLETED SUCCESSFULLY")
print("=" * 75)

print("\nCreated:")
print(results_file)
print(predictions_file)

print("\n✓ No model retraining performed")
print("✓ Ensemble weights unchanged")
print("✓ Five thresholds evaluated")
print("✓ Threshold sensitivity results saved")

print("\nSTOP HERE.")
print("Do NOT change the locked threshold based on these results.")