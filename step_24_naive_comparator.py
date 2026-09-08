from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)


# ============================================================
# STEP 24
# NAIVE MAJORITY-CLASS COMPARATOR
# ============================================================

print("=" * 75)
print("STEP 24 — NAIVE MAJORITY-CLASS COMPARATOR")
print("=" * 75)


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"


# ------------------------------------------------------------
# 2. LOAD TEST DATA
# ------------------------------------------------------------

test = pd.read_csv(
    DATA_DIR / "test_student_level.csv"
)

y_test = (
    test["at_risk"]
    .astype(int)
    .to_numpy()
)


# ------------------------------------------------------------
# 3. DETERMINE MAJORITY CLASS FROM TRAINING DATA ONLY
#
# IMPORTANT:
# The naive rule is determined from the training data.
# The test set is not used to determine the majority class.
# ------------------------------------------------------------

train = pd.read_csv(
    DATA_DIR / "train_student_level.csv"
)

y_train = (
    train["at_risk"]
    .astype(int)
    .to_numpy()
)


train_class_counts = pd.Series(
    y_train
).value_counts()


MAJORITY_CLASS = int(
    train_class_counts.idxmax()
)


print("\nTraining-set target counts:")

print(
    train_class_counts
    .sort_index()
)


print(
    "\nMajority class selected from training data:"
)

if MAJORITY_CLASS == 0:
    print("0 = Not at risk")
else:
    print("1 = At risk")


# ------------------------------------------------------------
# 4. CREATE NAIVE PREDICTIONS
# ------------------------------------------------------------

naive_prediction = np.full(
    len(y_test),
    MAJORITY_CLASS,
    dtype=int
)


# ------------------------------------------------------------
# 5. NAIVE PROBABILITY
#
# Use the training-set majority-class prevalence as the
# constant probability estimate.
# ------------------------------------------------------------

majority_probability = (
    np.mean(y_train)
)

if MAJORITY_CLASS == 1:

    naive_probability = np.full(
        len(y_test),
        majority_probability
    )

else:

    naive_probability = np.full(
        len(y_test),
        majority_probability
    )


# ------------------------------------------------------------
# 6. METRICS
# ------------------------------------------------------------

tn, fp, fn, tp = confusion_matrix(
    y_test,
    naive_prediction,
    labels=[0, 1]
).ravel()


accuracy = accuracy_score(
    y_test,
    naive_prediction
)

precision = precision_score(
    y_test,
    naive_prediction,
    zero_division=0
)

recall = recall_score(
    y_test,
    naive_prediction,
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    naive_prediction,
    average="macro",
    zero_division=0
)


# AUC metrics are undefined for a constant prediction score.
# We therefore record them as NaN rather than inventing a value.

try:
    auc_roc = roc_auc_score(
        y_test,
        naive_probability
    )
except ValueError:
    auc_roc = np.nan


try:
    auc_pr = average_precision_score(
        y_test,
        naive_probability
    )
except ValueError:
    auc_pr = np.nan


brier = brier_score_loss(
    y_test,
    naive_probability
)


# ------------------------------------------------------------
# 7. DISPLAY RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("NAIVE COMPARATOR TEST RESULTS")
print("=" * 75)

print(
    f"Accuracy       : {accuracy:.6f}"
)

print(
    f"Precision      : {precision:.6f}"
)

print(
    f"Recall         : {recall:.6f}"
)

print(
    f"Macro-F1       : {macro_f1:.6f}"
)

print(
    f"AUC-ROC        : {auc_roc}"
)

print(
    f"AUC-PR         : {auc_pr}"
)

print(
    f"Brier score    : {brier:.6f}"
)

print("\nConfusion matrix components:")

print(
    f"TN = {tn}"
)

print(
    f"FP = {fp}"
)

print(
    f"FN = {fn}"
)

print(
    f"TP = {tp}"
)


# ------------------------------------------------------------
# 8. SAVE RESULTS
# ------------------------------------------------------------

result = pd.DataFrame(
    [
        {
            "model": "Naive Majority-Class Comparator",
            "split": "test",
            "majority_class": MAJORITY_CLASS,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "macro_f1": macro_f1,
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
            "brier_score": brier,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }
    ]
)


output_file = (
    DATA_DIR
    / "naive_comparator_results.csv"
)


result.to_csv(
    output_file,
    index=False
)


# ------------------------------------------------------------
# 9. SAVE PREDICTIONS
# ------------------------------------------------------------

predictions = pd.DataFrame(
    {
        "y_true": y_test,
        "naive_probability": naive_probability,
        "naive_prediction": naive_prediction,
    }
)


predictions_file = (
    DATA_DIR
    / "naive_comparator_predictions.csv"
)


predictions.to_csv(
    predictions_file,
    index=False
)


# ------------------------------------------------------------
# 10. FINAL CHECKS
# ------------------------------------------------------------

assert len(result) == 1

assert len(predictions) == len(test)

assert MAJORITY_CLASS in [0, 1]

assert output_file.exists()
assert predictions_file.exists()


print("\n" + "=" * 75)
print("STEP 24 COMPLETED SUCCESSFULLY")
print("=" * 75)

print("\nCreated:")
print(output_file)
print(predictions_file)

print("\n✓ Majority class determined from training data")
print("✓ Test data used only for comparator evaluation")
print("✓ No model fitting or tuning performed")
print("✓ Naive comparator ready for comparison with ML models")

print("\nSTOP HERE.")
print("Do NOT modify the comparator based on its test performance.")