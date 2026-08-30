import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

print("=" * 70)
print("STEP 16.15B — CLASS-LEVEL FAIRNESS ANALYSIS")
print("=" * 70)

INPUT_FILE = "data/processed/fairness_test_predictions.csv"
OUTPUT_FILE = "data/processed/fairness_class_results.csv"
GAP_FILE = "data/processed/fairness_class_gaps.csv"

# ------------------------------------------------------------
# LOAD EXISTING HELD-OUT TEST PREDICTIONS
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("\nLoaded fairness test predictions:")
print("Shape:", df.shape)

required_columns = [
    "class_level",
    "actual_score",
    "prediction"
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Required column missing: {col}")

print("✓ Required columns confirmed.")

# ------------------------------------------------------------
# DEFINE AT-RISK TARGET
# ------------------------------------------------------------

# Same operational definition used in the classification analysis:
# overall average score below 50 = at risk.

df["actual_at_risk"] = (
    df["actual_score"] < 50
).astype(int)

df["predicted_at_risk"] = (
    df["prediction"] < 50
).astype(int)

print("\nAt-risk definition:")
print("Score < 50 = At Risk")
print("Score >= 50 = Not At Risk")

# ------------------------------------------------------------
# FAIRNESS BY CLASS LEVEL
# ------------------------------------------------------------

results = []

print("\n" + "=" * 70)
print("FAIRNESS ANALYSIS BY CLASS LEVEL")
print("=" * 70)

for class_level in ["JHS1", "JHS2", "JHS3"]:

    group = df[df["class_level"] == class_level].copy()

    if len(group) == 0:
        print(f"\nWARNING: No records found for {class_level}")
        continue

    y_true = group["actual_at_risk"]
    y_pred = group["predicted_at_risk"]

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    if (fp + tn) > 0:
        fpr = fp / (fp + tn)
    else:
        fpr = 0.0

    results.append({
        "class_level": class_level,
        "n": len(group),
        "accuracy": accuracy,
        "precision": precision,
        "recall_tpr": recall,
        "fpr": fpr,
        "macro_f1": macro_f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    })

results_df = pd.DataFrame(results)

print("\n")
print(
    results_df.to_string(
        index=False,
        formatters={
            "accuracy": "{:.4f}".format,
            "precision": "{:.4f}".format,
            "recall_tpr": "{:.4f}".format,
            "fpr": "{:.4f}".format,
            "macro_f1": "{:.4f}".format
        }
    )
)

# ------------------------------------------------------------
# FAIRNESS GAPS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CLASS-LEVEL FAIRNESS GAPS")
print("=" * 70)

metrics = [
    "accuracy",
    "precision",
    "recall_tpr",
    "fpr",
    "macro_f1"
]

gap_results = []

for metric in metrics:

    maximum = results_df[metric].max()
    minimum = results_df[metric].min()

    max_group = results_df.loc[
        results_df[metric].idxmax(),
        "class_level"
    ]

    min_group = results_df.loc[
        results_df[metric].idxmin(),
        "class_level"
    ]

    gap = maximum - minimum

    gap_results.append({
        "metric": metric,
        "max_group": max_group,
        "min_group": min_group,
        "gap": gap
    })

    print(
        f"{metric:15s}: "
        f"{gap:.4f} "
        f"({max_group} vs {min_group})"
    )

gaps_df = pd.DataFrame(gap_results)

# ------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

gaps_df.to_csv(
    GAP_FILE,
    index=False
)

print("\n" + "=" * 70)
print("STEP 16.15B COMPLETE")
print("=" * 70)

print("\nSaved:")
print(OUTPUT_FILE)
print(GAP_FILE)

print("\n✓ Class-level fairness calculated.")
print("✓ JHS1 evaluated.")
print("✓ JHS2 evaluated.")
print("✓ JHS3 evaluated.")
print("✓ Accuracy calculated.")
print("✓ Precision calculated.")
print("✓ Recall/TPR calculated.")
print("✓ FPR calculated.")
print("✓ Macro-F1 calculated.")
print("✓ Fairness gaps calculated.")

print("\nNext stage:")
print("Update final Results and Analysis tables.")