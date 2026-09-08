"""
STEP 27B — FAIRNESS / SUBGROUP PERFORMANCE ANALYSIS

Purpose:
    Evaluate the locked primary ensemble across gender and
    class-level subgroups.

Subgroups:
    Gender: Female (F), Male (M)
    Class: JHS1, JHS2

Protocol:
    - Uses the fairness-ready mapped prediction file.
    - Calculates metrics separately for each seed.
    - Reports mean ± SD across ten seeds.
    - Does not retrain models.
    - Does not modify the locked model.
    - Does not select a threshold or model based on subgroup
      test performance.

Fairness metrics:
    Accuracy
    Precision
    Recall / TPR
    Macro-F1
    FPR
    FNR
    TN, FP, FN, TP

Fairness gaps:
    - Absolute TPR gap
    - Absolute FPR gap
    - Absolute precision gap
    - Absolute accuracy gap
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

INPUT_FILE = (
    PROCESSED_DIR /
    "fairness_analysis_mapped_predictions.csv"
)

THRESHOLD = 0.50


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("STEP 27B — FAIRNESS / SUBGROUP PERFORMANCE ANALYSIS")
print("=" * 75)

print("\nLoading fairness-ready predictions...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df)}")
print(f"Columns loaded: {len(df.columns)}")


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "student_id",
    "academic_year",
    "class_level",
    "gender",
    "at_risk",
    "seed",
    "ensemble_probability",
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("\n✓ All required columns are present.")


# ============================================================
# BASIC VALIDATION
# ============================================================

expected_seeds = list(range(42, 52))

actual_seeds = sorted(
    df["seed"].unique().tolist()
)

if actual_seeds != expected_seeds:
    raise ValueError(
        f"Expected seeds {expected_seeds}, "
        f"found {actual_seeds}"
    )

print(
    f"✓ Ten seeds confirmed: {expected_seeds}"
)


# ============================================================
# PREDICTION GENERATION
# ============================================================

df["prediction_at_05"] = (
    df["ensemble_probability"] >= THRESHOLD
).astype(int)


# ============================================================
# METRIC FUNCTION
# ============================================================

def subgroup_metrics(group):

    y_true = group["at_risk"].astype(int)
    y_pred = group["prediction_at_05"].astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

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

    # False Positive Rate:
    # FP / (FP + TN)
    if (fp + tn) > 0:
        fpr = fp / (fp + tn)
    else:
        fpr = np.nan

    # False Negative Rate:
    # FN / (FN + TP)
    if (fn + tp) > 0:
        fnr = fn / (fn + tp)
    else:
        fnr = np.nan

    return {
        "n": len(group),
        "not_at_risk": int((y_true == 0).sum()),
        "at_risk": int((y_true == 1).sum()),
        "accuracy": accuracy,
        "precision": precision,
        "recall_TPR": recall,
        "macro_f1": macro_f1,
        "FPR": fpr,
        "FNR": fnr,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


# ============================================================
# PER-SEED SUBGROUP RESULTS
# ============================================================

results = []

subgroup_definitions = {
    "gender": ["F", "M"],
    "class_level": ["JHS1", "JHS2"],
}


print("\n")
print("=" * 75)
print("CALCULATING SUBGROUP METRICS")
print("=" * 75)


for seed in expected_seeds:

    seed_df = df[
        df["seed"] == seed
    ].copy()

    for subgroup_variable, subgroup_values in (
        subgroup_definitions.items()
    ):

        for subgroup_value in subgroup_values:

            group = seed_df[
                seed_df[subgroup_variable] == subgroup_value
            ].copy()

            if len(group) == 0:
                continue

            metrics = subgroup_metrics(group)

            results.append({
                "seed": seed,
                "subgroup_variable": subgroup_variable,
                "subgroup": subgroup_value,
                **metrics
            })


results_df = pd.DataFrame(results)


# ============================================================
# SAVE PER-SEED RESULTS
# ============================================================

per_seed_file = (
    PROCESSED_DIR /
    "fairness_subgroup_results_by_seed.csv"
)

results_df.to_csv(
    per_seed_file,
    index=False
)


# ============================================================
# MEAN ± SD
# ============================================================

metric_columns = [
    "accuracy",
    "precision",
    "recall_TPR",
    "macro_f1",
    "FPR",
    "FNR",
]


summary_rows = []

for (
    subgroup_variable,
    subgroup
), group in results_df.groupby(
    [
        "subgroup_variable",
        "subgroup"
    ]
):

    row = {
        "subgroup_variable": subgroup_variable,
        "subgroup": subgroup,
        "n_seeds": len(group),
        "n_test_observations": int(group["n"].iloc[0]),
        "not_at_risk": int(group["not_at_risk"].iloc[0]),
        "at_risk": int(group["at_risk"].iloc[0]),
    }

    for metric in metric_columns:

        row[f"{metric}_mean"] = (
            group[metric].mean()
        )

        row[f"{metric}_sd"] = (
            group[metric].std(ddof=1)
        )

    summary_rows.append(row)


summary_df = pd.DataFrame(summary_rows)


# ============================================================
# FAIRNESS GAPS
# ============================================================

fairness_gap_rows = []


# ------------------------------------------------------------
# Gender gap
# ------------------------------------------------------------

gender_df = summary_df[
    summary_df["subgroup_variable"] == "gender"
].copy()


if set(gender_df["subgroup"]) == {"F", "M"}:

    female = gender_df[
        gender_df["subgroup"] == "F"
    ].iloc[0]

    male = gender_df[
        gender_df["subgroup"] == "M"
    ].iloc[0]

    fairness_gap_rows.append({
        "comparison": "Female vs Male",
        "TPR_gap_absolute": abs(
            female["recall_TPR_mean"]
            - male["recall_TPR_mean"]
        ),
        "FPR_gap_absolute": abs(
            female["FPR_mean"]
            - male["FPR_mean"]
        ),
        "precision_gap_absolute": abs(
            female["precision_mean"]
            - male["precision_mean"]
        ),
        "accuracy_gap_absolute": abs(
            female["accuracy_mean"]
            - male["accuracy_mean"]
        ),
    })


# ------------------------------------------------------------
# Class-level gap
# ------------------------------------------------------------

class_df = summary_df[
    summary_df["subgroup_variable"] == "class_level"
].copy()


if set(class_df["subgroup"]) == {"JHS1", "JHS2"}:

    jhs1 = class_df[
        class_df["subgroup"] == "JHS1"
    ].iloc[0]

    jhs2 = class_df[
        class_df["subgroup"] == "JHS2"
    ].iloc[0]

    fairness_gap_rows.append({
        "comparison": "JHS1 vs JHS2",
        "TPR_gap_absolute": abs(
            jhs1["recall_TPR_mean"]
            - jhs2["recall_TPR_mean"]
        ),
        "FPR_gap_absolute": abs(
            jhs1["FPR_mean"]
            - jhs2["FPR_mean"]
        ),
        "precision_gap_absolute": abs(
            jhs1["precision_mean"]
            - jhs2["precision_mean"]
        ),
        "accuracy_gap_absolute": abs(
            jhs1["accuracy_mean"]
            - jhs2["accuracy_mean"]
        ),
    })


fairness_gaps_df = pd.DataFrame(
    fairness_gap_rows
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_file = (
    PROCESSED_DIR /
    "fairness_subgroup_summary_mean_sd.csv"
)

gap_file = (
    PROCESSED_DIR /
    "fairness_gap_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)

fairness_gaps_df.to_csv(
    gap_file,
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 75)
print("FAIRNESS SUBGROUP RESULTS — MEAN ± SD")
print("=" * 75)

for _, row in summary_df.iterrows():

    print(
        f"\n{row['subgroup_variable']} = "
        f"{row['subgroup']}"
    )

    print(
        f"n={int(row['n_test_observations'])}, "
        f"not-at-risk={int(row['not_at_risk'])}, "
        f"at-risk={int(row['at_risk'])}"
    )

    print(
        f"Accuracy : "
        f"{row['accuracy_mean']:.6f} ± "
        f"{row['accuracy_sd']:.6f}"
    )

    print(
        f"Precision: "
        f"{row['precision_mean']:.6f} ± "
        f"{row['precision_sd']:.6f}"
    )

    print(
        f"TPR      : "
        f"{row['recall_TPR_mean']:.6f} ± "
        f"{row['recall_TPR_sd']:.6f}"
    )

    print(
        f"Macro-F1 : "
        f"{row['macro_f1_mean']:.6f} ± "
        f"{row['macro_f1_sd']:.6f}"
    )

    print(
        f"FPR      : "
        f"{row['FPR_mean']:.6f} ± "
        f"{row['FPR_sd']:.6f}"
    )

    print(
        f"FNR      : "
        f"{row['FNR_mean']:.6f} ± "
        f"{row['FNR_sd']:.6f}"
    )


print("\n")
print("=" * 75)
print("FAIRNESS GAPS")
print("=" * 75)

print(
    fairness_gaps_df.to_string(index=False)
)


# ============================================================
# COMPLETION
# ============================================================

print("\n")
print("=" * 75)
print("STEP 27B COMPLETED SUCCESSFULLY")
print("=" * 75)

print("\nCreated:")
print(per_seed_file)
print(summary_file)
print(gap_file)

print("\n✓ Gender subgroup analysis completed")
print("✓ Class-level subgroup analysis completed")
print("✓ Ten-seed mean ± SD calculated")
print("✓ TPR, FPR and FNR calculated")
print("✓ Fairness gaps calculated")
print("✓ No model retraining performed")
print("✓ No threshold selection performed")

print("\nSTOP HERE.")
print("Do NOT modify the model based on subgroup test performance.")