"""
STEP 27A — CELL A PRIMARY FAIRNESS ANALYSIS

Purpose
-------
Evaluate Cell A primary-model subgroup performance across:

1. Class level: JHS1 vs JHS2
2. Gender: Female vs Male

Cell A:
- 50:50 XGBoost–LightGBM ensemble
- No attendance_decay feature
- Ten split/model seed combinations
- Test predictions generated after model fitting
- Threshold = 0.50
- No test-set threshold tuning

Inputs
------
data/processed/cell_a_primary_test_predictions.csv
data/processed/test_student_level.csv

Outputs
-------
data/processed/cell_a_primary_fairness_class_results.csv
data/processed/cell_a_primary_fairness_gender_results.csv
data/processed/cell_a_primary_fairness_subgroup_gaps.csv
data/processed/cell_a_primary_fairness_audit.json
"""

from pathlib import Path
import json
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
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"

PRED_PATH = DATA_DIR / "cell_a_primary_test_predictions.csv"
META_PATH = DATA_DIR / "test_student_level.csv"

CLASS_OUT = DATA_DIR / "cell_a_primary_fairness_class_results.csv"
GENDER_OUT = DATA_DIR / "cell_a_primary_fairness_gender_results.csv"
GAP_OUT = DATA_DIR / "cell_a_primary_fairness_subgroup_gaps.csv"
AUDIT_OUT = DATA_DIR / "cell_a_primary_fairness_audit.json"


# ============================================================
# 2. LOAD DATA
# ============================================================

pred = pd.read_csv(PRED_PATH)
meta = pd.read_csv(META_PATH)

print("=" * 72)
print("STEP 27A — CELL A PRIMARY FAIRNESS ANALYSIS")
print("=" * 72)

print(f"Prediction rows: {len(pred):,}")
print(f"Metadata rows:   {len(meta):,}")

print("\nPrediction columns:")
print(pred.columns.tolist())

print("\nMetadata columns:")
print(meta.columns.tolist())


# ============================================================
# 3. REQUIRED COLUMNS
# ============================================================

required_pred = {
    "seed",
    "row_index",
    "y_true",
    "ensemble_probability",
    "ensemble_prediction",
    "threshold",
    "model_cell",
    "attendance_decay",
}

required_meta = {
    "student_id",
    "academic_year",
    "class_level",
    "gender",
    "at_risk",
}

missing_pred = required_pred - set(pred.columns)
missing_meta = required_meta - set(meta.columns)

if missing_pred:
    raise ValueError(
        f"Missing prediction columns: {sorted(missing_pred)}"
    )

if missing_meta:
    raise ValueError(
        f"Missing metadata columns: {sorted(missing_meta)}"
    )


# ============================================================
# 4. NORMALISE TYPES
# ============================================================

pred["seed"] = pred["seed"].astype(int)
pred["row_index"] = pred["row_index"].astype(int)
pred["y_true"] = pred["y_true"].astype(int)
pred["ensemble_prediction"] = pred["ensemble_prediction"].astype(int)

meta["student_id"] = meta["student_id"].astype(str)
meta["academic_year"] = meta["academic_year"].astype(str)
meta["class_level"] = meta["class_level"].astype(str)
meta["gender"] = meta["gender"].astype(str)
meta["at_risk"] = meta["at_risk"].astype(int)


# ============================================================
# 5. VERIFY CELL A
# ============================================================

cell_values = pred["model_cell"].dropna().unique().tolist()

if cell_values != ["Cell A"]:
    raise ValueError(
        f"Unexpected model_cell values: {cell_values}"
    )

if pred["attendance_decay"].astype(bool).any():
    raise ValueError(
        "Cell A predictions contain attendance_decay=True."
    )

threshold_values = sorted(
    pred["threshold"].dropna().unique().tolist()
)

if threshold_values != [0.5]:
    raise ValueError(
        f"Unexpected threshold values: {threshold_values}"
    )


# ============================================================
# 6. VERIFY ROW-INDEX MAPPING
# ============================================================

# The prediction artifact stores row_index values referring to
# the original row positions in test_student_level.csv.

metadata_indices = set(range(len(meta)))
prediction_indices = set(pred["row_index"].unique())

if not prediction_indices.issubset(metadata_indices):
    bad_indices = sorted(
        prediction_indices - metadata_indices
    )[:20]

    raise ValueError(
        "Prediction row_index values fall outside "
        f"test_student_level.csv. Examples: {bad_indices}"
    )

expected_indices = set(range(len(meta)))

if prediction_indices != expected_indices:
    missing_indices = sorted(
        expected_indices - prediction_indices
    )[:20]

    extra_indices = sorted(
        prediction_indices - expected_indices
    )[:20]

    raise ValueError(
        "Prediction row_index coverage does not match "
        "the 110-row test metadata.\n"
        f"Missing indices: {missing_indices}\n"
        f"Extra indices: {extra_indices}"
    )


# ============================================================
# 7. JOIN METADATA USING row_index
# ============================================================

meta = meta.copy()
meta["row_index"] = np.arange(len(meta))

metadata_columns = [
    "row_index",
    "student_id",
    "academic_year",
    "class_level",
    "gender",
    "at_risk",
]

merged = pred.merge(
    meta[metadata_columns],
    on="row_index",
    how="left",
    validate="many_to_one",
)


# ============================================================
# 8. ALIGNMENT CHECKS
# ============================================================

if merged["student_id"].isna().any():
    missing = int(
        merged["student_id"].isna().sum()
    )

    raise ValueError(
        f"{missing} prediction rows could not be "
        "matched to test metadata."
    )


target_mismatch = int(
    (
        merged["y_true"].astype(int)
        != merged["at_risk"].astype(int)
    ).sum()
)

prediction_mismatch = int(
    (
        merged["ensemble_prediction"].astype(int)
        != (
            merged["ensemble_probability"] >= merged["threshold"]
        ).astype(int)
    ).sum()
)

if target_mismatch != 0:
    raise ValueError(
        f"Target mismatch between predictions and metadata: "
        f"{target_mismatch}"
    )

if prediction_mismatch != 0:
    raise ValueError(
        f"Prediction/threshold mismatch: "
        f"{prediction_mismatch}"
    )


# ============================================================
# 9. VERIFY SEED STRUCTURE
# ============================================================

seeds = sorted(
    merged["seed"].unique().tolist()
)

expected_seeds = [
    42, 43, 44, 45, 46,
    47, 48, 49, 50, 51
]

if seeds != expected_seeds:
    raise ValueError(
        f"Unexpected seed set: {seeds}"
    )

rows_by_seed = (
    merged.groupby("seed")
    .size()
    .to_dict()
)

bad_seed_sizes = {
    int(seed): int(n)
    for seed, n in rows_by_seed.items()
    if n != len(meta)
}

if bad_seed_sizes:
    raise ValueError(
        "Each seed must contain exactly 110 test predictions. "
        f"Observed: {bad_seed_sizes}"
    )


# ============================================================
# 10. BASIC AUDIT
# ============================================================

student_year_duplicates = int(
    merged.duplicated(
        subset=[
            "seed",
            "student_id",
            "academic_year",
        ]
    ).sum()
)

missing_demographics = int(
    merged[
        ["class_level", "gender"]
    ].isna().any(axis=1).sum()
)

print("\nAUDIT")
print("-" * 72)
print(f"Seeds:                         {seeds}")
print(f"Prediction rows:               {len(merged):,}")
print(f"Rows per seed:                 {len(meta):,}")
print(
    "Unique student-year pairs:     "
    f"{merged[['student_id', 'academic_year']].drop_duplicates().shape[0]:,}"
)
print(
    "Student-year duplicates/seed:  "
    f"{student_year_duplicates}"
)
print(
    "Missing demographic/class:     "
    f"{missing_demographics}"
)
print(
    "Target mismatches:             "
    f"{target_mismatch}"
)
print(
    "Prediction mismatches:         "
    f"{prediction_mismatch}"
)


# ============================================================
# 11. METRIC FUNCTION
# ============================================================

def calculate_metrics(df):

    y_true = df["y_true"].astype(int)
    y_pred = df["ensemble_prediction"].astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "n": int(len(df)),
        "at_risk_n": int(y_true.sum()),

        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred
            )
        ),

        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),

        "recall_tpr": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),

        "fpr": float(
            fp / (fp + tn)
            if (fp + tn) > 0
            else np.nan
        ),

        "fnr": float(
            fn / (fn + tp)
            if (fn + tp) > 0
            else np.nan
        ),

        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ============================================================
# 12. SUBGROUP ANALYSIS
# ============================================================

def subgroup_records(df, dimension, groups):

    records = []

    for seed in sorted(df["seed"].unique()):

        seed_df = df[
            df["seed"] == seed
        ]

        for group in groups:

            subgroup = seed_df[
                seed_df[dimension] == group
            ].copy()

            if len(subgroup) == 0:
                continue

            metrics = calculate_metrics(
                subgroup
            )

            metrics["seed"] = int(seed)
            metrics["dimension"] = dimension
            metrics["group"] = group

            records.append(metrics)

    return pd.DataFrame(records)


class_order = [
    "JHS1",
    "JHS2",
]

gender_order = [
    "F",
    "M",
]

class_df = subgroup_records(
    merged,
    "class_level",
    class_order,
)

gender_df = subgroup_records(
    merged,
    "gender",
    gender_order,
)


# ============================================================
# 13. AGGREGATE ACROSS TEN SEEDS
# ============================================================

metric_cols = [
    "n",
    "at_risk_n",
    "accuracy",
    "precision",
    "recall_tpr",
    "fpr",
    "fnr",
    "macro_f1",
    "tn",
    "fp",
    "fn",
    "tp",
]


def aggregate_subgroups(df):

    result = (
        df.groupby(
            ["dimension", "group"],
            as_index=False
        )[metric_cols]
        .agg(["mean", "std"])
    )

    result.columns = [
        "_".join(
            str(x)
            for x in column
            if str(x) != ""
        ).rstrip("_")
        for column in result.columns
    ]

    return result


class_summary = aggregate_subgroups(
    class_df
)

gender_summary = aggregate_subgroups(
    gender_df
)


# ============================================================
# 14. GAP CALCULATIONS
# ============================================================

gap_records = []


def calculate_gap(
    dimension,
    group_a,
    group_b,
    metric,
    summary_df,
):

    a = summary_df[
        (summary_df["dimension"] == dimension)
        & (summary_df["group"] == group_a)
    ]

    b = summary_df[
        (summary_df["dimension"] == dimension)
        & (summary_df["group"] == group_b)
    ]

    if len(a) != 1 or len(b) != 1:
        return

    a_value = float(
        a[f"{metric}_mean"].iloc[0]
    )

    b_value = float(
        b[f"{metric}_mean"].iloc[0]
    )

    gap_records.append(
        {
            "dimension": dimension,
            "group_a": group_a,
            "group_b": group_b,
            "metric": metric,
            "group_a_minus_group_b": (
                a_value - b_value
            ),
            "absolute_gap": abs(
                a_value - b_value
            ),
        }
    )


fairness_metrics = [
    "accuracy",
    "precision",
    "recall_tpr",
    "fpr",
    "fnr",
    "macro_f1",
]


# JHS1 − JHS2
for metric in fairness_metrics:

    calculate_gap(
        "class_level",
        "JHS1",
        "JHS2",
        metric,
        class_summary,
    )


# Female − Male
for metric in fairness_metrics:

    calculate_gap(
        "gender",
        "F",
        "M",
        metric,
        gender_summary,
    )


gap_df = pd.DataFrame(
    gap_records
)


# ============================================================
# 15. ROUND NUMERIC OUTPUTS
# ============================================================

for dataframe in [
    class_df,
    gender_df,
    class_summary,
    gender_summary,
    gap_df,
]:

    numeric_cols = dataframe.select_dtypes(
        include=[np.number]
    ).columns

    dataframe[numeric_cols] = (
        dataframe[numeric_cols]
        .round(6)
    )


# ============================================================
# 16. SAVE OUTPUTS
# ============================================================

class_df.to_csv(
    CLASS_OUT,
    index=False,
)

gender_df.to_csv(
    GENDER_OUT,
    index=False,
)

combined_summary = pd.concat(
    [
        class_summary,
        gender_summary,
    ],
    ignore_index=True,
)

combined_summary.to_csv(
    GAP_OUT,
    index=False,
)


# ============================================================
# 17. AUDIT JSON
# ============================================================

audit = {
    "analysis": (
        "Cell A primary fairness analysis"
    ),
    "cell": "A",

    "primary_model": {
        "ensemble_weight_xgboost": 0.5,
        "ensemble_weight_lightgbm": 0.5,
        "attendance_decay_in_features": False,
        "threshold": 0.5,
    },

    "prediction_rows": int(
        len(merged)
    ),

    "metadata_rows": int(
        len(meta)
    ),

    "seeds": seeds,

    "rows_per_seed": {
        str(seed): int(rows_by_seed[seed])
        for seed in seeds
    },

    "unique_student_year_pairs": int(
        merged[
            [
                "student_id",
                "academic_year",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    ),

    "student_year_duplicates_per_seed": (
        student_year_duplicates
    ),

    "target_mismatches": target_mismatch,

    "prediction_mismatches": (
        prediction_mismatch
    ),

    "missing_demographics": (
        missing_demographics
    ),

    "class_levels": sorted(
        merged[
            "class_level"
        ].unique().tolist()
    ),

    "genders": sorted(
        merged[
            "gender"
        ].unique().tolist()
    ),

    "metadata_join_method": (
        "row_index to test_student_level.csv row position"
    ),

    "class_results_file": (
        CLASS_OUT.name
    ),

    "gender_results_file": (
        GENDER_OUT.name
    ),

    "combined_summary_file": (
        GAP_OUT.name
    ),
}


with open(
    AUDIT_OUT,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        audit,
        f,
        indent=2,
    )


# ============================================================
# 18. PRINT RESULTS
# ============================================================

print("\n")
print("=" * 72)
print("CELL A — CLASS-LEVEL FAIRNESS")
print("=" * 72)

print(
    class_summary.to_string(
        index=False
    )
)

print("\n")
print("=" * 72)
print("CELL A — GENDER FAIRNESS")
print("=" * 72)

print(
    gender_summary.to_string(
        index=False
    )
)

print("\n")
print("=" * 72)
print("CELL A — SUBGROUP GAPS")
print("=" * 72)

print(
    gap_df.to_string(
        index=False
    )
)

print("\n")
print("=" * 72)
print("OUTPUT FILES")
print("=" * 72)

print(CLASS_OUT)
print(GENDER_OUT)
print(GAP_OUT)
print(AUDIT_OUT)

print("\nSTEP 27A COMPLETE.")
