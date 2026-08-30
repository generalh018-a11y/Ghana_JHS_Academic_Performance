
# ================================================================
# STEP 16.17 — TRUE TEMPORAL ATTENDANCE-DECAY ENGINEERED MODEL
# Ghana JHS Academic Performance Thesis
# ================================================================

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

print("=" * 70)
print("STEP 16.17 — TRUE TEMPORAL ATTENDANCE-DECAY ENGINEERED MODEL")
print("=" * 70)

# ================================================================
# PATHS
# ================================================================

BASE_DIR = r"C:\Users\hills\OneDrive\Documents\MSc_Thesis\Ghana_JHS_Academic_Performance"

EXCEL_FILE = r"C:\Users\hills\OneDrive\Documents\Data Collection__Thesis\JHS_Data_Collection_1.xlsx"

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

os.makedirs(DATA_DIR, exist_ok=True)

MODEL_DATA = os.path.join(
    DATA_DIR,
    "modeling_dataset.csv"
)

# ================================================================
# LOAD MODELING DATA
# ================================================================

print("\nLoading modeling dataset...")

model_df = pd.read_csv(MODEL_DATA)

print("Dataset shape:", model_df.shape)

required_model = [
    "class_level",
    "academic_year",
    "gender",
    "age_years",
    "avg_days_present",
    "avg_total_school_days",
    "avg_attendance_rate",
    "overall_average_score"
]

missing = [
    c for c in required_model
    if c not in model_df.columns
]

if missing:
    raise ValueError(
        "Missing columns in modeling dataset: "
        + ", ".join(missing)
    )

# ================================================================
# LOAD STUDENT-YEAR IDENTIFIERS FROM AVERAGED DATASET
# ================================================================

print("\nLoading student identifiers from Excel AVERAGED DATASET...")

average_raw = pd.read_excel(
    EXCEL_FILE,
    sheet_name="📊 AVERAGED DATASET",
    header=4
)

average_raw.columns = [
    str(c).strip().lower()
    for c in average_raw.columns
]
# Normalize descriptive Excel headers to their actual field names.

average_raw = average_raw.rename(
    columns={
        c: c.split("\n")[0].strip()
        for c in average_raw.columns
    }
)
# Identify required columns despite descriptive header text.

average_map = {}

for col in average_raw.columns:

    if col.startswith("student_id"):
        average_map[col] = "student_id"

    elif col.startswith("academic_year"):
        average_map[col] = "academic_year"

average_raw = average_raw.rename(
    columns=average_map
)

if "student_id" not in average_raw.columns:
    raise ValueError(
        "student_id could not be located in AVERAGED DATASET."
    )

if "academic_year" not in average_raw.columns:
    raise ValueError(
        "academic_year could not be located in AVERAGED DATASET."
    )

student_year = average_raw[
    [
        "student_id",
        "academic_year"
    ]
].copy()

student_year["student_id"] = (
    student_year["student_id"]
    .astype(str)
    .str.strip()
)

student_year["academic_year"] = (
    student_year["academic_year"]
    .astype(str)
    .str.strip()
    .str.replace("–", "-", regex=False)
)

# Keep only genuine student records.

student_year = student_year[
    student_year["student_id"].str.startswith("STU_")
].copy()

# Normalize academic-year notation.

student_year["academic_year"] = (
    student_year["academic_year"]
    .replace({
        "2022-23": "2022-2023",
        "2023-24": "2023-2024"
    })
)

student_year = student_year.drop_duplicates(
    subset=[
        "student_id",
        "academic_year"
    ]
)

print(
    "Student-year identifiers:",
    student_year.shape
)

# ================================================================
# ATTACH IDENTIFIERS TO MODELING DATA
# ================================================================

print("\nAttaching student identifiers from AVERAGED DATASET...")

# The modeling_dataset.csv was created from the same 800
# student-year records as the AVERAGED DATASET, but student_id
# was excluded from the modeling feature file.
#
# We therefore reconstruct the identifier by matching the
# COMPLETE academic record rather than using demographic
# attributes alone.

# Normalize year representation.

model_df["academic_year"] = (
    model_df["academic_year"]
    .astype(str)
    .str.strip()
    .str.replace("–", "-", regex=False)
    .replace({
        "2022-23": "2022-2023",
        "2023-24": "2023-2024"
    })
)

average_raw["academic_year"] = (
    average_raw["academic_year"]
    .astype(str)
    .str.strip()
    .str.replace("–", "-", regex=False)
    .replace({
        "2022-23": "2022-2023",
        "2023-24": "2023-2024"
    })
)

# Extract clean student-level fields.

average_raw["student_id"] = (
    average_raw["student_id"]
    .astype(str)
    .str.strip()
)

average_raw["class_level"] = (
    average_raw["class_level"]
    .astype(str)
    .str.strip()
)

# Extract gender and age from the combined Excel field.

gender_age_col = next(
    c for c in average_raw.columns
    if c.startswith("gender / age")
)

average_raw["gender"] = (
    average_raw[gender_age_col]
    .astype(str)
    .str.extract(r"([MF])", expand=False)
)

average_raw["age_years"] = pd.to_numeric(
    average_raw[gender_age_col]
    .astype(str)
    .str.extract(r"(\d+(?:\.\d+)?)", expand=False),
    errors="coerce"
)

# Convert numerical fields to comparable values.

numeric_fields = [
    "avg_days_present",
    "avg_total_school_days",
    "avg_attendance_rate_%",
    "avg_mathematics",
    "avg_english",
    "avg_science_social"
]

for col in numeric_fields:

    average_raw[col] = pd.to_numeric(
        average_raw[col],
        errors="coerce"
    )

for col in [
    "age_years",
    "avg_days_present",
    "avg_total_school_days",
    "avg_attendance_rate"
]:

    model_df[col] = pd.to_numeric(
        model_df[col],
        errors="coerce"
    )

# The modeling dataset contains the averaged attendance,
# so use the original averaged attendance values together
# with class/year/gender/age and the academic score fields
# to establish the exact record.

average_raw["overall_average_score"] = (
    average_raw[
        [
            "avg_mathematics",
            "avg_english",
            "avg_science_social"
        ]
    ].mean(axis=1)
)

# Build a deterministic record signature.

def make_signature(row):

    return (
        str(row["class_level"]).strip(),
        str(row["academic_year"]).strip(),
        str(row["gender"]).strip(),
        round(float(row["age_years"]), 6)
        if pd.notna(row["age_years"])
        else np.nan,
        round(float(row["avg_days_present"]), 6)
        if pd.notna(row["avg_days_present"])
        else np.nan,
        round(float(row["avg_total_school_days"]), 6)
        if pd.notna(row["avg_total_school_days"])
        else np.nan,
        round(float(row["avg_attendance_rate"]), 6)
        if pd.notna(row["avg_attendance_rate"])
        else np.nan,
        round(float(row["overall_average_score"]), 6)
        if pd.notna(row["overall_average_score"])
        else np.nan
    )

# IMPORTANT:
# The original averaged dataset contains duplicate-looking
# demographic records. Therefore we do NOT use a many-to-many
# merge. We match each modeling record to exactly one source row.

source_records = []

for _, row in average_raw.iterrows():

    source_records.append({
        "student_id": row["student_id"],
        "signature": make_signature({
            "class_level": row["class_level"],
            "academic_year": row["academic_year"],
            "gender": row["gender"],
            "age_years": row["age_years"],
            "avg_days_present": row["avg_days_present"],
            "avg_total_school_days": row["avg_total_school_days"],
            "avg_attendance_rate": row["avg_attendance_rate_%"],
            "overall_average_score": row["overall_average_score"]
        })
    })

source_signature_df = pd.DataFrame(
    source_records
)

model_records = []

for _, row in model_df.iterrows():

    model_records.append({
        "signature": make_signature({
            "class_level": row["class_level"],
            "academic_year": row["academic_year"],
            "gender": row["gender"],
            "age_years": row["age_years"],
            "avg_days_present": row["avg_days_present"],
            "avg_total_school_days": row["avg_total_school_days"],
            "avg_attendance_rate": row["avg_attendance_rate"],
            "overall_average_score": row["overall_average_score"]
        })
    })

model_signature_df = pd.DataFrame(
    model_records
)

# Map each signature to the available student IDs.

signature_map = {}

for _, row in source_signature_df.iterrows():

    signature = row["signature"]

    if signature not in signature_map:

        signature_map[signature] = []

    signature_map[signature].append(
        row["student_id"]
    )

# Check whether every modeling record can be matched.

unmatched = []

ambiguous = []

assigned_ids = []

for i, signature in enumerate(
    model_signature_df["signature"]
):

    candidates = signature_map.get(
        signature,
        []
    )

    if len(candidates) == 0:

        unmatched.append(i)
        assigned_ids.append(None)

    elif len(candidates) == 1:

        assigned_ids.append(
            candidates[0]
        )

    else:

        ambiguous.append(
            (i, candidates)
        )

        assigned_ids.append(None)

print(
    "Unmatched records:",
    len(unmatched)
)

print(
    "Ambiguous records:",
    len(ambiguous)
)

if len(unmatched) > 0:

    raise ValueError(
        "Some modeling records could not be "
        "matched uniquely to the AVERAGED DATASET."
    )

if len(ambiguous) > 0:

    raise ValueError(
        "Some modeling records have multiple "
        "possible student IDs. Exact identity "
        "cannot be established safely."
    )

model_df["student_id"] = assigned_ids

print(
    "✓ Student identifiers attached uniquely."
)

print(
    "Verified records:",
    len(model_df)
)

print("\nSample matched records:")

print(
    model_df[
        [
            "student_id",
            "class_level",
            "academic_year",
            "gender",
            "age_years"
        ]
    ]
    .head(10)
    .to_string(index=False)
)

# ================================================================
# TARGET
# ================================================================

model_df["at_risk"] = (
    model_df["overall_average_score"] < 50
).astype(int)

print("\nTarget definition:")
print("Score < 50 = At Risk")
print("Score >= 50 = Not At Risk")

print("\nTarget distribution:")
print(
    model_df["at_risk"]
    .value_counts()
    .sort_index()
)

# ================================================================
# READ TERM-LEVEL ATTENDANCE
# ================================================================

print("\n" + "=" * 70)
print("READING TERM-LEVEL ATTENDANCE")
print("=" * 70)

term_sheets = {
    ("2022-2023", 1): "📘 2022-23 Term 1",
    ("2022-2023", 2): "📘 2022-23 Term 2",
    ("2022-2023", 3): "📘 2022-23 Term 3",
    ("2023-2024", 1): "📗 2023-24 Term 1",
    ("2023-2024", 2): "📗 2023-24 Term 2",
    ("2023-2024", 3): "📗 2023-24 Term 3"
}

term_records = []

for (year, term), sheet in term_sheets.items():

    print(f"\nReading {sheet}...")

    raw = pd.read_excel(
        EXCEL_FILE,
        sheet_name=sheet,
        header=6
    )

    raw.columns = [
        str(c).strip().lower()
        for c in raw.columns
    ]

    column_map = {}

    for col in raw.columns:

        if col.startswith("student_id"):
            column_map[col] = "student_id"

        elif col.startswith("attendance_rate"):
            column_map[col] = "attendance_rate"

    raw = raw.rename(
        columns=column_map
    )

    required_term = [
        "student_id",
        "attendance_rate"
    ]

    missing_term = [
        c for c in required_term
        if c not in raw.columns
    ]

    if missing_term:
        raise ValueError(
            f"{sheet}: missing "
            + ", ".join(missing_term)
        )

    term_data = raw[
        [
            "student_id",
            "attendance_rate"
        ]
    ].copy()

    term_data["academic_year"] = year
    term_data["term"] = term

    term_data["student_id"] = (
        term_data["student_id"]
        .astype(str)
        .str.strip()
    )

    term_data["attendance_rate"] = pd.to_numeric(
        term_data["attendance_rate"],
        errors="coerce"
    )

    term_data = term_data[
        term_data["student_id"].str.startswith("STU_")
    ]

    term_data = term_data.dropna(
        subset=["attendance_rate"]
    )

    # Convert percentage values such as 92.9 to 0.929.

    if term_data["attendance_rate"].max() > 1:

        term_data["attendance_rate"] = (
            term_data["attendance_rate"] / 100
        )

    term_records.append(
        term_data
    )

    print(
        f"  Records loaded: {len(term_data)}"
    )

attendance_terms = pd.concat(
    term_records,
    ignore_index=True
)

print(
    "\nCombined term records:",
    attendance_terms.shape
)

# ================================================================
# EXPONENTIAL TEMPORAL DECAY
# ================================================================

print("\n" + "=" * 70)
print("CALCULATING EXPONENTIAL ATTENDANCE DECAY")
print("=" * 70)

LAMBDA = 0.5

# More recent terms receive greater weight.

raw_weights = np.exp(
    LAMBDA * np.array([1, 2, 3])
)

weights = (
    raw_weights /
    raw_weights.sum()
)

print(
    f"Lambda: {LAMBDA}"
)

print(
    f"Term 1 weight: {weights[0]:.6f}"
)

print(
    f"Term 2 weight: {weights[1]:.6f}"
)

print(
    f"Term 3 weight: {weights[2]:.6f}"
)

# ================================================================
# CREATE STUDENT-YEAR ATTENDANCE TABLE
# ================================================================

decay_table = (
    attendance_terms
    .pivot_table(
        index=[
            "student_id",
            "academic_year"
        ],
        columns="term",
        values="attendance_rate",
        aggfunc="mean"
    )
    .reset_index()
)

decay_table = decay_table.rename(
    columns={
        1: "attendance_term1",
        2: "attendance_term2",
        3: "attendance_term3"
    }
)

for col in [
    "attendance_term1",
    "attendance_term2",
    "attendance_term3"
]:

    if col not in decay_table.columns:

        decay_table[col] = np.nan

# ================================================================
# CALCULATE DECAY FEATURE
# ================================================================

term_columns = [
    "attendance_term1",
    "attendance_term2",
    "attendance_term3"
]

term_values = decay_table[
    term_columns
]

weighted_sum = (
    term_values
    .mul(weights, axis=1)
    .sum(axis=1)
)

available_weight = (
    term_values.notna()
    .mul(weights, axis=1)
    .sum(axis=1)
)

decay_table[
    "attendance_decay"
] = (
    weighted_sum /
    available_weight
)

print("\nSample attendance-decay values:")

print(
    decay_table[
        [
            "student_id",
            "academic_year",
            "attendance_term1",
            "attendance_term2",
            "attendance_term3",
            "attendance_decay"
        ]
    ]
    .head(10)
    .to_string(index=False)
)

# ================================================================
# SAVE DECAY FEATURES
# ================================================================

decay_path = os.path.join(
    DATA_DIR,
    "attendance_decay_features.csv"
)

decay_table.to_csv(
    decay_path,
    index=False
)

print(
    "\n✓ Saved:",
    decay_path
)

# ================================================================
# MERGE ENGINEERED FEATURE
# ================================================================

print("\n" + "=" * 70)
print("MERGING ENGINEERED FEATURE")
print("=" * 70)

engineered_df = model_df.merge(
    decay_table,
    on=[
        "student_id",
        "academic_year"
    ],
    how="left"
)

print(
    "Merged dataset shape:",
    engineered_df.shape
)

print(
    "Missing attendance_decay:",
    engineered_df["attendance_decay"]
    .isna()
    .sum()
)

# Fallback only for unmatched records.

engineered_df["attendance_decay"] = (
    engineered_df["attendance_decay"]
    .fillna(
        engineered_df["avg_attendance_rate"] / 100
        if engineered_df["avg_attendance_rate"].max() > 1
        else engineered_df["avg_attendance_rate"]
    )
)

# ================================================================
# SAVE ENGINEERED DATASET
# ================================================================

engineered_dataset_path = os.path.join(
    DATA_DIR,
    "engineered_modeling_dataset.csv"
)

engineered_df.to_csv(
    engineered_dataset_path,
    index=False
)

print(
    "✓ Saved:",
    engineered_dataset_path
)

# ================================================================
# FEATURES
# ================================================================

feature_cols = [
    "age_years",
    "avg_days_present",
    "avg_total_school_days",
    "attendance_decay",
    "class_level",
    "academic_year",
    "gender"
]

X = engineered_df[
    feature_cols
].copy()

y = engineered_df[
    "at_risk"
].astype(int)

# ================================================================
# ENCODE CATEGORICAL VARIABLES
# ================================================================

X = pd.get_dummies(
    X,
    columns=[
        "class_level",
        "academic_year",
        "gender"
    ],
    dtype=int
)

print(
    "\nEncoded feature matrix:",
    X.shape
)

print("\nFeatures:")

for i, feature in enumerate(
    X.columns,
    start=1
):

    print(
        f"{i}. {feature}"
    )

# ================================================================
# TRAIN / VALIDATION / TEST
# ================================================================

print("\n" + "=" * 70)
print("TRAIN / VALIDATION / TEST SPLIT")
print("=" * 70)

(
    X_train_full,
    X_test,
    y_train_full,
    y_test,
    idx_train_full,
    idx_test
) = train_test_split(
    X,
    y,
    engineered_df.index,
    test_size=0.20,
    random_state=42,
    stratify=y
)

(
    X_train,
    X_val,
    y_train,
    y_val,
    idx_train,
    idx_val
) = train_test_split(
    X_train_full,
    y_train_full,
    idx_train_full,
    test_size=0.20,
    random_state=42,
    stratify=y_train_full
)

print(
    "Training:",
    X_train.shape
)

print(
    "Validation:",
    X_val.shape
)

print(
    "Testing:",
    X_test.shape
)

# ================================================================
# XGBOOST
# ================================================================

print("\n" + "=" * 70)
print("TRAINING XGBOOST")
print("=" * 70)

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

xgb.fit(
    X_train,
    y_train
)

print(
    "✓ XGBoost trained."
)

# ================================================================
# LIGHTGBM
# ================================================================

print("\n" + "=" * 70)
print("TRAINING LIGHTGBM")
print("=" * 70)

lgb = LGBMClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary",
    random_state=42,
    verbosity=-1
)

lgb.fit(
    X_train,
    y_train
)

print(
    "✓ LightGBM trained."
)

# ================================================================
# TEST PROBABILITIES
# ================================================================

xgb_test = xgb.predict_proba(
    X_test
)[:, 1]

lgb_test = lgb.predict_proba(
    X_test
)[:, 1]

standard_prob = (
    0.50 * xgb_test
    +
    0.50 * lgb_test
)

# ================================================================
# CLASS-SENSITIVE FUSION
# ================================================================

print("\n" + "=" * 70)
print("CLASS-SENSITIVE FUSION")
print("=" * 70)

xgb_val = xgb.predict_proba(
    X_val
)[:, 1]

lgb_val = lgb.predict_proba(
    X_val
)[:, 1]

validation_classes = engineered_df.loc[
    idx_val,
    "class_level"
].values

class_alpha = {}

for cls in [
    "JHS1",
    "JHS2",
    "JHS3"
]:

    mask = (
        validation_classes == cls
    )

    if mask.sum() == 0:

        class_alpha[cls] = 0.50
        continue

    y_cls = y_val.to_numpy()[mask]

    if len(np.unique(y_cls)) < 2:

        alpha = 0.50

    else:

        fusion_X = np.column_stack([
            xgb_val[mask],
            lgb_val[mask]
        ])

        calibration = LogisticRegression(
            max_iter=1000,
            random_state=42
        )

        calibration.fit(
            fusion_X,
            y_cls
        )

        coef = np.abs(
            calibration.coef_[0]
        )

        if coef.sum() == 0:

            alpha = 0.50

        else:

            alpha = (
                coef[0] /
                coef.sum()
            )

    alpha = float(
        np.clip(
            alpha,
            0.10,
            0.90
        )
    )

    class_alpha[cls] = alpha

    print(
        f"{cls}: "
        f"XGBoost={alpha:.4f}, "
        f"LightGBM={1-alpha:.4f}"
    )

# ================================================================
# CLASS-SENSITIVE TEST PROBABILITY
# ================================================================

test_classes = engineered_df.loc[
    idx_test,
    "class_level"
].values

class_sensitive_prob = np.zeros(
    len(X_test)
)

for i, cls in enumerate(
    test_classes
):

    alpha = class_alpha.get(
        cls,
        0.50
    )

    class_sensitive_prob[i] = (
        alpha * xgb_test[i]
        +
        (1 - alpha) * lgb_test[i]
    )

# ================================================================
# EVALUATION
# ================================================================

def evaluate_model(
    name,
    y_true,
    probability
):

    prediction = (
        probability >= 0.50
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        prediction,
        labels=[0, 1]
    ).ravel()

    return {
        "condition": name,
        "accuracy": accuracy_score(
            y_true,
            prediction
        ),
        "precision": precision_score(
            y_true,
            prediction,
            zero_division=0
        ),
        "recall_tpr": recall_score(
            y_true,
            prediction,
            zero_division=0
        ),
        "macro_f1": f1_score(
            y_true,
            prediction,
            average="macro",
            zero_division=0
        ),
        "auc_roc": roc_auc_score(
            y_true,
            probability
        ),
        "auc_pr": average_precision_score(
            y_true,
            probability
        ),
        "brier_score": brier_score_loss(
            y_true,
            probability
        ),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }

# ================================================================
# FINAL ENGINEERED RESULTS
# ================================================================

print("\n" + "=" * 70)
print("FINAL ENGINEERED CLASSIFICATION RESULTS")
print("=" * 70)

results = pd.DataFrame([
    evaluate_model(
        "Temporal Attendance-Decay 50:50 Ensemble",
        y_test,
        standard_prob
    ),
    evaluate_model(
        "Class-Sensitive Temporal Attendance-Decay Ensemble",
        y_test,
        class_sensitive_prob
    )
])

print(
    results.to_string(
        index=False,
        formatters={
            "accuracy": "{:.4f}".format,
            "precision": "{:.4f}".format,
            "recall_tpr": "{:.4f}".format,
            "macro_f1": "{:.4f}".format,
            "auc_roc": "{:.4f}".format,
            "auc_pr": "{:.4f}".format,
            "brier_score": "{:.4f}".format
        }
    )
)

# ================================================================
# SAVE RESULTS
# ================================================================

results_path = os.path.join(
    DATA_DIR,
    "engineered_classification_results.csv"
)

results.to_csv(
    results_path,
    index=False
)

print(
    "\n✓ Saved:",
    results_path
)

# ================================================================
# SAVE TEST PREDICTIONS
# ================================================================

predictions = engineered_df.loc[
    idx_test,
    [
        "student_id",
        "class_level",
        "academic_year",
        "gender",
        "age_years",
        "attendance_decay",
        "overall_average_score"
    ]
].copy()

predictions["actual_at_risk"] = (
    y_test.to_numpy()
)

predictions[
    "standard_probability"
] = standard_prob

predictions[
    "class_sensitive_probability"
] = class_sensitive_prob

predictions[
    "standard_prediction"
] = (
    standard_prob >= 0.50
).astype(int)

predictions[
    "class_sensitive_prediction"
] = (
    class_sensitive_prob >= 0.50
).astype(int)

prediction_path = os.path.join(
    DATA_DIR,
    "engineered_classification_predictions.csv"
)

predictions.to_csv(
    prediction_path,
    index=False
)

print(
    "✓ Saved:",
    prediction_path
)

# ================================================================
# SAVE FUSION WEIGHTS
# ================================================================

alpha_df = pd.DataFrame([
    {
        "class_level": cls,
        "xgboost_weight": alpha,
        "lightgbm_weight": 1 - alpha
    }
    for cls, alpha
    in class_alpha.items()
])

alpha_path = os.path.join(
    DATA_DIR,
    "class_sensitive_fusion_weights.csv"
)

alpha_df.to_csv(
    alpha_path,
    index=False
)

print(
    "✓ Saved:",
    alpha_path
)

# ================================================================
# GENDER FAIRNESS
# ================================================================

print("\n" + "=" * 70)
print("ENGINEERED MODEL FAIRNESS BY GENDER")
print("=" * 70)

gender_results = []

for gender in sorted(
    predictions["gender"]
    .dropna()
    .unique()
):

    group = predictions[
        predictions["gender"] == gender
    ]

    true = group[
        "actual_at_risk"
    ]

    pred = group[
        "class_sensitive_prediction"
    ]

    gender_results.append({
        "gender": gender,
        "n": len(group),
        "accuracy": accuracy_score(
            true,
            pred
        ),
        "precision": precision_score(
            true,
            pred,
            zero_division=0
        ),
        "recall_tpr": recall_score(
            true,
            pred,
            zero_division=0
        ),
        "macro_f1": f1_score(
            true,
            pred,
            average="macro",
            zero_division=0
        )
    })

gender_df = pd.DataFrame(
    gender_results
)

gender_path = os.path.join(
    DATA_DIR,
    "engineered_fairness_gender.csv"
)

gender_df.to_csv(
    gender_path,
    index=False
)

print(
    gender_df.to_string(
        index=False
    )
)

print(
    "✓ Saved:",
    gender_path
)

# ================================================================
# CLASS FAIRNESS
# ================================================================
print("\n" + "=" * 70)
print("ENGINEERED MODEL FAIRNESS BY CLASS LEVEL")
print("=" * 70)

class_results = []

for cls in [
    "JHS1",
    "JHS2",
    "JHS3"
]:

    group = predictions[
        predictions["class_level"] == cls
    ]

    if len(group) == 0:
        continue

    true = group[
        "actual_at_risk"
    ]

    pred = group[
        "class_sensitive_prediction"
    ]

    tn, fp, fn, tp = confusion_matrix(
        true,
        pred,
        labels=[0, 1]
    ).ravel()

    # ------------------------------------------------------------
    # FAIRNESS METRICS
    # ------------------------------------------------------------

    # FPR is undefined when there are no actual negatives.
    if (fp + tn) > 0:
        fpr = fp / (fp + tn)
    else:
        fpr = float("nan")

    # Precision is undefined when there are no predicted positives.
    if (tp + fp) > 0:
        precision = tp / (tp + fp)
    else:
        precision = float("nan")

    # TPR / Recall is undefined when there are no actual positives.
    if (tp + fn) > 0:
        recall_tpr = tp / (tp + fn)
    else:
        recall_tpr = float("nan")

    # Macro-F1 is not meaningful when one class is absent.
    if (tp + fn) > 0 and (tn + fp) > 0:
        macro_f1 = f1_score(
            true,
            pred,
            average="macro",
            zero_division=0
        )
    else:
        macro_f1 = float("nan")

    class_results.append({
        "class_level": cls,
        "n": len(group),
        "accuracy": accuracy_score(
            true,
            pred
        ),
        "precision": precision,
        "recall_tpr": recall_tpr,
        "fpr": fpr,
        "macro_f1": macro_f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    })

class_df = pd.DataFrame(
    class_results
)

class_path = os.path.join(
    DATA_DIR,
    "engineered_fairness_class.csv"
)

class_df.to_csv(
    class_path,
    index=False
)

print(
    class_df.to_string(
        index=False
    )
)

print(
    "✓ Saved:",
    class_path
)

# ================================================================
# FAIRNESS GAPS
# ================================================================

gap_results = []

for metric in [
    "accuracy",
    "precision",
    "recall_tpr",
    "fpr",
    "macro_f1"
]:

    valid = class_df[
        metric
    ].dropna()

    if len(valid) == 0:
        gap_results.append({
            "metric": metric,
            "max_group": "N/A",
            "min_group": "N/A",
            "gap": np.nan
        })
        continue

    maximum = valid.max()
    minimum = valid.min()

    max_group = class_df.loc[
        class_df[metric].idxmax(),
        "class_level"
    ]

    min_group = class_df.loc[
        class_df[metric].idxmin(),
        "class_level"
    ]

    gap_results.append({
        "metric": metric,
        "max_group": max_group,
        "min_group": min_group,
        "gap": maximum - minimum
    })

gap_df = pd.DataFrame(
    gap_results
)

gap_path = os.path.join(
    DATA_DIR,
    "engineered_fairness_gaps.csv"
)

gap_df.to_csv(
    gap_path,
    index=False
)

print(
    "\nClass-level fairness gaps:"
)

print(
    gap_df.to_string(
        index=False
    )
)

print(
    "✓ Saved:",
    gap_path
)

# ================================================================
# COMPLETE
# ================================================================

print("\n" + "=" * 70)
print("STEP 16.17 COMPLETE")
print("=" * 70)

print("✓ Student-year identifiers recovered.")
print("✓ Six term sheets processed.")
print("✓ True temporal attendance decay calculated.")
print("✓ XGBoost trained.")
print("✓ LightGBM trained.")
print("✓ Ensemble evaluated.")
print("✓ Macro-F1 calculated.")
print("✓ AUC-ROC calculated.")
print("✓ AUC-PR calculated.")
print("✓ Brier Score calculated.")
print("✓ Gender fairness calculated.")
print("✓ Class-level fairness calculated.")
print("✓ Fairness gaps calculated.")

print("\nDO NOT COMMIT OR UPDATE THE THESIS YET.")
print("Verify the numerical results first.")

print("=" * 70)