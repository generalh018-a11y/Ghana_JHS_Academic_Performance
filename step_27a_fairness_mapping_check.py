"""
STEP 27A — FAIRNESS DATA MAPPING VERIFICATION

Purpose:
    Verify that the locked primary ensemble predictions can be
    correctly mapped to the corresponding test students and their
    subgroup attributes.

Checks:
    - Test student-level dataset
    - Ten-seed prediction file
    - Student IDs
    - Gender
    - Class level
    - Academic year
    - Target values
    - Prediction row structure
    - One prediction per student per seed

IMPORTANT:
    This script does not retrain any model.
    This script does not modify any prediction.
    This script only verifies the mapping required for fairness analysis.
"""

from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TEST_STUDENT_FILE = (
    PROCESSED_DIR /
    "test_student_level.csv"
)

PREDICTION_FILE = (
    PROCESSED_DIR /
    "ten_seed_ensemble_test_predictions.csv"
)


# ============================================================
# START
# ============================================================

print("=" * 75)
print("STEP 27A — FAIRNESS DATA MAPPING VERIFICATION")
print("=" * 75)


# ============================================================
# LOAD FILES
# ============================================================

print("\nLoading test student-level dataset...")

test_df = pd.read_csv(TEST_STUDENT_FILE)

print(f"Test student-level rows : {len(test_df)}")
print(f"Test student-level cols : {len(test_df.columns)}")

print("\nTest dataset columns:")
print(list(test_df.columns))


print("\nLoading ten-seed prediction file...")

pred_df = pd.read_csv(PREDICTION_FILE)

print(f"Prediction rows : {len(pred_df)}")
print(f"Prediction cols : {len(pred_df.columns)}")

print("\nPrediction columns:")
print(list(pred_df.columns))


# ============================================================
# REQUIRED TEST DATA COLUMNS
# ============================================================

required_test_columns = [
    "student_id",
    "academic_year",
    "class_level",
    "gender",
    "at_risk",
]


print("\nChecking required test columns...")

missing_test_columns = [
    col
    for col in required_test_columns
    if col not in test_df.columns
]

if missing_test_columns:
    raise ValueError(
        "Missing required test columns: "
        + str(missing_test_columns)
    )

print("✓ All required test columns are present.")


# ============================================================
# REQUIRED PREDICTION COLUMNS
# ============================================================

required_prediction_columns = [
    "seed",
    "y_true",
    "ensemble_probability",
    "ensemble_prediction",
]


print("\nChecking required prediction columns...")

missing_prediction_columns = [
    col
    for col in required_prediction_columns
    if col not in pred_df.columns
]

if missing_prediction_columns:
    raise ValueError(
        "Missing required prediction columns: "
        + str(missing_prediction_columns)
    )

print("✓ All required prediction columns are present.")




# ============================================================
# TEST STUDENT-YEAR UNIQUENESS
# ============================================================

print("\nChecking test student-year uniqueness...")

duplicate_student_year = (
    test_df
    .duplicated(
        subset=["student_id", "academic_year"]
    )
    .sum()
)

print(
    f"Duplicate student-year combinations: "
    f"{duplicate_student_year}"
)

if duplicate_student_year != 0:
    raise ValueError(
        "Duplicate student-year combinations found "
        "in test_student_level.csv."
    )

print(
    "✓ Each student-year trajectory appears once."
)

print(
    f"Unique students represented in test set: "
    f"{test_df['student_id'].nunique()}"
)

print(
    f"Test student-year trajectories: "
    f"{len(test_df)}"
)


# ============================================================
# PREDICTION SEEDS
# ============================================================

print("\nChecking prediction seeds...")

seeds = sorted(
    pred_df["seed"].unique().tolist()
)

print(f"Seeds found: {seeds}")

expected_seeds = list(range(42, 52))

if seeds != expected_seeds:
    raise ValueError(
        f"Unexpected seeds. Expected {expected_seeds}, "
        f"found {seeds}"
    )

print("✓ All ten expected seeds are present.")


# ============================================================
# ROW COUNT CHECK
# ============================================================

expected_prediction_rows = (
    len(test_df) * len(expected_seeds)
)

print("\nChecking prediction row count...")

print(
    f"Expected prediction rows: "
    f"{len(test_df)} × {len(expected_seeds)} "
    f"= {expected_prediction_rows}"
)

print(
    f"Actual prediction rows  : "
    f"{len(pred_df)}"
)

if len(pred_df) != expected_prediction_rows:
    raise ValueError(
        "Prediction row count does not match "
        "test students × number of seeds."
    )

print("✓ Prediction row count is correct.")


# ============================================================
# ROW COUNT PER SEED
# ============================================================

print("\nChecking rows per seed...")

rows_per_seed = (
    pred_df
    .groupby("seed")
    .size()
)

print(rows_per_seed.to_string())

if not all(rows_per_seed == len(test_df)):
    raise ValueError(
        "At least one seed does not contain exactly "
        "one prediction for every test row."
    )

print("✓ Every seed contains one prediction per test row.")


# ============================================================
# TARGET CONSISTENCY
# ============================================================

print("\nChecking target consistency...")

test_targets = (
    test_df["at_risk"]
    .astype(int)
    .reset_index(drop=True)
)

target_mismatches = 0

for seed in expected_seeds:

    seed_pred = (
        pred_df[pred_df["seed"] == seed]
        .reset_index(drop=True)
    )

    seed_targets = (
        seed_pred["y_true"]
        .astype(int)
        .reset_index(drop=True)
    )

    mismatches = (
        seed_targets != test_targets
    ).sum()

    print(
        f"Seed {seed}: target mismatches = {mismatches}"
    )

    target_mismatches += mismatches


if target_mismatches != 0:
    raise ValueError(
        "Target values do not align with the test dataset."
    )

print("✓ Target values align across all seeds.")


# ============================================================
# PREDICTION ORDER CHECK
# ============================================================

print("\nChecking prediction ordering...")

reference_targets = (
    pred_df[
        pred_df["seed"] == expected_seeds[0]
    ]["y_true"]
    .astype(int)
    .reset_index(drop=True)
)

order_mismatches = (
    reference_targets != test_targets
).sum()

print(
    f"Order mismatches for reference seed "
    f"{expected_seeds[0]}: {order_mismatches}"
)

if order_mismatches != 0:
    raise ValueError(
        "Prediction ordering does not match "
        "test_student_level.csv."
    )

print("✓ Prediction order matches test dataset.")


# ============================================================
# BUILD MAPPING
# ============================================================

print("\nBuilding fairness mapping...")

mapping_df = test_df[
    [
        "student_id",
        "academic_year",
        "class_level",
        "gender",
        "at_risk",
    ]
].copy()

mapping_df = mapping_df.reset_index(drop=True)

mapping_df["test_row_index"] = mapping_df.index


# ============================================================
# MAP PREDICTIONS TO STUDENTS
# ============================================================

mapped_rows = []

for seed in expected_seeds:

    seed_predictions = (
        pred_df[
            pred_df["seed"] == seed
        ]
        .reset_index(drop=True)
        .copy()
    )

    seed_predictions["test_row_index"] = (
        seed_predictions.index
    )

    merged = mapping_df.merge(
        seed_predictions[
            [
                "test_row_index",
                "seed",
                "ensemble_probability",
                "ensemble_prediction",
                "y_true",
            ]
        ],
        on="test_row_index",
        how="left",
        validate="one_to_one",
    )

    mapped_rows.append(merged)


fairness_df = pd.concat(
    mapped_rows,
    ignore_index=True
)


# ============================================================
# VERIFY MAPPING
# ============================================================

print("\nVerifying final mapping...")

expected_rows = (
    len(test_df) * len(expected_seeds)
)

print(
    f"Mapped rows   : {len(fairness_df)}"
)

print(
    f"Expected rows : {expected_rows}"
)

if len(fairness_df) != expected_rows:
    raise ValueError(
        "Mapped row count is incorrect."
    )


# ------------------------------------------------------------
# Missing subgroup values
# ------------------------------------------------------------

mapping_columns = [
    "student_id",
    "gender",
    "class_level",
    "academic_year",
]

print("\nMissing values after mapping:")

for col in mapping_columns:

    missing = fairness_df[col].isna().sum()

    print(
        f"{col}: {missing}"
    )

    if missing != 0:
        raise ValueError(
            f"Missing values found in mapped column: {col}"
        )


# ============================================================
# STUDENT / SEED UNIQUENESS
# ============================================================

print("\nChecking student-seed uniqueness...")


duplicate_student_year_seed = (
    fairness_df
    .duplicated(
        subset=["student_id", "academic_year", "seed"]
    )
    .sum()
)

print(
    f"Duplicate student-year-seed combinations: "
    f"{duplicate_student_year_seed}"
)

if duplicate_student_year_seed != 0:
    raise ValueError(
        "Duplicate student-year-seed combinations detected."
    )

print(
    "✓ Each student-year trajectory has exactly "
    "one prediction per seed."
)





# ============================================================
# SUBGROUP COUNTS
# ============================================================

print("\n")
print("=" * 75)
print("TEST SUBGROUP STRUCTURE")
print("=" * 75)

print("\nGender:")
print(
    mapping_df["gender"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nClass level:")
print(
    mapping_df["class_level"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nAcademic year:")
print(
    mapping_df["academic_year"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# TARGET BY SUBGROUP
# ============================================================

print("\n")
print("=" * 75)
print("AT-RISK DISTRIBUTION BY SUBGROUP")
print("=" * 75)

print("\nGender × target:")

gender_target = pd.crosstab(
    mapping_df["gender"],
    mapping_df["at_risk"]
)

print(gender_target.to_string())


print("\nClass level × target:")

class_target = pd.crosstab(
    mapping_df["class_level"],
    mapping_df["at_risk"]
)

print(class_target.to_string())


# ============================================================
# SAVE
# ============================================================

output_file = (
    PROCESSED_DIR /
    "fairness_analysis_mapped_predictions.csv"
)

fairness_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# COMPLETION
# ============================================================

print("\n")
print("=" * 75)
print("STEP 27A COMPLETED SUCCESSFULLY")
print("=" * 75)

print("\nCreated:")
print(output_file)

print("\n✓ Test student attributes verified")
print("✓ Ten seeds verified")
print("✓ Target alignment verified")
print("✓ Prediction ordering verified")
print("✓ Student-seed uniqueness verified")
print("✓ Gender and class attributes mapped")
print("✓ Fairness-ready prediction dataset created")

print("\nSTOP HERE.")
print("Do NOT calculate or modify fairness results manually.")