from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# STEP 19A
# PREPARE CANONICAL MODEL FEATURES
# ============================================================

print("=" * 75)
print("STEP 19A — PREPARE CANONICAL MODEL FEATURES")
print("=" * 75)


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_FILE = DATA_DIR / "train_student_level.csv"
VALIDATION_FILE = DATA_DIR / "validation_student_level.csv"
TEST_FILE = DATA_DIR / "test_student_level.csv"


# ------------------------------------------------------------
# 2. LOAD SPLITS
# ------------------------------------------------------------

train = pd.read_csv(TRAIN_FILE)
validation = pd.read_csv(VALIDATION_FILE)
test = pd.read_csv(TEST_FILE)


print("\nLoaded datasets:")
print(f"Train:      {len(train)} rows")
print(f"Validation: {len(validation)} rows")
print(f"Test:       {len(test)} rows")


# ------------------------------------------------------------
# 3. DEFINE TARGET
# ------------------------------------------------------------

TARGET = "at_risk"


# ------------------------------------------------------------
# 4. DEFINE PREDICTORS
# ------------------------------------------------------------

NUMERIC_FEATURES = [
    "score_t1",
    "score_t2",
    "attendance_t1",
    "attendance_t2",
    "attendance_decay",
    "age_years",
]

CATEGORICAL_FEATURES = [
    "gender",
    "class_level",
    "academic_year",
]

FEATURES = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)


print("\nCanonical numeric features:")
for feature in NUMERIC_FEATURES:
    print("  -", feature)

print("\nCanonical categorical features:")
for feature in CATEGORICAL_FEATURES:
    print("  -", feature)


# ------------------------------------------------------------
# 5. CHECK FEATURE AVAILABILITY
# ------------------------------------------------------------

for name, data in [
    ("Train", train),
    ("Validation", validation),
    ("Test", test),
]:

    missing_features = [
        col for col in FEATURES
        if col not in data.columns
    ]

    assert not missing_features, (
        f"{name} is missing features: "
        f"{missing_features}"
    )


# ------------------------------------------------------------
# 6. CHECK NO TARGET LEAKAGE
# ------------------------------------------------------------

FORBIDDEN_FEATURES = [
    "target_score_t3",
    "at_risk",
    "at_risk_label",
]

for name, data in [
    ("Train", train),
    ("Validation", validation),
    ("Test", test),
]:

    # This check is conceptual:
    # forbidden target fields may exist in the source split,
    # but they must not appear in X.

    X_columns = FEATURES

    leakage = set(X_columns) & set(FORBIDDEN_FEATURES)

    assert len(leakage) == 0, (
        f"Target leakage detected in {name}: {leakage}"
    )


# ------------------------------------------------------------
# 7. CREATE FEATURE MATRICES
# ------------------------------------------------------------

X_train = train[FEATURES].copy()
y_train = train[TARGET].astype(int).copy()

X_validation = validation[FEATURES].copy()
y_validation = validation[TARGET].astype(int).copy()

X_test = test[FEATURES].copy()
y_test = test[TARGET].astype(int).copy()


# ------------------------------------------------------------
# 8. ONE-HOT ENCODE CATEGORICAL FEATURES
#
# IMPORTANT:
# Fit encoding using TRAIN only.
# Validation and test are transformed to the same columns.
# ------------------------------------------------------------

X_train = pd.get_dummies(
    X_train,
    columns=CATEGORICAL_FEATURES,
    drop_first=False,
    dtype=int,
)

X_validation = pd.get_dummies(
    X_validation,
    columns=CATEGORICAL_FEATURES,
    drop_first=False,
    dtype=int,
)

X_test = pd.get_dummies(
    X_test,
    columns=CATEGORICAL_FEATURES,
    drop_first=False,
    dtype=int,
)


# ------------------------------------------------------------
# 9. ALIGN VALIDATION AND TEST TO TRAIN FEATURES
# ------------------------------------------------------------

X_validation = X_validation.reindex(
    columns=X_train.columns,
    fill_value=0,
)

X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0,
)


# ------------------------------------------------------------
# 10. CHECK MISSING VALUES
# ------------------------------------------------------------

assert not X_train.isna().any().any()
assert not X_validation.isna().any().any()
assert not X_test.isna().any().any()


# ------------------------------------------------------------
# 11. CHECK FEATURE ORDER
# ------------------------------------------------------------

assert list(X_train.columns) == list(
    X_validation.columns
)

assert list(X_train.columns) == list(
    X_test.columns
)


# ------------------------------------------------------------
# 12. PRINT FEATURE MATRIX INFORMATION
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("FEATURE MATRIX")
print("=" * 75)

print(
    "X_train shape:",
    X_train.shape
)

print(
    "X_validation shape:",
    X_validation.shape
)

print(
    "X_test shape:",
    X_test.shape
)

print("\nFinal feature columns:")

for i, feature in enumerate(
    X_train.columns,
    start=1
):
    print(
        f"{i:2d}. {feature}"
    )


# ------------------------------------------------------------
# 13. TARGET CHECK
# ------------------------------------------------------------

print("\nTarget sizes:")

print(
    "y_train:",
    len(y_train),
    "| At risk:",
    y_train.sum()
)

print(
    "y_validation:",
    len(y_validation),
    "| At risk:",
    y_validation.sum()
)

print(
    "y_test:",
    len(y_test),
    "| At risk:",
    y_test.sum()
)


# ------------------------------------------------------------
# 14. SAVE FEATURE MATRICES
# ------------------------------------------------------------

X_train_file = DATA_DIR / "X_train.csv"
X_validation_file = DATA_DIR / "X_validation.csv"
X_test_file = DATA_DIR / "X_test.csv"

y_train_file = DATA_DIR / "y_train.csv"
y_validation_file = DATA_DIR / "y_validation.csv"
y_test_file = DATA_DIR / "y_test.csv"


X_train.to_csv(
    X_train_file,
    index=False
)

X_validation.to_csv(
    X_validation_file,
    index=False
)

X_test.to_csv(
    X_test_file,
    index=False
)

y_train.to_csv(
    y_train_file,
    index=False,
    header=True
)

y_validation.to_csv(
    y_validation_file,
    index=False,
    header=True
)

y_test.to_csv(
    y_test_file,
    index=False,
    header=True
)


# ------------------------------------------------------------
# 15. SAVE FEATURE MANIFEST
# ------------------------------------------------------------

feature_manifest = pd.DataFrame(
    {
        "feature_order": range(
            1,
            len(X_train.columns) + 1
        ),
        "feature": X_train.columns,
    }
)

manifest_file = (
    DATA_DIR
    / "canonical_feature_manifest.csv"
)

feature_manifest.to_csv(
    manifest_file,
    index=False
)


# ------------------------------------------------------------
# 16. FINAL CHECKS
# ------------------------------------------------------------

assert X_train.shape[0] == len(train)
assert X_validation.shape[0] == len(validation)
assert X_test.shape[0] == len(test)

assert len(y_train) == len(train)
assert len(y_validation) == len(validation)
assert len(y_test) == len(test)

assert X_train.shape[1] == X_validation.shape[1]
assert X_train.shape[1] == X_test.shape[1]


print("\n" + "=" * 75)
print("STEP 19A COMPLETED SUCCESSFULLY")
print("=" * 75)

print("\nCreated:")
print(X_train_file)
print(X_validation_file)
print(X_test_file)
print(y_train_file)
print(y_validation_file)
print(y_test_file)
print(manifest_file)

print("\n✓ Canonical feature set established")
print("✓ No target variables included as predictors")
print("✓ Train/validation/test feature columns aligned")
print("✓ No missing feature values")
print("✓ Ready for controlled model training")

print("\nDo NOT train models yet.")