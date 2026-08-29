import pandas as pd
from pathlib import Path

print("=" * 70)
print("STEP 16.10 — MODELING DATA PREPARATION")
print("=" * 70)

# ---------------------------------------------------------
# 1. PROJECT PATHS
# ---------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent

INPUT_FILE = PROJECT_DIR / "data" / "processed" / "model_data.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

MODEL_DATA_FILE = OUTPUT_DIR / "modeling_dataset.csv"
X_FILE = OUTPUT_DIR / "X_features.csv"
Y_FILE = OUTPUT_DIR / "y_target.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\nInput file:")
print(INPUT_FILE)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nModel data not found:\n{INPUT_FILE}"
    )

# ---------------------------------------------------------
# 2. LOAD VERIFIED DATASET
# ---------------------------------------------------------
df = pd.read_csv(INPUT_FILE)

print(f"\nLoaded dataset shape: {df.shape}")

# ---------------------------------------------------------
# 3. VERIFY EXPECTED RECORD COUNT
# ---------------------------------------------------------
if len(df) != 800:
    raise ValueError(
        f"Expected 800 records, but found {len(df)}."
    )

print("✓ Exactly 800 student-year records confirmed.")

# ---------------------------------------------------------
# 4. CHECK REQUIRED COLUMNS
# ---------------------------------------------------------
required_columns = [
    "student_id",
    "class_level",
    "academic_year",
    "gender",
    "age_years",
    "avg_days_present",
    "avg_total_school_days",
    "avg_attendance_rate",
    "avg_mathematics",
    "avg_english",
    "avg_science_social",
    "overall_average_score"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("✓ Required columns confirmed.")

# ---------------------------------------------------------
# 5. CHECK TARGET
# ---------------------------------------------------------
TARGET = "overall_average_score"

if df[TARGET].isna().any():
    raise ValueError("Target contains missing values.")

print("\nTarget variable:")
print(f"  {TARGET}")

print("\nTarget statistics:")
print(df[TARGET].describe())

# ---------------------------------------------------------
# 6. PREVENT TARGET LEAKAGE
# ---------------------------------------------------------
# These variables are mathematical components of the target.
#
# overall_average_score =
# mean(avg_mathematics, avg_english, avg_science_social)
#
# Therefore they MUST NOT be used as predictors in the
# primary predictive model.

LEAKAGE_COLUMNS = [
    "avg_mathematics",
    "avg_english",
    "avg_science_social"
]

print("\nTarget leakage protection:")
for col in LEAKAGE_COLUMNS:
    print(f"  Excluding: {col}")

# ---------------------------------------------------------
# 7. DEFINE IDENTIFIER / NON-PREDICTIVE COLUMNS
# ---------------------------------------------------------
IDENTIFIER_COLUMNS = [
    "student_id"
]

# ---------------------------------------------------------
# 8. DEFINE PREDICTORS
# ---------------------------------------------------------
FEATURE_COLUMNS = [
    "class_level",
    "academic_year",
    "gender",
    "age_years",
    "avg_days_present",
    "avg_total_school_days",
    "avg_attendance_rate"
]

X = df[FEATURE_COLUMNS].copy()
y = df[TARGET].copy()

# ---------------------------------------------------------
# 9. CHECK FEATURE MISSING VALUES
# ---------------------------------------------------------
feature_missing = X.isna().sum()

print("\nFeature missing values:")
print(feature_missing)

if feature_missing.sum() > 0:
    raise ValueError(
        "Missing predictor values detected. "
        "Resolve before model training."
    )

print("✓ No missing predictor values.")

# ---------------------------------------------------------
# 10. CHECK TARGET LEAKAGE
# ---------------------------------------------------------
for col in LEAKAGE_COLUMNS:
    if col in X.columns:
        raise ValueError(
            f"TARGET LEAKAGE DETECTED: {col} is in X."
        )

print("\n✓ Target leakage check passed.")

# ---------------------------------------------------------
# 11. DISPLAY FEATURES
# ---------------------------------------------------------
print("\nPredictor variables:")
for i, col in enumerate(FEATURE_COLUMNS, start=1):
    print(f"  {i}. {col}")

print(f"\nNumber of predictors: {len(FEATURE_COLUMNS)}")

# ---------------------------------------------------------
# 12. CATEGORICAL VARIABLE DISTRIBUTIONS
# ---------------------------------------------------------
print("\nCLASS LEVEL DISTRIBUTION:")
print(df["class_level"].value_counts())

print("\nACADEMIC YEAR DISTRIBUTION:")
print(df["academic_year"].value_counts())

print("\nGENDER DISTRIBUTION:")
print(df["gender"].value_counts())

# ---------------------------------------------------------
# 13. SAVE CLEAN MODELING DATASET
# ---------------------------------------------------------
modeling_df = df[
    FEATURE_COLUMNS + [TARGET]
].copy()

modeling_df.to_csv(
    MODEL_DATA_FILE,
    index=False
)

X.to_csv(
    X_FILE,
    index=False
)

y.to_csv(
    Y_FILE,
    index=False,
    header=True
)

# ---------------------------------------------------------
# 14. FINAL SUMMARY
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 16.10 COMPLETE")
print("=" * 70)

print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")

print(f"\nModeling dataset saved to:")
print(MODEL_DATA_FILE)

print(f"\nFeatures saved to:")
print(X_FILE)

print(f"\nTarget saved to:")
print(Y_FILE)

print("\nFinal predictor columns:")
print(list(X.columns))

print("\nTarget:")
print(TARGET)

print("\n✓ Dataset is ready for model training.")