
import pandas as pd
from pathlib import Path

print("=" * 70)
print("STEP 16.9 — PREPARING VERIFIED MODELING DATASET")
print("=" * 70)

# ---------------------------------------------------------
# 1. FILE PATH
# ---------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
INPUT_FILE = PROJECT_DIR / "data" / "raw" / "JHS_Data_Collection_1.xlsx"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "model_data.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nExcel file:")
print(INPUT_FILE)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Excel file not found:\n{INPUT_FILE}")

# ---------------------------------------------------------
# 2. LOAD AVERAGED DATASET
# ---------------------------------------------------------
df_raw = pd.read_excel(
    INPUT_FILE,
    sheet_name="📊 AVERAGED DATASET",
    header=None
)

print(f"\nOriginal shape: {df_raw.shape}")

# ---------------------------------------------------------
# 3. USE THE KNOWN DATA HEADER
# ---------------------------------------------------------
columns = [
    "student_id",
    "class_level",
    "academic_year",
    "gender_age",
    "avg_days_present",
    "avg_total_school_days",
    "avg_attendance_rate",
    "avg_mathematics",
    "avg_english",
    "avg_science_social",
    "promotion_status",
    "at_risk"
]

# The actual student records begin after the instruction/header rows.
# We identify valid records using the student ID pattern.
df = df_raw.copy()

df = df[
    ~(
        (df[0].astype(str) == "STU_001") &
        (df[2].astype(str).isin(["2022-23", "2023-24"]))
    )
].copy()

df.columns = columns

print(f"Valid student-year records found: {len(df)}")

# ---------------------------------------------------------
# 4. CLEAN DATA TYPES
# ---------------------------------------------------------
numeric_columns = [
    "avg_days_present",
    "avg_total_school_days",
    "avg_attendance_rate",
    "avg_mathematics",
    "avg_english",
    "avg_science_social"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["academic_year"] = (
    df["academic_year"]
    .astype(str)
    .str.replace("–", "-", regex=False)
    .str.replace("—", "-", regex=False)
)

# ---------------------------------------------------------
# 5. EXTRACT GENDER AND AGE
# ---------------------------------------------------------
gender_age_clean = (
    df["gender_age"]
    .astype(str)
    .str.replace(" ", "", regex=False)
)

df["gender"] = gender_age_clean.str.extract(r"^([MF])", expand=False)
df["age_years"] = pd.to_numeric(
    gender_age_clean.str.extract(r"/(\d+)", expand=False),
    errors="coerce"
)

# ---------------------------------------------------------
# 6. CREATE OVERALL ACADEMIC PERFORMANCE TARGET
# ---------------------------------------------------------
# Academic performance is represented by the mean of:
# Mathematics, English, and Science/Social Studies.

df["overall_average_score"] = df[
    [
        "avg_mathematics",
        "avg_english",
        "avg_science_social"
    ]
].mean(axis=1)

# ---------------------------------------------------------
# 7. REMOVE RECORDS WITHOUT A VALID TARGET
# ---------------------------------------------------------
before_target_filter = len(df)

df = df.dropna(
    subset=["overall_average_score"]
).copy()

removed = before_target_filter - len(df)

print(f"\nRecords removed because target was missing: {removed}")

# ---------------------------------------------------------
# 8. CHECK TARGET
# ---------------------------------------------------------
print("\nTARGET VARIABLE")
print("-" * 70)
print("Target: overall_average_score")
print("Type: Continuous academic-performance score")

print("\nTarget summary:")
print(df["overall_average_score"].describe())

# ---------------------------------------------------------
# 9. CHECK CLASS/GENDER DISTRIBUTION
# ---------------------------------------------------------
print("\nCLASS LEVELS")
print("-" * 70)
print(df["class_level"].value_counts(dropna=False))

print("\nACADEMIC YEARS")
print("-" * 70)
print(df["academic_year"].value_counts(dropna=False))

print("\nGENDER")
print("-" * 70)
print(df["gender"].value_counts(dropna=False))

# ---------------------------------------------------------
# 10. CHECK MISSING VALUES
# ---------------------------------------------------------
print("\nMISSING VALUES")
print("-" * 70)

missing = df.isna().sum()
print(missing[missing > 0])

# ---------------------------------------------------------
# 11. CHECK DUPLICATE STUDENT-YEAR RECORDS
# ---------------------------------------------------------
duplicates = df.duplicated(
    subset=["student_id", "academic_year"]
).sum()

print("\nDUPLICATE STUDENT-YEAR RECORDS")
print("-" * 70)
print(f"Duplicates: {duplicates}")

# ---------------------------------------------------------
# 12. CREATE MODELING DATASET
# ---------------------------------------------------------
model_columns = [
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

model_df = df[model_columns].copy()

# ---------------------------------------------------------
# 13. SAVE
# ---------------------------------------------------------
model_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("STEP 16.9 COMPLETE")
print("=" * 70)

print(f"\nFinal modeling dataset shape: {model_df.shape}")
print(f"Saved to:\n{OUTPUT_FILE}")

print("\nFinal columns:")
print(list(model_df.columns))

print("\nFirst 5 records:")
print(model_df.head().to_string(index=False))

print("\nDataset ready for the next modeling-preparation stage.")