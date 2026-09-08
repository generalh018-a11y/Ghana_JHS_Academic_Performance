from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# STEP 17
# BUILD TERM-LEVEL EARLY-WARNING DATASET
# ============================================================

print("=" * 75)
print("STEP 17 — BUILD TERM-LEVEL EARLY-WARNING DATASET")
print("=" * 75)


# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_FILE = (
    Path(r"C:\Users\hills\OneDrive\Documents\Data Collection__Thesis")
    / "JHS_Data_Collection_1.xlsx"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "term_level_early_warning_dataset.csv"


# ------------------------------------------------------------
# 2. SHEET NAMES
# ------------------------------------------------------------

SHEETS = {
    "2022-23": {
        "T1": "📘 2022-23 Term 1",
        "T2": "📘 2022-23 Term 2",
        "T3": "📘 2022-23 Term 3",
    },
    "2023-24": {
        "T1": "📗 2023-24 Term 1",
        "T2": "📗 2023-24 Term 2",
        "T3": "📗 2023-24 Term 3",
    },
}


# ------------------------------------------------------------
# 3. READ AND CLEAN TERM SHEETS
# ------------------------------------------------------------

def load_term_sheet(sheet_name):
    """
    Load one term sheet.

    Excel row 6 contains the real column headings,
    while the following instructional row is removed by
    retaining only genuine STU_ student records.
    """

    df = pd.read_excel(
        DATA_FILE,
        sheet_name=sheet_name,
        header=6
    )

    # Clean column names.
    df.columns = [
        str(col).split("\n")[0].strip()
        for col in df.columns
    ]

    # Keep only genuine student records.
    df["student_id"] = df["student_id"].astype(str).str.strip()

    df = df[
        df["student_id"].str.startswith("STU_")
    ].copy()

    # Convert relevant numeric columns.
    numeric_columns = [
        "age_years",
        "days_present",
        "total_school_days",
        "attendance_rate_%",
        "term_average_score",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    return df


# ------------------------------------------------------------
# 4. LOAD ALL SIX TERM SHEETS
# ------------------------------------------------------------

data = {}

for academic_year, terms in SHEETS.items():

    print(f"\nLoading academic year: {academic_year}")

    for term, sheet_name in terms.items():

        df = load_term_sheet(sheet_name)

        data[(academic_year, term)] = df

        print(
            f"  {term}: {len(df)} student records | "
            f"{df['student_id'].nunique()} unique students"
        )


# ------------------------------------------------------------
# 5. CONSTRUCT EACH STUDENT-YEAR TRAJECTORY
# ------------------------------------------------------------

records = []

for academic_year in SHEETS.keys():

    t1 = data[(academic_year, "T1")].copy()
    t2 = data[(academic_year, "T2")].copy()
    t3 = data[(academic_year, "T3")].copy()

    # --------------------------------------------------------
    # Check uniqueness before merging
    # --------------------------------------------------------

    assert t1["student_id"].is_unique, (
        f"Duplicate student IDs found in {academic_year} T1"
    )

    assert t2["student_id"].is_unique, (
        f"Duplicate student IDs found in {academic_year} T2"
    )

    assert t3["student_id"].is_unique, (
        f"Duplicate student IDs found in {academic_year} T3"
    )

    # --------------------------------------------------------
    # Rename columns so their temporal origin is explicit
    # --------------------------------------------------------

    t1 = t1.rename(
        columns={
            "age_years": "age_t1",
            "attendance_rate_%": "attendance_t1",
            "term_average_score": "score_t1",
            "class_level": "class_t1",
            "gender": "gender_t1",
        }
    )

    t2 = t2.rename(
        columns={
            "age_years": "age_t2",
            "attendance_rate_%": "attendance_t2",
            "term_average_score": "score_t2",
            "class_level": "class_t2",
            "gender": "gender_t2",
        }
    )

    t3 = t3.rename(
        columns={
            "age_years": "age_t3",
            "attendance_rate_%": "attendance_t3",
            "term_average_score": "score_t3",
            "class_level": "class_t3",
            "gender": "gender_t3",
        }
    )

    # --------------------------------------------------------
    # Keep only variables required for early-warning modeling
    # --------------------------------------------------------

    t1_keep = t1[
        [
            "student_id",
            "age_t1",
            "attendance_t1",
            "score_t1",
            "class_t1",
            "gender_t1",
        ]
    ]

    t2_keep = t2[
        [
            "student_id",
            "age_t2",
            "attendance_t2",
            "score_t2",
            "class_t2",
            "gender_t2",
        ]
    ]

    t3_keep = t3[
        [
            "student_id",
            "age_t3",
            "attendance_t3",
            "score_t3",
            "class_t3",
            "gender_t3",
        ]
    ]

    # --------------------------------------------------------
    # Inner merge:
    # Student must exist in T1, T2 and T3.
    # --------------------------------------------------------

    merged = t1_keep.merge(
        t2_keep,
        on="student_id",
        how="inner",
        validate="one_to_one",
    )

    merged = merged.merge(
        t3_keep,
        on="student_id",
        how="inner",
        validate="one_to_one",
    )

    merged["academic_year"] = academic_year

    records.append(merged)


# ------------------------------------------------------------
# 6. COMBINE BOTH ACADEMIC YEARS
# ------------------------------------------------------------

df = pd.concat(
    records,
    ignore_index=True
)

print("\nCombined student-year trajectories:", len(df))


# ------------------------------------------------------------
# 7. VERIFY STUDENT-YEAR UNIQUENESS
# ------------------------------------------------------------

duplicates = df.duplicated(
    subset=["student_id", "academic_year"]
).sum()

print(
    "Duplicate student-year trajectories:",
    duplicates
)

assert duplicates == 0, (
    "Duplicate student-year trajectories detected."
)


# ------------------------------------------------------------
# 8. CHECK TERM-LEVEL PREDICTOR COMPLETENESS
# ------------------------------------------------------------

predictor_columns = [
    "age_t1",
    "attendance_t1",
    "score_t1",
    "age_t2",
    "attendance_t2",
    "score_t2",
]

print("\nMissing predictor values:")
print(
    df[predictor_columns].isna().sum()
)


# ------------------------------------------------------------
# 9. CHECK TERM 3 OUTCOME
# ------------------------------------------------------------

print("\nMissing Term 3 outcomes before filtering:")

print(
    df["score_t3"].isna().sum()
)


# ------------------------------------------------------------
# 10. KEEP ONLY OBSERVED TERM 3 OUTCOMES
# ------------------------------------------------------------

before_filter = len(df)

df = df[
    df["score_t3"].notna()
].copy()

after_filter = len(df)

print(
    "\nRows removed because Term 3 outcome was unavailable:",
    before_filter - after_filter
)

print(
    "Eligible student-year trajectories:",
    after_filter
)


# ------------------------------------------------------------
# 11. VERIFY NO PREDICTOR MISSINGNESS
# ------------------------------------------------------------

missing_predictors = (
    df[predictor_columns]
    .isna()
    .sum()
)

print("\nMissing predictors after eligibility filtering:")

print(missing_predictors)

assert missing_predictors.sum() == 0, (
    "Missing predictor values remain."
)


# ------------------------------------------------------------
# 12. VERIFY TEMPORAL ORDER
# ------------------------------------------------------------

# T2 should not precede T1.
# We retain age from T2 for checking progression.

age_difference = (
    df["age_t2"] - df["age_t1"]
)

print("\nAge progression from T1 to T2:")

print(
    age_difference.value_counts()
    .sort_index()
)


# ------------------------------------------------------------
# 13. CHECK CLASS AND GENDER CONSISTENCY
# ------------------------------------------------------------

class_changes = (
    df["class_t1"] != df["class_t2"]
).sum()

gender_changes = (
    df["gender_t1"] != df["gender_t2"]
).sum()

print(
    "\nT1 → T2 class-level changes:",
    class_changes
)

print(
    "T1 → T2 gender changes:",
    gender_changes
)


# ------------------------------------------------------------
# 14. CREATE MODELING VARIABLES
# ------------------------------------------------------------

df["age_years"] = df["age_t2"]

df["class_level"] = df["class_t2"]

df["gender"] = df["gender_t2"]

# Term 3 target.
df["target_score_t3"] = df["score_t3"]

# Binary classification target.
df["at_risk"] = np.where(
    df["target_score_t3"] < 50,
    1,
    0
)

df["at_risk_label"] = np.where(
    df["at_risk"] == 1,
    "At risk",
    "Not at risk"
)


# ------------------------------------------------------------
# 15. ATTENDANCE-DECAY FEATURE
# ------------------------------------------------------------

# Current planned decay parameter.
# This is an engineered feature and will be sensitivity-tested later.

LAMBDA = 0.5

w1 = np.exp(-LAMBDA * 1)
w2 = np.exp(-LAMBDA * 0)

weight_sum = w1 + w2

w1 = w1 / weight_sum
w2 = w2 / weight_sum

df["attendance_decay"] = (
    w1 * df["attendance_t1"]
    + w2 * df["attendance_t2"]
)

print("\nAttendance-decay weights:")

print(
    f"  Lambda = {LAMBDA}"
)

print(
    f"  T1 weight = {w1:.6f}"
)

print(
    f"  T2 weight = {w2:.6f}"
)


# ------------------------------------------------------------
# 16. SELECT FINAL DATASET COLUMNS
# ------------------------------------------------------------

final_columns = [
    "student_id",
    "academic_year",
    "class_level",
    "gender",
    "age_years",

    # Historical academic performance
    "score_t1",
    "score_t2",

    # Historical attendance
    "attendance_t1",
    "attendance_t2",

    # Engineered attendance feature
    "attendance_decay",

    # Future target
    "target_score_t3",
    "at_risk",
    "at_risk_label",
]

final_df = df[
    final_columns
].copy()


# ------------------------------------------------------------
# 17. SORT DATA
# ------------------------------------------------------------

final_df = final_df.sort_values(
    by=[
        "academic_year",
        "student_id",
    ]
).reset_index(drop=True)


# ------------------------------------------------------------
# 18. FINAL INTEGRITY CHECKS
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("FINAL DATASET VALIDATION")
print("=" * 75)

print(
    "Rows:",
    len(final_df)
)

print(
    "Columns:",
    len(final_df.columns)
)

print(
    "Unique students:",
    final_df["student_id"].nunique()
)

print(
    "Unique student-year trajectories:",
    final_df[
        ["student_id", "academic_year"]
    ].drop_duplicates().shape[0]
)

print("\nAcademic year distribution:")

print(
    final_df["academic_year"]
    .value_counts()
    .sort_index()
)

print("\nClass distribution:")

print(
    final_df["class_level"]
    .value_counts()
    .sort_index()
)

print("\nGender distribution:")

print(
    final_df["gender"]
    .value_counts()
    .sort_index()
)

print("\nTarget distribution:")

print(
    final_df["at_risk_label"]
    .value_counts()
)

print("\nTarget percentages:")

print(
    (
        final_df["at_risk_label"]
        .value_counts(normalize=True)
        * 100
    ).round(2)
)

print("\nMissing values:")

print(
    final_df.isna().sum()
)


# ------------------------------------------------------------
# 19. FINAL ASSERTIONS
# ------------------------------------------------------------

assert len(final_df) == 541, (
    f"Expected 541 eligible observations, "
    f"but obtained {len(final_df)}."
)

assert final_df["student_id"].notna().all()

assert final_df["academic_year"].notna().all()

assert final_df["at_risk"].isin([0, 1]).all()

assert final_df[
    ["score_t1", "score_t2",
     "attendance_t1", "attendance_t2",
     "attendance_decay"]
].notna().all().all()

assert final_df[
    ["student_id", "academic_year"]
].duplicated().sum() == 0


# ------------------------------------------------------------
# 20. SAVE DATASET
# ------------------------------------------------------------

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 75)
print("DATASET SUCCESSFULLY CREATED")
print("=" * 75)

print(
    "Saved to:",
    OUTPUT_FILE
)

print("\nFirst five rows:")

print(
    final_df.head().to_string(index=False)
)

print("\nStep 17 completed successfully.")