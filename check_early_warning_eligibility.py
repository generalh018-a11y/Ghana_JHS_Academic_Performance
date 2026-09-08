import pandas as pd

FILE = r"C:\Users\hills\OneDrive\Documents\Data Collection__Thesis\JHS_Data_Collection_1.xlsx"

SHEETS = {
    ("2022-23", "T1"): "📘 2022-23 Term 1",
    ("2022-23", "T2"): "📘 2022-23 Term 2",
    ("2022-23", "T3"): "📘 2022-23 Term 3",
    ("2023-24", "T1"): "📗 2023-24 Term 1",
    ("2023-24", "T2"): "📗 2023-24 Term 2",
    ("2023-24", "T3"): "📗 2023-24 Term 3",
}

def load_term(sheet):
    df = pd.read_excel(
        FILE,
        sheet_name=sheet,
        header=6
    )

    # Clean column names
    df.columns = [
        str(col).split("\n")[0].strip()
        for col in df.columns
    ]

    # Keep real student records only
    df = df[
        df["student_id"]
        .astype(str)
        .str.startswith("STU_")
    ].copy()

    return df


# ------------------------------------------------------------
# Load all six terms
# ------------------------------------------------------------

data = {}

for key, sheet in SHEETS.items():
    data[key] = load_term(sheet)


# ------------------------------------------------------------
# Create one student-year trajectory
# T1 + T2 -> T3
# ------------------------------------------------------------

trajectories = []

for year in ["2022-23", "2023-24"]:

    t1 = data[(year, "T1")].copy()
    t2 = data[(year, "T2")].copy()
    t3 = data[(year, "T3")].copy()

    # Rename variables by term
    t1 = t1.rename(columns={
        col: f"T1_{col}"
        for col in t1.columns
        if col != "student_id"
    })

    t2 = t2.rename(columns={
        col: f"T2_{col}"
        for col in t2.columns
        if col != "student_id"
    })

    t3 = t3.rename(columns={
        col: f"T3_{col}"
        for col in t3.columns
        if col != "student_id"
    })

    merged = t1.merge(
        t2,
        on="student_id",
        how="inner"
    )

    merged = merged.merge(
        t3,
        on="student_id",
        how="inner"
    )

    merged["academic_year"] = year

    trajectories.append(merged)


trajectory_df = pd.concat(
    trajectories,
    ignore_index=True
)


# ------------------------------------------------------------
# Basic trajectory count
# ------------------------------------------------------------

print("=" * 75)
print("EARLY-WARNING TRAJECTORY AUDIT")
print("=" * 75)

print("Total possible student-year trajectories:",
      len(trajectory_df))

print("\nBy academic year:")
print(
    trajectory_df["academic_year"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ------------------------------------------------------------
# Target availability
# ------------------------------------------------------------

target_available = (
    trajectory_df["T3_term_average_score"]
    .notna()
)

print("\n" + "=" * 75)
print("TERM 3 TARGET AVAILABILITY")
print("=" * 75)

print("With observed T3 target:",
      target_available.sum())

print("Without observed T3 target:",
      (~target_available).sum())


# ------------------------------------------------------------
# Target availability by class
# ------------------------------------------------------------

print("\nTarget availability by T3 class level:")

class_target = (
    trajectory_df
    .groupby("T3_class_level")["T3_term_average_score"]
    .agg(
        records="size",
        observed="count",
        missing=lambda x: x.isna().sum()
    )
)

print(class_target.to_string())


# ------------------------------------------------------------
# Predictor completeness
# ------------------------------------------------------------

predictor_columns = [
    "T1_age_years",
    "T1_days_present",
    "T1_total_school_days",
    "T1_attendance_rate_%",
    "T1_term_average_score",

    "T2_age_years",
    "T2_days_present",
    "T2_total_school_days",
    "T2_attendance_rate_%",
    "T2_term_average_score",
]

print("\n" + "=" * 75)
print("PREDICTOR COMPLETENESS")
print("=" * 75)

for col in predictor_columns:
    print(
        f"{col:30} missing = "
        f"{trajectory_df[col].isna().sum()}"
    )


# ------------------------------------------------------------
# Complete-case eligibility
# ------------------------------------------------------------

predictors_complete = (
    trajectory_df[predictor_columns]
    .notna()
    .all(axis=1)
)

eligible = (
    target_available &
    predictors_complete
)

print("\n" + "=" * 75)
print("FINAL ELIGIBILITY CHECK")
print("=" * 75)

print("Total trajectories:", len(trajectory_df))
print("Valid T3 target:", target_available.sum())
print("Complete T1/T2 predictors:", predictors_complete.sum())
print("Eligible for supervised modelling:", eligible.sum())

print("\nExcluded because T3 target is missing:",
      (~target_available).sum())

print("Excluded because predictor data are incomplete:",
      (target_available & ~predictors_complete).sum())

print("\nEligible trajectories by academic year:")

print(
    trajectory_df.loc[eligible, "academic_year"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nEligible trajectories by T3 class:")

print(
    trajectory_df.loc[eligible, "T3_class_level"]
    .value_counts()
    .sort_index()
    .to_string()
)
