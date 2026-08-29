import pandas as pd

file = r"data\raw\JHS_Data_Collection_1.xlsx"

sheets = [
    "📘 2022-23 Term 1",
    "📘 2022-23 Term 2",
    "📘 2022-23 Term 3",
    "📗 2023-24 Term 1",
    "📗 2023-24 Term 2",
    "📗 2023-24 Term 3"
]

columns = [
    "student_id",
    "class_level",
    "gender",
    "age_years",
    "days_present",
    "total_school_days",
    "attendance_rate",
    "score_mathematics",
    "score_english",
    "score_science",
    "score_social_studies",
    "score_rme_other",
    "term_average_score",
    "promotion_status",
    "repeat_student",
    "meals_access",
    "distance_km",
    "data_source",
    "entry_date"
]

print("DATA QUALITY AUDIT")
print("=" * 70)

for sheet in sheets:

    df = pd.read_excel(
        file,
        sheet_name=sheet,
        header=None
    )

    # Start from row 8 and keep genuine student records
    df = df.iloc[7:].copy()

    ids = df.iloc[:, 0].astype(str).str.strip()

    df = df[
        ids.str.match(r"^STU_\d+$", na=False)
    ].copy()

    df.columns = columns

    print("\n" + sheet)
    print("-" * 70)

    print("Records:", len(df))
    print("Unique students:", df["student_id"].nunique())

    print("\nMissing values:")
    print(df.isna().sum().to_string())

    print("\nScore ranges:")


score_columns = [
    "score_mathematics",
    "score_english",
    "score_science",
    "score_social_studies",
    "score_rme_other",
    "term_average_score"
]

for column in score_columns:
    values = pd.to_numeric(df[column], errors="coerce")

    print(
        f"{column}: "
        f"min={values.min()}, "
        f"max={values.max()}"
    )
    

    print("\nMeals access:")
    print(df["meals_access"].value_counts(dropna=False).to_dict())

    print("\nRepeat student:")
    print(df["repeat_student"].value_counts(dropna=False).to_dict())

    print("\nDuplicate student IDs:")
    print(df["student_id"].duplicated().sum())

print("\n" + "=" * 70)
print("AUDIT COMPLETE")