import pandas as pd

FILE = r"C:\Users\hills\OneDrive\Documents\Data Collection__Thesis\JHS_Data_Collection_1.xlsx"

SHEETS = [
    "📘 2022-23 Term 1",
    "📘 2022-23 Term 2",
    "📘 2022-23 Term 3",
    "📗 2023-24 Term 1",
    "📗 2023-24 Term 2",
    "📗 2023-24 Term 3",
]

KEY_COLUMNS = [
    "student_id",
    "class_level",
    "gender",
    "age_years",
    "days_present",
    "total_school_days",
    "attendance_rate_%",
    "score_mathematics",
    "score_english",
    "score_science",
    "score_social_studies",
    "term_average_score",
]

for sheet in SHEETS:

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

    # Keep only actual student rows
    df = df[
        df["student_id"]
        .astype(str)
        .str.startswith("STU_")
    ].copy()

    print("\n" + "=" * 75)
    print(sheet)
    print("=" * 75)

    print("Total student records:", len(df))

    print("\nMissing values — ALL STUDENTS:")

    for col in KEY_COLUMNS:
        missing = df[col].isna().sum()
        print(f"{col:25} {missing}")

    print("\nMissing values — BY CLASS LEVEL:")

    for level in ["JHS1", "JHS2", "JHS3"]:

        subset = df[df["class_level"] == level]

        print(f"\n{level} — records: {len(subset)}")

        for col in KEY_COLUMNS:
            missing = subset[col].isna().sum()
            print(f"  {col:23} {missing}")