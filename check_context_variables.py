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

VARIABLES = [
    "repeat_student",
    "meals_access",
    "distance_km",
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

    # Keep actual student records
    df = df[
        df["student_id"]
        .astype(str)
        .str.startswith("STU_")
    ].copy()

    print("\n" + "=" * 75)
    print(sheet)
    print("=" * 75)

    print("Records:", len(df))

    for col in VARIABLES:

        print(f"\n--- {col} ---")

        print("Missing:", df[col].isna().sum())

        print("Unique values:")
        print(
            df[col]
            .value_counts(dropna=False)
            .sort_index()
            .to_string()
        )

        if col == "distance_km":
            print("\nDescriptive statistics:")
            print(df[col].describe().to_string())