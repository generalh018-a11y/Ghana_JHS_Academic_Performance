import pandas as pd

FILE = r"C:\Users\hills\OneDrive\Documents\Data Collection__Thesis\JHS_Data_Collection_1.xlsx"

SHEETS = {
    "2022-23_T1": "📘 2022-23 Term 1",
    "2022-23_T2": "📘 2022-23 Term 2",
    "2022-23_T3": "📘 2022-23 Term 3",
    "2023-24_T1": "📗 2023-24 Term 1",
    "2023-24_T2": "📗 2023-24 Term 2",
    "2023-24_T3": "📗 2023-24 Term 3",
}

data = {}

for label, sheet in SHEETS.items():

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

    # Keep only real student records
    df = df[
        df["student_id"]
        .astype(str)
        .str.startswith("STU_")
    ].copy()

    data[label] = df[
        ["student_id", "class_level", "gender", "age_years"]
    ]

# ------------------------------------------------------------
# Create year-level student records
# ------------------------------------------------------------

year_1 = (
    data["2022-23_T1"]
    .rename(columns={
        "class_level": "class_2022_23",
        "gender": "gender_2022_23",
        "age_years": "age_2022_23"
    })
)

year_2 = (
    data["2023-24_T1"]
    .rename(columns={
        "class_level": "class_2023_24",
        "gender": "gender_2023_24",
        "age_years": "age_2023_24"
    })
)

# Keep only students appearing in both years
overlap = pd.merge(
    year_1,
    year_2,
    on="student_id",
    how="inner"
)

print("=" * 70)
print("CROSS-YEAR STUDENT PROGRESSION")
print("=" * 70)

print("Students appearing in both years:", len(overlap))

print("\nCLASS TRANSITIONS:")
print(
    overlap
    .groupby(["class_2022_23", "class_2023_24"])
    .size()
    .sort_index()
    .to_string()
)

print("\nDETAILED TRANSITION COUNTS:")

transitions = (
    overlap
    .groupby(["class_2022_23", "class_2023_24"])
    .size()
    .reset_index(name="students")
)

print(transitions.to_string(index=False))

print("\nAGE CHANGES:")
overlap["age_change"] = (
    overlap["age_2023_24"] -
    overlap["age_2022_23"]
)

print(
    overlap["age_change"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nGENDER CHANGES:")
gender_changes = (
    overlap["gender_2022_23"] !=
    overlap["gender_2023_24"]
).sum()

print("Students with changed gender value:", gender_changes)

print("\nFIRST 20 CROSS-YEAR MATCHES:")
print(overlap.head(20).to_string(index=False))