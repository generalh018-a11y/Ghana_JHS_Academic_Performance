import pandas as pd

FILE = r"C:\Users\hills\OneDrive\Documents\Data Collection__Thesis\JHS_Data_Collection_1.xlsx"

SHEETS = {
    "2022-23": "📘 2022-23 Term 3",
    "2023-24": "📗 2023-24 Term 3",
}

frames = []

for year, sheet in SHEETS.items():

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

    df["academic_year"] = year

    frames.append(df)

t3 = pd.concat(frames, ignore_index=True)

# Only students with observed T3 academic outcomes
t3 = t3[
    t3["term_average_score"].notna()
].copy()

# Classification target
t3["at_risk"] = (
    t3["term_average_score"] < 50
).astype(int)

print("=" * 75)
print("TERM 3 TARGET DISTRIBUTION")
print("=" * 75)

print("Observed Term 3 outcomes:", len(t3))

print("\nOverall descriptive statistics:")
print(
    t3["term_average_score"]
    .describe()
    .to_string()
)

print("\nAt-risk distribution:")
print(
    t3["at_risk"]
    .value_counts()
    .sort_index()
    .rename(index={
        0: "Not at risk (>=50)",
        1: "At risk (<50)"
    })
    .to_string()
)

print("\nAt-risk percentages:")
print(
    (t3["at_risk"].value_counts(normalize=True) * 100)
    .sort_index()
    .rename(index={
        0: "Not at risk (>=50)",
        1: "At risk (<50)"
    })
    .round(2)
    .to_string()
)

print("\nBy academic year:")
print(
    pd.crosstab(
        t3["academic_year"],
        t3["at_risk"],
        margins=True
    )
    .rename(columns={
        0: "Not at risk",
        1: "At risk"
    })
    .to_string()
)

print("\nBy class level:")
print(
    pd.crosstab(
        t3["class_level"],
        t3["at_risk"],
        margins=True
    )
    .rename(columns={
        0: "Not at risk",
        1: "At risk"
    })
    .to_string()
)

print("\nBy gender:")
print(
    pd.crosstab(
        t3["gender"],
        t3["at_risk"],
        margins=True
    )
    .rename(columns={
        0: "Not at risk",
        1: "At risk"
    })
    .to_string()
)

print("\nMean T3 score by class:")
print(
    t3.groupby("class_level")["term_average_score"]
    .agg(["count", "mean", "std", "min", "median", "max"])
    .round(3)
    .to_string()
)

print("\nMean T3 score by gender:")
print(
    t3.groupby("gender")["term_average_score"]
    .agg(["count", "mean", "std", "min", "median", "max"])
    .round(3)
    .to_string()
)