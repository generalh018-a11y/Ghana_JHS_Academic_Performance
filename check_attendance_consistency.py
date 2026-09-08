import pandas as pd
import numpy as np

FILE = r"C:\Users\hills\OneDrive\Documents\Data Collection__Thesis\JHS_Data_Collection_1.xlsx"

SHEETS = [
    "📘 2022-23 Term 1",
    "📘 2022-23 Term 2",
    "📘 2022-23 Term 3",
    "📗 2023-24 Term 1",
    "📗 2023-24 Term 2",
    "📗 2023-24 Term 3",
]

all_results = []

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

    # Calculate attendance independently
    df["calculated_attendance_rate"] = (
        df["days_present"] /
        df["total_school_days"] *
        100
    )

    df["attendance_difference"] = (
        df["attendance_rate_%"] -
        df["calculated_attendance_rate"]
    ).abs()

    all_results.append(
        df[
            [
                "student_id",
                "class_level",
                "days_present",
                "total_school_days",
                "attendance_rate_%",
                "calculated_attendance_rate",
                "attendance_difference",
            ]
        ]
    )

combined = pd.concat(
    all_results,
    ignore_index=True
)

print("=" * 75)
print("ATTENDANCE CONSISTENCY AUDIT")
print("=" * 75)

print("Total term records:", len(combined))

print(
    "\nMaximum absolute difference:",
    combined["attendance_difference"].max()
)

print(
    "Mean absolute difference:",
    combined["attendance_difference"].mean()
)

print(
    "Records with difference > 0.1 percentage points:",
    (combined["attendance_difference"] > 0.1).sum()
)

print(
    "Records with difference > 1 percentage point:",
    (combined["attendance_difference"] > 1).sum()
)

print("\nAttendance descriptive statistics:")
print(
    combined[
        [
            "days_present",
            "total_school_days",
            "attendance_rate_%",
        ]
    ].describe().to_string()
)

print("\nLargest discrepancies:")

print(
    combined
    .sort_values("attendance_difference", ascending=False)
    .head(20)
    .to_string(index=False)
)

print("\nCorrelation between attendance variables:")

print(
    combined[
        [
            "days_present",
            "total_school_days",
            "attendance_rate_%",
        ]
    ]
    .corr()
    .to_string()
)