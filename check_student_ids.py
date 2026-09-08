import pandas as pd

# ============================================================
# CHECK STUDENT IDs ACROSS ALL SIX TERM SHEETS
# ============================================================

FILE = r"C:\Users\hills\OneDrive\Documents\Data Collection__Thesis\JHS_Data_Collection_1.xlsx"

SHEETS = [
    "📘 2022-23 Term 1",
    "📘 2022-23 Term 2",
    "📘 2022-23 Term 3",
    "📗 2023-24 Term 1",
    "📗 2023-24 Term 2",
    "📗 2023-24 Term 3",
]

dfs = {}

for sheet in SHEETS:
    df = pd.read_excel(FILE, sheet_name=sheet, header=6)

    # Clean column names
    df.columns = [
        str(col).split("\n")[0].strip()
        for col in df.columns
    ]

    # Remove the orange instruction row and keep actual student records
    df = df[df["student_id"].astype(str).str.startswith("STU_")].copy()

    dfs[sheet] = df

    print(f"{sheet}")
    print(f"  Actual records: {len(df)}")
    print(f"  Unique student IDs: {df['student_id'].nunique()}")
    print(f"  Duplicate student-ID rows: {df['student_id'].duplicated().sum()}")
    print()

# ------------------------------------------------------------
# Within-year continuity
# ------------------------------------------------------------

y1_t1 = set(dfs[SHEETS[0]]["student_id"])
y1_t2 = set(dfs[SHEETS[1]]["student_id"])
y1_t3 = set(dfs[SHEETS[2]]["student_id"])

y2_t1 = set(dfs[SHEETS[3]]["student_id"])
y2_t2 = set(dfs[SHEETS[4]]["student_id"])
y2_t3 = set(dfs[SHEETS[5]]["student_id"])

print("=" * 60)
print("WITHIN-YEAR STUDENT-ID OVERLAP")
print("=" * 60)

print("2022-23:")
print("  T1 ∩ T2:", len(y1_t1 & y1_t2))
print("  T1 ∩ T3:", len(y1_t1 & y1_t3))
print("  T2 ∩ T3:", len(y1_t2 & y1_t3))
print("  T1 ∩ T2 ∩ T3:", len(y1_t1 & y1_t2 & y1_t3))

print()

print("2023-24:")
print("  T1 ∩ T2:", len(y2_t1 & y2_t2))
print("  T1 ∩ T3:", len(y2_t1 & y2_t3))
print("  T2 ∩ T3:", len(y2_t2 & y2_t3))
print("  T1 ∩ T2 ∩ T3:", len(y2_t1 & y2_t2 & y2_t3))

# ------------------------------------------------------------
# Cross-year overlap
# ------------------------------------------------------------

all_y1 = y1_t1 | y1_t2 | y1_t3
all_y2 = y2_t1 | y2_t2 | y2_t3

print()
print("=" * 60)
print("CROSS-YEAR STUDENT-ID OVERLAP")
print("=" * 60)

print("IDs appearing in BOTH academic years:", len(all_y1 & all_y2))
print("Unique IDs across both years:", len(all_y1 | all_y2))