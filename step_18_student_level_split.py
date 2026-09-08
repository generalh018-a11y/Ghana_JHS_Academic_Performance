from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# STEP 18
# LEAKAGE-SAFE STUDENT-LEVEL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("=" * 75)
print("STEP 18 — LEAKAGE-SAFE STUDENT-LEVEL DATA SPLIT")
print("=" * 75)


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "term_level_early_warning_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("\nLoaded dataset:")
print("Rows:", len(df))
print("Unique students:", df["student_id"].nunique())


# ------------------------------------------------------------
# 3. BASIC VALIDATION
# ------------------------------------------------------------

assert len(df) == 541

assert (
    df[["student_id", "academic_year"]]
    .duplicated()
    .sum()
    == 0
)

assert df["student_id"].notna().all()

assert df["at_risk"].isin([0, 1]).all()


# ------------------------------------------------------------
# 4. STUDENT-LEVEL GROUPS
# ------------------------------------------------------------

groups = df["student_id"]

unique_students = df["student_id"].unique()

print("\nUnique student groups:", len(unique_students))


# ------------------------------------------------------------
# 5. FIRST SPLIT
#
# 80% development
# 20% final test
#
# IMPORTANT:
# Split by student_id, NOT individual rows.
# ------------------------------------------------------------

gss_test = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

development_idx, test_idx = next(
    gss_test.split(
        df,
        y=df["at_risk"],
        groups=groups
    )
)

development = df.iloc[
    development_idx
].copy()

test = df.iloc[
    test_idx
].copy()


# ------------------------------------------------------------
# 6. SECOND SPLIT
#
# From the 80% development set:
#
# 75% → training
# 25% → validation
#
# Overall approximately:
#
# Train      60%
# Validation 20%
# Test       20%
#
# Again split by student_id.
# ------------------------------------------------------------

gss_validation = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=43
)

train_idx, validation_idx = next(
    gss_validation.split(
        development,
        y=development["at_risk"],
        groups=development["student_id"]
    )
)

train = development.iloc[
    train_idx
].copy()

validation = development.iloc[
    validation_idx
].copy()


# ------------------------------------------------------------
# 7. RESET INDEX
# ------------------------------------------------------------

train = train.reset_index(drop=True)
validation = validation.reset_index(drop=True)
test = test.reset_index(drop=True)


# ------------------------------------------------------------
# 8. CHECK STUDENT OVERLAP
# ------------------------------------------------------------

train_students = set(train["student_id"])
validation_students = set(validation["student_id"])
test_students = set(test["student_id"])


train_validation_overlap = (
    train_students & validation_students
)

train_test_overlap = (
    train_students & test_students
)

validation_test_overlap = (
    validation_students & test_students
)


print("\n" + "=" * 75)
print("STUDENT-LEVEL LEAKAGE CHECK")
print("=" * 75)

print(
    "Train ∩ Validation:",
    len(train_validation_overlap)
)

print(
    "Train ∩ Test:",
    len(train_test_overlap)
)

print(
    "Validation ∩ Test:",
    len(validation_test_overlap)
)

assert len(train_validation_overlap) == 0
assert len(train_test_overlap) == 0
assert len(validation_test_overlap) == 0

print("\n✓ No student appears in more than one split.")


# ------------------------------------------------------------
# 9. CHECK ROW COUNTS
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("SPLIT SIZES")
print("=" * 75)

print(
    f"Train:      {len(train)} rows "
    f"({len(train) / len(df) * 100:.2f}%)"
)

print(
    f"Validation: {len(validation)} rows "
    f"({len(validation) / len(df) * 100:.2f}%)"
)

print(
    f"Test:       {len(test)} rows "
    f"({len(test) / len(df) * 100:.2f}%)"
)

print(
    f"Total:      {len(train) + len(validation) + len(test)} rows"
)

assert (
    len(train)
    + len(validation)
    + len(test)
    == len(df)
)


# ------------------------------------------------------------
# 10. UNIQUE STUDENT COUNTS
# ------------------------------------------------------------

print("\nUnique students by split:")

print(
    "Train:",
    train["student_id"].nunique()
)

print(
    "Validation:",
    validation["student_id"].nunique()
)

print(
    "Test:",
    test["student_id"].nunique()
)

print(
    "Total unique students across original dataset:",
    df["student_id"].nunique()
)


# ------------------------------------------------------------
# 11. TARGET DISTRIBUTION
# ------------------------------------------------------------

def print_target_distribution(name, data):

    counts = (
        data["at_risk"]
        .value_counts()
        .sort_index()
    )

    percentages = (
        data["at_risk"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )

    print(f"\n{name} target distribution:")

    print(
        pd.DataFrame(
            {
                "Count": counts,
                "Percentage": percentages.round(2)
            }
        )
        .rename(
            index={
                0: "Not at risk",
                1: "At risk"
            }
        )
    )


print("\n" + "=" * 75)
print("TARGET DISTRIBUTION BY SPLIT")
print("=" * 75)

print_target_distribution(
    "TRAIN",
    train
)

print_target_distribution(
    "VALIDATION",
    validation
)

print_target_distribution(
    "TEST",
    test
)


# ------------------------------------------------------------
# 12. CLASS DISTRIBUTION
# ------------------------------------------------------------

def print_class_distribution(name, data):

    print(f"\n{name} class distribution:")

    print(
        data["class_level"]
        .value_counts()
        .sort_index()
    )


print("\n" + "=" * 75)
print("CLASS DISTRIBUTION BY SPLIT")
print("=" * 75)

print_class_distribution(
    "TRAIN",
    train
)

print_class_distribution(
    "VALIDATION",
    validation
)

print_class_distribution(
    "TEST",
    test
)


# ------------------------------------------------------------
# 13. GENDER DISTRIBUTION
# ------------------------------------------------------------

def print_gender_distribution(name, data):

    print(f"\n{name} gender distribution:")

    print(
        data["gender"]
        .value_counts()
        .sort_index()
    )


print("\n" + "=" * 75)
print("GENDER DISTRIBUTION BY SPLIT")
print("=" * 75)

print_gender_distribution(
    "TRAIN",
    train
)

print_gender_distribution(
    "VALIDATION",
    validation
)

print_gender_distribution(
    "TEST",
    test
)


# ------------------------------------------------------------
# 14. ACADEMIC YEAR DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("ACADEMIC YEAR DISTRIBUTION BY SPLIT")
print("=" * 75)

for name, data in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test),
]:

    print(f"\n{name}:")

    print(
        data["academic_year"]
        .value_counts()
        .sort_index()
    )


# ------------------------------------------------------------
# 15. VERIFY TARGET CLASSES EXIST
# ------------------------------------------------------------

for name, data in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test),
]:

    unique_targets = set(
        data["at_risk"].unique()
    )

    print(
        f"\n{name} target classes:",
        unique_targets
    )

    assert unique_targets == {0, 1}, (
        f"{name} does not contain both target classes."
    )


# ------------------------------------------------------------
# 16. SAVE SPLITS
# ------------------------------------------------------------

train_file = (
    OUTPUT_DIR
    / "train_student_level.csv"
)

validation_file = (
    OUTPUT_DIR
    / "validation_student_level.csv"
)

test_file = (
    OUTPUT_DIR
    / "test_student_level.csv"
)


train.to_csv(
    train_file,
    index=False
)

validation.to_csv(
    validation_file,
    index=False
)

test.to_csv(
    test_file,
    index=False
)


# ------------------------------------------------------------
# 17. CREATE SPLIT MANIFEST
# ------------------------------------------------------------

manifest = pd.DataFrame(
    {
        "split": (
            ["train"] * len(train)
            + ["validation"] * len(validation)
            + ["test"] * len(test)
        ),
        "student_id": (
            train["student_id"].tolist()
            + validation["student_id"].tolist()
            + test["student_id"].tolist()
        ),
        "academic_year": (
            train["academic_year"].tolist()
            + validation["academic_year"].tolist()
            + test["academic_year"].tolist()
        ),
    }
)

manifest_file = (
    OUTPUT_DIR
    / "student_level_split_manifest.csv"
)

manifest.to_csv(
    manifest_file,
    index=False
)


# ------------------------------------------------------------
# 18. FINAL VALIDATION
# ------------------------------------------------------------

assert train_file.exists()
assert validation_file.exists()
assert test_file.exists()
assert manifest_file.exists()


print("\n" + "=" * 75)
print("STEP 18 COMPLETED SUCCESSFULLY")
print("=" * 75)

print("\nFiles created:")

print(train_file)
print(validation_file)
print(test_file)
print(manifest_file)

print("\nLeakage status:")
print("✓ Student-level grouping enforced")
print("✓ No student overlap between splits")
print("✓ Both target classes present in all splits")
print("✓ All 541 observations accounted for")

print("\nDo NOT proceed to model training yet.")
print("Review the output before Step 19.")