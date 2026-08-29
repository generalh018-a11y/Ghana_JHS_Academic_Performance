import pandas as pd
from sklearn.model_selection import train_test_split

print("=" * 70)
print("STEP 16.11 — TRAIN/TEST SPLIT")
print("=" * 70)

# ---------------------------------------------------------
# 1. LOAD MODELING DATASET
# ---------------------------------------------------------

input_file = r"C:\Users\hills\OneDrive\Documents\MSc_Thesis\Ghana_JHS_Academic_Performance\data\processed\modeling_dataset.csv"

df = pd.read_csv(input_file)

print("\nInput dataset:")
print(f"Shape: {df.shape}")

# ---------------------------------------------------------
# 2. DEFINE TARGET AND PREDICTORS
# ---------------------------------------------------------

target = "overall_average_score"

X = df.drop(columns=[target])
y = df[target]

print("\nTarget:")
print(target)

print("\nPredictors:")
for i, col in enumerate(X.columns, 1):
    print(f"  {i}. {col}")

# ---------------------------------------------------------
# 3. CONVERT CATEGORICAL VARIABLES
# ---------------------------------------------------------

categorical_columns = [
    "class_level",
    "academic_year",
    "gender"
]

X_encoded = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=False,
    dtype=int
)

print("\nEncoded predictor shape:")
print(X_encoded.shape)

print("\nEncoded predictor columns:")
for i, col in enumerate(X_encoded.columns, 1):
    print(f"  {i}. {col}")

# ---------------------------------------------------------
# 4. TRAIN / TEST SPLIT
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.20,
    random_state=42
)

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print(f"\nTraining records: {len(X_train)}")
print(f"Testing records:  {len(X_test)}")

print(f"\nX_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape:  {y_test.shape}")

# ---------------------------------------------------------
# 5. CHECK TARGET DISTRIBUTION
# ---------------------------------------------------------

print("\nTraining target summary:")
print(y_train.describe())

print("\nTesting target summary:")
print(y_test.describe())

# ---------------------------------------------------------
# 6. CHECK MISSING VALUES
# ---------------------------------------------------------

print("\nMissing values:")
print(X_train.isnull().sum().sum())
print(X_test.isnull().sum().sum())

# ---------------------------------------------------------
# 7. SAVE FILES
# ---------------------------------------------------------

output_dir = r"C:\Users\hills\OneDrive\Documents\MSc_Thesis\Ghana_JHS_Academic_Performance\data\processed"

X_train.to_csv(output_dir + r"\X_train.csv", index=False)
X_test.to_csv(output_dir + r"\X_test.csv", index=False)
y_train.to_csv(output_dir + r"\y_train.csv", index=False)
y_test.to_csv(output_dir + r"\y_test.csv", index=False)

# Save complete encoded dataset as well
X_encoded.to_csv(output_dir + r"\X_encoded.csv", index=False)

print("\n" + "=" * 70)
print("STEP 16.11 COMPLETE")
print("=" * 70)

print("\nFiles saved:")
print(output_dir + r"\X_train.csv")
print(output_dir + r"\X_test.csv")
print(output_dir + r"\y_train.csv")
print(output_dir + r"\y_test.csv")
print(output_dir + r"\X_encoded.csv")

print("\n✓ Data successfully split into training and testing sets.")
print("✓ Categorical variables encoded.")
print("✓ Random state = 42.")
print("✓ Test size = 20%.")