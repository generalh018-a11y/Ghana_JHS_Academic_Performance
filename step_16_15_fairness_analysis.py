# ================================================================
# STEP 16.15 — FAIRNESS ANALYSIS
# SHAP-Driven Fairness-Aware XGBoost–LightGBM Ensemble
# ================================================================

import os
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

print("=" * 70)
print("STEP 16.15 — FAIRNESS ANALYSIS")
print("=" * 70)

# ------------------------------------------------
# PATHS
# ------------------------------------------------

BASE_DIR = r"C:\Users\hills\OneDrive\Documents\MSc_Thesis\Ghana_JHS_Academic_Performance"

X_TRAIN_PATH = os.path.join(
    BASE_DIR, "data", "processed", "X_train.csv"
)

X_TEST_PATH = os.path.join(
    BASE_DIR, "data", "processed", "X_test.csv"
)

Y_TRAIN_PATH = os.path.join(
    BASE_DIR, "data", "processed", "y_train.csv"
)

Y_TEST_PATH = os.path.join(
    BASE_DIR, "data", "processed", "y_test.csv"
)

MODEL_DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed", "modeling_dataset.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "data", "processed"
)

# ------------------------------------------------
# LOAD MODELING DATA
# ------------------------------------------------

print("\nLoading modeling data...")

model_data = pd.read_csv(MODEL_DATA_PATH)

print(f"Modeling dataset shape: {model_data.shape}")

# ------------------------------------------------
# LOAD TRAIN / TEST DATA
# ------------------------------------------------

print("\nLoading train/test data...")

X_train = pd.read_csv(X_TRAIN_PATH)
X_test = pd.read_csv(X_TEST_PATH)

y_train = pd.read_csv(Y_TRAIN_PATH).squeeze("columns")
y_test = pd.read_csv(Y_TEST_PATH).squeeze("columns")

print(f"X_train: {X_train.shape}")
print(f"X_test : {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test : {y_test.shape}")

# ------------------------------------------------
# RECONSTRUCT TEST-GROUP INFORMATION
# ------------------------------------------------

# The original modeling dataset contains the demographic
# information. The train/test split used random_state=42
# and test_size=0.20.

from sklearn.model_selection import train_test_split

X_full = model_data[
    [
        "class_level",
        "academic_year",
        "gender",
        "age_years",
        "avg_days_present",
        "avg_total_school_days",
        "avg_attendance_rate"
    ]
]

y_full = model_data["overall_average_score"]

_, X_test_original, _, y_test_original = train_test_split(
    X_full,
    y_full,
    test_size=0.20,
    random_state=42
)

# Preserve the original test-set row order
X_test_original = X_test_original.copy()
y_test_original = y_test_original.copy()

print("\nTest-set demographic information reconstructed.")
print(f"Test records: {len(X_test_original)}")

# ------------------------------------------------
# TRAIN XGBOOST
# ------------------------------------------------

print("\n" + "=" * 70)
print("TRAINING XGBOOST")
print("=" * 70)

xgb_model = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)

xgb_predictions = xgb_model.predict(X_test)

print("✓ XGBoost trained.")

# ------------------------------------------------
# TRAIN LIGHTGBM
# ------------------------------------------------

print("\n" + "=" * 70)
print("TRAINING LIGHTGBM")
print("=" * 70)

lgb_model = LGBMRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="regression",
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

lgb_model.fit(X_train, y_train)

lgb_predictions = lgb_model.predict(X_test)

print("✓ LightGBM trained.")

# ------------------------------------------------
# FINAL 50:50 ENSEMBLE
# ------------------------------------------------

print("\n" + "=" * 70)
print("FINAL 50:50 ENSEMBLE")
print("=" * 70)

ensemble_predictions = (
    0.50 * xgb_predictions +
    0.50 * lgb_predictions
)

print("✓ XGBoost weight : 50%")
print("✓ LightGBM weight : 50%")

# ------------------------------------------------
# OVERALL TEST PERFORMANCE
# ------------------------------------------------

overall_mae = mean_absolute_error(
    y_test,
    ensemble_predictions
)

overall_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        ensemble_predictions
    )
)

overall_r2 = r2_score(
    y_test,
    ensemble_predictions
)

print("\nOverall ensemble performance:")
print(f"MAE  : {overall_mae:.4f}")
print(f"RMSE : {overall_rmse:.4f}")
print(f"R²   : {overall_r2:.4f}")

# ------------------------------------------------
# ALIGN TEST DATA
# ------------------------------------------------

test_results = X_test_original.copy()

test_results["actual_score"] = y_test_original.values
test_results["prediction"] = ensemble_predictions
test_results["error"] = (
    test_results["prediction"] -
    test_results["actual_score"]
)
test_results["absolute_error"] = (
    test_results["error"].abs()
)

# ------------------------------------------------
# FAIRNESS ANALYSIS BY GENDER
# ------------------------------------------------

print("\n" + "=" * 70)
print("FAIRNESS ANALYSIS BY GENDER")
print("=" * 70)

fairness_results = []

for group in sorted(test_results["gender"].unique()):

    group_data = test_results[
        test_results["gender"] == group
    ]

    actual = group_data["actual_score"]
    predicted = group_data["prediction"]

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    # R² requires more than one distinct target value
    if len(group_data) > 1 and actual.nunique() > 1:
        r2 = r2_score(
            actual,
            predicted
        )
    else:
        r2 = np.nan

    mean_actual = actual.mean()
    mean_prediction = predicted.mean()
    mean_error = group_data["error"].mean()
    mean_absolute_error_value = group_data[
        "absolute_error"
    ].mean()

    fairness_results.append({
        "gender": group,
        "n": len(group_data),
        "mean_actual_score": mean_actual,
        "mean_predicted_score": mean_prediction,
        "mean_error": mean_error,
        "mean_absolute_error": mean_absolute_error_value,
        "rmse": rmse,
        "r2": r2
    })

fairness_df = pd.DataFrame(
    fairness_results
)

print(
    fairness_df.to_string(index=False)
)

# ------------------------------------------------
# FAIRNESS GAP CALCULATIONS
# ------------------------------------------------

print("\n" + "=" * 70)
print("FAIRNESS GAP ANALYSIS")
print("=" * 70)

if len(fairness_df) == 2:

    groups = fairness_df["gender"].tolist()

    g1 = fairness_df[
        fairness_df["gender"] == groups[0]
    ].iloc[0]

    g2 = fairness_df[
        fairness_df["gender"] == groups[1]
    ].iloc[0]

    mae_gap = abs(
        g1["mean_absolute_error"] -
        g2["mean_absolute_error"]
    )

    rmse_gap = abs(
        g1["rmse"] -
        g2["rmse"]
    )

    r2_gap = abs(
        g1["r2"] -
        g2["r2"]
    )

    mean_error_gap = abs(
        g1["mean_error"] -
        g2["mean_error"]
    )

    mean_prediction_gap = abs(
        g1["mean_predicted_score"] -
        g2["mean_predicted_score"]
    )

    mean_actual_gap = abs(
        g1["mean_actual_score"] -
        g2["mean_actual_score"]
    )

    print(
        f"Groups compared: {groups[0]} vs {groups[1]}"
    )

    print(f"MAE gap               : {mae_gap:.4f}")
    print(f"RMSE gap              : {rmse_gap:.4f}")
    print(f"R² gap                : {r2_gap:.4f}")
    print(f"Mean error gap        : {mean_error_gap:.4f}")
    print(f"Mean prediction gap   : {mean_prediction_gap:.4f}")
    print(f"Mean actual-score gap : {mean_actual_gap:.4f}")

    gap_results = pd.DataFrame([{
        "group_1": groups[0],
        "group_2": groups[1],
        "mae_gap": mae_gap,
        "rmse_gap": rmse_gap,
        "r2_gap": r2_gap,
        "mean_error_gap": mean_error_gap,
        "mean_prediction_gap": mean_prediction_gap,
        "mean_actual_score_gap": mean_actual_gap
    }])

else:

    print(
        "More than two gender groups detected."
    )

    gap_results = pd.DataFrame()

# ------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------

fairness_path = os.path.join(
    OUTPUT_DIR,
    "fairness_gender_results.csv"
)

fairness_df.to_csv(
    fairness_path,
    index=False
)

print(
    f"\n✓ Saved: {fairness_path}"
)

if not gap_results.empty:

    gap_path = os.path.join(
        OUTPUT_DIR,
        "fairness_gender_gaps.csv"
    )

    gap_results.to_csv(
        gap_path,
        index=False
    )

    print(
        f"✓ Saved: {gap_path}"
    )

# ------------------------------------------------
# SAVE TEST PREDICTIONS WITH GROUPS
# ------------------------------------------------

predictions_path = os.path.join(
    OUTPUT_DIR,
    "fairness_test_predictions.csv"
)

test_results.to_csv(
    predictions_path,
    index=False
)

print(
    f"✓ Saved: {predictions_path}"
)

# ------------------------------------------------
# SUMMARY
# ------------------------------------------------

print("\n" + "=" * 70)
print("STEP 16.15 COMPLETE")
print("=" * 70)

print("\n✓ Final 50:50 ensemble evaluated.")
print("✓ Test-set predictions generated.")
print("✓ Gender-specific MAE calculated.")
print("✓ Gender-specific RMSE calculated.")
print("✓ Gender-specific R² calculated.")
print("✓ Mean prediction error calculated.")
print("✓ Fairness gaps calculated.")
print("✓ Fairness results saved.")

print("\nNext stage:")
print("STEP 16.16 — FINAL RESULTS AND THESIS TABLES")
print("=" * 70)