import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("=" * 70)
print("STEP 16.13 — XGBOOST–LIGHTGBM ENSEMBLE")
print("=" * 70)

# ---------------------------------------------------------
# 1. LOAD BASELINE PREDICTIONS
# ---------------------------------------------------------

input_dir = r"C:\Users\hills\OneDrive\Documents\MSc_Thesis\Ghana_JHS_Academic_Performance\data\processed"

predictions_file = input_dir + r"\baseline_predictions.csv"

df = pd.read_csv(predictions_file)

print("\nBaseline predictions loaded:")
print(f"Shape: {df.shape}")

print("\nColumns:")
print(list(df.columns))

# ---------------------------------------------------------
# 2. EXTRACT ACTUAL AND MODEL PREDICTIONS
# ---------------------------------------------------------

y_actual = df["actual_score"]

xgb_pred = df["xgboost_prediction"]

lgb_pred = df["lightgbm_prediction"]

# ---------------------------------------------------------
# 3. DEFINE ENSEMBLE WEIGHTS
# ---------------------------------------------------------

weights = [
    (0.50, 0.50),
    (0.60, 0.40),
    (0.70, 0.30),
    (0.80, 0.20),
    (0.40, 0.60),
    (0.30, 0.70),
    (0.20, 0.80)
]

results = []

# ---------------------------------------------------------
# 4. EVALUATE EACH ENSEMBLE
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("ENSEMBLE WEIGHT EVALUATION")
print("=" * 70)

for xgb_weight, lgb_weight in weights:

    ensemble_pred = (
        xgb_weight * xgb_pred
        + lgb_weight * lgb_pred
    )

    mae = mean_absolute_error(
        y_actual,
        ensemble_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_actual,
            ensemble_pred
        )
    )

    r2 = r2_score(
        y_actual,
        ensemble_pred
    )

    results.append({
        "XGBoost_weight": xgb_weight,
        "LightGBM_weight": lgb_weight,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    print(
        f"\nXGBoost = {xgb_weight:.0%} | "
        f"LightGBM = {lgb_weight:.0%}"
    )

    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

# ---------------------------------------------------------
# 5. CREATE RESULTS TABLE
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

# Sort primarily by RMSE
results_df = results_df.sort_values(
    by="RMSE",
    ascending=True
).reset_index(drop=True)

print("\n" + "=" * 70)
print("ENSEMBLE PERFORMANCE RANKING")
print("=" * 70)

print(results_df.to_string(index=False))

# ---------------------------------------------------------
# 6. IDENTIFY BEST ENSEMBLE
# ---------------------------------------------------------

best = results_df.iloc[0]

best_xgb_weight = best["XGBoost_weight"]
best_lgb_weight = best["LightGBM_weight"]

best_mae = best["MAE"]
best_rmse = best["RMSE"]
best_r2 = best["R2"]

print("\n" + "=" * 70)
print("BEST ENSEMBLE")
print("=" * 70)

print(
    f"\nBest weighting:"
    f"\nXGBoost  = {best_xgb_weight:.0%}"
    f"\nLightGBM = {best_lgb_weight:.0%}"
)

print(f"\nBest MAE  : {best_mae:.4f}")
print(f"Best RMSE : {best_rmse:.4f}")
print(f"Best R²   : {best_r2:.4f}")

# ---------------------------------------------------------
# 7. COMPARE BEST ENSEMBLE WITH BASELINE MODELS
# ---------------------------------------------------------

xgb_mae = mean_absolute_error(y_actual, xgb_pred)
xgb_rmse = np.sqrt(mean_squared_error(y_actual, xgb_pred))
xgb_r2 = r2_score(y_actual, xgb_pred)

lgb_mae = mean_absolute_error(y_actual, lgb_pred)
lgb_rmse = np.sqrt(mean_squared_error(y_actual, lgb_pred))
lgb_r2 = r2_score(y_actual, lgb_pred)

comparison = pd.DataFrame({
    "Model": [
        "XGBoost",
        "LightGBM",
        "XGBoost-LightGBM Ensemble"
    ],
    "MAE": [
        xgb_mae,
        lgb_mae,
        best_mae
    ],
    "RMSE": [
        xgb_rmse,
        lgb_rmse,
        best_rmse
    ],
    "R2": [
        xgb_r2,
        lgb_r2,
        best_r2
    ]
})

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(comparison.to_string(index=False))

# ---------------------------------------------------------
# 8. GENERATE BEST ENSEMBLE PREDICTIONS
# ---------------------------------------------------------

best_ensemble_prediction = (
    best_xgb_weight * xgb_pred
    + best_lgb_weight * lgb_pred
)

final_predictions = pd.DataFrame({
    "actual_score": y_actual,
    "xgboost_prediction": xgb_pred,
    "lightgbm_prediction": lgb_pred,
    "ensemble_prediction": best_ensemble_prediction
})

# ---------------------------------------------------------
# 9. SAVE RESULTS
# ---------------------------------------------------------

results_df.to_csv(
    input_dir + r"\ensemble_weight_results.csv",
    index=False
)

comparison.to_csv(
    input_dir + r"\final_model_comparison.csv",
    index=False
)

final_predictions.to_csv(
    input_dir + r"\ensemble_predictions.csv",
    index=False
)

print("\n" + "=" * 70)
print("STEP 16.13 COMPLETE")
print("=" * 70)

print("\nFiles saved:")

print(
    input_dir
    + r"\ensemble_weight_results.csv"
)

print(
    input_dir
    + r"\final_model_comparison.csv"
)

print(
    input_dir
    + r"\ensemble_predictions.csv"
)

print("\n✓ Multiple ensemble weights evaluated.")
print("✓ Best ensemble identified.")
print("✓ Ensemble compared with XGBoost and LightGBM.")
print("✓ Final ensemble predictions saved.")