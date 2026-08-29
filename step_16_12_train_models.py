import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("=" * 70)
print("STEP 16.12 — TRAINING XGBOOST AND LIGHTGBM")
print("=" * 70)

# ---------------------------------------------------------
# 1. FILE PATHS
# ---------------------------------------------------------

input_dir = r"C:\Users\hills\OneDrive\Documents\MSc_Thesis\Ghana_JHS_Academic_Performance\data\processed"

# ---------------------------------------------------------
# 2. LOAD TRAINING AND TESTING DATA
# ---------------------------------------------------------

X_train = pd.read_csv(input_dir + r"\X_train.csv")
X_test = pd.read_csv(input_dir + r"\X_test.csv")

y_train = pd.read_csv(input_dir + r"\y_train.csv").squeeze("columns")
y_test = pd.read_csv(input_dir + r"\y_test.csv").squeeze("columns")

print("\nData loaded successfully.")

print(f"X_train: {X_train.shape}")
print(f"X_test:  {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test:  {y_test.shape}")

# ---------------------------------------------------------
# 3. TRAIN XGBOOST
# ---------------------------------------------------------

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

xgb_mae = mean_absolute_error(y_test, xgb_predictions)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))
xgb_r2 = r2_score(y_test, xgb_predictions)

print("\nXGBoost results:")
print(f"MAE  : {xgb_mae:.4f}")
print(f"RMSE : {xgb_rmse:.4f}")
print(f"R²   : {xgb_r2:.4f}")

# ---------------------------------------------------------
# 4. TRAIN LIGHTGBM
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TRAINING LIGHTGBM")
print("=" * 70)

lgb_model = LGBMRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    num_leaves=15,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

lgb_model.fit(X_train, y_train)

lgb_predictions = lgb_model.predict(X_test)

lgb_mae = mean_absolute_error(y_test, lgb_predictions)
lgb_rmse = np.sqrt(mean_squared_error(y_test, lgb_predictions))
lgb_r2 = r2_score(y_test, lgb_predictions)

print("\nLightGBM results:")
print(f"MAE  : {lgb_mae:.4f}")
print(f"RMSE : {lgb_rmse:.4f}")
print(f"R²   : {lgb_r2:.4f}")

# ---------------------------------------------------------
# 5. MODEL COMPARISON
# ---------------------------------------------------------

results = pd.DataFrame({
    "Model": [
        "XGBoost",
        "LightGBM"
    ],
    "MAE": [
        xgb_mae,
        lgb_mae
    ],
    "RMSE": [
        xgb_rmse,
        lgb_rmse
    ],
    "R2": [
        xgb_r2,
        lgb_r2
    ]
})

print("\n" + "=" * 70)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 70)

print(results.to_string(index=False))

# ---------------------------------------------------------
# 6. SAVE RESULTS
# ---------------------------------------------------------

results.to_csv(
    input_dir + r"\baseline_model_results.csv",
    index=False
)

# Save predictions for later analysis
predictions = pd.DataFrame({
    "actual_score": y_test.values,
    "xgboost_prediction": xgb_predictions,
    "lightgbm_prediction": lgb_predictions
})

predictions.to_csv(
    input_dir + r"\baseline_predictions.csv",
    index=False
)

print("\n" + "=" * 70)
print("STEP 16.12 COMPLETE")
print("=" * 70)

print("\nSaved files:")
print(input_dir + r"\baseline_model_results.csv")
print(input_dir + r"\baseline_predictions.csv")

print("\n✓ XGBoost trained successfully.")
print("✓ LightGBM trained successfully.")
print("✓ Test-set predictions generated.")
print("✓ MAE, RMSE and R² calculated.")
print("✓ Results saved for thesis analysis.")