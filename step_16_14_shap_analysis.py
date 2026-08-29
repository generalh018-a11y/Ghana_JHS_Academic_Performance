# ================================================================
# STEP 16.14 — SHAP EXPLAINABILITY ANALYSIS
# Ghana JHS Academic Performance Thesis
# ================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from xgboost import XGBRegressor

print("=" * 70)
print("STEP 16.14 — SHAP EXPLAINABILITY ANALYSIS")
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

OUTPUT_DIR = os.path.join(
    BASE_DIR, "data", "processed"
)

FIGURE_DIR = os.path.join(
    BASE_DIR, "results", "figures"
)

os.makedirs(FIGURE_DIR, exist_ok=True)

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

print("\nLoading training and testing data...")

X_train = pd.read_csv(X_TRAIN_PATH)
X_test = pd.read_csv(X_TEST_PATH)

y_train = pd.read_csv(Y_TRAIN_PATH).squeeze("columns")
y_test = pd.read_csv(Y_TEST_PATH).squeeze("columns")

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape : {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape : {y_test.shape}")

# ------------------------------------------------
# TRAIN XGBOOST MODEL
# ------------------------------------------------

print("\n" + "=" * 70)
print("TRAINING XGBOOST FOR SHAP ANALYSIS")
print("=" * 70)

model = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("✓ XGBoost model trained successfully.")

# ------------------------------------------------
# SHAP EXPLAINER
# ------------------------------------------------

print("\n" + "=" * 70)
print("CALCULATING SHAP VALUES")
print("=" * 70)

# Use a model-agnostic SHAP explainer.
# This avoids compatibility problems between
# SHAP and newer XGBoost model formats.

background_data = X_train.sample(
    n=min(100, len(X_train)),
    random_state=42
)

prediction_function = lambda data: model.predict(
    pd.DataFrame(data, columns=X_train.columns)
)

explainer = shap.Explainer(
    prediction_function,
    background_data
)

shap_explanation = explainer(
    X_test,
    max_evals=5000
)

shap_values = shap_explanation.values

print("✓ Model-agnostic SHAP values calculated successfully.")
print(f"SHAP matrix shape: {shap_values.shape}")

print("✓ SHAP values calculated successfully.")
print(f"SHAP matrix shape: {np.array(shap_values).shape}")

# ------------------------------------------------
# FEATURE NAMES
# ------------------------------------------------

feature_names = X_test.columns.tolist()

print("\nFeatures analyzed:")

for i, feature in enumerate(feature_names, start=1):
    print(f"  {i}. {feature}")

# ------------------------------------------------
# GLOBAL SHAP IMPORTANCE
# ------------------------------------------------

mean_abs_shap = np.abs(shap_values).mean(axis=0)

shap_importance = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": mean_abs_shap
})

shap_importance = shap_importance.sort_values(
    "mean_abs_shap",
    ascending=False
).reset_index(drop=True)

shap_importance["rank"] = np.arange(
    1, len(shap_importance) + 1
)

print("\n" + "=" * 70)
print("GLOBAL SHAP FEATURE IMPORTANCE")
print("=" * 70)

print(
    shap_importance[
        ["rank", "feature", "mean_abs_shap"]
    ].to_string(index=False)
)

# Save importance table

importance_path = os.path.join(
    OUTPUT_DIR,
    "shap_feature_importance.csv"
)

shap_importance.to_csv(
    importance_path,
    index=False
)

print(f"\n✓ Saved: {importance_path}")

# ------------------------------------------------
# SHAP VALUES DATASET
# ------------------------------------------------

shap_values_df = pd.DataFrame(
    shap_values,
    columns=feature_names
)

shap_values_path = os.path.join(
    OUTPUT_DIR,
    "shap_values_test.csv"
)

shap_values_df.to_csv(
    shap_values_path,
    index=False
)

print(f"✓ Saved: {shap_values_path}")

# ------------------------------------------------
# SHAP SUMMARY PLOT
# ------------------------------------------------

print("\nGenerating SHAP summary plot...")

plt.figure()

shap.summary_plot(
    shap_explanation,
    X_test,
    show=False
)


plt.tight_layout()

summary_path = os.path.join(
    FIGURE_DIR,
    "shap_summary_plot.png"
)

plt.savefig(
    summary_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"✓ Saved: {summary_path}")

# ------------------------------------------------
# SHAP BAR PLOT
# ------------------------------------------------

print("\nGenerating SHAP importance bar plot...")

plt.figure()

shap.summary_plot(
    shap_explanation,
    X_test,
    plot_type="bar",
    show=False
)


plt.tight_layout()

bar_path = os.path.join(
    FIGURE_DIR,
    "shap_feature_importance_bar.png"
)

plt.savefig(
    bar_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"✓ Saved: {bar_path}")

# ------------------------------------------------
# FEATURE EFFECT DIRECTION
# ------------------------------------------------

print("\n" + "=" * 70)
print("SHAP EFFECT DIRECTION")
print("=" * 70)

direction_results = []

for feature in feature_names:

    values = shap_values_df[feature]

    positive_count = (values > 0).sum()
    negative_count = (values < 0).sum()
    zero_count = (values == 0).sum()

    mean_shap = values.mean()

    if mean_shap > 0:
        overall_direction = "Positive"
    elif mean_shap < 0:
        overall_direction = "Negative"
    else:
        overall_direction = "Neutral"

    direction_results.append({
        "feature": feature,
        "mean_shap": mean_shap,
        "mean_abs_shap": np.abs(values).mean(),
        "positive_effects": positive_count,
        "negative_effects": negative_count,
        "zero_effects": zero_count,
        "overall_direction": overall_direction
    })

direction_df = pd.DataFrame(direction_results)

direction_df = direction_df.sort_values(
    "mean_abs_shap",
    ascending=False
).reset_index(drop=True)

direction_path = os.path.join(
    OUTPUT_DIR,
    "shap_effect_direction.csv"
)

direction_df.to_csv(
    direction_path,
    index=False
)

print(
    direction_df.to_string(index=False)
)

print(f"\n✓ Saved: {direction_path}")

# ------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------

print("\n" + "=" * 70)
print("STEP 16.14 COMPLETE")
print("=" * 70)

print("\nSHAP analysis completed successfully.")

print("\nGenerated files:")

print(f"1. {importance_path}")
print(f"2. {shap_values_path}")
print(f"3. {direction_path}")
print(f"4. {summary_path}")
print(f"5. {bar_path}")

print("\n✓ Global feature importance calculated.")
print("✓ SHAP values calculated.")
print("✓ SHAP summary plot generated.")
print("✓ SHAP bar plot generated.")
print("✓ SHAP effect direction calculated.")

print("\nNext stage: STEP 16.15 — FAIRNESS ANALYSIS")
print("=" * 70)