# ================================================================
# STEP 16.16 — FINAL RESULTS AND THESIS TABLES
# SHAP-Driven Fairness-Aware XGBoost–LightGBM Ensemble
# ================================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------
# 1. PROJECT PATHS
# ----------------------------------------------------------------

BASE_DIR = r"C:\Users\hills\OneDrive\Documents\MSc_Thesis\Ghana_JHS_Academic_Performance"

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 70)
print("STEP 16.16 — FINAL RESULTS AND THESIS TABLES")
print("=" * 70)


# ----------------------------------------------------------------
# 2. HELPER FUNCTION
# ----------------------------------------------------------------

def load_csv(filename):
    path = os.path.join(PROCESSED_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )

    return pd.read_csv(path)


# ----------------------------------------------------------------
# 3. LOAD COMPLETED RESULTS
# ----------------------------------------------------------------

print("\nLoading completed analysis results...")

model_results = load_csv("final_model_comparison.csv")
ensemble_weights = load_csv("ensemble_weight_results.csv")
shap_importance = load_csv("shap_feature_importance.csv")
shap_direction = load_csv("shap_effect_direction.csv")
gender_results = load_csv("fairness_gender_results.csv")
gender_gaps = load_csv("fairness_gender_gaps.csv")

print("✓ Model comparison loaded.")
print("✓ Ensemble weight results loaded.")
print("✓ SHAP importance loaded.")
print("✓ SHAP effect direction loaded.")
print("✓ Gender fairness results loaded.")
print("✓ Gender fairness gaps loaded.")


# ----------------------------------------------------------------
# 4. MODEL PERFORMANCE TABLE
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("TABLE 1 — MODEL PERFORMANCE COMPARISON")
print("=" * 70)

table_model = model_results.copy()

# Standardize column names where necessary
table_model.columns = [
    str(col).strip() for col in table_model.columns
]

# Round numerical values
for col in ["MAE", "RMSE", "R2"]:
    if col in table_model.columns:
        table_model[col] = table_model[col].round(4)

print(table_model.to_string(index=False))

table_model_path = os.path.join(
    PROCESSED_DIR,
    "table_model_comparison.csv"
)

table_model.to_csv(table_model_path, index=False)

print(
    f"\n✓ Saved: {table_model_path}"
)


# ----------------------------------------------------------------
# 5. ENSEMBLE WEIGHT TABLE
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("TABLE 2 — ENSEMBLE WEIGHT COMPARISON")
print("=" * 70)

table_weights = ensemble_weights.copy()

table_weights.columns = [
    str(col).strip() for col in table_weights.columns
]

for col in ["XGBoost_weight", "LightGBM_weight"]:
    if col in table_weights.columns:
        table_weights[col] = table_weights[col].round(2)

for col in ["MAE", "RMSE", "R2"]:
    if col in table_weights.columns:
        table_weights[col] = table_weights[col].round(4)

print(table_weights.to_string(index=False))

table_weights_path = os.path.join(
    PROCESSED_DIR,
    "table_ensemble_weights.csv"
)

table_weights.to_csv(table_weights_path, index=False)

print(
    f"\n✓ Saved: {table_weights_path}"
)


# ----------------------------------------------------------------
# 6. SHAP FEATURE IMPORTANCE TABLE
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("TABLE 3 — GLOBAL SHAP FEATURE IMPORTANCE")
print("=" * 70)

table_shap = shap_importance.copy()

table_shap.columns = [
    str(col).strip() for col in table_shap.columns
]

if "mean_abs_shap" in table_shap.columns:
    table_shap["mean_abs_shap"] = (
        table_shap["mean_abs_shap"].round(4)
    )

print(table_shap.to_string(index=False))

table_shap_path = os.path.join(
    PROCESSED_DIR,
    "table_shap_importance.csv"
)

table_shap.to_csv(table_shap_path, index=False)

print(
    f"\n✓ Saved: {table_shap_path}"
)


# ----------------------------------------------------------------
# 7. SHAP EFFECT DIRECTION TABLE
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("TABLE 4 — SHAP EFFECT DIRECTION")
print("=" * 70)

table_direction = shap_direction.copy()

table_direction.columns = [
    str(col).strip() for col in table_direction.columns
]

for col in [
    "mean_shap",
    "mean_abs_shap"
]:
    if col in table_direction.columns:
        table_direction[col] = (
            table_direction[col].round(4)
        )

print(table_direction.to_string(index=False))

table_direction_path = os.path.join(
    PROCESSED_DIR,
    "table_shap_direction.csv"
)

table_direction.to_csv(
    table_direction_path,
    index=False
)

print(
    f"\n✓ Saved: {table_direction_path}"
)


# ----------------------------------------------------------------
# 8. GENDER FAIRNESS TABLE
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("TABLE 5 — FAIRNESS ANALYSIS BY GENDER")
print("=" * 70)

table_gender = gender_results.copy()

table_gender.columns = [
    str(col).strip() for col in table_gender.columns
]

for col in [
    "mean_actual_score",
    "mean_predicted_score",
    "mean_error",
    "mean_absolute_error",
    "rmse",
    "r2"
]:
    if col in table_gender.columns:
        table_gender[col] = (
            table_gender[col].round(4)
        )

print(table_gender.to_string(index=False))

table_gender_path = os.path.join(
    PROCESSED_DIR,
    "table_gender_fairness.csv"
)

table_gender.to_csv(
    table_gender_path,
    index=False
)

print(
    f"\n✓ Saved: {table_gender_path}"
)


# ----------------------------------------------------------------
# 9. GENDER FAIRNESS GAPS
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("TABLE 6 — GENDER FAIRNESS GAPS")
print("=" * 70)

table_gaps = gender_gaps.copy()

table_gaps.columns = [
    str(col).strip() for col in table_gaps.columns
]

for col in table_gaps.columns:
    if pd.api.types.is_numeric_dtype(table_gaps[col]):
        table_gaps[col] = table_gaps[col].round(4)

print(table_gaps.to_string(index=False))

table_gaps_path = os.path.join(
    PROCESSED_DIR,
    "table_gender_fairness_gaps.csv"
)

table_gaps.to_csv(
    table_gaps_path,
    index=False
)

print(
    f"\n✓ Saved: {table_gaps_path}"
)


# ----------------------------------------------------------------
# 10. CONSOLIDATED FINAL RESULTS SUMMARY
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("CREATING CONSOLIDATED RESULTS SUMMARY")
print("=" * 70)

# Find best model according to lowest MAE
best_mae_row = model_results.loc[
    model_results["MAE"].idxmin()
]

# Find best model according to highest R2
best_r2_row = model_results.loc[
    model_results["R2"].idxmax()
]

# Find best ensemble weight according to lowest MAE
best_weight_row = ensemble_weights.loc[
    ensemble_weights["MAE"].idxmin()
]

summary = pd.DataFrame({
    "Metric": [
        "Best model by MAE",
        "Best MAE",
        "Best model by R2",
        "Best R2",
        "Best ensemble XGBoost weight",
        "Best ensemble LightGBM weight"
    ],
    "Value": [
        best_mae_row["Model"],
        round(best_mae_row["MAE"], 4),
        best_r2_row["Model"],
        round(best_r2_row["R2"], 4),
        round(best_weight_row["XGBoost_weight"], 2),
        round(best_weight_row["LightGBM_weight"], 2)
    ]
})

print(summary.to_string(index=False))

summary_path = os.path.join(
    PROCESSED_DIR,
    "final_results_summary.csv"
)

summary.to_csv(summary_path, index=False)

print(
    f"\n✓ Saved: {summary_path}"
)


# ----------------------------------------------------------------
# 11. FIGURE 1 — MODEL PERFORMANCE
# ----------------------------------------------------------------

print("\nGenerating model performance figure...")

models = model_results["Model"].astype(str)
mae_values = model_results["MAE"]
rmse_values = model_results["RMSE"]

plt.figure(figsize=(9, 6))

x = range(len(models))

plt.bar(
    [i - 0.18 for i in x],
    mae_values,
    width=0.36,
    label="MAE"
)

plt.bar(
    [i + 0.18 for i in x],
    rmse_values,
    width=0.36,
    label="RMSE"
)

plt.xticks(
    list(x),
    models,
    rotation=20,
    ha="right"
)

plt.ylabel("Error")
plt.title("Comparison of Model Prediction Errors")
plt.legend()
plt.tight_layout()

model_fig_path = os.path.join(
    FIGURES_DIR,
    "model_performance_comparison.png"
)

plt.savefig(
    model_fig_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"✓ Saved: {model_fig_path}"
)


# ----------------------------------------------------------------
# 12. FIGURE 2 — R² COMPARISON
# ----------------------------------------------------------------

print("\nGenerating R² comparison figure...")

plt.figure(figsize=(9, 6))

plt.bar(
    models,
    model_results["R2"]
)

plt.ylabel("R²")
plt.xlabel("Model")
plt.title("Comparison of Model R² Performance")

plt.xticks(
    rotation=20,
    ha="right"
)

plt.tight_layout()

r2_fig_path = os.path.join(
    FIGURES_DIR,
    "model_r2_comparison.png"
)

plt.savefig(
    r2_fig_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"✓ Saved: {r2_fig_path}"
)


# ----------------------------------------------------------------
# 13. FIGURE 3 — ENSEMBLE WEIGHTS
# ----------------------------------------------------------------

print("\nGenerating ensemble-weight figure...")

plt.figure(figsize=(9, 6))

weight_labels = (
    ensemble_weights["XGBoost_weight"].astype(str)
    + " / "
    + ensemble_weights["LightGBM_weight"].astype(str)
)

plt.plot(
    weight_labels,
    ensemble_weights["MAE"],
    marker="o",
    label="MAE"
)

plt.plot(
    weight_labels,
    ensemble_weights["RMSE"],
    marker="o",
    label="RMSE"
)

plt.xlabel("XGBoost / LightGBM Weight")
plt.ylabel("Error")
plt.title("Ensemble Performance Across Weight Combinations")
plt.legend()

plt.xticks(rotation=30)

plt.tight_layout()

weight_fig_path = os.path.join(
    FIGURES_DIR,
    "ensemble_weight_comparison.png"
)

plt.savefig(
    weight_fig_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"✓ Saved: {weight_fig_path}"
)


# ----------------------------------------------------------------
# 14. FIGURE 4 — SHAP FEATURE IMPORTANCE
# ----------------------------------------------------------------

print("\nGenerating SHAP feature importance figure...")

shap_plot = shap_importance.sort_values(
    "mean_abs_shap",
    ascending=True
)

plt.figure(figsize=(9, 7))

plt.barh(
    shap_plot["feature"],
    shap_plot["mean_abs_shap"]
)

plt.xlabel("Mean Absolute SHAP Value")
plt.ylabel("Feature")
plt.title("Global SHAP Feature Importance")

plt.tight_layout()

shap_fig_path = os.path.join(
    FIGURES_DIR,
    "final_shap_feature_importance.png"
)

plt.savefig(
    shap_fig_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"✓ Saved: {shap_fig_path}"
)


# ----------------------------------------------------------------
# 15. FIGURE 5 — GENDER FAIRNESS MAE
# ----------------------------------------------------------------

print("\nGenerating gender fairness MAE figure...")

plt.figure(figsize=(7, 6))

plt.bar(
    gender_results["gender"].astype(str),
    gender_results["mean_absolute_error"]
)

plt.xlabel("Gender")
plt.ylabel("Mean Absolute Error")
plt.title("Prediction Error by Gender")

plt.tight_layout()

gender_mae_fig_path = os.path.join(
    FIGURES_DIR,
    "gender_fairness_mae.png"
)

plt.savefig(
    gender_mae_fig_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"✓ Saved: {gender_mae_fig_path}"
)


# ----------------------------------------------------------------
# 16. FINAL CONSOLE SUMMARY
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 16.16 COMPLETE")
print("=" * 70)

print("\nFINAL MODEL:")
print(
    f"Best model by MAE: {best_mae_row['Model']}"
)
print(
    f"MAE : {best_mae_row['MAE']:.4f}"
)
print(
    f"RMSE: {best_mae_row['RMSE']:.4f}"
)
print(
    f"R²  : {best_mae_row['R2']:.4f}"
)

print("\nBEST ENSEMBLE:")
print(
    f"XGBoost weight : "
    f"{best_weight_row['XGBoost_weight']:.0%}"
)
print(
    f"LightGBM weight: "
    f"{best_weight_row['LightGBM_weight']:.0%}"
)

print(
    f"MAE : {best_weight_row['MAE']:.4f}"
)
print(
    f"RMSE: {best_weight_row['RMSE']:.4f}"
)
print(
    f"R²  : {best_weight_row['R2']:.4f}"
)

print("\nOUTPUT TABLES:")
print("✓ table_model_comparison.csv")
print("✓ table_ensemble_weights.csv")
print("✓ table_shap_importance.csv")
print("✓ table_shap_direction.csv")
print("✓ table_gender_fairness.csv")
print("✓ table_gender_fairness_gaps.csv")
print("✓ final_results_summary.csv")

print("\nOUTPUT FIGURES:")
print("✓ model_performance_comparison.png")
print("✓ model_r2_comparison.png")
print("✓ ensemble_weight_comparison.png")
print("✓ final_shap_feature_importance.png")
print("✓ gender_fairness_mae.png")

print("\n" + "=" * 70)
print("ALL FINAL RESULTS HAVE BEEN CONSOLIDATED.")
print("=" * 70)