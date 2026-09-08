"""
Step 28B: SHAP visualization and thesis-ready figures.

Uses the SHAP values already generated in Step 28A.
No model retraining.
No model selection.
No threshold tuning.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
SHAP_DIR = PROJECT_ROOT / "results" / "shap"

SHAP_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

X_test = pd.read_csv(DATA_DIR / "X_test.csv")

shap_values_df = pd.read_csv(
    SHAP_DIR / "shap_values_test_ensemble.csv"
)

test_with_shap = pd.read_csv(
    SHAP_DIR / "test_predictions_with_shap.csv"
)


print("=" * 70)
print("STEP 28B — SHAP VISUALIZATIONS")
print("=" * 70)


# ============================================================
# 3. VERIFY INPUTS
# ============================================================

feature_columns = list(X_test.columns)

shap_feature_columns = [
    col for col in shap_values_df.columns
    if col != "test_row"
]

if feature_columns != shap_feature_columns:
    raise ValueError(
        "X_test features and SHAP features do not match."
    )


shap_values = shap_values_df[
    feature_columns
].to_numpy()

print("\nX_test shape:", X_test.shape)
print("SHAP shape:", shap_values.shape)

if shap_values.shape != X_test.shape:
    raise ValueError(
        "SHAP matrix shape does not match X_test."
    )


# ============================================================
# 4. GLOBAL SHAP IMPORTANCE
# ============================================================

global_importance = pd.DataFrame({
    "feature": feature_columns,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0)
})

global_importance = global_importance.sort_values(
    "mean_abs_shap",
    ascending=True
)


# ============================================================
# 5. GLOBAL SHAP BAR PLOT
# ============================================================

plt.figure(figsize=(9, 6))

plt.barh(
    global_importance["feature"],
    global_importance["mean_abs_shap"]
)

plt.xlabel("Mean Absolute SHAP Value")
plt.ylabel("Feature")
plt.title(
    "Global SHAP Feature Importance — Locked Ensemble"
)

plt.tight_layout()

bar_path = SHAP_DIR / "Figure_SHAP_Global_Bar.png"

plt.savefig(
    bar_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSaved:")
print(bar_path)


# ============================================================
# 6. SHAP BEESWARM PLOT
# ============================================================

# Build a SHAP Explanation object using the already-computed
# SHAP values. No new model explanation is performed.

explanation = shap.Explanation(
    values=shap_values,
    base_values=np.zeros(len(X_test)),
    data=X_test.to_numpy(),
    feature_names=feature_columns
)

plt.figure()

shap.plots.beeswarm(
    explanation,
    max_display=len(feature_columns),
    show=False
)

plt.title(
    "SHAP Summary Plot — Locked XGBoost–LightGBM Ensemble"
)

plt.tight_layout()

beeswarm_path = SHAP_DIR / "Figure_SHAP_Beeswarm.png"

plt.savefig(
    beeswarm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(beeswarm_path)


# ============================================================
# 7. DEPENDENCE PLOT FUNCTION
# ============================================================

def create_dependence_plot(feature_name):

    feature_index = feature_columns.index(feature_name)

    feature_values = X_test[feature_name].to_numpy()

    feature_shap_values = shap_values[:, feature_index]

    plt.figure(figsize=(8, 6))

    plt.scatter(
        feature_values,
        feature_shap_values,
        alpha=0.75
    )

    plt.axhline(
        y=0,
        linestyle="--"
    )

    plt.xlabel(feature_name)
    plt.ylabel("SHAP Value")
    plt.title(
        f"SHAP Dependence Plot — {feature_name}"
    )

    plt.tight_layout()

    safe_name = feature_name.replace(
        "/", "_"
    ).replace(
        "-", "_"
    )

    output_path = (
        SHAP_DIR /
        f"Figure_SHAP_Dependence_{safe_name}.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(output_path)


# ============================================================
# 8. KEY DEPENDENCE PLOTS
# ============================================================

print("\nGenerating dependence plots...")

create_dependence_plot("score_t1")

create_dependence_plot("score_t2")

create_dependence_plot("attendance_decay")


# ============================================================
# 9. EXPORT PLOT DATA
# ============================================================

plot_data = X_test.copy()

for feature in feature_columns:
    plot_data[
        f"shap_{feature}"
    ] = shap_values_df[feature]

if "actual_at_risk" in test_with_shap.columns:
    plot_data["actual_at_risk"] = (
        test_with_shap["actual_at_risk"]
    )

if "ensemble_probability" in test_with_shap.columns:
    plot_data["ensemble_probability"] = (
        test_with_shap["ensemble_probability"]
    )

if "ensemble_prediction" in test_with_shap.columns:
    plot_data["ensemble_prediction"] = (
        test_with_shap["ensemble_prediction"]
    )

plot_data.to_csv(
    SHAP_DIR / "shap_visualization_data.csv",
    index=False
)


# ============================================================
# 10. FINISH
# ============================================================

print("\nSaved:")
print(
    SHAP_DIR / "shap_visualization_data.csv"
)

print("\n" + "=" * 70)
print("STEP 28B COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nSHAP figures generated in:")
print(SHAP_DIR)