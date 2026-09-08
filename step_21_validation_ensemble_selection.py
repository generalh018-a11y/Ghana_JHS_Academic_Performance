from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)


# ============================================================
# STEP 21
# VALIDATION-BASED ENSEMBLE WEIGHT SELECTION
# ============================================================

print("=" * 75)
print("STEP 21 — VALIDATION-BASED ENSEMBLE WEIGHT SELECTION")
print("=" * 75)


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


# ------------------------------------------------------------
# 2. LOAD VALIDATION PREDICTIONS
# ------------------------------------------------------------

validation_file = (
    DATA_DIR
    / "base_models_validation_predictions.csv"
)

val = pd.read_csv(validation_file)


print("\nValidation predictions loaded:")
print("Rows:", len(val))

assert len(val) == 113


# ------------------------------------------------------------
# 3. VERIFY REQUIRED COLUMNS
# ------------------------------------------------------------

required_columns = [
    "y_true",
    "xgb_probability",
    "lgbm_probability",
]

for column in required_columns:

    assert column in val.columns, (
        f"Missing required column: {column}"
    )


# ------------------------------------------------------------
# 4. DEFINE METRIC FUNCTION
# ------------------------------------------------------------

THRESHOLD = 0.50


def calculate_metrics(
    y_true,
    probability
):

    prediction = (
        probability >= THRESHOLD
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        prediction,
        labels=[0, 1]
    ).ravel()

    return {
        "accuracy": accuracy_score(
            y_true,
            prediction
        ),

        "precision": precision_score(
            y_true,
            prediction,
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            prediction,
            zero_division=0
        ),

        "macro_f1": f1_score(
            y_true,
            prediction,
            average="macro",
            zero_division=0
        ),

        "auc_roc": roc_auc_score(
            y_true,
            probability
        ),

        "auc_pr": average_precision_score(
            y_true,
            probability
        ),

        "brier_score": brier_score_loss(
            y_true,
            probability
        ),

        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


# ------------------------------------------------------------
# 5. GENERATE ENSEMBLE WEIGHTS
# ------------------------------------------------------------

weights = np.round(
    np.arange(
        0.0,
        1.01,
        0.10
    ),
    2
)


# ------------------------------------------------------------
# 6. EVALUATE ALL WEIGHTS
# ------------------------------------------------------------

results = []

y_true = val["y_true"].astype(int).to_numpy()

xgb_probability = (
    val["xgb_probability"].to_numpy()
)

lgbm_probability = (
    val["lgbm_probability"].to_numpy()
)


for xgb_weight in weights:

    lgbm_weight = 1.0 - xgb_weight

    ensemble_probability = (
        xgb_weight * xgb_probability
        + lgbm_weight * lgbm_probability
    )

    metrics = calculate_metrics(
        y_true,
        ensemble_probability
    )

    row = {
        "xgb_weight": xgb_weight,
        "lgbm_weight": lgbm_weight,
    }

    row.update(metrics)

    results.append(row)


results_df = pd.DataFrame(results)


# ------------------------------------------------------------
# 7. RANK BY VALIDATION MACRO-F1
# ------------------------------------------------------------

results_df = results_df.sort_values(
    by=[
        "macro_f1",
        "auc_pr",
        "brier_score",
    ],
    ascending=[
        False,
        False,
        True,
    ]
).reset_index(drop=True)


# ------------------------------------------------------------
# 8. SELECT BEST WEIGHT
# ------------------------------------------------------------

best = results_df.iloc[0]

BEST_XGB_WEIGHT = float(
    best["xgb_weight"]
)

BEST_LGBM_WEIGHT = float(
    best["lgbm_weight"]
)


# ------------------------------------------------------------
# 9. DISPLAY RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("ALL VALIDATION ENSEMBLE WEIGHTS")
print("=" * 75)

display_columns = [
    "xgb_weight",
    "lgbm_weight",
    "accuracy",
    "precision",
    "recall",
    "macro_f1",
    "auc_roc",
    "auc_pr",
    "brier_score",
]

print(
    results_df[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


print("\n" + "=" * 75)
print("SELECTED ENSEMBLE CONFIGURATION")
print("=" * 75)

print(
    f"XGBoost weight : {BEST_XGB_WEIGHT:.2f}"
)

print(
    f"LightGBM weight: {BEST_LGBM_WEIGHT:.2f}"
)

print(
    "Selection criterion: Validation Macro-F1"
)

print(
    f"Validation Macro-F1: {best['macro_f1']:.6f}"
)


# ------------------------------------------------------------
# 10. CHECK WEIGHT SUM
# ------------------------------------------------------------

assert np.isclose(
    BEST_XGB_WEIGHT
    + BEST_LGBM_WEIGHT,
    1.0
)


# ------------------------------------------------------------
# 11. SAVE ALL WEIGHTS
# ------------------------------------------------------------

all_weights_file = (
    DATA_DIR
    / "validation_ensemble_weight_search.csv"
)

results_df.to_csv(
    all_weights_file,
    index=False
)


# ------------------------------------------------------------
# 12. SAVE LOCKED CONFIGURATION
# ------------------------------------------------------------

locked_config = pd.DataFrame(
    [
        {
            "selection_split": "validation",
            "selection_metric": "macro_f1",
            "threshold": THRESHOLD,
            "xgb_weight": BEST_XGB_WEIGHT,
            "lgbm_weight": BEST_LGBM_WEIGHT,
            "seed": 42,
        }
    ]
)

locked_config_file = (
    DATA_DIR
    / "locked_ensemble_configuration.csv"
)

locked_config.to_csv(
    locked_config_file,
    index=False
)


# ------------------------------------------------------------
# 13. FINAL MESSAGE
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("STEP 21 COMPLETED SUCCESSFULLY")
print("=" * 75)

print("\nCreated:")
print(all_weights_file)
print(locked_config_file)

print("\n✓ Ensemble weights selected using validation data")
print("✓ Test data was NOT used for weight selection")
print("✓ Selection criterion was predefined")
print("✓ Configuration is now ready to be frozen")

print("\nSTOP HERE.")
print("Do NOT evaluate the final ensemble on the test set yet.")