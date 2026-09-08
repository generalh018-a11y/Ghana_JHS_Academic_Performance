"""
Step 28A: SHAP explanation for the locked XGBoost-LightGBM ensemble.

Purpose:
- Rebuild the locked seed-42 XGBoost and LightGBM models.
- Use the pre-established 10% XGBoost / 90% LightGBM ensemble.
- Explain the ensemble using model-independent SHAP.
- Save the SHAP values and the actual SHAP base values.

Important:
- No model selection is performed.
- No threshold tuning is performed.
- No test-set optimization is performed.
- SHAP is used only for post-hoc explainability.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import lightgbm as lgb


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "shap"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD PREPARED DATA
# ============================================================

X_train = pd.read_csv(
    DATA_DIR / "X_train.csv"
)

X_validation = pd.read_csv(
    DATA_DIR / "X_validation.csv"
)

X_test = pd.read_csv(
    DATA_DIR / "X_test.csv"
)

y_train = pd.read_csv(
    DATA_DIR / "y_train.csv"
).squeeze("columns")

y_validation = pd.read_csv(
    DATA_DIR / "y_validation.csv"
).squeeze("columns")

y_test = pd.read_csv(
    DATA_DIR / "y_test.csv"
).squeeze("columns")


print("=" * 70)
print("STEP 28A — SHAP ENSEMBLE EXPLAINABILITY")
print("=" * 70)


# ============================================================
# 3. DATA SHAPE CHECK
# ============================================================

print("\nFeature shapes:")
print("X_train:", X_train.shape)
print("X_validation:", X_validation.shape)
print("X_test:", X_test.shape)


# ============================================================
# 4. FEATURE CONSISTENCY CHECK
# ============================================================

if list(X_train.columns) != list(X_validation.columns):
    raise ValueError(
        "Training and validation feature columns do not match."
    )

if list(X_train.columns) != list(X_test.columns):
    raise ValueError(
        "Training and test feature columns do not match."
    )


FEATURES = list(X_train.columns)


print("\nCanonical features:")

for i, feature in enumerate(FEATURES, start=1):
    print(f"{i:2d}. {feature}")


# ============================================================
# 5. MISSING-VALUE CHECK
# ============================================================

for name, dataframe in [
    ("X_train", X_train),
    ("X_validation", X_validation),
    ("X_test", X_test)
]:

    missing_count = int(
        dataframe.isna().sum().sum()
    )

    print(
        f"\n{name} missing values: {missing_count}"
    )

    if missing_count > 0:
        raise ValueError(
            f"{name} contains missing values."
        )


# ============================================================
# 6. LOCKED ENSEMBLE CONFIGURATION
# ============================================================

SEED = 42

XGB_WEIGHT = 0.10
LGBM_WEIGHT = 0.90

THRESHOLD = 0.50


print("\nLocked configuration:")
print(f"XGBoost weight : {XGB_WEIGHT:.2f}")
print(f"LightGBM weight: {LGBM_WEIGHT:.2f}")
print(f"Threshold      : {THRESHOLD:.2f}")
print(f"Seed           : {SEED}")


# ============================================================
# 7. TRAIN LOCKED XGBOOST MODEL
# ============================================================

print("\nTraining locked XGBoost model...")


xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.80,
    colsample_bytree=0.80,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=SEED,
    n_jobs=-1
)


xgb_model.fit(
    X_train,
    y_train
)


print("XGBoost training complete.")


# ============================================================
# 8. TRAIN LOCKED LIGHTGBM MODEL
# ============================================================

print("\nTraining locked LightGBM model...")


lgbm_model = lgb.LGBMClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.80,
    colsample_bytree=0.80,
    objective="binary",
    random_state=SEED,
    n_jobs=-1,
    verbosity=-1
)


lgbm_model.fit(
    X_train,
    y_train
)


print("LightGBM training complete.")


# ============================================================
# 9. DEFINE LOCKED ENSEMBLE
# ============================================================

def ensemble_predict(X):
    """
    Return probability of the positive class (at-risk).

    Locked ensemble:

        0.10 * XGBoost probability
      + 0.90 * LightGBM probability
    """

    X_df = pd.DataFrame(
        X,
        columns=FEATURES
    )

    xgb_probability = (
        xgb_model.predict_proba(X_df)[:, 1]
    )

    lgbm_probability = (
        lgbm_model.predict_proba(X_df)[:, 1]
    )

    ensemble_probability = (
        XGB_WEIGHT * xgb_probability
        + LGBM_WEIGHT * lgbm_probability
    )

    return ensemble_probability


# ============================================================
# 10. VERIFY ENSEMBLE PROBABILITIES
# ============================================================

test_probabilities = ensemble_predict(
    X_test
)


if np.any(test_probabilities < 0):
    raise ValueError(
        "Ensemble probabilities below 0 detected."
    )

if np.any(test_probabilities > 1):
    raise ValueError(
        "Ensemble probabilities above 1 detected."
    )


print("\nEnsemble probability verification:")
print("Minimum:", test_probabilities.min())
print("Maximum:", test_probabilities.max())
print("Mean   :", test_probabilities.mean())


# ============================================================
# 11. CREATE SHAP BACKGROUND
# ============================================================

BACKGROUND_SIZE = min(
    100,
    len(X_train)
)


background = X_train.sample(
    n=BACKGROUND_SIZE,
    random_state=SEED
).reset_index(drop=True)


print("\nSHAP background:")
print(
    "Background rows:",
    len(background)
)


# ============================================================
# 12. CREATE MODEL-INDEPENDENT SHAP EXPLAINER
# ============================================================

print(
    "\nCreating model-independent SHAP explainer..."
)


explainer = shap.Explainer(
    ensemble_predict,
    background,
    algorithm="permutation"
)


print(
    "SHAP explainer created."
)


# ============================================================
# 13. CALCULATE SHAP VALUES
# ============================================================

print(
    "\nCalculating SHAP values for test observations..."
)

print(
    "This may take some time because the ensemble "
    "is treated as a black-box prediction function."
)


shap_explanation = explainer(
    X_test,
    max_evals=1000
)


print(
    "SHAP calculation complete."
)


# ============================================================
# 14. EXTRACT SHAP VALUES
# ============================================================

shap_values = np.asarray(
    shap_explanation.values
)


print(
    "\nSHAP value shape:",
    shap_values.shape
)


if shap_values.ndim != 2:
    raise ValueError(
        f"Unexpected SHAP dimensions: "
        f"{shap_values.shape}"
    )


if shap_values.shape != X_test.shape:
    raise ValueError(
        "SHAP values and X_test dimensions do not match."
    )


# ============================================================
# 15. EXTRACT ACTUAL SHAP BASE VALUES
# ============================================================

base_values = np.asarray(
    shap_explanation.base_values
).reshape(-1)


print(
    "\nSHAP base value verification:"
)

print(
    "Number of base values:",
    len(base_values)
)

print(
    "Minimum:",
    base_values.min()
)

print(
    "Maximum:",
    base_values.max()
)

print(
    "Mean   :",
    base_values.mean()
)


# ============================================================
# 16. SAVE SHAP BASE VALUES
# ============================================================

base_values_df = pd.DataFrame({
    "test_row": np.arange(
        len(base_values)
    ),
    "shap_base_value": base_values
})


base_values_df.to_csv(
    RESULTS_DIR /
    "shap_base_values_test.csv",
    index=False
)


print(
    "\nSaved:",
    RESULTS_DIR /
    "shap_base_values_test.csv"
)


# ============================================================
# 17. SAVE RAW SHAP VALUES
# ============================================================

shap_values_df = pd.DataFrame(
    shap_values,
    columns=FEATURES
)


shap_values_df.insert(
    0,
    "test_row",
    np.arange(
        len(X_test)
    )
)


shap_values_df.to_csv(
    RESULTS_DIR /
    "shap_values_test_ensemble.csv",
    index=False
)


print(
    "Saved:",
    RESULTS_DIR /
    "shap_values_test_ensemble.csv"
)


# ============================================================
# 18. SAVE TEST DATA + SHAP VALUES
# ============================================================

test_with_shap = X_test.copy()


test_with_shap.insert(
    0,
    "test_row",
    np.arange(
        len(X_test)
    )
)


test_with_shap[
    "actual_at_risk"
] = np.asarray(
    y_test
)


test_with_shap[
    "ensemble_probability"
] = test_probabilities


test_with_shap[
    "ensemble_prediction"
] = (
    test_probabilities >= THRESHOLD
).astype(int)


test_with_shap[
    "shap_base_value"
] = base_values


for feature in FEATURES:

    test_with_shap[
        f"shap_{feature}"
    ] = shap_values_df[
        feature
    ]


test_with_shap.to_csv(
    RESULTS_DIR /
    "test_predictions_with_shap.csv",
    index=False
)


print(
    "Saved:",
    RESULTS_DIR /
    "test_predictions_with_shap.csv"
)


# ============================================================
# 19. GLOBAL SHAP FEATURE IMPORTANCE
# ============================================================

mean_abs_shap = (
    np.abs(shap_values)
    .mean(axis=0)
)


global_importance = pd.DataFrame({
    "feature": FEATURES,
    "mean_abs_shap": mean_abs_shap
})


global_importance = (
    global_importance
    .sort_values(
        "mean_abs_shap",
        ascending=False
    )
    .reset_index(drop=True)
)


global_importance[
    "rank"
] = (
    np.arange(
        len(global_importance)
    ) + 1
)


global_importance = global_importance[
    [
        "rank",
        "feature",
        "mean_abs_shap"
    ]
]


global_importance.to_csv(
    RESULTS_DIR /
    "shap_global_feature_importance.csv",
    index=False
)


print(
    "\nGlobal SHAP feature importance:"
)

print(
    global_importance.to_string(
        index=False
    )
)


# ============================================================
# 20. MEAN SIGNED SHAP VALUES
# ============================================================

mean_signed_shap = (
    shap_values.mean(axis=0)
)


signed_importance = pd.DataFrame({
    "feature": FEATURES,
    "mean_shap": mean_signed_shap
})


signed_importance = (
    signed_importance
    .sort_values(
        "mean_shap",
        ascending=False
    )
    .reset_index(drop=True)
)


signed_importance.to_csv(
    RESULTS_DIR /
    "shap_mean_signed_values.csv",
    index=False
)


print(
    "\nMean signed SHAP values:"
)

print(
    signed_importance.to_string(
        index=False
    )
)


# ============================================================
# 21. SHAP RUN MANIFEST
# ============================================================

summary = pd.DataFrame({
    "metric": [
        "shap_version",
        "xgboost_version",
        "lightgbm_version",
        "seed",
        "xgb_weight",
        "lgbm_weight",
        "threshold",
        "background_rows",
        "test_rows",
        "number_of_features"
    ],

    "value": [
        shap.__version__,
        xgb.__version__,
        lgb.__version__,
        SEED,
        XGB_WEIGHT,
        LGBM_WEIGHT,
        THRESHOLD,
        BACKGROUND_SIZE,
        len(X_test),
        len(FEATURES)
    ]
})


summary.to_csv(
    RESULTS_DIR /
    "shap_run_manifest.csv",
    index=False
)


print(
    "\nSHAP run manifest saved."
)


# ============================================================
# 22. FINISH
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "STEP 28A COMPLETED SUCCESSFULLY"
)

print(
    "=" * 70
)


print(
    "\nGenerated files:"
)


for file in sorted(
    RESULTS_DIR.glob("*.csv")
):

    print(
        " -",
        file.name
    )


print(
    "\nIMPORTANT:"
)

print(
    "SHAP was used only for post-hoc explainability."
)

print(
    "No model selection or threshold tuning was performed."
)
