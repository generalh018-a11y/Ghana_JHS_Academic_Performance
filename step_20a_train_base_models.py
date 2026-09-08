from pathlib import Path
import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

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
# STEP 20A
# CONTROLLED TRAINING OF BASE XGBOOST AND LIGHTGBM MODELS
# ============================================================

print("=" * 75)
print("STEP 20A — CONTROLLED BASE MODEL TRAINING")
print("=" * 75)


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"


# ------------------------------------------------------------
# 2. LOAD FEATURES
# ------------------------------------------------------------

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_validation = pd.read_csv(DATA_DIR / "X_validation.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")

y_train = pd.read_csv(
    DATA_DIR / "y_train.csv"
).squeeze("columns").astype(int)

y_validation = pd.read_csv(
    DATA_DIR / "y_validation.csv"
).squeeze("columns").astype(int)

y_test = pd.read_csv(
    DATA_DIR / "y_test.csv"
).squeeze("columns").astype(int)


print("\nDataset shapes:")
print("X_train:", X_train.shape)
print("X_validation:", X_validation.shape)
print("X_test:", X_test.shape)


# ------------------------------------------------------------
# 3. FIXED EXPERIMENTAL SEED
# ------------------------------------------------------------

SEED = 42

print("\nExperimental seed:", SEED)


# ------------------------------------------------------------
# 4. DEFINE BASE MODELS
# ------------------------------------------------------------

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=SEED,
    n_jobs=-1,
)

lgbm_model = LGBMClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary",
    random_state=SEED,
    n_jobs=-1,
    verbosity=-1,
)


# ------------------------------------------------------------
# 5. TRAIN XGBOOST
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("TRAINING XGBOOST")
print("=" * 75)

xgb_model.fit(
    X_train,
    y_train
)

print("✓ XGBoost training completed.")


# ------------------------------------------------------------
# 6. TRAIN LIGHTGBM
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("TRAINING LIGHTGBM")
print("=" * 75)

lgbm_model.fit(
    X_train,
    y_train
)

print("✓ LightGBM training completed.")


# ------------------------------------------------------------
# 7. PREDICT VALIDATION AND TEST
# ------------------------------------------------------------

xgb_val_prob = xgb_model.predict_proba(
    X_validation
)[:, 1]

xgb_test_prob = xgb_model.predict_proba(
    X_test
)[:, 1]

lgbm_val_prob = lgbm_model.predict_proba(
    X_validation
)[:, 1]

lgbm_test_prob = lgbm_model.predict_proba(
    X_test
)[:, 1]


# ------------------------------------------------------------
# 8. CLASSIFICATION THRESHOLD
# ------------------------------------------------------------

THRESHOLD = 0.50

xgb_val_pred = (
    xgb_val_prob >= THRESHOLD
).astype(int)

xgb_test_pred = (
    xgb_test_prob >= THRESHOLD
).astype(int)

lgbm_val_pred = (
    lgbm_val_prob >= THRESHOLD
).astype(int)

lgbm_test_pred = (
    lgbm_test_prob >= THRESHOLD
).astype(int)


# ------------------------------------------------------------
# 9. METRIC FUNCTION
# ------------------------------------------------------------

def calculate_metrics(
    y_true,
    y_pred,
    y_prob
):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "auc_roc": roc_auc_score(
            y_true,
            y_prob
        ),

        "auc_pr": average_precision_score(
            y_true,
            y_prob
        ),

        "brier_score": brier_score_loss(
            y_true,
            y_prob
        ),

        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


# ------------------------------------------------------------
# 10. VALIDATION RESULTS
# ------------------------------------------------------------

xgb_val_metrics = calculate_metrics(
    y_validation,
    xgb_val_pred,
    xgb_val_prob
)

lgbm_val_metrics = calculate_metrics(
    y_validation,
    lgbm_val_pred,
    lgbm_val_prob
)


print("\n" + "=" * 75)
print("VALIDATION RESULTS")
print("=" * 75)

print("\nXGBoost:")

for key, value in xgb_val_metrics.items():
    print(f"{key:15s}: {value:.6f}")


print("\nLightGBM:")

for key, value in lgbm_val_metrics.items():
    print(f"{key:15s}: {value:.6f}")


# ------------------------------------------------------------
# 11. TEST RESULTS
#
# IMPORTANT:
# These are reported for transparency only.
# They must NOT be used to select a model or ensemble weight.
# ------------------------------------------------------------

xgb_test_metrics = calculate_metrics(
    y_test,
    xgb_test_pred,
    xgb_test_prob
)

lgbm_test_metrics = calculate_metrics(
    y_test,
    lgbm_test_pred,
    lgbm_test_prob
)


print("\n" + "=" * 75)
print("TEST RESULTS — DESCRIPTIVE ONLY")
print("=" * 75)

print("\nXGBoost:")

for key, value in xgb_test_metrics.items():
    print(f"{key:15s}: {value:.6f}")


print("\nLightGBM:")

for key, value in lgbm_test_metrics.items():
    print(f"{key:15s}: {value:.6f}")


# ------------------------------------------------------------
# 12. SAVE PROBABILITIES
# ------------------------------------------------------------

validation_predictions = pd.DataFrame(
    {
        "y_true": y_validation,
        "xgb_probability": xgb_val_prob,
        "xgb_prediction": xgb_val_pred,
        "lgbm_probability": lgbm_val_prob,
        "lgbm_prediction": lgbm_val_pred,
    }
)

test_predictions = pd.DataFrame(
    {
        "y_true": y_test,
        "xgb_probability": xgb_test_prob,
        "xgb_prediction": xgb_test_pred,
        "lgbm_probability": lgbm_test_prob,
        "lgbm_prediction": lgbm_test_pred,
    }
)


validation_predictions.to_csv(
    DATA_DIR / "base_models_validation_predictions.csv",
    index=False
)

test_predictions.to_csv(
    DATA_DIR / "base_models_test_predictions.csv",
    index=False
)


# ------------------------------------------------------------
# 13. SAVE MODEL RESULTS
# ------------------------------------------------------------

results = []

for model_name, split_name, metrics in [
    (
        "XGBoost",
        "validation",
        xgb_val_metrics
    ),
    (
        "LightGBM",
        "validation",
        lgbm_val_metrics
    ),
    (
        "XGBoost",
        "test",
        xgb_test_metrics
    ),
    (
        "LightGBM",
        "test",
        lgbm_test_metrics
    ),
]:

    row = {
        "model": model_name,
        "split": split_name,
        "seed": SEED,
    }

    row.update(metrics)

    results.append(row)


results_df = pd.DataFrame(results)

results_df.to_csv(
    DATA_DIR / "base_model_results_seed42.csv",
    index=False
)


# ------------------------------------------------------------
# 14. FINAL CHECKS
# ------------------------------------------------------------

assert len(xgb_val_prob) == len(y_validation)
assert len(lgbm_val_prob) == len(y_validation)

assert len(xgb_test_prob) == len(y_test)
assert len(lgbm_test_prob) == len(y_test)

assert np.isfinite(
    xgb_val_prob
).all()

assert np.isfinite(
    lgbm_val_prob
).all()

print("\n" + "=" * 75)
print("STEP 20A COMPLETED SUCCESSFULLY")
print("=" * 75)

print("\nSaved:")
print(DATA_DIR / "base_models_validation_predictions.csv")
print(DATA_DIR / "base_models_test_predictions.csv")
print(DATA_DIR / "base_model_results_seed42.csv")

print("\n✓ XGBoost trained")
print("✓ LightGBM trained")
print("✓ Validation predictions generated")
print("✓ Test predictions generated for descriptive comparison")
print("✓ Test results were NOT used for model selection")

print("\nSTOP HERE.")
print("Do NOT select an ensemble weight yet.")