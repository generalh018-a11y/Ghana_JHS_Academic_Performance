"""
STEP 28A — CELL A PRIMARY SHAP ANALYSIS

Purpose
-------
Explain the locked Cell A primary model using SHAP.

Cell A definition:
- XGBoost + LightGBM ensemble
- Fixed 50:50 weighting
- attendance_decay EXCLUDED
- Seed = 42
- Threshold = 0.50
- Exact verified model parameters from Step 23

SHAP:
- Model-independent PermutationExplainer
- Background = 100 training observations
- max_evals = 1000
- Exact Cell A feature matrix reconstructed from the canonical
  12-feature matrices by removing attendance_decay.

Important
---------
The canonical X_train/X_validation/X_test files remain unchanged.
Cell A uses an 11-feature view derived from those matrices.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results" / "shap" / "cell_a"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

X_TRAIN_PATH = DATA_DIR / "X_train.csv"
X_VALIDATION_PATH = DATA_DIR / "X_validation.csv"
X_TEST_PATH = DATA_DIR / "X_test.csv"

Y_TRAIN_PATH = DATA_DIR / "y_train.csv"
Y_VALIDATION_PATH = DATA_DIR / "y_validation.csv"
Y_TEST_PATH = DATA_DIR / "y_test.csv"

CELL_A_PREDICTIONS_PATH = (
    DATA_DIR / "cell_a_primary_test_predictions.csv"
)

# Output files
SHAP_VALUES_PATH = RESULTS_DIR / "cell_a_shap_values.csv"
BASE_VALUE_PATH = RESULTS_DIR / "cell_a_shap_base_value.json"
TEST_PREDICTIONS_PATH = RESULTS_DIR / "cell_a_shap_test_predictions.csv"
GLOBAL_IMPORTANCE_PATH = RESULTS_DIR / "cell_a_shap_global_importance.csv"
SIGNED_SHAP_PATH = RESULTS_DIR / "cell_a_shap_signed_importance.csv"
MANIFEST_PATH = RESULTS_DIR / "cell_a_shap_manifest.json"
ADDITIVITY_PATH = RESULTS_DIR / "cell_a_shap_additivity_audit.json"

GLOBAL_BAR_PATH = RESULTS_DIR / "cell_a_shap_global_bar.png"
BEESWARM_PATH = RESULTS_DIR / "cell_a_shap_beeswarm.png"
DEPENDENCE_SCORE_T1_PATH = RESULTS_DIR / "cell_a_shap_dependence_score_t1.png"
DEPENDENCE_SCORE_T2_PATH = RESULTS_DIR / "cell_a_shap_dependence_score_t2.png"
DEPENDENCE_ATTENDANCE_T1_PATH = RESULTS_DIR / "cell_a_shap_dependence_attendance_t1.png"

WATERFALL_TP_PATH = RESULTS_DIR / "cell_a_shap_waterfall_tp.png"
WATERFALL_TN_PATH = RESULTS_DIR / "cell_a_shap_waterfall_tn.png"
WATERFALL_FN_PATH = RESULTS_DIR / "cell_a_shap_waterfall_fn.png"


# ============================================================
# 2. LOCKED EXPERIMENT CONFIGURATION
# ============================================================

SEED = 42

XGB_WEIGHT = 0.50
LGBM_WEIGHT = 0.50

THRESHOLD = 0.50

EXCLUDED_FEATURE = "attendance_decay"

BACKGROUND_SIZE = 100
MAX_EVALS = 1000


# Exact verified parameters from Step 23
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_jobs": -1,
}

LGBM_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary",
    "n_jobs": -1,
    "verbosity": -1,
}


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def read_target(path):
    """
    Read a target CSV as a one-dimensional integer vector.
    """

    df = pd.read_csv(path)

    if df.shape[1] == 1:
        return df.squeeze("columns").astype(int)

    if "at_risk" in df.columns:
        return df["at_risk"].astype(int)

    if "target" in df.columns:
        return df["target"].astype(int)

    return df.iloc[:, -1].astype(int)


def save_json(path, payload):
    """
    Save JSON with stable formatting.
    """

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            payload,
            file,
            indent=2
        )


# ============================================================
# 4. START
# ============================================================

print("=" * 80)
print("STEP 28A — CELL A PRIMARY SHAP ANALYSIS")
print("=" * 80)

print("\nCell A definition:")
print("  XGBoost + LightGBM")
print("  Fixed 50:50 weighting")
print("  attendance_decay EXCLUDED")
print("  Seed = 42")
print("  Threshold = 0.50")

print("\nSHAP configuration:")
print("  Explainer = PermutationExplainer")
print(f"  Background observations = {BACKGROUND_SIZE}")
print(f"  max_evals = {MAX_EVALS}")


# ============================================================
# 5. LOAD CANONICAL MATRICES
# ============================================================

print("\n[1/10] Loading canonical model matrices...")

X_train_full = pd.read_csv(
    X_TRAIN_PATH
)

X_validation_full = pd.read_csv(
    X_VALIDATION_PATH
)

X_test_full = pd.read_csv(
    X_TEST_PATH
)

y_train = read_target(
    Y_TRAIN_PATH
)

y_validation = read_target(
    Y_VALIDATION_PATH
)

y_test = read_target(
    Y_TEST_PATH
)

print(f"X_train canonical:      {X_train_full.shape}")
print(f"X_validation canonical: {X_validation_full.shape}")
print(f"X_test canonical:       {X_test_full.shape}")


# ============================================================
# 6. VERIFY CANONICAL 12-FEATURE STRUCTURE
# ============================================================

EXPECTED_FEATURES = [
    "score_t1",
    "score_t2",
    "attendance_t1",
    "attendance_t2",
    "attendance_decay",
    "age_years",
    "gender_F",
    "gender_M",
    "class_level_JHS1",
    "class_level_JHS2",
    "academic_year_2022-23",
    "academic_year_2023-24",
]

print("\n[2/10] Verifying canonical feature structure...")

assert list(X_train_full.columns) == EXPECTED_FEATURES
assert list(X_validation_full.columns) == EXPECTED_FEATURES
assert list(X_test_full.columns) == EXPECTED_FEATURES

print("✓ Canonical 12-feature structure verified.")


# ============================================================
# 7. RECONSTRUCT EXACT CELL A MATRICES
# ============================================================

print("\n[3/10] Constructing exact Cell A feature matrices...")

CELL_A_FEATURES = [
    feature
    for feature in EXPECTED_FEATURES
    if feature != EXCLUDED_FEATURE
]

X_train = X_train_full[
    CELL_A_FEATURES
].copy()

X_validation = X_validation_full[
    CELL_A_FEATURES
].copy()

X_test = X_test_full[
    CELL_A_FEATURES
].copy()

print("\nCell A features:")

for number, feature in enumerate(
    CELL_A_FEATURES,
    start=1
):

    print(
        f"  {number:2d}. {feature}"
    )

print(
    f"\nCell A feature count: "
    f"{len(CELL_A_FEATURES)}"
)

assert EXCLUDED_FEATURE not in X_train.columns
assert EXCLUDED_FEATURE not in X_validation.columns
assert EXCLUDED_FEATURE not in X_test.columns

assert list(X_train.columns) == CELL_A_FEATURES
assert list(X_validation.columns) == CELL_A_FEATURES
assert list(X_test.columns) == CELL_A_FEATURES

assert len(CELL_A_FEATURES) == 11

print(
    f"✓ Exact Cell A 11-feature representation confirmed."
)


# ============================================================
# 8. ROW COUNT VALIDATION
# ============================================================

print("\n[4/10] Validating rows and targets...")

assert len(X_train) == len(y_train)
assert len(X_validation) == len(y_validation)
assert len(X_test) == len(y_test)

assert X_train.shape == (318, 11)
assert X_validation.shape == (113, 11)
assert X_test.shape == (110, 11)

print("✓ Train:      318 × 11")
print("✓ Validation: 113 × 11")
print("✓ Test:       110 × 11")


# ============================================================
# 9. REBUILD LOCKED CELL A MODEL
# ============================================================

print("\n[5/10] Training locked Cell A models...")

xgb_params = dict(
    XGB_PARAMS
)

xgb_params["random_state"] = SEED

xgb_model = XGBClassifier(
    **xgb_params
)

xgb_model.fit(
    X_train,
    y_train
)

print("✓ XGBoost trained.")

lgbm_params = dict(
    LGBM_PARAMS
)

lgbm_params["random_state"] = SEED

lgbm_model = LGBMClassifier(
    **lgbm_params
)

lgbm_model.fit(
    X_train,
    y_train
)

print("✓ LightGBM trained.")


# ============================================================
# 10. VERIFY LOCKED ENSEMBLE
# ============================================================

print("\n[6/10] Computing locked Cell A probabilities...")

xgb_test_probability = (
    xgb_model
    .predict_proba(X_test)[:, 1]
)

lgbm_test_probability = (
    lgbm_model
    .predict_proba(X_test)[:, 1]
)

ensemble_test_probability = (
    XGB_WEIGHT * xgb_test_probability
    +
    LGBM_WEIGHT * lgbm_test_probability
)

ensemble_prediction = (
    ensemble_test_probability >= THRESHOLD
).astype(int)

print(
    f"XGBoost weight:  {XGB_WEIGHT:.2f}"
)

print(
    f"LightGBM weight: {LGBM_WEIGHT:.2f}"
)

print(
    f"Threshold:       {THRESHOLD:.2f}"
)


# ============================================================
# 11. RECONCILE AGAINST AUTHORITATIVE CELL A ARTIFACT
# ============================================================

print("\n[7/10] Reconciling against Cell A test artifact...")

assert CELL_A_PREDICTIONS_PATH.exists(), (
    f"Missing authoritative Cell A artifact:\n"
    f"{CELL_A_PREDICTIONS_PATH}"
)

cell_a_predictions = pd.read_csv(
    CELL_A_PREDICTIONS_PATH
)

seed42_artifact = (
    cell_a_predictions[
        cell_a_predictions["seed"] == SEED
    ]
    .sort_values("row_index")
    .reset_index(drop=True)
)

assert len(seed42_artifact) == len(y_test)

artifact_probability = (
    seed42_artifact[
        "ensemble_probability"
    ]
    .to_numpy(dtype=float)
)

artifact_prediction = (
    seed42_artifact[
        "ensemble_prediction"
    ]
    .to_numpy(dtype=int)
)

probability_difference = np.max(
    np.abs(
        ensemble_test_probability
        -
        artifact_probability
    )
)

prediction_difference = np.max(
    np.abs(
        ensemble_prediction
        -
        artifact_prediction
    )
)

print(
    f"Maximum probability difference: "
    f"{probability_difference:.12e}"
)

print(
    f"Maximum prediction difference: "
    f"{prediction_difference}"
)

assert probability_difference < 1e-9, (
    "SHAP model probability does not reconcile "
    "with the authoritative Cell A artifact."
)

assert prediction_difference == 0, (
    "SHAP model predictions do not reconcile "
    "with the authoritative Cell A artifact."
)

print(
    "✓ Cell A probability/prediction reconciliation PASS."
)


# ============================================================
# 12. BUILD MODEL-INDEPENDENT SHAP EXPLAINER
# ============================================================

print("\n[8/10] Building PermutationExplainer...")

background_rng = np.random.default_rng(
    SEED
)

background_size = min(
    BACKGROUND_SIZE,
    len(X_train)
)

background_indices = (
    background_rng
    .choice(
        len(X_train),
        size=background_size,
        replace=False
    )
)

background = X_train.iloc[
    background_indices
].copy()

print(
    f"Background size: "
    f"{len(background)}"
)


def ensemble_predict(data):
    """
    Probability of the positive class for the
    exact locked Cell A 50:50 ensemble.
    """

    if isinstance(data, pd.DataFrame):
        data_frame = data.copy()
    else:
        data_frame = pd.DataFrame(
            data,
            columns=CELL_A_FEATURES
        )

    xgb_probability = (
        xgb_model
        .predict_proba(data_frame)[:, 1]
    )

    lgbm_probability = (
        lgbm_model
        .predict_proba(data_frame)[:, 1]
    )

    return (
        XGB_WEIGHT * xgb_probability
        +
        LGBM_WEIGHT * lgbm_probability
    )


explainer = shap.PermutationExplainer(
    ensemble_predict,
    background,
    seed=SEED
)

print("✓ PermutationExplainer created.")


# ============================================================
# 13. COMPUTE SHAP VALUES
# ============================================================

print("\n[9/10] Computing SHAP values...")

shap_result = explainer(
    X_test,
    max_evals=MAX_EVALS
)

shap_values = np.asarray(
    shap_result.values
)

base_values = np.asarray(
    shap_result.base_values
)

if shap_values.ndim == 3:
    shap_values = shap_values[:, :, 0]

if base_values.ndim > 1:
    base_values = base_values[:, 0]

assert shap_values.shape == (
    len(X_test),
    len(CELL_A_FEATURES)
)

print(
    f"SHAP matrix: "
    f"{shap_values.shape}"
)

base_value = float(
    np.mean(base_values)
)

print(
    f"Mean SHAP base value: "
    f"{base_value:.15f}"
)


# ============================================================
# 14. ADDITIVITY AUDIT
# ============================================================

print("\n[10/10] Performing SHAP additivity audit...")

shap_probability_reconstruction = (
    base_values
    +
    shap_values.sum(axis=1)
)

additivity_error = (
    shap_probability_reconstruction
    -
    ensemble_test_probability
)

max_additivity_error = float(
    np.max(
        np.abs(additivity_error)
    )
)

mean_additivity_error = float(
    np.mean(
        np.abs(additivity_error)
    )
)

print(
    f"Maximum absolute additivity error: "
    f"{max_additivity_error:.15e}"
)

print(
    f"Mean absolute additivity error: "
    f"{mean_additivity_error:.15e}"
)

assert max_additivity_error < 1e-9

print(
    "✓ SHAP additivity audit PASS."
)


# ============================================================
# 15. SAVE SHAP VALUES
# ============================================================

shap_df = pd.DataFrame(
    shap_values,
    columns=[
        f"SHAP_{feature}"
        for feature in CELL_A_FEATURES
    ]
)

shap_df.insert(
    0,
    "row_index",
    np.arange(len(X_test))
)

shap_df.insert(
    1,
    "y_true",
    y_test.to_numpy()
)

shap_df.insert(
    2,
    "ensemble_probability",
    ensemble_test_probability
)

shap_df.insert(
    3,
    "ensemble_prediction",
    ensemble_prediction
)

shap_df.to_csv(
    SHAP_VALUES_PATH,
    index=False
)


# ============================================================
# 16. SAVE BASE VALUE
# ============================================================

save_json(
    BASE_VALUE_PATH,
    {
        "model_cell": "Cell A",
        "seed": SEED,
        "xgb_weight": XGB_WEIGHT,
        "lgbm_weight": LGBM_WEIGHT,
        "threshold": THRESHOLD,
        "excluded_feature": EXCLUDED_FEATURE,
        "n_features": len(CELL_A_FEATURES),
        "background_size": int(background_size),
        "base_value_mean": base_value,
        "base_values_min": float(np.min(base_values)),
        "base_values_max": float(np.max(base_values)),
    }
)


# ============================================================
# 17. SAVE TEST PREDICTIONS
# ============================================================

test_predictions_df = pd.DataFrame({
    "row_index":
        np.arange(len(X_test)),

    "y_true":
        y_test.to_numpy(),

    "xgb_probability":
        xgb_test_probability,

    "lgbm_probability":
        lgbm_test_probability,

    "ensemble_probability":
        ensemble_test_probability,

    "ensemble_prediction":
        ensemble_prediction,

    "threshold":
        THRESHOLD,

    "xgb_weight":
        XGB_WEIGHT,

    "lgbm_weight":
        LGBM_WEIGHT,

    "model_cell":
        "Cell A",

    "seed":
        SEED,

    "attendance_decay":
        False,
})

test_predictions_df.to_csv(
    TEST_PREDICTIONS_PATH,
    index=False
)


# ============================================================
# 18. GLOBAL SHAP IMPORTANCE
# ============================================================

mean_abs_shap = (
    np.mean(
        np.abs(shap_values),
        axis=0
    )
)

global_importance_df = pd.DataFrame({
    "feature":
        CELL_A_FEATURES,

    "mean_abs_shap":
        mean_abs_shap,
})

global_importance_df = (
    global_importance_df
    .sort_values(
        "mean_abs_shap",
        ascending=False
    )
    .reset_index(drop=True)
)

global_importance_df.insert(
    0,
    "rank",
    np.arange(
        1,
        len(global_importance_df) + 1
    )
)

global_importance_df.to_csv(
    GLOBAL_IMPORTANCE_PATH,
    index=False
)


# ============================================================
# 19. SIGNED SHAP IMPORTANCE
# ============================================================

mean_signed_shap = (
    np.mean(
        shap_values,
        axis=0
    )
)

signed_importance_df = pd.DataFrame({
    "feature":
        CELL_A_FEATURES,

    "mean_signed_shap":
        mean_signed_shap,
})

signed_importance_df.to_csv(
    SIGNED_SHAP_PATH,
    index=False
)


# ============================================================
# 20. SHAP PLOTS
# ============================================================

print("\nCreating SHAP figures...")


# ------------------------------------------------------------
# Global bar
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 7)
)

shap.summary_plot(
    shap_values,
    X_test,
    feature_names=CELL_A_FEATURES,
    plot_type="bar",
    show=False
)

plt.tight_layout()

plt.savefig(
    GLOBAL_BAR_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# Beeswarm
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 7)
)

shap.summary_plot(
    shap_values,
    X_test,
    feature_names=CELL_A_FEATURES,
    show=False
)

plt.tight_layout()

plt.savefig(
    BEESWARM_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# Dependence: score_t1
# ------------------------------------------------------------

plt.figure(
    figsize=(9, 6)
)

shap.dependence_plot(
    "score_t1",
    shap_values,
    X_test,
    feature_names=CELL_A_FEATURES,
    show=False
)

plt.tight_layout()

plt.savefig(
    DEPENDENCE_SCORE_T1_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# Dependence: score_t2
# ------------------------------------------------------------

plt.figure(
    figsize=(9, 6)
)

shap.dependence_plot(
    "score_t2",
    shap_values,
    X_test,
    feature_names=CELL_A_FEATURES,
    show=False
)

plt.tight_layout()

plt.savefig(
    DEPENDENCE_SCORE_T2_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# Dependence: attendance_t1
# ------------------------------------------------------------

plt.figure(
    figsize=(9, 6)
)

shap.dependence_plot(
    "attendance_t1",
    shap_values,
    X_test,
    feature_names=CELL_A_FEATURES,
    show=False
)

plt.tight_layout()

plt.savefig(
    DEPENDENCE_ATTENDANCE_T1_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 21. SELECT REPRESENTATIVE WATERFALL CASES
# ============================================================

def find_case(
    true_label,
    predicted_label,
    probability_condition
):
    """
    Find a deterministic representative case.
    """

    candidates = np.where(
        (
            y_test.to_numpy() == true_label
        )
        &
        (
            ensemble_prediction == predicted_label
        )
        &
        probability_condition(
            ensemble_test_probability
        )
    )[0]

    if len(candidates) == 0:
        return None

    return int(candidates[0])


tp_index = find_case(
    true_label=1,
    predicted_label=1,
    probability_condition=lambda p: p >= 0.50
)

tn_index = find_case(
    true_label=0,
    predicted_label=0,
    probability_condition=lambda p: p < 0.50
)

fn_index = find_case(
    true_label=1,
    predicted_label=0,
    probability_condition=lambda p: p >= 0.20
)


def save_waterfall(
    index,
    path,
    title
):
    """
    Save one SHAP waterfall plot.
    """

    if index is None:
        print(
            f"Skipping {title}: "
            "no matching case found."
        )
        return

    explanation = shap.Explanation(
        values=shap_values[index],
        base_values=base_values[index],
        data=X_test.iloc[index].to_numpy(),
        feature_names=CELL_A_FEATURES
    )

    plt.figure(
        figsize=(12, 8)
    )

    shap.plots.waterfall(
        explanation,
        max_display=len(CELL_A_FEATURES),
        show=False
    )

    plt.title(title)

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


save_waterfall(
    tp_index,
    WATERFALL_TP_PATH,
    "Cell A SHAP Waterfall — True Positive"
)

save_waterfall(
    tn_index,
    WATERFALL_TN_PATH,
    "Cell A SHAP Waterfall — True Negative"
)

save_waterfall(
    fn_index,
    WATERFALL_FN_PATH,
    "Cell A SHAP Waterfall — False Negative"
)


# ============================================================
# 22. SAVE AUDIT
# ============================================================

save_json(
    ADDITIVITY_PATH,
    {
        "model_cell":
            "Cell A",

        "seed":
            SEED,

        "n_test_rows":
            int(len(X_test)),

        "n_features":
            int(len(CELL_A_FEATURES)),

        "excluded_feature":
            EXCLUDED_FEATURE,

        "background_size":
            int(background_size),

        "max_evals":
            int(MAX_EVALS),

        "base_value_mean":
            base_value,

        "max_absolute_additivity_error":
            max_additivity_error,

        "mean_absolute_additivity_error":
            mean_additivity_error,

        "probability_reconciliation_max_abs_difference":
            float(probability_difference),

        "prediction_reconciliation_max_difference":
            int(prediction_difference),

        "additivity_pass":
            bool(max_additivity_error < 1e-9),

        "cell_a_reconciliation_pass":
            bool(
                probability_difference < 1e-9
                and prediction_difference == 0
            ),
    }
)


# ============================================================
# 23. SAVE MANIFEST
# ============================================================

manifest = {
    "model_cell": "Cell A",
    "description": (
        "Fixed 50:50 XGBoost-LightGBM ensemble "
        "without attendance_decay."
    ),
    "seed": SEED,
    "xgb_weight": XGB_WEIGHT,
    "lgbm_weight": LGBM_WEIGHT,
    "threshold": THRESHOLD,
    "excluded_feature": EXCLUDED_FEATURE,
    "features": CELL_A_FEATURES,
    "n_features": len(CELL_A_FEATURES),
    "n_train_rows": len(X_train),
    "n_validation_rows": len(X_validation),
    "n_test_rows": len(X_test),
    "background_size": background_size,
    "max_evals": MAX_EVALS,
    "explainer": "PermutationExplainer",
    "xgb_parameters": XGB_PARAMS,
    "lgbm_parameters": LGBM_PARAMS,
    "canonical_input_features": EXPECTED_FEATURES,
    "protocol_note": (
        "The canonical 12-feature matrices are retained unchanged. "
        "Cell A is reconstructed exactly by removing "
        "attendance_decay, yielding the verified 11-feature "
        "Cell A representation."
    ),
    "reconciliation": {
        "max_probability_difference":
            float(probability_difference),
        "max_prediction_difference":
            int(prediction_difference),
        "passed":
            bool(
                probability_difference < 1e-9
                and prediction_difference == 0
            ),
    },
    "additivity": {
        "max_absolute_error":
            max_additivity_error,
        "mean_absolute_error":
            mean_additivity_error,
        "passed":
            bool(max_additivity_error < 1e-9),
    },
}

save_json(
    MANIFEST_PATH,
    manifest
)


# ============================================================
# 24. FINAL VALIDATION
# ============================================================

assert shap_values.shape == (
    110,
    11
)

assert len(global_importance_df) == 11

assert np.isfinite(
    shap_values
).all()

assert np.isfinite(
    ensemble_test_probability
).all()

assert probability_difference < 1e-9
assert prediction_difference == 0
assert max_additivity_error < 1e-9


# ============================================================
# 25. FINAL REPORT
# ============================================================

print("\n")
print("=" * 80)
print("STEP 28A COMPLETED SUCCESSFULLY")
print("=" * 80)

print("\n✓ Cell A reconstructed exactly")
print("✓ attendance_decay excluded")
print("✓ 11 features used")
print("✓ Seed 42")
print("✓ Fixed 50:50 XGBoost/LightGBM")
print("✓ Threshold = 0.50")
print("✓ Exact verified model parameters")
print("✓ Cell A artifact reconciliation PASS")
print("✓ SHAP additivity audit PASS")

print(
    f"\nMaximum probability reconciliation error: "
    f"{probability_difference:.15e}"
)

print(
    f"Maximum SHAP additivity error: "
    f"{max_additivity_error:.15e}"
)

print("\nTop SHAP features:")

print(
    global_importance_df
    .head(10)
    .to_string(index=False)
)

print("\nOutput directory:")
print(RESULTS_DIR)

print("\nKey files:")
print(SHAP_VALUES_PATH)
print(BASE_VALUE_PATH)
print(TEST_PREDICTIONS_PATH)
print(GLOBAL_IMPORTANCE_PATH)
print(SIGNED_SHAP_PATH)
print(MANIFEST_PATH)
print(ADDITIVITY_PATH)

print("\nFigures:")
print(GLOBAL_BAR_PATH)
print(BEESWARM_PATH)
print(DEPENDENCE_SCORE_T1_PATH)
print(DEPENDENCE_SCORE_T2_PATH)
print(DEPENDENCE_ATTENDANCE_T1_PATH)
print(WATERFALL_TP_PATH)
print(WATERFALL_TN_PATH)
print(WATERFALL_FN_PATH)

print("\n" + "=" * 80)
print("READY FOR CHAPTER 4 SHAP TABLES/FIGURES")
print("=" * 80)
