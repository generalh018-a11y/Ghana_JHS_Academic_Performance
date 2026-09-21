"""
Step 23A — Cell A Primary Model Stability Analysis

Cell A:
    XGBoost + LightGBM fixed 50:50 ensemble
    WITHOUT attendance_decay

Protocol:
    - Same existing train/validation/test matrices
    - Same fixed model hyperparameters as original Step 23
    - Model seeds 42–51
    - Fixed 50:50 ensemble
    - Threshold = 0.50
    - No validation-based weight selection
    - No test-based selection
    - attendance_decay excluded

Purpose:
    Establish Cell A as the candidate primary model and verify
    its ten-seed stability using the same model specification as
    the original verified experiment.

Outputs:
    data/processed/cell_a_primary_test_results.csv
    data/processed/cell_a_primary_summary_mean_sd.csv
    data/processed/cell_a_primary_test_predictions.csv
    data/processed/cell_a_primary_audit.json
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

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


warnings.filterwarnings("ignore")


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"

X_TRAIN_PATH = DATA_DIR / "X_train.csv"
X_VALIDATION_PATH = DATA_DIR / "X_validation.csv"
X_TEST_PATH = DATA_DIR / "X_test.csv"

Y_TRAIN_PATH = DATA_DIR / "y_train.csv"
Y_VALIDATION_PATH = DATA_DIR / "y_validation.csv"
Y_TEST_PATH = DATA_DIR / "y_test.csv"

OUT_RESULTS = (
    DATA_DIR /
    "cell_a_primary_test_results.csv"
)

OUT_SUMMARY = (
    DATA_DIR /
    "cell_a_primary_summary_mean_sd.csv"
)

OUT_PREDICTIONS = (
    DATA_DIR /
    "cell_a_primary_test_predictions.csv"
)

OUT_AUDIT = (
    DATA_DIR /
    "cell_a_primary_audit.json"
)


# ============================================================
# 2. EXPERIMENTAL PARAMETERS
# ============================================================

SEEDS = list(range(42, 52))

XGB_WEIGHT = 0.50
LGBM_WEIGHT = 0.50

THRESHOLD = 0.50

EXCLUDED_FEATURE = "attendance_decay"


# ============================================================
# 3. EXACT MODEL PARAMETERS FROM ORIGINAL STEP 23
# ============================================================

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
# 4. HELPER FUNCTIONS
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


def calculate_metrics(
    y_true,
    probability,
    threshold=0.50
):
    """
    Calculate the same metrics used by the original
    Step 23 experiment.
    """

    prediction = (
        probability >= threshold
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

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ============================================================
# 5. START
# ============================================================

print("=" * 75)
print("STEP 23A — CELL A PRIMARY MODEL STABILITY")
print("=" * 75)

print("\nCell A definition:")
print("  XGBoost + LightGBM")
print("  Fixed 50:50 weighting")
print("  attendance_decay EXCLUDED")
print("  Threshold = 0.50")

print("\nSeeds:")
print(SEEDS)


# ============================================================
# 6. LOAD DATA
# ============================================================

print("\n[1/8] Loading model matrices...")

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


print(
    f"Train:      "
    f"{X_train_full.shape}"
)

print(
    f"Validation: "
    f"{X_validation_full.shape}"
)

print(
    f"Test:       "
    f"{X_test_full.shape}"
)


# ============================================================
# 7. ROW COUNT VALIDATION
# ============================================================

print("\n[2/8] Checking row counts...")

assert len(X_train_full) == len(y_train), (
    "Training X/y row count mismatch."
)

assert len(X_validation_full) == len(
    y_validation
), (
    "Validation X/y row count mismatch."
)

assert len(X_test_full) == len(y_test), (
    "Test X/y row count mismatch."
)


print("✓ Train rows match")
print("✓ Validation rows match")
print("✓ Test rows match")


# ============================================================
# 8. VERIFY ORIGINAL 12-FEATURE STRUCTURE
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


print(
    "\n[3/8] Verifying canonical feature structure..."
)

assert list(X_train_full.columns) == (
    EXPECTED_FEATURES
)

assert list(X_validation_full.columns) == (
    EXPECTED_FEATURES
)

assert list(X_test_full.columns) == (
    EXPECTED_FEATURES
)

print(
    "✓ All three matrices contain the "
    "same canonical 12 features."
)


# ============================================================
# 9. CREATE CELL A MATRICES
# ============================================================

print(
    "\n[4/8] Removing attendance_decay "
    "for Cell A..."
)

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

print(
    f"Excluded: "
    f"{EXCLUDED_FEATURE}"
)


# ============================================================
# 10. TARGET DISTRIBUTION
# ============================================================

print(
    "\n[5/8] Checking target distributions..."
)

print(
    "Train:",
    dict(
        pd.Series(y_train)
        .value_counts()
        .sort_index()
    )
)

print(
    "Validation:",
    dict(
        pd.Series(y_validation)
        .value_counts()
        .sort_index()
    )
)

print(
    "Test:",
    dict(
        pd.Series(y_test)
        .value_counts()
        .sort_index()
    )
)


# ============================================================
# 11. RUN TEN SEEDS
# ============================================================

print(
    "\n[6/8] Running Cell A across "
    "ten model seeds..."
)

all_results = []
all_predictions = []


for seed in SEEDS:

    print("\n")
    print("-" * 75)
    print(
        f"RUNNING CELL A — SEED {seed}"
    )
    print("-" * 75)

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    xgb_params = dict(
        XGB_PARAMS
    )

    xgb_params[
        "random_state"
    ] = seed

    xgb_model = XGBClassifier(
        **xgb_params
    )

    xgb_model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # LightGBM
    # --------------------------------------------------------

    lgbm_params = dict(
        LGBM_PARAMS
    )

    lgbm_params[
        "random_state"
    ] = seed

    lgbm_model = LGBMClassifier(
        **lgbm_params
    )

    lgbm_model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # Validation probabilities
    #
    # Validation is retained for auditability but is NOT
    # used to select the Cell A 50:50 weight.
    # --------------------------------------------------------

    xgb_validation_probability = (
        xgb_model
        .predict_proba(
            X_validation
        )[:, 1]
    )

    lgbm_validation_probability = (
        lgbm_model
        .predict_proba(
            X_validation
        )[:, 1]
    )


    ensemble_validation_probability = (
        XGB_WEIGHT
        * xgb_validation_probability
        +
        LGBM_WEIGHT
        * lgbm_validation_probability
    )


    # --------------------------------------------------------
    # Test probabilities
    # --------------------------------------------------------

    xgb_test_probability = (
        xgb_model
        .predict_proba(
            X_test
        )[:, 1]
    )

    lgbm_test_probability = (
        lgbm_model
        .predict_proba(
            X_test
        )[:, 1]
    )


    ensemble_test_probability = (
        XGB_WEIGHT
        * xgb_test_probability
        +
        LGBM_WEIGHT
        * lgbm_test_probability
    )


    # --------------------------------------------------------
    # Validation metrics
    # --------------------------------------------------------

    validation_metrics = calculate_metrics(
        y_validation,
        ensemble_validation_probability,
        THRESHOLD
    )


    # --------------------------------------------------------
    # Test metrics
    # --------------------------------------------------------

    test_metrics = calculate_metrics(
        y_test,
        ensemble_test_probability,
        THRESHOLD
    )


    # --------------------------------------------------------
    # Store test result
    # --------------------------------------------------------

    result = {

        "model_cell":
            "Cell A",

        "seed":
            seed,

        "xgb_weight":
            XGB_WEIGHT,

        "lgbm_weight":
            LGBM_WEIGHT,

        "threshold":
            THRESHOLD,

        "attendance_decay":
            False,


        "validation_accuracy":
            validation_metrics[
                "accuracy"
            ],

        "validation_precision":
            validation_metrics[
                "precision"
            ],

        "validation_recall":
            validation_metrics[
                "recall"
            ],

        "validation_macro_f1":
            validation_metrics[
                "macro_f1"
            ],

        "validation_auc_roc":
            validation_metrics[
                "auc_roc"
            ],

        "validation_auc_pr":
            validation_metrics[
                "auc_pr"
            ],

        "validation_brier":
            validation_metrics[
                "brier_score"
            ],


        "accuracy":
            test_metrics[
                "accuracy"
            ],

        "precision":
            test_metrics[
                "precision"
            ],

        "recall":
            test_metrics[
                "recall"
            ],

        "macro_f1":
            test_metrics[
                "macro_f1"
            ],

        "auc_roc":
            test_metrics[
                "auc_roc"
            ],

        "auc_pr":
            test_metrics[
                "auc_pr"
            ],

        "brier_score":
            test_metrics[
                "brier_score"
            ],


        "tn":
            test_metrics["tn"],

        "fp":
            test_metrics["fp"],

        "fn":
            test_metrics["fn"],

        "tp":
            test_metrics["tp"],
    }


    all_results.append(
        result
    )


    # --------------------------------------------------------
    # Store row-level test predictions
    # --------------------------------------------------------

    seed_predictions = pd.DataFrame({

        "seed":
            seed,

        "row_index":
            np.arange(
                len(y_test)
            ),

        "y_true":
            y_test,

        "xgb_probability":
            xgb_test_probability,

        "lgbm_probability":
            lgbm_test_probability,

        "ensemble_probability":
            ensemble_test_probability,

        "ensemble_prediction":
            (
                ensemble_test_probability
                >= THRESHOLD
            ).astype(int),

        "xgb_weight":
            XGB_WEIGHT,

        "lgbm_weight":
            LGBM_WEIGHT,

        "threshold":
            THRESHOLD,

        "model_cell":
            "Cell A",

        "attendance_decay":
            False,
    })


    all_predictions.append(
        seed_predictions
    )


    print(
        f"Test Accuracy: "
        f"{test_metrics['accuracy']:.6f}"
    )

    print(
        f"Test Precision: "
        f"{test_metrics['precision']:.6f}"
    )

    print(
        f"Test Recall: "
        f"{test_metrics['recall']:.6f}"
    )

    print(
        f"Test Macro-F1: "
        f"{test_metrics['macro_f1']:.6f}"
    )

    print(
        f"Test ROC-AUC: "
        f"{test_metrics['auc_roc']:.6f}"
    )

    print(
        f"Test PR-AUC: "
        f"{test_metrics['auc_pr']:.6f}"
    )

    print(
        f"Test Brier: "
        f"{test_metrics['brier_score']:.6f}"
    )


# ============================================================
# 12. COMBINE RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)

predictions_df = pd.concat(
    all_predictions,
    ignore_index=True
)


# ============================================================
# 13. MEAN ± SD
# ============================================================

print(
    "\n[7/8] Calculating ten-seed "
    "mean ± SD..."
)

METRICS = [
    "accuracy",
    "precision",
    "recall",
    "macro_f1",
    "auc_roc",
    "auc_pr",
    "brier_score",
]


summary_rows = []


for metric in METRICS:

    values = (
        results_df[metric]
        .astype(float)
    )

    summary_rows.append({

        "model_cell":
            "Cell A",

        "attendance_decay":
            False,

        "xgb_weight":
            XGB_WEIGHT,

        "lgbm_weight":
            LGBM_WEIGHT,

        "threshold":
            THRESHOLD,

        "metric":
            metric,

        "mean":
            values.mean(),

        "sd":
            values.std(
                ddof=1
            ),

        "n_seeds":
            len(values),
    })


for metric in [
    "tn",
    "fp",
    "fn",
    "tp",
]:

    values = (
        results_df[metric]
        .astype(float)
    )

    summary_rows.append({

        "model_cell":
            "Cell A",

        "attendance_decay":
            False,

        "xgb_weight":
            XGB_WEIGHT,

        "lgbm_weight":
            LGBM_WEIGHT,

        "threshold":
            THRESHOLD,

        "metric":
            metric.upper(),

        "mean":
            values.mean(),

        "sd":
            values.std(
                ddof=1
            ),

        "n_seeds":
            len(values),
    })


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# 14. SAVE OUTPUTS
# ============================================================

print(
    "\n[8/8] Saving Cell A artifacts..."
)

results_df.to_csv(
    OUT_RESULTS,
    index=False
)

summary_df.to_csv(
    OUT_SUMMARY,
    index=False
)

predictions_df.to_csv(
    OUT_PREDICTIONS,
    index=False
)


audit = {

    "model_cell":
        "Cell A",

    "description":
        (
            "Fixed 50:50 XGBoost-LightGBM "
            "ensemble without attendance_decay."
        ),

    "seeds":
        SEEDS,

    "n_seeds":
        len(SEEDS),

    "xgb_weight":
        XGB_WEIGHT,

    "lgbm_weight":
        LGBM_WEIGHT,

    "threshold":
        THRESHOLD,

    "excluded_feature":
        EXCLUDED_FEATURE,

    "features":
        CELL_A_FEATURES,

    "n_features":
        len(CELL_A_FEATURES),

    "train_rows":
        len(X_train),

    "validation_rows":
        len(X_validation),

    "test_rows":
        len(X_test),

    "xgb_parameters":
        XGB_PARAMS,

    "lgbm_parameters":
        LGBM_PARAMS,

    "protocol_note":
        (
            "Model parameters copied from the original "
            "verified Step 23 experiment. Cell A uses "
            "fixed 50:50 weighting and does not perform "
            "validation-based ensemble-weight selection."
        ),
}


with open(
    OUT_AUDIT,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        audit,
        file,
        indent=2
    )


# ============================================================
# 15. DISPLAY FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 75)
print("CELL A PRIMARY — TEN-SEED SUMMARY")
print("=" * 75)

for metric in METRICS:

    row = summary_df[
        summary_df["metric"]
        == metric
    ].iloc[0]

    print(
        f"{metric:15s}: "
        f"{row['mean']:.6f} ± "
        f"{row['sd']:.6f}"
    )


print("\nMean confusion counts:")

for metric in [
    "TN",
    "FP",
    "FN",
    "TP",
]:

    row = summary_df[
        summary_df["metric"]
        == metric
    ].iloc[0]

    print(
        f"{metric}: "
        f"{row['mean']:.4f} ± "
        f"{row['sd']:.4f}"
    )


print("\nFiles created:")

print(
    OUT_RESULTS
)

print(
    OUT_SUMMARY
)

print(
    OUT_PREDICTIONS
)

print(
    OUT_AUDIT
)


# ============================================================
# 16. FINAL VALIDATION
# ============================================================

assert len(results_df) == 10

assert results_df[
    "seed"
].nunique() == 10

assert predictions_df[
    "seed"
].nunique() == 10

assert (
    predictions_df
    .groupby("seed")
    .size()
    .eq(110)
    .all()
)

assert np.isfinite(
    results_df[METRICS]
).all().all()


print("\n")
print("=" * 75)
print("STEP 23A COMPLETED SUCCESSFULLY")
print("=" * 75)

print(
    "\n✓ Exact original Step 23 model parameters used"
)

print(
    "✓ attendance_decay excluded"
)

print(
    "✓ Fixed 50:50 XGBoost/LightGBM ensemble"
)

print(
    "✓ Ten seeds: 42–51"
)

print(
    "✓ Threshold = 0.50"
)

print(
    "✓ Test predictions saved"
)

print(
    "✓ Existing Cell C artifacts were not overwritten"
)
