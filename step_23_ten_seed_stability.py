from pathlib import Path

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


# ============================================================
# STEP 23
# TEN-SEED STABILITY EXPERIMENT
# ============================================================

print("=" * 75)
print("STEP 23 — TEN-SEED STABILITY EXPERIMENT")
print("=" * 75)


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_FILE = DATA_DIR / "train_student_level.csv"
VALIDATION_FILE = DATA_DIR / "validation_student_level.csv"
TEST_FILE = DATA_DIR / "test_student_level.csv"


# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

train = pd.read_csv(TRAIN_FILE)
validation = pd.read_csv(VALIDATION_FILE)
test = pd.read_csv(TEST_FILE)


print("\nDataset sizes:")
print(f"Train:      {len(train)}")
print(f"Validation: {len(validation)}")
print(f"Test:       {len(test)}")


# ------------------------------------------------------------
# 3. CANONICAL FEATURES
# ------------------------------------------------------------

FEATURES = [
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


# ------------------------------------------------------------
# 4. LOAD CANONICAL FEATURE MATRICES
# ------------------------------------------------------------

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
).squeeze("columns").astype(int)


y_validation = pd.read_csv(
    DATA_DIR / "y_validation.csv"
).squeeze("columns").astype(int)


y_test = pd.read_csv(
    DATA_DIR / "y_test.csv"
).squeeze("columns").astype(int)


# ------------------------------------------------------------
# 5. VERIFY FEATURE STRUCTURE
# ------------------------------------------------------------

assert list(X_train.columns) == FEATURES
assert list(X_validation.columns) == FEATURES
assert list(X_test.columns) == FEATURES

assert len(X_train) == 318
assert len(X_validation) == 113
assert len(X_test) == 110


# ------------------------------------------------------------
# 6. EXPERIMENTAL PARAMETERS
# ------------------------------------------------------------

SEEDS = list(range(42, 52))

ENSEMBLE_WEIGHTS = np.round(
    np.arange(
        0.0,
        1.01,
        0.10
    ),
    2
)

THRESHOLD = 0.50


print("\nSeeds:")
print(SEEDS)

print("\nCandidate XGBoost weights:")
print(ENSEMBLE_WEIGHTS)

print("\nLightGBM weight = 1 - XGBoost weight")

print(
    f"\nClassification threshold: {THRESHOLD:.2f}"
)

print(
    "\nPrimary ensemble-selection criterion: "
    "Validation Macro-F1"
)


# ------------------------------------------------------------
# 7. METRIC FUNCTION
# ------------------------------------------------------------

def calculate_metrics(
    y_true,
    probability,
    threshold=0.50
):

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

        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


# ------------------------------------------------------------
# 8. STORAGE
# ------------------------------------------------------------

all_results = []
all_weight_results = []
all_predictions = []


# ------------------------------------------------------------
# 9. RUN TEN SEEDS
# ------------------------------------------------------------

for seed in SEEDS:

    print("\n")
    print("=" * 75)
    print(f"RUNNING SEED {seed}")
    print("=" * 75)


    # --------------------------------------------------------
    # 9.1 XGBOOST
    # --------------------------------------------------------

    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
    )


    # --------------------------------------------------------
    # 9.2 LIGHTGBM
    # --------------------------------------------------------

    lgbm_model = LGBMClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


    # --------------------------------------------------------
    # 9.3 TRAIN
    # --------------------------------------------------------

    xgb_model.fit(
        X_train,
        y_train
    )

    lgbm_model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # 9.4 VALIDATION PROBABILITIES
    # --------------------------------------------------------

    xgb_val_probability = (
        xgb_model.predict_proba(
            X_validation
        )[:, 1]
    )

    lgbm_val_probability = (
        lgbm_model.predict_proba(
            X_validation
        )[:, 1]
    )


    # --------------------------------------------------------
    # 9.5 SEARCH ENSEMBLE WEIGHTS ON VALIDATION ONLY
    # --------------------------------------------------------

    weight_results = []

    for xgb_weight in ENSEMBLE_WEIGHTS:

        lgbm_weight = (
            1.0 - xgb_weight
        )

        ensemble_val_probability = (
            xgb_weight
            * xgb_val_probability
            +
            lgbm_weight
            * lgbm_val_probability
        )

        metrics = calculate_metrics(
            y_validation,
            ensemble_val_probability,
            THRESHOLD
        )

        row = {
            "seed": seed,
            "xgb_weight": xgb_weight,
            "lgbm_weight": lgbm_weight,
        }

        row.update(metrics)

        weight_results.append(row)


    weight_results_df = pd.DataFrame(
        weight_results
    )


    # --------------------------------------------------------
    # 9.6 DETERMINISTIC WEIGHT SELECTION
    #
    # Priority:
    # 1. Macro-F1 higher is better
    # 2. AUC-PR higher is better
    # 3. Brier lower is better
    # 4. XGBoost weight higher is preferred only as the
    #    final deterministic tie-breaker.
    # --------------------------------------------------------

    weight_results_df = (
        weight_results_df
        .sort_values(
            by=[
                "macro_f1",
                "auc_pr",
                "brier_score",
                "xgb_weight",
            ],
            ascending=[
                False,
                False,
                True,
                False,
            ]
        )
        .reset_index(drop=True)
    )


    best = weight_results_df.iloc[0]


    best_xgb_weight = float(
        best["xgb_weight"]
    )

    best_lgbm_weight = float(
        best["lgbm_weight"]
    )


    all_weight_results.append(
        weight_results_df
    )


    print(
        f"\nSelected weight for seed {seed}:"
    )

    print(
        f"XGBoost = {best_xgb_weight:.2f}"
    )

    print(
        f"LightGBM = {best_lgbm_weight:.2f}"
    )

    print(
        f"Validation Macro-F1 = "
        f"{best['macro_f1']:.6f}"
    )


    # --------------------------------------------------------
    # 9.7 TEST PROBABILITIES
    #
    # The weight is now frozen.
    # --------------------------------------------------------

    xgb_test_probability = (
        xgb_model.predict_proba(
            X_test
        )[:, 1]
    )

    lgbm_test_probability = (
        lgbm_model.predict_proba(
            X_test
        )[:, 1]
    )


    # --------------------------------------------------------
    # 9.8 LOCKED ENSEMBLE
    # --------------------------------------------------------

    ensemble_test_probability = (
        best_xgb_weight
        * xgb_test_probability
        +
        best_lgbm_weight
        * lgbm_test_probability
    )


    # --------------------------------------------------------
    # 9.9 FINAL TEST METRICS FOR THIS SEED
    # --------------------------------------------------------

    test_metrics = calculate_metrics(
        y_test,
        ensemble_test_probability,
        THRESHOLD
    )


    result_row = {
        "model": "XGBoost-LightGBM Ensemble",
        "seed": seed,
        "xgb_weight": best_xgb_weight,
        "lgbm_weight": best_lgbm_weight,
        "threshold": THRESHOLD,
    }

    result_row.update(
        test_metrics
    )

    all_results.append(
        result_row
    )


    # --------------------------------------------------------
    # 9.10 SAVE PREDICTIONS FOR THIS SEED
    # --------------------------------------------------------

    predictions = pd.DataFrame(
        {
            "seed": seed,
            "y_true": y_test,
            "xgb_probability": xgb_test_probability,
            "lgbm_probability": lgbm_test_probability,
            "ensemble_probability": ensemble_test_probability,
            "ensemble_prediction": (
                ensemble_test_probability
                >= THRESHOLD
            ).astype(int),
        }
    )

    all_predictions.append(
        predictions
    )


    print(
        f"Test Accuracy = "
        f"{test_metrics['accuracy']:.6f}"
    )

    print(
        f"Test Macro-F1 = "
        f"{test_metrics['macro_f1']:.6f}"
    )

    print(
        f"Test AUC-ROC = "
        f"{test_metrics['auc_roc']:.6f}"
    )


# ------------------------------------------------------------
# 10. COMBINE RESULTS
# ------------------------------------------------------------

results_df = pd.DataFrame(
    all_results
)

weight_search_df = pd.concat(
    all_weight_results,
    ignore_index=True
)

predictions_df = pd.concat(
    all_predictions,
    ignore_index=True
)


# ------------------------------------------------------------
# 11. CALCULATE MEAN ± SD
# ------------------------------------------------------------

metric_columns = [
    "accuracy",
    "precision",
    "recall",
    "macro_f1",
    "auc_roc",
    "auc_pr",
    "brier_score",
]


summary_rows = []

for metric in metric_columns:

    summary_rows.append(
        {
            "metric": metric,
            "mean": results_df[metric].mean(),
            "sd": results_df[metric].std(
                ddof=1
            ),
            "mean_minus_sd": (
                results_df[metric].mean()
                - results_df[metric].std(
                    ddof=1
                )
            ),
            "mean_plus_sd": (
                results_df[metric].mean()
                + results_df[metric].std(
                    ddof=1
                )
            ),
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)


# ------------------------------------------------------------
# 12. WEIGHT STABILITY
# ------------------------------------------------------------

weight_summary = pd.DataFrame(
    [
        {
            "parameter": "xgb_weight",
            "mean": results_df[
                "xgb_weight"
            ].mean(),
            "sd": results_df[
                "xgb_weight"
            ].std(ddof=1),
            "min": results_df[
                "xgb_weight"
            ].min(),
            "max": results_df[
                "xgb_weight"
            ].max(),
        },

        {
            "parameter": "lgbm_weight",
            "mean": results_df[
                "lgbm_weight"
            ].mean(),
            "sd": results_df[
                "lgbm_weight"
            ].std(ddof=1),
            "min": results_df[
                "lgbm_weight"
            ].min(),
            "max": results_df[
                "lgbm_weight"
            ].max(),
        },
    ]
)


# ------------------------------------------------------------
# 13. DISPLAY SEED RESULTS
# ------------------------------------------------------------

print("\n")
print("=" * 75)
print("TEN-SEED TEST RESULTS")
print("=" * 75)

display_columns = [
    "seed",
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


# ------------------------------------------------------------
# 14. DISPLAY MEAN ± SD
# ------------------------------------------------------------

print("\n")
print("=" * 75)
print("TEN-SEED SUMMARY — MEAN ± SD")
print("=" * 75)

for _, row in summary_df.iterrows():

    print(
        f"{row['metric']:15s}: "
        f"{row['mean']:.6f} ± "
        f"{row['sd']:.6f}"
    )


# ------------------------------------------------------------
# 15. DISPLAY WEIGHT STABILITY
# ------------------------------------------------------------

print("\n")
print("=" * 75)
print("ENSEMBLE WEIGHT STABILITY")
print("=" * 75)

print(
    weight_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ------------------------------------------------------------
# 16. CONFUSION COUNTS ACROSS SEEDS
# ------------------------------------------------------------

print("\n")
print("=" * 75)
print("CONFUSION COUNTS ACROSS SEEDS")
print("=" * 75)

confusion_summary = results_df[
    [
        "seed",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
]

print(
    confusion_summary.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 17. SAVE RESULTS
# ------------------------------------------------------------

results_file = (
    DATA_DIR
    / "ten_seed_ensemble_test_results.csv"
)

summary_file = (
    DATA_DIR
    / "ten_seed_ensemble_summary_mean_sd.csv"
)

weights_file = (
    DATA_DIR
    / "ten_seed_ensemble_weight_search.csv"
)

weight_summary_file = (
    DATA_DIR
    / "ten_seed_ensemble_weight_stability.csv"
)

predictions_file = (
    DATA_DIR
    / "ten_seed_ensemble_test_predictions.csv"
)


results_df.to_csv(
    results_file,
    index=False
)

summary_df.to_csv(
    summary_file,
    index=False
)

weight_search_df.to_csv(
    weights_file,
    index=False
)

weight_summary.to_csv(
    weight_summary_file,
    index=False
)

predictions_df.to_csv(
    predictions_file,
    index=False
)


# ------------------------------------------------------------
# 18. FINAL VALIDATION
# ------------------------------------------------------------

assert len(results_df) == 10

assert len(
    summary_df
) == len(metric_columns)

assert (
    results_df["seed"].nunique()
    == 10
)

assert (
    predictions_df["seed"].nunique()
    == 10
)

assert (
    predictions_df.groupby("seed").size()
    .eq(110)
    .all()
)

assert np.isfinite(
    results_df[metric_columns]
).all().all()


# ------------------------------------------------------------
# 19. FINAL MESSAGE
# ------------------------------------------------------------

print("\n")
print("=" * 75)
print("STEP 23 COMPLETED SUCCESSFULLY")
print("=" * 75)

print("\nFiles created:")

print(results_file)
print(summary_file)
print(weights_file)
print(weight_summary_file)
print(predictions_file)

print("\n✓ Ten independent seeds completed")
print("✓ Ensemble weight selected using validation only")
print("✓ Test set evaluated only after weight selection")
print("✓ Mean ± SD calculated across ten seeds")
print("✓ Ensemble-weight stability recorded")
print("✓ Per-seed test predictions saved")

print("\nIMPORTANT:")
print("Do NOT modify or select results based on the test performance.")
print("Do NOT write the final Chapter 4 result yet.")
print("The naive comparator and ablation are still required.")