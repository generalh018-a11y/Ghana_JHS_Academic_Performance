"""
STEP 25 — THREE-CELL ABLATION STUDY

Cell A:
    50:50 XGBoost–LightGBM ensemble WITHOUT attendance_decay

Cell B:
    50:50 XGBoost–LightGBM ensemble WITH attendance_decay

Cell C:
    Validation-selected XGBoost–LightGBM ensemble WITH attendance_decay

Protocol:
    - Student-level train/validation/test split is fixed.
    - Seeds 42–51 are evaluated.
    - Cell A and Cell B use a pre-specified 50:50 weight.
    - Cell C selects the ensemble weight using VALIDATION ONLY.
    - Test data are not used for model/weight selection.
    - No test result is used to modify the experiment.
"""

from pathlib import Path
import numpy as np
import pandas as pd

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

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SEEDS = list(range(42, 52))

# Standard ensemble condition
STANDARD_XGB_WEIGHT = 0.50
STANDARD_LGBM_WEIGHT = 0.50

# Candidate weights for validation-selected primary model
WEIGHT_GRID = np.round(np.arange(0.0, 1.01, 0.1), 10)

THRESHOLD = 0.50


# ============================================================
# FILES
# ============================================================

TRAIN_X_FILE = PROCESSED_DIR / "X_train.csv"
TRAIN_Y_FILE = PROCESSED_DIR / "y_train.csv"

VAL_X_FILE = PROCESSED_DIR / "X_validation.csv"
VAL_Y_FILE = PROCESSED_DIR / "y_validation.csv"

TEST_X_FILE = PROCESSED_DIR / "X_test.csv"
TEST_Y_FILE = PROCESSED_DIR / "y_test.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("STEP 25 — THREE-CELL ABLATION STUDY")
print("=" * 75)

print("\nLoading canonical model datasets...")

X_train = pd.read_csv(TRAIN_X_FILE)
y_train = pd.read_csv(TRAIN_Y_FILE).iloc[:, 0].astype(int)

X_val = pd.read_csv(VAL_X_FILE)
y_val = pd.read_csv(VAL_Y_FILE).iloc[:, 0].astype(int)

X_test = pd.read_csv(TEST_X_FILE)
y_test = pd.read_csv(TEST_Y_FILE).iloc[:, 0].astype(int)

print(f"Training shape   : {X_train.shape}")
print(f"Validation shape : {X_val.shape}")
print(f"Test shape       : {X_test.shape}")

print("\nTarget distribution:")
print("Training:")
print(y_train.value_counts().sort_index())

print("\nValidation:")
print(y_val.value_counts().sort_index())

print("\nTest:")
print(y_test.value_counts().sort_index())


# ============================================================
# FEATURE SETS
# ============================================================

ALL_FEATURES = list(X_train.columns)

DECAY_FEATURE = "attendance_decay"

if DECAY_FEATURE not in ALL_FEATURES:
    raise ValueError(
        f"Required engineered feature '{DECAY_FEATURE}' was not found."
    )

FEATURES_WITH_DECAY = ALL_FEATURES.copy()

FEATURES_WITHOUT_DECAY = [
    col for col in ALL_FEATURES
    if col != DECAY_FEATURE
]

print("\nFeature configuration:")
print(f"Total features with decay    : {len(FEATURES_WITH_DECAY)}")
print(f"Total features without decay : {len(FEATURES_WITHOUT_DECAY)}")

print("\nRemoved feature for Cell A:")
print(f"  {DECAY_FEATURE}")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def build_xgb(seed):
    """Create XGBoost using the fixed canonical configuration."""
    return XGBClassifier(
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


def build_lgbm(seed):
    """Create LightGBM using the fixed canonical configuration."""
    return LGBMClassifier(
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


def calculate_metrics(y_true, probabilities, threshold=0.50):
    """Calculate classification and probability metrics."""
    
    predictions = (probabilities >= threshold).astype(int)

    accuracy = accuracy_score(y_true, predictions)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0
    )

    auc_roc = roc_auc_score(y_true, probabilities)

    auc_pr = average_precision_score(
        y_true,
        probabilities
    )

    brier = brier_score_loss(
        y_true,
        probabilities
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "macro_f1": macro_f1,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "brier_score": brier,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def train_models(Xtr, ytr, Xv, yv, Xt, seed):
    """Train XGBoost and LightGBM and return probabilities."""

    xgb = build_xgb(seed)
    lgbm = build_lgbm(seed)

    xgb.fit(
        Xtr,
        ytr,
        eval_set=[(Xv, yv)],
        verbose=False
    )

    lgbm.fit(
        Xtr,
        ytr,
        eval_set=[(Xv, yv)],
        callbacks=[]
    )

    xgb_val = xgb.predict_proba(Xv)[:, 1]
    lgbm_val = lgbm.predict_proba(Xv)[:, 1]

    xgb_test = xgb.predict_proba(Xt)[:, 1]
    lgbm_test = lgbm.predict_proba(Xt)[:, 1]

    return (
        xgb_val,
        lgbm_val,
        xgb_test,
        lgbm_test
    )


def select_validation_weight(
    y_true,
    xgb_probability,
    lgbm_probability
):
    """
    Select XGBoost weight using validation data only.

    Primary criterion:
        Macro-F1

    Tie breakers:
        1. Higher AUC-PR
        2. Lower Brier score
    """

    candidates = []

    for xgb_weight in WEIGHT_GRID:

        lgbm_weight = 1.0 - xgb_weight

        ensemble_probability = (
            xgb_weight * xgb_probability
            + lgbm_weight * lgbm_probability
        )

        metrics = calculate_metrics(
            y_true,
            ensemble_probability,
            THRESHOLD
        )

        candidates.append({
            "xgb_weight": xgb_weight,
            "lgbm_weight": lgbm_weight,
            "macro_f1": metrics["macro_f1"],
            "auc_pr": metrics["auc_pr"],
            "brier_score": metrics["brier_score"],
        })

    candidates_df = pd.DataFrame(candidates)

    candidates_df = candidates_df.sort_values(
        by=[
            "macro_f1",
            "auc_pr",
            "brier_score"
        ],
        ascending=[
            False,
            False,
            True
        ]
    ).reset_index(drop=True)

    best = candidates_df.iloc[0]

    return (
        float(best["xgb_weight"]),
        float(best["lgbm_weight"]),
        candidates_df
    )


# ============================================================
# STORAGE
# ============================================================

all_results = []
weight_results = []


# ============================================================
# TEN-SEED EXPERIMENT
# ============================================================

for seed in SEEDS:

    print("\n" + "=" * 75)
    print(f"SEED {seed}")
    print("=" * 75)

    # --------------------------------------------------------
    # CELL A — WITHOUT ATTENDANCE DECAY
    # --------------------------------------------------------

    print("\nCELL A — 50:50 WITHOUT attendance_decay")

    Xtr_A = X_train[FEATURES_WITHOUT_DECAY]
    Xv_A = X_val[FEATURES_WITHOUT_DECAY]
    Xt_A = X_test[FEATURES_WITHOUT_DECAY]

    (
        xgb_val_A,
        lgbm_val_A,
        xgb_test_A,
        lgbm_test_A
    ) = train_models(
        Xtr_A,
        y_train,
        Xv_A,
        y_val,
        Xt_A,
        seed
    )

    prob_test_A = (
        STANDARD_XGB_WEIGHT * xgb_test_A
        + STANDARD_LGBM_WEIGHT * lgbm_test_A
    )

    metrics_A = calculate_metrics(
        y_test,
        prob_test_A,
        THRESHOLD
    )

    all_results.append({
        "cell": "A",
        "cell_description": "50:50 ensemble without attendance_decay",
        "seed": seed,
        "xgb_weight": STANDARD_XGB_WEIGHT,
        "lgbm_weight": STANDARD_LGBM_WEIGHT,
        "attendance_decay_included": False,
        **metrics_A
    })

    print(
        f"Accuracy={metrics_A['accuracy']:.6f} | "
        f"Macro-F1={metrics_A['macro_f1']:.6f} | "
        f"AUC-ROC={metrics_A['auc_roc']:.6f}"
    )


    # --------------------------------------------------------
    # CELL B — WITH ATTENDANCE DECAY, STANDARD 50:50
    # --------------------------------------------------------

    print("\nCELL B — 50:50 WITH attendance_decay")

    Xtr_B = X_train[FEATURES_WITH_DECAY]
    Xv_B = X_val[FEATURES_WITH_DECAY]
    Xt_B = X_test[FEATURES_WITH_DECAY]

    (
        xgb_val_B,
        lgbm_val_B,
        xgb_test_B,
        lgbm_test_B
    ) = train_models(
        Xtr_B,
        y_train,
        Xv_B,
        y_val,
        Xt_B,
        seed
    )

    prob_test_B = (
        STANDARD_XGB_WEIGHT * xgb_test_B
        + STANDARD_LGBM_WEIGHT * lgbm_test_B
    )

    metrics_B = calculate_metrics(
        y_test,
        prob_test_B,
        THRESHOLD
    )

    all_results.append({
        "cell": "B",
        "cell_description": "50:50 ensemble with attendance_decay",
        "seed": seed,
        "xgb_weight": STANDARD_XGB_WEIGHT,
        "lgbm_weight": STANDARD_LGBM_WEIGHT,
        "attendance_decay_included": True,
        **metrics_B
    })

    print(
        f"Accuracy={metrics_B['accuracy']:.6f} | "
        f"Macro-F1={metrics_B['macro_f1']:.6f} | "
        f"AUC-ROC={metrics_B['auc_roc']:.6f}"
    )


    # --------------------------------------------------------
    # CELL C — VALIDATION-SELECTED PRIMARY MODEL
    # --------------------------------------------------------

    print("\nCELL C — VALIDATION-SELECTED ENSEMBLE WITH attendance_decay")

    (
        selected_xgb_weight,
        selected_lgbm_weight,
        validation_search
    ) = select_validation_weight(
        y_val,
        xgb_val_B,
        lgbm_val_B
    )

    selected_test_probability = (
        selected_xgb_weight * xgb_test_B
        + selected_lgbm_weight * lgbm_test_B
    )

    metrics_C = calculate_metrics(
        y_test,
        selected_test_probability,
        THRESHOLD
    )

    all_results.append({
        "cell": "C",
        "cell_description": "validation-selected ensemble with attendance_decay",
        "seed": seed,
        "xgb_weight": selected_xgb_weight,
        "lgbm_weight": selected_lgbm_weight,
        "attendance_decay_included": True,
        **metrics_C
    })

    weight_results.append(
        validation_search.assign(
            seed=seed,
            selected=validation_search.index == 0
        )
    )

    print(
        f"Selected weights: "
        f"XGB={selected_xgb_weight:.2f}, "
        f"LGBM={selected_lgbm_weight:.2f}"
    )

    print(
        f"Accuracy={metrics_C['accuracy']:.6f} | "
        f"Macro-F1={metrics_C['macro_f1']:.6f} | "
        f"AUC-ROC={metrics_C['auc_roc']:.6f}"
    )


# ============================================================
# SAVE RAW RESULTS
# ============================================================

results_df = pd.DataFrame(all_results)

weights_df = pd.concat(
    weight_results,
    ignore_index=True
)


raw_results_file = (
    PROCESSED_DIR /
    "three_cell_ablation_results_by_seed.csv"
)

weights_file = (
    PROCESSED_DIR /
    "three_cell_ablation_validation_weight_search.csv"
)

results_df.to_csv(
    raw_results_file,
    index=False
)

weights_df.to_csv(
    weights_file,
    index=False
)


# ============================================================
# MEAN ± SD SUMMARY
# ============================================================

metric_columns = [
    "accuracy",
    "precision",
    "recall",
    "macro_f1",
    "auc_roc",
    "auc_pr",
    "brier_score",
    "TN",
    "FP",
    "FN",
    "TP",
]


summary_rows = []

for cell, group in results_df.groupby("cell"):

    row = {
        "cell": cell,
        "cell_description": group["cell_description"].iloc[0],
        "n_seeds": len(group),
        "xgb_weight_mean": group["xgb_weight"].mean(),
        "xgb_weight_sd": group["xgb_weight"].std(ddof=1),
        "lgbm_weight_mean": group["lgbm_weight"].mean(),
        "lgbm_weight_sd": group["lgbm_weight"].std(ddof=1),
    }

    for metric in metric_columns:

        row[f"{metric}_mean"] = group[metric].mean()
        row[f"{metric}_sd"] = group[metric].std(ddof=1)

    summary_rows.append(row)


summary_df = pd.DataFrame(summary_rows)

summary_file = (
    PROCESSED_DIR /
    "three_cell_ablation_summary_mean_sd.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# PRINT FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 75)
print("STEP 25 — THREE-CELL ABLATION SUMMARY")
print("=" * 75)

for _, row in summary_df.iterrows():

    print("\n" + row["cell"])
    print(row["cell_description"])

    print(
        f"Accuracy : "
        f"{row['accuracy_mean']:.6f} ± "
        f"{row['accuracy_sd']:.6f}"
    )

    print(
        f"Precision: "
        f"{row['precision_mean']:.6f} ± "
        f"{row['precision_sd']:.6f}"
    )

    print(
        f"Recall   : "
        f"{row['recall_mean']:.6f} ± "
        f"{row['recall_sd']:.6f}"
    )

    print(
        f"Macro-F1 : "
        f"{row['macro_f1_mean']:.6f} ± "
        f"{row['macro_f1_sd']:.6f}"
    )

    print(
        f"AUC-ROC  : "
        f"{row['auc_roc_mean']:.6f} ± "
        f"{row['auc_roc_sd']:.6f}"
    )

    print(
        f"AUC-PR   : "
        f"{row['auc_pr_mean']:.6f} ± "
        f"{row['auc_pr_sd']:.6f}"
    )

    print(
        f"Brier    : "
        f"{row['brier_score_mean']:.6f} ± "
        f"{row['brier_score_sd']:.6f}"
    )

    print(
        f"XGB weight: "
        f"{row['xgb_weight_mean']:.4f} ± "
        f"{row['xgb_weight_sd']:.4f}"
    )

    print(
        f"LGBM weight: "
        f"{row['lgbm_weight_mean']:.4f} ± "
        f"{row['lgbm_weight_sd']:.4f}"
    )


print("\n" + "=" * 75)
print("STEP 25 COMPLETED SUCCESSFULLY")
print("=" * 75)

print("\nCreated:")
print(raw_results_file)
print(weights_file)
print(summary_file)

print("\n✓ Cell A: 50:50 without attendance_decay")
print("✓ Cell B: 50:50 with attendance_decay")
print("✓ Cell C: validation-selected ensemble with attendance_decay")
print("✓ Ten seeds: 42–51")
print("✓ Weight selection performed using validation data only")
print("✓ Test data not used for weight selection")
print("✓ Mean ± SD summary generated")

print("\nSTOP HERE.")
print("Do NOT modify any ablation cell based on test performance.")