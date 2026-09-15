from __future__ import annotations

"""
STEP 23B — REPEATED GROUPED-SPLIT × MODEL-SEED VARIANCE

Primary configuration:
    Cell A: 50:50 XGBoost–LightGBM ensemble WITHOUT attendance_decay.

Design:
    10 grouped student-level partitions × 10 model seeds.
    Split seeds: 42, 52, ..., 132.
    Model seeds: 42, ..., 51.

For each partition:
    1. Draw the test partition by GroupShuffleSplit using the split seed.
    2. Draw validation from the remaining training/validation pool using
       GroupShuffleSplit with split_seed + 1 and test_size=0.25.
    3. Fit preprocessing/encoding separately within that partition.
    4. Fit XGBoost and LightGBM for each model seed.
    5. Evaluate the frozen 50:50 primary ensemble on the held-out test set.

Legacy reconciliation:
    Partition 0 (split seed 42; validation seed 43) is additionally run using
    the former attendance-decay + validation-selected configuration. Its ten
    model-seed results are compared with step_23's canonical CSV to 1e-9.
    This preserves the old reproducibility anchor while making Cell A primary.

Outputs are aggregate-only and contain no student-level rows.
"""

from pathlib import Path
import json
import math
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Repository-relative paths
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
DATA_PATH = DATA_DIR / "term_level_early_warning_dataset.csv"
LEGACY_RESULTS_PATH = ROOT / "data" / "processed" / "ten_seed_ensemble_test_results.csv"
OUT_DIR = ROOT / "results" / "variance"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Experiment constants
# -----------------------------------------------------------------------------
SPLIT_SEEDS = [42, 52, 62, 72, 82, 92, 102, 112, 122, 132]
MODEL_SEEDS = list(range(42, 52))
TEST_FRACTION = 0.20
VALIDATION_FRACTION = 0.25
THRESHOLD = 0.50
WEIGHT_GRID = np.round(np.arange(0.0, 1.01, 0.1), 10)
RECON_TOL = 1e-9
EXPECTED_ROWS = 541

# Primary Cell A — no attendance decay, fixed 50:50.
PRIMARY_NUMERIC = [
    "score_t1",
    "score_t2",
    "attendance_t1",
    "attendance_t2",
    "age_years",
]
PRIMARY_CATEGORICAL = ["gender", "class_level", "academic_year"]
PRIMARY_COLUMNS = [
    "score_t1",
    "score_t2",
    "attendance_t1",
    "attendance_t2",
    "age_years",
    "gender_F",
    "gender_M",
    "class_level_JHS1",
    "class_level_JHS2",
    "academic_year_2022-23",
    "academic_year_2023-24",
]

# Former Cell C — only for reconciliation of partition 0 against step_23.
LEGACY_NUMERIC = [
    "score_t1",
    "score_t2",
    "attendance_t1",
    "attendance_t2",
    "attendance_decay",
    "age_years",
]
LEGACY_COLUMNS = [
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

# -----------------------------------------------------------------------------
# Fixed model-development configuration from Chapter 3
# -----------------------------------------------------------------------------
XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    n_jobs=1,
    verbosity=0,
)

LGB_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary",
    n_jobs=1,
    verbosity=-1,
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def validate_dataset(df: pd.DataFrame) -> None:
    required = {
        "student_id",
        "class_level",
        "gender",
        "academic_year",
        "score_t1",
        "score_t2",
        "attendance_t1",
        "attendance_t2",
        "attendance_decay",
        "age_years",
        "at_risk",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        fail(f"Dataset missing required columns: {missing}")
    if len(df) != EXPECTED_ROWS:
        fail(f"Expected {EXPECTED_ROWS} rows, found {len(df)}")
    if df["student_id"].isna().any():
        fail("student_id contains missing values")
    if df["at_risk"].isna().any():
        fail("at_risk contains missing values")
    if not set(pd.Series(df["at_risk"]).astype(int).unique()).issubset({0, 1}):
        fail("at_risk is not binary")
    if df.duplicated(["student_id", "academic_year"]).any():
        fail("Duplicate student-year observations detected")
    for c in PRIMARY_NUMERIC + ["attendance_decay"]:
        if df[c].isna().any():
            fail(f"Missing values found in {c}")


def split_partition(df: pd.DataFrame, split_seed: int):
    outer = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_FRACTION,
        random_state=split_seed,
    )
    trval_idx, test_idx = next(
        outer.split(df, df["at_risk"], groups=df["student_id"])
    )
    trval = df.iloc[trval_idx].copy().reset_index(drop=True)
    test = df.iloc[test_idx].copy().reset_index(drop=True)

    inner = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_FRACTION,
        random_state=split_seed + 1,
    )
    train_idx, val_idx = next(
        inner.split(trval, trval["at_risk"], groups=trval["student_id"])
    )
    train = trval.iloc[train_idx].copy().reset_index(drop=True)
    validation = trval.iloc[val_idx].copy().reset_index(drop=True)

    train_students = set(train.student_id)
    val_students = set(validation.student_id)
    test_students = set(test.student_id)
    if train_students & val_students:
        fail(f"Student overlap train/validation for split seed {split_seed}")
    if train_students & test_students:
        fail(f"Student overlap train/test for split seed {split_seed}")
    if val_students & test_students:
        fail(f"Student overlap validation/test for split seed {split_seed}")

    for name, part in [("train", train), ("validation", validation), ("test", test)]:
        if part["at_risk"].nunique() != 2:
            return None, {
                "split_seed": split_seed,
                "status": "EXCLUDED",
                "reason": f"{name} partition lacks both target classes",
                "train_rows": len(train),
                "validation_rows": len(validation),
                "test_rows": len(test),
            }

    return (train, validation, test), {
        "split_seed": split_seed,
        "validation_seed": split_seed + 1,
        "status": "PASS",
        "reason": "",
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "train_students": train.student_id.nunique(),
        "validation_students": validation.student_id.nunique(),
        "test_students": test.student_id.nunique(),
        "train_at_risk": int(train.at_risk.sum()),
        "validation_at_risk": int(validation.at_risk.sum()),
        "test_at_risk": int(test.at_risk.sum()),
    }


def encode(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, use_decay: bool):
    numeric = LEGACY_NUMERIC if use_decay else PRIMARY_NUMERIC
    categorical = PRIMARY_CATEGORICAL

    def make(part: pd.DataFrame) -> pd.DataFrame:
        X = part[numeric + categorical].copy()
        return pd.get_dummies(X, columns=categorical, dtype=float)

    X_train = make(train)
    X_val = make(validation)
    X_test = make(test)
    columns = LEGACY_COLUMNS if use_decay else PRIMARY_COLUMNS
    X_train = X_train.reindex(columns=columns, fill_value=0.0)
    X_val = X_val.reindex(columns=columns, fill_value=0.0)
    X_test = X_test.reindex(columns=columns, fill_value=0.0)

    if list(X_train.columns) != columns or list(X_val.columns) != columns or list(X_test.columns) != columns:
        fail("Canonical column contract failed")
    if X_train.isna().any().any() or X_val.isna().any().any() or X_test.isna().any().any():
        fail("Encoded feature matrix contains missing values")
    return X_train, X_val, X_test


def fit_models(X_train, y_train, seed: int):
    xgb = XGBClassifier(**XGB_PARAMS, random_state=seed)
    lgbm = LGBMClassifier(**LGB_PARAMS, random_state=seed)
    xgb.fit(X_train, y_train)
    lgbm.fit(X_train, y_train)
    return xgb, lgbm


def metric_row(y_true, probability, prediction):
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, probability)),
        "auc_pr": float(average_precision_score(y_true, probability)),
        "brier_score": float(np.mean((np.asarray(y_true) - probability) ** 2)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def validation_score(y_val, probability):
    pred = (probability >= THRESHOLD).astype(int)
    return {
        "macro_f1": f1_score(y_val, pred, average="macro", zero_division=0),
        "auc_pr": average_precision_score(y_val, probability),
        "brier": np.mean((np.asarray(y_val) - probability) ** 2),
    }


def select_weight(y_val, p_xgb, p_lgbm):
    candidates = []
    for w in WEIGHT_GRID:
        p = w * p_xgb + (1.0 - w) * p_lgbm
        s = validation_score(y_val, p)
        candidates.append((float(w), s))
    # Maximise Macro-F1; tie-break by higher AUPR, lower Brier, then higher XGB weight.
    candidates.sort(
        key=lambda item: (
            item[1]["macro_f1"],
            item[1]["auc_pr"],
            -item[1]["brier"],
            item[0],
        ),
        reverse=True,
    )
    return candidates[0][0]


def run_primary_partition(train, validation, test, split_seed, model_seed):
    X_train, X_val, X_test = encode(train, validation, test, use_decay=False)
    y_train = train["at_risk"].astype(int).to_numpy()
    y_test = test["at_risk"].astype(int).to_numpy()
    xgb, lgbm = fit_models(X_train, y_train, model_seed)
    p_xgb = xgb.predict_proba(X_test)[:, 1]
    p_lgbm = lgbm.predict_proba(X_test)[:, 1]
    probability = 0.5 * p_xgb + 0.5 * p_lgbm
    prediction = (probability >= THRESHOLD).astype(int)
    m = metric_row(y_test, probability, prediction)
    return {
        "configuration": "Cell_A_primary_50_50_without_attendance_decay",
        "split_seed": split_seed,
        "validation_seed": split_seed + 1,
        "model_seed": model_seed,
        "xgb_weight": 0.5,
        "lgbm_weight": 0.5,
        **m,
    }


def run_legacy_reconciliation_partition(train, validation, test, split_seed, model_seed):
    X_train, X_val, X_test = encode(train, validation, test, use_decay=True)
    y_train = train["at_risk"].astype(int).to_numpy()
    y_val = validation["at_risk"].astype(int).to_numpy()
    y_test = test["at_risk"].astype(int).to_numpy()
    xgb, lgbm = fit_models(X_train, y_train, model_seed)
    p_xgb_val = xgb.predict_proba(X_val)[:, 1]
    p_lgbm_val = lgbm.predict_proba(X_val)[:, 1]
    w = select_weight(y_val, p_xgb_val, p_lgbm_val)
    p_xgb_test = xgb.predict_proba(X_test)[:, 1]
    p_lgbm_test = lgbm.predict_proba(X_test)[:, 1]
    probability = w * p_xgb_test + (1.0 - w) * p_lgbm_test
    prediction = (probability >= THRESHOLD).astype(int)
    m = metric_row(y_test, probability, prediction)
    return {
        "split_seed": split_seed,
        "validation_seed": split_seed + 1,
        "model_seed": model_seed,
        "xgb_weight": float(w),
        "lgbm_weight": float(1.0 - w),
        **m,
    }


def build_legacy_canonical_results() -> pd.DataFrame:
    """Re-run the original Step 23 experiment on its exact canonical matrices."""
    required_files = [
        DATA_DIR / "X_train.csv",
        DATA_DIR / "X_validation.csv",
        DATA_DIR / "X_test.csv",
        DATA_DIR / "y_train.csv",
        DATA_DIR / "y_validation.csv",
        DATA_DIR / "y_test.csv",
    ]
    missing = [str(p) for p in required_files if not p.exists()]
    if missing:
        fail("Canonical Step 23 matrices missing: " + "; ".join(missing))

    features = LEGACY_COLUMNS
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_validation = pd.read_csv(DATA_DIR / "X_validation.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns").astype(int)
    y_validation = pd.read_csv(DATA_DIR / "y_validation.csv").squeeze("columns").astype(int)
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns").astype(int)

    if list(X_train.columns) != features or list(X_validation.columns) != features or list(X_test.columns) != features:
        fail("Canonical Step 23 feature structure does not match the legacy 12-column contract")
    if len(X_train) != 318 or len(X_validation) != 113 or len(X_test) != 110:
        fail("Canonical Step 23 matrices do not have the expected 318/113/110 row counts")

    rows = []
    for seed in MODEL_SEEDS:
        legacy_xgb_params = XGB_PARAMS.copy()
        legacy_lgb_params = LGB_PARAMS.copy()
        legacy_xgb_params["n_jobs"] = -1
        legacy_lgb_params["n_jobs"] = -1
        xgb = XGBClassifier(**legacy_xgb_params, random_state=seed)
        lgbm = LGBMClassifier(**legacy_lgb_params, random_state=seed)
        xgb.fit(X_train, y_train)
        lgbm.fit(X_train, y_train)
        pxv = xgb.predict_proba(X_validation)[:, 1]
        plv = lgbm.predict_proba(X_validation)[:, 1]

        weight_rows = []
        for w in WEIGHT_GRID:
            p = w * pxv + (1.0 - w) * plv
            pred = (p >= THRESHOLD).astype(int)
            m = metric_row(y_validation, p, pred)
            m["brier_score"] = float(brier_score_loss(y_validation, p))
            weight_rows.append({"xgb_weight": float(w), "lgbm_weight": float(1.0-w), **m})
        wr = pd.DataFrame(weight_rows).sort_values(
            by=["macro_f1", "auc_pr", "brier_score", "xgb_weight"],
            ascending=[False, False, True, False],
        ).reset_index(drop=True)
        best = wr.iloc[0]
        w = float(best["xgb_weight"])
        pxt = xgb.predict_proba(X_test)[:, 1]
        plt = lgbm.predict_proba(X_test)[:, 1]
        p = w * pxt + (1.0-w) * plt
        pred = (p >= THRESHOLD).astype(int)
        m = metric_row(y_test, p, pred)
        m["brier_score"] = float(brier_score_loss(y_test, p))
        rows.append({"model_seed": seed, "xgb_weight": w, "lgbm_weight": 1.0-w, **m})
    return pd.DataFrame(rows).sort_values("model_seed").reset_index(drop=True)


def reconcile_legacy() -> tuple[bool, pd.DataFrame]:
    if not LEGACY_RESULTS_PATH.exists():
        fail(f"Canonical step_23 results not found for required reconciliation: {LEGACY_RESULTS_PATH}")
    old = pd.read_csv(LEGACY_RESULTS_PATH).rename(columns={"seed": "model_seed"})
    required = ["model_seed", "accuracy", "precision", "recall", "macro_f1", "auc_roc", "auc_pr", "brier_score", "xgb_weight", "lgbm_weight"]
    missing = [c for c in required if c not in old.columns]
    if missing:
        fail(f"Legacy reconciliation CSV missing columns: {missing}")
    old = old[old.model_seed.astype(int).isin(MODEL_SEEDS)].copy().sort_values("model_seed").reset_index(drop=True)
    new = build_legacy_canonical_results()
    max_diff = 0.0
    worst = None
    for c in required[1:]:
        a = old[c].astype(float).to_numpy()
        b = new[c].astype(float).to_numpy()
        diff = np.abs(a-b)
        i = int(np.argmax(diff))
        if diff[i] > max_diff:
            max_diff = float(diff[i])
            worst = (c, int(new.loc[i, "model_seed"]), float(a[i]), float(b[i]))
    print(f"LEGACY RECONCILIATION MAX ABS DIFF: {max_diff:.12g}")
    if max_diff > RECON_TOL:
        print(f"LEGACY RECONCILIATION: FAIL — worst={worst}")
        return False, new
    print("LEGACY RECONCILIATION: PASS")
    return True, new

def variance_summary(primary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "accuracy",
        "precision",
        "recall",
        "macro_f1",
        "auc_roc",
        "auc_pr",
        "brier_score",
    ]
    rows = []
    M = len(MODEL_SEEDS)
    K = primary.split_seed.nunique()
    for metric in metrics:
        per_split = primary.groupby("split_seed")[metric]
        means = per_split.mean()
        within_vars = per_split.var(ddof=1).fillna(0.0)
        within_var = float(within_vars.mean())
        mean_of_split_means = float(means.mean())
        observed_partition_mean_variance = float(means.var(ddof=1)) if K > 1 else 0.0
        # Method-of-moments random-effects component:
        # Var(split means) = between + within/M.
        between_var = max(0.0, observed_partition_mean_variance - within_var / M)
        total_var = between_var + within_var
        rows.append({
            "configuration": "Cell_A_primary_50_50_without_attendance_decay",
            "metric": metric,
            "n_partitions": K,
            "n_model_seeds_per_partition": M,
            "mean": mean_of_split_means,
            "within_partition_variance": within_var,
            "within_partition_sd": math.sqrt(within_var),
            "observed_partition_mean_variance": observed_partition_mean_variance,
            "observed_partition_mean_sd": math.sqrt(observed_partition_mean_variance),
            "between_partition_variance_component": between_var,
            "between_partition_sd": math.sqrt(between_var),
            "total_variance_component": total_var,
            "total_sd": math.sqrt(total_var),
        })
    return pd.DataFrame(rows)


def main():
    print("=" * 79)
    print("STEP 23B — REPEATED GROUPED-SPLIT × MODEL-SEED VARIANCE")
    print("=" * 79)
    print(f"Repository root : {ROOT}")
    print(f"Dataset         : {DATA_PATH}")
    print(f"Split seeds     : {SPLIT_SEEDS}")
    print(f"Model seeds     : {MODEL_SEEDS}")
    print("Primary         : Cell A, 50:50, WITHOUT attendance_decay")
    print(f"Script path     : {Path(__file__).resolve()}")

    expected_data_dir = ROOT / "data" / "processed"
    if not expected_data_dir.exists():
        fail(f"Repository root appears incorrect: expected data folder at {expected_data_dir}")

    if not DATA_PATH.exists():
        fail(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    validate_dataset(df)
    print(f"Dataset integrity: PASS ({len(df)} rows, {df.student_id.nunique()} students)")

    primary_rows = []
    legacy_rows = []
    manifests = []

    for partition_index, split_seed in enumerate(SPLIT_SEEDS):
        parts, manifest = split_partition(df, split_seed)
        manifest["partition_index"] = partition_index
        manifests.append(manifest)
        print("-" * 79)
        print(
            f"PARTITION {partition_index} | split_seed={split_seed} | "
            f"validation_seed={split_seed + 1} | status={manifest['status']}"
        )
        if parts is None:
            print(f"Excluded: {manifest['reason']}")
            continue

        train, validation, test = parts
        print(
            f"Rows train/val/test: {len(train)}/{len(validation)}/{len(test)} | "
            f"students: {train.student_id.nunique()}/{validation.student_id.nunique()}/{test.student_id.nunique()}"
        )

        for model_seed in MODEL_SEEDS:
            row = run_primary_partition(train, validation, test, split_seed, model_seed)
            row["partition_index"] = partition_index
            primary_rows.append(row)
            print(
                f"  seed {model_seed}: Macro-F1={row['macro_f1']:.6f} "
                f"ROC-AUC={row['auc_roc']:.6f} Brier={row['brier_score']:.6f}"
            )


    primary = pd.DataFrame(primary_rows)
    if primary.empty:
        fail("No valid partitions were produced")
    if primary.split_seed.nunique() < 2:
        fail("Fewer than two valid partitions; split variance cannot be estimated")

    manifest_df = pd.DataFrame(manifests)
    primary = primary.sort_values(["partition_index", "model_seed"]).reset_index(drop=True)
    decomposition = variance_summary(primary)

    per_partition = (
        primary.groupby("split_seed")
        .agg(
            partition_index=("partition_index", "first"),
            model_seed_count=("model_seed", "nunique"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_sd=("accuracy", "std"),
            precision_mean=("precision", "mean"),
            precision_sd=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_sd=("recall", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", "std"),
            auc_roc_mean=("auc_roc", "mean"),
            auc_roc_sd=("auc_roc", "std"),
            auc_pr_mean=("auc_pr", "mean"),
            auc_pr_sd=("auc_pr", "std"),
            brier_score_mean=("brier_score", "mean"),
            brier_score_sd=("brier_score", "std"),
        )
        .reset_index()
    )

    primary_path = OUT_DIR / "split_seed_variance_grid.csv"
    decomposition_path = OUT_DIR / "split_seed_variance_decomposition.csv"
    partition_path = OUT_DIR / "per_partition_summary.csv"
    manifest_path = OUT_DIR / "partition_manifest.csv"
    legacy_path = OUT_DIR / "legacy_partition0_reconciliation.csv"
    status_path = OUT_DIR / "step_23b_status.json"

    primary.to_csv(primary_path, index=False)
    decomposition.to_csv(decomposition_path, index=False)
    per_partition.to_csv(partition_path, index=False)
    manifest_df.to_csv(manifest_path, index=False)

    legacy_pass, legacy_df = reconcile_legacy()
    if not legacy_pass:
        fail("Legacy partition-0 reconciliation failed; stop before using Step 23B outputs")
    legacy_df.to_csv(legacy_path, index=False)

    status = {
        "step": "23B",
        "primary_configuration": "Cell A — 50:50 XGBoost–LightGBM without attendance_decay",
        "split_seeds": SPLIT_SEEDS,
        "model_seeds": MODEL_SEEDS,
        "valid_partitions": int(primary.split_seed.nunique()),
        "expected_partitions": len(SPLIT_SEEDS),
        "legacy_partition0_reconciliation": "PASS",
        "outputs": [
            str(primary_path.relative_to(ROOT)),
            str(decomposition_path.relative_to(ROOT)),
            str(partition_path.relative_to(ROOT)),
            str(manifest_path.relative_to(ROOT)),
            str(legacy_path.relative_to(ROOT)),
        ],
    }
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    print("=" * 79)
    print("STEP 23B SUMMARY — PRIMARY CELL A")
    print("=" * 79)
    print(decomposition.to_string(index=False))
    print("=" * 79)
    print("OUTPUTS")
    for p in [primary_path, decomposition_path, partition_path, manifest_path, legacy_path, status_path]:
        print(f"[OK] {p}")
    print("=" * 79)
    print("STEP 23B COMPLETED")
    print("=" * 79)


if __name__ == "__main__":
    main()

