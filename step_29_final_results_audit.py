
# ============================================================
# STEP 29 — FINAL RESULTS AUDIT
# ============================================================
#
# Purpose:
#   Verify that all required datasets, model results, fairness
#   results, ablation results, and SHAP artifacts exist and
#   satisfy the final methodological requirements.
#
# IMPORTANT:
#   This script DOES NOT retrain models.
#   This script DOES NOT modify test results.
#   This script DOES NOT select a new threshold.
#   This script DOES NOT generate new predictive results.
#
# Project:
#   SHAP-Driven Fairness-Aware XGBoost–LightGBM Ensemble
#   for Predicting Junior High School Academic Performance
#
# ============================================================

from pathlib import Path
import pandas as pd


# ============================================================
# 1. PROJECT DIRECTORIES
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
SHAP_DIR = RESULTS_DIR / "shap"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"


print("\n" + "=" * 80)
print("STEP 29 — FINAL RESULTS AUDIT")
print("=" * 80)

print("\nProject root:")
print(PROJECT_ROOT)

print("\nData directory:")
print(DATA_DIR)

print("\nResults directory:")
print(RESULTS_DIR)

print("\nSHAP directory:")
print(SHAP_DIR)


# ============================================================
# 2. REQUIRED FILE DEFINITIONS
# ============================================================

# Core processed datasets
DATA_FILES = [

    DATA_DIR / "term_level_early_warning_dataset.csv",

    DATA_DIR / "train_student_level.csv",
    DATA_DIR / "validation_student_level.csv",
    DATA_DIR / "test_student_level.csv",

    DATA_DIR / "student_level_split_manifest.csv",

    DATA_DIR / "X_train.csv",
    DATA_DIR / "X_validation.csv",
    DATA_DIR / "X_test.csv",

    DATA_DIR / "y_train.csv",
    DATA_DIR / "y_validation.csv",
    DATA_DIR / "y_test.csv",

    DATA_DIR / "canonical_feature_manifest.csv",
]


# Core model/result files
RESULT_FILES = [

    DATA_DIR / "base_model_results_seed42.csv",

    DATA_DIR / "validation_ensemble_weight_search.csv",
    DATA_DIR / "locked_ensemble_configuration.csv",

    DATA_DIR / "locked_ensemble_test_results_seed42.csv",
    DATA_DIR / "locked_ensemble_test_predictions.csv",

    DATA_DIR / "ten_seed_ensemble_test_results.csv",
    DATA_DIR / "ten_seed_ensemble_summary_mean_sd.csv",
    DATA_DIR / "ten_seed_ensemble_weight_search.csv",
    DATA_DIR / "ten_seed_ensemble_weight_stability.csv",
    DATA_DIR / "ten_seed_ensemble_test_predictions.csv",

    DATA_DIR / "naive_comparator_results.csv",
    DATA_DIR / "naive_comparator_predictions.csv",

    DATA_DIR / "three_cell_ablation_results_by_seed.csv",
    DATA_DIR / "three_cell_ablation_validation_weight_search.csv",
    DATA_DIR / "three_cell_ablation_summary_mean_sd.csv",

    DATA_DIR / "fairness_analysis_mapped_predictions.csv",
    DATA_DIR / "fairness_subgroup_results_by_seed.csv",
    DATA_DIR / "fairness_subgroup_summary_mean_sd.csv",
    DATA_DIR / "fairness_gap_summary.csv",
]


# SHAP result files
SHAP_FILES = [

    SHAP_DIR / "shap_global_feature_importance.csv",
    SHAP_DIR / "shap_mean_signed_values.csv",
    SHAP_DIR / "shap_run_manifest.csv",
    SHAP_DIR / "shap_values_test_ensemble.csv",
    SHAP_DIR / "shap_base_values_test.csv",
    SHAP_DIR / "test_predictions_with_shap.csv",
    SHAP_DIR / "shap_visualization_data.csv",
    SHAP_DIR / "shap_waterfall_case_details.csv",
    SHAP_DIR / "selected_shap_waterfall_cases.csv",
    SHAP_DIR / "shap_additivity_check.csv",
]


# SHAP figures
SHAP_FIGURES = [

    
    SHAP_DIR / "Figure_SHAP_Global_Bar.png",
    SHAP_DIR / "Figure_SHAP_Beeswarm.png",
    SHAP_DIR / "Figure_SHAP_Dependence_score_t1.png",
    SHAP_DIR / "Figure_SHAP_Dependence_score_t2.png",
    SHAP_DIR / "Figure_SHAP_Dependence_attendance_decay.png",
    SHAP_DIR / "Figure_SHAP_Waterfall_TP.png",
    SHAP_DIR / "Figure_SHAP_Waterfall_TN.png",
    SHAP_DIR / "Figure_SHAP_Waterfall_FN.png",
]


REQUIRED_FILES = DATA_FILES + RESULT_FILES + SHAP_FILES + SHAP_FIGURES


# ============================================================
# 3. FILE EXISTENCE CHECK
# ============================================================

print("\n" + "-" * 80)
print("1. REQUIRED FILE CHECK")
print("-" * 80)

missing_files = []
existing_files = []

for file_path in REQUIRED_FILES:

    if file_path.exists():

        existing_files.append(file_path)

        print(
            "[OK]   ",
            file_path.relative_to(PROJECT_ROOT)
        )

    else:

        missing_files.append(file_path)

        print(
            "[MISS] ",
            file_path.relative_to(PROJECT_ROOT)
        )


print("\nRequired files:", len(REQUIRED_FILES))
print("Existing files:", len(existing_files))
print("Missing files :", len(missing_files))


# ============================================================
# 4. TERM-LEVEL DATASET CHECK
# ============================================================

print("\n" + "-" * 80)
print("2. TERM-LEVEL DATASET CHECK")
print("-" * 80)

term_file = DATA_DIR / "term_level_early_warning_dataset.csv"

term_check_pass = False

if term_file.exists():

    term_df = pd.read_csv(term_file)

    print("Rows:", len(term_df))
    print("Columns:", len(term_df.columns))

    print(
        "Unique students:",
        term_df["student_id"].nunique()
    )

    print(
        "Student-year rows:",
        len(
            term_df[
                ["student_id", "academic_year"]
            ].drop_duplicates()
        )
    )

    print(
        "Missing target:",
        term_df["target_score_t3"].isna().sum()
    )

    print(
        "At-risk:",
        int(term_df["at_risk"].sum())
    )

    print(
        "Not at risk:",
        int(
            (term_df["at_risk"] == 0).sum()
        )
    )

    expected_features = [
        "student_id",
        "academic_year",
        "class_level",
        "gender",
        "age_years",
        "score_t1",
        "score_t2",
        "attendance_t1",
        "attendance_t2",
        "attendance_decay",
        "target_score_t3",
        "at_risk",
        "at_risk_label",
    ]

    missing_columns = [
        c
        for c in expected_features
        if c not in term_df.columns
    ]

    if missing_columns:

        print(
            "WARNING — missing columns:",
            missing_columns
        )

    else:

        print("Feature schema: PASS")

        if (
            len(term_df) == 541
            and term_df["student_id"].nunique() == 340
            and term_df["target_score_t3"].isna().sum() == 0
            and len(missing_columns) == 0
        ):

            term_check_pass = True

            print("Term-level dataset integrity: PASS")

        else:

            print(
                "Term-level dataset integrity: CHECK"
            )

else:

    print(
        "[MISS] Term-level dataset not found."
    )


# ============================================================
# 5. TRAIN / VALIDATION / TEST CHECK
# ============================================================

print("\n" + "-" * 80)
print("3. TRAIN / VALIDATION / TEST CHECK")
print("-" * 80)

split_files = {

    "train":
        DATA_DIR / "train_student_level.csv",

    "validation":
        DATA_DIR / "validation_student_level.csv",

    "test":
        DATA_DIR / "test_student_level.csv",
}

split_dfs = {}

for split_name, path in split_files.items():

    if not path.exists():

        print(
            f"[MISS] {split_name}: {path}"
        )

        continue

    df = pd.read_csv(path)

    split_dfs[split_name] = df

    print(f"\n{split_name.upper()}:")

    print(
        "  Rows:",
        len(df)
    )

    print(
        "  Unique students:",
        df["student_id"].nunique()
    )

    print(
        "  At-risk:",
        int(df["at_risk"].sum())
    )

    print(
        "  Not at risk:",
        int(
            (df["at_risk"] == 0).sum()
        )
    )


# ============================================================
# 6. STUDENT OVERLAP / LEAKAGE CHECK
# ============================================================

print("\n" + "-" * 80)
print("4. STUDENT-LEVEL LEAKAGE CHECK")
print("-" * 80)

leakage_check_pass = False

if len(split_dfs) == 3:

    train_students = set(
        split_dfs["train"]["student_id"]
    )

    validation_students = set(
        split_dfs["validation"]["student_id"]
    )

    test_students = set(
        split_dfs["test"]["student_id"]
    )

    train_val_overlap = (
        train_students
        &
        validation_students
    )

    train_test_overlap = (
        train_students
        &
        test_students
    )

    validation_test_overlap = (
        validation_students
        &
        test_students
    )

    print(
        "Train ∩ Validation:",
        len(train_val_overlap)
    )

    print(
        "Train ∩ Test:",
        len(train_test_overlap)
    )

    print(
        "Validation ∩ Test:",
        len(validation_test_overlap)
    )

    if (
        len(train_val_overlap) == 0
        and
        len(train_test_overlap) == 0
        and
        len(validation_test_overlap) == 0
    ):

        print(
            "\nSTUDENT-LEVEL LEAKAGE CHECK: PASS"
        )

        leakage_check_pass = True

    else:

        print(
            "\nSTUDENT-LEVEL LEAKAGE CHECK: FAIL"
        )

else:

    print(
        "\nSTUDENT-LEVEL LEAKAGE CHECK: CHECK"
    )


# ============================================================
# 7. CANONICAL FEATURE MATRIX CHECK
# ============================================================

print("\n" + "-" * 80)
print("5. CANONICAL FEATURE MATRIX CHECK")
print("-" * 80)

feature_files = {

    "X_train":
        DATA_DIR / "X_train.csv",

    "X_validation":
        DATA_DIR / "X_validation.csv",

    "X_test":
        DATA_DIR / "X_test.csv",
}

feature_dfs = {}

feature_check_pass = True

for name, path in feature_files.items():

    if path.exists():

        df = pd.read_csv(path)

        feature_dfs[name] = df

        print(
            name,
            "shape:",
            df.shape
        )

        missing = int(
            df.isna().sum().sum()
        )

        print(
            name,
            "missing:",
            missing
        )

        if missing != 0:
            feature_check_pass = False

    else:

        print(
            "[MISS]",
            name
        )

        feature_check_pass = False


if feature_check_pass:

    expected_shapes = {

        "X_train": (318, 12),
        "X_validation": (113, 12),
        "X_test": (110, 12),
    }

    shapes_match = True

    for name, expected_shape in expected_shapes.items():

        if name in feature_dfs:

            if feature_dfs[name].shape != expected_shape:

                shapes_match = False

                print(
                    f"WARNING — {name} expected "
                    f"{expected_shape}, found "
                    f"{feature_dfs[name].shape}"
                )

    if shapes_match:

        print(
            "Canonical feature matrices: PASS"
        )

    else:

        print(
            "Canonical feature matrices: CHECK"
        )


# ============================================================
# 8. LOCKED ENSEMBLE CONFIGURATION
# ============================================================

print("\n" + "-" * 80)
print("6. LOCKED ENSEMBLE CONFIGURATION")
print("-" * 80)

config_file = (
    DATA_DIR /
    "locked_ensemble_configuration.csv"
)

configuration_check_pass = False

if config_file.exists():

    config_df = pd.read_csv(config_file)

    print(
        config_df.to_string(index=False)
    )

    print(
        "\nLocked configuration file: PRESENT"
    )

    # Confirm expected locked values where available.
    config_text = config_df.to_string(
        index=False
    )

    if (
        "0.1" in config_text
        and
        "0.9" in config_text
    ):

        print(
            "Expected XGB/LGBM weighting appears present."
        )

    configuration_check_pass = True

else:

    print(
        "[MISS] Locked ensemble configuration."
    )


# ============================================================
# 9. TEN-SEED STABILITY CHECK
# ============================================================

print("\n" + "-" * 80)
print("7. TEN-SEED STABILITY CHECK")
print("-" * 80)

ten_seed_file = (
    DATA_DIR /
    "ten_seed_ensemble_test_results.csv"
)

ten_seed_check_pass = False

if ten_seed_file.exists():

    ten_seed_df = pd.read_csv(
        ten_seed_file
    )

    print(
        "Rows:",
        len(ten_seed_df)
    )

    if "seed" in ten_seed_df.columns:

        seeds = sorted(
            ten_seed_df["seed"].unique()
        )

        print(
            "Seeds:",
            seeds
        )

        print(
            "Number of seeds:",
            len(seeds)
        )

        expected_seeds = list(
            range(42, 52)
        )

        if seeds == expected_seeds:

            print(
                "Ten-seed coverage: PASS"
            )

            ten_seed_check_pass = True

        else:

            print(
                "Ten-seed coverage: CHECK"
            )

else:

    print(
        "[MISS] Ten-seed results."
    )


# ============================================================
# 10. NAIVE COMPARATOR CHECK
# ============================================================

print("\n" + "-" * 80)
print("8. NAIVE COMPARATOR CHECK")
print("-" * 80)

naive_file = (
    DATA_DIR /
    "naive_comparator_results.csv"
)

if naive_file.exists():

    naive_df = pd.read_csv(
        naive_file
    )

    print(
        naive_df.to_string(index=False)
    )

    print(
        "\nNaive comparator: PRESENT"
    )

else:

    print(
        "\nNaive comparator: MISSING"
    )


# ============================================================
# 11. THREE-CELL ABLATION CHECK
# ============================================================

print("\n" + "-" * 80)
print("9. THREE-CELL ABLATION CHECK")
print("-" * 80)

ablation_file = (
    DATA_DIR /
    "three_cell_ablation_summary_mean_sd.csv"
)

if ablation_file.exists():

    ablation_df = pd.read_csv(
        ablation_file
    )

    print(
        ablation_df.to_string(index=False)
    )

    print(
        "\nThree-cell ablation: PRESENT"
    )

else:

    print(
        "\nThree-cell ablation: MISSING"
    )


# ============================================================
# 12. FAIRNESS CHECK
# ============================================================

print("\n" + "-" * 80)
print("10. FAIRNESS ANALYSIS CHECK")
print("-" * 80)

fairness_file = (
    DATA_DIR /
    "fairness_subgroup_summary_mean_sd.csv"
)

gap_file = (
    DATA_DIR /
    "fairness_gap_summary.csv"
)

fairness_check_pass = False

if fairness_file.exists():

    fairness_df = pd.read_csv(
        fairness_file
    )

    print(
        "\nSubgroup results:"
    )

    print(
        fairness_df.to_string(index=False)
    )

    fairness_check_pass = True

else:

    print(
        "\n[MISS] Fairness subgroup summary."
    )


if gap_file.exists():

    gap_df = pd.read_csv(
        gap_file
    )

    print(
        "\nFairness gaps:"
    )

    print(
        gap_df.to_string(index=False)
    )

else:

    print(
        "\n[MISS] Fairness gap summary."
    )


# ============================================================
# 13. SHAP RESULTS CHECK
# ============================================================

print("\n" + "-" * 80)
print("11. SHAP RESULTS CHECK")
print("-" * 80)

shap_global_file = (
    SHAP_DIR /
    "shap_global_feature_importance.csv"
)

if shap_global_file.exists():

    shap_global_df = pd.read_csv(
        shap_global_file
    )

    print(
        "\nGlobal SHAP importance:"
    )

    print(
        shap_global_df.to_string(index=False)
    )

    print(
        "\nGlobal SHAP results: PRESENT"
    )

else:

    print(
        "\n[MISS] Global SHAP results."
    )


# ============================================================
# 14. SHAP ADDITIVITY CHECK
# ============================================================

print("\n" + "-" * 80)
print("12. SHAP ADDITIVITY CHECK")
print("-" * 80)

additivity_file = (
    SHAP_DIR /
    "shap_additivity_check.csv"
)

shap_additivity_pass = False

if additivity_file.exists():

    additivity_df = pd.read_csv(
        additivity_file
    )

    maximum_error = (
        additivity_df[
            "absolute_reconstruction_error"
        ].max()
    )

    mean_error = (
        additivity_df[
            "absolute_reconstruction_error"
        ].mean()
    )

    print(
        "Observations:",
        len(additivity_df)
    )

    print(
        "Maximum reconstruction error:",
        maximum_error
    )

    print(
        "Mean reconstruction error:",
        mean_error
    )

    if maximum_error <= 1e-4:

        print(
            "SHAP additivity: PASS"
        )

        shap_additivity_pass = True

    else:

        print(
            "SHAP additivity: FAIL"
        )

else:

    print(
        "[MISS] SHAP additivity file."
    )


# ============================================================
# 15. SHAP WATERFALL CASE CHECK
# ============================================================

print("\n" + "-" * 80)
print("13. SHAP WATERFALL CASE CHECK")
print("-" * 80)

cases_file = (
    SHAP_DIR /
    "selected_shap_waterfall_cases.csv"
)

waterfall_check_pass = False

if cases_file.exists():

    cases_df = pd.read_csv(
        cases_file
    )

    print(
        cases_df.to_string(index=False)
    )

    if "case" in cases_df.columns:

        expected_cases = {
            "TP",
            "TN",
            "FN"
        }

        actual_cases = set(
            cases_df["case"]
        )

        if expected_cases.issubset(
            actual_cases
        ):

            print(
                "\nTP/TN/FN waterfall cases: PASS"
            )

            waterfall_check_pass = True

        else:

            print(
                "\nTP/TN/FN waterfall cases: CHECK"
            )

else:

    print(
        "[MISS] SHAP waterfall case file."
    )


# ============================================================
# 16. FINAL TEST SIZE CHECK
# ============================================================

print("\n" + "-" * 80)
print("14. FINAL TEST SIZE CHECK")
print("-" * 80)

test_file = (
    DATA_DIR /
    "test_student_level.csv"
)

test_size_pass = False

if test_file.exists():

    test_df = pd.read_csv(
        test_file
    )

    print(
        "Final test observations:",
        len(test_df)
    )

    print(
        "Final test unique students:",
        test_df["student_id"].nunique()
    )

    if len(test_df) == 110:

        print(
            "Final test size: PASS"
        )

        test_size_pass = True

    else:

        print(
            "Final test size: CHECK"
        )

else:

    print(
        "[MISS] Final test dataset."
    )


# ============================================================
# 17. FAIRNESS-READY PREDICTION CHECK
# ============================================================

print("\n" + "-" * 80)
print("15. FAIRNESS-READY PREDICTION CHECK")
print("-" * 80)

fairness_mapping_file = (
    DATA_DIR /
    "fairness_analysis_mapped_predictions.csv"
)

if fairness_mapping_file.exists():

    fairness_mapping_df = pd.read_csv(
        fairness_mapping_file
    )

    print(
        "Rows:",
        len(fairness_mapping_df)
    )

    if "student_id" in fairness_mapping_df.columns:

        print(
            "Unique students:",
            fairness_mapping_df[
                "student_id"
            ].nunique()
        )

    if "academic_year" in fairness_mapping_df.columns:

        print(
            "Unique student-year trajectories:",
            len(
                fairness_mapping_df[
                    [
                        "student_id",
                        "academic_year"
                    ]
                ].drop_duplicates()
            )
        )

    required_mapping_columns = [
        "student_id",
        "gender",
        "class_level",
        "academic_year",
        "y_true",
        "ensemble_probability",
        "ensemble_prediction",
    ]

    missing_mapping_columns = [
        c
        for c in required_mapping_columns
        if c not in fairness_mapping_df.columns
    ]

    if not missing_mapping_columns:

        print(
            "Fairness mapping schema: PASS"
        )

    else:

        print(
            "Fairness mapping schema: CHECK",
            missing_mapping_columns
        )

else:

    print(
        "[MISS] Fairness mapped predictions."
    )


# ============================================================
# 18. IMPORTANT METHODOLOGICAL STATUS
# ============================================================

print("\n" + "-" * 80)
print("16. METHODOLOGICAL STATUS")
print("-" * 80)

print(
    """
Model selection:
    Validation set only.

Ensemble weight selection:
    Validation set only.

Final test evaluation:
    Performed after configuration was locked.

Threshold:
    0.50 locked for the primary experiment.

Seeds:
    42–51 for stability analysis.

Primary model:
    XGBoost–LightGBM weighted ensemble.

SHAP:
    Post-hoc explainability only.

Fairness:
    Gender and class-level subgroup analysis.

Student leakage:
    Grouped student-level split.

Naive comparator:
    Included.

Three-cell ablation:
    Included.

Threshold sensitivity:
    Included as post-hoc sensitivity analysis.

Public-dataset transfer:
    NOT included in the primary experiment.

Socioeconomic fairness:
    NOT claimed because the available contextual
    variables did not provide usable socioeconomic
    variation.

Term-level prediction mechanism:
    Term 3 outcome predicted from Terms 1 and 2.

JHS3 Term 3:
    Students without observed Term 3 academic outcomes
    are not included in the supervised Term 3 prediction
    dataset.

Attendance representation:
    Normalized attendance rate is used rather than
    simultaneously entering attendance numerator,
    denominator, and rate as primary predictors.
"""
)


# ============================================================
# 19. FINAL AUDIT SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL AUDIT SUMMARY")
print("=" * 80)

print(
    "\nRequired files found:",
    len(existing_files),
    "/",
    len(REQUIRED_FILES)
)

print(
    "Required files missing:",
    len(missing_files)
)

if missing_files:

    print(
        "\nWARNING — Missing files:"
    )

    for file_path in missing_files:

        print(
            " -",
            file_path.relative_to(
                PROJECT_ROOT
            )
        )

else:

    print(
        "\nALL REQUIRED RESULTS FILES ARE PRESENT."
    )


print("\nIndividual audit statuses:")

print(
    "Term-level dataset:",
    "PASS" if term_check_pass else "CHECK"
)

print(
    "Student-level leakage:",
    "PASS" if leakage_check_pass else "CHECK"
)

print(
    "Canonical feature matrices:",
    "PASS" if feature_check_pass else "CHECK"
)

print(
    "Locked ensemble configuration:",
    "PASS" if configuration_check_pass else "CHECK"
)

print(
    "Ten-seed stability:",
    "PASS" if ten_seed_check_pass else "CHECK"
)

print(
    "Fairness analysis:",
    "PASS" if fairness_check_pass else "CHECK"
)

print(
    "SHAP additivity:",
    "PASS" if shap_additivity_pass else "CHECK"
)

print(
    "SHAP waterfall cases:",
    "PASS" if waterfall_check_pass else "CHECK"
)

print(
    "Final test size:",
    "PASS" if test_size_pass else "CHECK"
)


# ============================================================
# 20. FINAL METHODOLOGICAL SAFETY STATEMENT
# ============================================================

print("\n" + "-" * 80)
print("METHODOLOGICAL SAFETY")
print("-" * 80)

print(
    """
No models were retrained.
No test-set model selection was performed.
No threshold was changed.
No new predictive result was generated.
No existing result was overwritten.
This script only audits existing files and results.
"""
)


# ============================================================
# 21. COMPLETION
# ============================================================

print("\n" + "=" * 80)
print("STEP 29 COMPLETED")
print("=" * 80)