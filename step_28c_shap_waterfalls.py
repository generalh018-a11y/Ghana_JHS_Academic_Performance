
"""
Step 28C: Individual SHAP waterfall case analysis.

Purpose:
- Select representative TP, TN and FN cases systematically.
- Use the already-generated SHAP values from Step 28A.
- Use the actual SHAP base values returned by PermutationExplainer.
- Verify SHAP additivity against the locked ensemble probability.
- Generate corrected individual waterfall plots.

No retraining.
No model selection.
No threshold tuning.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

SHAP_DIR = PROJECT_ROOT / "results" / "shap"

SHAP_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. LOAD SHAP RESULTS
# ============================================================

data = pd.read_csv(
    SHAP_DIR / "test_predictions_with_shap.csv"
)

shap_values_df = pd.read_csv(
    SHAP_DIR / "shap_values_test_ensemble.csv"
)

base_values_df = pd.read_csv(
    SHAP_DIR / "shap_base_values_test.csv"
)

global_importance = pd.read_csv(
    SHAP_DIR / "shap_global_feature_importance.csv"
)


# ============================================================
# 3. DEFINE FEATURES
# ============================================================

feature_columns = list(
    global_importance["feature"]
)

shap_columns = [
    f"shap_{feature}"
    for feature in feature_columns
]


# ============================================================
# 4. BASIC DATA CHECKS
# ============================================================

print("=" * 70)
print("STEP 28C — CORRECTED INDIVIDUAL SHAP WATERFALL ANALYSIS")
print("=" * 70)


print("\nTest observations:", len(data))

if len(data) != 110:
    raise ValueError(
        f"Expected 110 test observations, "
        f"found {len(data)}."
    )


if len(shap_values_df) != 110:
    raise ValueError(
        "SHAP values file does not contain 110 observations."
    )


if len(base_values_df) != 110:
    raise ValueError(
        "SHAP base values file does not contain 110 observations."
    )


# ============================================================
# 5. EXTRACT SHAP VALUES
# ============================================================

X = data[
    feature_columns
].copy()


shap_values = data[
    shap_columns
].to_numpy()


base_values = base_values_df[
    "shap_base_value"
].to_numpy()


ensemble_probabilities = data[
    "ensemble_probability"
].to_numpy()


actual_values = data[
    "actual_at_risk"
].astype(int).to_numpy()


predictions = data[
    "ensemble_prediction"
].astype(int).to_numpy()


# ============================================================
# 6. VERIFY SHAP DIMENSIONS
# ============================================================

print("\nFeature matrix shape:", X.shape)

print(
    "SHAP matrix shape:",
    shap_values.shape
)

print(
    "Base-value vector shape:",
    base_values.shape
)


if shap_values.shape != X.shape:
    raise ValueError(
        "SHAP matrix and feature matrix do not match."
    )


# ============================================================
# 7. VERIFY TEST ROW ALIGNMENT
# ============================================================

if not np.array_equal(
    data["test_row"].to_numpy(),
    shap_values_df["test_row"].to_numpy()
):

    raise ValueError(
        "Test-row ordering differs between prediction "
        "and SHAP files."
    )


if not np.array_equal(
    data["test_row"].to_numpy(),
    base_values_df["test_row"].to_numpy()
):

    raise ValueError(
        "Test-row ordering differs between prediction "
        "and SHAP base-value files."
    )


# ============================================================
# 8. SHAP ADDITIVITY VERIFICATION
# ============================================================

print("\n" + "-" * 70)
print("SHAP ADDITIVITY VERIFICATION")
print("-" * 70)


# For a model-independent SHAP explanation:

# base value + sum(feature SHAP values)
#
# should approximately equal:
#
# ensemble prediction probability


reconstructed_probabilities = (
    base_values
    + shap_values.sum(axis=1)
)


absolute_errors = np.abs(
    reconstructed_probabilities
    - ensemble_probabilities
)


print(
    "Maximum absolute reconstruction error:",
    absolute_errors.max()
)

print(
    "Mean absolute reconstruction error:",
    absolute_errors.mean()
)


# SHAP numerical approximation is allowed to have
# a small numerical error.

ADDITIVITY_TOLERANCE = 1e-4


if absolute_errors.max() > ADDITIVITY_TOLERANCE:

    print(
        "\nWARNING:"
    )

    print(
        "SHAP reconstruction error exceeds "
        f"the tolerance of {ADDITIVITY_TOLERANCE}."
    )

    print(
        "Waterfall figures will NOT be generated."
    )

    raise ValueError(
        "SHAP additivity verification failed."
    )


print(
    "SHAP additivity verification PASSED."
)


# ============================================================
# 9. SAVE ADDITIVITY CHECK
# ============================================================

additivity_check = pd.DataFrame({
    "test_row": data["test_row"],
    "ensemble_probability": ensemble_probabilities,
    "shap_base_value": base_values,
    "sum_shap_values": shap_values.sum(axis=1),
    "reconstructed_probability": reconstructed_probabilities,
    "absolute_reconstruction_error": absolute_errors
})


additivity_check.to_csv(
    SHAP_DIR /
    "shap_additivity_check.csv",
    index=False
)


print(
    "\nSaved:"
)

print(
    SHAP_DIR /
    "shap_additivity_check.csv"
)


# ============================================================
# 10. IDENTIFY CASE TYPES
# ============================================================

data["case_type"] = "Other"


# True Positive
data.loc[
    (data["actual_at_risk"] == 1)
    &
    (data["ensemble_prediction"] == 1),
    "case_type"
] = "True Positive"


# True Negative
data.loc[
    (data["actual_at_risk"] == 0)
    &
    (data["ensemble_prediction"] == 0),
    "case_type"
] = "True Negative"


# False Negative
data.loc[
    (data["actual_at_risk"] == 1)
    &
    (data["ensemble_prediction"] == 0),
    "case_type"
] = "False Negative"


# False Positive
data.loc[
    (data["actual_at_risk"] == 0)
    &
    (data["ensemble_prediction"] == 1),
    "case_type"
] = "False Positive"


# ============================================================
# 11. DISPLAY CASE COUNTS
# ============================================================

print(
    "\nCase counts:"
)

print(
    data["case_type"]
    .value_counts()
    .to_string()
)


# ============================================================
# 12. SYSTEMATIC CASE SELECTION
# ============================================================

# TP:
# Highest predicted probability among correctly
# identified at-risk students.
#
# TN:
# Lowest predicted probability among correctly
# identified non-risk students.
#
# FN:
# Highest predicted probability among missed
# at-risk students.
#
# These rules were fixed before examining the
# individual SHAP explanations.


tp_candidates = data[
    data["case_type"] == "True Positive"
].copy()


tn_candidates = data[
    data["case_type"] == "True Negative"
].copy()


fn_candidates = data[
    data["case_type"] == "False Negative"
].copy()


if len(tp_candidates) == 0:
    raise ValueError(
        "No True Positive case available."
    )


if len(tn_candidates) == 0:
    raise ValueError(
        "No True Negative case available."
    )


if len(fn_candidates) == 0:
    raise ValueError(
        "No False Negative case available."
    )


tp_case = tp_candidates.sort_values(
    "ensemble_probability",
    ascending=False
).iloc[0]


tn_case = tn_candidates.sort_values(
    "ensemble_probability",
    ascending=True
).iloc[0]


fn_case = fn_candidates.sort_values(
    "ensemble_probability",
    ascending=False
).iloc[0]


# ============================================================
# 13. SAVE CASE SELECTION
# ============================================================

selected_cases = pd.DataFrame([
    {
        "case": "TP",
        "case_type": "True Positive",
        "test_row": int(tp_case["test_row"]),
        "actual_at_risk": int(
            tp_case["actual_at_risk"]
        ),
        "ensemble_probability": float(
            tp_case["ensemble_probability"]
        ),
        "ensemble_prediction": int(
            tp_case["ensemble_prediction"]
        ),
        "selection_rule":
            "Highest predicted probability among true positives"
    },

    {
        "case": "TN",
        "case_type": "True Negative",
        "test_row": int(tn_case["test_row"]),
        "actual_at_risk": int(
            tn_case["actual_at_risk"]
        ),
        "ensemble_probability": float(
            tn_case["ensemble_probability"]
        ),
        "ensemble_prediction": int(
            tn_case["ensemble_prediction"]
        ),
        "selection_rule":
            "Lowest predicted probability among true negatives"
    },

    {
        "case": "FN",
        "case_type": "False Negative",
        "test_row": int(fn_case["test_row"]),
        "actual_at_risk": int(
            fn_case["actual_at_risk"]
        ),
        "ensemble_probability": float(
            fn_case["ensemble_probability"]
        ),
        "ensemble_prediction": int(
            fn_case["ensemble_prediction"]
        ),
        "selection_rule":
            "Highest predicted probability among false negatives"
    }
])


selected_cases.to_csv(
    SHAP_DIR /
    "selected_shap_waterfall_cases.csv",
    index=False
)


print(
    "\nSelected cases:"
)

print(
    selected_cases.to_string(
        index=False
    )
)


# ============================================================
# 14. GENERATE CORRECTED WATERFALLS
# ============================================================

case_lookup = {
    "TP": tp_case,
    "TN": tn_case,
    "FN": fn_case
}


for case_name, row in case_lookup.items():

    test_row = int(
        row["test_row"]
    )


    # Locate the corresponding row.
    row_indices = np.where(
        data["test_row"].to_numpy()
        == test_row
    )[0]


    if len(row_indices) != 1:
        raise ValueError(
            f"Could not uniquely locate test row "
            f"{test_row}."
        )


    row_index = int(
        row_indices[0]
    )


    # Get SHAP values for this observation.
    values = shap_values[
        row_index
    ]


    # Get actual feature values.
    feature_values = X.iloc[
        row_index
    ].to_numpy()


    # IMPORTANT:
    # Use the actual base value returned by SHAP,
    # not the mean ensemble probability.
    base_value = float(
        base_values[row_index]
    )


    actual_probability = float(
        ensemble_probabilities[row_index]
    )


    reconstructed_probability = (
        base_value
        + values.sum()
    )


    reconstruction_error = abs(
        reconstructed_probability
        - actual_probability
    )


    print(
        f"\n{case_name} case:"
    )

    print(
        "Test row:",
        test_row
    )

    print(
        "SHAP base value:",
        base_value
    )

    print(
        "Actual ensemble probability:",
        actual_probability
    )

    print(
        "Reconstructed probability:",
        reconstructed_probability
    )

    print(
        "Reconstruction error:",
        reconstruction_error
    )


    # --------------------------------------------------------
    # Create SHAP Explanation
    # --------------------------------------------------------

    explanation = shap.Explanation(
        values=values,
        base_values=base_value,
        data=feature_values,
        feature_names=feature_columns
    )


    # --------------------------------------------------------
    # Create waterfall plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 7)
    )


    shap.plots.waterfall(
        explanation,
        max_display=len(feature_columns),
        show=False
    )


    plt.title(
        f"SHAP Waterfall — "
        f"{case_name} ({row['case_type']})"
    )


    plt.tight_layout()


    output_path = (
        SHAP_DIR
        /
        f"Figure_SHAP_Waterfall_{case_name}.png"
    )


    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        "Saved:"
    )

    print(
        output_path
    )


# ============================================================
# 15. SAVE DETAILED CASE INFORMATION
# ============================================================

case_details = []


for case_name, row in case_lookup.items():

    test_row = int(
        row["test_row"]
    )


    row_index = int(
        np.where(
            data["test_row"].to_numpy()
            == test_row
        )[0][0]
    )


    values = shap_values[
        row_index
    ]


    feature_values = X.iloc[
        row_index
    ].to_numpy()


    base_value = float(
        base_values[row_index]
    )


    actual_probability = float(
        ensemble_probabilities[row_index]
    )


    reconstructed_probability = (
        base_value
        + values.sum()
    )


    for feature, feature_value, shap_value in zip(
        feature_columns,
        feature_values,
        values
    ):

        case_details.append({

            "case": case_name,

            "case_type":
                row["case_type"],

            "test_row":
                test_row,

            "actual_at_risk":
                int(row["actual_at_risk"]),

            "ensemble_probability":
                actual_probability,

            "ensemble_prediction":
                int(row["ensemble_prediction"]),

            "shap_base_value":
                base_value,

            "reconstructed_probability":
                reconstructed_probability,

            "feature":
                feature,

            "feature_value":
                feature_value,

            "shap_value":
                shap_value,

            "absolute_shap_value":
                abs(shap_value)
        })


case_details_df = pd.DataFrame(
    case_details
)


case_details_df.to_csv(
    SHAP_DIR /
    "shap_waterfall_case_details.csv",
    index=False
)


print(
    "\nSaved:"
)

print(
    SHAP_DIR /
    "shap_waterfall_case_details.csv"
)


# ============================================================
# 16. FINISH
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "STEP 28C COMPLETED SUCCESSFULLY"
)

print(
    "=" * 70
)

print(
    "\nCorrected waterfall figures generated:"
)

print(
    " - Figure_SHAP_Waterfall_TP.png"
)

print(
    " - Figure_SHAP_Waterfall_TN.png"
)

print(
    " - Figure_SHAP_Waterfall_FN.png"
)

print(
    "\nSHAP additivity was verified before "
    "generating the figures."
)