# ================================================================
# STEP 16.14B — LOCAL SHAP WATERFALL EXPLANATIONS
# Ghana JHS Academic Performance Thesis
# ================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from xgboost import XGBRegressor

print("=" * 70)
print("STEP 16.14B — LOCAL SHAP WATERFALL EXPLANATIONS")
print("=" * 70)

# ------------------------------------------------
# PATHS
# ------------------------------------------------

BASE_DIR = r"C:\Users\hills\OneDrive\Documents\MSc_Thesis\Ghana_JHS_Academic_Performance"

X_TRAIN_PATH = os.path.join(
    BASE_DIR, "data", "processed", "X_train.csv"
)

X_TEST_PATH = os.path.join(
    BASE_DIR, "data", "processed", "X_test.csv"
)

Y_TRAIN_PATH = os.path.join(
    BASE_DIR, "data", "processed", "y_train.csv"
)

Y_TEST_PATH = os.path.join(
    BASE_DIR, "data", "processed", "y_test.csv"
)

FIGURE_DIR = os.path.join(
    BASE_DIR, "results", "figures"
)

os.makedirs(FIGURE_DIR, exist_ok=True)

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

print("\nLoading data...")

X_train = pd.read_csv(X_TRAIN_PATH)
X_test = pd.read_csv(X_TEST_PATH)

y_train = pd.read_csv(
    Y_TRAIN_PATH
).squeeze("columns")

y_test = pd.read_csv(
    Y_TEST_PATH
).squeeze("columns")

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

# ------------------------------------------------
# TRAIN SAME XGBOOST MODEL USED IN STEP 16.14
# ------------------------------------------------

print("\nTraining XGBoost model...")

model = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("✓ XGBoost trained.")

# ------------------------------------------------
# SHAP EXPLAINER
# ------------------------------------------------

print("\nCalculating SHAP explanations...")

background_data = X_train.sample(
    n=min(100, len(X_train)),
    random_state=42
)

prediction_function = lambda data: model.predict(
    pd.DataFrame(
        data,
        columns=X_train.columns
    )
)

explainer = shap.Explainer(
    prediction_function,
    background_data
)

shap_explanation = explainer(
    X_test,
    max_evals=5000
)

print("✓ SHAP explanations calculated.")

# ------------------------------------------------
# IDENTIFY AT-RISK TEST CASES
# ------------------------------------------------

# Operational definition:
# overall average score < 50 = At Risk

at_risk_indices = np.where(
    y_test.values < 50
)[0]

print("\nAt-risk test observations:", len(at_risk_indices))

if len(at_risk_indices) < 3:
    raise ValueError(
        "Fewer than three at-risk test observations were found."
    )

# ------------------------------------------------
# SELECT THREE AT-RISK CASES
# ------------------------------------------------

# Select three representative cases:
# lowest actual score,
# middle at-risk score,
# highest at-risk score.

at_risk_scores = y_test.iloc[
    at_risk_indices
]

sorted_indices = at_risk_scores.sort_values().index.tolist()

selected = [
    sorted_indices[0],
    sorted_indices[len(sorted_indices) // 2],
    sorted_indices[-1]
]

print("\nSelected cases:")

for i, idx in enumerate(selected, start=1):

    actual_score = y_test.iloc[idx]
    prediction = model.predict(
        X_test.iloc[[idx]]
    )[0]

    print(
        f"Case {i}: "
        f"test index={idx}, "
        f"actual score={actual_score:.2f}, "
        f"predicted score={prediction:.2f}"
    )

# ------------------------------------------------
# GENERATE WATERFALL PLOTS
# ------------------------------------------------

for case_number, idx in enumerate(
    selected,
    start=1
):

    print(
        f"\nGenerating waterfall plot "
        f"for Case {case_number}..."
    )

    explanation = shap.Explanation(
        values=shap_explanation.values[idx],
        base_values=shap_explanation.base_values[idx],
        data=X_test.iloc[idx].values,
        feature_names=X_test.columns.tolist()
    )

    plt.figure(
        figsize=(10, 7)
    )

    shap.plots.waterfall(
        explanation,
        max_display=11,
        show=False
    )

    plt.tight_layout()

    output_path = os.path.join(
        FIGURE_DIR,
        f"shap_waterfall_case_{case_number}.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"✓ Saved: {output_path}"
    )

# ------------------------------------------------
# SAVE CASE INFORMATION
# ------------------------------------------------

case_rows = []

for case_number, idx in enumerate(
    selected,
    start=1
):

    actual_score = float(
        y_test.iloc[idx]
    )

    prediction = float(
        model.predict(
            X_test.iloc[[idx]]
        )[0]
    )

    case_rows.append({
        "case": f"Case {case_number}",
        "test_index": idx,
        "actual_score": actual_score,
        "predicted_score": prediction,
        "actual_status": "At Risk"
    })

case_df = pd.DataFrame(case_rows)

case_path = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "local_shap_cases.csv"
)

case_df.to_csv(
    case_path,
    index=False
)

print("\n✓ Saved case information:")
print(case_path)

# ------------------------------------------------
# COMPLETE
# ------------------------------------------------

print("\n" + "=" * 70)
print("STEP 16.14B COMPLETE")
print("=" * 70)

print("\nGenerated figures:")

for case_number in range(1, 4):

    print(
        os.path.join(
            FIGURE_DIR,
            f"shap_waterfall_case_{case_number}.png"
        )
    )

print("\n✓ Three local SHAP waterfall plots generated.")
print("✓ All three cases are actual at-risk test observations.")
print("✓ Same XGBoost model and SHAP methodology as Step 16.14.")