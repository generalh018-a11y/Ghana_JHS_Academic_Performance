# Ghana JHS Academic Performance — Final Thesis Analysis

## Purpose
This repository contains the reproducible Python analysis pipeline and non-sensitive final outputs for the MSc Cybersecurity and Digital Forensics thesis.

## Final research workflow
1. Data quality audit
2. Student-year data preparation
3. Target-leakage-controlled predictor preparation
4. Train/test split
5. XGBoost and LightGBM baseline modelling
6. XGBoost–LightGBM ensemble
7. SHAP explainability
8. Gender fairness analysis
9. Final tables and figures

## Engineered contribution
The model-engineering contribution is the integration of a recency-sensitive attendance-decay representation and class-sensitive probability fusion into the XGBoost–LightGBM ensemble. The attendance-decay feature gives greater weight to more recent terms, while the class-sensitive fusion is evaluated against the standard 50:50 ensemble.

## Key final regression result
The 50:50 XGBoost–LightGBM ensemble achieved:
- MAE = 4.9530
- RMSE = 6.0893
- R² = 0.6137

## Key classification results
The secondary at-risk classification analysis reports Accuracy, Precision, Recall, Macro-F1, AUC-ROC, AUC-PR and Brier Score.

## Reproducibility
The raw institutional student-level spreadsheet is intentionally excluded from this public repository for privacy and ethical reasons. The Python scripts in the repository operate on the locally prepared data files.

## Repository contents
The existing Python scripts in the repository contain the data preparation, splitting, baseline modelling, ensemble, SHAP, fairness and final-results stages. The `results/` directory contains non-sensitive final figures and result tables.

## Privacy
No direct student identifiers or raw student-level institutional data are published.

## Thesis
The final Methodology and Results and Analysis Word documents are supplied separately under `thesis_documents/`.

## Author
Name: ______________________________
