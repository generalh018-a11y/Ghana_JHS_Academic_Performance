# SHAP-Driven Fairness-Aware XGBoost–LightGBM Ensemble for Predicting Junior High School Academic Performance in Ghana

## Overview

This repository contains the reproducible machine-learning workflow supporting the MSc thesis:

> **SHAP-Driven Fairness-Aware XGBoost–LightGBM Ensemble for Predicting Junior High School Academic Performance in Ghana: An Explainable Decision Support Study**

The study develops and evaluates a leakage-controlled XGBoost–LightGBM ensemble for predicting **Term 3 at-risk status** using academic and attendance information available from **Terms 1 and 2**.

The study combines predictive modelling, fairness-aware subgroup analysis, and SHAP-based explainability to investigate how academic, attendance, demographic, and temporal variables contribute to Term 3 at-risk predictions.

---

## Research Objectives

### Objective 1

To develop and evaluate a leakage-controlled XGBoost–LightGBM ensemble for predicting Term 3 at-risk status, defined from academic performance, using term-level academic and attendance records available in Terms 1 and 2.

### Objective 2

To apply SHAP-based global and individual-level explanations to identify and interpret the contributions of academic, attendance, demographic, and temporal predictors to the ensemble's Term 3 at-risk predictions.

### Objective 3

To evaluate predictive disparities in the ensemble across gender and class-level subgroups using subgroup-disaggregated performance measures.

---

## Prediction Design

The prediction task is defined at the **student-academic-year trajectory** level.

- **Prediction target:** Term 3 at-risk status.
- **Predictors:** information available from Terms 1 and 2 only.
- **Prediction horizon:** Term 3.
- **Student-level grouping:** all observations belonging to the same student are kept within the same data partition.
- **Train/validation/test split:** grouped at the student level.
- **Test set:** used only after model and ensemble decisions were frozen.
- **Decision threshold:** 0.50.
- **Primary model:** Cell A.

JHS3 trajectories without an observed Term 3 outcome are excluded from the prediction analysis because those students had not completed the academic period before writing the Basic Education Certificate Examination (BECE).

---

## Final Primary Model — Cell A

The final primary model is a **50:50 XGBoost–LightGBM ensemble without the derived `attendance_decay` feature**.

### XGBoost

- `n_estimators = 300`
- `max_depth = 4`
- `learning_rate = 0.05`
- `subsample = 0.8`
- `colsample_bytree = 0.8`
- `objective = binary:logistic`
- `eval_metric = logloss`
- `n_jobs = -1`

### LightGBM

- `n_estimators = 300`
- `max_depth = 4`
- `learning_rate = 0.05`
- `subsample = 0.8`
- `colsample_bytree = 0.8`
- `objective = binary`
- `n_jobs = -1`
- `verbosity = -1`

The primary locked run uses **seed 42**, equal model weights, and a decision threshold of **0.50**.

---

## Dataset and Eligibility

The modelling workflow is based on student academic and attendance records collected across multiple public Junior High Schools.

The source workbook contains term-level records covering:

- 2022–2023 Term 1
- 2022–2023 Term 2
- 2022–2023 Term 3
- 2023–2024 Term 1
- 2023–2024 Term 2
- 2023–2024 Term 3

The processed dataset contains:

- **2,400** term-level records initially considered
- **800** student-academic-year trajectories after preparation
- **259** trajectories without an observed Term 3 outcome
- **541** eligible trajectories for the prediction task

The 259 excluded trajectories correspond to JHS3 cases without completed Term 3 outcomes and therefore define an explicit design boundary for the empirical prediction analysis.

---

## Primary Feature Set

The final Cell A model uses **11 predictors**:

1. `score_t1`
2. `score_t2`
3. `attendance_t1`
4. `attendance_t2`
5. `age_years`
6. `gender_F`
7. `gender_M`
8. `class_level_JHS1`
9. `class_level_JHS2`
10. `academic_year_2022-23`
11. `academic_year_2023-24`

The derived `attendance_decay` feature is excluded from the primary Cell A model.

---

## Validation and Robustness

Model development uses validation data for model-selection decisions and preserves the test set for final evaluation.

The repository includes:

- Ten-seed primary-model stability analysis.
- Student-level grouped splitting.
- Split-variance analysis across multiple grouped partitions.
- Prior-score ablation.
- Majority-class naive comparator.
- Threshold-sensitivity analysis.
- Three-cell ablation analysis.
- Fairness analysis by gender and class level.

The primary stability analysis uses model seeds **42–51**.

The additional split-variance analysis evaluates multiple student-level partitions while keeping the modelling procedure consistent.

---

## Explainability

SHAP-based explainability is applied to the locked Cell A ensemble.

The final SHAP analysis uses:

- **Seed:** 42
- **Test observations:** 110
- **Features:** 11
- **Background observations:** 100
- **Explainer:** model-independent `PermutationExplainer`
- **Maximum evaluations:** 1,000

The SHAP workflow includes:

- Global feature-importance analysis.
- SHAP beeswarm visualization.
- Feature-dependence analysis.
- Individual prediction explanations.
- True-positive, true-negative, and false-negative waterfall examples.
- Additivity verification.
- Probability reconciliation against the locked ensemble predictions.

The maximum observed SHAP additivity error is approximately **2.44 × 10⁻¹⁵**, and the ensemble-probability reconciliation error is approximately **1.11 × 10⁻¹⁶**.

Row-level SHAP values and test-prediction records containing student-level information are excluded from version control.

---

## Fairness Analysis

Predictive performance is disaggregated across:

- **Gender:** female and male.
- **Class level:** JHS1 and JHS2.

The analysis reports measures including:

- Accuracy
- Precision
- Recall / true-positive rate
- False-positive rate
- False-negative rate
- Macro-F1

Subgroup gaps are reported descriptively rather than collapsed into a single fairness score.

---

## Repository Structure

```text
Ghana_JHS_Academic_Performance/
│
├── data/
│   └── processed/
│       ├── model-ready datasets
│       ├── locked model outputs
│       └── ablation outputs
│
├── results/
│   ├── shap/
│   │   └── cell_a/
│   └── summary_tables/
│
├── step_23A_cellA_primary_stability.py
├── step_27a_cellA_fairness_analysis.py
├── step_28a_shap_ensemble.py
├── 00_CLOSEOUT_OSAFO_SIMON_YEBOAH_20260920_READY_TO_RUN.py
│
├── Chapter_3_FINAL_SUBMISSION_MANUSCRIPT_FINAL_20260920.docx
├── Chapter_4_FINAL_SUBMISSION_MANUSCRIPT_FINAL_20260920.docx
├── Chapters_3_4_FINAL_SUBMISSION_MANUSCRIPT_FINAL_20260920.docx
│
├── README.md
├── requirements.txt
└── .gitignore
