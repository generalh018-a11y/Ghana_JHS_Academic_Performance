
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

The final experiment follows a term-level temporal prediction design:

```text
Term 1 academic + attendance information
                    +
Term 2 academic + attendance information
                    |
                    v
             Model prediction
                    |
                    v
             Term 3 at-risk status