# DSAA2011 Final Project — Student Dropout Dataset

This repository contains a complete end-to-end machine learning project for **DSAA2011 (Spring 2026)** using the selected dataset **`data.csv`** (Student Dropout Dataset).

## What is included

- **Full project analysis pipeline**: `analysis.py`
- **Generated experiment outputs**: `results/`
  - `model_metrics.csv`
  - `clustering_metrics.csv`
  - `embedding_points.csv`
  - `roc_lr.csv`, `roc_knn.csv`
  - `summary.txt`
- **Final report (LaTeX source using provided template)**: `Report/neurips_2025.tex`

## Tasks completed (aligned with project guideline)

### 1) Data preprocessing
- Loaded dataset from `data.csv` (semicolon-separated, UTF-8 with BOM handling).
- Parsed all 36 predictors as numeric values.
- Encoded prediction target as binary:
  - `Dropout = 1`
  - `Graduate/Enrolled = 0`
- Applied z-score standardization to all features.

### 2) Data visualization
- Produced a 2D embedding dataset (`results/embedding_points.csv`) for visual cluster inspection.
- In this environment (no external scientific packages), a PCA-based embedding was implemented directly in Python as a deterministic projection baseline.

### 3) Clustering analysis
- Implemented and compared **two clustering algorithms** from scratch:
  - K-Means
  - K-Medoids
- Evaluated with multiple metrics:
  - Silhouette score
  - Davies–Bouldin index

### 4) Prediction (training and testing)
- Classification target: dropout risk.
- Implemented and evaluated two models:
  - Logistic Regression (from scratch, L2-regularized)
  - k-NN classifier
- Performed stratified train/test split (80/20).
- Reported confusion matrix values and metrics.

### 5) Evaluation and model selection
- Computed:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC curve and AUC
- Generated test-set ROC points for each model in CSV format.
- Selected the stronger model based on generalization metrics.

### 6) Open-ended exploration
- Added algorithmic comparison beyond a single baseline model.
- Included additional unsupervised comparison (K-Means vs K-Medoids).

---

## How to run

From repository root:

```bash
python analysis.py
```

This will regenerate all files under `results/`.

---

## Key results snapshot

From `results/summary.txt`:
- Dataset size: 4,424 samples, 36 features.
- Class counts: Dropout = 1,421; Graduate = 2,209; Enrolled = 794.
- Logistic Regression (test): Accuracy 0.8713, F1 0.7833, AUC 0.9047.
- k-NN (test): Accuracy 0.8363, F1 0.6742, AUC 0.8842.
- Clustering: K-Means outperformed K-Medoids on both silhouette and DB index.

---

## Notes

- The project environment does not provide pandas/scikit-learn/matplotlib; therefore, all core algorithms and metrics were implemented directly in Python standard library for full reproducibility.
- Report content follows the required structure in the announcement and is written in the provided NeurIPS-style template source.
