# DSAA2011 Final Project — Student Dropout Dataset (Refined Version)

This refined submission now includes a **full Jupyter Notebook pipeline** using modern ML/visualization libraries (**numpy, pandas, matplotlib, seaborn, scikit-learn**) and supports polished figures plus full multiclass confusion-matrix analysis.

## ✅ Main deliverables

- `project.ipynb` (**primary deliverable**): complete end-to-end notebook workflow.
- `analysis.py`: lightweight script version (kept for reproducibility in constrained environments).
- `Report/neurips_2025.tex`: report source.
- `requirements.txt`: dependency list for notebook execution.
- `results_ml/` (created after running notebook): polished figures and tables.

## What is now covered (per your feedback)

### 1) Jupyter notebook deliverable
A full `.ipynb` file is provided: **`project.ipynb`**.

### 2) Library-based workflow
Notebook uses:
- `pandas` / `numpy` for data handling
- `matplotlib` / `seaborn` for polished visualizations
- `scikit-learn` for preprocessing, t-SNE, clustering, classification, ROC/AUC, and validation

### 3) Polished visual outputs
Notebook exports:
- PCA projection plot
- t-SNE projection plot
- KMeans cluster plot on t-SNE space
- ROC curves (OvR) for multiclass setup
- Confusion matrix heatmaps for each model on train/test/all

### 4) Full multiclass confusion-matrix analysis
The notebook evaluates **all target categories** (`Dropout`, `Enrolled`, `Graduate`) and exports:
- confusion matrices per split and model
- per-class precision/recall/F1 (classification report)
- macro and weighted metrics

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook project.ipynb
```

Run all notebook cells to generate final outputs in `results_ml/`.

## Output structure after notebook execution

- `results_ml/figures/*.png`
- `results_ml/tables/*.csv`

Key exported tables include:
- `model_performance.csv`
- `clustering_metrics.csv`
- `validation_summary.csv`
- `confusion_matrix_<model>_<split>.csv`
- `classification_report_<model>_<split>.csv`

## Important note
If your grading environment already has these dependencies available, just open and run `project.ipynb` directly. If not, install from `requirements.txt` first.
