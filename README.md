# California Housing — Regression Models
**Maincrafts Technology | AI & ML Internship**

---

## Overview

This repository contains two tasks from the AI & ML internship, both using the California Housing dataset to predict median house values. The tasks progress from a simple baseline model to a multi-model comparison with proper preprocessing.

**Dataset:** [California Housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) — built into scikit-learn, no download needed.
- 20,640 samples | 8 features | Target: median house value in $100k units

---

## Notebooks

| Task | Notebook | Description |
|---|---|---|
| Task 1 | [task1_ml_linear_regression.ipynb](https://github.com/rajathhskht-create/california-housing-regression/blob/main/task1_ml_linear_regression.ipynb) | Baseline linear regression with EDA |
| Task 2 | [task2_ml_model_comparison.ipynb](https://github.com/rajathhskht-create/california-housing-regression/blob/main/task2_ml_model_comparison.ipynb) | Feature scaling + model comparison |

---

## Task 1 — Linear Regression Baseline

**Goal:** Build a baseline regression model and understand the data.

**What's covered:**
- Exploratory Data Analysis (distributions, correlation heatmap)
- Train/test split (80/20)
- Linear Regression — no feature scaling
- Evaluation: MAE, RMSE, R²
- Visualizations: Actual vs Predicted, Residual Plot, Feature Coefficients

**Results:**

| Metric | Score |
|---|---|
| MAE | ~0.533 |
| RMSE | ~0.745 |
| R² | ~0.596 |

**Key finding:** `MedInc` (median income) has the strongest correlation with house value (~0.688). R² of ~0.60 is a reasonable baseline but linear regression is hitting its ceiling — house prices aren't purely linear.

---

## Task 2 — Feature Scaling & Model Comparison

**Goal:** Improve on Task 1 by adding proper preprocessing and comparing multiple models.

**What's covered:**
- Feature scaling with `StandardScaler`
- Three models trained and compared side by side:
  - Linear Regression (scaled baseline)
  - Ridge Regression (L2 regularisation, alpha=1.0)
  - Decision Tree Regressor (max depth=5)
- Visualizations: R² bar chart, Actual vs Predicted (all 3 models), Residual plots
- Best model saved with `joblib`

**Results:**

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | ~0.745 | ~0.533 | ~0.596 |
| Ridge Regression | ~0.745 | ~0.533 | ~0.596 |
| Decision Tree (depth=5) | ~0.694 | ~0.485 | ~0.647 |

**Best model: Decision Tree Regressor** — ~4.7 percentage point improvement over Task 1 baseline.

**Key finding:** Ridge and Linear came out nearly identical (no extreme multicollinearity in this dataset). Decision Tree outperformed both by capturing non-linear patterns between location, income, and price.

---

## How to Run

1. Clone the repo
   ```bash
   git clone https://github.com/rajathhskht-create/california-housing-regression.git
   cd california-housing-regression
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Open any notebook
   ```bash
   jupyter notebook task1_ml_linear_regression.ipynb
   # or
   jupyter notebook task2_ml_model_comparison.ipynb
   ```

4. Run all cells (`Kernel > Restart & Run All`)

---

## Repository Structure

```
california-housing-regression/
├── task1_ml_linear_regression.ipynb    # Task 1 — baseline linear regression
├── task2_ml_model_comparison.ipynb     # Task 2 — feature scaling + model comparison
├── requirements.txt                     # Python dependencies
├── .gitignore                           # Excludes .pkl model files
└── README.md                            # This file
```

> **Note:** Saved model files (`*.pkl`) are excluded via `.gitignore`. They are generated locally when you run the notebooks.

---

## Next Steps

- Random Forest / Gradient Boosting — expected to push R² to 0.80+
- Hyperparameter tuning with `GridSearchCV`
- Feature importance analysis (especially for tree-based models)

---

*Maincrafts Technology — AI & ML Internship | Tasks 1 & 2*
