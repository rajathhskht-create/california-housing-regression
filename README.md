# 🏠 California Housing Price Prediction

A machine learning project that predicts median house values across California districts using **Linear Regression**, **Ridge**, and **Lasso** models — with full EDA, feature scaling, cross-validation, and model persistence.

---

## 📊 Results

| Model              | MAE    | RMSE   | R²     | CV R² (5-fold) |
|--------------------|--------|--------|--------|----------------|
| Linear Regression  | ~0.53  | ~0.73  | ~0.60  | ~0.60          |
| Ridge Regression   | ~0.53  | ~0.73  | ~0.60  | ~0.60          |
| Lasso Regression   | ~0.53  | ~0.73  | ~0.60  | ~0.60          |

> Values in $100k units. The best model explains ~60% of variance in house prices with an average prediction error of ~$53,000.

---

## 📁 Project Structure

```
california_housing_regression/
│
├── src/
│   └── linear_regression.py     # Main ML pipeline
│
├── notebooks/
│   └── (Jupyter notebooks — optional exploration)
│
├── outputs/                     # Auto-generated (gitignored for .pkl)
│   ├── feature_distributions.png
│   ├── target_distribution.png
│   ├── correlation_heatmap.png
│   ├── actual_vs_predicted.png
│   ├── feature_coefficients.png
│   ├── best_model.pkl           # Saved model (not tracked by Git)
│   └── scaler.pkl               # Saved scaler (not tracked by Git)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔍 Dataset

**California Housing Dataset** — built into `scikit-learn` (no download needed).

| Feature      | Description                                      |
|--------------|--------------------------------------------------|
| `MedInc`     | Median income in block group                     |
| `HouseAge`   | Median house age in block group                  |
| `AveRooms`   | Average number of rooms per household            |
| `AveBedrms`  | Average number of bedrooms per household         |
| `Population` | Block group population                           |
| `AveOccup`   | Average number of household members              |
| `Latitude`   | Block group latitude                             |
| `Longitude`  | Block group longitude                            |
| `MedHouseVal`| **Target** — Median house value (in $100k)      |

- **Source**: U.S. Census 1990, California
- **Samples**: 20,640
- **Features**: 8

---

## ⚙️ Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/california_housing_regression.git
cd california_housing_regression
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the pipeline
```bash
python src/linear_regression.py
```

Plots are saved to `outputs/`. The best model and scaler are saved as `.pkl` files.

---

## 🧪 What the Pipeline Does

1. **Load** — Fetches the California Housing dataset via `sklearn.datasets`
2. **EDA** — Feature distributions, target distribution, correlation heatmap
3. **Preprocess** — Train/test split (80/20), feature scaling (`StandardScaler`)
4. **Train** — Fits Linear Regression, Ridge, and Lasso models
5. **Evaluate** — MAE, RMSE, R², and 5-fold cross-validation
6. **Visualize** — Actual vs Predicted plot, Residual plot, Coefficient chart
7. **Save** — Best model and scaler persisted with `joblib`
8. **Inference** — Sample predictions from the saved model

---

## 📈 Key Visualizations

| Plot | Description |
|------|-------------|
| Feature Distributions | Histogram of each input feature |
| Correlation Heatmap | Pairwise correlations including target |
| Actual vs Predicted | Scatter plot of true vs model predictions |
| Residual Plot | Checks for heteroscedasticity |
| Feature Coefficients | Importance and direction of each feature |

---

## 🛠️ Tech Stack

- **Python** 3.9+
- **scikit-learn** — Models, preprocessing, evaluation
- **pandas / numpy** — Data manipulation
- **matplotlib / seaborn** — Visualizations
- **joblib** — Model persistence
