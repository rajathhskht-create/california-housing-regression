"""
California Housing Price Prediction
====================================
Linear Regression model with EDA, evaluation, and model persistence.
Dataset: California Housing (scikit-learn built-in)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Config
plt.rcParams['figure.dpi'] = 120
sns.set_theme(style='whitegrid')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load Data 
print("=" * 50)
print("  California Housing Price Prediction")
print("=" * 50)

data = fetch_california_housing(as_frame=True)
df = pd.concat([data.data, data.target.rename('MedHouseVal')], axis=1)

print(f'\n[1] Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns')
print('\nFeature descriptions:')
print(data.DESCR[:800])

# 2. EDA 
print('\n[2] Exploratory Data Analysis')
print(df.describe().round(2))

print('\nMissing values:')
print(df.isnull().sum())

# Feature distributions
features = data.feature_names
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, feat in enumerate(features):
    axes[i // 4][i % 4].hist(df[feat], bins=40, color='steelblue',
                               edgecolor='white', alpha=0.85)
    axes[i // 4][i % 4].set_title(feat, fontsize=10, fontweight='bold')
plt.suptitle('Feature Distributions – California Housing', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_distributions.png'))
plt.show()

# Target distribution
plt.figure(figsize=(7, 4))
df['MedHouseVal'].hist(bins=50, color='coral', edgecolor='white')
plt.title('Target: Median House Value Distribution', fontweight='bold')
plt.xlabel('Median House Value ($100k)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'target_distribution.png'))
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt='.2f',
            cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap.png'))
plt.show()

print('\nCorrelation with target (MedHouseVal):')
print(df.corr(numeric_only=True)['MedHouseVal'].sort_values(ascending=False).round(3))

# 3. Preprocessing 
print('\n[3] Preprocessing')

X = df.drop(columns='MedHouseVal')
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f'  Training samples : {len(X_train):,}')
print(f'  Testing  samples : {len(X_test):,}')
print('  Features scaled with StandardScaler ✓')

# 4. Model Training & Comparison 
print('\n[4] Model Training')

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression':  Ridge(alpha=1.0),
    'Lasso Regression':  Lasso(alpha=0.01),
}

results = {}
for name, mdl in models.items():
    mdl.fit(X_train_scaled, y_train)
    y_pred = mdl.predict(X_test_scaled)
    cv_scores = cross_val_score(mdl, X_train_scaled, y_train,
                                cv=5, scoring='r2')
    results[name] = {
        'model':   mdl,
        'y_pred':  y_pred,
        'MAE':     mean_absolute_error(y_test, y_pred),
        'RMSE':    np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2':      r2_score(y_test, y_pred),
        'CV_R2':   cv_scores.mean(),
        'CV_Std':  cv_scores.std(),
    }

# 5. Evaluation 
print('\n[5] Model Evaluation')
print(f'\n{"Model":<22} {"MAE":>7} {"RMSE":>7} {"R²":>7} {"CV R²":>8} {"CV Std":>8}')
print('-' * 65)
for name, r in results.items():
    print(f'{name:<22} {r["MAE"]:>7.4f} {r["RMSE"]:>7.4f} '
          f'{r["R2"]:>7.4f} {r["CV_R2"]:>8.4f} {r["CV_Std"]:>8.4f}')

# Best model = highest R²
best_name = max(results, key=lambda k: results[k]['R2'])
best = results[best_name]
print(f'\n  Best model : {best_name}')
print(f'  R²         : {best["R2"]:.4f}  → explains {best["R2"]*100:.1f}% of variance')
print(f'  MAE        : {best["MAE"]:.4f} → predictions off by ~${best["MAE"]*100:.0f}k on average')

# 6. Visualisations 
print('\n[6] Generating plots')

y_pred_best = best['y_pred']
residuals   = y_test - y_pred_best

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Actual vs Predicted
axes[0].scatter(y_test, y_pred_best, alpha=0.3, color='steelblue', s=8)
lims = [min(y_test.min(), y_pred_best.min()) - 0.1,
        max(y_test.max(), y_pred_best.max()) + 0.1]
axes[0].plot(lims, lims, 'r--', linewidth=1.8, label='Perfect Fit')
axes[0].set_xlabel('Actual Value ($100k)', fontsize=11)
axes[0].set_ylabel('Predicted Value ($100k)', fontsize=11)
axes[0].set_title(f'Actual vs Predicted – {best_name}', fontweight='bold')
axes[0].legend()

# Residuals
axes[1].scatter(y_pred_best, residuals, alpha=0.3, color='darkorange', s=8)
axes[1].axhline(0, color='red', linestyle='--', linewidth=1.8)
axes[1].set_xlabel('Predicted Value ($100k)', fontsize=11)
axes[1].set_ylabel('Residual', fontsize=11)
axes[1].set_title('Residual Plot', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'actual_vs_predicted.png'))
plt.show()

# Feature Coefficients (best model)
coef = best['model'].coef_
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': coef})
coef_df = coef_df.sort_values('Coefficient', key=abs, ascending=False)
colors = ['steelblue' if c > 0 else 'tomato' for c in coef_df['Coefficient']]

plt.figure(figsize=(8, 5))
plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
plt.axvline(0, color='black', linewidth=0.8)
plt.title(f'Feature Coefficients – {best_name}', fontweight='bold')
plt.xlabel('Coefficient Value (scaled features)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_coefficients.png'))
plt.show()

# 7. Save Model & Scaler
print('\n[7] Saving model and scaler')
joblib.dump(best['model'], os.path.join(OUTPUT_DIR, 'best_model.pkl'))
joblib.dump(scaler,        os.path.join(OUTPUT_DIR, 'scaler.pkl'))
print(f'  Saved: outputs/best_model.pkl  ({best_name})')
print(f'  Saved: outputs/scaler.pkl')

# 8. Quick Inference Test
print('\n[8] Sample predictions from saved model')
loaded_model  = joblib.load(os.path.join(OUTPUT_DIR, 'best_model.pkl'))
loaded_scaler = joblib.load(os.path.join(OUTPUT_DIR, 'scaler.pkl'))

sample   = X_test.iloc[:5]
s_scaled = loaded_scaler.transform(sample)
preds    = loaded_model.predict(s_scaled)

print(f'\n  {"Sample":<8} {"Predicted":>12} {"Actual":>12} {"Error":>10}')
print('  ' + '-' * 46)
for i, p in enumerate(preds):
    actual = y_test.iloc[i]
    print(f'  {i+1:<8} ${p*100:>8.1f}k   ${actual*100:>8.1f}k  '
          f'  {abs(p-actual)*100:>6.1f}k')

print('\n✓ Pipeline complete.')
