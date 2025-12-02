"""
Fairness analysis for the baseline Boston Housing model
Analyzes if predictions are biased across racial demographics (B feature)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, max_error
import lightgbm as lgb

# Load data
data = pd.read_csv("workspaces/boston-baseline/input/train.csv")
X = data.drop("MEDV", axis=1)
y = data["MEDV"]

# Split into train and validation sets (same as baseline)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train a simple model (using best params from AIDE would be better, but this is quick)
model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42, verbose=-1)
model.fit(X_train_scaled, y_train)

# Get predictions
y_pred = model.predict(X_val_scaled)

# Overall RMSE
overall_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Overall Validation RMSE: {overall_rmse:.4f}\n")

# Fairness Analysis: Split by racial demographic (B feature)
# The B feature represents: 1000(Bk - 0.63)^2 where Bk is proportion of Black residents
# Higher B = higher proportion of Black residents
X_val_reset = X_val.reset_index(drop=True)
y_val_reset = y_val.reset_index(drop=True)

# Create groups based on B feature (median split)
b_median = X_val_reset['B'].median()
high_b_mask = X_val_reset['B'] >= b_median  # Higher proportion Black residents
low_b_mask = X_val_reset['B'] < b_median    # Lower proportion Black residents

# Calculate RMSE for each group
y_val_high_b = y_val_reset[high_b_mask]
y_pred_high_b = y_pred[high_b_mask]
rmse_high_b = np.sqrt(mean_squared_error(y_val_high_b, y_pred_high_b))

y_val_low_b = y_val_reset[low_b_mask]
y_pred_low_b = y_pred[low_b_mask]
rmse_low_b = np.sqrt(mean_squared_error(y_val_low_b, y_pred_low_b))

print("=" * 60)
print("FAIRNESS ANALYSIS BY RACIAL DEMOGRAPHICS")
print("=" * 60)
print(f"\nHigh B group (higher % Black residents):")
print(f"  - Count: {len(y_val_high_b)}")
print(f"  - RMSE: {rmse_high_b:.4f}")
print(f"\nLow B group (lower % Black residents):")
print(f"  - Count: {len(y_val_low_b)}")
print(f"  - RMSE: {rmse_low_b:.4f}")
print(f"\nRMSE Difference: {abs(rmse_high_b - rmse_low_b):.4f}")
print(f"RMSE Ratio: {max(rmse_high_b, rmse_low_b) / min(rmse_high_b, rmse_low_b):.2f}x")

# Also check mean prediction error per group
mean_error_high_b = np.mean(y_pred_high_b - y_val_high_b)
mean_error_low_b = np.mean(y_pred_low_b - y_val_low_b)

print(f"\nMean Prediction Error (bias):")
print(f"  - High B group: {mean_error_high_b:.4f}")
print(f"  - Low B group: {mean_error_low_b:.4f}")
print(f"  - Difference: {abs(mean_error_high_b - mean_error_low_b):.4f}")

# Additional fairness metrics
print("\n" + "-" * 60)
print("ADDITIONAL FAIRNESS METRICS")
print("-" * 60)

# MAE (Mean Absolute Error) per group
mae_high_b = mean_absolute_error(y_val_high_b, y_pred_high_b)
mae_low_b = mean_absolute_error(y_val_low_b, y_pred_low_b)
print(f"\nMean Absolute Error (MAE):")
print(f"  - High B group: {mae_high_b:.4f}")
print(f"  - Low B group: {mae_low_b:.4f}")
print(f"  - Difference: {abs(mae_high_b - mae_low_b):.4f}")
print(f"  - Ratio: {max(mae_high_b, mae_low_b) / min(mae_high_b, mae_low_b):.2f}x")

# R² score per group (predictive quality)
r2_high_b = r2_score(y_val_high_b, y_pred_high_b)
r2_low_b = r2_score(y_val_low_b, y_pred_low_b)
print(f"\nR² Score (predictive quality):")
print(f"  - High B group: {r2_high_b:.4f}")
print(f"  - Low B group: {r2_low_b:.4f}")
print(f"  - Difference: {abs(r2_high_b - r2_low_b):.4f}")

# Max error per group (worst-case fairness)
max_err_high_b = max_error(y_val_high_b, y_pred_high_b)
max_err_low_b = max_error(y_val_low_b, y_pred_low_b)
print(f"\nMax Error (worst prediction):")
print(f"  - High B group: {max_err_high_b:.4f}")
print(f"  - Low B group: {max_err_low_b:.4f}")
print(f"  - Difference: {abs(max_err_high_b - max_err_low_b):.4f}")

# Average prediction per group (demographic parity-like)
avg_pred_high_b = np.mean(y_pred_high_b)
avg_pred_low_b = np.mean(y_pred_low_b)
avg_actual_high_b = np.mean(y_val_high_b)
avg_actual_low_b = np.mean(y_val_low_b)
print(f"\nAverage Predictions (demographic parity check):")
print(f"  - High B group: pred={avg_pred_high_b:.2f}, actual={avg_actual_high_b:.2f}")
print(f"  - Low B group: pred={avg_pred_low_b:.2f}, actual={avg_actual_low_b:.2f}")
print(f"  - Prediction gap: {abs(avg_pred_high_b - avg_pred_low_b):.2f}")
print(f"  - Actual gap: {abs(avg_actual_high_b - avg_actual_low_b):.2f}")

print("\n" + "=" * 60)
