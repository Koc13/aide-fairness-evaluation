import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import lightgbm as lgb
from fairlearn.postprocessing import ThresholdOptimizer

# Load data
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")

# Prepare features, target, sensitive attribute
X = train.drop(["PINCP", "RAC1P"], axis=1)
y = train["PINCP"].astype(int)
race = train["RAC1P"]

# Split into train/validation
X_train, X_val, y_train, y_val, race_train, race_val = train_test_split(
    X, y, race, test_size=0.2, stratify=y, random_state=42
)

# Train LightGBM classifier
model = lgb.LGBMClassifier(random_state=42)
model.fit(X_train, y_train)

# Post-process with Fairlearn ThresholdOptimizer for demographic parity (epsilon=0.05)
to = ThresholdOptimizer(
    estimator=model,
    constraints="demographic_parity",
    predict_method="predict_proba",
    prefit=True,
    grid_size=100,
)
to.fit(X_train, y_train, sensitive_features=race_train)

# Predict on validation
y_val_pred = to.predict(X_val, sensitive_features=race_val)

# Compute balanced accuracy
bal_acc = balanced_accuracy_score(y_val, y_val_pred)

# Compute demographic parity difference
group_rates = []
for r in np.unique(race_val):
    mask = race_val == r
    if mask.sum() > 0:
        group_rates.append(y_val_pred[mask].mean())
dp_diff = max(group_rates) - min(group_rates)

# Compute equalized odds difference
tprs, fprs = [], []
for r in np.unique(race_val):
    mask = race_val == r
    if mask.sum() == 0:
        tprs.append(0)
        fprs.append(0)
        continue
    y_true_r = y_val[mask].values
    y_pred_r = y_val_pred[mask]
    tp = ((y_pred_r == 1) & (y_true_r == 1)).sum()
    fn = ((y_pred_r == 0) & (y_true_r == 1)).sum()
    fp = ((y_pred_r == 1) & (y_true_r == 0)).sum()
    tn = ((y_pred_r == 0) & (y_true_r == 0)).sum()
    tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
    fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
eq_odds_diff = (max(tprs) - min(tprs)) + (max(fprs) - min(fprs))

# Print metrics
print(f"Balanced Accuracy: {bal_acc:.4f}")
print(f"Demographic Parity Difference: {dp_diff:.4f}")
print(f"Equalized Odds Difference: {eq_odds_diff:.4f}")

# Predict on test set and save submission
X_test = test.drop([c for c in ["PINCP", "RAC1P"] if c in test.columns], axis=1)
race_test = test["RAC1P"]
y_test_pred = to.predict(X_test, sensitive_features=race_test)

submission = pd.DataFrame({"PINCP": y_test_pred})
os.makedirs("./working", exist_ok=True)
submission.to_csv("./working/submission.csv", index=False)
