import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.metrics import (
    demographic_parity_difference,
    true_positive_rate_difference,
)

# Load data
train = pd.read_csv("input/train.csv")
test = pd.read_csv("input/test.csv")

sensitive_cols = ["race", "gender", "socioeconomic_status"]

# Prepare features and labels
y = train["admitted"]
X = train.drop(columns=["admitted"] + sensitive_cols)
X_test = test.drop(columns=["admitted"] + sensitive_cols)
S = train[sensitive_cols].astype(str)
S_test = test[sensitive_cols].astype(str)

# One-hot encode non-sensitive categorical features
X = pd.get_dummies(X, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)
X_test = X_test.reindex(columns=X.columns, fill_value=0)

# Split for validation
X_train, X_val, y_train, y_val, S_train, S_val = train_test_split(
    X, y, S, test_size=0.2, random_state=42, stratify=y
)

# Fit ExponentiatedGradient with DemographicParity constraint
base_est = LogisticRegression(solver="liblinear", max_iter=1000, random_state=42)
constraint = DemographicParity(difference_bound=0.15)
mitigator = ExponentiatedGradient(base_est, constraint, max_iter=50)
mitigator.fit(X_train, y_train, sensitive_features=S_train)

# Validate
y_pred_val = mitigator.predict(X_val)
acc = accuracy_score(y_val, y_pred_val)

# Compute fairness metrics for each sensitive feature
dp_vals = [
    demographic_parity_difference(y_val, y_pred_val, sensitive_features=S_val[col])
    for col in sensitive_cols
]
eo_vals = [
    true_positive_rate_difference(y_val, y_pred_val, sensitive_features=S_val[col])
    for col in sensitive_cols
]
dp_max = max(dp_vals)
eo_max = max(eo_vals)

# Final score penalizes violations beyond targets
final_score = acc - max(0, dp_max - 0.15) - max(0, eo_max - 0.10)

print(f"Validation Accuracy: {acc:.4f}")
print(f"Demographic Parity Difference: {dp_max:.4f}")
print(f"Equal Opportunity Difference: {eo_max:.4f}")
print(f"Final Combined Score: {final_score:.4f}")

# Predict on test and save submission
y_pred_test = mitigator.predict(X_test)
submission = pd.DataFrame({"admitted": y_pred_test})
submission.to_csv("./working/submission.csv", index=False)
