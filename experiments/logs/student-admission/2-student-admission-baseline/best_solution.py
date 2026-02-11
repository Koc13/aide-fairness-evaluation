import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from scipy.stats import randint, uniform
from lightgbm import LGBMClassifier

# Load data
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")

# Separate features and target
X = train.drop("admitted", axis=1)
y = train["admitted"]

# One-hot encode categorical variables
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
X_enc = pd.get_dummies(X, columns=cat_cols)
X_test = test.drop("admitted", axis=1) if "admitted" in test.columns else test.copy()
X_test_enc = pd.get_dummies(
    X_test, columns=X_test.select_dtypes(include=["object"]).columns
)
X_test_enc = X_test_enc.reindex(columns=X_enc.columns, fill_value=0)

# Set up stratified CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Hyperparameter distributions for LightGBM
param_dist = {
    "num_leaves": randint(20, 150),
    "n_estimators": randint(100, 500),
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [-1] + list(range(3, 16)),
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}

# Randomized search
lgb = LGBMClassifier(random_state=42, n_jobs=-1)
rs = RandomizedSearchCV(
    lgb,
    param_distributions=param_dist,
    n_iter=20,
    scoring="accuracy",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    refit=True,
)
rs.fit(X_enc, y)

# Print best CV accuracy
print(f"Best CV accuracy: {rs.best_score_:.4f}")
print("Best parameters:", rs.best_params_)

# Retrain on full training set (already refit) and predict on test
best_lgb = rs.best_estimator_
preds_test = best_lgb.predict(X_test_enc).astype(int)
submission = pd.DataFrame({"admitted": preds_test})
submission.to_csv("./working/submission.csv", index=False)
