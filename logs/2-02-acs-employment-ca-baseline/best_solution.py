import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
import lightgbm as lgb

# Ensure working directory exists
os.makedirs("working", exist_ok=True)

# Load data
train = pd.read_csv("input/train.csv")
test = pd.read_csv("input/test.csv")

# Features and target
feature_cols = [c for c in train.columns if c != "ESR"]
X = train[feature_cols]
y = train["ESR"].astype(int)

# Identify categorical features (<=10 unique values)
cat_cols = [c for c in feature_cols if X[c].nunique() <= 10]

# Split out a hold‐out validation set
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(sss.split(X, y))
X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

# LightGBM parameters
init_n_estimators = 1000
lgb_params = {
    "boosting_type": "dart",
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbosity": -1,
}

# Train with early stopping via callbacks
model = lgb.LGBMClassifier(n_estimators=init_n_estimators, **lgb_params)
model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="binary_logloss",
    categorical_feature=cat_cols,
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)],
)

# Determine best iteration, ensure it's >=1
best_iter = model.best_iteration_
if not best_iter or best_iter < 1:
    best_iter = init_n_estimators

# Validate
val_preds = model.predict(X_val, num_iteration=best_iter)
val_acc = accuracy_score(y_val, val_preds)
print(f"Validation Accuracy: {val_acc:.5f}")

# Retrain on full data with best_iter
final_model = lgb.LGBMClassifier(n_estimators=best_iter, **lgb_params)
final_model.fit(X, y, categorical_feature=cat_cols)

# Predict on test and save submission
test_preds = final_model.predict(test[feature_cols]).astype(int)
submission = pd.DataFrame({"ESR": test_preds})
submission.to_csv("working/submission.csv", index=False)
