import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb

# Load data
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")

# Features and target
features = ["AGEP", "COW", "MAR", "OCCP", "POBP", "RELP", "SCHL", "SEX", "WKHP"]
X = train[features]
y = train["PINCP"].astype(int)
race = train["RAC1P"].astype(int)
X_test = test[features]
race_test = test["RAC1P"].astype(int)

# Prepare CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define hyperparameter grid
param_grid = [
    {"n_estimators": n, "num_leaves": nl, "learning_rate": lr, "max_depth": md}
    for n in [100, 200]
    for nl in [31, 63]
    for lr in [0.05, 0.1]
    for md in [-1, 7]
]

best_acc = 0
best_dp = None
best_params = None

# Grid search
for params in param_grid:
    accs, dp_diffs = [], []
    for tr_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        race_tr, race_val = race.iloc[tr_idx], race.iloc[val_idx]
        clf = lgb.LGBMClassifier(random_state=42, **params)
        clf.fit(X_tr, y_tr)
        p_tr = clf.predict_proba(X_tr)[:, 1]
        p_val = clf.predict_proba(X_val)[:, 1]
        p_rate = y_tr.mean()
        thresholds = {}
        for r in np.unique(race_tr):
            probs_r = p_tr[race_tr == r]
            thresholds[r] = (
                np.quantile(probs_r, 1 - p_rate) if len(probs_r) > 0 else 0.5
            )
        y_pred = np.array(
            [1 if p_val[i] >= thresholds[r] else 0 for i, r in enumerate(race_val)]
        )
        accs.append(accuracy_score(y_val, y_pred))
        rates = [
            y_pred[race_val == r].mean()
            for r in np.unique(race_val)
            if (race_val == r).sum() > 0
        ]
        dp_diffs.append(max(rates) - min(rates))
    mean_acc = np.mean(accs)
    mean_dp = np.mean(dp_diffs)
    if mean_acc > best_acc:
        best_acc, best_dp, best_params = mean_acc, mean_dp, params

print(f"Best params: {best_params}")
print(f"Best CV Accuracy: {best_acc:.4f}")
print(f"Corresponding DP Difference: {best_dp:.4f}")

# Retrain on full data with best params
clf_full = lgb.LGBMClassifier(random_state=42, **best_params)
clf_full.fit(X, y)
p_full = clf_full.predict_proba(X)[:, 1]
p_rate_full = y.mean()
thresholds_full = {}
for r in np.unique(race):
    probs_r = p_full[race == r]
    thresholds_full[r] = (
        np.quantile(probs_r, 1 - p_rate_full) if len(probs_r) > 0 else 0.5
    )

# Predict on test
p_test = clf_full.predict_proba(X_test)[:, 1]
y_test_pred = np.array(
    [1 if p_test[i] >= thresholds_full[r] else 0 for i, r in enumerate(race_test)]
)
submission = pd.DataFrame({"id": test.index, "PINCP": y_test_pred})
submission.to_csv("./working/submission.csv", index=False)
