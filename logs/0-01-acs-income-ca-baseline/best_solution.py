import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb

# Load data
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")

# Features and target
X = train.drop("PINCP", axis=1)
y = train["PINCP"].astype(int)
X_test = test.drop("PINCP", axis=1)

# Categorical columns
categorical_cols = ["COW", "MAR", "RAC1P", "RELP", "SCHL", "SEX"]

# Cross-validation setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Hyperparameter grid
learning_rates = [0.05, 0.1]
num_leaves_list = [31, 63, 127]
feature_fractions = [0.8, 1.0]
bagging_fractions = [0.8, 1.0]

best_acc = 0.0
best_params = {}
best_iter = 0

for lr in learning_rates:
    for nl in num_leaves_list:
        for ff in feature_fractions:
            for bf in bagging_fractions:
                fold_accuracies = []
                fold_iters = []
                for train_idx, val_idx in skf.split(X, y):
                    X_tr, X_val = X.iloc[train_idx].reset_index(drop=True), X.iloc[
                        val_idx
                    ].reset_index(drop=True)
                    y_tr, y_val = y.iloc[train_idx].reset_index(drop=True), y.iloc[
                        val_idx
                    ].reset_index(drop=True)

                    # Ordinal encoding
                    encoder = OrdinalEncoder()
                    X_tr_enc = X_tr.copy()
                    X_val_enc = X_val.copy()
                    X_tr_enc[categorical_cols] = encoder.fit_transform(
                        X_tr[categorical_cols]
                    )
                    X_val_enc[categorical_cols] = encoder.transform(
                        X_val[categorical_cols]
                    )

                    # Train model with early stopping and subsampling
                    clf = lgb.LGBMClassifier(
                        n_estimators=10000,
                        learning_rate=lr,
                        num_leaves=nl,
                        feature_fraction=ff,
                        bagging_fraction=bf,
                        bagging_freq=5,
                        random_state=42,
                    )
                    clf.fit(
                        X_tr_enc,
                        y_tr,
                        eval_set=[(X_val_enc, y_val)],
                        eval_metric="binary_error",
                        categorical_feature=categorical_cols,
                        callbacks=[
                            lgb.early_stopping(stopping_rounds=50),
                            lgb.log_evaluation(period=0),
                        ],
                    )

                    preds = clf.predict(X_val_enc)
                    fold_accuracies.append(accuracy_score(y_val, preds))
                    fold_iters.append(clf.best_iteration_)

                mean_acc = np.mean(fold_accuracies)
                mean_iter = int(np.mean(fold_iters))
                print(
                    f"Params lr={lr}, nl={nl}, ff={ff}, bf={bf} -> CV Acc = {mean_acc:.4f}, Iter = {mean_iter}"
                )
                if mean_acc > best_acc:
                    best_acc = mean_acc
                    best_params = {
                        "learning_rate": lr,
                        "num_leaves": nl,
                        "feature_fraction": ff,
                        "bagging_fraction": bf,
                    }
                    best_iter = mean_iter

print(
    f"Best CV Accuracy: {best_acc:.4f} with params {best_params} and n_estimators={best_iter}"
)

# Retrain on full data with best params
encoder_full = OrdinalEncoder()
X_full_enc = X.copy()
X_full_enc[categorical_cols] = encoder_full.fit_transform(X_full_enc[categorical_cols])
X_test_enc = X_test.copy()
X_test_enc[categorical_cols] = encoder_full.transform(X_test_enc[categorical_cols])

final_clf = lgb.LGBMClassifier(
    n_estimators=best_iter,
    learning_rate=best_params["learning_rate"],
    num_leaves=best_params["num_leaves"],
    feature_fraction=best_params["feature_fraction"],
    bagging_fraction=best_params["bagging_fraction"],
    bagging_freq=5,
    random_state=42,
)
final_clf.fit(X_full_enc, y, categorical_feature=categorical_cols)

# Predict on test set and save submission
test_preds = final_clf.predict(X_test_enc).astype(int)
submission = pd.DataFrame({"PINCP": test_preds})
os.makedirs("./working", exist_ok=True)
submission.to_csv("./working/submission.csv", index=False)
