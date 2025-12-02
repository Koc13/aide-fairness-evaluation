import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.metrics import demographic_parity_difference

# Configuration
DATASET_DIR = "datasets/acs-income-ca"
TRAIN_PATH = os.path.join(DATASET_DIR, "train.csv")
TEST_PATH = os.path.join(DATASET_DIR, "test.csv")
TARGET_COL = "PINCP"
SENSITIVE_COL = "RAC1P"

# Features used by the "Fairness Prompted" AIDE solution
# Note: RAC1P is NOT in this list (Fairness through Unawareness)
FEATURES = ["AGEP", "COW", "MAR", "OCCP", "POBP", "RELP", "SCHL", "SEX", "WKHP"]
CATEGORICAL_COLS = ["COW", "MAR", "RELP", "SCHL", "SEX"] # Removed RAC1P

def main():
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Prepare X and y
    X = train_df[FEATURES]
    y = train_df[TARGET_COL].astype(int)
    
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET_COL].astype(int)

    # Sensitive attributes (kept separate)
    A = train_df[SENSITIVE_COL]
    A_test = test_df[SENSITIVE_COL]

    # Stratification Key
    stratify_col = y.astype(str) + "_" + A.astype(str)

    # Split train into train_main (for model) and train_calib (for threshold optimizer)
    X_train_main, X_train_calib, y_train_main, y_train_calib = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify_col
    )
    
    # We also need the sensitive attribute for the calibration set
    # We can't just split A because we need the indices to match.
    # The easiest way is to split the indices or dataframe, but train_test_split returns copies.
    # Let's re-do the split including A in the dataframe to keep it aligned.
    
    # Combine X and A for splitting
    X_with_A = X.copy()
    X_with_A[SENSITIVE_COL] = A
    
    X_train_main_full, X_train_calib_full, y_train_main, y_train_calib = train_test_split(
        X_with_A, y, test_size=0.25, random_state=42, stratify=stratify_col
    )
    
    # Separate A back out
    A_train_calib = X_train_calib_full[SENSITIVE_COL]
    X_train_main = X_train_main_full.drop(SENSITIVE_COL, axis=1)
    X_train_calib = X_train_calib_full.drop(SENSITIVE_COL, axis=1)

    # Encoding
    print("Encoding features...")
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    
    X_train_main_enc = X_train_main.copy()
    X_train_main_enc[CATEGORICAL_COLS] = encoder.fit_transform(X_train_main[CATEGORICAL_COLS])
    
    X_train_calib_enc = X_train_calib.copy()
    X_train_calib_enc[CATEGORICAL_COLS] = encoder.transform(X_train_calib[CATEGORICAL_COLS])
    
    X_test_enc = X_test.copy()
    X_test_enc[CATEGORICAL_COLS] = encoder.transform(X_test[CATEGORICAL_COLS])

    # Train Model (Mimicking Fairness Prompted Params)
    print("Training LightGBM (Fairness Config)...")
    # Params from AIDE's best_solution.py
    # {'n_estimators': 200, 'num_leaves': 31, 'learning_rate': 0.05, 'max_depth': 7}
    lgbm = lgb.LGBMClassifier(
        n_estimators=200,
        num_leaves=31,
        learning_rate=0.05,
        max_depth=7,
        random_state=42,
        n_jobs=-1
    )
    
    lgbm.fit(
        X_train_main_enc, 
        y_train_main, 
        categorical_feature=CATEGORICAL_COLS
    )

    # Evaluate Baseline (Unawareness)
    print("Evaluating Unawareness Baseline...")
    y_pred_baseline = lgbm.predict(X_test_enc)
    acc_baseline = accuracy_score(y_test, y_pred_baseline)
    dp_diff_baseline = demographic_parity_difference(y_test, y_pred_baseline, sensitive_features=A_test)
    
    print(f"Unawareness Accuracy: {acc_baseline:.4f}")
    print(f"Unawareness DP Diff:  {dp_diff_baseline:.4f}")

    # Post-processing with ThresholdOptimizer
    print("\nRunning ThresholdOptimizer on Unawareness Model...")
    optimizer = ThresholdOptimizer(
        estimator=lgbm,
        constraints="demographic_parity",
        predict_method='predict_proba',
        prefit=True
    )

    optimizer.fit(X_train_calib_enc, y_train_calib, sensitive_features=A_train_calib)

    print("Predicting with Fairlearn...")
    y_pred_fair = optimizer.predict(X_test_enc, sensitive_features=A_test)

    acc_fair = accuracy_score(y_test, y_pred_fair)
    dp_diff_fair = demographic_parity_difference(y_test, y_pred_fair, sensitive_features=A_test)

    print(f"Fairlearn Accuracy: {acc_fair:.4f}")
    print(f"Fairlearn DP Diff:  {dp_diff_fair:.4f}")
    
    print("\nComparison vs AIDE Manual Heuristic (approx 0.28 bias):")
    print(f"Fairlearn Improvement: {0.28 - dp_diff_fair:.4f} (if positive, Fairlearn is better)")

if __name__ == "__main__":
    main()
