import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

# Configuration
DATASET_DIR = "datasets/acs-income-ca"
TRAIN_PATH = os.path.join(DATASET_DIR, "train.csv")
TEST_PATH = os.path.join(DATASET_DIR, "test.csv")
TARGET_COL = "PINCP"
SENSITIVE_COL = "RAC1P"
CATEGORICAL_COLS = ["COW", "MAR", "RAC1P", "RELP", "SCHL", "SEX"]

def main():
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Prepare X and y
    X = train_df.drop(TARGET_COL, axis=1)
    y = train_df[TARGET_COL].astype(int)
    
    X_test = test_df.drop(TARGET_COL, axis=1)
    y_test = test_df[TARGET_COL].astype(int)

    # Create a stratification key combining Target and Sensitive Attribute
    # This ensures every race/outcome combination is represented in both splits
    # We use the sensitive column from X (before splitting)
    stratify_col = y.astype(str) + "_" + X[SENSITIVE_COL].astype(str)

    # Split train into train_main (for model) and train_calib (for threshold optimizer)
    # Using 75/25 split
    X_train_main, X_train_calib, y_train_main, y_train_calib = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify_col
    )

    # Extract sensitive features for calibration and testing
    # Note: We need to do this BEFORE encoding if we want to preserve original values, 
    # but for the optimizer it just needs a vector.
    A_train_calib = X_train_calib[SENSITIVE_COL]
    A_test = X_test[SENSITIVE_COL]

    # Encoding
    print("Encoding features...")
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    
    # Fit on main training data
    X_train_main_enc = X_train_main.copy()
    X_train_main_enc[CATEGORICAL_COLS] = encoder.fit_transform(X_train_main[CATEGORICAL_COLS])
    
    # Transform calibration and test data
    X_train_calib_enc = X_train_calib.copy()
    X_train_calib_enc[CATEGORICAL_COLS] = encoder.transform(X_train_calib[CATEGORICAL_COLS])
    
    X_test_enc = X_test.copy()
    X_test_enc[CATEGORICAL_COLS] = encoder.transform(X_test[CATEGORICAL_COLS])

    # Train Baseline Model (LightGBM)
    print("Training baseline LightGBM model...")
    # Using parameters similar to best_solution.py but fixed for speed
    lgbm = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1
    )
    
    lgbm.fit(
        X_train_main_enc, 
        y_train_main, 
        categorical_feature=CATEGORICAL_COLS
    )

    # Evaluate Baseline on Test
    print("Evaluating baseline on test set...")
    y_pred_baseline = lgbm.predict(X_test_enc)
    acc_baseline = accuracy_score(y_test, y_pred_baseline)
    dp_diff_baseline = demographic_parity_difference(y_test, y_pred_baseline, sensitive_features=A_test)
    
    print(f"Baseline Accuracy: {acc_baseline:.4f}")
    print(f"Baseline DP Diff:  {dp_diff_baseline:.4f}")

    # Post-processing with ThresholdOptimizer
    print("\nRunning ThresholdOptimizer (Demographic Parity)...")
    optimizer = ThresholdOptimizer(
        estimator=lgbm,
        constraints="demographic_parity",
        predict_method='predict_proba',
        prefit=True
    )

    # Fit optimizer on calibration set
    # We need to pass the encoded features because the estimator expects them
    optimizer.fit(X_train_calib_enc, y_train_calib, sensitive_features=A_train_calib)

    # Predict on test set
    print("Predicting with fair model...")
    y_pred_fair = optimizer.predict(X_test_enc, sensitive_features=A_test)

    # Evaluate Fair Model
    acc_fair = accuracy_score(y_test, y_pred_fair)
    dp_diff_fair = demographic_parity_difference(y_test, y_pred_fair, sensitive_features=A_test)

    print(f"Fair Model Accuracy: {acc_fair:.4f}")
    print(f"Fair Model DP Diff:  {dp_diff_fair:.4f}")
    
    print("\nComparison:")
    print(f"Accuracy Change: {acc_fair - acc_baseline:.4f}")
    print(f"DP Diff Change:  {dp_diff_fair - dp_diff_baseline:.4f}")

if __name__ == "__main__":
    main()
