import os
import json
import sys
import pandas as pd
import numpy as np
import importlib.util
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from sklearn.metrics import accuracy_score, mean_squared_error

# Paths
REGISTRY_PATH = "fairness_configs/registry.json"
LOGS_DIR = "logs"

def load_registry():
    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)

def audit_experiment(exp_key, config):
    print(f"\n--- Auditing: {exp_key} ---")
    
    # 1. Load Data (Ground Truth)
    data_path = config["dataset_path"]
    test_csv = os.path.join(data_path, "test.csv")
    
    if not os.path.exists(test_csv):
        print(f"Error: Test file not found at {test_csv}")
        return

    df_test = pd.read_csv(test_csv)
    
    # 2. Find Predictions (submission.csv)
    # AIDE creates folders like "0-experiment_name" in workspaces/
    # We search for the folder that contains the experiment key
    experiment_name = exp_key
    workspaces_root = "workspaces"
    
    if not os.path.exists(workspaces_root):
         print(f"Error: workspaces directory not found.")
         return

    # Find folder containing experiment_name
    found_dirs = [d for d in os.listdir(workspaces_root) if experiment_name in d]
    
    if not found_dirs:
        print(f"Error: No workspace directory found for experiment: {experiment_name}")
        return
        
    # Pick the most recent one (highest index/lexicographically last)
    workspace_dir_name = sorted(found_dirs)[-1]
    submission_path = os.path.join(workspaces_root, workspace_dir_name, "working", "submission.csv")
    
    if not os.path.exists(submission_path):
        print("No submission.csv found. Cannot audit.")
        return

    preds_df = pd.read_csv(submission_path)
    
    # 3. Align and Validate
    if len(preds_df) != len(df_test):
        print(f"Warning: Length mismatch. Test: {len(df_test)}, Preds: {len(preds_df)}")
        return

    # Assume single column prediction or specific column name
    if len(preds_df.columns) == 1:
        y_pred = preds_df.iloc[:, 0].values
    elif config["target"] in preds_df.columns:
        y_pred = preds_df[config["target"]].values
    elif 'prediction' in preds_df.columns:
        y_pred = preds_df['prediction'].values
    else:
        y_pred = preds_df.iloc[:, 0].values # Fallback

    # 4. Calculate Metrics
    target_col = config["target"]
    protected_col = config["protected_attribute"]
    
    y_true = df_test[target_col].values
    sensitive_features = df_test[protected_col].values
    
    # Performance
    if config["performance_metric"] == "accuracy":
        # Ensure predictions are binary
        y_pred_binary = (y_pred > 0.5).astype(int)
        # Ensure y_true is binary (it might be boolean in CSV)
        y_true_binary = y_true.astype(int)
        
        perf_score = accuracy_score(y_true_binary, y_pred_binary)
        print(f"Accuracy: {perf_score:.4f}")
        
    # Fairness
    if config["fairness_metric"] == "demographic_parity_difference":
        fair_score = demographic_parity_difference(
            y_true_binary, 
            y_pred_binary, 
            sensitive_features=sensitive_features
        )
        print(f"Demographic Parity Difference: {fair_score:.4f}")
        
    # Always calculate Equalized Odds for reference
    eq_odds = equalized_odds_difference(
        y_true_binary,
        y_pred_binary,
        sensitive_features=sensitive_features
    )
    print(f"Equalized Odds Difference: {eq_odds:.4f}")

if __name__ == "__main__":
    registry = load_registry()
    
    # You can pass a specific key as arg, or run all
    if len(sys.argv) > 1:
        keys = [sys.argv[1]]
    else:
        keys = registry.keys()
        
    for key in keys:
        audit_experiment(key, registry[key])
