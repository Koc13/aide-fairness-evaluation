import os
import json
import sys
import pandas as pd
from fairlearn.metrics import demographic_parity_difference

# Paths
REGISTRY_PATH = "fairness_configs/registry.json"

def load_registry():
    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)

def check_data_bias(exp_key, config):
    print(f"\n--- Checking Data Bias: {exp_key} ---")
    
    # 1. Load Train Data
    data_path = config["dataset_path"]
    train_csv = os.path.join(data_path, "train.csv")
    
    if not os.path.exists(train_csv):
        print(f"Error: Train file not found at {train_csv}")
        return

    df = pd.read_csv(train_csv)
    
    target_col = config["target"]
    protected_col = config["protected_attribute"]
    
    if target_col not in df.columns or protected_col not in df.columns:
        print(f"Error: Columns not found. Target: {target_col}, Protected: {protected_col}")
        return

    y_true = df[target_col].values
    sensitive_features = df[protected_col].values
    
    # Ensure binary
    # In the CSV, PINCP is True/False (boolean) or 0/1.
    y_true_binary = y_true.astype(int)
    
    # Calculate Demographic Parity on Ground Truth
    # We pass y_true_binary as the "predictions" to see the bias in the labels themselves
    # demographic_parity_difference(y_true, y_pred, sensitive_features=...)
    # Here y_pred is y_true_binary because we want to measure the disparity in the actual data
    bias_score = demographic_parity_difference(
        y_true_binary, 
        y_true_binary, 
        sensitive_features=sensitive_features
    )
    
    print(f"Dataset: {train_csv}")
    print(f"Ground Truth Demographic Parity Difference: {bias_score:.4f}")
    
    # Also print base rates for context
    print("\nBase Rates (Selection Rate) per Group:")
    groups = sorted(list(set(sensitive_features)))
    for g in groups:
        mask = sensitive_features == g
        rate = y_true_binary[mask].mean()
        count = mask.sum()
        print(f"  Group {g}: {rate:.4f} ({count} samples)")

if __name__ == "__main__":
    registry = load_registry()
    
    if len(sys.argv) > 1:
        keys = [sys.argv[1]]
    else:
        # Filter to unique datasets to avoid duplicate work
        seen_paths = set()
        keys = []
        for k, v in registry.items():
            if v["dataset_path"] not in seen_paths:
                seen_paths.add(v["dataset_path"])
                keys.append(k)
        
    for key in keys:
        check_data_bias(key, registry[key])
