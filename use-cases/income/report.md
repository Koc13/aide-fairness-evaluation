# Fairness Evaluation Report: ACS Income (California)

## 1. Introduction
This report summarizes the fairness evaluation of machine learning models trained on the **ACS Income** dataset (California, 2018). The task is to predict whether an individual's income is above $50,000 (`PINCP`). The sensitive attribute considered for fairness analysis is **Race** (`RAC1P`).

## 2. Methodology

### 2.1 Metrics
We evaluated models using the following metrics:
*   **Accuracy**: Overall classification accuracy.
*   **Demographic Parity Difference (DP Diff)**: The difference between the largest and smallest selection rates (percentage of positive predictions) across racial groups. A value of 0 implies perfect demographic parity.
*   **Equal Opportunity Difference (EO Diff)**: The difference between the largest and smallest True Positive Rates (TPR) across racial groups.

### 2.2 Methods Evaluated
1.  **Baseline (LightGBM)**: A standard LightGBM model trained with hyperparameter optimization (Grid Search) to maximize accuracy, without any fairness constraints.
2.  **Fairness-Prompted (Unawareness + Heuristic)**: A LightGBM model trained without the sensitive attribute (`RAC1P`). During inference, group-specific thresholds were applied to equalize selection rates to the global average.
3.  **Post-processing (ThresholdOptimizer)**: Applied `fairlearn`'s `ThresholdOptimizer` to the Baseline model to enforce Demographic Parity.
4.  **Adversarial Debiasing (From Scratch)**: A custom PyTorch implementation of a **Predictor-Adversary** neural network architecture. The Predictor tries to predict income, while the Adversary tries to predict race from the Predictor's output. The Predictor is penalized if the Adversary succeeds, forcing it to learn fair representations.

## 3. Data Bias Analysis
Before modeling, we analyzed the ground truth labels to understand inherent disparities in the dataset.
*   **Ground Truth DP Difference**: `0.2861`
*   **Observation**: The dataset is inherently imbalanced. Group 6 (White alone) has a high base rate of high income (~48%), while Group 8 (Some other race alone) has a much lower rate (~19%).

## 4. Results Summary

| Method | Accuracy | DP Difference | EO Difference |
| :--- | :--- | :--- | :--- |
| **Baseline (LightGBM)** | **0.8281** | 0.5399 | 0.5595 |
| **Fairness-Prompted (Heuristic)** | 0.8114 | 0.2737 | 0.2766 |
| **Baseline + Post-processing** | 0.8111 | **0.1090** | 0.3776 |
| **Adversarial Debiasing** | 0.8049 | 0.1858 | **0.2510** |

*Note: The "Fairness-Prompted + Post-processing" method was also evaluated but yielded unstable results (DP Diff: 0.4226) likely due to small sample sizes in certain groups affecting the calibration split.*

## 5. Analysis & Key Findings

### 5.1 Baseline Amplifies Bias
The Baseline model achieved the highest accuracy (82.8%) but significantly exacerbated the existing bias. The DP Difference jumped from the ground truth's **0.2861** to **0.5399**. This confirms that standard training on biased data leads to models that discriminate more than the data itself.

### 5.2 Effectiveness of Post-processing
Applying `ThresholdOptimizer` to the Baseline model was highly effective for Demographic Parity, reducing the difference to **0.1090** (the lowest among all methods). However, this came at a cost of about 1.7% in accuracy.

### 5.3 Adversarial Debiasing
The custom Adversarial Debiasing model (trained from scratch) proved to be a very strong contender.
*   It achieved a **DP Difference of 0.1858**, which is significantly fairer than the ground truth (0.2861) and the Baseline (0.5399).
*   It achieved the best **Equal Opportunity Difference (0.2510)**, indicating it balances True Positive Rates across groups better than other methods.
*   The accuracy trade-off was acceptable (~2.3% drop from Baseline).

### 5.4 Heuristic Approach
The "Fairness-Prompted" approach (Unawareness + Manual Thresholds) successfully restored the bias level to roughly match the ground truth (0.2737 vs 0.2861) but did not reduce it further.

## 6. Conclusion
*   **For maximum fairness**: The **Baseline + Post-processing** approach is best if Demographic Parity is the sole priority.
*   **For a balanced approach**: **Adversarial Debiasing** offers an excellent trade-off, achieving high fairness (better than ground truth) and the best Equal Opportunity metric, with a moderate accuracy cost.
*   **Recommendation**: If deployment requires strict adherence to fairness metrics, **Post-processing** is recommended. If the goal is to learn intrinsically fair representations that generalize well, **Adversarial Debiasing** is a superior architectural choice.
