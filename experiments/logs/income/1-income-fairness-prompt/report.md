# Technical Report: Predicting PINCP with Demographic Parity Constraints Across Racial Groups

## Introduction

The task is to predict the PINCP variable (a target class) for individual records, subject to the dual objectives of high predictive accuracy and low demographic parity difference across racial groups (as represented by the RAC1P attribute). This report summarizes empirical findings and technical decisions from multiple modeling iterations, focusing on trade-offs between accuracy and fairness.

---

## Preprocessing

### Data Preparation

- **Data Source**: Training and test data were consistently loaded from CSV files.
- **Feature Selection**: Candidate features typically included `AGEP`, `COW`, `MAR`, `OCCP`, `POBP`, `RELP`, `SCHL`, `SEX`, `WKHP`, with some variants also including `RAC1P`.
- **Handling Categorical Features**:
    - For logistic regression, categorical predictors were one-hot encoded.
    - In LightGBM pipelines, categorical features were either one-hot encoded or, preferably, designated directly using LightGBM’s `categorical_feature` support, with dtype coercion to `category`.

### Data Splitting and Scaling

- **Cross-validation**: Predominantly 5-fold stratified CV, ensuring label proportions were preserved.
- **Holdout Set**: Some experiments added a separate holdout validation set for 'final' unbiased evaluation.
- **Scaling**: Numerical features were standardized for linear models.

---

## Modeling Methods

### Baseline Logistic Regression

- **Encoding**: One-hot encoding for categorical predictors.
- **Reweighing**: Several variants computed sample weights to debias the joint race–label distribution.
    - Reweighing aimed to compensate for label/race imbalances by adjusting the loss contribution per sample.
- **Postprocessing**: Probability thresholds were calibrated per race group to match the observed base rate, enforcing demographic parity at prediction time.

### Gradient Boosted Trees (LightGBM)

- **Feature Encoding**: Two approaches:
    1. Manual one-hot encoding of categories.
    2. Native handling of categorical features, specifying `categorical_feature` in LightGBM.
- **Hyperparameter Optimization**: A consistent hyperparameter grid search evaluated combinations of `n_estimators`, `num_leaves`, `learning_rate`, and `max_depth` using CV accuracy.
- **Fairness Postprocessing**:
    - Per-race calibrated thresholds for demographic parity using quantiles of predicted probabilities, matching each group’s positive prediction rate to the overall rate in the training set.
    - Some attempts replaced manual thresholding with [Fairlearn](https://fairlearn.org/)’s `ThresholdOptimizer` under fairness constraints.

---

## Results Discussion

### Logistic Regression Findings

- **Raw Accuracy**: ~0.768 (CV, one-hot encoded, group-thresholded), ~0.785 (with reweighing).
- **Demographic Parity Difference**: Postprocessing with calibrated thresholds yielded differences around 0.44–0.57, indicating substantial residual disparity.
- **Reweighing**: Only marginally improved fairness; accuracy–parity trade-off persisted.

### LightGBM Approaches

- **Baseline (no hyperparameter tuning, manual thresholding)**: CV accuracy ~0.805; DP difference ~0.44–0.45.
- **Grid Search with Per-Race Thresholding**:
    - **Best results**: CV accuracy of up to **0.8068**, DP difference around **0.4572**, with native categorical feature support and optimal hyperparameters.
    - **Including `RAC1P` as a feature** increased complexity but did not improve fairness or accuracy significantly.
    - **Holdout Validation**: Showed accuracy up to ~0.808, but DP difference remained >0.46.
- **Fairlearn ThresholdOptimizer**:
    - Markedly reduced overall accuracy (down to ~0.41) with little improvement or even worsening of demographic parity compared to quantile thresholding.

### Technical Decisions

- **LightGBM with categorical feature support proved most effective** for both accuracy and transparency.
- **Manual quantile-based calibration** of per-group thresholds provided a practical fairness adjustment without excessive complexity.
- **Grid search for hyperparameters** robustly improved outcomes over default model settings.
- **Including `RAC1P` as a feature did not outperform using it *solely* for postprocessing**.
- **Fairlearn's ThresholdOptimizer implementation led to strong accuracy degradation**, suggesting further tuning or alternative fairness regularization would be necessary for practical use.

---

## Future Work

- **Alternative Fairness Constraints**: Explore loss-level fairness regularization (e.g., adversarial debiasing or fairness-aware tree splits) vs. postprocessing-only approaches.
- **Fine-Tuning Threshold Calibration**: Investigate methods (e.g., ROC curve smoothing or isotonic regression) that might yield better trade-offs for small or unbalanced groups.
- **Comprehensive Metric Tracking**: Report and optimize for additional fairness metrics (equalized odds, predictive parity) to evaluate other dimensions.
- **Group-Aware Hyperparameter Tuning**: Jointly select model hyperparameters and group thresholds considering both accuracy and fairness on a validation set.
- **Feature Engineering**: Investigate additional socio-economic variables or nonlinear feature transformations to boost accuracy without hurting fairness.
- **Robustness & Generalization**: Assess model performance/fairness stability over time or under distribution shift, possibly with simulated or real temporal splits.

---

## Summary Table: Main Results

| Method                           | CV/Val Accuracy | DP Difference | Notes                                          |
|-----------------------------------|----------------:|--------------:|------------------------------------------------|
| Logistic Regression, One-hot      |      ~0.768     |  ~0.44        | Group thresholded                              |
| Logistic Regression, Reweighed    |      ~0.785     |  ~0.51–0.57   | Moderate improvement in DP, slight acc gain     |
| LightGBM + Grid Search            |      ~0.8068    |  ~0.457       | Best Pareto: categorical feature support,      |
|                                   |                 |               | race thresholding, tuned                       |
| LightGBM + Fairlearn ThresholdOpt |      ~0.41      |  ~0.36        | Large drop in accuracy, modest parity change    |
| LightGBM + Hold-out validation    |      ~0.808     |  ~0.461       | Best for accuracy, parity still moderate        |

---

```
**Key Recommendation:**  
For the task of PINCP prediction with demographic parity constraints, use LightGBM with native categorical feature support, grid search hyperparameter tuning, and calibrated per-race probability thresholds. This approach maximizes accuracy while reducing (but not eliminating) demographic parity differences. Further fairness improvements likely require model-level fairness regularization and more sophisticated postprocessing strategies.