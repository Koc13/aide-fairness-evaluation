```markdown
# Technical Report: ESR Prediction Model Development

## Introduction

The objective of this project is to develop a predictive model for the binary ESR classification task, evaluated by accuracy. Multiple modeling strategies, leveraging both tree-based and linear methods, were iteratively implemented and empirically assessed using cross-validation and validation set performance. This report summarizes the preprocessing strategies, modeling approaches, validation outcomes, technical reasoning, and directions for future work.

## Preprocessing

### Feature Handling

- **Categorical Features:**  
  Features with low cardinality (≤10 unique values) were consistently treated as categorical. Depending on the model, preprocessing was handled in two distinct ways:
  - **Tree-Based Methods (LightGBM/Random Forest):**  
    Categorical features were either directly specified via model interfaces (LightGBM) or transformed via one-hot encoding (Random Forest).
  - **Linear Models (Logistic Regression):**  
    Applied one-hot encoding to categorical features using `OneHotEncoder`.

- **Numerical Features:**  
  - **Tree-Based Models:** Numeric columns were used as-is, with no explicit scaling.
  - **Linear Models:** Applied standard scaling to numeric features via `StandardScaler`.

- **Consistent Train/Test Feature Space:**  
  When encoding features (e.g., one-hot), train and test data were concatenated prior to encoding to ensure consistency across splits.

## Modeling Methods

A variety of modeling strategies were attempted. Below is a summary of each, including principal design decisions and validation outcomes.

### 1. LightGBM Baseline

- Used all features, passed low-cardinality features as categorical.
- Training via 5-fold stratified cross-validation.
- **Validation Accuracy:** ~0.8222

### 2. Random Forest

- One-hot encoded categorical features.
- 5-fold stratified cross-validation for evaluation.
- **Validation Accuracy:** ~0.7961

### 3. Logistic Regression Pipeline

- Preprocessed with scaling and one-hot encoding in a `ColumnTransformer`.
- Evaluated with 5-fold cross-validation.
- **Validation Accuracy:** ~0.7817

### 4. LightGBM with Parameter Tuning

- DART boosting, subsample and colsample set to 0.8, learning rate 0.05.
- Initial early stopping via hold-out; addressed edge cases where `best_iteration_` was invalid.
- **Hold-out Validation Accuracy:** ~0.8234

### 5. LightGBM with Cross-Validation and Early Stopping

Several variants explored the use of 5-fold stratified cross-validation with early stopping, recording both the best iteration and accuracy per fold:

- For each fold:
  - Early stopping set to 50 rounds.
  - Tracked best iteration for boosting rounds.
- Early stopping did _not_ trigger; models used the full 1000 boosting rounds.
- Aggregated best iteration and accuracy across folds, then retrained final model on full data.
- **Cross-Validation Accuracy:** ~0.8227 (across all CV-based LightGBM methods)

### 6. LightGBM Cross-Validation Ensemble

- In addition to averaging validation accuracy and best iterations, predicted probabilities for the test set were averaged across the 5 LightGBM models.
- Final predictions made by thresholding the mean probabilities.
- **Cross-Validation Accuracy:** ~0.8227

## Results Discussion

- **Performance:**  
  Tree-based methods (LightGBM) consistently outperformed linear and Random Forest baselines (~0.82 vs. ~0.78–0.80).
- **Impact of Categorical Feature Handling:**  
  LightGBM's native categorical support provided a simple and performant pipeline. One-hot encoding was less effective for complex models (Random Forest) and required greater care for feature alignment.
- **Ensembling/Cross-Validation:**  
  Using test-set ensembling within CV folds did **not** yield a higher mean validation accuracy versus retraining on the full dataset with mean best iteration, but it further stabilized predictions.
- **Early Stopping:**  
  Early stopping never triggered within 1000 boosting rounds; the validation accuracy plateaued, hinting that higher values for n_estimators could be explored or more regularization might be applicable.
- **Robustness:**  
  Stratified cross-validation provided a more reliable estimate of model generalization as opposed to a single hold-out split.

## Future Work

- **Feature Engineering:**  
  Explore additional derived features, interaction terms, or embedding encodings for high-dimensional categoricals, which may further improve accuracy.
- **Hyperparameter Tuning:**  
  Systematic tuning, possibly via Bayesian optimization, could improve LightGBM accuracy beyond the current plateau.
- **Ensemble Methods:**  
  Expand on current ensembling (e.g., stacking or blending with models such as CatBoost, if feasible).
- **Regularization and Early Stopping:**  
  Increase `n_estimators` and explore stronger regularization or early stopping parameters.
- **Interpretability:**  
  Apply SHAP or permutation feature importance to better understand key predictors of ESR.

---
**Summary Table:**

| Model                        | Preprocessing                 | Validation Accuracy |
|------------------------------|-------------------------------|--------------------|
| LightGBM (baseline)          | Native categoricals           | ~0.8222            |
| Random Forest                | One-hot encoding              | ~0.7961            |
| Logistic Regression Pipeline | Scaling + One-hot encoding    | ~0.7817            |
| LightGBM (hold-out)          | Native categoricals           | ~0.8234*           |
| LightGBM (CV, early stop)    | Native categoricals           | ~0.8227            |
| LightGBM (CV, ensemble)      | Native categoricals, ensemble | ~0.8227            |

*hold-out validation; all others are 5-fold stratified cross-validation averages.

---
```