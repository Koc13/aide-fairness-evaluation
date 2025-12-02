# Technical Report: Predicting PINCP with Machine Learning

## Introduction

The goal of this study is to develop and optimize machine learning models to accurately predict the PINCP variable in the provided dataset. PINCP prediction is evaluated using accuracy as the primary metric. Multiple modeling strategies were explored, with successive rounds of technical improvements aiming for higher validation accuracy.

---

## Preprocessing

### Categorical Variable Handling
- **Ordinal Encoding:** Most LightGBM runs used OrdinalEncoder on columns COW, MAR, RAC1P, RELP, SCHL, SEX, mapping each category to an integer.
- **Label Encoding:** The initial LightGBM trial utilized LabelEncoder, guaranteeing consistent category-to-integer mapping across train and test splits.
- **One-Hot Encoding:** Logistic regression, Random Forest, and HistGradientBoostingClassifier used OneHotEncoder for categorical features, enabling models to consider each category separately.
- **Mean Target Encoding:** One experiment applied smoothed target mean encoding, mapping each category to its mean PINCP in the training split using a smoothing factor.
- **Frequency Encoding:** Another experiment encoded categories by their prevalence (relative frequency) in the training data.

### Numeric Features
- Consistently, numeric features such as AGEP and WKHP were used directly or standard scaled, depending on the classifier.

### Data Splitting and Consistency
- Stratified splitting ensured representative class proportions during cross-validation and validation splits.
- Encoders were fitted strictly on training data within each fold to avoid data leakage.

---

## Modeling Methods

### Baseline Models
- **Logistic Regression:**
  - Pipeline with one-hot encoding and standard scaling.
  - Evaluated on an 80/20 stratified hold-out split.
- **Random Forest:**
  - Similar preprocessing as logistic regression with one-hot encoding and scaling.
  - Evaluated via hold-out stratified validation.

### Gradient Boosting Methods
- **LightGBM:**
  - Models used categorical variable encoding (ordinal/label/target/frequency).
  - 5-fold stratified CV used to estimate accuracy; early stopping (via `lgb.early_stopping`) tuned iteration count.
  - Hyperparameter grids included learning rate, number of leaves, and subsampling parameters.
  - Final training on full data with optimal settings and boosting rounds.
- **HistGradientBoostingClassifier:**
  - Used one-hot encoded categoricals and matched feature columns via reindexing.
  - 5-fold stratified cross-validation for validation metrics.

### Encoding Innovations
- **Smoothed Target Mean Encoding:** Applied on categorical columns to encode with a blend of global and in-category mean target, reducing overfitting.
- **Frequency Encoding:** Categories encoded by their relative frequency in the data for continuous signal.

### Hyperparameter Search
- **LightGBM Grid Search:**
  - Explored learning_rate `[0.05, 0.1]`, num_leaves `[31, 63, 127]`.
  - Subsampling: feature_fraction and bagging_fraction tested at [0.8, 1.0].
  - Grid search combined with early stopping within cross-validation to optimize parameters and boosting rounds.

---

## Results Discussion

### Validation Metrics (Accuracy)

| Model / Encoding                 | CV/Val Accuracy |
|----------------------------------|-----------------|
| Logistic Regression (OHE)        | 0.7872          |
| Random Forest (OHE)              | 0.8139          |
| LightGBM (Label/Ordinal)         | 0.8224–0.8270   |
| LightGBM (Mean Target Encoding)  | 0.8265          |
| LightGBM (Frequency Encoding)    | 0.8275          |
| HistGradientBoosting (OHE)       | 0.8242          |
| LightGBM (Grid+Subsampling)      | **0.8275–0.8279**|

#### Key Observations
- **Encoding Impact:** Basic ordinal/label encoding performs well with tree models; target and frequency encoding yielded competitive or marginally improved results (~0.826–0.827).
- **Classical ML vs. Boosting:** Logistic regression underperforms relative to tree ensembles, highlighting the value of non-linear modeling for PINCP.
- **Early Stopping & CV:** Applying early stopping in cross-validation greatly aids generalization, with best iteration numbers tracked and averaged for final retraining.
- **Hyperparameter Optimization:** Systematic tuning of boosting parameters and addition of feature/data subsampling led to the best performance (0.8279).
- **Bagging/feature_fraction:** Subsampling contributed small but measurable gains in ensemble generalization.

---

## Future Work

- **Feature Engineering:** Explore non-linear feature interactions, polynomial features, and external data enrichment.
- **Encoding Strategies:** Test advanced encodings, including target encoding with cross-fold/leave-one-out prevention, embedding-based categorical encoding, or combining multiple schemes.
- **Ensemble Methods:** Blend top-performing models (e.g., LightGBM, HistGBM, Random Forest) to potentially surpass individual model accuracy.
- **Automated Hyperparameter Tuning:** Employ Bayesian optimization or random search with more granular search spaces.
- **Calibration & Uncertainty Quantification:** Measure and improve probability calibration, exploring models capable of quantifying prediction uncertainty.
- **Interpretability:** Apply SHAP or similar model explainability tools to understand the features most predictive of PINCP.

---

**Summary:**  
Through methodical feature encoding, modeling, and hyperparameter optimization, the best LightGBM models achieved validation accuracy of approximately 0.8279 on PINCP prediction. Encoding schemes and boosting frameworks were the major drivers of success, with further accuracy gains possible through advanced feature engineering and ensembling.