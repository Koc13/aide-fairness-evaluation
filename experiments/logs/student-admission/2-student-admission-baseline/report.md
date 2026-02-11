```markdown
# Technical Report: Predicting Student Admission Decisions

## Introduction

This report summarizes the empirical findings and technical decisions from a series of modeling experiments for the task of predicting student admission decisions using tabular data. The evaluation metric is **accuracy**. Various pipelines involving preprocessing, feature engineering, and machine learning models were explored, including linear models, tree ensembles, gradient boosting, and model ensembling, as well as hyperparameter optimization.

## Preprocessing

### Data Loading and Splitting
- Data was loaded from CSV files, with `admitted` used as the binary target.
- The data was consistently split into training and test sets.
- Stratified strategies for train/validation splits ensured class balance.

### Feature Engineering

#### Categorical Variables
- **One-Hot Encoding:** Used for tree-based models (RandomForest, LightGBM for most runs), ensuring all categorical variables are represented in a numeric form and aligning features across train and test sets.
- **Ordinal Encoding:** Used in one LightGBM experiment to prevent passing `object` dtypes; unknown categories in the test set mapped to a placeholder value.

#### Numerical Variables
- **Scaling:** StandardScaler applied to numerical features for linear models and some pipelines.
- **Polynomial/Interaction Features:** Generated using `PolynomialFeatures`, both full degree-2 and interaction-only expansions, to enable tree-models to pick up nonlinearities and feature interactions.

#### Feature Alignment
- After encoding, columns were aligned between train and test data to ensure consistency.

## Modeling Methods

### Baseline Models
- **Logistic Regression:** Fitted after one-hot encoding and scaling, serving as a linear baseline.
    - Validation Accuracy: **0.8828**

### Tree-Based Models
- **Random Forest Classifier:** Used with one-hot encoded features. Implemented both with scikit-learn pipelines and with direct one-hot encoding.
    - Validation/CV Accuracies: **0.8766** (holdout), **0.8771** (pipeline), **0.8850** (5-fold CV)
    - **Hyperparameter Tuning:** Randomized search for n_estimators, max_depth, min_samples_split, and min_samples_leaf improved mean CV accuracy up to **0.8864**.

- **LightGBM Classifier:**
    - Used with both one-hot and ordinal/categorical feature encoding.
    - With default settings and 5-fold CV: **0.8847**
    - With early stopping: **0.8844**
    - With randomized hyperparameter search (default and extended, including reg_alpha and reg_lambda): **0.8877** (best), **0.8868**
    - When using only ordinal encoding for categories: **0.8125** (indicating it may be suboptimal on this dataset)

### Feature Augmentation
- **Polynomial and Interaction Features:**
    - LightGBM trained with polynomial numeric features and one-hot encoded categorical features.
        - Degree-2, all polys: **0.8872**
        - Interaction-only: **0.8688**

### Ensemble Methods
- **Voting Ensemble:** Soft-voting classifier combining the best-tuned LightGBM and RandomForest, granting higher weight to LightGBM.
    - CV accuracy: **0.8872**, very similar to the tuned LightGBM alone

## Results Discussion

- **Best Model Performance:** The highest validation accuracy achieved was **0.8877** using a LightGBM model with thorough hyperparameter optimization, closely matched by ensembles and augmented-features runs.
- **Preprocessing Impact:** One-hot encoding of categoricals consistently outperformed ordinal encoding; interaction and polynomial value augmentation produced marginal gains, particularly with LightGBM.
- **Tree Models vs. Linear Models:** Both RandomForest and LightGBM outperformed logistic regression, albeit logistic regression’s performance (**0.8828**) was also strong, suggesting strong linear separability.
- **Hyperparameter Tuning:** Randomized search on both RandomForest and LightGBM reliably improved results over default values by a modest margin (about +0.002-0.003 in accuracy).
- **Ensembling:** Provided no clear advantage over a well-tuned LightGBM, implying limited complementarity between RandomForest and LightGBM for this dataset.
- **Early Stopping:** LGBM with early stopping yielded similar performance as full cross-validation, confirming model robustness.

## Future Work

- **Advanced Feature Engineering:** Explore domain-driven or learned embeddings for categorical values, or feature selection to reduce dimension.
- **Stacking/Blending Ensembles:** Investigate meta-ensembling or stacking diverse model types.
- **Automated Feature Generation:** Consider unsupervised feature extraction (e.g., via autoencoders) or advanced interaction mining.
- **Model Calibration:** Assess outputs for probabilistic calibration, useful if admission probabilities are required.
- **Explainability:** Incorporate SHAP or permutation feature importance for interpretability.
- **Handling Class Imbalance:** If admission ratios are skewed, try sampling strategies or weighted losses.
- **Alternative Algorithms:** Try other gradient boosting packages (e.g., CatBoost, if available), or neural network approaches for tabular data.

---

**Summary:**  
A careful combination of one-hot encoding, thoughtful numeric augmentation, and comprehensive hyperparameter tuning with LightGBM yields the best admission prediction performance, achieving up to **0.8877** accuracy. Further gains may depend on advanced feature engineering or model ensembling strategies.

```