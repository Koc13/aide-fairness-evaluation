```markdown
# Technical Report: Fair Admissions Prediction with Strict Fairness Constraints

## Introduction

The goal of this study is to predict student admission decisions while enforcing stringent fairness constraints: **demographic parity difference ≤ 0.15** and **equal opportunity (true positive rate) difference ≤ 0.10** across race, gender, and socioeconomic status. All models are evaluated using a balanced metric: **accuracy minus penalties for fairness violations**, with penalties applied if demographic parity or equal opportunity targets are breached. To address these requirements, we systematically explored and compared a variety of preprocessing, in-processing, and post-processing fairness techniques using the `fairlearn` library.

---

## Preprocessing

Across all design attempts, the core principles of preprocessing were:

- **Sensitive Features Handling**:  
  - Sensitive columns (`race`, `gender`, `socioeconomic_status`) are completely removed from the feature set during model training.  
  - Sensitive columns are retained separately for post-processing and fairness metric evaluation.

- **Feature Engineering**:  
  - Categorical non-sensitive features are one-hot encoded (`get_dummies` or `OneHotEncoder`), ensuring alignment between train and test columns.
  - Numerical features are standardized (`StandardScaler`) when used in pipelines.
  - For post-processing methods using composite groupings (e.g., `race_gender_ses`), some attempts mapped degenerate groups (those with only one class) to a valid reference group to avoid errors during threshold optimization.

- **Data Splitting**:  
  - Consistently, 20% of the training data was set aside as a validation split, stratified by the admission label.

---

## Modeling Methods

### 1. In-processing Fairness Methods

#### ExponentiatedGradient (with `fairlearn.reductions`)
- **Constraint: Demographic Parity**  
  - Difference bound set to 0.15.
  - Logistic Regression base model (`liblinear`, C varied via hyperparameter search).
  - Provided reasonable accuracy but imperfect fairness; often could not satisfy both parity and opportunity constraints simultaneously.

- **Constraint: Equalized Odds**  
  - Difference bound set to 0.10 to target strict true positive/false positive rate parity.
  - Grid search over logistic regression regularization (C) values performed for optimal trade-off.
  - Increased maximum iterations (from 50 to 100) tested for improved convergence.

#### Grid/Hyperparameter Search
- Searched over regularization strengths (`C` in Logistic Regression) in conjunction with fairness constraints to seek optimal balance between fairness and accuracy.

- **Base learners considered**: Primarily Logistic Regression, with one LightGBM attempt tested within a pipeline.

### 2. Post-processing Fairness Methods

#### ThresholdOptimizer (with `fairlearn.postprocessing`)
- **Constraints**: Both demographic parity and equalized odds constraints were explored.
- **Sensitive Feature Grouping**:  
  - Custom logic to avoid degenerate groups (groups with only one class label) for threshold fitting.
  - When necessary, composite groups (across all sensitive attributes) were mapped to a reference group with both classes.
  - In one experiment, limiting post-processing to `race` only (the largest groups) to avoid degenerate group errors.

- **Base learners**: Always used prefit logistic regression.

---

## Results Discussion

### Empirical Performance

| Method & Constraint                         | Validation Metric | Notable Fairness Metrics | Outcome Summary                        |
|---------------------------------------------|-------------------|-------------------------|----------------------------------------|
| ExponentiatedGradient (Demographic Parity)  | 0.6223            | DP: 0.16, EO: 0.21      | Good accuracy; fairness not fully met. |
| ExponentiatedGradient (Equalized Odds)      | 0.3447–0.5194     | DP, EO exceeded targets | Accuracy moderate, fairness unmet.     |
| ExponentiatedGradient (Hyperparam. Search)  | 0.5685            | DP, EO above targets    | No C fully solved constraints.         |
| ThresholdOptimizer (Equalized Odds, composite) | 0.4908         | Moderate violations     | Better trade-off, but not perfect.     |
| ThresholdOptimizer (composite group mapping)| 0.4862            | Constraints respected   | Effective group mapping, solid trade-off.|
| ThresholdOptimizer (race only)              | -0.4626           | DP: 0.77, EO: 0.81      | High accuracy, fairness failed.        |

#### Observations

- **ExponentiatedGradient** (with either Demographic Parity or Equalized Odds) provided reasonable accuracy, but could not jointly achieve both fairness constraints—especially the strict 0.10 threshold for EO.
- **Hyperparameter tuning** marginally improved the best combined metric but did not overcome the fundamental constraint tightness.
- **ThresholdOptimizer post-processing** permitted finer per-group adjustment, but required careful handling of degenerate groups to avoid fitting errors.
- **Composite group mapping** (mapping degenerate groups to a reference) allowed successful completion of threshold optimization and resulted in competitive balanced metrics, despite minor accuracy losses.
- **Post-processing with a single sensitive attribute** (race) failed to generalize fairness mitigation to other attributes (gender, SES).

### Technical Decisions

- Removing sensitive features during training is essential to mitigate predictive bias.
- When using post-processing, either restrict sensitive groups to those with both classes, or explicitly map degenerate combinations to a valid reference group.
- Hyperparameter search offers incremental improvements but is insufficient where fairness-accuracy trade-offs are steep due to strict constraint bounds.

---

## Future Work

- **Multi-constraint or joint-fairness optimization**: Explore techniques or custom constraints that jointly enforce both demographic parity and equal opportunity.
- **More flexible post-processing**: Investigate alternative or layered post-processing methods, such as soft group calibration or multi-attribute thresholding beyond race/gender/SES individually.
- **Alternative in-processing mitigators**: Test `fairlearn`’s `GridSearch` or other fairness-focused algorithms (e.g., adversarial debiasing, reweighting strategies).
- **Data augmentation/balancing**: Address data-group imbalances that lead to degenerate groups, possibly via resampling or synthetic data.
- **Model diversity**: Incorporate more expressive base models (advanced tree ensembles, neural networks) if robustly compatible with fairness constraints.


---

**Summary**:  
Despite extensive attempts with both in-processing and post-processing fairness algorithms, strict simultaneous control of demographic parity and equal opportunity differences remains elusive—mainly due to the tightness of prescribed thresholds and data group sparsity. The most robust solution involved careful group mapping for post-processing, modestly outperforming constraint-based in-processing methods. Future work should prioritize approaches capable of handling multiple fairness constraints and problematic group sizes.
```