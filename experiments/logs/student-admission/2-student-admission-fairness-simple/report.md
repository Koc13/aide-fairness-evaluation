```markdown
# Technical Report: Fair Student Admission Prediction

## Introduction

This study explores empirical strategies for predicting student admission decisions with an explicit focus on fairness across race, gender, and socioeconomic status. Task evaluation emphasizes both accuracy and fairness, quantified through composite metrics that penalize disparate impact between protected groups. This report summarizes key technical decisions, their rationales, and empirical findings from a sequence of modeling, preprocessing, and fairness-aware interventions.

## Preprocessing

### Feature Handling
- **Protected Attributes**: Race, gender, and socioeconomic status were systematically excluded from model inputs when fairness was being audited or enforced.
- **Categorical Encodings**:
  - One-hot encoding was used for most categorical features, ensuring consistent feature space alignment between train and test sets.
  - Ordinal encoding was trialed in some baselines.
- **Numeric Features**: All numeric features were standardized using `StandardScaler` when included in pipelines.
- **Alignment**: One-hot encoded feature spaces for train and test data were strictly aligned to avoid column mismatches.

## Modeling Methods

### Baseline Predictors
- **LightGBM**: Used initially to jointly predict admission, with one-hot encoded data. Baseline fairness was assessed via the demographic parity gap.
- **Logistic Regression**: Deployed with L2 regularization and fairness measured with the equal opportunity gap, after excluding protected attributes.

### Fairness-Oriented Design Choices

#### Exclusion of Protected Attributes
- Early experiments omitted race, gender, and socioeconomic status from model features to assess 'fairness-through-unawareness'. Performance was moderate, but fairness gaps persisted.

#### Fairness Metrics
- **Demographic Parity Gap**: Maximum difference in positive prediction rates between groups.
- **Equal Opportunity Gap**: Difference in true positive rates (TPR) across groups.
- **Equalized Odds Gap**: Maximum of TPR and false positive rate (FPR) differences across groups.
- **Composite Metric**: Most experiments report accuracy minus average group gap.

#### Algorithm Selection
- **RandomForestClassifier**: Served as a reliable, well-performing estimator on tabular data, with or without fairness interventions.
- **LightGBM**: Adopted for faster training and improved predictive accuracy, consistently yielding high composite metrics.

#### Preprocessing Pipelines
- Feature preprocessing was encapsulated in `sklearn` pipelines and column transformers for reproducibility.

### Fairness-Enhancing Interventions

#### Sample Reweighting
- Inverse-probability weights were assigned so that the label became statistically independent of the joint protected group membership.
- Variants:
  - **Clipping**: Weights were clipped at the 95th percentile to dampen the influence of rare groups and reduce variance.
  - **Sqrt Damping**: Square roots of weights were used to further mitigate overfitting risks due to highly variable weights.

#### Threshold Tuning
- For each cross-validation fold, the classification threshold was optimized post-hoc on training data to maximize the composite metric; the selected threshold was then applied to the validation set.
- The same approach was applied to the full training set before generating final test predictions.

#### Calibration
- Probability calibration (using `CalibratedClassifierCV`) was incorporated post-training:
  - **Isotonic Calibration**: Improved reliability of probability outputs for threshold selection, especially in larger data.
  - **Sigmoid Calibration**: Platt scaling (logistic/sigmoid) was tested for greater robustness in smaller samples; it yielded competitive or superior results with reduced risk of overfitting.

#### LightGBM Hyperparameter Tuning
- Increasing the number of estimators and tree depth, reducing learning rate, and adding regularization parameters improved generalization without compromising fairness.

## Results Discussion

| Approach                                            | Composite Metric (Val.) |
|-----------------------------------------------------|------------------------|
| LightGBM, all features (baseline)                   | 0.4580                 |
| Logistic Regression, excl. protected                | 0.5180                 |
| RandomForest, excl. protected                       | 0.5362                 |
| RandomForest, sample reweighting                    | 0.5493                 |
| LGBM + sample reweighting                           | 0.5330                 |
| LGBM + threshold tuning                             | 0.5777                 |
| RandomForest + threshold tuning                     | 0.5727                 |
| LGBM, reweighted, isotonic calibration              | 0.5810                 |
| LGBM, reweighted, tuned hyperparameters, isotonic   | 0.5725                 |
| LGBM, reweighted, sqrt damping                      | 0.5705                 |
| LGBM, reweighted, sigmoid calibration               | 0.5761                 |

**Key Empirical Observations:**
- **Fairness interventions** such as reweighting and exclusion of protected attributes yielded substantial improvements over naive baselines.
- **Sample reweighting** alone improved the composite score (+~0.04), and **threshold tuning** further increased it (+~0.03 over plain reweighting).
- **Probability calibration** (especially isotonic) paired with threshold tuning achieved the best observed trade-off, marginally improving the composite metric.
- Using **LightGBM** with appropriate regularization and calibration consistently outperformed RandomForest in composite fairness-accuracy metrics.
- **Damping or clipping** extreme weights helped stabilize results, particularly in the presence of rare intersectional group samples; however, they did not outperform calibration and threshold tuning.
- The final best model combined: LightGBM, square-root-damped reweighting, sigmoid (Platt) calibration, and threshold tuning, which balanced fairness and accuracy (composite metric up to 0.581).

## Future Work

- **Intersectional Analysis**: Further breakdown of fairness metrics by intersectional subgroups to detect subtler disparities.
- **Alternative Fairness Constraints**: Test in-processing fairness algorithms (e.g., adversarial training, fairness regularization).
- **Causal Fairness**: Explore approaches modeling causal relations to better separate direct and indirect effects of protected attributes.
- **Robustness and Generalization**: Assess impact under covariate shift or with simulated adversarial subpopulations.
- **Automated Hyperparameter and Threshold Search**: Integrate Bayesian optimization for more principled threshold and hyperparameter selection.
- **Model Explanation**: Use post-hoc explainers (e.g., SHAP, LIME) to ensure transparency and to inspect group-level variable importances relevant to fairness concerns.

```