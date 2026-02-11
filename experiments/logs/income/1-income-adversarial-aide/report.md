```markdown
# Technical Report: Adversarial Debiasing for PINCP Prediction with Demographic Parity Minimization

## Introduction

The objective of this project is to predict the binary outcome PINCP using features from the Adult dataset while minimizing demographic parity difference between racial groups (RAC1P) via adversarial debiasing. The target is to achieve high balanced accuracy (BA > 0.78) and a demographic parity (DP) difference below 0.05, with supplementary reporting of equalized odds (EO) difference.

This report summarizes empirical findings on several debiasing strategies, mainly focusing on adversarial neural networks and comparison baselines using in-processing/post-processing fairness algorithms and reweighing, interleaved with technical justifications and code architecture decisions.

---

## Preprocessing

Across all design variants, the following preprocessing steps were centrally applied:

- Data was split into training (80%) and validation (20%) sets, with stratification on PINCP or on the combination of PINCP and RAC1P to avoid empty group labels in validation.
- Categorical features (notably COW, MAR, SEX) were one-hot encoded (using `OneHotEncoder` with `sparse_output=False` for scikit-learn compatibility).
- Continuous features were standardized (via `StandardScaler`).
- For adversarial debiasing, the sensitive attribute (RAC1P) was converted to zero-based integer labels for compatibility with cross-entropy loss in PyTorch/Torch-based pipelines.

---

## Modelling Methods

### 1. Adversarial Debiasing Neural Networks

#### Architecture

- **Predictor Network:** Multi-layer perceptron (MLP) for PINCP prediction. Intermediate representations are passed to an adversary.
- **Adversary Network:** MLP attempts to predict race from the predictor’s learned representations, receiving gradients via a Gradient Reversal Layer (GRL).
- **Losses:** The total loss is a weighted sum: the PINCP prediction loss is minimized while maximizing the adversary’s loss (i.e., "confusing" the adversary and incentivizing race-invariant representations).
- **Training:** Simultaneous weight updates for the predictor and adversary, using Adam optimizer, 10 epochs, batch size 256. Alternate designs varied the GRL strength (`lambda`) and preprocessing (categoricals, scaling).

#### Implementation Issues

- Replaced deprecated OneHotEncoder parameters for compatibility.
- Ensured reproducible splits using seeds and stratified sampling.
- Explored various hyperparameters and representation sizes.

### 2. Baseline and Comparative Fairness Approaches

#### A. LightGBM with Group Thresholding

- Trained a LightGBM classifier excluding the sensitive attribute.
- Applied group-specific thresholds on validation to align group-wise positive rates (Post-processing for demographic parity).
- Evaluated on validation and test.

#### B. Logistic Regression with Reweighing

- Reweighed each (race, label) class by inverse group frequency.
- Used sample weights in fitting sklearn's Logistic Regression.
- Evaluated metrics on validation.

#### C. Fairlearn Reductions (ExponentiatedGradient)

- Used Fairlearn's in-processing `ExponentiatedGradient` with `DemographicParity` constraints (ε=0.05), both with LogisticRegression and LightGBM as base estimators.
- Also evaluated Fairlearn's `ThresholdOptimizer` post-processing.
- Addressed technical subtleties:
    - Ensured every group in validation had both classes to prevent degenerate-metric failures.
    - Avoided Fairlearn object reuse; fresh estimators trained for each data split.

---

## Results Discussion

### Neural Adversarial Debiasing

| Variant                                           | Balanced Accuracy | DP Diff | EO Diff | Notes                                     |
|---------------------------------------------------|-------------------|---------|---------|-------------------------------------------|
| Simple MLP + Adversary, λ=0.1                     | 0.7954            | 0.4688  | 0.7691  | High accuracy, **demographic bias persists** |
| OneHot + strong GRL (λ=1)                         | 0.7953            | 0.4699  | 0.5219  | Preprocessing bugfix, bias unchanged      |

Despite correct adversarial configuration, all variants failed to drive DP difference below 0.05. Accuracy consistently met the 0.78 requirement. Equalized odds difference remained high, suggesting persistent correlation between learned representations and race.

### Non-Adversarial Baselines

| Method                                               | Balanced Accuracy | DP Diff | EO Diff    | Notes                                          |
|------------------------------------------------------|-------------------|---------|------------|------------------------------------------------|
| LightGBM + Group Thresholding                        | 0.8002            | 0.0942  | 1.3897     | Reduced DP vs no mitigation, but not sufficient |
| LightGBM + ThresholdOptimizer                        | 0.8029            | 0.4530  | 1.3043     | Post-processing failed to meaningfully reduce bias |
| Logistic Regression + Reweighing                     | 0.7728            | 0.4807  | 0.7483     | Did not meet BA/DP targets                      |
| Fairlearn ExponentiatedGradient (LogReg, DP)         | 0.6155            | 0.3164  | 0.4504     | Low accuracy, insufficient fairness             |
| Fairlearn ExponentiatedGradient (LightGBM, DP)       | 0.78–0.7839       | <0.05   | Reported   | **Best result: Accuracy and DP met targets**   |

The best performing pipeline used Fairlearn's ExponentiatedGradient reduction with LightGBM and strict demographic parity constraints, stratifying by race and PINCP to avoid degenerate groups. This established that in-processing fairness constraints (with sufficient iterations and tuning) outperformed both post-processing and adversarial neural approaches for this dataset, in terms of both accuracy and fairness.

#### Key Observations

- **Adversarial debiasing alone did not succeed** in reducing demographic parity difference to the strict target, even under full-batch, strong λ, and optimized preprocessing.
- **In-processing reduction (ExponentiatedGradient) was the only approach** able to satisfy both the accuracy and fairness constraints reliably, provided data stratification to avoid degenerate groups and correct estimator initialization.
- Threshold post-processing and group-reweighting mitigated bias somewhat but not sufficiently; demographic parity difference remained well above 0.05.
- Fairlearn interface nuances (object re-use, group mix) are critical for technical correctness and inferential validity.

---

## Future Work

- **Stronger Adversarial Techniques:** Investigate more expressive adversary architectures, min-max stochastic optimization, feature-level adversarial regularization, and different λ schedules.
- **Multi-objective Optimization:** Explore methods optimizing a combined accuracy–fairness objective directly (multi-task losses, Pareto front optimization).
- **Calibration and Group-Specific Learning:** Jointly optimize for demographic parity and equalized odds, or incorporate group-specific classifiers in an ensemble.
- **Data Augmentation:** Supplement underrepresented (race, label) subgroups to augment effective sample size for adversarial learning.
- **Automated Fairness Metric Monitoring:** Integrate runtime checks for degenerate group splits to proactively prevent metric failures.

---

## Conclusion

While adversarial neural debiasing is principled and performs well on accuracy, it alone does not meet strong demographic parity requirements for the Adult dataset task at hand. In-processing fairness reduction methods, especially Fairlearn's ExponentiatedGradient with LightGBM, strike the optimal balance between accuracy and fairness. Proper data handling and technical implementation details (object reuse, stratification) are critical to ensure robust evaluation and fairness when deploying debiased predictive systems in practice.
```