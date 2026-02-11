```markdown
# Technical Report: Adversarial Debiasing for Student Admission Prediction

## Introduction

The objective of this project was to predict student admission decisions using adversarial debiasing with neural networks. The core approach involved training a predictor network to classify admission outcomes while simultaneously hiding information about protected demographic attributes (race, gender, socioeconomic status, first generation) in its hidden representations. An adversary network with a separate head for each protected attribute attempted to infer these attributes from the predictor's hidden layer. The predictor was optimized to minimize prediction loss minus a weighted adversarial loss, penalizing leakage of demographic information. The training protocol followed alternating optimization, often with multiple adversary updates per predictor update. The primary metric was balanced accuracy, complemented by fairness metrics: demographic parity difference (DPD) and equal opportunity difference (EOD) for each protected attribute.

## Preprocessing

### Data Preparation

- **Protected Attribute Encoding:** Protected attributes were label-encoded for use as adversary targets.
- **Feature Selection:** Protected attributes and admission label were excluded from predictor input features. After one-hot encoding (for non-numeric features), inputs were ensured to be numeric (float32).
- **Scaling:** Numerical features were standardized using `StandardScaler` for neural networks.
- **Splitting:** Training data was split into training and validation sets, stratified by the admission outcome. For LightGBM, train/val and sample weight balancing by protected attribute marginals were also performed.

### Handling Data Types and Consistency

- Tensor conversion and DataLoader creation were robustified by explicit data type casting.
- Test sets lacking protected attribute labels were handled gracefully, only using feature columns during prediction.

## Modeling Methods

### Adversarial Debiasing with Neural Networks

- **Predictor Network:** Configurations evolved from a single hidden layer (64 units, ReLU) to architectures with two hidden layers (128 and 64 units), with dropout (0.5) and optionally batch normalization for stability and better generalization.
- **Adversary Network:** Multi-head adversaries, each head predicting one protected attribute using cross-entropy loss.
- **Alternating Optimization:** Most designs alternated one predictor update with multiple adversary updates (ranging 1–5 adversary steps), updating gradients accordingly.
- **Loss Function:**
  - Main objective: predictor minimizes binary cross-entropy for admission, minus the weighted sum of adversarial cross-entropy losses.
  - Class imbalance addressed via positive class weighting in BCE.
  - Adversarial loss weight (`lambda_adv`) was experimentally set (often 0.1) or scheduled to ramp up mid-training.
  - In some variants, a **gradient reversal layer (GRL)** was used for joint optimization, negating gradients with respect to protected information in the predictor's hidden state.
- **Regularization:** Dropout and L2 regularization (weight decay) were implemented to mitigate overfitting.
- **Early Stopping:** Validation balanced accuracy was monitored with early stopping based on non-improving epochs.

### Baseline: LightGBM

As a non-neural baseline, LightGBM was trained on one-hot encoded features (excluding protected attributes), using sample weights to equalize protected group marginals.

### Fairness Evaluation and Metrics

- **Demographic Parity Difference (DPD):** Difference between highest and lowest predicted positive rates across groups.
- **Equal Opportunity Difference (EOD):** Difference in true positive rates (i.e., recall on positive examples) across groups.

## Results Discussion

### Predictive Performance

- **Best Balanced Accuracy:** The top-performing neural models (two hidden layers, dropout, L2 regularization, class weighting) consistently achieved validation balanced accuracy between **0.854** and **0.855**.
  - Batch normalization and architectural refinements (deeper predictors) provided marginal improvements.
  - LightGBM with sample weights reached a balanced accuracy of **0.826**, indicating neural methods' advantage.
  - Simpler baselines and naïve adversarial models without robust tuning frequently performed at or near chance (**0.5** balanced accuracy), especially when adversarial strength was excessive.

### Fairness Analysis

- In early and most later neural network models, **fairness metrics (DPD and EOD)** were usually not computed or reported, meaning bias reduction could only be inferred indirectly (from adversarial training success).
- When fairness metrics were finally evaluated:
  - **DPD and EOD were near zero in failed models with random accuracy** (likely due to model collapse).
  - In a more stable debiasing run, DPD for gender and race was low, but substantial DPD (0.15) remained for socioeconomic status, suggesting **not all group disparities were fully mitigated** by standard adversarial debiasing.
- **Conclusion:** Adversarial debiasing can attenuate fairness violations but requires careful balancing of adversarial loss and main task accuracy. Excessive adversarial strength can degrade performance to random, with perfect fairness but no utility.

### Other Observations

- **Multiple Adversary Steps** per predictor update did not universally improve performance, sometimes leading to training instability unless the adversarial loss weight was reduced accordingly.
- **Class Imbalance and Regularization** (dropout, L2) were crucial for both stable training and high validation accuracy.
- Models with **early stopping** and scheduled adversarial weights showed improved convergence and generalization.

## Future Work

1. **Comprehensive Fairness Reporting:** Integrate detailed reporting and tracking of DPD and EOD throughout training to ensure continual progress toward fairness, not only accuracy.
2. **Adaptive Adversarial Weighting:** Implement dynamic adversarial coefficient adjustment, increasing adversarial importance for attributes with higher observed bias.
3. **Model Calibration:** Explore post-processing techniques (thresholding, reweighting) to further reduce fairness gaps, especially where residual bias remains for attributes such as socioeconomic status.
4. **Alternative Architectures:** Investigate more expressive adversaries or alternative feature hiding mechanisms, such as mutual information minimization.
5. **Longer Training with Early Stopping:** Combine extended training schedules with robust early stopping/patience to balance convergence and overfitting risks.
6. **Robustness Checks:** Evaluate model fairness and performance across multiple random seeds and on group-specific metrics (subgroup accuracy, TPR/FPR parity).
7. **Broader Application:** Assess transferability of adversarial debiasing pipeline to other outcome variables or educational datasets with similar fairness goals.

---

**Summary Table of Top Experiments:**

| Design Variant                              | Balanced Accuracy | DPD (Best Reported) | EOD (Best Reported) | Notes                                 |
|---------------------------------------------|-------------------|---------------------|---------------------|---------------------------------------|
| Deep Predictor + Dropout + L2 + Class Weights | 0.854–0.855       | Low (where measured)| Low (where measured)| Best overall; robust to overfitting   |
| Baseline LightGBM (Marginal Weights)        | 0.826             | Not reported        | Not reported        | Simple, competitive baseline          |
| Early Stopping                             | 0.852             | Not reported        | Not reported        | Good gen.; fastest convergence        |
| Strong Adversarial (no tuning)              | 0.50              | 0                   | 0                   | No utility; fairness artifact         |
| Improved Tensor Conversion + Debiasing      | 0.507             | Low–Med (SES high)  | Low                 | Fairness possible, accuracy low       |

---

*Success in adversarial debiasing for fairness requires careful tuning, ongoing fairness monitoring, and balanced optimization to maintain predictive accuracy while effectively mitigating group disparities.*
```
