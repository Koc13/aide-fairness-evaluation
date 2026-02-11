# Income Prediction Use Case

## Overview
The income prediction use case focuses on predicting whether an individual's income exceeds $50,000 using the **ACS Income (California)** dataset from the US Census Bureau's American Community Survey. This use case explores various fairness techniques to minimize bias across racial groups (RAC1P) while maintaining high prediction accuracy.

## Dataset: ACS Income (California)
- **Source**: US Census Bureau's American Community Survey (2018)
- **Target**: Predict whether PINCP (Total person's income) > $50,000 (binary classification)
- **Protected Attribute**: RAC1P (Race - 9 categories: White, Black/African American, Asian, etc.)
- **Features**: 
  - Demographics: Age (AGEP), Sex, Marital Status (MAR)
  - Economic: Occupation (OCCP), Class of Worker (COW), Usual Hours Worked (WKHP)
  - Education: School Level (SCHL)
  - Other: Place of Birth (POBP), Relationship Status (RELP)

## Notebooks and Techniques

### 1. california_income.ipynb
**Main notebook demonstrating 4 different approaches to fairness-aware income prediction:**

#### 1.1 Baseline Model (No Fairness)
- **Algorithm**: LightGBM Classifier
- **Features**: All available features including RAC1P
- **Hyperparameter Tuning**: Grid search with 5-fold cross-validation
- **Goal**: Maximize accuracy without fairness constraints
- **Results**: Highest accuracy but significant demographic parity violation

#### 1.2 Fairness-Prompted Model (AIDE-Generated)
- **Algorithm**: LightGBM Classifier with fairness interventions
- **Technique**: Combines two strategies:
  1. **Fairness Through Unawareness**: Removes RAC1P from training features
  2. **Group-Specific Thresholding**: Uses different classification thresholds per racial group to enforce demographic parity
- **AIDE Prompt**: "Predict PINCP while minimizing the demographic parity difference between racial groups (RAC1P)"
- **Approach**: Optimizes group-specific thresholds to match overall positive rate
- **Results**: Reduced demographic parity difference with minimal accuracy loss

#### 1.3 Post-Processing: ThresholdOptimizer (Fairlearn)
- **Algorithm**: Post-processes baseline model predictions
- **Technique**: Fairlearn's ThresholdOptimizer
- **Constraint**: Demographic parity (equal selection rates across groups)
- **Method**: Finds optimal per-group thresholds to satisfy fairness constraints
- **Applied To**: Both baseline and fairness-prompted models
- **Results**: Significantly improved fairness metrics with controlled accuracy trade-off

#### 1.4 Visual Comparison
- **Metrics Tracked**:
  - Accuracy
  - Demographic Parity Difference (selection rate disparity across racial groups)
  - Equal Opportunity Difference (TPR disparity for positive class)
- **Visualizations**:
  - Bar charts comparing accuracy and fairness across all 4 methods
  - Selection rate distributions by racial group
  - Heatmap of fairness metrics

### 2. adversarial_debiasing.ipynb
**Implementation of adversarial debiasing using neural networks from scratch:**

#### Technique: Predictor-Adversary Architecture
- **Concept**: Min-max game between two neural networks
  1. **Predictor Network**: Predicts income (PINCP > $50k)
  2. **Adversary Network**: Tries to predict race (RAC1P) from predictor's hidden representations
- **Loss Function**: 
  ```
  L = L_predictor - λ * L_adversary
  ```
  where λ controls the fairness-accuracy trade-off
- **Goal**: Force predictor to learn representations that are informative for income but hide racial information
- **Implementation**:
  - PyTorch-based custom neural networks
  - Multi-class adversary (9 racial categories)
  - Gradient reversal during backpropagation
  - Batch training with Adam optimizer
- **Results**: Demonstrates the theoretical foundation of adversarial fairness

### 3. aide_adversarial.ipynb
**AIDE-ML's attempt at adversarial debiasing - An interesting case study:**

#### The Prompt
- **Goal**: "Predict PINCP using adversarial debiasing to minimize demographic parity difference between racial groups (RAC1P) while maintaining high accuracy. Train two neural networks simultaneously in a min-max game..."
- **Success Criteria**: Accuracy > 0.78 AND Demographic Parity < 0.05

#### What AIDE Actually Generated
**Surprising Result**: Despite being explicitly prompted for neural adversarial debiasing, AIDE's best solution used:
1. **LightGBM** (gradient boosting, not neural networks)
2. **Fairlearn's ThresholdOptimizer** (post-processing)
3. **Demographic parity constraint** with grid search

#### Why This Happened (From AIDE's Report)
- AIDE **did try** adversarial neural networks with:
  - Simple MLP architectures
  - Gradient Reversal Layer (GRL)
  - Various λ values (trade-off parameter)
- **Result**: Best neural approach achieved ~0.47 DP difference (far from 0.05 target)
- AIDE autonomously recognized failure and pivoted to alternative approaches
- **Best performer**: ExponentiatedGradient + ThresholdOptimizer (achieved DP < 0.05)


## Fairness Techniques Summary

| Technique | Type | Description | Notebook |
|-----------|------|-------------|----------|
| **Fairness Through Unawareness** | Pre-processing | Remove protected attribute (RAC1P) from features | california_income.ipynb |
| **Group-Specific Thresholding** | In-processing | Different classification thresholds per group | california_income.ipynb |
| **ThresholdOptimizer** | Post-processing | Optimize per-group thresholds using Fairlearn | california_income.ipynb |
| **Adversarial Debiasing** | In-processing | Neural min-max game to hide protected attribute | adversarial_debiasing.ipynb |
| **AIDE Adaptive Approach** | Hybrid | Automatic exploration of multiple techniques | aide_adversarial.ipynb |

## Key Findings

1. **No Free Lunch**: All fairness interventions involve an accuracy-fairness trade-off
2. **Post-processing is Effective**: ThresholdOptimizer achieved strong fairness improvements with controlled accuracy loss
3. **Fairness Through Unawareness is Limited**: Simply removing RAC1P doesn't eliminate bias due to correlated features
4. **Group Thresholding Works**: Adjusting decision thresholds per group is a practical fairness intervention
5. **Neural Adversarial Debiasing is Challenging**: Achieving strict fairness criteria (DP < 0.05) with adversarial networks proved difficult
6. **AIDE Demonstrates Intelligence**: When prompted approach fails, AIDE can autonomously find better solutions

## Metrics Used

- **Accuracy**: Overall classification accuracy
- **Demographic Parity (DP) Difference**: `max(selection_rate_group_i) - min(selection_rate_group_j)` across racial groups
- **Equal Opportunity (EO) Difference**: `max(TPR_group_i) - min(TPR_group_j)` across racial groups (TPR = True Positive Rate)

Lower DP and EO differences indicate better fairness (0 = perfect fairness).

## How to Run

1. **Data Download**: Run the data downloading cell in `california_income.ipynb` to fetch ACS Income data
2. **Baseline**: Execute baseline model training and evaluation
3. **Fairness Interventions**: Run fairness-prompted model and post-processing sections
4. **Adversarial Debiasing**: Open `adversarial_debiasing.ipynb` for neural network implementation
5. **AIDE Analysis**: Review `aide_adversarial.ipynb` to see AIDE's adaptive approach

