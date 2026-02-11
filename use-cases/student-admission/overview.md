# Student Admission Use Case

## Overview
The student admission use case focuses on predicting whether a student will be **admitted** or **rejected** using a **synthetic dataset of 12,000 student applications** with built-in systemic biases. This use case tests AIDE-ML's ability to create fair admission prediction models across multiple protected demographic groups while maintaining high accuracy.

## Dataset: Student Admission (Synthetic)
- **Size**: 12,000 applications (9,600 train / 2,400 test)
- **Target**: Predict admission decision (binary: admitted=1, rejected=0)
- **Protected Attributes**: 
  - **race**: White, Black, Hispanic, Asian, Native American, Other (6 categories)
  - **gender**: Male, Female, Non-binary (3 categories)
  - **socioeconomic_status**: Low, Middle, High (3 categories)
  - **first_generation**: Yes/No (first-generation college student)
  - **region**: Northeast, Southeast, Midwest, Southwest, West
  - **urban_rural**: Urban, Suburban, Rural
  - **disability_status**: Yes/No

- **Academic Features**: GPA, SAT score, ACT score, AP courses, honors courses, class rank percentile
- **Activities**: Extracurriculars, leadership positions, community service hours, awards/honors
- **Qualitative**: Essay score, recommendation strength

### Built-in Systemic Biases
The dataset intentionally replicates realistic systemic biases:

1. **Socioeconomic Inequality**: 
   - Higher SES → better test scores (test prep access)
   - Higher SES → more AP courses (school resource disparity)
   - Higher SES → stronger recommendations (network access)

2. **Racial Bias**: 
   - Race correlated with SES (structural inequality)
   - Test score gaps across racial groups
   - Admission decisions favor White and Asian applicants

3. **Geographic Disadvantage**: 
   - Rural students have fewer resources and opportunities
   - Urban/suburban students have more access to advanced courses

4. **First-Generation Penalty**: 
   - First-gen students disadvantaged despite diversity initiatives
   - Correlated with low SES (65% of low SES are first-gen)

5. **Gender Bias**: 
   - Subtle admission bias favoring male applicants

### Fairness Goals
- **Accuracy Target**: ≥ 80%
- **Demographic Parity Difference**: ≤ 0.15
- **Equal Opportunity Difference**: ≤ 0.10

## Notebooks and Techniques

### 1. student_admission.ipynb
**Comprehensive notebook comparing 3 AIDE-generated approaches:**

#### 1.1 Baseline Model (No Fairness)
- **Algorithm**: LightGBM Classifier (AIDE-generated)
- **Features**: All features including protected attributes
- **Approach**: Standard ML pipeline with cross-validation
- **AIDE Prompt**: "Predict student admission decisions"
- **Results**: 
  - Highest accuracy (~84-86%)
  - Significant fairness violations across race, gender, and SES
  - Example: DP difference for race often > 0.30

#### 1.2 Simple Fairness-Prompted Model
- **Algorithm**: LightGBM Classifier with fairness interventions
- **Technique**: 
  1. **Fairness Through Unawareness**: Removes race, gender, and SES from training features
  2. **Calibrated Classification**: Uses CalibratedClassifierCV to adjust probabilities
- **AIDE Prompt**: "Predict student admission decisions with fairness across race, gender, and socioeconomic status"
- **Approach**: Remove protected attributes, rely on LightGBM to learn unbiased patterns
- **Results**: 
  - Moderate accuracy improvement over baseline
  - **Limited fairness improvement** - bias persists through correlated features
  - Example: Removing race doesn't eliminate racial bias due to GPA/SAT correlations

#### Analysis: Why Simple Fairness Failed
The notebook includes detailed analysis showing:
- **Proxy features**: Test scores, GPA, and AP courses strongly correlate with SES and race
- **Fairness through unawareness is insufficient**: Protected attributes can be inferred from other features
- **Calibration alone doesn't enforce fairness**: Without explicit constraints, bias propagates

#### 1.3 Advanced Fairness-Constrained Model
- **Algorithm**: LightGBM + Fairlearn's ExponentiatedGradient
- **Technique**: In-processing fairness with demographic parity constraints
- **AIDE Prompt**: "Predict student admission with strong fairness constraints across all protected attributes"
- **Approach**: 
  1. Train base LightGBM estimator
  2. Wrap with ExponentiatedGradient reduction
  3. Enforce demographic parity constraint during training
  4. Optimize fairness-accuracy trade-off
- **Results**: 
  - Better fairness-accuracy balance
  - Significant reduction in DP difference
  - Controlled accuracy trade-off

#### 1.4 Fairness Evaluation Across All Protected Attributes
The notebook evaluates fairness across 4 dimensions:
- **Race** (6 groups): White, Black, Hispanic, Asian, Native American, Other
- **Gender** (3 groups): Male, Female, Non-binary
- **Socioeconomic Status** (3 groups): Low, Middle, High
- **First Generation** (2 groups): Yes, No

**Metrics tracked**:
- Accuracy
- Demographic Parity Difference (selection rate equality)
- Equal Opportunity Difference (TPR equality for qualified candidates)
- Selection rates per group
- True Positive Rates per group

### 2. fair_model.ipynb
**Custom implementation of adversarial debiasing neural networks:**

#### Technique: Multi-Attribute Predictor-Adversary Architecture
- **Innovation**: Unlike single-attribute adversarial debiasing, this handles **multiple protected attributes simultaneously**
- **Architecture**:
  1. **Predictor Network**: 
     - Input → 256 → 128 → 64 → Admission prediction
     - Learns hidden representations for admission worthiness
  2. **Multi-Head Adversary Network**:
     - Predictor's hidden layer (64) → Separate adversary heads
     - **Race head**: 64 → 96 → 48 → 6 classes
     - **Gender head**: 64 → 96 → 48 → 3 classes
     - **SES head**: 64 → 96 → 48 → 3 classes
     - **First-gen head**: 64 → 96 → 48 → 2 classes

- **Loss Function**:
  ```
  L = L_admission - (λ_race * L_race + λ_gender * L_gender + λ_SES * L_SES + λ_firstgen * L_firstgen)
  ```
  where each λ controls fairness pressure for that attribute

- **Attribute-Specific Lambda Values**:
  - **λ_race = 12.0**: Highest (race bias most severe in dataset)
  - **λ_SES = 10.0**: High (strong SES bias)
  - **λ_gender = 5.0**: Moderate
  - **λ_firstgen = 3.0**: Lower (binary attribute, easier to handle)

- **Training Strategy**:
  - Alternating optimization (2 adversary updates per predictor update)
  - 250 epochs for convergence
  - Batch size: 1024
  - Monitors both accuracy and fairness throughout training

- **Results**:
  - Accuracy: ~81% (controlled trade-off)
  - Significantly reduced demographic parity across all 4 attributes
  - Learned representations that hide demographic information

### 3. aide_adversarial.ipynb
**AIDE's adversarial debiasing attempt - Another case study:**

#### The Detailed Prompt
- **Goal**: "Predict student admission using adversarial debiasing with neural networks. Train two networks simultaneously in a min-max game where a predictor network learns to predict admission decisions while hiding protected demographic information in its hidden representations, and an adversary network with separate heads for each protected attribute (race, gender, socioeconomic status, first generation) tries to predict these attributes from the predictor hidden layer..."
- **Success Criteria**: "High accuracy while achieving low fairness violations across all protected groups"

#### What AIDE Generated
**Architecture**:
- **Predictor**: 128 → 64 → Admission (simpler than custom)
- **Adversary**: Single layer per head (64 → classes)
- **Lambda**: 0.1 (uniform across all attributes)
- **Training**: 10 epochs only
- **Adversary steps**: 1 per predictor update

#### Results & Analysis
- **Accuracy**: ~85.4% (high, similar to baseline)
- **Fairness**: SES bias persisted, limited improvement
- **Why**: 
  1. **Weak fairness pressure**: λ=0.1 vs custom's λ=12.0 for race (100x weaker!)
  2. **Insufficient training**: 10 epochs vs 250 (min-max games need longer convergence)
  3. **Simpler adversary**: Easier to fool, less pressure on predictor
  4. **Uniform lambda**: Doesn't account for varying bias severity across attributes

#### Comparison Table (in notebook)
| Component | AIDE | Custom | Impact |
|-----------|------|--------|---------|
| Encoder depth | 128→64 | 256→128→64 | AIDE faster but less capacity |
| Adversary | 1 layer | 2 layers (96→48) | AIDE easier to fool |
| Lambda | 0.1 uniform | 12.0 (race), 10.0 (SES) | AIDE 100x weaker fairness |
| Epochs | 10 | 250 | AIDE insufficient convergence |
| Test Accuracy | ~85% | ~81% | AIDE prioritized accuracy |
| Key Issue | SES bias persisted | Better fairness trade-off | - |

#### Key Insight
AIDE's implementation shows **accuracy prioritization** - even with explicit fairness requirements, the weak lambda and short training suggest AIDE may have:
1. Tested adversarial debiasing but found accuracy loss unacceptable
2. Settled on a configuration that maintained high accuracy
3. Not fully explored the fairness-accuracy Pareto frontier

This contrasts with the income adversarial case where AIDE completely abandoned neural networks for a better approach.

## Fairness Techniques Summary

| Technique | Type | Description | Notebook |
|-----------|------|-------------|----------|
| **Fairness Through Unawareness** | Pre-processing | Remove protected attributes from features | student_admission.ipynb (Simple) |
| **Calibrated Classification** | Post-processing | Adjust prediction probabilities | student_admission.ipynb (Simple) |
| **ExponentiatedGradient** | In-processing | Fairlearn reduction with DP constraints | student_admission.ipynb (Advanced) |
| **Multi-Attribute Adversarial** | In-processing | Neural min-max with separate adversary heads | fair_model.ipynb |
| **Attribute-Weighted Adversarial** | In-processing | Different λ values per attribute severity | fair_model.ipynb |

## Key Findings

1. **Fairness Through Unawareness is Insufficient**: Simply removing protected attributes doesn't eliminate bias when proxy features exist
2. **Proxy Features are Powerful**: Test scores, GPA, and AP courses strongly predict race and SES
3. **Multi-Attribute Fairness is Complex**: Different protected attributes require different fairness pressures (λ values)
4. **In-Processing Works Best**: ExponentiatedGradient and adversarial debiasing outperform simple pre-processing
5. **Convergence Takes Time**: Adversarial min-max games need 200+ epochs, not 10
6. **Trade-offs are Real**: Best fairness (custom adversarial ~81% accuracy) vs best accuracy (baseline ~86%)
7. **AIDE's Priorities**: When fairness and accuracy conflict, AIDE may prioritize accuracy even with explicit fairness prompts

## Metrics Used

- **Accuracy**: Overall classification accuracy
- **Demographic Parity (DP) Difference**: `max(selection_rate_i) - min(selection_rate_j)` across groups (target: ≤0.15)
- **Equal Opportunity (EO) Difference**: `max(TPR_i) - min(TPR_j)` across groups (target: ≤0.10)
- **Selection Rates**: Percentage of positive predictions per group
- **True Positive Rates (TPR)**: Accuracy for qualified candidates per group

## How to Run

1. **Generate Dataset**: Run data generation cell in [student_admission.ipynb](student_admission.ipynb) (creates synthetic 12k applications)
2. **Baseline**: Execute baseline model training and multi-attribute fairness evaluation
3. **Simple Fairness**: Run fairness-through-unawareness approach and analyze why it fails
4. **Advanced Fairness**: Execute ExponentiatedGradient approach for better fairness
5. **Custom Adversarial**: Open [fair_model.ipynb](fair_model.ipynb) for multi-head adversarial neural network implementation
6. **AIDE Analysis**: Review [aide_adversarial.ipynb](aide_adversarial.ipynb) to compare AIDE's approach to custom implementation