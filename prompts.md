# Prompts

This file contains the prompts used in AIDE for all experiments done for each use case, including the goal and evaluation criteria, which are requested by AIDE to run the pipelines.

## Healthcare Admission

1. Accuracy-first model

- Goal: Predict accurately the probability of an individual being admitted at a healthcare facility using demographic and socioeconomic features. The dataset includes the sensitive attributes race and sex
- Evaluation criteria: Maximize performance measured by accuracy and roc auc

2. Fairness-aware model

- Goal: Predict the probability of an individual being admitted at a healthcare facility using demographic and socioeconomic features. The dataset includes the sensitive attributes race and sex. Focus on fairness on these attributes
- Evaluation criteria: Maximize performance measured by accuracy and roc auc while reducing unfairness in predictions regarding race and sex attributes

3. Accuracy and Fairness model (reweighting)

- Goal: Predict the probability of an individual being admitted at a healthcare facility using demographic and socioeconomic features. The dataset includes the sensitive attributes race and sex. Focus on fairness on these attributes, without excluding them from the feature set
- Evaluation criteria: Maximize performance measured by accuracy and roc auc while reducing unfairness in predictions regarding race and sex attributes

4. Adaptive reweighting model

- Goal: Predict the probability of an individual being admitted at a healthcare facility using demographic and socioeconomic features. The dataset includes the sensitive attributes race and sex. Focus on fairness on these attributes, without excluding them from the feature set. Use different group-based reweighting factors to explore the balance between predictive accuracy and fairness.
- Evaluation criteria: Compare the results for each weight based on accuracy, roc auc and fairness metrics to understand how increasing reweighting affects both performance and group-level fairness.

5. Fairness-aware scoring model

- Goal: Predict the probability of an individual being admitted at a healthcare facility using demographic and socioeconomic features. The dataset includes the sensitive attributes race and sex. Focus on fairness on these attributes, without excluding them from the feature set. Select the employment prediction pipeline that optimizes both predictive performance and fairness. Focus on the accuracy, penalising unfairness by reducing the fairness gap for race and sex attributes.
- Evaluation criteria: Instead of maximizing predictive performance alone, rank results using a joint objective that penalizes unfairness. Score the results considering roc auc and penalties for demographic parity gaps across race and sex.


## Hiring

1. Accuracy-first model

- Goal: Predict accurately the probability of an individual being hired using demographic and socioeconomic features. The dataset includes the sensitive attributes race and sex
- Evaluation criteria: Maximize performance measured by accuracy and roc auc

2. Fairness-aware model

- Goal: Predict the probability of an individual being hired using demographic and socioeconomic features. The dataset includes the sensitive attributes race and sex. Focus on fairness on these attributes
- Evaluation criteria: Maximize performance measured by accuracy and roc auc while reducing unfairness in predictions regarding race and sex attributes

3. Accuracy and Fairness model (reweighting)

- Goal: Predict the probability of an individual being hired using demographic and socioeconomic features. The dataset includes the sensitive attributes race and sex. Focus on fairness on these attributes, without excluding them from the feature set
- Evaluation criteria: Maximize performance measured by accuracy and roc auc while reducing unfairness in predictions regarding race and sex attributes

4. Adaptive reweighting model

- Goal: Predict the probability of an individual being hired using demographic and socioeconomic features. The dataset includes the sensitive attributes race and sex. Focus on fairness on these attributes, without excluding them from the feature set. Use different group-based reweighting factors to explore the balance between predictive accuracy and fairness.
- Evaluation criteria: Compare the results for each weight based on accuracy, roc auc and fairness metrics to understand how increasing reweighting affects both performance and group-level fairness.

5. Fairness-aware scoring model

- Goal: Predict the probability of an individual being hired using demographic and socioeconomic features. The dataset includes the sensitive attributes race and sex. Focus on fairness on these attributes, without excluding them from the feature set. Select the employment prediction pipeline that optimizes both predictive performance and fairness. Focus on the accuracy, penalising unfairness by reducing the fairness gap for race and sex attributes.
- Evaluation criteria: Instead of maximizing predictive performance alone, rank results using a joint objective that penalizes unfairness. Score the results considering roc auc and penalties for demographic parity gaps across race and sex.

6. Fairness percentage for each group in sensitive attributes

- Goal: Determine how fair the hiring prediction is for each group of the sensitive attributes race and sex
- Evaluation criteria: Compute fairness percentages for each race and sex group relative to a reference group, considering disparities in predicted hiring probabilities

**Note**: The prompts for the pilot experiments done for the Hiring use case are listed directly on the scripts (all files that are not `hiring.ipynb` in the `use-cases/hiring` folder). This includes the goal, evaluation criteria and AIDE's logic description to generate the pipelines.

## House Application Approval

1. Accuracy-first model

- Goal: Predict accurately the probability of an individual being approved for a house application using demographic and socioeconomic features. The dataset includes the sensitive attributes race and sex
- Evaluation criteria: Maximize performance measured by accuracy and roc auc

2. Fairness-aware model

- Goal: Predict the probability of an individual being approved for a house application using demographic and socioeconomic features. The dataset includes the sensitive attributes race and sex. Focus on fairness on these attributes
- Evaluation criteria: Maximize performance measured by accuracy and roc auc while reducing unfairness in predictions regarding race and sex attributes

3. Accuracy and Fairness model (reweighting)

- Goal: Predict the probability of an individual being approved for a house application using demographic and socioeconomic features. The dataset includes the sensitive attributes race and sex. Focus on fairness on these attributes, without excluding them from the feature set
- Evaluation criteria: Maximize performance measured by accuracy and roc auc while reducing unfairness in predictions regarding race and sex attributes

4. Adaptive reweighting model

- Goal: Predict the probability of an individual being approved for a house application using demographic and socioeconomic features. The dataset includes the sensitive attributes race and sex. Focus on fairness on these attributes, without excluding them from the feature set. Use different group-based reweighting factors to explore the balance between predictive accuracy and fairness.
- Evaluation criteria: Compare the results for each weight based on accuracy, roc auc and fairness metrics to understand how increasing reweighting affects both performance and group-level fairness.

5. Fairness-aware scoring model

- Goal: Predict the probability of an individual being approved for a house application using demographic and socioeconomic features. The dataset includes the sensitive attributes race and sex. Focus on fairness on these attributes, without excluding them from the feature set. Select the employment prediction pipeline that optimizes both predictive performance and fairness. Focus on the accuracy, penalising unfairness by reducing the fairness gap for race and sex attributes.
- Evaluation criteria: Instead of maximizing predictive performance alone, rank results using a joint objective that penalizes unfairness. Score the results considering roc auc and penalties for demographic parity gaps across race and sex.
 ask-Specific Prompt Design for Additional Use Cases

## Using Prompt Design to Balance Fairness and Accuracy

Across the four use cases—BankBias, Crime, Facial Recognition (FairFace), and Image Generation for Professions—fairness improvements were not pursued at the expense of predictive accuracy. Instead, prompt design was guided by a central principle: **introducing fairness constraints while preserving as much predictive performance as possible**.

In the accuracy-first baseline setting, prompts primarily emphasize overall predictive performance, such as accuracy or ROC AUC. Under this formulation, models naturally prioritize dominant patterns in the data, which often leads to strong overall performance but may amplify disparities between different groups.

In contrast, fairness-aware prompts do not abandon performance objectives. Rather, they reformulate task goals so that fairness becomes an **additional, controlled consideration**. Models are still encouraged to make accurate predictions, but they are no longer allowed to improve performance by disproportionately disadvantaging certain groups.

---

### BankBias: Reducing Direct Discrimination While Preserving Predictive Power

In the BankBias use case, fairness-aware prompt design focuses on limiting the influence of gender as a decision factor, without undermining the core objective of credit-risk prediction. By constraining the model’s reliance on gender-related patterns, the prompt encourages greater use of task-relevant financial features.

The goal is not to eliminate all performance differences across groups, but to reduce clearly unfair decision patterns while maintaining overall predictive capability. In this sense, the fairness-aware prompt restricts shortcut behavior rather than weakening the model’s fundamental predictive capacity.

---

### Crime: Balancing Group Disparities and Overall Performance in a Multi-Class Setting

For the Crime use case, fairness-aware prompt design does not require equal performance across all crime categories and gender groups. Instead, it emphasizes that group-level disparities should be considered alongside predictive performance during model selection.

This reflects a deliberate balancing strategy: fairness is treated as an important optimization factor, but not one that overrides the multi-class classification objective. As a result, selected models represent a compromise between minimizing group disparities and maintaining reasonable overall classification performance.

---

### Facial Recognition (FairFace): Avoiding Over-Constraint Under Limited Data Conditions

In the Facial Recognition use case, severe data imbalance and limited sample size make strict fairness constraints unrealistic. The fairness-aware prompt therefore does not demand equal performance across all racial groups.

Instead, it emphasizes improving outcomes for underrepresented groups while preserving baseline classification ability. This reflects a pragmatic approach in which fairness improvements are pursued incrementally, without destabilizing the model’s overall performance under constrained data conditions.

---

### Image Generation for Professions: Controlling Source Bias While Preserving Semantic Consistency

In the Image Generation for Professions use case, the fairness-aware prompt aims to prevent the model from over-relying on image generation style—a factor unrelated to the semantic task of profession prediction.

By emphasizing profession-related content and de-emphasizing non-semantic visual cues, the prompt encourages consistent decision logic across image sources. This supports fairness by reducing source-based bias while preserving the model’s ability to recognize task-relevant visual information.

---

## Summary: Fairness as a Constraint, Not a Replacement for Accuracy

Overall, prompt design in these four use cases treats fairness not as a substitute for accuracy, but as a **constraint on how accuracy is achieved**. Rather than allowing performance gains through amplified group disparities, fairness-aware prompts encourage models to operate within more balanced decision boundaries.

This approach does not guarantee an ideal fairness–accuracy trade-off in all settings. However, it provides a systematic way to explore this trade-off, avoiding extreme solutions that prioritize either fairness or accuracy in isolation.
