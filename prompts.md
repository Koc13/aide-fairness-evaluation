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
