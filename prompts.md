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

## Task-Specific Prompt Design for Additional Use Cases

While the Healthcare Admission, Hiring, and Housing Application Approval use cases follow a unified prompt structure, this formulation cannot be consistently applied across all tasks. Differences in data modality, prediction objectives, and the conceptual role of sensitive attributes necessitate task-specific prompt designs for the remaining use cases.

For the BankBias, Crime, Facial Recognition (FairFace), and Image Generation for Professions use cases, prompts were deliberately designed to **define the prediction task and evaluation criteria without directly enforcing fairness constraints**. Instead, fairness was handled through explicit methodological mechanisms at the data, loss, or model-selection level. This separation ensures that observed fairness effects can be attributed to well-defined interventions rather than to variations in prompt wording.

---

## BankBias

- **Goal**  
  Predict the probability of a positive credit decision using financial and demographic features. Gender is treated as the sensitive attribute but is not included in the predictive decision logic described by the prompt.

- **Evaluation Criteria**  
  Model performance is evaluated using accuracy and ROC AUC. Fairness is assessed externally through group-level metrics across gender groups.

- **Methodological Rationale**  
  The prompt is intentionally neutral and task-focused, serving only to specify the prediction objective. This design aligns with the use of sample-based reweighting at the data level, ensuring that fairness effects arise from changes in training distribution rather than from prompt-induced behavioral constraints. By excluding explicit fairness instructions from the prompt, the experimental setup isolates reweighting as the primary fairness intervention under limited feature expressiveness.

---

## Crime

- **Goal**  
  Classify reported incidents into multiple crime categories based on available case features. Gender is considered a sensitive attribute but is not referenced explicitly in the prediction task.

- **Evaluation Criteria**  
  Predictive performance is evaluated using multi-class classification metrics. Fairness is assessed across gender groups and crime categories during validation.

- **Methodological Rationale**  
  Given the multi-class nature of the task and heterogeneous group distributions across categories, fairness considerations are incorporated at the model-selection stage rather than at the prompt or feature level. Prompts are therefore kept neutral, allowing fairness-aware scoring during validation to capture category-specific disparities without constraining the underlying model architecture or introducing prompt-level bias.

---

## Facial Recognition (FairFace)

- **Goal**  
  Perform image-based facial classification where race-based group membership is inherently embedded in the visual input.

- **Evaluation Criteria**  
  Model performance is evaluated using standard classification metrics, with fairness assessed across race-based groups.

- **Methodological Rationale**  
  In this setting, sensitive attributes cannot be meaningfully removed or abstracted at the prompt level, as they are intrinsic to the data representation. Consequently, prompts focus exclusively on the classification task, while fairness is addressed through group-based reweighting and fairness-aware loss functions during training. This design ensures that fairness interventions operate at the optimization level rather than through artificial suppression of sensitive visual information.

---

## Image Generation for Professions

- **Goal**  
  Predict profession labels from generated images. Image generation style or source is treated as the sensitive attribute but is not explicitly mentioned in the prediction task.

- **Evaluation Criteria**  
  Predictive performance is evaluated using classification accuracy, with fairness assessed by comparing performance across image sources.

- **Methodological Rationale**  
  The sensitive attribute in this use case arises from the data generation process rather than from task-relevant semantic content. Accordingly, prompts are designed to standardize semantic information across images while avoiding explicit references to generation style. Fairness is addressed through source-based reweighting at the data level, enabling an analysis of whether distributional interventions can mitigate bias introduced by dataset construction.

---

## Summary: Role of Prompts in the Fairness Methodology

Across all use cases, prompts function as **controlled task specifications rather than fairness enforcement mechanisms**. By maintaining neutral and consistent prompt formulations, fairness effects can be attributed to explicit methodological choices—such as reweighting strategies, loss design, or fairness-aware model selection—rather than to implicit prompt conditioning.

This design choice is essential for preserving methodological transparency and for enabling interpretable comparisons of fairness–accuracy trade-offs across heterogeneous tasks and data modalities.
