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

While the Healthcare Admission, Hiring, and Housing Application Approval use cases follow a unified prompt structure, not all tasks could be addressed using the same formulation. Differences in data modality, prediction objectives, and the role of sensitive attributes required more specialized prompt designs for the remaining use cases.

For the BankBias, Crime, Facial Recognition (FairFace), and Image Generation for Professions use cases, prompts were therefore adapted to reflect task-specific methodological constraints. In these settings, prompts primarily define the prediction objective and evaluation criteria, while fairness interventions are handled explicitly at the data, loss, or model-selection level rather than through direct prompt conditioning. This separation ensures methodological clarity and prevents fairness effects from being conflated with prompt-induced behavior.

BankBias

Goal: Predict the probability of a positive credit decision using financial and demographic features. Gender is treated as the sensitive attribute but is not included as a predictive feature in the decision logic. The prompt is designed to define the task neutrally, allowing fairness interventions to be applied at the data level.

Evaluation criteria: Evaluate predictive performance using accuracy and roc auc. Fairness is assessed separately through group-level metrics computed across gender groups.

Crime

Goal: Classify reported incidents into multiple crime categories based on available case features. Gender is treated as the sensitive attribute but is not explicitly referenced in the prediction task. The prompt formulation remains neutral with respect to fairness constraints.

Evaluation criteria: Evaluate predictive performance using multi-class classification metrics. Fairness is incorporated during model selection through fairness-aware scoring, allowing disparities across gender groups and crime categories to be evaluated externally.

Facial Recognition (FairFace)

Goal: Perform facial classification using image-based inputs, where race-based group membership is inherent to the visual data. The prompt focuses exclusively on the classification task without attempting to suppress or abstract sensitive visual attributes.

Evaluation criteria: Evaluate model performance using standard classification metrics. Fairness is assessed across race-based groups and addressed during training through group-based reweighting and fairness-aware loss functions rather than through prompt-level constraints.

Image Generation for Professions

Goal: Predict profession labels from generated images. Image generation style or source is treated as the sensitive attribute, but is not explicitly mentioned in the prediction task. The prompt is designed to standardize semantic task content across different image sources.

Evaluation criteria: Evaluate predictive performance using classification accuracy. Fairness is assessed by comparing performance across image sources, with mitigation handled through source-based reweighting during training rather than through prompt conditioning.

Summary of Prompt Design Strategy

Across all use cases, prompts were designed to serve as task-specification mechanisms rather than as direct fairness enforcement tools. By keeping prompts neutral and consistent with the prediction objectives, fairness effects can be attributed to explicit methodological choices—such as reweighting strategies, loss design, or fairness-aware model selection—rather than to variations in prompt wording. This design choice ensures interpretability and comparability of fairness–accuracy trade-offs across heterogeneous tasks. 
