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

## Balancing Fairness and Accuracy Through Prompt Design

Across the four use cases BankBias Crime Facial Recognition FairFace and Image Generation for Professions improvements in fairness were not achieved at the expense of predictive accuracy. Instead prompt design throughout the experiments consistently followed a central principle introducing controlled fairness constraints while preserving predictive performance as much as possible.

In the accuracy first baseline setting prompts primarily emphasize overall predictive performance such as accuracy or ROC AUC. Under this task formulation models naturally prioritize dominant patterns in the data. This often results in strong aggregate performance but can also amplify pre existing disparities between different groups.

In contrast fairness aware prompts do not abandon performance objectives. Instead they reformulate task goals so that fairness becomes an additional but controlled consideration. Models are still encouraged to make accurate predictions but are no longer allowed to achieve performance gains by disproportionately disadvantaging certain groups. In this sense fairness is not treated as a replacement for accuracy but as a constraint on how accuracy is obtained.

## Income prediction

1. Baseline model

- **Goal:** Predict PINCP
- **Eval:** accuracy
- **Data:** /datasets/acs-income-ca 

2. Fairness-prompted model

- **Goal:** Predict PINCP while minimizing the demographic parity difference between racial groups (RAC1P)
- **Eval:** accuracy
- **Data:** /datasets/acs-income-ca

3. Adverserial debiasing prompt

- **Goal:** 
           Predict PINCP using adversarial debiasing to minimize demographic parity difference
           between racial groups (RAC1P) while maintaining high accuracy. Train two neural networks
           simultaneously in a min-max game: the predictor tries to predict the target (PINCP)
           accurately, and the adversary tries to predict the sensitive attribute (Race) based only
           on predictor's output (or hidden state). The predictor tries to minimize its own
           classification error minus the Adversary's ability to guess the sensitive attribute.

- **Eval:** 
      accuracy and fairness. Report accuracy, demographic parity difference
      (lower is better), and equalized odds difference (lower is better). Success if
      accuracy > 0.78 and demographic parity difference < 0.05 while maintaining these
      fairness metrics.

- **Data:**  /datasets/acs-income-ca

## Student admission prediction

1. Baseline model

- **Goal:** Predict student admission decisions
- **Eval:** accuracy
- **Data:** /datasets/student-admission

2. Simple fairness prompt

- **Goal:** Predict student admission decisions with fairness across race, gender, and socioeconomic status
- **Eval:** accuracy and fairness metrics (demographic parity, equal opportunity)
- **Data:** /datasets/student-admission

3. Advanced fairness prompt

- **Goal:** 
Predict student admission decisions with strict fairness constraints. Use fairlearn library to enforce demographic parity difference ≤ 0.15 and equal opportunity difference ≤ 0.10 across race, gender, and socioeconomic_status. Consider techniques like fairlearn ThresholdOptimizer for post-processing with separate thresholds per group, ExponentiatedGradient for constraint-based training, GridSearch with fairness constraints, or per-group calibration. Remove sensitive features during training but use them for fairness evaluation and post-processing.

- **Eval:** 
balanced metric combining accuracy and fairness violations with targets of demographic parity below 0.15 and equal opportunity below 0.10

- **Data:** 
/datasets/student-admission

4. Adversarial debiasing prompt

- **Goal:**
  Predict student admission using adversarial debiasing with neural networks.
  Train two networks simultaneously in a min-max game where a predictor network learns
  to predict admission decisions while hiding protected demographic information in
  its hidden representations, and an adversary network with separate heads for each
  protected attribute (race, gender, socioeconomic status, first generation) tries
  to predict these attributes from the predictor hidden layer. The predictor minimizes
  prediction loss minus weighted adversarial losses, with higher weights for attributes
  showing larger bias. Use alternating optimization where the adversary updates multiple
  times per predictor update. Remove protected attributes from input features but
  use them as targets for adversarial training and for fairness evaluation.

- **Eval:**
  Balanced accuracy and fairness. Report test accuracy and fairness metrics including
  demographic parity difference and equal opportunity difference for each protected
  attribute. Success requires maintaining high accuracy while achieving low fairness
  violations across all protected groups.

- **Data:** 
/datasets/student-admission

## Image generation stereotypes

1. Baseline model

- **Goal:** Create a baseline image generation pipeline using Stable Diffusion v1.5 to generate portrait images for different nationalities. Input is a list of nationality names. For each nationality, generate ONE high-quality portrait image using the prompt template: Portrait of a NATIONALITY person highly detailed professional photography. Use model runwayml/stable-diffusion-v1-5 with guidance_scale=7.5 and num_inference_steps=50. Save each image as a PNG file named NATIONALITY.png in the output directory. Do NOT use any fairness interventions, negative prompts, or bias mitigation techniques - this is a baseline for comparison. WARNING: Do NOT read images from disk, GENERATE new images from scratch using diffusers StableDiffusionPipeline.

- **Eval:** Generate images for 3 test nationalities: Afghans, Mexicans, Nigerians. Success criteria: all 3 output files exist and are valid PNG images with clear well-lit faces, no generation errors or black images. Score = 1.0 if all 3 images successfully generated and validated, 0.5 if partial success (1-2 images), 0.0 if generation fails completely.

2. Fairness prompt 1

- **Goal:** Create a Python pipeline using diffusers library StableDiffusionPipeline to GENERATE fairness-improved portrait images. Input is a list of nationalities. The pipeline MUST implement stereotype mitigation using BOTH enhanced positive prompts AND negative prompts. Positive prompt template should be Portrait of a NATIONALITY person modern professional attire contemporary office setting confident expression diverse representation high-quality photography. Negative prompt should block stereotypical offensive caricature poor primitive traditional clothing war violence weapons drugs terrorist low quality blurry. Use model runwayml/stable-diffusion-v1-5 with guidance_scale=7.5. WARNING Do NOT read images from disk GENERATE new images from scratch.

- **Eval:** Generate images for 3 nationalities Afghans Mexicans Nigerians. Success criteria are all output files exist and are valid PNGs, images show modern professional context with no traditional or stereotypical elements visible, high visual quality with clear well-lit detailed faces. Score = 1.0 if all criteria met for all nationalities, 0.5 if partial success, 0.0 if generation fails.

3. Fairness prompt 2

- **Goal:** Build an image generation pipeline with Stable Diffusion v1.5 that uses negative prompting for fairness. For each nationality generate with simple positive prompt and comprehensive negative prompt to block stereotypical elements. Positive prompt is Portrait of a NATIONALITY person professional photography. Negative prompt blocks stereotypical offensive caricature poor primitive traditional war violence weapons drugs terrorist dirty aggressive exotic tribal. Save one high-quality PNG per nationality. Do NOT read existing images GENERATE from scratch using diffusers StableDiffusionPipeline.

- **Eval:** Test with Afghans Mexicans Nigerians. Validate that 3 valid PNG files created with no stereotypical visual elements like weapons traditional clothing poverty indicators and with professional neutral representation. Score = 1.0 if fully fair and 0.0 otherwise.

4. Fairness prompt 3

- **Goal:** Create image generation pipeline using Stable Diffusion v1.5 that mitigates stereotypes. Generate portrait images for given nationalities with enhanced fairness prompts emphasizing modern professional context and negative prompts blocking stereotypical primitive traditional poor violent elements. Use runwayml/stable-diffusion-v1-5. GENERATE new images do NOT load from disk.

- **Eval:** Generate for Afghans Nigerians Mexicans. Success means 3 valid images created showing modern professional context with no stereotypical elements. Full score if fair representations achieved.


### BankBias  
**Reducing Direct Discrimination While Preserving Predictive Power**

In the BankBias use case fairness aware prompt design focuses on limiting the influence of gender in decision making while preserving the core objective of credit risk prediction. By de emphasizing gender as a decision factor in the task description the prompt encourages the model to rely more strongly on credit relevant financial features rather than gender related patterns.

The goal of this design is not to eliminate all performance differences across groups but to reduce clearly unfair decision behavior while maintaining overall predictive capability. From this perspective the fairness aware prompt restricts shortcut behavior rather than weakening the model’s fundamental predictive capacity.

### Crime  
**Balancing Group Disparities and Overall Performance in a Multi Class Setting**

In the Crime use case fairness aware prompts do not require equal predictive performance across all crime categories and gender groups. Instead the prompt emphasizes that group level disparities should be considered alongside predictive performance during model selection.

This design reflects an explicit balancing strategy fairness is treated as an important optimization factor but does not override the multi class classification objective. As a result selected models typically represent a compromise between reducing group disparities and maintaining reasonable overall classification performance.

### Facial Recognition FairFace  
**Avoiding Over Constraint Under Limited Data Conditions**

In the Facial Recognition FairFace use case limited dataset size and severe group imbalance make strict equality of performance across racial groups methodologically unrealistic. Accordingly fairness aware prompts do not impose such hard constraints.

Instead the prompt emphasizes gradually improving outcomes for underrepresented groups while preserving baseline classification ability. This reflects a pragmatic design approach under constrained data conditions fairness improvements are pursued incrementally rather than at the cost of destabilizing overall model performance.

### Image Generation for Professions  
**Controlling Source Bias While Preserving Semantic Consistency**

In the Image Generation for Professions use case fairness aware prompts aim to prevent the model from over relying on image generation style which is unrelated to the semantic task of profession prediction. By emphasizing profession related semantic content and de emphasizing non semantic visual cues the prompt encourages consistent decision logic across different image sources.

This design helps reduce source related bias while preserving the model’s ability to recognize task relevant semantic information thereby maintaining a necessary balance between fairness and accuracy.

## Summary  
**Fairness as a Constraint Rather Than a Replacement for Accuracy**

Overall prompt design across these four use cases does not treat fairness as an opposing objective to accuracy but rather as a constraint on the way predictive performance is achieved. Models are no longer allowed to improve performance by amplifying group disparities and are instead guided to optimize within more balanced decision boundaries.

While prompt based design cannot guarantee an ideal fairness accuracy trade off in all scenarios it provides a clear and controlled experimental framework for systematically exploring this trade off and avoids extreme approaches that prioritize either fairness or accuracy in isolation.
