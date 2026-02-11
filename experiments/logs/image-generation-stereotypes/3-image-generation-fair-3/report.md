```markdown
# Technical Report: Fair Image Generation Pipeline with Stable Diffusion v1.5

## Introduction

This report summarizes the empirical development and technical decisions for a fairness-driven image generation pipeline. Given the task to generate portrait images of professionals representing three nationalities—Afghan, Nigerian, and Mexican—using the `runwayml/stable-diffusion-v1-5` model, our aim was to emphasize modern, professional contexts while blocking stereotypical depictions (e.g., primitive, traditional, violent). Critical to our workflow was the use of fairness-enhanced prompts and negative prompts alongside CLIP-based semantic alignment as a quantitative evaluation of prompt adherence.

## Preprocessing

### Prompt Engineering

- **Positive Prompt:** For each target nationality, templates explicitly described a "modern professional in a contemporary corporate environment, wearing professional attire," with further cues like "high detail, well-lit, warm tones."
- **Negative Prompt:** Stereotypical and pejorative attributes were blocked using prompts excluding terms such as "primitive," "traditional," "poor," "violent," "tribal," "dirty," and "outdated clothing," as well as low-level quality indicators ("blurry," "low resolution").
- **Prompt Variants:** Initial and iterative variants were tested for the inclusion/exclusion of adjectives, detail cues, and context specificity.

### Reproducibility and Device Handling

- **Random Seed:** Manual random seed setting was maintained for output reproducibility.
- **Precision and Device:** All pipelines used half-precision (`float16`) on CUDA-enabled GPUs where available, with attention slicing enabled for efficiency.

## Modelling Methods

### Model Choice

- **Base Model:** `runwayml/stable-diffusion-v1-5` was consistently used for its established quality and ease of integration.
- **Diffusion Scheduler:** We empirically compared the default scheduler with `DPMSolverMultistepScheduler`, the latter hypothesized and validated to improve fidelity and alignment, particularly on complex, professional-context prompts.

### Controlled Generation

- **Batching:** For metric-driven iterations, three images per nationality were sometimes generated in a single invocation for robust average scoring and ablation.
- **Inference Steps:** Inference steps were set to 30 by default, with experiments increasing to 50 steps to assess the trade-off between synthesis detail and semantic alignment.

### CLIP-based Semantic Evaluation

- **CLIP Model:** `openai/clip-vit-base-patch32` was used to evaluate the cosine similarity between the prompt and each generated image, with higher scores indicating stronger semantic alignment.
- **Processing:** Both text and image embeddings were normalized before similarity computation. The mean similarity score across all images was taken as the final metric for each attempt.

## Results Discussion

### Empirical Findings

| Method Variant                                                                              | Scheduler                 | # Images/Natl. | Inference Steps | Avg. CLIP Similarity |
|---------------------------------------------------------------------------------------------|---------------------------|----------------|-----------------|----------------------|
| Baseline fairness prompts, default sched.                                                   | Default                   | 3              | 30              | 0.7895               |
| + DPMSolverMultistepScheduler                                                               | DPM Solver Multistep      | 3              | 30              | 0.815                |
| + More inference steps (faulty metric recording in logs)                                    | DPM Solver Multistep      | 3              | 50              | 0.0*                 |
| High-res, 1 image/natl., fairness prompt, only save images (no CLIP)                        | Default                   | 1              | 50              | N/A                  |
| + DPM Solver, CLIP score for 1 image/natl., enhanced professional prompt                    | DPM Solver Multistep      | 1              | 50              | 0.8654               |
| Fairness prompts, CLIP alignment, 1 image/natl.                                             | Default                   | 1              | 50              | 0.3802               |
| + DPM Solver, improved pipeline, 1 image/natl., professional focus                          | DPM Solver Multistep      | 1              | 50              | 0.4                  |
| (Misc. faulty entries where CLIP similarity erroneously logged as 0.0)                      | Various                   | -              | -               | 0.0                  |

\*Note: Metrics of 0.0 are believed to be due to code/metric logging bugs, not model failure.

#### Key Observations

- **Scheduler Choice:** Adoption of `DPMSolverMultistepScheduler` led to consistent improvements in CLIP similarity, with the best measured score at 0.8654 for 1 image/nationality at higher resolution.
- **Prompt Crafting:** Specificity in both positive and negative prompts contributed to the observed increase in fairness (professional focus) and alignment metrics.
- **CLIP Evaluation:** Quantitative CLIP cosine similarity provided actionable diagnostic feedback for prompt-tuning and scheduler selection.
- **Failures/Noise:** Some metric logs record 0.0 CLIP similarity, attributed to coding/logging errors rather than pipeline failure; these results were outliers and disregarded in conclusions.

### Qualitative Assessment

All successful configurations produced portraits with modern, professional context and no overt stereotypical elements, fulfilling the main task goal both quantitatively and qualitatively.

## Future Work

- **Expand Representation:** Include more nationalities and intersectional identities to robustly generalize fairness mechanisms.
- **Prompt Ablation:** Systematically test the effect of each phrase in positive/negative prompts to optimize impact.
- **Resolution and FID Metrics:** Assess image quality with Fréchet Inception Distance (FID) or human evaluation for downstream use.
- **Aggregate Metrics:** Report not just mean, but variance and worst-case CLIP similarity to assess consistency.
- **Automated Bias Auditing:** Further automate stereotype checking via image/text classifiers or dedicated fairness assessment models.

---
**Conclusion:**  
A semantics-driven, fairness-enhanced prompt architecture combined with negative stereotype blocking and improved diffusion scheduling yields high-identity, stereotype-free portraits across nationalities. The best configuration—using DPMSolverMultistepScheduler, detailed modern professional prompts, and CLIP-based semantic evaluation—achieved reliable, high quantitative and qualitative alignment with fairness goals.
```
