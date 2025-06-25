# Project Tasks

This document tracks the active work, backlog, and completed milestones for the SISR research paper.

## ✅ Completed Milestones

*   **Dataset Curation:** Combined DIV2K and Flickr2K datasets.
*   **Model Development & Experimentation:**
    *   Implemented `pure REAL-ESRGAN` baseline.
    *   Implemented `real_esrgan with unet` for ablation.
    *   Implemented `DINO Ranger` and `DINO Ranger with degradation` to test novel components.
    *   Finalized `proposed merge` model architecture.
*   **Model Training:** All models have been trained for 500 epochs on the combined dataset.
*   **Results Generation:** Inference has been run and results are available for analysis.

## ✏️ Active Tasks: Paper Writing

*   [ ] **Structure Paper in LaTeX:** Lay out the main sections of the paper in `main.tex`.
*   [ ] **Write Abstract:** Summarize the problem, proposed solution, and key results.
*   [ ] **Write Introduction:**
    *   [ ] Introduce SISR.
    *   [ ] Discuss the problem of real-world degradation.
    *   [ ] Briefly introduce the proposed solution and its contributions.
    *   [ ] Outline the structure of the paper.
*   [ ] **Write Related Work:**
    *   [ ] Survey of classic and deep learning-based SISR methods (e.g., SRCNN, EDSR, ESRGAN).
    *   [ ] Review of methods focusing on real-world degradation (e.g., Real-ESRGAN).
    *   [ ] Discussion of relevant architectures (Transformers in vision) and loss functions (perceptual losses, self-supervised features like DINO).
*   [ ] **Write Proposed Method Section:**
    *   [ ] Detail the `TransformerESRGAN` generator architecture.
    *   [ ] Detail the `MultiScaleDiscriminator` architecture.
    *   [ ] Fully describe the `NovelDegradationPipeline`.
    *   [ ] Explain the combined loss function, with special focus on the `DINOPerceptualLoss` and `FrequencyLoss`.
*   [ ] **Write Experiments & Results Section:**
    *   [ ] Describe the dataset (DIV2K + Flickr2K).
    *   [ ] Detail the training parameters (epochs, batch size, optimizer, etc.).
    *   [ ] Present quantitative results (PSNR, SSIM, etc.) comparing `proposed merge` against the baseline and ablation models. Use tables for clarity.
    *   [ ] Present qualitative results with side-by-side image comparisons.
    *   [ ] Discuss the results of the ablation studies to justify architectural choices.

## 🗒️ Backlog / Future Work

*   [ ] **Create Figures and Diagrams:**
    *   [ ] Architecture diagram for the `TransformerESRGAN` generator.
    *   [ ] Diagram illustrating the `NovelDegradationPipeline`.
    *   [ ] Plots for quantitative comparisons.
*   [ ] **Write Conclusion:** Summarize the findings and discuss potential future research directions.
*   [ ] **Add References:** Compile and format all citations.
*   [ ] **Review and Refine:** Proofread the entire paper for clarity, grammar, and style. 