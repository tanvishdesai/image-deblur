# PLANNING.md - Phase 1: Proof of Concept

## High-Level Vision

**Core Hypothesis:** Physics-informed neural networks can outperform traditional black-box approaches for image deblurring by embedding the mathematical structure of the deconvolution process directly into the network architecture.

**Phase 1 Goal:** Validate that unrolled Richardson-Lucy deconvolution can be trained end-to-end and outperform classical Richardson-Lucy on simple, uniform motion blur scenarios.

## Research Innovation

**Novel Contribution:** Instead of treating deblurring as image-to-image translation, we model it as a learnable optimization process where each network layer represents one iteration of a classical deconvolution algorithm.

**Key Insight:** By respecting the underlying physics of blur formation (B = S ⊗ K + N), we can build networks with strong inductive biases that should generalize better than purely data-driven approaches.

## Architecture Overview

### Network Structure
```
Input: Blurred Image (256x256)
├── Initial Estimates: S₀, K₀ (learnable initialization)
├── Physics Block 1: (S₁, K₁) = RLBlock₁(B, S₀, K₀)
├── Physics Block 2: (S₂, K₂) = RLBlock₂(B, S₁, K₁)
├── Physics Block 3: (S₃, K₃) = RLBlock₃(B, S₂, K₂)
├── Physics Block 4: (S₄, K₄) = RLBlock₄(B, S₃, K₃)
└── Physics Block 5: (S₅, K₅) = RLBlock₅(B, S₄, K₄)
Output: Sharp Image S₅, Blur Kernel K₅
```

### Richardson-Lucy Physics Block
Each block implements one iteration of Richardson-Lucy deconvolution:
```python
# Classical R-L update (what we're learning to do):
S_new = S_old * (K_flipped ⊗ (B / (S_old ⊗ K)))
K_new = K_old * (S_flipped ⊗ (B / (S_old ⊗ K)))

# Learnable parameters:
- Step sizes (α, β)
- Regularization weights (λ₁, λ₂)
- Smoothness constraints
- Initialization strategies
```

## Technical Constraints

### Mathematical Constraints
- **Convolution Operations:** All convolutions must be differentiable
- **Kernel Size:** Fixed 15×15 blur kernels for Phase 1
- **Boundary Conditions:** Handle edge effects in convolution
- **Numerical Stability:** Prevent division by zero in R-L updates

### Data Constraints
- **Resolution:** 256×256 patches only
- **Blur Types:** Uniform motion blur (horizontal, vertical, diagonal)
- **Noise Level:** Clean conditions (minimal noise)
- **Ground Truth:** Perfect sharp images and known blur kernels

## Data Pipeline

### Input Data Requirements
- **Base Images:** 500 high-quality sharp images (1024×1024 minimum)
- **Categories:** Portraits (100), Natural scenes (150), Architecture (100), Objects (100), Textures (50)
- **Format:** PNG or high-quality JPEG
- **Storage:** ~2GB for base images

### Synthetic Data Generation
```python
# Per base image:
blur_angles = [0°, 45°, 90°, 135°]  # 4 directions
blur_lengths = [5, 10, 15]          # 3 lengths
crops_per_image = 2                 # 2 random crops
# Total: 500 × 4 × 3 × 2 = 12,000 training samples
```

### Data Augmentation
- **Geometric:** Horizontal flips, small rotations (±5°)
- **Photometric:** Slight brightness/contrast variation (±10%)
- **Spatial:** Random 256×256 crops from 1024×1024 images
- **Final Dataset:** ~5,000 training pairs, 500 validation, 500 test

## Evaluation Metrics

### Primary Metrics
- **PSNR (Peak Signal-to-Noise Ratio):** >28 dB target
- **SSIM (Structural Similarity Index):** >0.85 target
- **Training Loss:** MSE between predicted and ground truth sharp images

### Secondary Metrics
- **Kernel Accuracy:** MSE between predicted and ground truth kernels

### Baseline Comparisons
- **Classical Richardson-Lucy:** 20 iterations, fixed parameters
- **Simple CNN:** Basic U-Net architecture
- **Bicubic Upsampling:** Simple interpolation baseline

## Success Criteria

### Minimum Viable Result
- **Performance:** Beat classical Richardson-Lucy by >2 dB PSNR
- **Stability:** Training converges within 100 epochs
- **Interpretability:** Can visualize learned kernel evolution through layers

### Stretch Goals
- **Performance:** Achieve >30 dB PSNR on test set
- **Generalization:** Works on unseen blur angles (e.g., 22.5°, 67.5°)
- **Efficiency:** Outperforms 20-iteration classical R-L in speed

### Failure Conditions
- **Cannot beat classical baseline:** Fundamental approach flawed
- **Training instability:** Gradient explosion/vanishing in physics blocks
- **Poor kernel estimation:** Network ignores blur kernel branch

## Risk Mitigation

### Technical Risks
- **Gradient Flow:** Use residual connections if needed
- **Numerical Issues:** Implement epsilon-smoothing in divisions
- **Memory Overflow:** Implement gradient checkpointing
- **Overfitting:** Early stopping and regularization

### Research Risks
- **Insufficient Novelty:** Emphasize physics-informed architecture
- **Limited Generalization:** Careful test set design
- **Hyperparameter Sensitivity:** Systematic grid search
