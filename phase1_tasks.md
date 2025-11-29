# TASK.md - Phase 1: Current Tasks & Backlog

## Current Sprint (Week 1-2)

### 🔄 **ACTIVE TASKS**

#### **[P1] Data Collection & Preparation**
- [ ] Collect 500 base high-quality images
  - [ ] 100 portraits (various lighting, ages)
  - [ ] 150 natural scenes (landscapes, nature)
  - [ ] 100 architecture (buildings, interiors)
  - [ ] 100 objects (various textures, shapes)
  - [ ] 50 texture samples (patterns, materials)
- [ ] Verify image quality (resolution >1024×1024, sharp)
- [ ] Organize images by category
- [ ] Create metadata file with image descriptions

#### **[P1] Synthetic Blur Generation**
- [ ] Implement motion blur kernel generation
  - [ ] Horizontal blur (0°)
  - [ ] Vertical blur (90°)
  - [ ] Diagonal blur (45°, 135°)
  - [ ] Variable lengths (5, 10, 15 pixels)
- [ ] Implement blur application using convolution
- [ ] Add edge handling (reflection padding)
- [ ] Validate blur quality visually

### 📋 **NEXT UP (Week 2-3)**

#### **[P1] Core Architecture Implementation**
- [ ] Implement Richardson-Lucy physics block
  - [ ] Classical R-L update equations
  - [ ] Learnable parameters (step sizes, regularization)
  - [ ] Numerical stability (epsilon smoothing)
- [ ] Build 5-layer unrolled network
- [ ] Implement learnable initialization for S₀, K₀
- [ ] Add residual connections if needed

#### **[P1] Training Infrastructure**
- [ ] Create dataset class for synthetic blur pairs
- [ ] Implement data loading with augmentation
- [ ] Set up training loop with validation
- [ ] Add checkpointing and model saving
- [ ] Integrate Weights & Biases logging

#### **[P2] Loss Functions**
- [ ] Implement MSE loss for image reconstruction
- [ ] Add kernel estimation loss
- [ ] Experiment with perceptual loss (VGG features)
- [ ] Implement combined loss weighting

## Backlog (Week 3-4)

### 🔬 **RESEARCH & EXPERIMENTS**

#### **[P1] Baseline Implementation**
- [ ] Classical Richardson-Lucy (20 iterations)
- [ ] Simple U-Net baseline
- [ ] Bicubic interpolation baseline
- [ ] Performance comparison framework

#### **[P1] Model Training & Optimization**
- [ ] Hyperparameter tuning (learning rates, batch sizes)
- [ ] Training schedule optimization
- [ ] Regularization techniques (weight decay, dropout)
- [ ] Learning rate scheduling

#### **[P2] Analysis & Visualization**
- [ ] Visualize kernel evolution through layers
- [ ] Plot loss curves and training metrics
- [ ] Analyze failure cases
- [ ] Create before/after comparison grids

### 🐛 **DEBUGGING & VALIDATION**

#### **[P1] Technical Validation**
- [ ] Verify gradient flow through physics blocks
- [ ] Check numerical stability in R-L updates
- [ ] Validate convolution operations
- [ ] Test memory usage and inference speed

#### **[P2] Quality Assurance**
- [ ] Unit tests for all major components
- [ ] Integration tests for training pipeline
- [ ] Reproducibility verification
- [ ] Code documentation and cleanup

## Discovered Issues & Solutions

### 🔍 **FINDINGS FROM INITIAL RESEARCH**

#### **Richardson-Lucy Stability Issues**
- **Issue:** Division by zero in R-L update equation
- **Solution:** Add epsilon smoothing: `B / (S ⊗ K + ε)`
- **Status:** To implement

#### **Kernel Estimation Challenges**
- **Issue:** Kernel estimation might be harder than image reconstruction
- **Solution:** Start with image-only training, add kernel loss gradually
- **Status:** Planned for experimentation

#### **Memory Constraints**
- **Issue:** 5 layers × 256×256 images might exceed 8GB GPU memory
- **Solution:** Implement gradient checkpointing or reduce batch size
- **Status:** Monitoring required

### 📊 **EXPERIMENTAL HYPOTHESES**

#### **Architecture Depth**
- **Hypothesis:** 5 layers optimal for simple blur
- **Test:** Compare 3, 5, 7, 10 layer networks
- **Status:** Planned

#### **Initialization Strategy**
- **Hypothesis:** Good S₀, K₀ initialization crucial for convergence
- **Test:** Compare random vs. identity vs. learned initialization
- **Status:** High priority

#### **Loss Function Weighting**
- **Hypothesis:** Image reconstruction loss should dominate initially
- **Test:** Curriculum learning with gradually increasing kernel loss weight
- **Status:** Planned

## Milestones & Checkpoints

### 🎯 **WEEK 1 MILESTONE**
- [ ] **Environment Ready:** All tools installed and tested
- [ ] **Data Collected:** 500 base images organized and validated
- [ ] **Blur Generation:** Synthetic blur pipeline working
- **Success Criteria:** Can generate perfect blur/sharp pairs

### 🎯 **WEEK 2 MILESTONE**
- [ ] **Core Architecture:** 5-layer physics network implemented
- [ ] **Training Pipeline:** End-to-end training working
- [ ] **Initial Results:** Model trains without crashing
- **Success Criteria:** Loss decreases consistently

### 🎯 **WEEK 3 MILESTONE**
- [ ] **Baseline Comparison:** Outperforms classical R-L
- [ ] **Stability:** Training converges reliably
- [ ] **Metrics:** Achieves >25 dB PSNR on validation
- **Success Criteria:** Clear improvement over baselines

### 🎯 **WEEK 4 MILESTONE (PHASE 1 COMPLETION)**
- [ ] **Target Performance:** >28 dB PSNR on test set
- [ ] **Generalization:** Works on unseen blur angles
- [ ] **Documentation:** Complete analysis and next phase plan
- **Success Criteria:** Ready to proceed to Phase 2

## Contingency Plans

### 🚨 **IF PERFORMANCE IS POOR**
- **Potential Causes:**
  - Architecture too simple
  - Insufficient training data
  - Poor hyperparameter choices
  - Numerical instability
- **Actions:**
  - Increase network depth to 7-10 layers
  - Generate more diverse training data
  - Systematic hyperparameter search
  - Implement gradient clipping

### 🚨 **IF TRAINING IS UNSTABLE**
- **Potential Causes:**
  - Gradient explosion in physics blocks
  - Learning rate too high
  - Batch size too large
- **Actions:**
  - Add gradient clipping (max_norm=1.0)
  - Reduce learning rate by 10x
  - Implement learning rate warmup
  - Use smaller batch sizes

### 🚨 **IF MEMORY ISSUES**
- **Potential Causes:**
  - Large intermediate activations
  - Batch size too large
  - Inefficient convolution operations
- **Actions:**
  - Implement gradient checkpointing
  - Reduce batch size to 8 or 16
  - Use mixed precision training
  - Optimize memory usage in physics blocks

## Research Notes & Ideas

### 💡 **ARCHITECTURAL INSIGHTS**
- Consider adding skip connections between layers
- Investigate different kernel initialization strategies
- Explore learnable blur kernel size adaptation
- Test different physics block variants

### 💡 **TRAINING IMPROVEMENTS**
- Implement curriculum learning (easy → hard blur)
- Add data augmentation during training
- Experiment with different optimizers (Adam, AdamW, SGD)
- Consider multi-scale training

### 💡 **EVALUATION ENHANCEMENTS**
- Add perceptual quality metrics (LPIPS)
- Implement kernel estimation accuracy metrics
- Create visual comparison tools
- Add runtime performance benchmarks

